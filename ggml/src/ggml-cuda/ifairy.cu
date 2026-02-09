#include "ifairy.cuh"
#include "convert.cuh"
#include <cstdint>

#define CUDA_IFAIRY_BLOCK_SIZE 256

// Split: Interleaved BF16 -> Planar FP32
// src: [ne0, ne1, ne2, ne3] (BF16)
// dst: [ne0, ne1, ne2, ne3] (FP32)
// Logic:
//   src[...][2*k]   (Real) -> dst[...][k]
//   src[...][2*k+1] (Imag) -> dst[...][ne0/2 + k]
static __global__ void ifairy_split_kernel(
    const char * __restrict__ src0, char * __restrict__ dst,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
    int64_t nb01, int64_t nb02, int64_t nb03,
    int64_t nb1, int64_t nb2, int64_t nb3) {

    // Grid X: Rows (nr)
    // Grid Y: Column Blocks (covering ne0/2)
    // Block Y: Column Threads

    // i0 is the index in the src dimension (0..ne0-1). We process pairs, so step by 2.
    const int64_t i0 = 2 * ((int64_t)blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne0) return;

    // Row index from Grid X
    int64_t r = blockIdx.x;

    // Decompose row index into i1, i2, i3
    const int64_t i1 = r % ne1; r /= ne1;
    const int64_t i2 = r % ne2; r /= ne2;
    const int64_t i3 = r;

    // Calculate base offsets
    const int64_t src_offset = i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * sizeof(nv_bfloat16);
    const int64_t dst_offset = i3 * nb3  + i2 * nb2  + i1 * nb1;

    // Pointers
    const nv_bfloat16 * src_ptr = (const nv_bfloat16 *)(src0 + src_offset);
    float * dst_ptr_r = (float *)(dst + dst_offset + (i0 / 2) * sizeof(float));
    float * dst_ptr_i = (float *)(dst + dst_offset + (ne0 / 2 + i0 / 2) * sizeof(float));

    // Load and Convert
    // Compiler should optimize contiguous load of v0, v1
    nv_bfloat16 v0 = src_ptr[0];
    nv_bfloat16 v1 = src_ptr[1];

    *dst_ptr_r = __bfloat162float(v0);
    *dst_ptr_i = __bfloat162float(v1);
}

// Merge: Planar FP32 -> Interleaved BF16
// src: [ne0, ne1, ne2, ne3] (FP32)
// dst: [ne0, ne1, ne2, ne3] (BF16)
// Logic:
//   src[...][k]             (Real) -> dst[...][2*k]
//   src[...][ne0/2 + k]     (Imag) -> dst[...][2*k+1]
static __global__ void ifairy_merge_kernel(
    const char * __restrict__ src0, char * __restrict__ dst,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
    int64_t nb01, int64_t nb02, int64_t nb03,
    int64_t nb1, int64_t nb2, int64_t nb3) {

    // Grid X: Rows (nr)
    // Grid Y: Column Blocks
    // Block Y: Column Threads

    // i0 is the index in the dst dimension (0..ne0-1). We process pairs, so step by 2.
    const int64_t i0 = 2 * ((int64_t)blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne0) return;

    // Row index from Grid X
    int64_t r = blockIdx.x;

    // Decompose row index into i1, i2, i3
    const int64_t i1 = r % ne1; r /= ne1;
    const int64_t i2 = r % ne2; r /= ne2;
    const int64_t i3 = r;

    // Calculate base offsets
    const int64_t src_offset = i3 * nb03 + i2 * nb02 + i1 * nb01;
    const int64_t dst_offset = i3 * nb3  + i2 * nb2  + i1 * nb1 + i0 * sizeof(nv_bfloat16);

    // Pointers
    const float * src_ptr_r = (const float *)(src0 + src_offset + (i0 / 2) * sizeof(float));
    const float * src_ptr_i = (const float *)(src0 + src_offset + (ne0 / 2 + i0 / 2) * sizeof(float));
    nv_bfloat16 * dst_ptr = (nv_bfloat16 *)(dst + dst_offset);

    // Load and Convert
    float f0 = *src_ptr_r;
    float f1 = *src_ptr_i;

    dst_ptr[0] = __float2bfloat16(f0);
    dst_ptr[1] = __float2bfloat16(f1);
}

void ggml_cuda_op_ifairy_split(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];
    const int64_t nr  = ggml_nrows(src0);

    // RoPE-style grid configuration
    // X dimension covers rows
    // Y dimension covers columns (inner dim)
    const dim3 block_dims(1, CUDA_IFAIRY_BLOCK_SIZE, 1);
    const int n_blocks_y = (ne0 + 2 * CUDA_IFAIRY_BLOCK_SIZE - 1) / (2 * CUDA_IFAIRY_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_y, 1);

    cudaStream_t stream = ctx.stream();
    ifairy_split_kernel<<<block_nums, block_dims, 0, stream>>>(
        (const char *)src0->data, (char *)dst->data,
        ne0, ne1, ne2, ne3,
        src0->nb[1], src0->nb[2], src0->nb[3],
        dst->nb[1], dst->nb[2], dst->nb[3]
    );
}

void ggml_cuda_op_ifairy_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const int64_t ne0 = dst->ne[0]; // dst determines the output size/shape
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];
    const int64_t nr  = ggml_nrows(dst);

    const dim3 block_dims(1, CUDA_IFAIRY_BLOCK_SIZE, 1);
    const int n_blocks_y = (ne0 + 2 * CUDA_IFAIRY_BLOCK_SIZE - 1) / (2 * CUDA_IFAIRY_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_y, 1);

    cudaStream_t stream = ctx.stream();
    ifairy_merge_kernel<<<block_nums, block_dims, 0, stream>>>(
        (const char *)src0->data, (char *)dst->data,
        ne0, ne1, ne2, ne3,
        src0->nb[1], src0->nb[2], src0->nb[3],
        dst->nb[1], dst->nb[2], dst->nb[3]
    );
}

void ggml_cuda_op_ifairy_rmsnorm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static bool warned = false;
    if (!warned) {
        fprintf(stderr, "%s: not implemented yet\n", __func__);
        warned = true;
    }
    // TODO: Implement using rope/norm patterns
}

// rope_yarn_ifairy: Simplified RoPE calculation for IFAIRY
// Only applies basic frequency scaling without YaRN complex logic
template<bool forward>
static __device__ void rope_yarn_ifairy(
        const float theta_extrap, const float freq_scale,
        float & cos_theta, float & sin_theta) {
    // Standard RoPE: theta = inv_freq * position_id
    const float theta = theta_extrap * freq_scale;
    cos_theta = cosf(theta);
    sin_theta = sinf(theta);
    if (!forward) {
        sin_theta *= -1.0f;  // Backward pass uses inverse rotation
    }
}

// ifairy_rope_kernel: Applies RoPE to complex numbers in BF16 interleaved format
// src and dst are in BF16 format with interleaved complex numbers: [real0, imag0, real1, imag1, ...]
// Each thread processes a pair of real and imaginary components using vectorized loads/stores
// gridDim.x: number of rows (ne1*ne2*ne3)
// gridDim.y: blocks for column dimension (covers ne0/2 complex pairs)
template<bool forward>
static __global__ void ifairy_rope_kernel(
        const nv_bfloat16 * __restrict__ x, nv_bfloat16 * __restrict__ dst,
        const int ne0, const int ne1, const int s1, const int s2,
        const int n_dims, const int32_t * __restrict__ pos,
        const float freq_scale, const float * __restrict__ freq_factors) {
    // i0 is the index in complex pairs (0..ne0/2-1)
    const int i0_complex = blockDim.y * blockIdx.y + threadIdx.y;
    const int i0 = i0_complex * 2;  // Convert to BF16 index

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x * blockIdx.x + threadIdx.x;

    const int row_x     = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    // Calculate indices for interleaved BF16 format - note that i0 is already multiplied by 2
    const int idst = row_dst * ne0 + i0;      // Destination index for this complex pair
    const int ix   = channel_x * s2 + row_x * s1 + i0;  // Source index

    // Return early if beyond n_dims (no rotation needed, just copy)
    if (i0_complex >= n_dims) {
        // Use vectorized load/store for better memory bandwidth
        nv_bfloat162 val = *reinterpret_cast<const nv_bfloat162*>(&x[ix]);
        *reinterpret_cast<nv_bfloat162*>(&dst[idst]) = val;
        return;
    }

    // Get position ID
    const int64_t p = pos[channel_x];

    // Get precomputed frequency factor
    const float theta = p * freq_factors[i0_complex];

    // Calculate cos and sin values
    float cos_theta, sin_theta;
    rope_yarn_ifairy<forward>(theta, freq_scale, cos_theta, sin_theta);

    // Read BF16 values using vectorized load
    nv_bfloat162 val2 = *reinterpret_cast<const nv_bfloat162*>(&x[ix]);
    float2 val = __bfloat1622float2(val2);

    // Apply rotation: (real + i*imag) * (cos + i*sin)
    const float y0 = val.x * cos_theta - val.y * sin_theta;  // New real part
    const float y1 = val.x * sin_theta + val.y * cos_theta;  // New imaginary part

    // Write result using vectorized store
    *reinterpret_cast<nv_bfloat162*>(&dst[idst]) = __float22bfloat162_rn(make_float2(y0, y1));
}

void ggml_cuda_op_ifairy_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // Input BF16 complex tensor
    const ggml_tensor * src1 = dst->src[1];  // Position tensor (I32)

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_BF16);

    // Extract operation parameters
    const int n_dims     = ((int32_t *) dst->op_params)[1] / 2;  // Actual dimension count
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *) dst->op_params + 5,  sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params + 6,  sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params + 7,  sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8,  sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params + 9,  sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));

    // Tensor dimensions for BF16 interleaved format
    const int64_t ne00 = src0->ne[0];  // Dimension with interleaved real/imag (2 * complex_dim)
    const int64_t ne01 = src0->ne[1];  // Number of attention heads
    const int64_t ne02 = src0->ne[2];  // Sequence length or other dimension
    const int64_t nr   = ggml_nrows(src0);

    // Strides in elements
    const size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = src0->nb[2] / ggml_type_size(src0->type);

    // Position data
    const int32_t * pos = (const int32_t *) src1->data;

    // Precompute frequency factors on host using optimized multiplicative accumulation
    // freq_factor[i] = powf(theta_scale, i) where theta_scale = base^(-2/n_dims_total)
    // This allows kernel to compute theta = p * freq_factor[i] with just a multiply
    const float theta_scale = powf(freq_base, -2.0f / (n_dims * 2));
    float * freq_factors_host = (float *) alloca(n_dims * sizeof(float));
    float cur_factor = 1.0f;  // Start with powf(theta_scale, 0) = 1
    for (int i = 0; i < n_dims; i++) {
        freq_factors_host[i] = cur_factor;
        cur_factor *= theta_scale;  // Multiply by theta_scale for next iteration (no powf)
    }

    // Use ggml_cuda_pool_alloc for efficient device memory management
    // This automatically handles memory pooling and avoids cudaMalloc/cudaFree overhead
    ggml_cuda_pool_alloc<float> freq_factors_device(ctx.pool(), n_dims);
    CUDA_CHECK(cudaMemcpyAsync(freq_factors_device.get(), freq_factors_host, n_dims * sizeof(float),
                                 cudaMemcpyHostToDevice, ctx.stream()));

    // Forward pass flag
    const bool forward = true;

    // Launch kernel
    cudaStream_t stream = ctx.stream();

    // Configure kernel launch parameters
    const dim3 block_dims(1, CUDA_IFAIRY_BLOCK_SIZE, 1);
    const int n_blocks_y = (ne00 + 2 * CUDA_IFAIRY_BLOCK_SIZE - 1) / (2 * CUDA_IFAIRY_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_y, 1);

    ifairy_rope_kernel<true><<<block_nums, block_dims, 0, stream>>>(
        (const nv_bfloat16 *) src0->data,
        (nv_bfloat16 *) dst->data,
        ne00, ne01, s01, s02,
        n_dims, pos,
        freq_scale, freq_factors_device.get()
    );

    GGML_UNUSED(ext_factor);
    GGML_UNUSED(attn_factor);
    GGML_UNUSED(beta_fast);
    GGML_UNUSED(beta_slow);
    GGML_UNUSED(mode);
    GGML_UNUSED(n_ctx_orig);
}

void ggml_cuda_op_ifairy_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static bool warned = false;
    if (!warned) {
        fprintf(stderr, "%s: not implemented yet\n", __func__);
        warned = true;
    }
}

void ggml_cuda_op_ifairy_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static bool warned = false;
    if (!warned) {
        fprintf(stderr, "%s: not implemented yet\n", __func__);
        warned = true;
    }
}

void ggml_cuda_op_ifairy_relu2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static bool warned = false;
    if (!warned) {
        fprintf(stderr, "%s: not implemented yet\n", __func__);
        warned = true;
    }
}
