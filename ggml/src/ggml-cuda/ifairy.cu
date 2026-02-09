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
// Each thread processes a pair of real and imaginary components
// gridDim.x: number of rows (ne1*ne2*ne3)
// gridDim.y: blocks for column dimension (covers ne0/2 complex pairs)
template<bool forward>
static __global__ void ifairy_rope_kernel(
        const nv_bfloat16 * __restrict__ x, nv_bfloat16 * __restrict__ dst,
        const int ne0, const int ne1, const int s1, const int s2,
        const int n_dims, const int32_t * __restrict__ pos,
        const float freq_scale, const float theta_scale) {
    // i0 is the index in the interleaved BF16 array (0..ne0-1), we process pairs so step by 2
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x * blockIdx.x + threadIdx.x;

    const int row_x     = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    // Calculate indices for interleaved BF16 format
    const int idst = row_dst * ne0 + i0;      // Destination index for this complex pair
    const int ix   = channel_x * s2 + row_x * s1 + i0;  // Source index

    // Return early if beyond n_dims (no rotation needed, just copy)
    // n_dims is the number of complex pairs, so compare with n_dims*2 for BF16 elements
    if (i0 >= n_dims * 2) {
        dst[idst + 0] = x[ix + 0];
        dst[idst + 1] = x[ix + 1];
        return;
    }

    // Get position ID
    const int64_t p = pos[channel_x];

    // Calculate theta using precomputed theta_scale for efficiency
    // theta = p * pow(theta_scale, i0/2.0f)
    // where theta_scale = base^(-2/n_dims_total) and n_dims_total = n_dims * 2
    const int head_idx = i0 / 2;  // Complex pair index (0..n_dims-1)
    const float theta = p * powf(theta_scale, (float)head_idx);

    // Calculate cos and sin values
    float cos_theta, sin_theta;
    rope_yarn_ifairy<forward>(theta, freq_scale, cos_theta, sin_theta);

    // Read BF16 values and convert to float
    const float x0 = __bfloat162float(x[ix + 0]);  // Real part
    const float x1 = __bfloat162float(x[ix + 1]);  // Imaginary part

    // Apply rotation: (x0 + i*x1) * (cos + i*sin)
    const float y0 = x0 * cos_theta - x1 * sin_theta;  // New real part
    const float y1 = x0 * sin_theta + x1 * cos_theta;  // New imaginary part

    // Convert back to BF16 and store
    dst[idst + 0] = __float2bfloat16(y0);
    dst[idst + 1] = __float2bfloat16(y1);
}

template<bool forward>
static void ifairy_rope_cuda(
        const nv_bfloat16 * x, nv_bfloat16 * dst,
        const int ne0, const int ne1, const int s1, const int s2,
        const int n_dims, const int nr, const int32_t * pos,
        const float freq_scale, const float theta_scale, cudaStream_t stream) {
    GGML_ASSERT(ne0 % 2 == 0);  // Must be even for interleaved format

    const dim3 block_dims(1, CUDA_IFAIRY_BLOCK_SIZE, 1);
    const int n_blocks_y = (ne0 + 2 * CUDA_IFAIRY_BLOCK_SIZE - 1) / (2 * CUDA_IFAIRY_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_y, 1);

    ifairy_rope_kernel<forward><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, theta_scale);
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

    // Calculate theta_scale for frequency calculation
    // theta_scale = base^(-2/n_dims_total) where n_dims_total = n_dims * 2
    const float theta_scale = powf(freq_base, -2.0f / (n_dims * 2));

    // Forward pass flag
    const bool forward = true;

    // Launch kernel
    cudaStream_t stream = ctx.stream();
    ifairy_rope_cuda<true>(
        (const nv_bfloat16 *) src0->data,
        (nv_bfloat16 *) dst->data,
        ne00, ne01, s01, s02,
        n_dims, nr, pos,
        freq_scale, theta_scale, stream
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
