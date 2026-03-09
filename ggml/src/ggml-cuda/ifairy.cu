#include "ifairy.cuh"
#include "convert.cuh"
#include "unary.cuh"
#include "binbcast.cuh"
#include "norm.cuh"
#include <cuda_bf16.h>
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

union bf16_bits_u {
    uint16_t       u16;
    __nv_bfloat16  bf16;
};

static __device__ __forceinline__ __nv_bfloat16 bits_to_bf16(uint16_t x) {
    bf16_bits_u v;
    v.u16 = x;
    return v.bf16;
}

static __device__ __forceinline__ uint16_t bf16_to_bits(__nv_bfloat16 x) {
    bf16_bits_u v;
    v.bf16 = x;
    return v.u16;
}

static __device__ __forceinline__ uint32_t ifairy_to_bits(float x) {
    return __float_as_uint(x);
}

static __device__ __forceinline__ float bits_to_ifairy(uint32_t x) {
    return __uint_as_float(x);
}

static __device__ __forceinline__ float ifairy_real(float x) {
    const uint32_t bits  = ifairy_to_bits(x);
    const uint16_t rbits = (uint16_t)((bits >> 16) & 0xFFFFu);
    return __bfloat162float(bits_to_bf16(rbits));
}

static __device__ __forceinline__ float ifairy_imag(float x) {
    const uint32_t bits  = ifairy_to_bits(x);
    const uint16_t ibits = (uint16_t)(bits & 0xFFFFu);
    return __bfloat162float(bits_to_bf16(ibits));
}

static __device__ __forceinline__ float make_ifairy(float real, float imag) {
    const uint32_t bits =
        (uint32_t(bf16_to_bits(__float2bfloat16(real))) << 16) |
         uint32_t(bf16_to_bits(__float2bfloat16(imag)));

    return bits_to_ifairy(bits);
}


static __device__ __forceinline__ float ifairy_pack_to_f32container(float real, float imag) {
    return make_ifairy(real, imag);
}

static void rms_norm_f32_cuda_ifairy(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        rms_norm_f32_ifairy<256, false><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_f32_ifairy<1024, false><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

template <int block_size, bool do_multiply = false, bool do_add = false>
static __global__ void rms_norm_f32_ifairy(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0)) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        const float r  = ifairy_real(xi);
        const float im = ifairy_imag(xi);
        tmp += r * r + im * im;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if constexpr (block_size > WARP_SIZE) {
        static_assert((block_size <= 1024) && (block_size % 32 == 0), "unexpected block_size");
        __shared__ float s_sum[32];
        const int        warp_id = tid / WARP_SIZE;
        const int        lane_id = tid % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = 0.0f;
        if (lane_id < (block_size / WARP_SIZE)) {
            tmp = s_sum[lane_id];
        }
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        const float rN  = ifairy_real(xi) * scale;
        const float iN  = ifairy_imag(xi) * scale;
        dst[col] = ifairy_pack_to_f32container(rN, iN); 
    }
}

void ggml_cuda_op_ifairy_rmsnorm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda_ifairy(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

// iFairy RoPE: applies rotation to complex numbers
// src0: packed BF16 (uint32_t: lower=real, upper=imag)
// dst: split FP32 [real: 0..n_dims-1, imag: n_dims..2*n_dims-1]
template<bool forward>
static __global__ void ifairy_rope_kernel(
        const uint32_t * __restrict__ x, float * __restrict__ dst,
        const int ne0, const int ne1, const int ne2, const int s1, const int s2,
        const int n_dims, const int32_t * __restrict__ pos,
        const float freq_scale, const float freq_base) {

    const int i0_complex = blockDim.y * blockIdx.y + threadIdx.y;
    if (i0_complex >= n_dims) return;

    const int row_dst = blockDim.x * blockIdx.x + threadIdx.x;

    // row_dst = i1 + i2*ne1 + i3*ne1*ne2
    const int i1 = row_dst % ne1;
    const int i2 = (row_dst / ne1) % ne2;
    const int i3 = row_dst / (ne1 * ne2);

    const int p = pos[i2];

    const int idst_real = row_dst * 2 * n_dims + i0_complex;
    const int idst_imag = row_dst * 2 * n_dims + n_dims + i0_complex;
    const int ix = i3 * s2 + i2 * s1 + i1 * ne0 + i0_complex;

    const float inv_freq = powf(freq_base, -(float)i0_complex / n_dims);
    const float theta = p * inv_freq * freq_scale;

    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);
    if (!forward) {
        sin_theta = -sin_theta;
    }

    const uint32_t packed = x[ix];
    const float real_val = __bfloat162float(reinterpret_cast<const nv_bfloat16*>(&packed)[0]);
    const float imag_val = __bfloat162float(reinterpret_cast<const nv_bfloat16*>(&packed)[1]);

    dst[idst_real] = real_val * cos_theta - imag_val * sin_theta;
    dst[idst_imag] = real_val * sin_theta + imag_val * cos_theta;
}


void ggml_cuda_op_ifairy_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // op_params layout matches CPU (ops.cpp)
    const int n_dims     = ((int32_t *) dst->op_params)[1] / 2;
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    float freq_base = 10000.0f;  // Fixed per CPU implementation
    float freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_scale,  (int32_t *) dst->op_params + 6,  sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params + 7,  sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8,  sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params + 9,  sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));

    // ne00 = complex_dim, ne01 = heads, ne02 = seq_len
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t nr   = ggml_nrows(src0);

    const size_t s01 = src0->nb[1] / sizeof(uint32_t);
    const size_t s02 = src0->nb[2] / sizeof(uint32_t);

    const int32_t * pos = (const int32_t *) src1->data;

    cudaStream_t stream = ctx.stream();

    const dim3 block_dims(1, CUDA_IFAIRY_BLOCK_SIZE, 1);
    const int n_blocks_y = (n_dims + CUDA_IFAIRY_BLOCK_SIZE - 1) / CUDA_IFAIRY_BLOCK_SIZE;
    const dim3 block_nums(nr, n_blocks_y, 1);

    ifairy_rope_kernel<true><<<block_nums, block_dims, 0, stream>>>(
        (const uint32_t *) src0->data,
        (float *) dst->data,
        ne00, ne01, ne02, s01, s02,
        n_dims, pos,
        freq_scale, freq_base
    );

    GGML_UNUSED(ext_factor);
    GGML_UNUSED(attn_factor);
    GGML_UNUSED(beta_fast);
    GGML_UNUSED(beta_slow);
    GGML_UNUSED(mode);
    GGML_UNUSED(n_ctx_orig);
}

