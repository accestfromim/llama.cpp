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

void ggml_cuda_op_ifairy_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static bool warned = false;
    if (!warned) {
        fprintf(stderr, "%s: not implemented yet\n", __func__);
        warned = true;
    }
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
