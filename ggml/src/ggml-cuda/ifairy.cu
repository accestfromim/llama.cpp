#include "ifairy.cuh"
#include "convert.cuh"
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

    rms_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_ifairy_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static bool warned = false;
    if (!warned) {
        fprintf(stderr, "%s: not implemented yet\n", __func__);
        warned = true;
    }
}

using ifairy_t = uint32_t;

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

static __device__ __forceinline__ float ifairy_real(ifairy_t x) {
    return __bfloat162float(bits_to_bf16((uint16_t)((x >> 16) & 0xFFFFu)));
}

static __device__ __forceinline__ float ifairy_imag(ifairy_t x) {
    return __bfloat162float(bits_to_bf16((uint16_t)(x & 0xFFFFu)));
}

static __device__ __forceinline__ ifairy_t make_ifairy(float real, float imag) {
    return (ifairy_t(bf16_to_bits(__float2bfloat16(real))) << 16) |
           ifairy_t(bf16_to_bits(__float2bfloat16(imag)));
}

static __device__ __forceinline__ float ifairy_pack_to_f32container(float r, float im) {
    const ifairy_t out = make_ifairy(r, im);        // float -> bf16 bits pack (uint32)
    return __uint_as_float((uint32_t)out);          // 存回 GGML_TYPE_F32 容器
}

static __device__ __forceinline__ ifairy_t op_ifairy_add_packed(const ifairy_t a, const ifairy_t b) {
    const float ar = ifairy_real(a);
    const float ai = ifairy_imag(a);
    const float br = ifairy_real(b);
    const float bi = ifairy_imag(b);

    return make_ifairy(ar + br, ai + bi);
}

static __device__ __forceinline__ ifairy_t op_ifairy_mul_packed(const ifairy_t a, const ifairy_t b) {
    const float ar = ifairy_real(a);
    const float ai = ifairy_imag(a);
    const float br = ifairy_real(b);
    const float bi = ifairy_imag(b);

    // Hermitian product: conj(a) * b
    const float real = __fmaf_rn(ar, br, ai * bi);       
    const float imag = __fmaf_rn(ar, bi, -ai * br);

    return make_ifairy(real, imag);
}

template <ifairy_t (*bin_op)(const ifairy_t, const ifairy_t),
          typename src0_t,
          typename src1_t,
          typename dst_t,
          typename... src1_ptrs>
static __global__ void k_bin_bcast_ifairy(const src0_t *         src0,
                                   const src1_t *         src1,
                                   dst_t *                dst,
                                   const int              ne0,
                                   const int              ne1,
                                   const int              ne2,
                                   const uint3            ne3,
                                   const uint3            ne10,
                                   const uint3            ne11,
                                   const uint3            ne12,
                                   const uint3            ne13,
                                   /*int s0, */ const int s1,
                                   const int              s2,
                                   const int              s3,
                                   /*int s00,*/ const int s01,
                                   const int              s02,
                                   const int              s03,
                                   /*int s10,*/ const int s11,
                                   const int              s12,
                                   const int              s13,
                                   src1_ptrs... src1s) {
    const uint32_t i0s = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t i1  = (blockDim.y * blockIdx.y + threadIdx.y);
    const uint32_t i2  = fastdiv((blockDim.z * blockIdx.z + threadIdx.z), ne3);
    const uint32_t i3  = (blockDim.z * blockIdx.z + threadIdx.z) - (i2 * ne3.z);

    if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3.z) {
        return;
    }

    const uint32_t i11 = fastmodulo(i1, ne11);
    const uint32_t i12 = fastmodulo(i2, ne12);
    const uint32_t i13 = fastmodulo(i3, ne13);

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 ? (src0 + i_src0) : nullptr;
    dst_t * dst_row = dst + i_dst;

    for (int i0 = i0s; i0 < ne0; i0 += blockDim.x * gridDim.x) {
        const uint32_t i10 = fastmodulo(i0, ne10);
        const ifairy_t a = src0_row ? src0_row[i0] : 0u;
        const ifairy_t b = src1[i_src1 + i10];
        dst_row[i0] = bin_op(a, b);
    }
}

template <ifairy_t (*bin_op)(const ifairy_t, const ifairy_t),
          typename src0_t,
          typename src1_t,
          typename dst_t,
          typename... src1_ptrs>
static __global__ void k_bin_bcast_unravel_ifairy(const src0_t *         src0,
                                           const src1_t *         src1,
                                           dst_t *                dst,
                                           const uint3            ne0,
                                           const uint3            ne1,
                                           const uint3            ne2,
                                           const uint32_t         ne3,
                                           const uint3            prod_012,
                                           const uint3            prod_01,
                                           const uint3            ne10,
                                           const uint3            ne11,
                                           const uint3            ne12,
                                           const uint3            ne13,
                                           /*int s0, */ const int s1,
                                           const int              s2,
                                           const int              s3,
                                           /*int s00,*/ const int s01,
                                           const int              s02,
                                           const int              s03,
                                           /*int s10,*/ const int s11,
                                           const int              s12,
                                           const int              s13,
                                           src1_ptrs... src1s) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = fastdiv(i, prod_012);
    const uint32_t i2 = fastdiv(i - i3 * prod_012.z, prod_01);
    const uint32_t i1 = fastdiv(i - i3 * prod_012.z - i2 * prod_01.z, ne0);
    const uint32_t i0 = i - i3 * prod_012.z - i2 * prod_01.z - i1 * ne0.z;

    if (i0 >= ne0.z || i1 >= ne1.z || i2 >= ne2.z || i3 >= ne3) {
        return;
    }

    const int i11 = fastmodulo(i1, ne11);
    const int i12 = fastmodulo(i2, ne12);
    const int i13 = fastmodulo(i3, ne13);

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 ? (src0 + i_src0) : nullptr;
    dst_t * dst_row = dst + i_dst;

    const uint32_t i10 = fastmodulo(i0, ne10);

    const ifairy_t a = src0_row ? src0_row[i0] : 0u;
    const ifairy_t b = src1[i_src1 + i10];
    dst_row[i0] = bin_op(a, b);
}

template <ifairy_t (*bin_op)(const ifairy_t, const ifairy_t), typename src0_t, typename src1_t, typename dst_t, size_t... I>
static void launch_bin_bcast_pack(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                                  const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd,
                                  cudaStream_t stream, std::index_sequence<I...>) {
    GGML_TENSOR_BINARY_OP_LOCALS

    int nr0 = ne10 / ne0;
    int nr1 = ne11 / ne1;
    int nr2 = ne12 / ne2;
    int nr3 = ne13 / ne3;

    int nr[4] = { nr0, nr1, nr2, nr3 };

    int64_t cne[]  = { ne0, ne1, ne2, ne3 };
    int64_t cne0[] = { ne00, ne01, ne02, ne03 };
    int64_t cne1[] = { ne10, ne11, ne12, ne13 };

    size_t cnb[]  = { nb0, nb1, nb2, nb3 };
    size_t cnb0[] = { nb00, nb01, nb02, nb03 };
    size_t cnb1[] = { nb10, nb11, nb12, nb13 };

    auto collapse = [](int64_t cne[]) {
        cne[0] *= cne[1];
        cne[1] = cne[2];
        cne[2] = cne[3];
        cne[3] = 1;
    };

    auto collapse_nb = [](size_t cnb[], const int64_t cne[]) {
        cnb[1] *= cne[1];
        cnb[2] *= cne[2];
        cnb[3] *= cne[3];
    };

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)) {
        for (int i = 0; i < 4; i++) {
            if (nr[i] != 1) {
                break;
            }
            if (i > 0) {
                collapse_nb(cnb, cne);
                collapse_nb(cnb0, cne0);
                collapse_nb(cnb1, cne1);
                collapse(cne);
                collapse(cne0);
                collapse(cne1);
            }
        }
    }

    {
        int64_t ne0 = cne[0];
        int64_t ne1 = cne[1];
        int64_t ne2 = cne[2];
        int64_t ne3 = cne[3];

        //int64_t ne00 = cne0[0]; GGML_UNUSED(ne00);
        //int64_t ne01 = cne0[1]; GGML_UNUSED(ne01);
        //int64_t ne02 = cne0[2]; GGML_UNUSED(ne02);
        //int64_t ne03 = cne0[3]; GGML_UNUSED(ne03);

        size_t nb0 = cnb[0];
        size_t nb1 = cnb[1];
        size_t nb2 = cnb[2];
        size_t nb3 = cnb[3];

        size_t nb00 = cnb0[0];
        size_t nb01 = cnb0[1];
        size_t nb02 = cnb0[2];
        size_t nb03 = cnb0[3];

        size_t nb10 = cnb1[0];
        size_t nb11 = cnb1[1];
        size_t nb12 = cnb1[2];
        size_t nb13 = cnb1[3];

        size_t s0 = nb0 / sizeof(dst_t);
        size_t s1 = nb1 / sizeof(dst_t);
        size_t s2 = nb2 / sizeof(dst_t);
        size_t s3 = nb3 / sizeof(dst_t);

        size_t s10 = nb10 / sizeof(src1_t);
        size_t s11 = nb11 / sizeof(src1_t);
        size_t s12 = nb12 / sizeof(src1_t);
        size_t s13 = nb13 / sizeof(src1_t);

        size_t s00 = nb00 / sizeof(src0_t);
        size_t s01 = nb01 / sizeof(src0_t);
        size_t s02 = nb02 / sizeof(src0_t);
        size_t s03 = nb03 / sizeof(src0_t);

        GGML_ASSERT(nb0 % sizeof(dst_t) == 0);
        GGML_ASSERT(nb1 % sizeof(dst_t) == 0);
        GGML_ASSERT(nb2 % sizeof(dst_t) == 0);
        GGML_ASSERT(nb3 % sizeof(dst_t) == 0);

        GGML_ASSERT(nb00 % sizeof(src0_t) == 0);
        GGML_ASSERT(nb01 % sizeof(src0_t) == 0);
        GGML_ASSERT(nb02 % sizeof(src0_t) == 0);
        GGML_ASSERT(nb03 % sizeof(src0_t) == 0);

        GGML_ASSERT(nb10 % sizeof(src1_t) == 0);
        GGML_ASSERT(nb11 % sizeof(src1_t) == 0);
        GGML_ASSERT(nb12 % sizeof(src1_t) == 0);
        GGML_ASSERT(nb13 % sizeof(src1_t) == 0);

        GGML_ASSERT(s0 == 1);
        GGML_ASSERT(s00 == 1);
        GGML_ASSERT(s10 == 1);

        const int block_size = 128;

        int64_t hne0 = std::max(ne0 / 2LL, 1LL);

        dim3 block_dims;
        block_dims.x = std::min<unsigned int>(hne0, block_size);
        block_dims.y = std::min<unsigned int>(ne1, block_size / block_dims.x);
        block_dims.z = std::min(std::min<unsigned int>(ne2 * ne3, block_size / block_dims.x / block_dims.y), 64U);

        dim3 block_nums((hne0 + block_dims.x - 1) / block_dims.x, (ne1 + block_dims.y - 1) / block_dims.y,
                        (ne2 * ne3 + block_dims.z - 1) / block_dims.z);

        const uint3 ne10 = init_fastdiv_values((uint32_t) cne1[0]);
        const uint3 ne11 = init_fastdiv_values((uint32_t) cne1[1]);
        const uint3 ne12 = init_fastdiv_values((uint32_t) cne1[2]);
        const uint3 ne13 = init_fastdiv_values((uint32_t) cne1[3]);

        if (block_nums.z > 65535) {
            int         block_num  = (ne0 * ne1 * ne2 * ne3 + block_size - 1) / block_size;
            const uint3 prod_012    = init_fastdiv_values((uint32_t) (ne0 * ne1 * ne2));
            const uint3 prod_01     = init_fastdiv_values((uint32_t) (ne0 * ne1));
            const uint3 ne0_fastdiv = init_fastdiv_values((uint32_t) ne0);
            const uint3 ne1_fastdiv = init_fastdiv_values((uint32_t) ne1);
            const uint3 ne2_fastdiv = init_fastdiv_values((uint32_t) ne2);

            if constexpr (sizeof...(I) > 0) {
                k_bin_bcast_unravel_ifairy<bin_op, src0_t, src1_t, dst_t><<<block_num, block_size, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd, ne0_fastdiv, ne1_fastdiv, ne2_fastdiv, ne3, prod_012, prod_01, ne10, ne11,
                    ne12, ne13,
                    /* s0, */ s1, s2, s3,
                    /* s00,*/ s01, s02, s03,
                    /* s10,*/ s11, s12, s13, (const src1_t *) dst->src[I + 1]->data...);
            } else {
                k_bin_bcast_unravel_ifairy<bin_op, src0_t, src1_t, dst_t>
                    <<<block_num, block_size, 0, stream>>>(src0_dd, src1_dd, dst_dd, ne0_fastdiv, ne1_fastdiv,
                                                           ne2_fastdiv, ne3, prod_012, prod_01, ne10, ne11, ne12, ne13,
                                                           /* s0, */ s1, s2, s3,
                                                           /* s00,*/ s01, s02, s03,
                                                           /* s10,*/ s11, s12, s13);
            }
        } else {
            const uint3 ne3_fastdiv = init_fastdiv_values((uint32_t) ne3);
            if constexpr (sizeof...(I) > 0) {
                k_bin_bcast_ifairy<bin_op, src0_t, src1_t, dst_t><<<block_nums, block_dims, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3_fastdiv, ne10, ne11, ne12, ne13,
                    /* s0, */ s1, s2, s3,
                    /* s00,*/ s01, s02, s03,
                    /* s10,*/ s11, s12, s13, (const src1_t *) dst->src[I + 1]->data...);
            } else {
                k_bin_bcast_ifairy<bin_op, src0_t, src1_t, dst_t><<<block_nums, block_dims, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3_fastdiv, ne10, ne11, ne12, ne13,
                    /* s0, */ s1, s2, s3,
                    /* s00,*/ s01, s02, s03,
                    /* s10,*/ s11, s12, s13);
            }
        }
    }
}

template <ifairy_t (*bin_op)(const ifairy_t, const ifairy_t), int n_fuse = 1>
struct bin_bcast_cuda {
    template<typename src0_t, typename src1_t, typename dst_t>
    void operator()(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst,
            const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd,
            cudaStream_t stream) {
        launch_bin_bcast_pack<bin_op, src0_t, src1_t, dst_t>(
            src0, src1, dst, src0_dd, src1_dd, dst_dd, stream, std::make_index_sequence<n_fuse>{});
    }
};


void ggml_cuda_op_ifairy_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_ifairy_add_packed>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_ifairy_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_ifairy_mul_packed>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());

}


template <float (*op)(float)>
void ggml_cuda_op_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        unary_cuda<op>((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        unary_cuda<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}


static __device__ __forceinline__ float op_ifairy_relu2(float x) {
    const ifairy_t a = (ifairy_t)__float_as_uint(x);

    // 注意：real 在高16位，imag在低16位
    const uint16_t ar = (uint16_t)((a >> 16) & 0xFFFFu);
    const uint16_t ai = (uint16_t)( a        & 0xFFFFu);

    const bool neg_r = (ar & 0x8000u) != 0;
    const bool neg_i = (ai & 0x8000u) != 0;

    // 你的原语义：real 和 imag 都为负（按 bf16 符号位）时整体置 0
    if (neg_r && neg_i) {
        return 0.0f; // 等价于 packed (0,0)
    }

    float r = ifairy_real(a); // 这里 ifairy_real/imag 是你已改成返回 float 的版本
    float i = ifairy_imag(a);

    r = r * r;
    i = i * i;

    const ifairy_t out = make_ifairy(r, i); // float -> bf16 bits pack
    return __uint_as_float((uint32_t)out);
}

void ggml_cuda_op_ifairy_relu2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_ifairy_relu2>(ctx, dst);
}
