#pragma once

#include "common.cuh"
#include <cuda_bf16.h>
#include <cstdint>

#define VDR_IFAIRY_Q8_1_MMVQ 1

static __device__ __forceinline__ int get_int_b4_ifairy(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}

static __device__ __forceinline__ float vec_dot_ifairy_ifairy_q16_impl_mmvq(
    const int & v, const int * __restrict__ u, const float * __restrict__ d8) {

    int sumi_realreal = 0;
    int sumi_realimag = 0;
    int sumi_imagreal = 0;
    int sumi_imagimag = 0;

    // LUT for Real weights: 00->-1(FF), 01->1(01), 10->0(00), 11->0(00)
    // Little-endian: 00, 00, 01, FF -> 0x000001FF
    const int lut_real = 0x000001FF;
    
    // LUT for Imag weights: 00->0(00), 01->0(00), 10->-1(FF), 11->1(01)
    // Little-endian: 01, FF, 00, 00 -> 0x01FF0000
    const int lut_imag = 0x01FF0000;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i ) {
        int weight_chunk = (v >> (8 * i)) & 0xFF;

        // Spread 8 bits (4x2 bits) into 4 bytes (4x2 bits at LSB of each byte)
        // weight_chunk: 00000000 00000000 00000000 dccbbaaa (where a,b,c,d are 2 bits)
        // Target t:     000000dd 000000cc 000000bb 000000aa
        
        unsigned int t = weight_chunk;
        t = (t | (t << 12)) & 0x000F000F;
        t = (t | (t << 6)) & 0x03030303;

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
        // Fallback for non-NVIDIA GPUs (AMD/Moore Threads)
        // Calculate value: l=0 -> -1 (0xFF), l=1 -> 1 (0x01)
        unsigned int l = t & 0x01010101;
        unsigned int h = (t >> 1) & 0x01010101;
        
        unsigned int mask_l = (l ^ 0x01010101) * 0xFF;
        unsigned int val = mask_l | 0x01010101;

        unsigned int mask_imag = h * 0xFF;
        unsigned int mask_real = ~mask_imag;

        int weight_real = val & mask_real;
        int weight_imag = val & mask_imag;
#else
        // NVIDIA CUDA: Use __byte_perm for register-based LUT lookup (Zero Memory Access)
        // t contains indices in the lowest 2 bits of each byte.
        // __byte_perm selects bytes from lut_real/lut_imag based on these indices.
        int weight_real = __byte_perm(lut_real, 0, t);
        int weight_imag = __byte_perm(lut_imag, 0, t);
#endif

        sumi_realreal = ggml_cuda_dp4a(u[i],       weight_real, sumi_realreal);
        sumi_realimag = ggml_cuda_dp4a(u[i+QR2_K], weight_real, sumi_realimag);
        sumi_imagreal = ggml_cuda_dp4a(u[i],       weight_imag, sumi_imagreal);
        sumi_imagimag = ggml_cuda_dp4a(u[i+QR2_K], weight_imag, sumi_imagimag);
    }

    float result_real = sumi_realreal*d8[0]+sumi_imagimag*d8[3];
    float result_imag = sumi_imagreal*d8[2]-sumi_realimag*d8[1];
    
    // 将结果打包为32位整型并返回
    __nv_bfloat16 result_real_bf16 = __float2bfloat16(result_real);
    __nv_bfloat16 result_imag_bf16 = __float2bfloat16(result_imag);

    int result = (*reinterpret_cast<uint16_t*>(&result_real_bf16)) | 
                 ((*reinterpret_cast<uint16_t*>(&result_imag_bf16)) << 16);
    return __int_as_float(result); 
}

static __device__ __forceinline__ float vec_dot_ifairy_ifairy_q16(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    // Inputs:
    //   vbq: pointer to quantized weights block (block_ifairy)
    //   bq8_1: pointer to quantized activations block (block_ifairy_q16) - cast it!

    const block_ifairy_q16 * bq16 = (const block_ifairy_q16 *) bq8_1;
    const block_ifairy * b = (const block_ifairy *) vbq + kbx;

    const int v = get_int_b4_ifairy(b->qs, iqs);

    int    u[QR2_K*2];
    float d8[QR2_K];

    d8[0] = __half2float(b->d_real) * __half2float(bq16->d_real);
    d8[1] = __half2float(b->d_real) * __half2float(bq16->d_imag);
    d8[2] = __half2float(b->d_imag) * __half2float(bq16->d_real);
    d8[3] = __half2float(b->d_imag) * __half2float(bq16->d_imag);

#pragma unroll
    for (int i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_b4_ifairy(bq16->x_real, iqs*QR2_K + i);
        u[i+QR2_K] = get_int_b4_ifairy(bq16->x_imag, iqs*QR2_K + i);
    }

    return vec_dot_ifairy_ifairy_q16_impl_mmvq(v, u, d8);
}
