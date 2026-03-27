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

    // LUT for __byte_perm:
    // byte 0..3:
    // real LUT: 00->FF(-1), 01->01(1), 10->00(0), 11->00(0)
    // imag LUT: 00->00(0), 01->00(0), 10->FF(-1), 11->01(1)
    const unsigned int lut_real = 0x000001FFu;
    const unsigned int lut_imag = 0x01FF0000u;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const unsigned int weight_chunk = (static_cast<unsigned int>(v) >> (8 * i)) & 0xFFu;

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
        // Fallback for non-NVIDIA GPUs (AMD/Moore Threads)
        // Spread 2-bit values into bytes and decode using bit arithmetic
        // Step 1: Spread bits [7:0] into 4 bytes at positions [1:0] of each byte
        // t will have: byte0=bits[1:0], byte1=bits[3:2], byte2=bits[5:4], byte3=bits[7:6]
        unsigned int t = weight_chunk;
        t = (t | (t << 12)) & 0x000F000F;
        t = (t | (t << 6)) & 0x03030303;

        // Extract low bit (l) and high bit (h) of each 2-bit value
        // l=0,h=0 (val=0): should decode to real=-1, imag=0
        // l=1,h=0 (val=1): should decode to real=1, imag=0
        // l=0,h=1 (val=2): should decode to real=0, imag=-1
        // l=1,h=1 (val=3): should decode to real=0, imag=1
        unsigned int l = t & 0x01010101;          // low bit of each 2-bit value
        unsigned int h = (t >> 1) & 0x01010101;   // high bit of each 2-bit value

        // For real part: val=0->-1(0xFF), val=1->1(0x01), val=2,3->0
        // When l=0: we want -1 (0xFF), when l=1: we want 1 (0x01), when h=1: we want 0
        // mask_l: l=0 -> 0xFF, l=1 -> 0x00
        unsigned int mask_l = (l ^ 0x01010101) * 0xFF;
        // val: when l=0 -> 0xFF, when l=1 -> 0x01
        unsigned int val = mask_l | 0x01010101;
        // mask_imag: h=1 -> 0xFF, h=0 -> 0x00 (determines if value goes to imag)
        unsigned int mask_imag = h * 0xFF;
        // mask_real: h=0 -> 0xFF, h=1 -> 0x00 (determines if value goes to real)
        unsigned int mask_real = ~mask_imag;

        int weight_real = static_cast<int>(val & mask_real);
        int weight_imag = static_cast<int>(val & mask_imag);
#else
        // NVIDIA CUDA: Use __byte_perm for register-based LUT lookup (Zero Memory Access)
        // Pack w0,w1,w2,w3 into 4 nibbles for __byte_perm selector
        // nibble0 = w0 (bits [1:0]), nibble1 = w1 (bits [3:2])
        // nibble2 = w2 (bits [5:4]), nibble3 = w3 (bits [7:6])
        const unsigned int sel =
              ((weight_chunk & 0x03u)      )
            | ((weight_chunk & 0x0Cu) << 2)
            | ((weight_chunk & 0x30u) << 4)
            | ((weight_chunk & 0xC0u) << 6);

        int weight_real = __byte_perm(lut_real, 0, sel);
        int weight_imag = __byte_perm(lut_imag, 0, sel);
#endif

        // dp4a computes: sum += dot_product(weight_bytes, activation_bytes)
        // Each call processes 4 int8 values in parallel
        // u[i]       = 4 real activations (int8 packed in int32)
        // u[i+QR2_K] = 4 imag activations (int8 packed in int32)
        sumi_realreal = ggml_cuda_dp4a(u[i],       weight_real, sumi_realreal);
        sumi_realimag = ggml_cuda_dp4a(u[i+QR2_K], weight_real, sumi_realimag);
        sumi_imagreal = ggml_cuda_dp4a(u[i],       weight_imag, sumi_imagreal);
        sumi_imagimag = ggml_cuda_dp4a(u[i+QR2_K], weight_imag, sumi_imagimag);
    }

    const float result_real = sumi_realreal * d8[0] + sumi_imagimag * d8[3];
    const float result_imag = sumi_imagreal * d8[2] - sumi_realimag * d8[1];

    const __nv_bfloat16 result_real_bf16 = __float2bfloat16(result_real);
    const __nv_bfloat16 result_imag_bf16 = __float2bfloat16(result_imag);

    const unsigned int packed =
          static_cast<unsigned int>(*reinterpret_cast<const uint16_t *>(&result_real_bf16))
        | (static_cast<unsigned int>(*reinterpret_cast<const uint16_t *>(&result_imag_bf16)) << 16);

    return __int_as_float(static_cast<int>(packed));
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
