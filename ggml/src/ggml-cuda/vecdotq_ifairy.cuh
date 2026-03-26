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

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        // Extract 8 bits containing 4 x 2-bit weights
        // Layout: [w3, w2, w1, w0] where each w is 2 bits
        // w0 is at bits [1:0], w1 at [3:2], w2 at [5:4], w3 at [7:6]
        int weight_chunk = (v >> (8 * i)) & 0xFF;

        // Decode 4 weights and pack into 32-bit integers for dp4a
        // Each weight is decoded to an 8-bit value stored in the corresponding byte
        // Encoding: 00->-1(0xFF), 01->1(0x01), 10->-i(0xFF for imag), 11->i(0x01 for imag)
        //
        // Byte layout in 32-bit integer (little-endian):
        // bits [7:0]   -> weight 0 (corresponds to activation bytes 0-3)
        // bits [15:8]  -> weight 1
        // bits [23:16] -> weight 2
        // bits [31:24] -> weight 3

        int weight_real = 0;
        int weight_imag = 0;

        // Weight 0: bits [1:0] of weight_chunk
        // Shift right 0, mask 0x03 -> position in byte 0 (shift left 0)
        int w0 = (weight_chunk >> 0) & 0x03;
        int r0 = (w0 == 0) ? 0xFF : (w0 == 1) ? 0x01 : 0x00;  // 00->-1, 01->1, else 0
        int i0 = (w0 == 2) ? 0xFF : (w0 == 3) ? 0x01 : 0x00;  // 10->-1, 11->1, else 0
        weight_real |= (r0 & 0xFF) << 0;
        weight_imag |= (i0 & 0xFF) << 0;

        // Weight 1: bits [3:2] of weight_chunk
        // Shift right 2, mask 0x03 -> position in byte 1 (shift left 8)
        int w1 = (weight_chunk >> 2) & 0x03;
        int r1 = (w1 == 0) ? 0xFF : (w1 == 1) ? 0x01 : 0x00;
        int i1 = (w1 == 2) ? 0xFF : (w1 == 3) ? 0x01 : 0x00;
        weight_real |= (r1 & 0xFF) << 8;
        weight_imag |= (i1 & 0xFF) << 8;

        // Weight 2: bits [5:4] of weight_chunk
        // Shift right 4, mask 0x03 -> position in byte 2 (shift left 16)
        int w2 = (weight_chunk >> 4) & 0x03;
        int r2 = (w2 == 0) ? 0xFF : (w2 == 1) ? 0x01 : 0x00;
        int i2 = (w2 == 2) ? 0xFF : (w2 == 3) ? 0x01 : 0x00;
        weight_real |= (r2 & 0xFF) << 16;
        weight_imag |= (i2 & 0xFF) << 16;

        // Weight 3: bits [7:6] of weight_chunk
        // Shift right 6, mask 0x03 -> position in byte 3 (shift left 24)
        int w3 = (weight_chunk >> 6) & 0x03;
        int r3 = (w3 == 0) ? 0xFF : (w3 == 1) ? 0x01 : 0x00;
        int i3 = (w3 == 2) ? 0xFF : (w3 == 3) ? 0x01 : 0x00;
        weight_real |= (r3 & 0xFF) << 24;
        weight_imag |= (i3 & 0xFF) << 24;

        // dp4a computes: sum += dot_product(weight_bytes, activation_bytes)
        // Each call processes 4 int8 values in parallel
        // u[i]       = 4 real activations (int8 packed in int32)
        // u[i+QR2_K] = 4 imag activations (int8 packed in int32)
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
