#pragma once

#include "ggml.h"
#include "ggml-common.h"
#include "ggml-impl.h"
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>


typedef struct {
    void* qlut;  // 存储 LUT (每个块 16384 字节)
    float* lut_scales;
    int K;
    size_t lut_size;
    size_t scales_size;
} ifairy_tensor_extra;

// 用来存量化后的信息

typedef struct {
    int8_t x_real[QK_K];
    int8_t x_imag[QK_K];
    float d_real;
    float d_imag;
} block_ifairy_q16_internal;

// 同时填充 2 个 LUT 的辅助函数 (AVX2)
// 该函数根据输入的实部和虚部 (ar, ai)，结合预定义的符号掩码 (SIG)，
// 生成用于后续查表的 LUT 数据。
// 它利用 AVX2 指令同时处理两组数据 (Pair 0 和 Pair 1)。
static inline void fill_luts_v3_avx2(
    int8_t ar0_0, int8_t ai0_0, int8_t ar1_0, int8_t ai1_0,
    int8_t ar0_1, int8_t ai0_1, int8_t ar1_1, int8_t ai1_1,
    __m256i* ymm_ac, __m256i* ymm_ad, __m256i* ymm_bc, __m256i* ymm_bd
) {
    // 1. 定义符号掩码 (Sign Masks)
    // 这些掩码定义了输入值 (ar, ai) 在 LUT 中的组合方式。
    // 对应于量化权重可能取的不同值 (例如 -1, 1, 0 等)。
    // _mm_setr_epi8 是按字节顺序设置 (低位到高位)。
    const __m128i SIG0_AC = _mm_setr_epi8(-1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0);
    const __m128i SIG1_AC = _mm_setr_epi8(-1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m128i SIG0_BC = _mm_setr_epi8(0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1);
    const __m128i SIG1_BC = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1);

    // 将 128 位掩码广播到 256 位，以便同时处理两组数据
    __m256i v_SIG0_AC = _mm256_broadcastsi128_si256(SIG0_AC);
    __m256i v_SIG1_AC = _mm256_broadcastsi128_si256(SIG1_AC);
    __m256i v_SIG0_BC = _mm256_broadcastsi128_si256(SIG0_BC);
    __m256i v_SIG1_BC = _mm256_broadcastsi128_si256(SIG1_BC);

    // 将标量输入广播到向量寄存器中。
    // 低 128 位存储 Pair 0 的数据 (后缀 _0)，高 128 位存储 Pair 1 的数据 (后缀 _1)。
    // _mm_set1_epi8 将一个 int8 复制填充到整个 128 位寄存器。
    // _mm256_set_m128i 将两个 128 位寄存器组合成一个 256 位寄存器 (高位在前，低位在后)。
    __m256i v_ar0 = _mm256_set_m128i(_mm_set1_epi8(ar0_1), _mm_set1_epi8(ar0_0));
    __m256i v_ai0 = _mm256_set_m128i(_mm_set1_epi8(ai0_1), _mm_set1_epi8(ai0_0));
    __m256i v_ar1 = _mm256_set_m128i(_mm_set1_epi8(ar1_1), _mm_set1_epi8(ar1_0));
    __m256i v_ai1 = _mm256_set_m128i(_mm_set1_epi8(ai1_1), _mm_set1_epi8(ai1_0));

    // 3. 计算 LUT 值
    // 使用 _mm256_sign_epi8 根据 SIG 掩码的符号对输入值进行操作：
    //   如果 SIG < 0，结果为 -val
    //   如果 SIG > 0，结果为 val
    //   如果 SIG = 0，结果为 0
    // 然后将两部分结果相加，得到最终的线性组合。
    // 例如：*ymm_ac = (ar0 * sgn(SIG0)) + (ar1 * sgn(SIG1))
    *ymm_ac = _mm256_add_epi8(_mm256_sign_epi8(v_ar0, v_SIG0_AC), _mm256_sign_epi8(v_ar1, v_SIG1_AC));
    *ymm_ad = _mm256_add_epi8(_mm256_sign_epi8(v_ai0, v_SIG0_AC), _mm256_sign_epi8(v_ai1, v_SIG1_AC));
    *ymm_bc = _mm256_add_epi8(_mm256_sign_epi8(v_ar0, v_SIG0_BC), _mm256_sign_epi8(v_ar1, v_SIG1_BC));
    *ymm_bd = _mm256_add_epi8(_mm256_sign_epi8(v_ai0, v_SIG0_BC), _mm256_sign_epi8(v_ai1, v_SIG1_BC));
}

static void generate_luts_ifairy_v3(int K, const float* src_real, const float* src_imag, uint8_t* dst_lut_buffer) {
    const int num_blocks = (K + QK_K - 1) / QK_K;
    const size_t lut_block_size = 16384; // 标准大小
    
    for (int b = 0; b < num_blocks; ++b) {
        // 每256行进行一次量化
        // 量化到临时缓冲区
        // 压缩到[-31, 31]防止溢出
        block_ifairy_q16_internal ab;
        float max_real = 0.0f;
        float max_imag = 0.0f;
        
        int start_k = b * QK_K;
        int end_k = (start_k + QK_K > K) ? K : start_k + QK_K;
        
        for (int i = start_k; i < end_k; ++i) {
            float abs_r = fabsf(src_real[i]);
            float abs_i = fabsf(src_imag[i]);
            if (abs_r > max_real) max_real = abs_r; 
            if (abs_i > max_imag) max_imag = abs_i;
        }
        
        float scale_real = max_real / 127.0f;
        float scale_imag = max_imag / 127.0f;
        if (scale_real == 0) scale_real = 1.0f;
        if (scale_imag == 0) scale_imag = 1.0f;
        
        ab.d_real = scale_real;
        ab.d_imag = scale_imag;
        
        float inv_real = 1.0f / scale_real;
        float inv_imag = 1.0f / scale_imag;
        
        for (int i = 0; i < QK_K; ++i) {
            int k_idx = start_k + i;
            if (k_idx >= K) {
                ab.x_real[i] = 0;
                ab.x_imag[i] = 0;
                continue;
            }
            
            int val_r = (int)roundf(src_real[k_idx] * inv_real);
            int val_i = (int)roundf(src_imag[k_idx] * inv_imag);
            // 这个地方把real和imag右移两位到[-31, 31]，使得后续运算时不会溢出，在最后的计算结果用乘以4.0予以补偿
            val_r >>= 2;
            val_i >>= 2;
            
            ab.x_real[i] = (int8_t)(val_r > 31 ? 31 : (val_r < -31 ? -31 : val_r));
            ab.x_imag[i] = (int8_t)(val_i > 31 ? 31 : (val_i < -31 ? -31 : val_i));
        }

        // 2. 生成 LUT
        uint8_t* block_lut_base = dst_lut_buffer + b * lut_block_size;
        
        float* s_ptr = (float*)block_lut_base;
        s_ptr[0] = ab.d_real;
        s_ptr[1] = ab.d_imag;
        
        // 计算lut，从偏移量 64 开始
        __m256i* lut_ptr = (__m256i*)(block_lut_base + 64);
        
        for (int k = 0; k < QK_K; k += 4) { // 一次性处理4个复数，分别取出实部虚部
            int8_t ar0_0 = ab.x_real[k], ai0_0 = ab.x_imag[k];
            int8_t ar1_0 = ab.x_real[k+1], ai1_0 = ab.x_imag[k+1];
            int8_t ar0_1 = ab.x_real[k+2], ai0_1 = ab.x_imag[k+2];
            int8_t ar1_1 = ab.x_real[k+3], ai1_1 = ab.x_imag[k+3];
            
            // 拆成ac ad bc bd，因为之前量化的时候部虚部是分开量化的
            __m256i yac, yad, ybc, ybd;
            fill_luts_v3_avx2(ar0_0, ai0_0, ar1_0, ai1_0, ar0_1, ai0_1, ar1_1, ai1_1, &yac, &yad, &ybc, &ybd);
            
            // 存储打包数据 (Pair0 在低 128 位，Pair1 在高 128 位)
            _mm256_storeu_si256(lut_ptr++, yac);
            _mm256_storeu_si256(lut_ptr++, yad);
            _mm256_storeu_si256(lut_ptr++, ybc);
            _mm256_storeu_si256(lut_ptr++, ybd);
        }
    }
}

static void ggml_ifairy_preprocessor(int M, int K,
                               const float* real_input,
                               const float* imag_input,
                               float* lut_scales,
                               void* qlut_buffer) {
    (void)M;

    uint8_t* lut_buffer = (uint8_t*)qlut_buffer;
    generate_luts_ifairy_v3(K, real_input, imag_input, lut_buffer);
    if (lut_scales != NULL) lut_scales[0] = 1.0f;
}

void ggml_ifairy_qgemm_lut_v3(
    int M, int K,
    const void* weights_ptr,
    const void* qlut_buffer,
    void* dst_ptr,
    int ith, int nth
) {
    const uint8_t* weight_base = (const uint8_t*)weights_ptr;
    const uint8_t* lut_base = (const uint8_t*)qlut_buffer;
    ggml_bf16_t* dst_bf16 = (ggml_bf16_t*)dst_ptr;
    
    const size_t block_size_bytes = sizeof(block_ifairy);
    const size_t lut_block_size = 16384;
    const int num_act_blocks = (K + QK_K - 1) / QK_K;
    
    int m_per_thread = (M + nth - 1) / nth;
    int m_start = ith * m_per_thread;
    int m_end = m_start + m_per_thread;
    if (m_end > M) m_end = M;
    if (m_start >= m_end) return;
    
    const __m256i mask_low = _mm256_set1_epi8(0x0F);
    const __m256 scale_4 = _mm256_set1_ps(4.0f);
    
    int m = m_start;
    // 一次处理 32 行
    for (; m <= m_end - 32; m += 32) {
        __m256 final_real_0 = _mm256_setzero_ps();
        __m256 final_imag_0 = _mm256_setzero_ps();
        __m256 final_real_1 = _mm256_setzero_ps();
        __m256 final_imag_1 = _mm256_setzero_ps();
        __m256 final_real_2 = _mm256_setzero_ps();
        __m256 final_imag_2 = _mm256_setzero_ps();
        __m256 final_real_3 = _mm256_setzero_ps();
        __m256 final_imag_3 = _mm256_setzero_ps();
        
        for (int blk = 0; blk < num_act_blocks; ++blk) {
            const uint8_t* block_lut = lut_base + blk * lut_block_size;
            const float* s_ptr = (const float*)block_lut;
            float d_real = s_ptr[0];
            float d_imag = s_ptr[1];
            
            const uint8_t* ptr_lut = block_lut + 64;
            
            // 32 行的累加器
            __m256i sum_ac_L = _mm256_setzero_si256();
            __m256i sum_ac_H = _mm256_setzero_si256();
            __m256i sum_ad_L = _mm256_setzero_si256();
            __m256i sum_ad_H = _mm256_setzero_si256();
            __m256i sum_bc_L = _mm256_setzero_si256();
            __m256i sum_bc_H = _mm256_setzero_si256();
            __m256i sum_bd_L = _mm256_setzero_si256();
            __m256i sum_bd_H = _mm256_setzero_si256();
            
            size_t row_stride = num_act_blocks * block_size_bytes;
            const uint8_t* w_ptr_base = weight_base + blk * block_size_bytes;

            // k 展开 16 (一次处理 4 字节权重?)
            for (int k = 0; k < QK_K; k += 16) {
                if (blk * QK_K + k >= K) break;
                
                uint32_t w_cols[32];
                const uint32_t* w_col_ptr = (const uint32_t*)(w_ptr_base + (k / 4));
                
                for(int r=0; r<32; ++r) {
                    w_cols[r] = *(const uint32_t*)((const uint8_t*)w_col_ptr + (m + r) * row_stride);
                }
                
                __m256i w_raw_0 = _mm256_loadu_si256((const __m256i*)&w_cols[0]);
                __m256i w_raw_1 = _mm256_loadu_si256((const __m256i*)&w_cols[8]);
                __m256i w_raw_2 = _mm256_loadu_si256((const __m256i*)&w_cols[16]);
                __m256i w_raw_3 = _mm256_loadu_si256((const __m256i*)&w_cols[24]);
                
                #define ACCUMULATE_PAIR(lut_ptr_offset, idx_p0, idx_p1, sum_L, sum_H) \
                    do { \
                        /* 广播 LUT 数据 */ \
                        __m256i lut_p0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(ptr_lut + lut_ptr_offset))); \
                        __m256i lut_p1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(ptr_lut + lut_ptr_offset + 16))); \
                        /* 使用 shuffle 查找值 */ \
                        __m256i val_p0 = _mm256_shuffle_epi8(lut_p0, idx_p0); \
                        __m256i val_p1 = _mm256_shuffle_epi8(lut_p1, idx_p1); \
                        /* 累加 */ \
                        __m256i val = _mm256_add_epi8(val_p0, val_p1); \
                        sum_L = _mm256_add_epi16(sum_L, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(val))); \
                        sum_H = _mm256_add_epi16(sum_H, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(val, 1))); \
                    } while(0)

                #define PROCESS_K_GROUP(byte_idx) \
                    do { \
                        __m256i shuf_mask = _mm256_set_epi8( \
                            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12+byte_idx, 8+byte_idx, 4+byte_idx, 0+byte_idx, \
                            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12+byte_idx, 8+byte_idx, 4+byte_idx, 0+byte_idx \
                        ); \
                        \
                        __m256i b0 = _mm256_shuffle_epi8(w_raw_0, shuf_mask); \
                        __m256i b1 = _mm256_shuffle_epi8(w_raw_1, shuf_mask); \
                        __m256i b2 = _mm256_shuffle_epi8(w_raw_2, shuf_mask); \
                        __m256i b3 = _mm256_shuffle_epi8(w_raw_3, shuf_mask); \
                        \
                        /* 重新排列数据以匹配 LUT 访问模式 */ \
                        __m256i t0 = _mm256_permute4x64_epi64(b0, 0xD8); \
                        __m256i t1 = _mm256_permute4x64_epi64(b1, 0xD8); \
                        __m128 t0_low = _mm_castsi128_ps(_mm256_castsi256_si128(t0)); \
                        __m128 t1_low = _mm_castsi128_ps(_mm256_castsi256_si128(t1)); \
                        __m128i w_L_128 = _mm_castps_si128(_mm_shuffle_ps(t0_low, t1_low, 0x88)); \
                        \
                        __m256i t2 = _mm256_permute4x64_epi64(b2, 0xD8); \
                        __m256i t3 = _mm256_permute4x64_epi64(b3, 0xD8); \
                        __m128 t2_low = _mm_castsi128_ps(_mm256_castsi256_si128(t2)); \
                        __m128 t3_low = _mm_castsi128_ps(_mm256_castsi256_si128(t3)); \
                        __m128i w_H_128 = _mm_castps_si128(_mm_shuffle_ps(t2_low, t3_low, 0x88)); \
                        \
                        /* 组合低位和高位索引 */ \
                        __m256i indices = _mm256_inserti128_si256(_mm256_castsi128_si256(w_L_128), w_H_128, 1); \
                        __m256i idx_p0 = _mm256_and_si256(indices, mask_low); \
                        __m256i idx_p1 = _mm256_and_si256(_mm256_srli_epi16(indices, 4), mask_low); \
                        \
                        ACCUMULATE_PAIR(0, idx_p0, idx_p1, sum_ac_L, sum_ac_H); \
                        ACCUMULATE_PAIR(32, idx_p0, idx_p1, sum_ad_L, sum_ad_H); \
                        ACCUMULATE_PAIR(64, idx_p0, idx_p1, sum_bc_L, sum_bc_H); \
                        ACCUMULATE_PAIR(96, idx_p0, idx_p1, sum_bd_L, sum_bd_H); \
                        ptr_lut += 128; \
                    } while(0)

                PROCESS_K_GROUP(0);
                PROCESS_K_GROUP(1);
                PROCESS_K_GROUP(2);
                PROCESS_K_GROUP(3);
                
                #undef PROCESS_K_GROUP
                #undef ACCUMULATE_PAIR
            }
            
            float w_scales_r[32], w_scales_i[32];
            for(int r=0; r<32; ++r) {
                const float* s_ptr = (const float*)(weight_base + (m+r) * row_stride + blk * block_size_bytes + 64);
                w_scales_r[r] = s_ptr[0];
                w_scales_i[r] = s_ptr[1];
            }
            
            __m256 a_sr = _mm256_set1_ps(d_real);
            __m256 a_si = _mm256_set1_ps(d_imag);

            #define PROCESS_16_ROWS(sum_ac, sum_ad, sum_bc, sum_bd, w_sr_ptr, w_si_ptr, f_r0, f_r1, f_i0, f_i1) \
                do { \
                    /* 将 16 位累加器转换为 32 位浮点数 */ \
                    __m256 fac0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum_ac))); \
                    __m256 fac1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_ac, 1))); \
                    __m256 fad0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum_ad))); \
                    __m256 fad1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_ad, 1))); \
                    __m256 fbc0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum_bc))); \
                    __m256 fbc1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_bc, 1))); \
                    __m256 fbd0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum_bd))); \
                    __m256 fbd1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_bd, 1))); \
                    /* 加载权重缩放因子 */ \
                    __m256 w_sr0 = _mm256_loadu_ps(w_sr_ptr); \
                    __m256 w_sr1 = _mm256_loadu_ps(w_sr_ptr + 8); \
                    __m256 w_si0 = _mm256_loadu_ps(w_si_ptr); \
                    __m256 w_si1 = _mm256_loadu_ps(w_si_ptr + 8); \
                    /* 计算实部 */ \
                    __m256 r0 = _mm256_sub_ps(_mm256_mul_ps(fac0, _mm256_mul_ps(w_sr0, a_sr)), _mm256_mul_ps(fbd0, _mm256_mul_ps(w_si0, a_si))); \
                    __m256 r1 = _mm256_sub_ps(_mm256_mul_ps(fac1, _mm256_mul_ps(w_sr1, a_sr)), _mm256_mul_ps(fbd1, _mm256_mul_ps(w_si1, a_si))); \
                    f_r0 = _mm256_add_ps(f_r0, _mm256_mul_ps(r0, scale_4)); \
                    f_r1 = _mm256_add_ps(f_r1, _mm256_mul_ps(r1, scale_4)); \
                    /* 计算虚部 */ \
                    __m256 i0 = _mm256_add_ps(_mm256_mul_ps(fbc0, _mm256_mul_ps(w_si0, a_sr)), _mm256_mul_ps(fad0, _mm256_mul_ps(w_sr0, a_si))); \
                    __m256 i1 = _mm256_add_ps(_mm256_mul_ps(fbc1, _mm256_mul_ps(w_si1, a_sr)), _mm256_mul_ps(fad1, _mm256_mul_ps(w_sr1, a_si))); \
                    f_i0 = _mm256_add_ps(f_i0, _mm256_mul_ps(i0, scale_4)); \
                    f_i1 = _mm256_add_ps(f_i1, _mm256_mul_ps(i1, scale_4)); \
                } while(0)

            PROCESS_16_ROWS(sum_ac_L, sum_ad_L, sum_bc_L, sum_bd_L, w_scales_r, w_scales_i, final_real_0, final_real_1, final_imag_0, final_imag_1);
            PROCESS_16_ROWS(sum_ac_H, sum_ad_H, sum_bc_H, sum_bd_H, w_scales_r + 16, w_scales_i + 16, final_real_2, final_real_3, final_imag_2, final_imag_3);
            #undef PROCESS_16_ROWS
        }
        
        float res_r[32], res_i[32];
        _mm256_storeu_ps(res_r, final_real_0);
        _mm256_storeu_ps(res_r + 8, final_real_1);
        _mm256_storeu_ps(res_r + 16, final_real_2);
        _mm256_storeu_ps(res_r + 24, final_real_3);
        _mm256_storeu_ps(res_i, final_imag_0);
        _mm256_storeu_ps(res_i + 8, final_imag_1);
        _mm256_storeu_ps(res_i + 16, final_imag_2);
        _mm256_storeu_ps(res_i + 24, final_imag_3);
        
        for(int r=0; r<32; ++r) {
            dst_bf16[(m + r) * 2 + 0] = GGML_FP32_TO_BF16(res_r[r]);
            dst_bf16[(m + r) * 2 + 1] = GGML_FP32_TO_BF16(res_i[r]);
        }
    }
    
    // 剩余行的回退处理 (标量)
    for (; m < m_end; ++m) {
        float sum_real = 0.0f;
        float sum_imag = 0.0f;
        
        for (int blk = 0; blk < num_act_blocks; ++blk) {
            const uint8_t* block_lut = lut_base + blk * lut_block_size;
            const float* s_ptr_lut = (const float*)block_lut;
            float d_real = s_ptr_lut[0];
            float d_imag = s_ptr_lut[1];
            
            const int8_t* ptr_ac = (const int8_t*)(block_lut + 64);
            const int8_t* ptr_ad = ptr_ac + 32;
            const int8_t* ptr_bc = ptr_ac + 64;
            const int8_t* ptr_bd = ptr_ac + 96;
            
            int32_t ac = 0, ad = 0, bc = 0, bd = 0;
            
            size_t row_stride = num_act_blocks * block_size_bytes;
            const uint8_t* w_ptr_base = weight_base + m * row_stride + blk * block_size_bytes;
            
            for (int k = 0; k < QK_K; k += 4) {
                if (blk * QK_K + k >= K) break;
                uint8_t w_byte = *(w_ptr_base + (k / 4));
                
                int idx0 = w_byte & 0x0F;
                int idx1 = (w_byte >> 4) & 0x0F;
                
                ac += ptr_ac[idx0] + ptr_ac[16 + idx1];
                ad += ptr_ad[idx0] + ptr_ad[16 + idx1];
                bc += ptr_bc[idx0] + ptr_bc[16 + idx1];
                bd += ptr_bd[idx0] + ptr_bd[16 + idx1];
                
                ptr_ac += 128;
                ptr_ad += 128;
                ptr_bc += 128;
                ptr_bd += 128;
            }
            
            const float* s_ptr = (const float*)(w_ptr_base + 64);
            float w_sr = s_ptr[0];
            float w_si = s_ptr[1];

            float a_sr = d_real;
            float a_si = d_imag;
            
            sum_real += (ac * w_sr * a_sr - bd * w_si * a_si) * 4.0f;
            sum_imag += (bc * w_si * a_sr + ad * w_sr * a_si) * 4.0f;

            if (m == 0 && blk == 0) {
                printf("DEBUG: QGEMM Scalar. ac=%d, bd=%d, bc=%d, ad=%d. w_sr=%f, w_si=%f, a_sr=%f, a_si=%f. sum_r=%f, sum_i=%f\n", 
                       ac, bd, bc, ad, w_sr, w_si, a_sr, a_si, sum_real, sum_imag);
            }
        }
        
        dst_bf16[m * 2 + 0] = GGML_FP32_TO_BF16(sum_real);
        dst_bf16[m * 2 + 1] = GGML_FP32_TO_BF16(sum_imag);
    }
}

// 分配lut空间
static void ggml_ifairy_transform_tensor(struct ggml_tensor* tensor) {
    if (tensor->extra != NULL) return;
    
    const int M = (int)tensor->ne[1]; // 列
    const int K = (int)tensor->ne[0]; // 行
    const int num_act_blocks = (K + QK_K - 1) / QK_K;
    
    ifairy_tensor_extra* extra = (ifairy_tensor_extra*)malloc(sizeof(ifairy_tensor_extra));
    
    // 为预计算 LUT 分配缓冲区 (每块 16384 字节)
    const size_t lut_block_size = 16384;
    extra->qlut = (void*)malloc(num_act_blocks * lut_block_size);
    extra->lut_scales = (float*)malloc(sizeof(float));  // 虚拟分配
    extra->K = K;
    extra->lut_size = num_act_blocks * lut_block_size;
    extra->scales_size = sizeof(float);
    
    tensor->extra = extra;
}
