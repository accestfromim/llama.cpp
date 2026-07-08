#include "lut-qgemm.h"

#include "ggml-fairy2i-lut-impl.h"
#include "ggml-fairy2i-lut.h"
#include "ggml-cpu.h"
#include "quants.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__aarch64__) && defined(GGML_USE_FAIRY2I_CPU_ARM_SVE2)
static bool ggml_fairy2i_lut_test_disable_arm_sve2(void) {
    const char * env = getenv("GGML_FAIRY2I_TEST_DISABLE_ARM_SVE2");
    return env && strcmp(env, "0") != 0;
}
#endif

const char * ggml_fairy2i_tile64_lut_qgemm_four_cpu_path_name(void) {
#if defined(__AVX2__)
    return "lut16";
#elif defined(__aarch64__) && defined(GGML_USE_FAIRY2I_CPU_ARM_SVE2)
    if (!ggml_fairy2i_lut_test_disable_arm_sve2() && ggml_cpu_has_sve2()) {
        return "lut16_sve2";
    }
#endif

    return "lut16";
}

#if defined(__AVX2__)
#    include <immintrin.h>

static inline __m256 ggml_fairy2i_tile64_lut_load_scale8(const ggml_half * src) {
#    if defined(__F16C__)
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) src));
#    else
    alignas(32) float tmp[8];
    for (int i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP16_TO_FP32(src[i]);
    }
    return _mm256_load_ps(tmp);
#    endif
}

static inline __m256 ggml_fairy2i_tile64_lut_madd(const __m256 & x, const __m256 & y, const __m256 & acc) {
#    if defined(__FMA__)
    return _mm256_fmadd_ps(x, y, acc);
#    else
    return _mm256_add_ps(_mm256_mul_ps(x, y), acc);
#    endif
}

static inline void ggml_fairy2i_tile64_lut_accumulate_pair_channels(const fairy2i_tile64_lut_wtile_16 * wt0,
                                                              const fairy2i_tile64_lut_wtile_16 * wt1,
                                                              const int8_t *                lut_blk,
                                                              __m256i &                     sum0_r_lo,
                                                              __m256i &                     sum0_r_hi,
                                                              __m256i &                     sum1_r_lo,
                                                              __m256i &                     sum1_r_hi,
                                                              __m256i &                     sum0_i_lo,
                                                              __m256i &                     sum0_i_hi,
                                                              __m256i &                     sum1_i_lo,
                                                              __m256i &                     sum1_i_hi) {
    sum0_r_lo = _mm256_setzero_si256();
    sum0_r_hi = _mm256_setzero_si256();
    sum1_r_lo = _mm256_setzero_si256();
    sum1_r_hi = _mm256_setzero_si256();
    sum0_i_lo = _mm256_setzero_si256();
    sum0_i_hi = _mm256_setzero_si256();
    sum1_i_lo = _mm256_setzero_si256();
    sum1_i_hi = _mm256_setzero_si256();
    const __m256i one      = _mm256_set1_epi8(1);
    const __m256i mask_idx = _mm256_set1_epi8(0x0f);

    for (int byte_idx = 0; byte_idx < QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK / 2; ++byte_idx) {
        const int8_t * lut_base = lut_blk + (size_t) byte_idx * 2u * k_fairy2i_lut_group_bytes;
        const __m256i lut_0_r   = _mm256_loadu_si256((const __m256i *) lut_base);
        const __m256i lut_1_r   = _mm256_loadu_si256((const __m256i *) (lut_base + 64));
        const __m256i lut_0_i   = _mm256_loadu_si256((const __m256i *) (lut_base + 32));
        const __m256i lut_1_i   = _mm256_loadu_si256((const __m256i *) (lut_base + 96));

#    define GGML_FAIRY2I_TILE64_LUT_ACCUMULATE(wt, suffix)                                                               \
        do {                                                                                                       \
            const __m256i packed = _mm256_broadcastsi128_si256(                                                    \
                _mm_load_si128((const __m128i *) &(wt)->qs[byte_idx]));                                            \
            const __m256i idx_lo = _mm256_and_si256(packed, mask_idx);                                             \
            const __m256i idx_hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_idx);                       \
            const __m256i out_0_r = _mm256_shuffle_epi8(lut_0_r, idx_lo);                                          \
            const __m256i out_1_r = _mm256_shuffle_epi8(lut_1_r, idx_hi);                                          \
            sum##suffix##_r_lo = _mm256_add_epi16(                                                                \
                sum##suffix##_r_lo, _mm256_maddubs_epi16(one, _mm256_unpacklo_epi8(out_0_r, out_1_r)));           \
            sum##suffix##_r_hi = _mm256_add_epi16(                                                                \
                sum##suffix##_r_hi, _mm256_maddubs_epi16(one, _mm256_unpackhi_epi8(out_0_r, out_1_r)));           \
            const __m256i out_0_i = _mm256_shuffle_epi8(lut_0_i, idx_lo);                                          \
            const __m256i out_1_i = _mm256_shuffle_epi8(lut_1_i, idx_hi);                                          \
            sum##suffix##_i_lo = _mm256_add_epi16(                                                                \
                sum##suffix##_i_lo, _mm256_maddubs_epi16(one, _mm256_unpacklo_epi8(out_0_i, out_1_i)));           \
            sum##suffix##_i_hi = _mm256_add_epi16(                                                                \
                sum##suffix##_i_hi, _mm256_maddubs_epi16(one, _mm256_unpackhi_epi8(out_0_i, out_1_i)));           \
        } while (false)

        GGML_FAIRY2I_TILE64_LUT_ACCUMULATE(wt0, 0);
        GGML_FAIRY2I_TILE64_LUT_ACCUMULATE(wt1, 1);
#    undef GGML_FAIRY2I_TILE64_LUT_ACCUMULATE
    }
}

static inline void ggml_fairy2i_tile64_lut_apply_pair_component(const fairy2i_tile64_lut_wtile_16 * wt0,
                                                          const fairy2i_tile64_lut_wtile_16 * wt1,
                                                          const __m256i &               sum0_lo,
                                                          const __m256i &               sum0_hi,
                                                          const __m256i &               sum1_lo,
                                                          const __m256i &               sum1_hi,
                                                          const ggml_half *             scale_a0,
                                                          const ggml_half *             scale_b0,
                                                          const ggml_half *             scale_a1,
                                                          const ggml_half *             scale_b1,
                                                          const __m256 &                v_a,
                                                          const __m256 &                v_b,
                                                          __m256 &                      acc_lo,
                                                          __m256 &                      acc_hi) {
    GGML_UNUSED(wt0);
    GGML_UNUSED(wt1);
    for (int half = 0; half < 2; ++half) {
        const __m128i s0_a = half == 0 ? _mm256_castsi256_si128(sum0_lo) : _mm256_castsi256_si128(sum0_hi);
        const __m128i s0_b = half == 0 ? _mm256_extracti128_si256(sum0_lo, 1) : _mm256_extracti128_si256(sum0_hi, 1);
        const __m128i s1_a = half == 0 ? _mm256_castsi256_si128(sum1_lo) : _mm256_castsi256_si128(sum1_hi);
        const __m128i s1_b = half == 0 ? _mm256_extracti128_si256(sum1_lo, 1) : _mm256_extracti128_si256(sum1_hi, 1);
        __m256 & acc = half == 0 ? acc_lo : acc_hi;
        const int offset = half * 8;

        acc = ggml_fairy2i_tile64_lut_madd(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(s0_a)),
                                     _mm256_mul_ps(v_a, ggml_fairy2i_tile64_lut_load_scale8(scale_a0 + offset)), acc);
        acc = ggml_fairy2i_tile64_lut_madd(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(s0_b)),
                                     _mm256_mul_ps(v_b, ggml_fairy2i_tile64_lut_load_scale8(scale_b0 + offset)), acc);
        acc = ggml_fairy2i_tile64_lut_madd(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(s1_a)),
                                     _mm256_mul_ps(v_a, ggml_fairy2i_tile64_lut_load_scale8(scale_a1 + offset)), acc);
        acc = ggml_fairy2i_tile64_lut_madd(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(s1_b)),
                                     _mm256_mul_ps(v_b, ggml_fairy2i_tile64_lut_load_scale8(scale_b1 + offset)), acc);
    }
}

static inline void ggml_fairy2i_tile64_lut_accumulate_pair(const fairy2i_tile64_lut_wtile_16 * wt0,
                                                     const fairy2i_tile64_lut_wtile_16 * wt1,
                                                     const int8_t *                lut_blk,
                                                     const __m256 &                v_lr,
                                                     const __m256 &                v_li,
                                                     __m256 &                      acc_r_lo,
                                                     __m256 &                      acc_r_hi,
                                                     __m256 &                      acc_i_lo,
                                                     __m256 &                      acc_i_hi) {
    __m256i sum0_r_lo;
    __m256i sum0_r_hi;
    __m256i sum1_r_lo;
    __m256i sum1_r_hi;
    __m256i sum0_i_lo;
    __m256i sum0_i_hi;
    __m256i sum1_i_lo;
    __m256i sum1_i_hi;

    ggml_fairy2i_tile64_lut_accumulate_pair_channels(wt0, wt1, lut_blk, sum0_r_lo, sum0_r_hi, sum1_r_lo, sum1_r_hi,
                                               sum0_i_lo, sum0_i_hi, sum1_i_lo, sum1_i_hi);
    ggml_fairy2i_tile64_lut_apply_pair_component(wt0, wt1, sum0_r_lo, sum0_r_hi, sum1_r_lo, sum1_r_hi, wt0->d_real,
                                           wt0->d_imag, wt1->d_real, wt1->d_imag, v_lr, v_li, acc_r_lo, acc_r_hi);
    ggml_fairy2i_tile64_lut_apply_pair_component(wt0, wt1, sum0_i_lo, sum0_i_hi, sum1_i_lo, sum1_i_hi, wt0->d_imag,
                                           wt0->d_real, wt1->d_imag, wt1->d_real, v_lr, v_li, acc_i_lo, acc_i_hi);
}

static inline void ggml_fairy2i_tile64_lut_store_pair(int            tile,
                                                int            m,
                                                uint8_t *      dst_col,
                                                size_t         dst_row_stride,
                                                bool           pack_bf16,
                                                bool           add,
                                                const __m256 & acc_r_lo,
                                                const __m256 & acc_r_hi,
                                                const __m256 & acc_i_lo,
                                                const __m256 & acc_i_hi) {
    alignas(32) float out_r[16];
    alignas(32) float out_i[16];
    _mm256_store_ps(out_r + 0, acc_r_lo);
    _mm256_store_ps(out_r + 8, acc_r_hi);
    _mm256_store_ps(out_i + 0, acc_i_lo);
    _mm256_store_ps(out_i + 8, acc_i_hi);

    for (int lane = 0; lane < 16 && (tile * 16 + lane) < m; ++lane) {
        uint8_t * out = dst_col + (size_t) (tile * 16 + lane) * dst_row_stride;
        if (pack_bf16) {
            const float prev_r = add ? ggml_bf16_to_fp32(((ggml_bf16_t *) out)[0]) : 0.0f;
            const float prev_i = add ? ggml_bf16_to_fp32(((ggml_bf16_t *) out)[1]) : 0.0f;
            ((ggml_bf16_t *) out)[0] = ggml_fp32_to_bf16(prev_r + out_r[lane]);
            ((ggml_bf16_t *) out)[1] = ggml_fp32_to_bf16(prev_i + out_i[lane]);
        } else {
            ((float *) out)[0] = (add ? ((float *) out)[0] : 0.0f) + out_r[lane];
            ((float *) out)[1] = (add ? ((float *) out)[1] : 0.0f) + out_i[lane];
        }
    }
}

#    if defined(_MSC_VER)
#        define GGML_FAIRY2I_NOINLINE __declspec(noinline)
#    else
#        define GGML_FAIRY2I_NOINLINE __attribute__((__noinline__))
#    endif

static GGML_FAIRY2I_NOINLINE void ggml_fairy2i_tile64_lut_qgemm_pair_avx2(int          m,
                                                                   int          k,
                                                                   int          n,
                                                                   const void * packed_wtiles0,
                                                                   const void * packed_wtiles1,
                                                                   const void * lut,
                                                                   const void * lut_scales,
                                                                   float *      dst,
                                                                   size_t       dst_col_stride,
                                                                   size_t       dst_row_stride,
                                                                   bool         pack_bf16,
                                                                   bool         negate_imag_scale,
                                                                   bool         add) {
    const int64_t blocks           = k / QK_FAIRY2I_TILE64;
    const int64_t groups_per_block = QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK;
    const int64_t groups           = blocks * groups_per_block;
    const int     tiles            = (m + 15) / 16;
    const auto *  wtiles0          = (const fairy2i_tile64_lut_wtile_16 *) packed_wtiles0;
    const auto *  wtiles1          = (const fairy2i_tile64_lut_wtile_16 *) packed_wtiles1;
    const __m256  imag_sign        = negate_imag_scale ? _mm256_set1_ps(-0.0f) : _mm256_setzero_ps();

    for (int col = 0; col < n; ++col) {
        const int8_t * lut_col = (const int8_t *) lut + (size_t) col * (size_t) groups * k_fairy2i_lut_group_bytes;
        const float * scales = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2u;
        uint8_t * dst_col = (uint8_t *) dst + (size_t) col * dst_col_stride;

        for (int tile = 0; tile < tiles; ++tile) {
            __m256 acc_r_lo = _mm256_setzero_ps();
            __m256 acc_r_hi = _mm256_setzero_ps();
            __m256 acc_i_lo = _mm256_setzero_ps();
            __m256 acc_i_hi = _mm256_setzero_ps();

            for (int64_t blk = 0; blk < blocks; ++blk) {
                const fairy2i_tile64_lut_wtile_16 * wt0 = wtiles0 + (size_t) tile * (size_t) blocks + (size_t) blk;
                const fairy2i_tile64_lut_wtile_16 * wt1 = wtiles1 + (size_t) tile * (size_t) blocks + (size_t) blk;
                const int8_t * lut_blk = lut_col + (size_t) blk * (size_t) groups_per_block * k_fairy2i_lut_group_bytes;
                const __m256 v_lr = _mm256_set1_ps(scales[blk * 2 + 0]);
                const __m256 v_li = _mm256_xor_ps(_mm256_set1_ps(scales[blk * 2 + 1]), imag_sign);
                ggml_fairy2i_tile64_lut_accumulate_pair(wt0, wt1, lut_blk, v_lr, v_li, acc_r_lo, acc_r_hi, acc_i_lo,
                                                  acc_i_hi);
            }
            ggml_fairy2i_tile64_lut_store_pair(tile, m, dst_col, dst_row_stride, pack_bf16, add, acc_r_lo, acc_r_hi,
                                         acc_i_lo, acc_i_hi);
        }
    }
}

#    undef GGML_FAIRY2I_NOINLINE
#endif

void ggml_fairy2i_tile64_lut_qgemm_pair_cpu(int          m,
                                      int          k,
                                      int          n,
                                      const void * packed_wtiles0,
                                      const void * packed_wtiles1,
                                      const void * lut,
                                      const void * lut_scales,
                                      float *      dst,
                                      size_t       dst_col_stride,
                                      size_t       dst_row_stride,
                                      bool         pack_bf16) {
    if (!packed_wtiles0 || !packed_wtiles1 || !dst || !lut || !lut_scales || m <= 0 || k <= 0 || n <= 0) {
        return;
    }

#if defined(__AVX2__)
    ggml_fairy2i_tile64_lut_qgemm_pair_avx2(m, k, n, packed_wtiles0, packed_wtiles1, lut, lut_scales, dst, dst_col_stride,
                                      dst_row_stride, pack_bf16, /*negate_imag_scale*/ false, /*add*/ false);
#else
    ggml_fairy2i_tile64_lut_qgemm_pair_lut16(m, k, n, packed_wtiles0, packed_wtiles1, lut, lut_scales, dst,
                                             dst_col_stride, dst_row_stride, pack_bf16,
                                             /*negate_imag_scale=*/false, /*add=*/false);
#endif
}

bool ggml_fairy2i_tile64_lut_qgemm_four_cpu(int          m,
                                      int          k,
                                      int          n,
                                      const void * packed_u0,
                                      const void * packed_u1,
                                      const void * packed_w0,
                                      const void * packed_w1,
                                      const void * lut,
                                      const void * lut_scales,
                                      float *      dst,
                                      size_t       dst_col_stride,
                                      size_t       dst_row_stride,
                                      bool         pack_bf16) {
    if (m == 0) {
        return true;
    }
    if (!packed_u0 || !packed_u1 || !packed_w0 || !packed_w1 || !dst || !lut || !lut_scales || m < 0 || k <= 0 ||
        n <= 0) {
        return false;
    }

#if defined(__AVX2__)
    // LUT preprocessing folds in conjugation. Negating the imaginary scale recovers U * x.
    ggml_fairy2i_tile64_lut_qgemm_pair_avx2(m, k, n, packed_u0, packed_u1, lut, lut_scales, dst, dst_col_stride,
                                      dst_row_stride, pack_bf16, /*negate_imag_scale*/ true, /*add*/ false);
    ggml_fairy2i_tile64_lut_qgemm_pair_avx2(m, k, n, packed_w0, packed_w1, lut, lut_scales, dst, dst_col_stride,
                                      dst_row_stride, pack_bf16, /*negate_imag_scale*/ false, /*add*/ true);
    return true;
#else
#    if defined(__aarch64__) && defined(GGML_USE_FAIRY2I_CPU_ARM_SVE2)
    if (!ggml_fairy2i_lut_test_disable_arm_sve2() && ggml_cpu_has_sve2()) {
        if (ggml_fairy2i_tile64_lut_qgemm_four_sve2(m, k, n, packed_u0, packed_u1, packed_w0, packed_w1, lut,
                                                    lut_scales, dst, dst_col_stride, dst_row_stride, pack_bf16)) {
            return true;
        }
    }
#    endif

    // LUT preprocessing folds in conjugation. Negating the imaginary scale recovers U * x.
    ggml_fairy2i_tile64_lut_qgemm_pair_lut16(m, k, n, packed_u0, packed_u1, lut, lut_scales, dst, dst_col_stride,
                                             dst_row_stride, pack_bf16, /*negate_imag_scale=*/true, /*add=*/false);
    ggml_fairy2i_tile64_lut_qgemm_pair_lut16(m, k, n, packed_w0, packed_w1, lut, lut_scales, dst, dst_col_stride,
                                             dst_row_stride, pack_bf16, /*negate_imag_scale=*/false, /*add=*/true);
    return true;
#endif
}
