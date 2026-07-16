#include "lut-qgemm.h"

#include "ggml-fairy2i-lut-impl.h"
#include "ggml-fairy2i-lut.h"
#include "quants.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) && defined(__aarch64__)
#    include <arm_neon.h>
#endif

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

static inline float ggml_fairy2i_bundle_lut_scale_to_f32(const ggml_half value) {
#    if defined(__F16C__)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) value)));
#    else
    return GGML_FP16_TO_FP32(value);
#    endif
}

static inline __m256 ggml_fairy2i_tile64_lut_madd(const __m256 & x, const __m256 & y, const __m256 & acc) {
#    if defined(__FMA__)
    return _mm256_fmadd_ps(x, y, acc);
#    else
    return _mm256_add_ps(_mm256_mul_ps(x, y), acc);
#    endif
}

template <bool adjacent_codes>
static inline void ggml_fairy2i_tile64_lut_accumulate_pair_channels_codes_impl(const uint8_t * codes0,
                                                                               size_t          code_stride0,
                                                                               const uint8_t * codes1,
                                                                               size_t          code_stride1,
                                                                               const int8_t *  lut_blk,
                                                                               __m256i &       sum0_r_lo,
                                                                               __m256i &       sum0_r_hi,
                                                                               __m256i &       sum1_r_lo,
                                                                               __m256i &       sum1_r_hi,
                                                                               __m256i &       sum0_i_lo,
                                                                               __m256i &       sum0_i_hi,
                                                                               __m256i &       sum1_i_lo,
                                                                               __m256i &       sum1_i_hi) {
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

        __m256i packed0;
        __m256i packed1;
        if constexpr (adjacent_codes) {
            const __m256i pair = _mm256_loadu_si256((const __m256i *) (codes0 + (size_t) byte_idx * code_stride0));
            packed0            = _mm256_permute2x128_si256(pair, pair, 0x00);
            packed1            = _mm256_permute2x128_si256(pair, pair, 0x11);
        } else {
            packed0 = _mm256_broadcastsi128_si256(
                _mm_loadu_si128((const __m128i *) (codes0 + (size_t) byte_idx * code_stride0)));
            packed1 = _mm256_broadcastsi128_si256(
                _mm_loadu_si128((const __m128i *) (codes1 + (size_t) byte_idx * code_stride1)));
        }

#    define GGML_FAIRY2I_TILE64_LUT_ACCUMULATE(packed, suffix)                                                           \
        do {                                                                                                             \
            const __m256i idx_lo  = _mm256_and_si256(packed, mask_idx);                                                  \
            const __m256i idx_hi  = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_idx);                            \
            const __m256i out_0_r = _mm256_shuffle_epi8(lut_0_r, idx_lo);                                                \
            const __m256i out_1_r = _mm256_shuffle_epi8(lut_1_r, idx_hi);                                                \
            sum##suffix##_r_lo    = _mm256_add_epi16(sum##suffix##_r_lo,                                                 \
                                                     _mm256_maddubs_epi16(one, _mm256_unpacklo_epi8(out_0_r, out_1_r))); \
            sum##suffix##_r_hi    = _mm256_add_epi16(sum##suffix##_r_hi,                                                 \
                                                     _mm256_maddubs_epi16(one, _mm256_unpackhi_epi8(out_0_r, out_1_r))); \
            const __m256i out_0_i = _mm256_shuffle_epi8(lut_0_i, idx_lo);                                                \
            const __m256i out_1_i = _mm256_shuffle_epi8(lut_1_i, idx_hi);                                                \
            sum##suffix##_i_lo    = _mm256_add_epi16(sum##suffix##_i_lo,                                                 \
                                                     _mm256_maddubs_epi16(one, _mm256_unpacklo_epi8(out_0_i, out_1_i))); \
            sum##suffix##_i_hi    = _mm256_add_epi16(sum##suffix##_i_hi,                                                 \
                                                     _mm256_maddubs_epi16(one, _mm256_unpackhi_epi8(out_0_i, out_1_i))); \
        } while (false)

        GGML_FAIRY2I_TILE64_LUT_ACCUMULATE(packed0, 0);
        GGML_FAIRY2I_TILE64_LUT_ACCUMULATE(packed1, 1);
#    undef GGML_FAIRY2I_TILE64_LUT_ACCUMULATE
    }
}

static inline void ggml_fairy2i_tile64_lut_accumulate_pair_channels_codes(const uint8_t * codes0,
                                                                          size_t          code_stride0,
                                                                          const uint8_t * codes1,
                                                                          size_t          code_stride1,
                                                                          const int8_t *  lut_blk,
                                                                          __m256i &       sum0_r_lo,
                                                                          __m256i &       sum0_r_hi,
                                                                          __m256i &       sum1_r_lo,
                                                                          __m256i &       sum1_r_hi,
                                                                          __m256i &       sum0_i_lo,
                                                                          __m256i &       sum0_i_hi,
                                                                          __m256i &       sum1_i_lo,
                                                                          __m256i &       sum1_i_hi) {
    ggml_fairy2i_tile64_lut_accumulate_pair_channels_codes_impl<false>(
        codes0, code_stride0, codes1, code_stride1, lut_blk, sum0_r_lo, sum0_r_hi, sum1_r_lo, sum1_r_hi, sum0_i_lo,
        sum0_i_hi, sum1_i_lo, sum1_i_hi);
}

static inline void ggml_fairy2i_tile64_lut_accumulate_pair_channels(const fairy2i_tile64_lut_wtile_16 * wt0,
                                                                    const fairy2i_tile64_lut_wtile_16 * wt1,
                                                                    const int8_t *                      lut_blk,
                                                                    __m256i &                           sum0_r_lo,
                                                                    __m256i &                           sum0_r_hi,
                                                                    __m256i &                           sum1_r_lo,
                                                                    __m256i &                           sum1_r_hi,
                                                                    __m256i &                           sum0_i_lo,
                                                                    __m256i &                           sum0_i_hi,
                                                                    __m256i &                           sum1_i_lo,
                                                                    __m256i &                           sum1_i_hi) {
    ggml_fairy2i_tile64_lut_accumulate_pair_channels_codes(&wt0->qs[0][0], sizeof(wt0->qs[0]), &wt1->qs[0][0],
                                                           sizeof(wt1->qs[0]), lut_blk, sum0_r_lo, sum0_r_hi, sum1_r_lo,
                                                           sum1_r_hi, sum0_i_lo, sum0_i_hi, sum1_i_lo, sum1_i_hi);
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

static inline void ggml_fairy2i_tile64_lut_apply_pair_component_mixed(const __m256i &   sum0_lo,
                                                                const __m256i &   sum0_hi,
                                                                const __m256i &   sum1_lo,
                                                                const __m256i &   sum1_hi,
                                                                const ggml_half * scale_a0,
                                                                const ggml_half * scale_b0,
                                                                const ggml_half * scale_a1,
                                                                const ggml_half * scale_b1,
                                                                const __m256 &    v_a0,
                                                                const __m256 &    v_b0,
                                                                const __m256 &    v_a1,
                                                                const __m256 &    v_b1,
                                                                __m256 &          acc_lo,
                                                                __m256 &          acc_hi) {
    for (int half = 0; half < 2; ++half) {
        const __m128i s0_a = half == 0 ? _mm256_castsi256_si128(sum0_lo) : _mm256_castsi256_si128(sum0_hi);
        const __m128i s0_b = half == 0 ? _mm256_extracti128_si256(sum0_lo, 1) : _mm256_extracti128_si256(sum0_hi, 1);
        const __m128i s1_a = half == 0 ? _mm256_castsi256_si128(sum1_lo) : _mm256_castsi256_si128(sum1_hi);
        const __m128i s1_b = half == 0 ? _mm256_extracti128_si256(sum1_lo, 1) : _mm256_extracti128_si256(sum1_hi, 1);
        __m256 &      acc  = half == 0 ? acc_lo : acc_hi;
        const int     offset = half * 8;

        acc = ggml_fairy2i_tile64_lut_madd(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(s0_a)),
                                     _mm256_mul_ps(v_a0, ggml_fairy2i_tile64_lut_load_scale8(scale_a0 + offset)), acc);
        acc = ggml_fairy2i_tile64_lut_madd(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(s0_b)),
                                     _mm256_mul_ps(v_b0, ggml_fairy2i_tile64_lut_load_scale8(scale_b0 + offset)), acc);
        acc = ggml_fairy2i_tile64_lut_madd(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(s1_a)),
                                     _mm256_mul_ps(v_a1, ggml_fairy2i_tile64_lut_load_scale8(scale_a1 + offset)), acc);
        acc = ggml_fairy2i_tile64_lut_madd(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(s1_b)),
                                     _mm256_mul_ps(v_b1, ggml_fairy2i_tile64_lut_load_scale8(scale_b1 + offset)), acc);
    }
}

static inline void ggml_fairy2i_bundle_lut_apply_pair_component_mixed(const __m256i & sum0_lo,
                                                                      const __m256i & sum0_hi,
                                                                      const __m256i & sum1_lo,
                                                                      const __m256i & sum1_hi,
                                                                      float           factor_a0,
                                                                      float           factor_b0,
                                                                      float           factor_a1,
                                                                      float           factor_b1,
                                                                      __m256 &        acc_lo,
                                                                      __m256 &        acc_hi) {
    const __m256 scaled_a0 = _mm256_set1_ps(factor_a0);
    const __m256 scaled_b0 = _mm256_set1_ps(factor_b0);
    const __m256 scaled_a1 = _mm256_set1_ps(factor_a1);
    const __m256 scaled_b1 = _mm256_set1_ps(factor_b1);

    for (int half = 0; half < 2; ++half) {
        const __m128i s0_a = half == 0 ? _mm256_castsi256_si128(sum0_lo) : _mm256_castsi256_si128(sum0_hi);
        const __m128i s0_b = half == 0 ? _mm256_extracti128_si256(sum0_lo, 1) : _mm256_extracti128_si256(sum0_hi, 1);
        const __m128i s1_a = half == 0 ? _mm256_castsi256_si128(sum1_lo) : _mm256_castsi256_si128(sum1_hi);
        const __m128i s1_b = half == 0 ? _mm256_extracti128_si256(sum1_lo, 1) : _mm256_extracti128_si256(sum1_hi, 1);
        __m256 &      acc  = half == 0 ? acc_lo : acc_hi;

        acc = ggml_fairy2i_tile64_lut_madd(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(s0_a)), scaled_a0, acc);
        acc = ggml_fairy2i_tile64_lut_madd(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(s0_b)), scaled_b0, acc);
        acc = ggml_fairy2i_tile64_lut_madd(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(s1_a)), scaled_a1, acc);
        acc = ggml_fairy2i_tile64_lut_madd(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(s1_b)), scaled_b1, acc);
    }
}

struct ggml_fairy2i_bundle_lut_pair_factors {
    float r_a0;
    float r_b0;
    float r_a1;
    float r_b1;
    float i_a0;
    float i_b0;
    float i_a1;
    float i_b1;
};

static inline struct ggml_fairy2i_bundle_lut_pair_factors
ggml_fairy2i_bundle_lut_make_pair_factors(float wr0, float wi0, float wr1, float wi1, float lr, float li0, float li1) {
    return {
        lr * wr0, li0 * wi0, lr * wr1, li1 * wi1, lr * wi0, li0 * wr0, lr * wi1, li1 * wr1,
    };
}

static inline void ggml_fairy2i_bundle_lut_accumulate_pair(const uint8_t * codes0,
                                                           const uint8_t * codes1,
                                                           size_t          code_stride,
                                                           const struct ggml_fairy2i_bundle_lut_pair_factors & factors,
                                                           const int8_t *                                      lut_blk,
                                                           __m256 &                                            acc_r_lo,
                                                           __m256 &                                            acc_r_hi,
                                                           __m256 &                                            acc_i_lo,
                                                           __m256 & acc_i_hi) {
    __m256i sum0_r_lo;
    __m256i sum0_r_hi;
    __m256i sum1_r_lo;
    __m256i sum1_r_hi;
    __m256i sum0_i_lo;
    __m256i sum0_i_hi;
    __m256i sum1_i_lo;
    __m256i sum1_i_hi;

    ggml_fairy2i_tile64_lut_accumulate_pair_channels_codes_impl<true>(codes0, code_stride, codes1, code_stride, lut_blk,
                                                                      sum0_r_lo, sum0_r_hi, sum1_r_lo, sum1_r_hi,
                                                                      sum0_i_lo, sum0_i_hi, sum1_i_lo, sum1_i_hi);

    ggml_fairy2i_bundle_lut_apply_pair_component_mixed(sum0_r_lo, sum0_r_hi, sum1_r_lo, sum1_r_hi, factors.r_a0,
                                                       factors.r_b0, factors.r_a1, factors.r_b1, acc_r_lo, acc_r_hi);
    ggml_fairy2i_bundle_lut_apply_pair_component_mixed(sum0_i_lo, sum0_i_hi, sum1_i_lo, sum1_i_hi, factors.i_a0,
                                                       factors.i_b0, factors.i_a1, factors.i_b1, acc_i_lo, acc_i_hi);
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

static inline void ggml_fairy2i_tile64_lut_accumulate_w1_pair(const fairy2i_tile64_lut_wtile_16 * u0,
                                                        const fairy2i_tile64_lut_wtile_16 * w0,
                                                        const int8_t *                lut_blk,
                                                        const __m256 &                v_lr,
                                                        const __m256 &                v_li_u,
                                                        const __m256 &                v_li_w,
                                                        __m256 &                      acc_r_lo,
                                                        __m256 &                      acc_r_hi,
                                                        __m256 &                      acc_i_lo,
                                                        __m256 &                      acc_i_hi) {
    __m256i sum_u_r_lo;
    __m256i sum_u_r_hi;
    __m256i sum_w_r_lo;
    __m256i sum_w_r_hi;
    __m256i sum_u_i_lo;
    __m256i sum_u_i_hi;
    __m256i sum_w_i_lo;
    __m256i sum_w_i_hi;

    ggml_fairy2i_tile64_lut_accumulate_pair_channels(u0, w0, lut_blk, sum_u_r_lo, sum_u_r_hi, sum_w_r_lo, sum_w_r_hi,
                                               sum_u_i_lo, sum_u_i_hi, sum_w_i_lo, sum_w_i_hi);
    ggml_fairy2i_tile64_lut_apply_pair_component_mixed(sum_u_r_lo, sum_u_r_hi, sum_w_r_lo, sum_w_r_hi, u0->d_real,
                                                 u0->d_imag, w0->d_real, w0->d_imag, v_lr, v_li_u, v_lr, v_li_w,
                                                 acc_r_lo, acc_r_hi);
    ggml_fairy2i_tile64_lut_apply_pair_component_mixed(sum_u_i_lo, sum_u_i_hi, sum_w_i_lo, sum_w_i_hi, u0->d_imag,
                                                 u0->d_real, w0->d_imag, w0->d_real, v_lr, v_li_u, v_lr, v_li_w,
                                                 acc_i_lo, acc_i_hi);
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

static GGML_FAIRY2I_NOINLINE void ggml_fairy2i_tile64_lut_qgemm_w1_pair_avx2(int          m,
                                                                       int          k,
                                                                       int          n,
                                                                       const void * packed_u0,
                                                                       const void * packed_w0,
                                                                       const void * lut,
                                                                       const void * lut_scales,
                                                                       float *      dst,
                                                                       size_t       dst_col_stride,
                                                                       size_t       dst_row_stride,
                                                                       bool         pack_bf16) {
    const int64_t blocks           = k / QK_FAIRY2I_TILE64;
    const int64_t groups_per_block = QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK;
    const int64_t groups           = blocks * groups_per_block;
    const int     tiles            = (m + 15) / 16;
    const auto *  utiles           = (const fairy2i_tile64_lut_wtile_16 *) packed_u0;
    const auto *  wtiles           = (const fairy2i_tile64_lut_wtile_16 *) packed_w0;
    const __m256  imag_sign_u      = _mm256_set1_ps(-0.0f);

    for (int col = 0; col < n; ++col) {
        const int8_t * lut_col = (const int8_t *) lut + (size_t) col * (size_t) groups * k_fairy2i_lut_group_bytes;
        const float *  scales  = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2u;
        uint8_t *      dst_col = (uint8_t *) dst + (size_t) col * dst_col_stride;

        for (int tile = 0; tile < tiles; ++tile) {
            __m256 acc_r_lo = _mm256_setzero_ps();
            __m256 acc_r_hi = _mm256_setzero_ps();
            __m256 acc_i_lo = _mm256_setzero_ps();
            __m256 acc_i_hi = _mm256_setzero_ps();

            for (int64_t blk = 0; blk < blocks; ++blk) {
                const fairy2i_tile64_lut_wtile_16 * u0 =
                    utiles + (size_t) tile * (size_t) blocks + (size_t) blk;
                const fairy2i_tile64_lut_wtile_16 * w0 =
                    wtiles + (size_t) tile * (size_t) blocks + (size_t) blk;
                const int8_t * lut_blk =
                    lut_col + (size_t) blk * (size_t) groups_per_block * k_fairy2i_lut_group_bytes;
                const __m256 v_lr   = _mm256_set1_ps(scales[blk * 2 + 0]);
                const __m256 v_li_w = _mm256_set1_ps(scales[blk * 2 + 1]);
                const __m256 v_li_u = _mm256_xor_ps(v_li_w, imag_sign_u);
                ggml_fairy2i_tile64_lut_accumulate_w1_pair(u0, w0, lut_blk, v_lr, v_li_u, v_li_w, acc_r_lo,
                                                     acc_r_hi, acc_i_lo, acc_i_hi);
            }
            ggml_fairy2i_tile64_lut_store_pair(tile, m, dst_col, dst_row_stride, pack_bf16, /*add*/ false, acc_r_lo,
                                         acc_r_hi, acc_i_lo, acc_i_hi);
        }
    }
}

static inline void ggml_fairy2i_bundle_lut_qgemm_pair_tile_avx2(int                                     m,
                                                                const struct ggml_fairy2i_bundle_desc * bundle,
                                                                int64_t        global_m16_offset,
                                                                int            branch0,
                                                                int            branch1,
                                                                bool           negate_imag0,
                                                                bool           negate_imag1,
                                                                int64_t        blocks,
                                                                int64_t        groups_per_block,
                                                                int            tile0,
                                                                int            tile_count,
                                                                const int8_t * lut_col,
                                                                const float *  scales,
                                                                uint8_t *      dst_col,
                                                                size_t         dst_row_stride,
                                                                bool           pack_bf16,
                                                                bool           add) {
    const size_t code_stride = (size_t) bundle->branches * 16u;
    __m256       acc_r_lo[4];
    __m256       acc_r_hi[4];
    __m256       acc_i_lo[4];
    __m256       acc_i_hi[4];
    for (int ti = 0; ti < tile_count; ++ti) {
        acc_r_lo[ti] = _mm256_setzero_ps();
        acc_r_hi[ti] = _mm256_setzero_ps();
        acc_i_lo[ti] = _mm256_setzero_ps();
        acc_i_hi[ti] = _mm256_setzero_ps();
    }

    for (int64_t blk = 0; blk < blocks; ++blk) {
        const int64_t first_global_m16 = global_m16_offset + tile0;
        const int64_t last_global_m16  = first_global_m16 + tile_count - 1;
        const bool    shared_m64       = first_global_m16 / 4 == last_global_m16 / 4;
        float         shared_wr0       = 0.0f;
        float         shared_wi0       = 0.0f;
        float         shared_wr1       = 0.0f;
        float         shared_wi1       = 0.0f;
        if (shared_m64) {
            const ggml_half * scales0 = ggml_fairy2i_bundle_scales_at(bundle, first_global_m16, blk, branch0);
            const ggml_half * scales1 = ggml_fairy2i_bundle_scales_at(bundle, first_global_m16, blk, branch1);
            shared_wr0                = ggml_fairy2i_bundle_lut_scale_to_f32(scales0[0]);
            shared_wi0                = ggml_fairy2i_bundle_lut_scale_to_f32(scales0[1]);
            shared_wr1                = ggml_fairy2i_bundle_lut_scale_to_f32(scales1[0]);
            shared_wi1                = ggml_fairy2i_bundle_lut_scale_to_f32(scales1[1]);
        }
        const int8_t * lut_blk = lut_col + (size_t) blk * (size_t) groups_per_block * k_fairy2i_lut_group_bytes;
        const float    lr      = scales[blk * 2 + 0];
        const float    li      = scales[blk * 2 + 1];
        const float    li0     = negate_imag0 ? -li : li;
        const float    li1     = negate_imag1 ? -li : li;
        struct ggml_fairy2i_bundle_lut_pair_factors shared_factors = {};
        if (shared_m64) {
            shared_factors =
                ggml_fairy2i_bundle_lut_make_pair_factors(shared_wr0, shared_wi0, shared_wr1, shared_wi1, lr, li0, li1);
        }
        for (int ti = 0; ti < tile_count; ++ti) {
            const int64_t   global_m16    = global_m16_offset + tile0 + ti;
            const int64_t   physical_tile = (global_m16 / 4) * bundle->k_blocks + blk;
            const int64_t   slot_base     = (global_m16 % 4) * 16;
            const uint8_t * code_base     = bundle->codes + ((physical_tile * 64 + slot_base) * bundle->branches) * 16;
            const uint8_t * codes0        = code_base + (size_t) branch0 * 16u;
            const uint8_t * codes1        = code_base + (size_t) branch1 * 16u;
            struct ggml_fairy2i_bundle_lut_pair_factors factors = shared_factors;
            if (!shared_m64) {
                const ggml_half * scales0 = bundle->scales + (physical_tile * bundle->branches + branch0) * 2;
                const ggml_half * scales1 = bundle->scales + (physical_tile * bundle->branches + branch1) * 2;
                const float       wr0     = ggml_fairy2i_bundle_lut_scale_to_f32(scales0[0]);
                const float       wi0     = ggml_fairy2i_bundle_lut_scale_to_f32(scales0[1]);
                const float       wr1     = ggml_fairy2i_bundle_lut_scale_to_f32(scales1[0]);
                const float       wi1     = ggml_fairy2i_bundle_lut_scale_to_f32(scales1[1]);
                factors                   = ggml_fairy2i_bundle_lut_make_pair_factors(wr0, wi0, wr1, wi1, lr, li0, li1);
            }
            ggml_fairy2i_bundle_lut_accumulate_pair(codes0, codes1, code_stride, factors, lut_blk, acc_r_lo[ti],
                                                    acc_r_hi[ti], acc_i_lo[ti], acc_i_hi[ti]);
        }
    }
    for (int ti = 0; ti < tile_count; ++ti) {
        ggml_fairy2i_tile64_lut_store_pair(tile0 + ti, m, dst_col, dst_row_stride, pack_bf16, add, acc_r_lo[ti],
                                           acc_r_hi[ti], acc_i_lo[ti], acc_i_hi[ti]);
    }
}

static GGML_FAIRY2I_NOINLINE void ggml_fairy2i_bundle_lut_qgemm_pair_avx2(
    int                                     m,
    int                                     k,
    int                                     n,
    const struct ggml_fairy2i_bundle_desc * bundle,
    int64_t                                 global_m16_offset,
    int                                     branch0,
    int                                     branch1,
    bool                                    negate_imag0,
    bool                                    negate_imag1,
    const void *                            lut,
    const void *                            lut_scales,
    float *                                 dst,
    size_t                                  dst_col_stride,
    size_t                                  dst_row_stride,
    bool                                    pack_bf16,
    bool                                    add) {
    const int64_t blocks           = k / QK_FAIRY2I_TILE64;
    const int64_t groups_per_block = QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK;
    const int64_t groups           = blocks * groups_per_block;
    const int     tiles            = (m + 15) / 16;
    const int     tiles_per_pass   = 4;
    const size_t  code_stride      = (size_t) bundle->branches * 16u;

    for (int col = 0; col < n; ++col) {
        const int8_t * lut_col = (const int8_t *) lut + (size_t) col * (size_t) groups * k_fairy2i_lut_group_bytes;
        const float *  scales  = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2u;
        uint8_t *      dst_col = (uint8_t *) dst + (size_t) col * dst_col_stride;

        for (int tile0 = 0; tile0 < tiles; tile0 += tiles_per_pass) {
            const int tile_count = tiles - tile0 < tiles_per_pass ? tiles - tile0 : tiles_per_pass;
            __m256    acc_r_lo[4];
            __m256    acc_r_hi[4];
            __m256    acc_i_lo[4];
            __m256    acc_i_hi[4];
            for (int ti = 0; ti < tile_count; ++ti) {
                acc_r_lo[ti] = _mm256_setzero_ps();
                acc_r_hi[ti] = _mm256_setzero_ps();
                acc_i_lo[ti] = _mm256_setzero_ps();
                acc_i_hi[ti] = _mm256_setzero_ps();
            }

            for (int64_t blk = 0; blk < blocks; ++blk) {
                const int64_t first_global_m16 = global_m16_offset + tile0;
                const int64_t last_global_m16  = first_global_m16 + tile_count - 1;
                const bool    shared_m64       = first_global_m16 / 4 == last_global_m16 / 4;
                float         shared_wr0       = 0.0f;
                float         shared_wi0       = 0.0f;
                float         shared_wr1       = 0.0f;
                float         shared_wi1       = 0.0f;
                if (shared_m64) {
                    const ggml_half * scales0 = ggml_fairy2i_bundle_scales_at(bundle, first_global_m16, blk, branch0);
                    const ggml_half * scales1 = ggml_fairy2i_bundle_scales_at(bundle, first_global_m16, blk, branch1);
                    shared_wr0                = ggml_fairy2i_bundle_lut_scale_to_f32(scales0[0]);
                    shared_wi0                = ggml_fairy2i_bundle_lut_scale_to_f32(scales0[1]);
                    shared_wr1                = ggml_fairy2i_bundle_lut_scale_to_f32(scales1[0]);
                    shared_wi1                = ggml_fairy2i_bundle_lut_scale_to_f32(scales1[1]);
                }
                const int8_t * lut_blk = lut_col + (size_t) blk * (size_t) groups_per_block * k_fairy2i_lut_group_bytes;
                const float    lr      = scales[blk * 2 + 0];
                const float    li      = scales[blk * 2 + 1];
                const float    li0     = negate_imag0 ? -li : li;
                const float    li1     = negate_imag1 ? -li : li;
                struct ggml_fairy2i_bundle_lut_pair_factors shared_factors = {};
                if (shared_m64) {
                    shared_factors = ggml_fairy2i_bundle_lut_make_pair_factors(shared_wr0, shared_wi0, shared_wr1,
                                                                               shared_wi1, lr, li0, li1);
                }
                for (int ti = 0; ti < tile_count; ++ti) {
                    const int64_t   global_m16    = global_m16_offset + tile0 + ti;
                    const int64_t   physical_tile = (global_m16 / 4) * bundle->k_blocks + blk;
                    const int64_t   slot_base     = (global_m16 % 4) * 16;
                    const uint8_t * code_base =
                        bundle->codes + ((physical_tile * 64 + slot_base) * bundle->branches) * 16;
                    const uint8_t *                             codes0  = code_base + (size_t) branch0 * 16u;
                    const uint8_t *                             codes1  = code_base + (size_t) branch1 * 16u;
                    struct ggml_fairy2i_bundle_lut_pair_factors factors = shared_factors;
                    if (!shared_m64) {
                        const ggml_half * scales0 = bundle->scales + (physical_tile * bundle->branches + branch0) * 2;
                        const ggml_half * scales1 = bundle->scales + (physical_tile * bundle->branches + branch1) * 2;
                        const float       wr0     = ggml_fairy2i_bundle_lut_scale_to_f32(scales0[0]);
                        const float       wi0     = ggml_fairy2i_bundle_lut_scale_to_f32(scales0[1]);
                        const float       wr1     = ggml_fairy2i_bundle_lut_scale_to_f32(scales1[0]);
                        const float       wi1     = ggml_fairy2i_bundle_lut_scale_to_f32(scales1[1]);
                        factors = ggml_fairy2i_bundle_lut_make_pair_factors(wr0, wi0, wr1, wi1, lr, li0, li1);
                    }
                    ggml_fairy2i_bundle_lut_accumulate_pair(codes0, codes1, code_stride, factors, lut_blk, acc_r_lo[ti],
                                                            acc_r_hi[ti], acc_i_lo[ti], acc_i_hi[ti]);
                }
            }
            for (int ti = 0; ti < tile_count; ++ti) {
                ggml_fairy2i_tile64_lut_store_pair(tile0 + ti, m, dst_col, dst_row_stride, pack_bf16, add, acc_r_lo[ti],
                                                   acc_r_hi[ti], acc_i_lo[ti], acc_i_hi[ti]);
            }
        }
    }
}

static GGML_FAIRY2I_NOINLINE void ggml_fairy2i_bundle_lut_qgemm_four_avx2(
    int                                     m,
    int                                     k,
    int                                     n,
    const struct ggml_fairy2i_bundle_desc * bundle,
    int64_t                                 global_m16_offset,
    const void *                            lut,
    const void *                            lut_scales,
    float *                                 dst,
    size_t                                  dst_col_stride,
    size_t                                  dst_row_stride,
    bool                                    pack_bf16) {
    const int64_t blocks           = k / QK_FAIRY2I_TILE64;
    const int64_t groups_per_block = QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK;
    const int64_t groups           = blocks * groups_per_block;
    const int     tiles            = (m + 15) / 16;

    for (int col = 0; col < n; ++col) {
        const int8_t * lut_col = (const int8_t *) lut + (size_t) col * (size_t) groups * k_fairy2i_lut_group_bytes;
        const float *  scales  = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2u;
        uint8_t *      dst_col = (uint8_t *) dst + (size_t) col * dst_col_stride;
        for (int tile0 = 0; tile0 < tiles; tile0 += 4) {
            const int tile_count = tiles - tile0 < 4 ? tiles - tile0 : 4;
            ggml_fairy2i_bundle_lut_qgemm_pair_tile_avx2(m, bundle, global_m16_offset, 0, 1, true, true, blocks,
                                                         groups_per_block, tile0, tile_count, lut_col, scales, dst_col,
                                                         dst_row_stride, pack_bf16, false);
            ggml_fairy2i_bundle_lut_qgemm_pair_tile_avx2(m, bundle, global_m16_offset, 2, 3, false, false, blocks,
                                                         groups_per_block, tile0, tile_count, lut_col, scales, dst_col,
                                                         dst_row_stride, pack_bf16, true);
        }
    }
}

#    undef GGML_FAIRY2I_NOINLINE
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)
#    if defined(_MSC_VER)
#        define GGML_FAIRY2I_NOINLINE __declspec(noinline)
#    else
#        define GGML_FAIRY2I_NOINLINE __attribute__((__noinline__))
#    endif

static inline float32x4_t ggml_fairy2i_tile64_lut_s16x4_to_f32(int16x4_t v) {
    return vcvtq_f32_s32(vmovl_s16(v));
}

static inline float32x4_t ggml_fairy2i_tile64_lut_load_scale4_arm(const ggml_half * src) {
    const uint16x4_t v = vld1_u16((const uint16_t *) src);
    return vcvt_f32_f16(vreinterpret_f16_u16(v));
}

static inline void ggml_fairy2i_tile64_lut_accumulate_codes_arm(const uint8_t *     codes,
                                                                const int8x16x4_t & ilut_0,
                                                                const int8x16x4_t & ilut_1,
                                                                const uint8x16_t    mask_4bit,
                                                                int16x8_t &         sum_ac_0,
                                                                int16x8_t &         sum_ac_1,
                                                                int16x8_t &         sum_bc_0,
                                                                int16x8_t &         sum_bc_1,
                                                                int16x8_t &         sum_ad_0,
                                                                int16x8_t &         sum_ad_1,
                                                                int16x8_t &         sum_bd_0,
                                                                int16x8_t &         sum_bd_1) {
    const uint8x16_t packed = vld1q_u8(codes);
    const uint8x16_t idx_lo = vandq_u8(packed, mask_4bit);
    const uint8x16_t idx_hi = vandq_u8(vshrq_n_u8(packed, 4), mask_4bit);

    const int8x16_t v_ac_0 = vqtbl1q_s8(ilut_0.val[0], idx_lo);
    const int8x16_t v_bd_0 = vqtbl1q_s8(ilut_0.val[1], idx_lo);
    const int8x16_t v_bc_0 = vqtbl1q_s8(ilut_0.val[2], idx_lo);
    const int8x16_t v_ad_0 = vqtbl1q_s8(ilut_0.val[3], idx_lo);

    const int8x16_t v_ac_1 = vqtbl1q_s8(ilut_1.val[0], idx_hi);
    const int8x16_t v_bd_1 = vqtbl1q_s8(ilut_1.val[1], idx_hi);
    const int8x16_t v_bc_1 = vqtbl1q_s8(ilut_1.val[2], idx_hi);
    const int8x16_t v_ad_1 = vqtbl1q_s8(ilut_1.val[3], idx_hi);

    sum_ac_0 = vaddw_s8(sum_ac_0, vget_low_s8(v_ac_0));
    sum_ac_1 = vaddw_s8(sum_ac_1, vget_high_s8(v_ac_0));
    sum_ac_0 = vaddw_s8(sum_ac_0, vget_low_s8(v_ac_1));
    sum_ac_1 = vaddw_s8(sum_ac_1, vget_high_s8(v_ac_1));

    sum_bc_0 = vaddw_s8(sum_bc_0, vget_low_s8(v_bc_0));
    sum_bc_1 = vaddw_s8(sum_bc_1, vget_high_s8(v_bc_0));
    sum_bc_0 = vaddw_s8(sum_bc_0, vget_low_s8(v_bc_1));
    sum_bc_1 = vaddw_s8(sum_bc_1, vget_high_s8(v_bc_1));

    sum_ad_0 = vaddw_s8(sum_ad_0, vget_low_s8(v_ad_0));
    sum_ad_1 = vaddw_s8(sum_ad_1, vget_high_s8(v_ad_0));
    sum_ad_0 = vaddw_s8(sum_ad_0, vget_low_s8(v_ad_1));
    sum_ad_1 = vaddw_s8(sum_ad_1, vget_high_s8(v_ad_1));

    sum_bd_0 = vaddw_s8(sum_bd_0, vget_low_s8(v_bd_0));
    sum_bd_1 = vaddw_s8(sum_bd_1, vget_high_s8(v_bd_0));
    sum_bd_0 = vaddw_s8(sum_bd_0, vget_low_s8(v_bd_1));
    sum_bd_1 = vaddw_s8(sum_bd_1, vget_high_s8(v_bd_1));
}

static inline void ggml_fairy2i_tile64_lut_accumulate_wtile_arm(const fairy2i_tile64_lut_wtile_16 * wt,
                                                                int                                 byte_idx,
                                                                const int8x16x4_t &                 ilut_0,
                                                                const int8x16x4_t &                 ilut_1,
                                                                const uint8x16_t                    mask_4bit,
                                                                int16x8_t &                         sum_ac_0,
                                                                int16x8_t &                         sum_ac_1,
                                                                int16x8_t &                         sum_bc_0,
                                                                int16x8_t &                         sum_bc_1,
                                                                int16x8_t &                         sum_ad_0,
                                                                int16x8_t &                         sum_ad_1,
                                                                int16x8_t &                         sum_bd_0,
                                                                int16x8_t &                         sum_bd_1) {
    ggml_fairy2i_tile64_lut_accumulate_codes_arm(wt->qs[byte_idx], ilut_0, ilut_1, mask_4bit, sum_ac_0, sum_ac_1,
                                                 sum_bc_0, sum_bc_1, sum_ad_0, sum_ad_1, sum_bd_0, sum_bd_1);
}

static inline void ggml_fairy2i_tile64_lut_apply_sums_arm(const fairy2i_tile64_lut_wtile_16 * wt,
                                                          const float32x4_t                   v_lr,
                                                          const float32x4_t                   v_li,
                                                          const int16x8_t                     sum_ac_0,
                                                          const int16x8_t                     sum_ac_1,
                                                          const int16x8_t                     sum_bc_0,
                                                          const int16x8_t                     sum_bc_1,
                                                          const int16x8_t                     sum_ad_0,
                                                          const int16x8_t                     sum_ad_1,
                                                          const int16x8_t                     sum_bd_0,
                                                          const int16x8_t                     sum_bd_1,
                                                          float32x4_t &                       acc_r0,
                                                          float32x4_t &                       acc_r1,
                                                          float32x4_t &                       acc_r2,
                                                          float32x4_t &                       acc_r3,
                                                          float32x4_t &                       acc_i0,
                                                          float32x4_t &                       acc_i1,
                                                          float32x4_t &                       acc_i2,
                                                          float32x4_t &                       acc_i3) {
    {
        const float32x4_t wr    = ggml_fairy2i_tile64_lut_load_scale4_arm(wt->d_real + 0);
        const float32x4_t wi    = ggml_fairy2i_tile64_lut_load_scale4_arm(wt->d_imag + 0);
        const float32x4_t lr_wr = vmulq_f32(v_lr, wr);
        const float32x4_t li_wi = vmulq_f32(v_li, wi);
        const float32x4_t lr_wi = vmulq_f32(v_lr, wi);
        const float32x4_t li_wr = vmulq_f32(v_li, wr);

        acc_r0 = vmlaq_f32(acc_r0, ggml_fairy2i_tile64_lut_s16x4_to_f32(vget_low_s16(sum_ac_0)), lr_wr);
        acc_r0 = vmlaq_f32(acc_r0, ggml_fairy2i_tile64_lut_s16x4_to_f32(vget_low_s16(sum_bd_0)), li_wi);
        acc_i0 = vmlaq_f32(acc_i0, ggml_fairy2i_tile64_lut_s16x4_to_f32(vget_low_s16(sum_bc_0)), lr_wi);
        acc_i0 = vmlaq_f32(acc_i0, ggml_fairy2i_tile64_lut_s16x4_to_f32(vget_low_s16(sum_ad_0)), li_wr);
    }
    {
        const float32x4_t wr    = ggml_fairy2i_tile64_lut_load_scale4_arm(wt->d_real + 4);
        const float32x4_t wi    = ggml_fairy2i_tile64_lut_load_scale4_arm(wt->d_imag + 4);
        const float32x4_t lr_wr = vmulq_f32(v_lr, wr);
        const float32x4_t li_wi = vmulq_f32(v_li, wi);
        const float32x4_t lr_wi = vmulq_f32(v_lr, wi);
        const float32x4_t li_wr = vmulq_f32(v_li, wr);

        acc_r1 = vmlaq_f32(acc_r1, ggml_fairy2i_tile64_lut_s16x4_to_f32(vget_high_s16(sum_ac_0)), lr_wr);
        acc_r1 = vmlaq_f32(acc_r1, ggml_fairy2i_tile64_lut_s16x4_to_f32(vget_high_s16(sum_bd_0)), li_wi);
        acc_i1 = vmlaq_f32(acc_i1, ggml_fairy2i_tile64_lut_s16x4_to_f32(vget_high_s16(sum_bc_0)), lr_wi);
        acc_i1 = vmlaq_f32(acc_i1, ggml_fairy2i_tile64_lut_s16x4_to_f32(vget_high_s16(sum_ad_0)), li_wr);
    }
    {
        const float32x4_t wr    = ggml_fairy2i_tile64_lut_load_scale4_arm(wt->d_real + 8);
        const float32x4_t wi    = ggml_fairy2i_tile64_lut_load_scale4_arm(wt->d_imag + 8);
        const float32x4_t lr_wr = vmulq_f32(v_lr, wr);
        const float32x4_t li_wi = vmulq_f32(v_li, wi);
        const float32x4_t lr_wi = vmulq_f32(v_lr, wi);
        const float32x4_t li_wr = vmulq_f32(v_li, wr);

        acc_r2 = vmlaq_f32(acc_r2, ggml_fairy2i_tile64_lut_s16x4_to_f32(vget_low_s16(sum_ac_1)), lr_wr);
        acc_r2 = vmlaq_f32(acc_r2, ggml_fairy2i_tile64_lut_s16x4_to_f32(vget_low_s16(sum_bd_1)), li_wi);
        acc_i2 = vmlaq_f32(acc_i2, ggml_fairy2i_tile64_lut_s16x4_to_f32(vget_low_s16(sum_bc_1)), lr_wi);
        acc_i2 = vmlaq_f32(acc_i2, ggml_fairy2i_tile64_lut_s16x4_to_f32(vget_low_s16(sum_ad_1)), li_wr);
    }
    {
        const float32x4_t wr    = ggml_fairy2i_tile64_lut_load_scale4_arm(wt->d_real + 12);
        const float32x4_t wi    = ggml_fairy2i_tile64_lut_load_scale4_arm(wt->d_imag + 12);
        const float32x4_t lr_wr = vmulq_f32(v_lr, wr);
        const float32x4_t li_wi = vmulq_f32(v_li, wi);
        const float32x4_t lr_wi = vmulq_f32(v_lr, wi);
        const float32x4_t li_wr = vmulq_f32(v_li, wr);

        acc_r3 = vmlaq_f32(acc_r3, ggml_fairy2i_tile64_lut_s16x4_to_f32(vget_high_s16(sum_ac_1)), lr_wr);
        acc_r3 = vmlaq_f32(acc_r3, ggml_fairy2i_tile64_lut_s16x4_to_f32(vget_high_s16(sum_bd_1)), li_wi);
        acc_i3 = vmlaq_f32(acc_i3, ggml_fairy2i_tile64_lut_s16x4_to_f32(vget_high_s16(sum_bc_1)), lr_wi);
        acc_i3 = vmlaq_f32(acc_i3, ggml_fairy2i_tile64_lut_s16x4_to_f32(vget_high_s16(sum_ad_1)), li_wr);
    }
}

static inline void ggml_fairy2i_bundle_lut_apply_sums4_arm(float32x4_t   v_lr,
                                                           float32x4_t   v_li,
                                                           float32x4_t   wr,
                                                           float32x4_t   wi,
                                                           int16x4_t     sum_ac,
                                                           int16x4_t     sum_bc,
                                                           int16x4_t     sum_ad,
                                                           int16x4_t     sum_bd,
                                                           float32x4_t & acc_r,
                                                           float32x4_t & acc_i) {
    const float32x4_t lr_wr = vmulq_f32(v_lr, wr);
    const float32x4_t li_wi = vmulq_f32(v_li, wi);
    const float32x4_t lr_wi = vmulq_f32(v_lr, wi);
    const float32x4_t li_wr = vmulq_f32(v_li, wr);
    acc_r                   = vmlaq_f32(acc_r, ggml_fairy2i_tile64_lut_s16x4_to_f32(sum_ac), lr_wr);
    acc_r                   = vmlaq_f32(acc_r, ggml_fairy2i_tile64_lut_s16x4_to_f32(sum_bd), li_wi);
    acc_i                   = vmlaq_f32(acc_i, ggml_fairy2i_tile64_lut_s16x4_to_f32(sum_bc), lr_wi);
    acc_i                   = vmlaq_f32(acc_i, ggml_fairy2i_tile64_lut_s16x4_to_f32(sum_ad), li_wr);
}

static inline void ggml_fairy2i_bundle_lut_apply_sums_arm(const ggml_half scales[2],
                                                          float32x4_t     v_lr,
                                                          float32x4_t     v_li,
                                                          int16x8_t       sum_ac_0,
                                                          int16x8_t       sum_ac_1,
                                                          int16x8_t       sum_bc_0,
                                                          int16x8_t       sum_bc_1,
                                                          int16x8_t       sum_ad_0,
                                                          int16x8_t       sum_ad_1,
                                                          int16x8_t       sum_bd_0,
                                                          int16x8_t       sum_bd_1,
                                                          float32x4_t &   acc_r0,
                                                          float32x4_t &   acc_r1,
                                                          float32x4_t &   acc_r2,
                                                          float32x4_t &   acc_r3,
                                                          float32x4_t &   acc_i0,
                                                          float32x4_t &   acc_i1,
                                                          float32x4_t &   acc_i2,
                                                          float32x4_t &   acc_i3) {
    const float32x4_t wr = vdupq_n_f32(GGML_FP16_TO_FP32(scales[0]));
    const float32x4_t wi = vdupq_n_f32(GGML_FP16_TO_FP32(scales[1]));
    ggml_fairy2i_bundle_lut_apply_sums4_arm(v_lr, v_li, wr, wi, vget_low_s16(sum_ac_0), vget_low_s16(sum_bc_0),
                                            vget_low_s16(sum_ad_0), vget_low_s16(sum_bd_0), acc_r0, acc_i0);
    ggml_fairy2i_bundle_lut_apply_sums4_arm(v_lr, v_li, wr, wi, vget_high_s16(sum_ac_0), vget_high_s16(sum_bc_0),
                                            vget_high_s16(sum_ad_0), vget_high_s16(sum_bd_0), acc_r1, acc_i1);
    ggml_fairy2i_bundle_lut_apply_sums4_arm(v_lr, v_li, wr, wi, vget_low_s16(sum_ac_1), vget_low_s16(sum_bc_1),
                                            vget_low_s16(sum_ad_1), vget_low_s16(sum_bd_1), acc_r2, acc_i2);
    ggml_fairy2i_bundle_lut_apply_sums4_arm(v_lr, v_li, wr, wi, vget_high_s16(sum_ac_1), vget_high_s16(sum_bc_1),
                                            vget_high_s16(sum_ad_1), vget_high_s16(sum_bd_1), acc_r3, acc_i3);
}

static inline void ggml_fairy2i_tile64_lut_store4_f32_arm(float * out, float32x4_t acc_r, float32x4_t acc_i) {
    const float32x4x2_t value = {
        { acc_r, acc_i }
    };
    vst2q_f32(out, value);
}

static inline void ggml_fairy2i_tile64_lut_store_tile_arm(int         tile,
                                                          int         m,
                                                          uint8_t *   dst_col,
                                                          size_t      dst_row_stride,
                                                          bool        pack_bf16,
                                                          float32x4_t acc_r0,
                                                          float32x4_t acc_r1,
                                                          float32x4_t acc_r2,
                                                          float32x4_t acc_r3,
                                                          float32x4_t acc_i0,
                                                          float32x4_t acc_i1,
                                                          float32x4_t acc_i2,
                                                          float32x4_t acc_i3) {
    const int row0 = tile << 4;
    if (!pack_bf16 && dst_row_stride == 2u * sizeof(float) && row0 + 16 <= m) {
        float * out = (float *) (dst_col + (size_t) row0 * dst_row_stride);
        ggml_fairy2i_tile64_lut_store4_f32_arm(out + 0, acc_r0, acc_i0);
        ggml_fairy2i_tile64_lut_store4_f32_arm(out + 8, acc_r1, acc_i1);
        ggml_fairy2i_tile64_lut_store4_f32_arm(out + 16, acc_r2, acc_i2);
        ggml_fairy2i_tile64_lut_store4_f32_arm(out + 24, acc_r3, acc_i3);
        return;
    }

    alignas(16) float out_r[16];
    alignas(16) float out_i[16];
    vst1q_f32(out_r + 0, acc_r0);
    vst1q_f32(out_r + 4, acc_r1);
    vst1q_f32(out_r + 8, acc_r2);
    vst1q_f32(out_r + 12, acc_r3);
    vst1q_f32(out_i + 0, acc_i0);
    vst1q_f32(out_i + 4, acc_i1);
    vst1q_f32(out_i + 8, acc_i2);
    vst1q_f32(out_i + 12, acc_i3);

    const int rows_in_tile = m - row0 >= 16 ? 16 : m - row0;
    for (int lane = 0; lane < rows_in_tile; ++lane) {
        uint8_t * out = dst_col + (size_t) (row0 + lane) * dst_row_stride;
        if (pack_bf16) {
            ((ggml_bf16_t *) out)[0] = ggml_fp32_to_bf16(out_r[lane]);
            ((ggml_bf16_t *) out)[1] = ggml_fp32_to_bf16(out_i[lane]);
        } else {
            ((float *) out)[0] = out_r[lane];
            ((float *) out)[1] = out_i[lane];
        }
    }
}

static GGML_FAIRY2I_NOINLINE void ggml_fairy2i_tile64_lut_qgemm_w1_pair_neon(int          m,
                                                                             int          k,
                                                                             int          n,
                                                                             const void * packed_u0,
                                                                             const void * packed_w0,
                                                                             const void * lut,
                                                                             const void * lut_scales,
                                                                             float *      dst,
                                                                             size_t       dst_col_stride,
                                                                             size_t       dst_row_stride,
                                                                             bool         pack_bf16) {
    const int64_t    blocks           = k / QK_FAIRY2I_TILE64;
    const int64_t    groups_per_block = QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK;
    const int64_t    groups           = blocks * groups_per_block;
    const int        tiles            = (m + 15) / 16;
    const auto *     utiles           = (const fairy2i_tile64_lut_wtile_16 *) packed_u0;
    const auto *     wtiles           = (const fairy2i_tile64_lut_wtile_16 *) packed_w0;
    const uint8x16_t mask_4bit        = vdupq_n_u8(0x0f);

    for (int col = 0; col < n; ++col) {
        const int8_t * lut_col = (const int8_t *) lut + (size_t) col * (size_t) groups * k_fairy2i_lut_group_bytes;
        const float *  scales  = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2u;
        uint8_t *      dst_col = (uint8_t *) dst + (size_t) col * dst_col_stride;

        for (int tile = 0; tile < tiles; ++tile) {
            float32x4_t acc_r0 = vdupq_n_f32(0.0f);
            float32x4_t acc_r1 = vdupq_n_f32(0.0f);
            float32x4_t acc_r2 = vdupq_n_f32(0.0f);
            float32x4_t acc_r3 = vdupq_n_f32(0.0f);
            float32x4_t acc_i0 = vdupq_n_f32(0.0f);
            float32x4_t acc_i1 = vdupq_n_f32(0.0f);
            float32x4_t acc_i2 = vdupq_n_f32(0.0f);
            float32x4_t acc_i3 = vdupq_n_f32(0.0f);

            for (int64_t blk = 0; blk < blocks; ++blk) {
                const fairy2i_tile64_lut_wtile_16 * u0 = utiles + (size_t) tile * (size_t) blocks + (size_t) blk;
                const fairy2i_tile64_lut_wtile_16 * w0 = wtiles + (size_t) tile * (size_t) blocks + (size_t) blk;
                const int8_t * lut_ptr = lut_col + (size_t) blk * (size_t) groups_per_block * k_fairy2i_lut_group_bytes;

                int16x8_t sum_u_ac_0 = vdupq_n_s16(0);
                int16x8_t sum_u_ac_1 = vdupq_n_s16(0);
                int16x8_t sum_u_bc_0 = vdupq_n_s16(0);
                int16x8_t sum_u_bc_1 = vdupq_n_s16(0);
                int16x8_t sum_u_ad_0 = vdupq_n_s16(0);
                int16x8_t sum_u_ad_1 = vdupq_n_s16(0);
                int16x8_t sum_u_bd_0 = vdupq_n_s16(0);
                int16x8_t sum_u_bd_1 = vdupq_n_s16(0);

                int16x8_t sum_w_ac_0 = vdupq_n_s16(0);
                int16x8_t sum_w_ac_1 = vdupq_n_s16(0);
                int16x8_t sum_w_bc_0 = vdupq_n_s16(0);
                int16x8_t sum_w_bc_1 = vdupq_n_s16(0);
                int16x8_t sum_w_ad_0 = vdupq_n_s16(0);
                int16x8_t sum_w_ad_1 = vdupq_n_s16(0);
                int16x8_t sum_w_bd_0 = vdupq_n_s16(0);
                int16x8_t sum_w_bd_1 = vdupq_n_s16(0);

                for (int byte_idx = 0; byte_idx < groups_per_block / 2; ++byte_idx) {
                    const int8x16x4_t ilut_0 = vld1q_s8_x4(lut_ptr + 0);
                    const int8x16x4_t ilut_1 = vld1q_s8_x4(lut_ptr + 64);
                    lut_ptr += 128;

                    ggml_fairy2i_tile64_lut_accumulate_wtile_arm(u0, byte_idx, ilut_0, ilut_1, mask_4bit, sum_u_ac_0,
                                                                 sum_u_ac_1, sum_u_bc_0, sum_u_bc_1, sum_u_ad_0,
                                                                 sum_u_ad_1, sum_u_bd_0, sum_u_bd_1);
                    ggml_fairy2i_tile64_lut_accumulate_wtile_arm(w0, byte_idx, ilut_0, ilut_1, mask_4bit, sum_w_ac_0,
                                                                 sum_w_ac_1, sum_w_bc_0, sum_w_bc_1, sum_w_ad_0,
                                                                 sum_w_ad_1, sum_w_bd_0, sum_w_bd_1);
                }

                const float32x4_t v_lr   = vdupq_n_f32(scales[blk * 2 + 0]);
                const float32x4_t v_li_w = vdupq_n_f32(scales[blk * 2 + 1]);
                const float32x4_t v_li_u = vnegq_f32(v_li_w);

                ggml_fairy2i_tile64_lut_apply_sums_arm(u0, v_lr, v_li_u, sum_u_ac_0, sum_u_ac_1, sum_u_bc_0, sum_u_bc_1,
                                                       sum_u_ad_0, sum_u_ad_1, sum_u_bd_0, sum_u_bd_1, acc_r0, acc_r1,
                                                       acc_r2, acc_r3, acc_i0, acc_i1, acc_i2, acc_i3);
                ggml_fairy2i_tile64_lut_apply_sums_arm(w0, v_lr, v_li_w, sum_w_ac_0, sum_w_ac_1, sum_w_bc_0, sum_w_bc_1,
                                                       sum_w_ad_0, sum_w_ad_1, sum_w_bd_0, sum_w_bd_1, acc_r0, acc_r1,
                                                       acc_r2, acc_r3, acc_i0, acc_i1, acc_i2, acc_i3);
            }

            ggml_fairy2i_tile64_lut_store_tile_arm(tile, m, dst_col, dst_row_stride, pack_bf16, acc_r0, acc_r1, acc_r2,
                                                   acc_r3, acc_i0, acc_i1, acc_i2, acc_i3);
        }
    }
}

static GGML_FAIRY2I_NOINLINE void ggml_fairy2i_bundle_lut_qgemm_w1_neon(int                                     m,
                                                                        int                                     k,
                                                                        int                                     n,
                                                                        const struct ggml_fairy2i_bundle_desc * bundle,
                                                                        int64_t      global_m16_offset,
                                                                        const void * lut,
                                                                        const void * lut_scales,
                                                                        float *      dst,
                                                                        size_t       dst_col_stride,
                                                                        size_t       dst_row_stride,
                                                                        bool         pack_bf16) {
    const int64_t    blocks           = k / QK_FAIRY2I_TILE64;
    const int64_t    groups_per_block = QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK;
    const int64_t    groups           = blocks * groups_per_block;
    const int        tiles            = (m + 15) / 16;
    const size_t     code_stride      = (size_t) bundle->branches * 16u;
    const uint8x16_t mask_4bit        = vdupq_n_u8(0x0f);

    for (int col = 0; col < n; ++col) {
        const int8_t * lut_col = (const int8_t *) lut + (size_t) col * (size_t) groups * k_fairy2i_lut_group_bytes;
        const float *  scales  = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2u;
        uint8_t *      dst_col = (uint8_t *) dst + (size_t) col * dst_col_stride;

        for (int tile = 0; tile < tiles; ++tile) {
            const int64_t global_m16 = global_m16_offset + tile;
            float32x4_t   acc_r0     = vdupq_n_f32(0.0f);
            float32x4_t   acc_r1     = vdupq_n_f32(0.0f);
            float32x4_t   acc_r2     = vdupq_n_f32(0.0f);
            float32x4_t   acc_r3     = vdupq_n_f32(0.0f);
            float32x4_t   acc_i0     = vdupq_n_f32(0.0f);
            float32x4_t   acc_i1     = vdupq_n_f32(0.0f);
            float32x4_t   acc_i2     = vdupq_n_f32(0.0f);
            float32x4_t   acc_i3     = vdupq_n_f32(0.0f);

            for (int64_t blk = 0; blk < blocks; ++blk) {
                const uint8_t *   codes_u  = ggml_fairy2i_bundle_codes_at(bundle, global_m16, blk, 0);
                const uint8_t *   codes_w  = ggml_fairy2i_bundle_codes_at(bundle, global_m16, blk, 1);
                const ggml_half * scales_u = ggml_fairy2i_bundle_scales_at(bundle, global_m16, blk, 0);
                const ggml_half * scales_w = ggml_fairy2i_bundle_scales_at(bundle, global_m16, blk, 1);
                const int8_t * lut_ptr = lut_col + (size_t) blk * (size_t) groups_per_block * k_fairy2i_lut_group_bytes;

#    define GGML_FAIRY2I_BUNDLE_NEON_SUMS(prefix) \
        int16x8_t prefix##_ac_0 = vdupq_n_s16(0); \
        int16x8_t prefix##_ac_1 = vdupq_n_s16(0); \
        int16x8_t prefix##_bc_0 = vdupq_n_s16(0); \
        int16x8_t prefix##_bc_1 = vdupq_n_s16(0); \
        int16x8_t prefix##_ad_0 = vdupq_n_s16(0); \
        int16x8_t prefix##_ad_1 = vdupq_n_s16(0); \
        int16x8_t prefix##_bd_0 = vdupq_n_s16(0); \
        int16x8_t prefix##_bd_1 = vdupq_n_s16(0)
                GGML_FAIRY2I_BUNDLE_NEON_SUMS(sum_u);
                GGML_FAIRY2I_BUNDLE_NEON_SUMS(sum_w);
#    undef GGML_FAIRY2I_BUNDLE_NEON_SUMS

                for (int q4 = 0; q4 < groups_per_block / 2; ++q4) {
                    const int8x16x4_t ilut_0 = vld1q_s8_x4(lut_ptr + 0);
                    const int8x16x4_t ilut_1 = vld1q_s8_x4(lut_ptr + 64);
                    lut_ptr += 128;
                    ggml_fairy2i_tile64_lut_accumulate_codes_arm(
                        codes_u + (size_t) q4 * code_stride, ilut_0, ilut_1, mask_4bit, sum_u_ac_0, sum_u_ac_1,
                        sum_u_bc_0, sum_u_bc_1, sum_u_ad_0, sum_u_ad_1, sum_u_bd_0, sum_u_bd_1);
                    ggml_fairy2i_tile64_lut_accumulate_codes_arm(
                        codes_w + (size_t) q4 * code_stride, ilut_0, ilut_1, mask_4bit, sum_w_ac_0, sum_w_ac_1,
                        sum_w_bc_0, sum_w_bc_1, sum_w_ad_0, sum_w_ad_1, sum_w_bd_0, sum_w_bd_1);
                }

                const float32x4_t v_lr   = vdupq_n_f32(scales[blk * 2 + 0]);
                const float32x4_t v_li_w = vdupq_n_f32(scales[blk * 2 + 1]);
                const float32x4_t v_li_u = vnegq_f32(v_li_w);
                ggml_fairy2i_bundle_lut_apply_sums_arm(scales_u, v_lr, v_li_u, sum_u_ac_0, sum_u_ac_1, sum_u_bc_0,
                                                       sum_u_bc_1, sum_u_ad_0, sum_u_ad_1, sum_u_bd_0, sum_u_bd_1,
                                                       acc_r0, acc_r1, acc_r2, acc_r3, acc_i0, acc_i1, acc_i2, acc_i3);
                ggml_fairy2i_bundle_lut_apply_sums_arm(scales_w, v_lr, v_li_w, sum_w_ac_0, sum_w_ac_1, sum_w_bc_0,
                                                       sum_w_bc_1, sum_w_ad_0, sum_w_ad_1, sum_w_bd_0, sum_w_bd_1,
                                                       acc_r0, acc_r1, acc_r2, acc_r3, acc_i0, acc_i1, acc_i2, acc_i3);
            }

            ggml_fairy2i_tile64_lut_store_tile_arm(tile, m, dst_col, dst_row_stride, pack_bf16, acc_r0, acc_r1, acc_r2,
                                                   acc_r3, acc_i0, acc_i1, acc_i2, acc_i3);
        }
    }
}

#    undef GGML_FAIRY2I_NOINLINE
#endif

static bool ggml_fairy2i_bundle_lut_qgemm_args_valid(int                                     m,
                                                     int                                     k,
                                                     int                                     n,
                                                     const struct ggml_fairy2i_bundle_desc * bundle,
                                                     int64_t                                 global_m16_offset,
                                                     const void *                            lut,
                                                     const void *                            lut_scales,
                                                     const float *                           dst,
                                                     int                                     branches) {
    if (m == 0) {
        return true;
    }
    return m > 0 && k > 0 && n > 0 && bundle && bundle->codes && bundle->scales && bundle->logical_k == k &&
           bundle->branches == branches && global_m16_offset >= 0 &&
           global_m16_offset + (m + 15) / 16 <= bundle->logical_m / 16 && lut && lut_scales && dst;
}

static bool ggml_fairy2i_bundle_lut_force_scalar_for_test(void) {
    const char * env = getenv("GGML_FAIRY2I_TEST_FORCE_SCALAR");
    return env && strcmp(env, "0") != 0;
}

static void ggml_fairy2i_bundle_lut_qgemm_branch_scalar(int                                     m,
                                                        int                                     k,
                                                        int                                     n,
                                                        const struct ggml_fairy2i_bundle_desc * bundle,
                                                        int64_t                                 global_m16_offset,
                                                        int                                     branch,
                                                        const void *                            lut,
                                                        const void *                            lut_scales,
                                                        float *                                 dst,
                                                        size_t                                  dst_col_stride,
                                                        size_t                                  dst_row_stride,
                                                        bool                                    pack_bf16,
                                                        bool                                    negate_imag_scale,
                                                        bool                                    add) {
    const int64_t blocks           = k / QK_FAIRY2I_TILE64;
    const int64_t groups_per_block = QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK;
    const int64_t groups           = blocks * groups_per_block;
    const size_t  code_stride      = (size_t) bundle->branches * 16u;

    for (int col = 0; col < n; ++col) {
        const int8_t * lut_col = (const int8_t *) lut + (size_t) col * (size_t) groups * k_fairy2i_lut_group_bytes;
        const float *  scales  = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2u;
        uint8_t *      dst_col = (uint8_t *) dst + (size_t) col * dst_col_stride;

        for (int row = 0; row < m; ++row) {
            const int64_t global_m16 = global_m16_offset + row / 16;
            const int     lane       = row % 16;
            uint8_t *     out_base   = dst_col + (size_t) row * dst_row_stride;
            float         out_r      = 0.0f;
            float         out_i      = 0.0f;
            if (add) {
                if (pack_bf16) {
                    out_r = GGML_BF16_TO_FP32(((const ggml_bf16_t *) out_base)[0]);
                    out_i = GGML_BF16_TO_FP32(((const ggml_bf16_t *) out_base)[1]);
                } else {
                    out_r = ((const float *) out_base)[0];
                    out_i = ((const float *) out_base)[1];
                }
            }

            for (int64_t blk = 0; blk < blocks; ++blk) {
                const uint8_t *   codes         = ggml_fairy2i_bundle_codes_at(bundle, global_m16, blk, branch);
                const ggml_half * weight_scales = ggml_fairy2i_bundle_scales_at(bundle, global_m16, blk, branch);
                const float       lr            = scales[blk * 2 + 0];
                const float       li            = negate_imag_scale ? -scales[blk * 2 + 1] : scales[blk * 2 + 1];
                const float       wr            = GGML_FP16_TO_FP32(weight_scales[0]);
                const float       wi            = GGML_FP16_TO_FP32(weight_scales[1]);

                int            sum_ac  = 0;
                int            sum_bd  = 0;
                int            sum_bc  = 0;
                int            sum_ad  = 0;
                const int8_t * lut_blk = lut_col + (size_t) blk * (size_t) groups_per_block * k_fairy2i_lut_group_bytes;
                for (int q4 = 0; q4 < groups_per_block / 2; ++q4) {
                    const uint8_t  packed = codes[(size_t) q4 * code_stride + (size_t) lane];
                    const int8_t * tbl0   = lut_blk + (size_t) (q4 * 2) * k_fairy2i_lut_group_bytes;
                    const int8_t * tbl1   = tbl0 + k_fairy2i_lut_group_bytes;
                    const uint8_t  lo     = packed & 0x0fu;
                    const uint8_t  hi     = (packed >> 4) & 0x0fu;
                    sum_ac += (int) tbl0[0 * 16 + lo] + (int) tbl1[0 * 16 + hi];
                    sum_bd += (int) tbl0[1 * 16 + lo] + (int) tbl1[1 * 16 + hi];
                    sum_bc += (int) tbl0[2 * 16 + lo] + (int) tbl1[2 * 16 + hi];
                    sum_ad += (int) tbl0[3 * 16 + lo] + (int) tbl1[3 * 16 + hi];
                }

                out_r += (float) sum_ac * (lr * wr) + (float) sum_bd * (li * wi);
                out_i += (float) sum_bc * (lr * wi) + (float) sum_ad * (li * wr);
            }

            if (pack_bf16) {
                ((ggml_bf16_t *) out_base)[0] = GGML_FP32_TO_BF16(out_r);
                ((ggml_bf16_t *) out_base)[1] = GGML_FP32_TO_BF16(out_i);
            } else {
                ((float *) out_base)[0] = out_r;
                ((float *) out_base)[1] = out_i;
            }
        }
    }
}

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

bool ggml_fairy2i_tile64_lut_qgemm_two_cpu(int          m,
                                      int          k,
                                      int          n,
                                      const void * packed_u0,
                                      const void * packed_w0,
                                      const void * lut,
                                      const void * lut_scales,
                                      float *      dst,
                                      size_t       dst_col_stride,
                                      size_t       dst_row_stride,
                                      bool         pack_bf16) {
    if (m == 0) {
        return true;
    }
    if (!packed_u0 || !packed_w0 || !dst || !lut || !lut_scales || m < 0 || k <= 0 || n <= 0) {
        return false;
    }

#if defined(__AVX2__)
    // LUT preprocessing folds in conjugation. Negating the imaginary scale recovers U * x while W stays W * conj(x).
    ggml_fairy2i_tile64_lut_qgemm_w1_pair_avx2(m, k, n, packed_u0, packed_w0, lut, lut_scales, dst, dst_col_stride,
                                               dst_row_stride, pack_bf16);
#elif defined(__ARM_NEON) && defined(__aarch64__)
    // LUT preprocessing folds in conjugation. Negating U's imaginary scale recovers U * x while W stays W * conj(x).
    ggml_fairy2i_tile64_lut_qgemm_w1_pair_neon(m, k, n, packed_u0, packed_w0, lut, lut_scales, dst, dst_col_stride,
                                               dst_row_stride, pack_bf16);
#else
    ggml_fairy2i_tile64_lut_qgemm_lut16(m, k, n, packed_u0, lut, lut_scales, dst, dst_col_stride, dst_row_stride,
                                        pack_bf16, /*negate_imag_scale=*/true, /*add=*/false);
    ggml_fairy2i_tile64_lut_qgemm_lut16(m, k, n, packed_w0, lut, lut_scales, dst, dst_col_stride, dst_row_stride,
                                        pack_bf16, /*negate_imag_scale=*/false, /*add=*/true);
#endif
    return true;
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
#if defined(__AVX2__)
    if (m == 0) {
        return true;
    }
    if (!packed_u0 || !packed_u1 || !packed_w0 || !packed_w1 || !dst || !lut || !lut_scales || m < 0 || k <= 0 ||
        n <= 0) {
        return false;
    }

    // LUT preprocessing folds in conjugation. Negating the imaginary scale recovers U * x.
    ggml_fairy2i_tile64_lut_qgemm_pair_avx2(m, k, n, packed_u0, packed_u1, lut, lut_scales, dst, dst_col_stride,
                                      dst_row_stride, pack_bf16, /*negate_imag_scale*/ true, /*add*/ false);
    ggml_fairy2i_tile64_lut_qgemm_pair_avx2(m, k, n, packed_w0, packed_w1, lut, lut_scales, dst, dst_col_stride,
                                      dst_row_stride, pack_bf16, /*negate_imag_scale*/ false, /*add*/ true);
    return true;
#else
    if (m == 0) {
        return true;
    }
    if (!packed_u0 || !packed_u1 || !packed_w0 || !packed_w1 || !dst || !lut || !lut_scales || m < 0 || k <= 0 ||
        n <= 0) {
        return false;
    }

    // LUT preprocessing folds in conjugation. Negating the imaginary scale recovers U * x.
    ggml_fairy2i_tile64_lut_qgemm_pair_lut16(m, k, n, packed_u0, packed_u1, lut, lut_scales, dst, dst_col_stride,
                                             dst_row_stride, pack_bf16, /*negate_imag_scale=*/true, /*add=*/false);
    ggml_fairy2i_tile64_lut_qgemm_pair_lut16(m, k, n, packed_w0, packed_w1, lut, lut_scales, dst, dst_col_stride,
                                             dst_row_stride, pack_bf16, /*negate_imag_scale=*/false, /*add=*/true);
    return true;
#endif
}

bool ggml_fairy2i_bundle_lut_qgemm_two_cpu(int                                     m,
                                           int                                     k,
                                           int                                     n,
                                           const struct ggml_fairy2i_bundle_desc * bundle,
                                           int64_t                                 global_m16_offset,
                                           const void *                            lut,
                                           const void *                            lut_scales,
                                           float *                                 dst,
                                           size_t                                  dst_col_stride,
                                           size_t                                  dst_row_stride,
                                           bool                                    pack_bf16) {
    if (!ggml_fairy2i_bundle_lut_qgemm_args_valid(m, k, n, bundle, global_m16_offset, lut, lut_scales, dst, 2)) {
        return false;
    }
    if (m == 0) {
        return true;
    }

#if defined(__AVX2__)
    if (!ggml_fairy2i_bundle_lut_force_scalar_for_test()) {
        ggml_fairy2i_bundle_lut_qgemm_pair_avx2(m, k, n, bundle, global_m16_offset, 0, 1,
                                                /*negate_imag0=*/true, /*negate_imag1=*/false, lut, lut_scales, dst,
                                                dst_col_stride, dst_row_stride, pack_bf16, /*add=*/false);
        return true;
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    if (!ggml_fairy2i_bundle_lut_force_scalar_for_test()) {
        ggml_fairy2i_bundle_lut_qgemm_w1_neon(m, k, n, bundle, global_m16_offset, lut, lut_scales, dst, dst_col_stride,
                                              dst_row_stride, pack_bf16);
        return true;
    }
#endif
    ggml_fairy2i_bundle_lut_qgemm_branch_scalar(m, k, n, bundle, global_m16_offset, 0, lut, lut_scales, dst,
                                                dst_col_stride, dst_row_stride, pack_bf16,
                                                /*negate_imag_scale=*/true, /*add=*/false);
    ggml_fairy2i_bundle_lut_qgemm_branch_scalar(m, k, n, bundle, global_m16_offset, 1, lut, lut_scales, dst,
                                                dst_col_stride, dst_row_stride, pack_bf16,
                                                /*negate_imag_scale=*/false, /*add=*/true);
    return true;
}

bool ggml_fairy2i_bundle_lut_qgemm_four_cpu(int                                     m,
                                            int                                     k,
                                            int                                     n,
                                            const struct ggml_fairy2i_bundle_desc * bundle,
                                            int64_t                                 global_m16_offset,
                                            const void *                            lut,
                                            const void *                            lut_scales,
                                            float *                                 dst,
                                            size_t                                  dst_col_stride,
                                            size_t                                  dst_row_stride,
                                            bool                                    pack_bf16) {
    if (!ggml_fairy2i_bundle_lut_qgemm_args_valid(m, k, n, bundle, global_m16_offset, lut, lut_scales, dst, 4)) {
        return false;
    }
    if (m == 0) {
        return true;
    }

#if defined(__AVX2__)
    if (!ggml_fairy2i_bundle_lut_force_scalar_for_test()) {
        if (n > 1) {
            ggml_fairy2i_bundle_lut_qgemm_four_avx2(m, k, n, bundle, global_m16_offset, lut, lut_scales, dst,
                                                    dst_col_stride, dst_row_stride, pack_bf16);
        } else {
            ggml_fairy2i_bundle_lut_qgemm_pair_avx2(m, k, n, bundle, global_m16_offset, 0, 1,
                                                    /*negate_imag0=*/true, /*negate_imag1=*/true, lut, lut_scales, dst,
                                                    dst_col_stride, dst_row_stride, pack_bf16, /*add=*/false);
            ggml_fairy2i_bundle_lut_qgemm_pair_avx2(m, k, n, bundle, global_m16_offset, 2, 3, /*negate_imag0=*/false,
                                                    /*negate_imag1=*/false, lut, lut_scales, dst, dst_col_stride,
                                                    dst_row_stride, pack_bf16, /*add=*/true);
        }
        return true;
    }
#endif
    for (int branch = 0; branch < 4; ++branch) {
        ggml_fairy2i_bundle_lut_qgemm_branch_scalar(m, k, n, bundle, global_m16_offset, branch, lut, lut_scales, dst,
                                                    dst_col_stride, dst_row_stride, pack_bf16,
                                                    /*negate_imag_scale=*/branch < 2, /*add=*/branch != 0);
    }
    return true;
}
