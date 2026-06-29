#include "wide-linear.h"
#include "lut-qgemm.h"

#ifdef GGML_USE_FAIRY2I_CPU_LUT

#include "ggml-fairy2i-lut-impl.h"
#include "ggml-fairy2i-lut.h"
#include "quants.h"

#include <algorithm>
#include <math.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#    include <immintrin.h>
#endif
#    if defined(__ARM_NEON) && defined(__aarch64__)
#        include <arm_neon.h>
#    endif

static inline float ggml_fairy2i_wide_linear_lut_bias_at(const struct ggml_tensor * bias,
                                                         int64_t                    i0,
                                                         int64_t                    i1,
                                                         int64_t                    i2,
                                                         int64_t                    i3) {
    const char * ptr = (const char *) bias->data + (i0 % bias->ne[0]) * bias->nb[0] +
                       (i1 % bias->ne[1]) * bias->nb[1] + (i2 % bias->ne[2]) * bias->nb[2] +
                       (i3 % bias->ne[3]) * bias->nb[3];
    return *(const float *) ptr;
}

static bool ggml_fairy2i_wide_linear_have_packed_weights(const struct ggml_tensor * dst, int first_src, int last_src) {
    for (int i = first_src; i <= last_src; ++i) {
        const struct ggml_tensor * weight = dst->src[i];
        const struct fairy2i_lut_extra * extra = weight ? (const struct fairy2i_lut_extra *) weight->extra : nullptr;
        if (!extra || !extra->packed_w) {
            return false;
        }
    }
    return true;
}

static bool ggml_fairy2i_wide_linear_w1_have_packed_weights(const struct ggml_tensor * dst) {
    return ggml_fairy2i_wide_linear_have_packed_weights(dst, 1, 2);
}

static bool ggml_fairy2i_wide_linear_w2_have_packed_weights(const struct ggml_tensor * dst) {
    return ggml_fairy2i_wide_linear_have_packed_weights(dst, 1, 4);
}

void ggml_fairy2i_tile64_lut_quantize_block_q16_64_for_test(const float *              x,
                                                            block_fairy2i_act_q16_64 * y,
                                                            bool                       force_scalar);

static void ggml_fairy2i_tile64_lut_quantize_block_q16_64_scalar(const float * x, block_fairy2i_act_q16_64 * y) {
    float max_real = 1e-5f;
    float max_imag = 1e-5f;
    for (int j = 0; j < QK_FAIRY2I_TILE64; ++j) {
        const ggml_bf16_t * value = (const ggml_bf16_t *) (x + j);
        max_real                  = std::max(max_real, fabsf(GGML_BF16_TO_FP32(value[0])));
        max_imag                  = std::max(max_imag, fabsf(GGML_BF16_TO_FP32(value[1])));
    }

    // A two-value LUT entry remains in int8 when each quantized activation is limited to +/-63.
    const float iscale_real = 63.0f / max_real;
    const float iscale_imag = 63.0f / max_imag;
    y->d_real               = GGML_FP32_TO_FP16(1.0f / iscale_real);
    y->d_imag               = GGML_FP32_TO_FP16(1.0f / iscale_imag);

    for (int j = 0; j < QK_FAIRY2I_TILE64; ++j) {
        const ggml_bf16_t * value = (const ggml_bf16_t *) (x + j);
        const int           qr    = (int) roundf(iscale_real * GGML_BF16_TO_FP32(value[0]));
        const int           qi    = (int) roundf(iscale_imag * GGML_BF16_TO_FP32(value[1]));
        y->x_real[j]              = (uint8_t) std::max(-63, std::min(63, qr));
        y->x_imag[j]              = (uint8_t) std::max(-63, std::min(63, qi));
    }
}

#    if defined(__ARM_NEON) && defined(__aarch64__)
static inline void ggml_fairy2i_tile64_lut_store_i32x4_as_i8_neon(uint8_t * dst, int32x4_t v) {
    const int16x4_t v16 = vqmovn_s32(v);
    const int8x8_t  v8  = vqmovn_s16(vcombine_s16(v16, vdup_n_s16(0)));
    vst1_lane_s32((int32_t *) (void *) dst, vreinterpret_s32_s8(v8), 0);
}

static void ggml_fairy2i_tile64_lut_quantize_block_q16_64_neon(const float * x, block_fairy2i_act_q16_64 * y) {
    const uint32_t * src_words = (const uint32_t *) x;
    const uint32x4_t imag_mask = vdupq_n_u32(0xffff0000u);
    float32x4_t      max_real  = vdupq_n_f32(1e-5f);
    float32x4_t      max_imag  = vdupq_n_f32(1e-5f);

    for (int j = 0; j < QK_FAIRY2I_TILE64; j += 4) {
        const uint32x4_t  words = vld1q_u32(src_words + j);
        const float32x4_t real  = vreinterpretq_f32_u32(vshlq_n_u32(words, 16));
        const float32x4_t imag  = vreinterpretq_f32_u32(vandq_u32(words, imag_mask));
        max_real                = vmaxq_f32(max_real, vabsq_f32(real));
        max_imag                = vmaxq_f32(max_imag, vabsq_f32(imag));
    }

    const float scalar_max_real = vmaxvq_f32(max_real);
    const float scalar_max_imag = vmaxvq_f32(max_imag);
    const float iscale_real     = 63.0f / scalar_max_real;
    const float iscale_imag     = 63.0f / scalar_max_imag;
    y->d_real                   = GGML_FP32_TO_FP16(1.0f / iscale_real);
    y->d_imag                   = GGML_FP32_TO_FP16(1.0f / iscale_imag);

    const float32x4_t scale_real = vdupq_n_f32(iscale_real);
    const float32x4_t scale_imag = vdupq_n_f32(iscale_imag);
    const int32x4_t   qmin       = vdupq_n_s32(-63);
    const int32x4_t   qmax       = vdupq_n_s32(63);
    for (int j = 0; j < QK_FAIRY2I_TILE64; j += 4) {
        const uint32x4_t  words = vld1q_u32(src_words + j);
        const float32x4_t real  = vreinterpretq_f32_u32(vshlq_n_u32(words, 16));
        const float32x4_t imag  = vreinterpretq_f32_u32(vandq_u32(words, imag_mask));
        const int32x4_t   qr    = vminq_s32(qmax, vmaxq_s32(qmin, vcvtaq_s32_f32(vmulq_f32(real, scale_real))));
        const int32x4_t   qi    = vminq_s32(qmax, vmaxq_s32(qmin, vcvtaq_s32_f32(vmulq_f32(imag, scale_imag))));
        ggml_fairy2i_tile64_lut_store_i32x4_as_i8_neon(y->x_real + j, qr);
        ggml_fairy2i_tile64_lut_store_i32x4_as_i8_neon(y->x_imag + j, qi);
    }
}
#    endif

static void ggml_fairy2i_tile64_lut_quantize_block_q16_64(const float * x, block_fairy2i_act_q16_64 * y, bool use_avx2) {
#if defined(__AVX2__)
    if (use_avx2) {
        const uint32_t * src_words = (const uint32_t *) x;
        const __m256     sign_mask = _mm256_set1_ps(-0.0f);
        const __m256i    imag_mask = _mm256_set1_epi32((int) 0xffff0000u);
        __m256           max_real  = _mm256_set1_ps(1e-5f);
        __m256           max_imag  = _mm256_set1_ps(1e-5f);

        for (int j = 0; j < QK_FAIRY2I_TILE64; j += 8) {
            const __m256i words = _mm256_loadu_si256((const __m256i *) (src_words + j));
            const __m256  real  = _mm256_castsi256_ps(_mm256_slli_epi32(words, 16));
            const __m256  imag  = _mm256_castsi256_ps(_mm256_and_si256(words, imag_mask));
            max_real = _mm256_max_ps(max_real, _mm256_andnot_ps(sign_mask, real));
            max_imag = _mm256_max_ps(max_imag, _mm256_andnot_ps(sign_mask, imag));
        }

        alignas(32) float maxima_real[8];
        alignas(32) float maxima_imag[8];
        _mm256_store_ps(maxima_real, max_real);
        _mm256_store_ps(maxima_imag, max_imag);
        float scalar_max_real = maxima_real[0];
        float scalar_max_imag = maxima_imag[0];
        for (int lane = 1; lane < 8; ++lane) {
            scalar_max_real = std::max(scalar_max_real, maxima_real[lane]);
            scalar_max_imag = std::max(scalar_max_imag, maxima_imag[lane]);
        }

        const float iscale_real = 63.0f / scalar_max_real;
        const float iscale_imag = 63.0f / scalar_max_imag;
        y->d_real               = GGML_FP32_TO_FP16(1.0f / iscale_real);
        y->d_imag               = GGML_FP32_TO_FP16(1.0f / iscale_imag);

        const __m256  scale_real = _mm256_set1_ps(iscale_real);
        const __m256  scale_imag = _mm256_set1_ps(iscale_imag);
        const __m256i qmin       = _mm256_set1_epi32(-63);
        const __m256i qmax       = _mm256_set1_epi32(63);
        for (int j = 0; j < QK_FAIRY2I_TILE64; j += 8) {
            const __m256i words = _mm256_loadu_si256((const __m256i *) (src_words + j));
            const __m256  real  = _mm256_castsi256_ps(_mm256_slli_epi32(words, 16));
            const __m256  imag  = _mm256_castsi256_ps(_mm256_and_si256(words, imag_mask));
            const __m256i qr =
                _mm256_max_epi32(qmin, _mm256_min_epi32(qmax, _mm256_cvtps_epi32(_mm256_mul_ps(real, scale_real))));
            const __m256i qi =
                _mm256_max_epi32(qmin, _mm256_min_epi32(qmax, _mm256_cvtps_epi32(_mm256_mul_ps(imag, scale_imag))));
            const __m128i qr16 = _mm_packs_epi32(_mm256_castsi256_si128(qr), _mm256_extracti128_si256(qr, 1));
            const __m128i qi16 = _mm_packs_epi32(_mm256_castsi256_si128(qi), _mm256_extracti128_si256(qi, 1));
            _mm_storel_epi64((__m128i *) (y->x_real + j), _mm_packs_epi16(qr16, qr16));
            _mm_storel_epi64((__m128i *) (y->x_imag + j), _mm_packs_epi16(qi16, qi16));
        }
        return;
    }
#else
    (void) use_avx2;
#endif

#    if defined(__ARM_NEON) && defined(__aarch64__)
    ggml_fairy2i_tile64_lut_quantize_block_q16_64_neon(x, y);
    return;
#    endif

    ggml_fairy2i_tile64_lut_quantize_block_q16_64_scalar(x, y);
}

void ggml_fairy2i_tile64_lut_quantize_block_q16_64_for_test(const float *              x,
                                                            block_fairy2i_act_q16_64 * y,
                                                            bool                       force_scalar) {
    if (force_scalar) {
        ggml_fairy2i_tile64_lut_quantize_block_q16_64_scalar(x, y);
        return;
    }
    ggml_fairy2i_tile64_lut_quantize_block_q16_64(x, y, true);
}

static size_t ggml_fairy2i_wide_linear_lut_wsize_common(const struct ggml_tensor * dst) {
    const struct ggml_tensor * x = dst ? dst->src[0] : nullptr;
    if (!x || x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || x->ne[0] % QK_FAIRY2I_TILE64 != 0) {
        return 0;
    }

    const char * enabled_env = getenv("GGML_FAIRY2I_LUT");
    if (!enabled_env || strcmp(enabled_env, "0") == 0) {
        return 0;
    }

    const size_t n             = (size_t) ggml_nrows(x);
    const size_t k             = (size_t) x->ne[0];
    const size_t m             = (size_t) dst->ne[0];
    const size_t weight_blocks = k / QK_FAIRY2I_TILE64;
    const size_t groups        = weight_blocks * QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK;

    GGML_ASSERT(n == 0 || weight_blocks <= SIZE_MAX / n);
    const size_t q_elems = n * weight_blocks;
    GGML_ASSERT(q_elems == 0 || sizeof(block_fairy2i_act_q16_64) <= SIZE_MAX / q_elems);
    const size_t q_bytes = GGML_PAD(q_elems * sizeof(block_fairy2i_act_q16_64), 64);

    GGML_ASSERT(n == 0 || groups <= SIZE_MAX / n);
    const size_t lut_groups = n * groups;
    GGML_ASSERT(lut_groups == 0 || k_fairy2i_lut_group_bytes <= SIZE_MAX / lut_groups);
    const size_t lut_bytes = lut_groups * k_fairy2i_lut_group_bytes;

    GGML_ASSERT(n == 0 || weight_blocks <= SIZE_MAX / n);
    const size_t scale_elems = n * weight_blocks;
    GGML_ASSERT(scale_elems <= SIZE_MAX / (2u * sizeof(float)));
    const size_t scale_bytes = scale_elems * 2u * sizeof(float);
    GGML_ASSERT(lut_bytes <= SIZE_MAX - scale_bytes);
    const size_t shared_bytes = GGML_PAD(lut_bytes + scale_bytes, 64);

    GGML_ASSERT(n == 0 || m <= SIZE_MAX / n);
    const size_t output_elems = n * m;
    GGML_ASSERT(output_elems <= SIZE_MAX / (2u * sizeof(float)));
    const size_t output_bytes = output_elems * 2u * sizeof(float);

    GGML_ASSERT(q_bytes <= SIZE_MAX - shared_bytes);
    GGML_ASSERT(q_bytes + shared_bytes <= SIZE_MAX - output_bytes);
    return q_bytes + shared_bytes + output_bytes;
}

size_t ggml_fairy2i_wide_linear_w1_lut_wsize(const struct ggml_tensor * dst) {
    return ggml_fairy2i_wide_linear_lut_wsize_common(dst);
}

size_t ggml_fairy2i_wide_linear_w2_lut_wsize(const struct ggml_tensor * dst) {
    return ggml_fairy2i_wide_linear_lut_wsize_common(dst);
}

bool ggml_fairy2i_wide_linear_w1_compute_lut(const struct ggml_compute_params * params,
                                              struct ggml_tensor *                dst,
                                              bool                                lut_c) {
    if (lut_c) {
        return false;
    }
    if (!ggml_fairy2i_wide_linear_w1_have_packed_weights(dst)) {
        return false;
    }

    const struct ggml_tensor * x      = dst->src[0];
    const struct ggml_tensor * bias   = dst->src[3];
    const int64_t K                   = x->ne[0];
    const int64_t M                   = dst->ne[0];
    const int64_t N                   = ggml_nrows(x);
    const int64_t weight_blocks       = K / QK_FAIRY2I_TILE64;
    const int64_t groups              = weight_blocks * QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK;
    const size_t  q_bytes             = GGML_PAD((size_t) N * (size_t) weight_blocks * sizeof(block_fairy2i_act_q16_64), 64);
    const size_t  lut_bytes           = (size_t) N * (size_t) groups * k_fairy2i_lut_group_bytes;
    const size_t  shared_bytes        =
        GGML_PAD(lut_bytes + (size_t) N * (size_t) weight_blocks * 2u * sizeof(float), 64);
    const size_t  need                = ggml_fairy2i_wide_linear_w1_lut_wsize(dst);

    if (!params->wdata || params->wsize < need) {
        return false;
    }

    block_fairy2i_act_q16_64 * q_x    = (block_fairy2i_act_q16_64 *) params->wdata;
    uint8_t *            shared = (uint8_t *) params->wdata + q_bytes;
    void *               lut    = shared;
    float *              scales = (float *) (shared + lut_bytes);
    float *              output = (float *) (shared + shared_bytes);

    if (N >= params->nth) {
        for (int64_t ir = params->ith; ir < N; ir += params->nth) {
            const float * x_row = (const float *) ((const char *) x->data + ir * x->nb[1]);
            block_fairy2i_act_q16_64 * q_row = q_x + ir * weight_blocks;
            for (int64_t ib = 0; ib < weight_blocks; ++ib) {
                ggml_fairy2i_tile64_lut_quantize_block_q16_64(x_row + ib * QK_FAIRY2I_TILE64, q_row + ib, true);
            }
        }
        ggml_fairy2i_tile64_lut_preprocess_ex_q16_64_lut16((int) M, (int) K, (int) N, q_x,
                                                       (size_t) weight_blocks * sizeof(block_fairy2i_act_q16_64), scales,
                                                       lut, params->ith, params->nth);
    } else {
        for (int64_t ir = 0; ir < N; ++ir) {
            const float * x_row = (const float *) ((const char *) x->data + ir * x->nb[1]);
            block_fairy2i_act_q16_64 * q_row = q_x + ir * weight_blocks;
            float * scale_row = scales + ir * weight_blocks * 2;
            int8_t * lut_row = (int8_t *) lut + ir * groups * k_fairy2i_lut_group_bytes;

            for (int64_t ib = params->ith; ib < weight_blocks; ib += params->nth) {
                ggml_fairy2i_tile64_lut_quantize_block_q16_64(x_row + ib * QK_FAIRY2I_TILE64, q_row + ib, false);
                ggml_fairy2i_tile64_lut_preprocess_q16_64_block_lut16(
                    q_row + ib, scale_row + ib * 2,
                    lut_row + ib * QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK * k_fairy2i_lut_group_bytes);
            }
        }
    }
    ggml_barrier(params->threadpool);

    const int64_t tiles_total = (M + 15) / 16;
    const int64_t tile0       = (tiles_total * params->ith) / params->nth;
    const int64_t tile1       = (tiles_total * (params->ith + 1)) / params->nth;
    const int64_t row0        = tile0 * 16;
    const int64_t row1        = std::min<int64_t>(tile1 * 16, M);
    const int64_t nrows       = row1 - row0;
    const size_t  packed_tile_bytes = (size_t) weight_blocks * sizeof(fairy2i_tile64_lut_wtile_16);

    const fairy2i_lut_extra * extra_u0 = (const fairy2i_lut_extra *) dst->src[1]->extra;
    const fairy2i_lut_extra * extra_w0 = (const fairy2i_lut_extra *) dst->src[2]->extra;
    const void * packed_u0 = (const uint8_t *) extra_u0->packed_w + (size_t) tile0 * packed_tile_bytes;
    const void * packed_w0 = (const uint8_t *) extra_w0->packed_w + (size_t) tile0 * packed_tile_bytes;

    ggml_fairy2i_tile64_lut_qgemm_lut16((int) nrows, (int) K, (int) N, packed_u0, lut, scales, output + row0 * 2,
                                        (size_t) M * 2u * sizeof(float), 2u * sizeof(float), false, true, false);
    ggml_fairy2i_tile64_lut_qgemm_lut16((int) nrows, (int) K, (int) N, packed_w0, lut, scales, output + row0 * 2,
                                        (size_t) M * 2u * sizeof(float), 2u * sizeof(float), false, false, true);

    for (int64_t col = 0; col < N; ++col) {
        const int64_t i1 = col % x->ne[1];
        const int64_t i2 = (col / x->ne[1]) % x->ne[2];
        const int64_t i3 = col / (x->ne[1] * x->ne[2]);
        const float * output_col = output + col * M * 2;
        char * dst_col = (char *) dst->data + col * dst->nb[1];
        for (int64_t row = row0; row < row1; ++row) {
            float real = output_col[row * 2 + 0];
            float imag = output_col[row * 2 + 1];
            if (bias) {
                real += ggml_fairy2i_wide_linear_lut_bias_at(bias, row, i1, i2, i3);
                imag += ggml_fairy2i_wide_linear_lut_bias_at(bias, row + M, i1, i2, i3);
            }
            ggml_bf16_t * out = (ggml_bf16_t *) (dst_col + row * dst->nb[0]);
            out[0]             = GGML_FP32_TO_BF16(real);
            out[1]             = GGML_FP32_TO_BF16(imag);
        }
    }
    return true;
}

bool ggml_fairy2i_wide_linear_w2_compute_lut(const struct ggml_compute_params * params,
                                              struct ggml_tensor *                dst,
                                              bool                                lut_c) {
    if (lut_c) {
        return false;
    }
    if (!ggml_fairy2i_wide_linear_w2_have_packed_weights(dst)) {
        return false;
    }

    const struct ggml_tensor * x      = dst->src[0];
    const struct ggml_tensor * bias   = dst->src[5];
    const int64_t K                   = x->ne[0];
    const int64_t M                   = dst->ne[0];
    const int64_t N                   = ggml_nrows(x);
    const int64_t weight_blocks       = K / QK_FAIRY2I_TILE64;
    const int64_t groups              = weight_blocks * QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK;
    const size_t  q_bytes             = GGML_PAD((size_t) N * (size_t) weight_blocks * sizeof(block_fairy2i_act_q16_64), 64);
    const size_t  lut_bytes           = (size_t) N * (size_t) groups * k_fairy2i_lut_group_bytes;
    const size_t  shared_bytes        =
        GGML_PAD(lut_bytes + (size_t) N * (size_t) weight_blocks * 2u * sizeof(float), 64);
    const size_t  need                = ggml_fairy2i_wide_linear_w2_lut_wsize(dst);

    if (!params->wdata || params->wsize < need) {
        return false;
    }

    block_fairy2i_act_q16_64 * q_x    = (block_fairy2i_act_q16_64 *) params->wdata;
    uint8_t *            shared = (uint8_t *) params->wdata + q_bytes;
    void *               lut    = shared;
    float *              scales = (float *) (shared + lut_bytes);
    float *              output = (float *) (shared + shared_bytes);

    if (N >= params->nth) {
        for (int64_t ir = params->ith; ir < N; ir += params->nth) {
            const float * x_row = (const float *) ((const char *) x->data + ir * x->nb[1]);
            block_fairy2i_act_q16_64 * q_row = q_x + ir * weight_blocks;
            for (int64_t ib = 0; ib < weight_blocks; ++ib) {
                ggml_fairy2i_tile64_lut_quantize_block_q16_64(x_row + ib * QK_FAIRY2I_TILE64, q_row + ib, true);
            }
        }
        ggml_fairy2i_tile64_lut_preprocess_ex_q16_64_lut16((int) M, (int) K, (int) N, q_x,
                                                       (size_t) weight_blocks * sizeof(block_fairy2i_act_q16_64), scales,
                                                       lut, params->ith, params->nth);
    } else {
        for (int64_t ir = 0; ir < N; ++ir) {
            const float * x_row = (const float *) ((const char *) x->data + ir * x->nb[1]);
            block_fairy2i_act_q16_64 * q_row = q_x + ir * weight_blocks;
            float * scale_row = scales + ir * weight_blocks * 2;
            int8_t * lut_row = (int8_t *) lut + ir * groups * k_fairy2i_lut_group_bytes;

            for (int64_t ib = params->ith; ib < weight_blocks; ib += params->nth) {
                ggml_fairy2i_tile64_lut_quantize_block_q16_64(x_row + ib * QK_FAIRY2I_TILE64, q_row + ib, false);
                ggml_fairy2i_tile64_lut_preprocess_q16_64_block_lut16(
                    q_row + ib, scale_row + ib * 2,
                    lut_row + ib * QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK * k_fairy2i_lut_group_bytes);
            }
        }
    }
    ggml_barrier(params->threadpool);

    const int64_t tiles_total = (M + 15) / 16;
    const int64_t tile0       = (tiles_total * params->ith) / params->nth;
    const int64_t tile1       = (tiles_total * (params->ith + 1)) / params->nth;
    const int64_t row0        = tile0 * 16;
    const int64_t row1        = std::min<int64_t>(tile1 * 16, M);
    const int64_t nrows       = row1 - row0;
    const size_t  packed_tile_bytes = (size_t) weight_blocks * sizeof(fairy2i_tile64_lut_wtile_16);

    const fairy2i_lut_extra * extra_u0 = (const fairy2i_lut_extra *) dst->src[1]->extra;
    const fairy2i_lut_extra * extra_u1 = (const fairy2i_lut_extra *) dst->src[2]->extra;
    const fairy2i_lut_extra * extra_w0 = (const fairy2i_lut_extra *) dst->src[3]->extra;
    const fairy2i_lut_extra * extra_w1 = (const fairy2i_lut_extra *) dst->src[4]->extra;
    const void * packed_u0 = (const uint8_t *) extra_u0->packed_w + (size_t) tile0 * packed_tile_bytes;
    const void * packed_u1 = (const uint8_t *) extra_u1->packed_w + (size_t) tile0 * packed_tile_bytes;
    const void * packed_w0 = (const uint8_t *) extra_w0->packed_w + (size_t) tile0 * packed_tile_bytes;
    const void * packed_w1 = (const uint8_t *) extra_w1->packed_w + (size_t) tile0 * packed_tile_bytes;
    const bool qgemm_ok = ggml_fairy2i_tile64_lut_qgemm_four_cpu(
        (int) nrows, (int) K, (int) N, packed_u0, packed_u1, packed_w0, packed_w1, lut, scales, output + row0 * 2,
        (size_t) M * 2u * sizeof(float), 2u * sizeof(float), false);
    if (!qgemm_ok) {
        return false;
    }

    for (int64_t col = 0; col < N; ++col) {
        const int64_t i1 = col % x->ne[1];
        const int64_t i2 = (col / x->ne[1]) % x->ne[2];
        const int64_t i3 = col / (x->ne[1] * x->ne[2]);
        const float * output_col = output + col * M * 2;
        char * dst_col = (char *) dst->data + col * dst->nb[1];
        for (int64_t row = row0; row < row1; ++row) {
            float real = output_col[row * 2 + 0];
            float imag = output_col[row * 2 + 1];
            if (bias) {
                real += ggml_fairy2i_wide_linear_lut_bias_at(bias, row, i1, i2, i3);
                imag += ggml_fairy2i_wide_linear_lut_bias_at(bias, row + M, i1, i2, i3);
            }
            ggml_bf16_t * out = (ggml_bf16_t *) (dst_col + row * dst->nb[0]);
            out[0]             = GGML_FP32_TO_BF16(real);
            out[1]             = GGML_FP32_TO_BF16(imag);
        }
    }
    return true;
}

#else

size_t ggml_fairy2i_wide_linear_w1_lut_wsize(const struct ggml_tensor * dst) {
    (void) dst;
    return 0;
}

size_t ggml_fairy2i_wide_linear_w2_lut_wsize(const struct ggml_tensor * dst) {
    (void) dst;
    return 0;
}

bool ggml_fairy2i_wide_linear_w1_compute_lut(const struct ggml_compute_params * params,
                                              struct ggml_tensor *                dst,
                                              bool                                lut_c) {
    (void) params;
    (void) dst;
    (void) lut_c;
    return false;
}

bool ggml_fairy2i_wide_linear_w2_compute_lut(const struct ggml_compute_params * params,
                                              struct ggml_tensor *                dst,
                                              bool                                lut_c) {
    (void) params;
    (void) dst;
    (void) lut_c;
    return false;
}

#endif
