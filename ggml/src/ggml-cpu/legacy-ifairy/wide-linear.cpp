#include "wide-linear.h"

#include "ggml-impl.h"
#include "ggml-threading.h"
#include "quants.h"

#include <stdint.h>

struct ggml_ifairy_complex_acc {
    float real;
    float imag;
};

static inline void ggml_ifairy64_fuse_accumulate_block_scalar(const block_ifairy64 *      w,
                                                               const block_ifairy64_q16 * x,
                                                               int32_t                     sums[4]) {
    for (int part = 0; part < 4; ++part) {
        for (int lane = 0; lane < 16; ++lane) {
            const int     idx  = part * 16 + lane;
            const uint8_t code = (w->qs[lane] >> (2 * part)) & 0x3u;
            const int     xr   = (int) ((const int8_t *) x->x_real)[idx];
            const int     xi   = (int) ((const int8_t *) x->x_imag)[idx];

            switch (code) {
                case 0:
                    sums[0] -= xr;
                    sums[1] -= xi;
                    break;
                case 1:
                    sums[0] += xr;
                    sums[1] += xi;
                    break;
                case 2:
                    sums[2] -= xr;
                    sums[3] -= xi;
                    break;
                case 3:
                    sums[2] += xr;
                    sums[3] += xi;
                    break;
                default:
                    GGML_UNREACHABLE();
            }
        }
    }
}

#if defined(__AVX2__)
static inline int32_t ggml_ifairy_hsum_i16x8_avx2(__m128i v) {
    const __m128i ones = _mm_set1_epi16(1);
    __m128i       s32  = _mm_madd_epi16(v, ones);
    s32                = _mm_add_epi32(s32, _mm_shuffle_epi32(s32, _MM_SHUFFLE(1, 0, 3, 2)));
    s32                = _mm_add_epi32(s32, _mm_shuffle_epi32(s32, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s32);
}

static inline __m256i ggml_ifairy64_pack_weight_pair(const block_ifairy64 * w0, const block_ifairy64 * w1) {
    const __m128i packed0 = _mm_loadu_si128((const __m128i *) w0->qs);
    const __m128i packed1 = _mm_loadu_si128((const __m128i *) w1->qs);
    return _mm256_inserti128_si256(_mm256_castsi128_si256(packed0), packed1, 1);
}

static inline void ggml_ifairy64_fuse_accumulate_code_pair_avx2(__m256i        codes,
                                                                 __m256i        xr,
                                                                 __m256i        xi,
                                                                 __m256i        neg_xr,
                                                                 __m256i        neg_xi,
                                                                 __m256i        zero,
                                                                 __m256i        one_byte,
                                                                 __m256i        two_byte,
                                                                 __m256i        madd_ones,
                                                                 __m256i        acc[4]) {
    const __m256i positive_mask = _mm256_cmpeq_epi8(_mm256_and_si256(codes, one_byte), one_byte);
    const __m256i real_mask     = _mm256_cmpeq_epi8(_mm256_and_si256(codes, two_byte), zero);
    const __m256i imag_mask     = _mm256_cmpeq_epi8(_mm256_and_si256(codes, two_byte), two_byte);

    const __m256i signed_xr = _mm256_blendv_epi8(neg_xr, xr, positive_mask);
    const __m256i signed_xi = _mm256_blendv_epi8(neg_xi, xi, positive_mask);

    acc[0] = _mm256_add_epi16(acc[0], _mm256_maddubs_epi16(madd_ones, _mm256_blendv_epi8(zero, signed_xr, real_mask)));
    acc[1] = _mm256_add_epi16(acc[1], _mm256_maddubs_epi16(madd_ones, _mm256_blendv_epi8(zero, signed_xi, real_mask)));
    acc[2] = _mm256_add_epi16(acc[2], _mm256_maddubs_epi16(madd_ones, _mm256_blendv_epi8(zero, signed_xr, imag_mask)));
    acc[3] = _mm256_add_epi16(acc[3], _mm256_maddubs_epi16(madd_ones, _mm256_blendv_epi8(zero, signed_xi, imag_mask)));
}

static inline void ggml_ifairy64_fuse_accumulate_block_four_avx2(const block_ifairy64 *      u0,
                                                                  const block_ifairy64 *      u1,
                                                                  const block_ifairy64 *      w0,
                                                                  const block_ifairy64 *      w1,
                                                                  const block_ifairy64_q16 * x,
                                                                  int32_t                      sums[4][4]) {
    const __m256i packed_u  = ggml_ifairy64_pack_weight_pair(u0, u1);
    const __m256i packed_w  = ggml_ifairy64_pack_weight_pair(w0, w1);
    const __m256i zero      = _mm256_setzero_si256();
    const __m256i one_byte  = _mm256_set1_epi8(1);
    const __m256i two_byte  = _mm256_set1_epi8(2);
    const __m256i code_mask = _mm256_set1_epi8(3);
    const __m256i madd_ones = _mm256_set1_epi8(1);
    __m256i       acc_u[4]  = { zero, zero, zero, zero };
    __m256i       acc_w[4]  = { zero, zero, zero, zero };

    for (int part = 0; part < 4; ++part) {
        const __m128i xr128 = _mm_loadu_si128((const __m128i *) ((const int8_t *) x->x_real + part * 16));
        const __m128i xi128 = _mm_loadu_si128((const __m128i *) ((const int8_t *) x->x_imag + part * 16));
        const __m256i xr    = _mm256_broadcastsi128_si256(xr128);
        const __m256i xi    = _mm256_broadcastsi128_si256(xi128);
        const __m256i neg_xr = _mm256_sub_epi8(zero, xr);
        const __m256i neg_xi = _mm256_sub_epi8(zero, xi);

        const __m256i codes_u = _mm256_and_si256(_mm256_srli_epi16(packed_u, 2 * part), code_mask);
        const __m256i codes_w = _mm256_and_si256(_mm256_srli_epi16(packed_w, 2 * part), code_mask);
        ggml_ifairy64_fuse_accumulate_code_pair_avx2(codes_u, xr, xi, neg_xr, neg_xi, zero, one_byte, two_byte,
                                                      madd_ones, acc_u);
        ggml_ifairy64_fuse_accumulate_code_pair_avx2(codes_w, xr, xi, neg_xr, neg_xi, zero, one_byte, two_byte,
                                                      madd_ones, acc_w);
    }

    for (int channel = 0; channel < 4; ++channel) {
        sums[0][channel] = ggml_ifairy_hsum_i16x8_avx2(_mm256_castsi256_si128(acc_u[channel]));
        sums[1][channel] = ggml_ifairy_hsum_i16x8_avx2(_mm256_extracti128_si256(acc_u[channel], 1));
        sums[2][channel] = ggml_ifairy_hsum_i16x8_avx2(_mm256_castsi256_si128(acc_w[channel]));
        sums[3][channel] = ggml_ifairy_hsum_i16x8_avx2(_mm256_extracti128_si256(acc_w[channel], 1));
    }
}
#endif

#if defined(GGML_USE_LEGACY_IFAIRY_CPU_AVX512) && defined(__AVX512F__) && defined(__AVX512BW__)
static inline __attribute__((always_inline)) __m512i ggml_ifairy64_pack_weight_four_avx512(const block_ifairy64 * u0,
                                                                                            const block_ifairy64 * u1,
                                                                                            const block_ifairy64 * w0,
                                                                                            const block_ifairy64 * w1) {
    __m512i packed = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i *) u0->qs));
    packed         = _mm512_inserti32x4(packed, _mm_loadu_si128((const __m128i *) u1->qs), 1);
    packed         = _mm512_inserti32x4(packed, _mm_loadu_si128((const __m128i *) w0->qs), 2);
    packed         = _mm512_inserti32x4(packed, _mm_loadu_si128((const __m128i *) w1->qs), 3);
    return packed;
}

static inline __attribute__((always_inline)) __m512i ggml_ifairy64_fuse_apply_sign_avx512(__m512i   values,
                                                                                          __mmask64 positive) {
    return _mm512_mask_sub_epi8(values, ~positive, _mm512_setzero_si512(), values);
}

static inline __attribute__((always_inline)) void ggml_ifairy64_fuse_accumulate_block_four_avx512(
    const block_ifairy64 * u0, const block_ifairy64 * u1, const block_ifairy64 * w0, const block_ifairy64 * w1,
    const block_ifairy64_q16 * x, int32_t sums[4][4]) {
    const __m512i packed    = ggml_ifairy64_pack_weight_four_avx512(u0, u1, w0, w1);
    const __m512i one_byte  = _mm512_set1_epi8(1);
    const __m512i two_byte  = _mm512_set1_epi8(2);
    const __m512i code_mask = _mm512_set1_epi8(3);
    const __m512i madd_ones = _mm512_set1_epi8(1);
    const __m512i zero      = _mm512_setzero_si512();
    __m512i       acc[4]    = { zero, zero, zero, zero };

    for (int part = 0; part < 4; ++part) {
        const __m128i xr128 = _mm_loadu_si128((const __m128i *) ((const int8_t *) x->x_real + part * 16));
        const __m128i xi128 = _mm_loadu_si128((const __m128i *) ((const int8_t *) x->x_imag + part * 16));
        const __m512i xr    = _mm512_broadcast_i32x4(xr128);
        const __m512i xi    = _mm512_broadcast_i32x4(xi128);
        const __m512i codes = _mm512_and_si512(_mm512_srli_epi16(packed, 2 * part), code_mask);

        const __mmask64 positive = _mm512_test_epi8_mask(codes, one_byte);
        const __mmask64 imag     = _mm512_test_epi8_mask(codes, two_byte);
        const __mmask64 real     = ~imag;

        const __m512i signed_xr = ggml_ifairy64_fuse_apply_sign_avx512(xr, positive);
        const __m512i signed_xi = ggml_ifairy64_fuse_apply_sign_avx512(xi, positive);

        acc[0] = _mm512_add_epi16(acc[0], _mm512_maddubs_epi16(madd_ones, _mm512_maskz_mov_epi8(real, signed_xr)));
        acc[1] = _mm512_add_epi16(acc[1], _mm512_maddubs_epi16(madd_ones, _mm512_maskz_mov_epi8(real, signed_xi)));
        acc[2] = _mm512_add_epi16(acc[2], _mm512_maddubs_epi16(madd_ones, _mm512_maskz_mov_epi8(imag, signed_xr)));
        acc[3] = _mm512_add_epi16(acc[3], _mm512_maddubs_epi16(madd_ones, _mm512_maskz_mov_epi8(imag, signed_xi)));
    }

    for (int channel = 0; channel < 4; ++channel) {
        sums[0][channel] = ggml_ifairy_hsum_i16x8_avx2(_mm512_castsi512_si128(acc[channel]));
        sums[1][channel] = ggml_ifairy_hsum_i16x8_avx2(_mm512_extracti32x4_epi32(acc[channel], 1));
        sums[2][channel] = ggml_ifairy_hsum_i16x8_avx2(_mm512_extracti32x4_epi32(acc[channel], 2));
        sums[3][channel] = ggml_ifairy_hsum_i16x8_avx2(_mm512_extracti32x4_epi32(acc[channel], 3));
    }
}
#endif

static inline void ggml_ifairy64_fuse_apply_branch(const block_ifairy64 *          w,
                                                    const block_ifairy64_q16 *      x,
                                                    const int32_t                   sums[4],
                                                    bool                            conjugate_input,
                                                    struct ggml_ifairy_complex_acc * acc) {
    const float xr = GGML_FP16_TO_FP32(x->d_real);
    const float xi = GGML_FP16_TO_FP32(x->d_imag);
    const float wr = GGML_FP16_TO_FP32(w->d_real);
    const float wi = GGML_FP16_TO_FP32(w->d_imag);

    if (conjugate_input) {
        acc->real += (float) sums[0] * (xr * wr) + (float) sums[3] * (xi * wi);
        acc->imag += (float) sums[2] * (xr * wi) - (float) sums[1] * (xi * wr);
    } else {
        acc->real += (float) sums[0] * (xr * wr) - (float) sums[3] * (xi * wi);
        acc->imag += (float) sums[2] * (xr * wi) + (float) sums[1] * (xi * wr);
    }
}

static inline void ggml_ifairy64_fuse_accumulate_four(const block_ifairy64 *          u0,
                                                       const block_ifairy64 *          u1,
                                                       const block_ifairy64 *          w0,
                                                       const block_ifairy64 *          w1,
                                                       const block_ifairy64_q16 *      x,
                                                       int64_t                         blocks,
                                                       struct ggml_ifairy_complex_acc * acc) {
    for (int64_t ib = 0; ib < blocks; ++ib) {
        int32_t sums[4][4] = {};

#if defined(GGML_USE_LEGACY_IFAIRY_CPU_AVX512) && defined(__AVX512F__) && defined(__AVX512BW__)
        ggml_ifairy64_fuse_accumulate_block_four_avx512(&u0[ib], &u1[ib], &w0[ib], &w1[ib], &x[ib], sums);
#elif defined(__AVX2__)
        ggml_ifairy64_fuse_accumulate_block_four_avx2(&u0[ib], &u1[ib], &w0[ib], &w1[ib], &x[ib], sums);
#else
        ggml_ifairy64_fuse_accumulate_block_scalar(&u0[ib], &x[ib], sums[0]);
        ggml_ifairy64_fuse_accumulate_block_scalar(&u1[ib], &x[ib], sums[1]);
        ggml_ifairy64_fuse_accumulate_block_scalar(&w0[ib], &x[ib], sums[2]);
        ggml_ifairy64_fuse_accumulate_block_scalar(&w1[ib], &x[ib], sums[3]);
#endif

        ggml_ifairy64_fuse_apply_branch(&u0[ib], &x[ib], sums[0], false, acc);
        ggml_ifairy64_fuse_apply_branch(&u1[ib], &x[ib], sums[1], false, acc);
        ggml_ifairy64_fuse_apply_branch(&w0[ib], &x[ib], sums[2], true, acc);
        ggml_ifairy64_fuse_apply_branch(&w1[ib], &x[ib], sums[3], true, acc);
    }
}

static inline float ggml_ifairy_wide_linear_bias_at(const struct ggml_tensor * bias,
                                                     int64_t                    i0,
                                                     int64_t                    i1,
                                                     int64_t                    i2,
                                                     int64_t                    i3) {
    const char * ptr = (const char *) bias->data + (i0 % bias->ne[0]) * bias->nb[0] +
                       (i1 % bias->ne[1]) * bias->nb[1] + (i2 % bias->ne[2]) * bias->nb[2] +
                       (i3 % bias->ne[3]) * bias->nb[3];
    return *(const float *) ptr;
}

static inline bool ggml_legacy_ifairy_wide_linear_weight_type(enum ggml_type type) {
    return type == GGML_TYPE_IFAIRY64;
}

void ggml_compute_forward_ifairy_wide_linear_w2(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * x      = dst->src[0];
    const struct ggml_tensor * u_s0   = dst->src[1];
    const struct ggml_tensor * u_s1   = dst->src[2];
    const struct ggml_tensor * w_s0   = dst->src[3];
    const struct ggml_tensor * w_s1   = dst->src[4];
    const struct ggml_tensor * bias   = dst->src[5];

    GGML_ASSERT(x && u_s0 && u_s1 && w_s0 && w_s1);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_legacy_ifairy_wide_linear_weight_type(u_s0->type));
    GGML_ASSERT(u_s0->type == u_s1->type && u_s0->type == w_s0->type && u_s0->type == w_s1->type);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(u_s0) && ggml_is_contiguous(u_s1));
    GGML_ASSERT(ggml_is_contiguous(w_s0) && ggml_is_contiguous(w_s1));
    GGML_ASSERT(u_s0->ne[2] == 1 && u_s0->ne[3] == 1);
    GGML_ASSERT(u_s1->ne[2] == 1 && u_s1->ne[3] == 1);
    GGML_ASSERT(w_s0->ne[2] == 1 && w_s0->ne[3] == 1);
    GGML_ASSERT(w_s1->ne[2] == 1 && w_s1->ne[3] == 1);
    GGML_ASSERT(!bias || bias->type == GGML_TYPE_F32);

    const int64_t K          = x->ne[0];
    const int64_t M          = dst->ne[0];
    const int64_t act_rows   = ggml_nrows(x);
    const int64_t blocks     = K / QK_IFAIRY64;
    const size_t  q_row_size = ggml_row_size(GGML_TYPE_IFAIRY64_Q16, K);
    const size_t  q_bytes    = GGML_PAD((size_t) act_rows * q_row_size, 64);

    GGML_ASSERT(K % QK_IFAIRY64 == 0);
    GGML_ASSERT(params->wdata && params->wsize >= q_bytes);

    block_ifairy64_q16 * q_x = (block_ifairy64_q16 *) params->wdata;
    for (int64_t ir = params->ith; ir < act_rows; ir += params->nth) {
        quantize_row_ifairy64_q16((const float *) ((const char *) x->data + ir * x->nb[1]),
                                  (char *) q_x + ir * q_row_size, K);
    }
    ggml_barrier(params->threadpool);

    const int64_t total = M * act_rows;
    const int64_t begin = (total * params->ith) / params->nth;
    const int64_t end   = (total * (params->ith + 1)) / params->nth;

    for (int64_t index = begin; index < end; ++index) {
        const int64_t row     = index % M;
        const int64_t act_row = index / M;
        const int64_t i1      = act_row % x->ne[1];
        const int64_t i2      = (act_row / x->ne[1]) % x->ne[2];
        const int64_t i3      = act_row / (x->ne[1] * x->ne[2]);

        const block_ifairy64 * u0_row = (const block_ifairy64 *) u_s0->data + row * blocks;
        const block_ifairy64 * u1_row = (const block_ifairy64 *) u_s1->data + row * blocks;
        const block_ifairy64 * w0_row = (const block_ifairy64 *) w_s0->data + row * blocks;
        const block_ifairy64 * w1_row = (const block_ifairy64 *) w_s1->data + row * blocks;
        const block_ifairy64_q16 * xq = (const block_ifairy64_q16 *) ((const char *) q_x + act_row * q_row_size);

        struct ggml_ifairy_complex_acc acc = { 0.0f, 0.0f };
        ggml_ifairy64_fuse_accumulate_four(u0_row, u1_row, w0_row, w1_row, xq, blocks, &acc);

        if (bias) {
            acc.real += ggml_ifairy_wide_linear_bias_at(bias, row, i1, i2, i3);
            acc.imag += ggml_ifairy_wide_linear_bias_at(bias, row + M, i1, i2, i3);
        }

        ggml_bf16_t * out = (ggml_bf16_t *) ((char *) dst->data + act_row * dst->nb[1] + row * dst->nb[0]);
        out[0]             = GGML_FP32_TO_BF16(acc.real);
        out[1]             = GGML_FP32_TO_BF16(acc.imag);
    }
}
