#include "fairy2i-quants.h"

#if defined(__ARM_NEON) && defined(__aarch64__)
#    include <arm_neon.h>
#endif

bool ggml_fairy2i_tile64_w2_arm_dotprod_available(void) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && defined(__aarch64__)
    return true;
#else
    return false;
#endif
}

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && defined(__aarch64__)
static inline uint8x16_t ggml_fairy2i_tile64_codes_part_dotprod(uint8x16_t packed, int part) {
    switch (part) {
        case 0:
            return vandq_u8(packed, vdupq_n_u8(0x03));
        case 1:
            return vandq_u8(vshrq_n_u8(packed, 2), vdupq_n_u8(0x03));
        case 2:
            return vandq_u8(vshrq_n_u8(packed, 4), vdupq_n_u8(0x03));
        case 3:
            return vshrq_n_u8(packed, 6);
        default:
            return vdupq_n_u8(0);
    }
}

static inline void ggml_fairy2i_tile64_fuse_accumulate_code_dotprod(int32x4_t acc[4],
                                                                    int8x16_t xr,
                                                                    int8x16_t xi,
                                                                    int8x16_t real_sign,
                                                                    int8x16_t imag_sign) {
    acc[0] = vdotq_s32(acc[0], xr, real_sign);
    acc[1] = vdotq_s32(acc[1], xi, real_sign);
    acc[2] = vdotq_s32(acc[2], xr, imag_sign);
    acc[3] = vdotq_s32(acc[3], xi, imag_sign);
}

static inline void ggml_fairy2i_tile64_store_accumulate_dotprod(int32_t sums[4], int32x4_t acc[4]) {
    for (int channel = 0; channel < 4; ++channel) {
        sums[channel] += vaddvq_s32(acc[channel]);
    }
}

static inline void ggml_fairy2i_tile64_fuse_accumulate_four_dotprod(const block_fairy2i_tile64_v2 *  u0,
                                                                    const block_fairy2i_tile64_v2 *  u1,
                                                                    const block_fairy2i_tile64_v2 *  w0,
                                                                    const block_fairy2i_tile64_v2 *  w1,
                                                                    const block_fairy2i_act_q16_64 * x,
                                                                    int32_t                          sums[4][4]) {
    static const int8_t lut_real_data[16] = {
        -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    static const int8_t lut_imag_data[16] = {
        0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    const int8x16_t  lut_real = vld1q_s8(lut_real_data);
    const int8x16_t  lut_imag = vld1q_s8(lut_imag_data);
    const uint8x16_t packed[4] = {
        vld1q_u8(u0->qs),
        vld1q_u8(u1->qs),
        vld1q_u8(w0->qs),
        vld1q_u8(w1->qs),
    };
    const int32x4_t zero      = vdupq_n_s32(0);
    int32x4_t       acc[4][4] = {
        { zero, zero, zero, zero },
        { zero, zero, zero, zero },
        { zero, zero, zero, zero },
        { zero, zero, zero, zero },
    };

    for (int part = 0; part < 4; ++part) {
        const int8x16_t xr = vld1q_s8((const int8_t *) x->x_real + part * 16);
        const int8x16_t xi = vld1q_s8((const int8_t *) x->x_imag + part * 16);

        for (int branch = 0; branch < 4; ++branch) {
            const uint8x16_t codes     = ggml_fairy2i_tile64_codes_part_dotprod(packed[branch], part);
            const int8x16_t  real_sign = vqtbl1q_s8(lut_real, codes);
            const int8x16_t  imag_sign = vqtbl1q_s8(lut_imag, codes);
            ggml_fairy2i_tile64_fuse_accumulate_code_dotprod(acc[branch], xr, xi, real_sign, imag_sign);
        }
    }

    for (int branch = 0; branch < 4; ++branch) {
        ggml_fairy2i_tile64_store_accumulate_dotprod(sums[branch], acc[branch]);
    }
}

static inline void ggml_fairy2i_tile64_fuse_accumulate_two_dotprod(const block_fairy2i_tile64_v2 *  u0,
                                                                   const block_fairy2i_tile64_v2 *  w0,
                                                                   const block_fairy2i_act_q16_64 * x,
                                                                   int32_t                          sums[2][4]) {
    static const int8_t lut_real_data[16] = {
        -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    static const int8_t lut_imag_data[16] = {
        0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    const int8x16_t  lut_real = vld1q_s8(lut_real_data);
    const int8x16_t  lut_imag = vld1q_s8(lut_imag_data);
    const uint8x16_t packed[2] = {
        vld1q_u8(u0->qs),
        vld1q_u8(w0->qs),
    };
    const int32x4_t zero      = vdupq_n_s32(0);
    int32x4_t       acc[2][4] = {
        { zero, zero, zero, zero },
        { zero, zero, zero, zero },
    };

    for (int part = 0; part < 4; ++part) {
        const int8x16_t xr = vld1q_s8((const int8_t *) x->x_real + part * 16);
        const int8x16_t xi = vld1q_s8((const int8_t *) x->x_imag + part * 16);

        for (int branch = 0; branch < 2; ++branch) {
            const uint8x16_t codes     = ggml_fairy2i_tile64_codes_part_dotprod(packed[branch], part);
            const int8x16_t  real_sign = vqtbl1q_s8(lut_real, codes);
            const int8x16_t  imag_sign = vqtbl1q_s8(lut_imag, codes);
            ggml_fairy2i_tile64_fuse_accumulate_code_dotprod(acc[branch], xr, xi, real_sign, imag_sign);
        }
    }

    for (int branch = 0; branch < 2; ++branch) {
        ggml_fairy2i_tile64_store_accumulate_dotprod(sums[branch], acc[branch]);
    }
}
#endif

void ggml_fairy2i_tile64_fuse_accumulate_block_four_dotprod(const block_fairy2i_tile64_v2 *  u0,
                                                            const block_fairy2i_tile64_v2 *  u1,
                                                            const block_fairy2i_tile64_v2 *  w0,
                                                            const block_fairy2i_tile64_v2 *  w1,
                                                            const block_fairy2i_act_q16_64 * x,
                                                            int32_t                          sums[4][4]) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && defined(__aarch64__)
    ggml_fairy2i_tile64_fuse_accumulate_four_dotprod(u0, u1, w0, w1, x, sums);
#else
    ggml_fairy2i_tile64_fuse_accumulate_block_four_neon(u0, u1, w0, w1, x, sums);
#endif
}

void ggml_fairy2i_tile64_fuse_accumulate_block_two_dotprod(const block_fairy2i_tile64_v2 *  u0,
                                                           const block_fairy2i_tile64_v2 *  w0,
                                                           const block_fairy2i_act_q16_64 * x,
                                                           int32_t                          sums[2][4]) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && defined(__aarch64__)
    ggml_fairy2i_tile64_fuse_accumulate_two_dotprod(u0, w0, x, sums);
#else
    ggml_fairy2i_tile64_fuse_accumulate_block_two_neon(u0, w0, x, sums);
#endif
}
