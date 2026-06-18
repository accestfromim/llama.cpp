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
static inline int32_t ggml_fairy2i_tile64_dot_i8x16_dotprod(int8x16_t a, int8x16_t b) {
    const int32x4_t acc = vdotq_s32(vdupq_n_s32(0), a, b);
    return vaddvq_s32(acc);
}

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

static inline void ggml_fairy2i_tile64_fuse_accumulate_one_dotprod(const block_fairy2i_tile64_v2 *  w,
                                                                   const block_fairy2i_act_q16_64 * x,
                                                                   int32_t                          sums[4]) {
    static const int8_t lut_real_data[16] = {
        -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    static const int8_t lut_imag_data[16] = {
        0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    const int8x16_t  lut_real = vld1q_s8(lut_real_data);
    const int8x16_t  lut_imag = vld1q_s8(lut_imag_data);
    const uint8x16_t packed   = vld1q_u8(w->qs);

    for (int part = 0; part < 4; ++part) {
        const uint8x16_t codes     = ggml_fairy2i_tile64_codes_part_dotprod(packed, part);
        const int8x16_t  real_sign = vqtbl1q_s8(lut_real, codes);
        const int8x16_t  imag_sign = vqtbl1q_s8(lut_imag, codes);
        const int8x16_t  xr        = vld1q_s8((const int8_t *) x->x_real + part * 16);
        const int8x16_t  xi        = vld1q_s8((const int8_t *) x->x_imag + part * 16);

        sums[0] += ggml_fairy2i_tile64_dot_i8x16_dotprod(xr, real_sign);
        sums[1] += ggml_fairy2i_tile64_dot_i8x16_dotprod(xi, real_sign);
        sums[2] += ggml_fairy2i_tile64_dot_i8x16_dotprod(xr, imag_sign);
        sums[3] += ggml_fairy2i_tile64_dot_i8x16_dotprod(xi, imag_sign);
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
    ggml_fairy2i_tile64_fuse_accumulate_one_dotprod(u0, x, sums[0]);
    ggml_fairy2i_tile64_fuse_accumulate_one_dotprod(u1, x, sums[1]);
    ggml_fairy2i_tile64_fuse_accumulate_one_dotprod(w0, x, sums[2]);
    ggml_fairy2i_tile64_fuse_accumulate_one_dotprod(w1, x, sums[3]);
#else
    ggml_fairy2i_tile64_fuse_accumulate_block_four_neon(u0, u1, w0, w1, x, sums);
#endif
}
