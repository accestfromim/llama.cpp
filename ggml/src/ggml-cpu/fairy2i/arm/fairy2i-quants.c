#include "fairy2i-quants.h"

#include "ggml-cpu.h"

#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) && defined(__aarch64__)
#    include <arm_neon.h>

static inline int32_t ggml_fairy2i_tile64_dot_i8x16_neon(int8x16_t a, int8x16_t b) {
    const int16x8_t prod_lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    const int16x8_t prod_hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));

    int32x4_t sum = vpaddlq_s16(prod_lo);
    sum           = vaddq_s32(sum, vpaddlq_s16(prod_hi));
    return vaddvq_s32(sum);
}

static inline uint8x16_t ggml_fairy2i_tile64_codes_part(uint8x16_t packed, int part) {
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

static inline void ggml_fairy2i_tile64_fuse_accumulate_one_neon(const block_fairy2i_tile64_v2 *  w,
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
        const uint8x16_t codes     = ggml_fairy2i_tile64_codes_part(packed, part);
        const int8x16_t  real_sign = vqtbl1q_s8(lut_real, codes);
        const int8x16_t  imag_sign = vqtbl1q_s8(lut_imag, codes);
        const int8x16_t  xr        = vld1q_s8((const int8_t *) x->x_real + part * 16);
        const int8x16_t  xi        = vld1q_s8((const int8_t *) x->x_imag + part * 16);

        sums[0] += ggml_fairy2i_tile64_dot_i8x16_neon(xr, real_sign);
        sums[1] += ggml_fairy2i_tile64_dot_i8x16_neon(xi, real_sign);
        sums[2] += ggml_fairy2i_tile64_dot_i8x16_neon(xr, imag_sign);
        sums[3] += ggml_fairy2i_tile64_dot_i8x16_neon(xi, imag_sign);
    }
}
#endif

bool ggml_fairy2i_tile64_w2_arm_neon_available(void) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    return ggml_cpu_has_neon() != 0;
#else
    return false;
#endif
}

#if !defined(GGML_USE_FAIRY2I_CPU_ARM_DOTPROD)
bool ggml_fairy2i_tile64_w2_arm_dotprod_available(void) {
    return false;
}

void ggml_fairy2i_tile64_fuse_accumulate_block_four_dotprod(const block_fairy2i_tile64_v2 *  u0,
                                                            const block_fairy2i_tile64_v2 *  u1,
                                                            const block_fairy2i_tile64_v2 *  w0,
                                                            const block_fairy2i_tile64_v2 *  w1,
                                                            const block_fairy2i_act_q16_64 * x,
                                                            int32_t                          sums[4][4]) {
    ggml_fairy2i_tile64_fuse_accumulate_block_four_neon(u0, u1, w0, w1, x, sums);
}
#endif

void ggml_fairy2i_tile64_fuse_accumulate_block_four_neon(const block_fairy2i_tile64_v2 *  u0,
                                                         const block_fairy2i_tile64_v2 *  u1,
                                                         const block_fairy2i_tile64_v2 *  w0,
                                                         const block_fairy2i_tile64_v2 *  w1,
                                                         const block_fairy2i_act_q16_64 * x,
                                                         int32_t                          sums[4][4]) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    ggml_fairy2i_tile64_fuse_accumulate_one_neon(u0, x, sums[0]);
    ggml_fairy2i_tile64_fuse_accumulate_one_neon(u1, x, sums[1]);
    ggml_fairy2i_tile64_fuse_accumulate_one_neon(w0, x, sums[2]);
    ggml_fairy2i_tile64_fuse_accumulate_one_neon(w1, x, sums[3]);
#else
    (void) u0;
    (void) u1;
    (void) w0;
    (void) w1;
    (void) x;
    (void) sums;
#endif
}

static bool ggml_fairy2i_test_disable_arm_dotprod(void) {
    const char * env = getenv("GGML_FAIRY2I_TEST_DISABLE_ARM_DOTPROD");
    return env && strcmp(env, "0") != 0;
}

bool ggml_fairy2i_tile64_fuse_accumulate_block_four_arm(const block_fairy2i_tile64_v2 *  u0,
                                                        const block_fairy2i_tile64_v2 *  u1,
                                                        const block_fairy2i_tile64_v2 *  w0,
                                                        const block_fairy2i_tile64_v2 *  w1,
                                                        const block_fairy2i_act_q16_64 * x,
                                                        int32_t                          sums[4][4]) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    if (!ggml_fairy2i_test_disable_arm_dotprod() && ggml_fairy2i_tile64_w2_arm_dotprod_available() &&
        ggml_cpu_has_dotprod()) {
        ggml_fairy2i_tile64_fuse_accumulate_block_four_dotprod(u0, u1, w0, w1, x, sums);
        return true;
    }

    if (ggml_fairy2i_tile64_w2_arm_neon_available()) {
        ggml_fairy2i_tile64_fuse_accumulate_block_four_neon(u0, u1, w0, w1, x, sums);
        return true;
    }
#else
    (void) u0;
    (void) u1;
    (void) w0;
    (void) w1;
    (void) x;
    (void) sums;
#endif

    return false;
}
