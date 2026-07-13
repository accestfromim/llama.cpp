#include "fairy2i-quants.h"

#include "ggml-cpu.h"

#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) && defined(__aarch64__)
#    include <arm_neon.h>

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

static inline int32_t ggml_fairy2i_tile64_hsum_i16x16_neon(int16x8_t lo, int16x8_t hi) {
    return vaddlvq_s16(lo) + vaddlvq_s16(hi);
}

static inline void ggml_fairy2i_tile64_fuse_accumulate_code_neon(int16x8_t acc[4][2],
                                                                 int8x16_t xr,
                                                                 int8x16_t xi,
                                                                 int8x16_t real_sign,
                                                                 int8x16_t imag_sign) {
    acc[0][0] = vmlal_s8(acc[0][0], vget_low_s8(xr), vget_low_s8(real_sign));
    acc[0][1] = vmlal_s8(acc[0][1], vget_high_s8(xr), vget_high_s8(real_sign));
    acc[1][0] = vmlal_s8(acc[1][0], vget_low_s8(xi), vget_low_s8(real_sign));
    acc[1][1] = vmlal_s8(acc[1][1], vget_high_s8(xi), vget_high_s8(real_sign));
    acc[2][0] = vmlal_s8(acc[2][0], vget_low_s8(xr), vget_low_s8(imag_sign));
    acc[2][1] = vmlal_s8(acc[2][1], vget_high_s8(xr), vget_high_s8(imag_sign));
    acc[3][0] = vmlal_s8(acc[3][0], vget_low_s8(xi), vget_low_s8(imag_sign));
    acc[3][1] = vmlal_s8(acc[3][1], vget_high_s8(xi), vget_high_s8(imag_sign));
}

static inline void ggml_fairy2i_tile64_store_accumulate_neon(int32_t sums[4], int16x8_t acc[4][2]) {
    for (int channel = 0; channel < 4; ++channel) {
        sums[channel] += ggml_fairy2i_tile64_hsum_i16x16_neon(acc[channel][0], acc[channel][1]);
    }
}

static inline void ggml_fairy2i_tile64_fuse_accumulate_pair_neon(const block_fairy2i_tile64_v2 *  w0,
                                                                 const block_fairy2i_tile64_v2 *  w1,
                                                                 const block_fairy2i_act_q16_64 * x,
                                                                 int32_t                          sums0[4],
                                                                 int32_t                          sums1[4]) {
    static const int8_t lut_real_data[16] = {
        -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    static const int8_t lut_imag_data[16] = {
        0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    const int8x16_t  lut_real = vld1q_s8(lut_real_data);
    const int8x16_t  lut_imag = vld1q_s8(lut_imag_data);
    const uint8x16_t packed0    = vld1q_u8(w0->qs);
    const uint8x16_t packed1    = vld1q_u8(w1->qs);
    const int16x8_t  zero       = vdupq_n_s16(0);
    int16x8_t        acc0[4][2] = {
        { zero, zero },
        { zero, zero },
        { zero, zero },
        { zero, zero },
    };
    int16x8_t acc1[4][2] = {
        { zero, zero },
        { zero, zero },
        { zero, zero },
        { zero, zero },
    };

    for (int part = 0; part < 4; ++part) {
        const int8x16_t  xr         = vld1q_s8((const int8_t *) x->x_real + part * 16);
        const int8x16_t  xi         = vld1q_s8((const int8_t *) x->x_imag + part * 16);
        const uint8x16_t codes0     = ggml_fairy2i_tile64_codes_part(packed0, part);
        const uint8x16_t codes1     = ggml_fairy2i_tile64_codes_part(packed1, part);
        const int8x16_t  real_sign0 = vqtbl1q_s8(lut_real, codes0);
        const int8x16_t  imag_sign0 = vqtbl1q_s8(lut_imag, codes0);
        const int8x16_t  real_sign1 = vqtbl1q_s8(lut_real, codes1);
        const int8x16_t  imag_sign1 = vqtbl1q_s8(lut_imag, codes1);

        ggml_fairy2i_tile64_fuse_accumulate_code_neon(acc0, xr, xi, real_sign0, imag_sign0);
        ggml_fairy2i_tile64_fuse_accumulate_code_neon(acc1, xr, xi, real_sign1, imag_sign1);
    }

    ggml_fairy2i_tile64_store_accumulate_neon(sums0, acc0);
    ggml_fairy2i_tile64_store_accumulate_neon(sums1, acc1);
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

void ggml_fairy2i_tile64_fuse_accumulate_block_two_dotprod(const block_fairy2i_tile64_v2 *  u0,
                                                           const block_fairy2i_tile64_v2 *  w0,
                                                           const block_fairy2i_act_q16_64 * x,
                                                           int32_t                          sums[2][4]) {
    ggml_fairy2i_tile64_fuse_accumulate_block_two_neon(u0, w0, x, sums);
}
#endif

#if !defined(GGML_USE_FAIRY2I_CPU_ARM_SVE2)
bool ggml_fairy2i_tile64_w2_arm_sve2_available(void) {
    return false;
}

void ggml_fairy2i_tile64_fuse_accumulate_block_four_sve2(const block_fairy2i_tile64_v2 *  u0,
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
    ggml_fairy2i_tile64_fuse_accumulate_pair_neon(u0, u1, x, sums[0], sums[1]);
    ggml_fairy2i_tile64_fuse_accumulate_pair_neon(w0, w1, x, sums[2], sums[3]);
#else
    (void) u0;
    (void) u1;
    (void) w0;
    (void) w1;
    (void) x;
    (void) sums;
#endif
}

void ggml_fairy2i_tile64_fuse_accumulate_block_two_neon(const block_fairy2i_tile64_v2 *  u0,
                                                        const block_fairy2i_tile64_v2 *  w0,
                                                        const block_fairy2i_act_q16_64 * x,
                                                        int32_t                          sums[2][4]) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    ggml_fairy2i_tile64_fuse_accumulate_pair_neon(u0, w0, x, sums[0], sums[1]);
#else
    (void) u0;
    (void) w0;
    (void) x;
    (void) sums;
#endif
}

static bool ggml_fairy2i_test_disable_arm_dotprod(void) {
    const char * env = getenv("GGML_FAIRY2I_TEST_DISABLE_ARM_DOTPROD");
    return env && strcmp(env, "0") != 0;
}

static bool ggml_fairy2i_test_disable_arm_sve2(void) {
    const char * env = getenv("GGML_FAIRY2I_TEST_DISABLE_ARM_SVE2");
    return env && strcmp(env, "0") != 0;
}

static bool ggml_fairy2i_test_require_arm_sve2(void) {
    const char * env = getenv("GGML_FAIRY2I_TEST_REQUIRE_ARM_SVE2");
    return env && strcmp(env, "0") != 0;
}

static bool ggml_fairy2i_tile64_w2_arm_sve2_enabled(void) {
    return !ggml_fairy2i_test_disable_arm_sve2() && ggml_fairy2i_tile64_w2_arm_sve2_available() &&
           ggml_cpu_has_sve2();
}

static void ggml_fairy2i_tile64_w2_arm_require_sve2(void) {
    if (ggml_fairy2i_test_require_arm_sve2() && !ggml_fairy2i_tile64_w2_arm_sve2_enabled()) {
        GGML_ABORT("GGML_FAIRY2I_TEST_REQUIRE_ARM_SVE2 is set, but the Fairy2i direct SVE2 path is unavailable "
                   "(disabled=%d, kernel=%d, cpu=%d)",
                   ggml_fairy2i_test_disable_arm_sve2() ? 1 : 0,
                   ggml_fairy2i_tile64_w2_arm_sve2_available() ? 1 : 0,
                   ggml_cpu_has_sve2() ? 1 : 0);
    }
}

const char * ggml_fairy2i_tile64_w2_arm_path_name(void) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    ggml_fairy2i_tile64_w2_arm_require_sve2();

    if (ggml_fairy2i_tile64_w2_arm_sve2_enabled()) {
        return "direct_sve2";
    }

    if (!ggml_fairy2i_test_disable_arm_dotprod() && ggml_fairy2i_tile64_w2_arm_dotprod_available() &&
        ggml_cpu_has_dotprod()) {
        return "direct_dotprod";
    }

    if (ggml_fairy2i_tile64_w2_arm_neon_available()) {
        return "direct_neon";
    }
#endif

    return "direct_scalar";
}

bool ggml_fairy2i_tile64_fuse_accumulate_block_four_arm(const block_fairy2i_tile64_v2 *  u0,
                                                        const block_fairy2i_tile64_v2 *  u1,
                                                        const block_fairy2i_tile64_v2 *  w0,
                                                        const block_fairy2i_tile64_v2 *  w1,
                                                        const block_fairy2i_act_q16_64 * x,
                                                        int32_t                          sums[4][4]) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    ggml_fairy2i_tile64_w2_arm_require_sve2();

    if (ggml_fairy2i_tile64_w2_arm_sve2_enabled()) {
        ggml_fairy2i_tile64_fuse_accumulate_block_four_sve2(u0, u1, w0, w1, x, sums);
        return true;
    }

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

bool ggml_fairy2i_tile64_fuse_accumulate_block_two_arm(const block_fairy2i_tile64_v2 *  u0,
                                                       const block_fairy2i_tile64_v2 *  w0,
                                                       const block_fairy2i_act_q16_64 * x,
                                                       int32_t                          sums[2][4]) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    if (!ggml_fairy2i_test_disable_arm_dotprod() && ggml_fairy2i_tile64_w2_arm_dotprod_available() &&
        ggml_cpu_has_dotprod()) {
        ggml_fairy2i_tile64_fuse_accumulate_block_two_dotprod(u0, w0, x, sums);
        return true;
    }

    if (ggml_fairy2i_tile64_w2_arm_neon_available()) {
        ggml_fairy2i_tile64_fuse_accumulate_block_two_neon(u0, w0, x, sums);
        return true;
    }
#else
    (void) u0;
    (void) w0;
    (void) x;
    (void) sums;
#endif

    return false;
}
