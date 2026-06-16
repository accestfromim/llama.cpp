#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "simd-mappings.h"

#include "../../ggml-cpu-impl.h"
#include "../../quants.h"
#include "quants-ifairy.h"

bool ggml_vec_dot_ifairy_q16_K_dotprod_available(void) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && defined(__aarch64__)
    return true;
#else
    return false;
#endif
}

void ggml_vec_dot_ifairy_q16_K_dotprod(int                        n,
                                       float * GGML_RESTRICT      s,
                                       size_t                     bs,
                                       const void * GGML_RESTRICT vx,
                                       size_t                     bx,
                                       const void * GGML_RESTRICT vy,
                                       size_t                     by,
                                       int                        nrc) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && defined(__aarch64__)
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);

    const block_ifairy * GGML_RESTRICT     w  = vx;
    const block_ifairy_q16 * GGML_RESTRICT x  = vy;
    const int                              nb = n / QK_IFAIRY;

    float sum_real_total = 0.0f;
    float sum_imag_total = 0.0f;

    const bool act_tensor = ggml_ifairy_vecdot_act_tensor_enabled();

    const uint8x16_t v_mask_0f = vdupq_n_u8(0x0F);
    const int32x4_t  vzero     = vdupq_n_s32(0);

    static const int8_t lut_real_data[16] = {
        -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0,
    };
    static const int8_t lut_imag_data[16] = {
        0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1,
    };
    static const int8_t lut_wr_idx1_data[16] = {
        -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    static const int8_t lut_wi_idx1_data[16] = {
        0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1,
    };

    const int8x16_t v_lut_wr_i0 = vld1q_s8(lut_real_data);
    const int8x16_t v_lut_wi_i0 = vld1q_s8(lut_imag_data);
    const int8x16_t v_lut_wr_i1 = vld1q_s8(lut_wr_idx1_data);
    const int8x16_t v_lut_wi_i1 = vld1q_s8(lut_wi_idx1_data);

    register uint8x16_t v_mask_0f_reg __asm__("v27")   = v_mask_0f;
    register int8x16_t  v_lut_wr_i0_reg __asm__("v28") = v_lut_wr_i0;
    register int8x16_t  v_lut_wi_i0_reg __asm__("v29") = v_lut_wi_i0;
    register int8x16_t  v_lut_wr_i1_reg __asm__("v30") = v_lut_wr_i1;
    register int8x16_t  v_lut_wi_i1_reg __asm__("v31") = v_lut_wi_i1;

    if (act_tensor) {
        int32_t sum_ac_total = 0;
        int32_t sum_ad_total = 0;
        int32_t sum_bc_total = 0;
        int32_t sum_bd_total = 0;

        for (int i = 0; i < nb; ++i) {
            __builtin_prefetch(w + i + 1, 0, 1);
            __builtin_prefetch(x + i + 1, 0, 1);

            register int32x4_t acc_ac0 __asm__("v20") = vzero;
            register int32x4_t acc_ad0 __asm__("v21") = vzero;
            register int32x4_t acc_bc0 __asm__("v22") = vzero;
            register int32x4_t acc_bd0 __asm__("v23") = vzero;
            register int32x4_t acc_ac1 __asm__("v24") = vzero;
            register int32x4_t acc_ad1 __asm__("v25") = vzero;
            register int32x4_t acc_bc1 __asm__("v26") = vzero;
            register int32x4_t acc_bd1 __asm__("v7")  = vzero;

            const uint8_t * GGML_RESTRICT w_ptr   = w[i].qs;
            const uint8_t * GGML_RESTRICT x_r_ptr = x[i].x_real;
            const uint8_t * GGML_RESTRICT x_i_ptr = x[i].x_imag;

            for (int j = 0; j < QK_IFAIRY; j += 128) {
                const uint8_t * GGML_RESTRICT w_iter = w_ptr + (j >> 2);
                const uint8_t * GGML_RESTRICT xr     = x_r_ptr + j;
                const uint8_t * GGML_RESTRICT xi     = x_i_ptr + j;

                __asm__ volatile(
                    "ldr            q0, [%[w]]                  \n"
                    "and            v1.16b, v0.16b, %[m0f].16b  \n"
                    "ushr           v2.16b, v0.16b, #4          \n"
                    "ldp            q3,  q4,  [%[xr]]           \n"
                    "ldp            q5,  q6,  [%[xi]]           \n"
                    "tbl            v16.16b, {%[wr0].16b}, v1.16b \n"
                    "tbl            v17.16b, {%[wi0].16b}, v1.16b \n"
                    "tbl            v18.16b, {%[wr1].16b}, v1.16b \n"
                    "tbl            v19.16b, {%[wi1].16b}, v1.16b \n"
                    "sdot           %[ac0].4s, v3.16b,  v16.16b \n"
                    "sdot           %[ad0].4s, v5.16b,  v16.16b \n"
                    "sdot           %[bc0].4s, v3.16b,  v17.16b \n"
                    "sdot           %[bd0].4s, v5.16b,  v17.16b \n"
                    "sdot           %[ac1].4s, v4.16b,  v18.16b \n"
                    "sdot           %[ad1].4s, v6.16b,  v18.16b \n"
                    "sdot           %[bc1].4s, v4.16b,  v19.16b \n"
                    "sdot           %[bd1].4s, v6.16b,  v19.16b \n"
                    "ldp            q3,  q4,  [%[xr], #32]      \n"
                    "ldp            q5,  q6,  [%[xi], #32]      \n"
                    "tbl            v16.16b, {%[wr0].16b}, v2.16b \n"
                    "tbl            v17.16b, {%[wi0].16b}, v2.16b \n"
                    "tbl            v18.16b, {%[wr1].16b}, v2.16b \n"
                    "tbl            v19.16b, {%[wi1].16b}, v2.16b \n"
                    "sdot           %[ac0].4s, v3.16b,  v16.16b \n"
                    "sdot           %[ad0].4s, v5.16b,  v16.16b \n"
                    "sdot           %[bc0].4s, v3.16b,  v17.16b \n"
                    "sdot           %[bd0].4s, v5.16b,  v17.16b \n"
                    "sdot           %[ac1].4s, v4.16b,  v18.16b \n"
                    "sdot           %[ad1].4s, v6.16b,  v18.16b \n"
                    "sdot           %[bc1].4s, v4.16b,  v19.16b \n"
                    "sdot           %[bd1].4s, v6.16b,  v19.16b \n"
                    "ldr            q0, [%[w],  #16]            \n"
                    "and            v1.16b, v0.16b, %[m0f].16b  \n"
                    "ushr           v2.16b, v0.16b, #4          \n"
                    "ldp            q3,  q4,  [%[xr], #64]      \n"
                    "ldp            q5,  q6,  [%[xi], #64]      \n"
                    "tbl            v16.16b, {%[wr0].16b}, v1.16b \n"
                    "tbl            v17.16b, {%[wi0].16b}, v1.16b \n"
                    "tbl            v18.16b, {%[wr1].16b}, v1.16b \n"
                    "tbl            v19.16b, {%[wi1].16b}, v1.16b \n"
                    "sdot           %[ac0].4s, v3.16b,  v16.16b \n"
                    "sdot           %[ad0].4s, v5.16b,  v16.16b \n"
                    "sdot           %[bc0].4s, v3.16b,  v17.16b \n"
                    "sdot           %[bd0].4s, v5.16b,  v17.16b \n"
                    "sdot           %[ac1].4s, v4.16b,  v18.16b \n"
                    "sdot           %[ad1].4s, v6.16b,  v18.16b \n"
                    "sdot           %[bc1].4s, v4.16b,  v19.16b \n"
                    "sdot           %[bd1].4s, v6.16b,  v19.16b \n"
                    "ldp            q3,  q4,  [%[xr], #96]      \n"
                    "ldp            q5,  q6,  [%[xi], #96]      \n"
                    "tbl            v16.16b, {%[wr0].16b}, v2.16b \n"
                    "tbl            v17.16b, {%[wi0].16b}, v2.16b \n"
                    "tbl            v18.16b, {%[wr1].16b}, v2.16b \n"
                    "tbl            v19.16b, {%[wi1].16b}, v2.16b \n"
                    "sdot           %[ac0].4s, v3.16b,  v16.16b \n"
                    "sdot           %[ad0].4s, v5.16b,  v16.16b \n"
                    "sdot           %[bc0].4s, v3.16b,  v17.16b \n"
                    "sdot           %[bd0].4s, v5.16b,  v17.16b \n"
                    "sdot           %[ac1].4s, v4.16b,  v18.16b \n"
                    "sdot           %[ad1].4s, v6.16b,  v18.16b \n"
                    "sdot           %[bc1].4s, v4.16b,  v19.16b \n"
                    "sdot           %[bd1].4s, v6.16b,  v19.16b \n"
                    : [ac0] "+w"(acc_ac0), [ad0] "+w"(acc_ad0), [bc0] "+w"(acc_bc0), [bd0] "+w"(acc_bd0),
                      [ac1] "+w"(acc_ac1), [ad1] "+w"(acc_ad1), [bc1] "+w"(acc_bc1), [bd1] "+w"(acc_bd1)
                    : [w] "r"(w_iter), [xr] "r"(xr), [xi] "r"(xi), [m0f] "w"(v_mask_0f_reg), [wr0] "w"(v_lut_wr_i0_reg),
                      [wi0] "w"(v_lut_wi_i0_reg), [wr1] "w"(v_lut_wr_i1_reg), [wi1] "w"(v_lut_wi_i1_reg)
                    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v16", "v17", "v18", "v19", "memory");
            }

            acc_ac0 = vaddq_s32(acc_ac0, acc_ac1);
            acc_ad0 = vaddq_s32(acc_ad0, acc_ad1);
            acc_bc0 = vaddq_s32(acc_bc0, acc_bc1);
            acc_bd0 = vaddq_s32(acc_bd0, acc_bd1);

            sum_ac_total += vaddvq_s32(acc_ac0);
            sum_ad_total += vaddvq_s32(acc_ad0);
            sum_bc_total += vaddvq_s32(acc_bc0);
            sum_bd_total += vaddvq_s32(acc_bd0);
        }

        const float coeff_w_real = GGML_CPU_FP16_TO_FP32(w[0].d_real);
        const float coeff_w_imag = GGML_CPU_FP16_TO_FP32(w[0].d_imag);
        const float x_real       = GGML_CPU_FP16_TO_FP32(x[0].d_real);
        const float x_imag       = GGML_CPU_FP16_TO_FP32(x[0].d_imag);

        sum_real_total = coeff_w_real * (x_real * (float) sum_ac_total) + coeff_w_imag * (x_imag * (float) sum_bd_total);
        sum_imag_total = coeff_w_imag * (x_real * (float) sum_bc_total) - coeff_w_real * (x_imag * (float) sum_ad_total);
    } else {
        float acc_ac_xr = 0.0f;
        float acc_bd_xi = 0.0f;
        float acc_bc_xr = 0.0f;
        float acc_ad_xi = 0.0f;

        for (int i = 0; i < nb; ++i) {
            __builtin_prefetch(w + i + 1, 0, 1);
            __builtin_prefetch(x + i + 1, 0, 1);

            register int32x4_t acc_ac0 __asm__("v20") = vzero;
            register int32x4_t acc_ad0 __asm__("v21") = vzero;
            register int32x4_t acc_bc0 __asm__("v22") = vzero;
            register int32x4_t acc_bd0 __asm__("v23") = vzero;
            register int32x4_t acc_ac1 __asm__("v24") = vzero;
            register int32x4_t acc_ad1 __asm__("v25") = vzero;
            register int32x4_t acc_bc1 __asm__("v26") = vzero;
            register int32x4_t acc_bd1 __asm__("v7")  = vzero;

            const uint8_t * GGML_RESTRICT w_ptr   = w[i].qs;
            const uint8_t * GGML_RESTRICT x_r_ptr = x[i].x_real;
            const uint8_t * GGML_RESTRICT x_i_ptr = x[i].x_imag;

            for (int j = 0; j < QK_IFAIRY; j += 128) {
                const uint8_t * GGML_RESTRICT w_iter = w_ptr + (j >> 2);
                const uint8_t * GGML_RESTRICT xr     = x_r_ptr + j;
                const uint8_t * GGML_RESTRICT xi     = x_i_ptr + j;

                __asm__ volatile(
                    "ldr            q0, [%[w]]                  \n"
                    "and            v1.16b, v0.16b, %[m0f].16b  \n"
                    "ushr           v2.16b, v0.16b, #4          \n"
                    "ldp            q3,  q4,  [%[xr]]           \n"
                    "ldp            q5,  q6,  [%[xi]]           \n"
                    "tbl            v16.16b, {%[wr0].16b}, v1.16b \n"
                    "tbl            v17.16b, {%[wi0].16b}, v1.16b \n"
                    "tbl            v18.16b, {%[wr1].16b}, v1.16b \n"
                    "tbl            v19.16b, {%[wi1].16b}, v1.16b \n"
                    "sdot           %[ac0].4s, v3.16b,  v16.16b \n"
                    "sdot           %[ad0].4s, v5.16b,  v16.16b \n"
                    "sdot           %[bc0].4s, v3.16b,  v17.16b \n"
                    "sdot           %[bd0].4s, v5.16b,  v17.16b \n"
                    "sdot           %[ac1].4s, v4.16b,  v18.16b \n"
                    "sdot           %[ad1].4s, v6.16b,  v18.16b \n"
                    "sdot           %[bc1].4s, v4.16b,  v19.16b \n"
                    "sdot           %[bd1].4s, v6.16b,  v19.16b \n"
                    "ldp            q3,  q4,  [%[xr], #32]      \n"
                    "ldp            q5,  q6,  [%[xi], #32]      \n"
                    "tbl            v16.16b, {%[wr0].16b}, v2.16b \n"
                    "tbl            v17.16b, {%[wi0].16b}, v2.16b \n"
                    "tbl            v18.16b, {%[wr1].16b}, v2.16b \n"
                    "tbl            v19.16b, {%[wi1].16b}, v2.16b \n"
                    "sdot           %[ac0].4s, v3.16b,  v16.16b \n"
                    "sdot           %[ad0].4s, v5.16b,  v16.16b \n"
                    "sdot           %[bc0].4s, v3.16b,  v17.16b \n"
                    "sdot           %[bd0].4s, v5.16b,  v17.16b \n"
                    "sdot           %[ac1].4s, v4.16b,  v18.16b \n"
                    "sdot           %[ad1].4s, v6.16b,  v18.16b \n"
                    "sdot           %[bc1].4s, v4.16b,  v19.16b \n"
                    "sdot           %[bd1].4s, v6.16b,  v19.16b \n"
                    "ldr            q0, [%[w],  #16]            \n"
                    "and            v1.16b, v0.16b, %[m0f].16b  \n"
                    "ushr           v2.16b, v0.16b, #4          \n"
                    "ldp            q3,  q4,  [%[xr], #64]      \n"
                    "ldp            q5,  q6,  [%[xi], #64]      \n"
                    "tbl            v16.16b, {%[wr0].16b}, v1.16b \n"
                    "tbl            v17.16b, {%[wi0].16b}, v1.16b \n"
                    "tbl            v18.16b, {%[wr1].16b}, v1.16b \n"
                    "tbl            v19.16b, {%[wi1].16b}, v1.16b \n"
                    "sdot           %[ac0].4s, v3.16b,  v16.16b \n"
                    "sdot           %[ad0].4s, v5.16b,  v16.16b \n"
                    "sdot           %[bc0].4s, v3.16b,  v17.16b \n"
                    "sdot           %[bd0].4s, v5.16b,  v17.16b \n"
                    "sdot           %[ac1].4s, v4.16b,  v18.16b \n"
                    "sdot           %[ad1].4s, v6.16b,  v18.16b \n"
                    "sdot           %[bc1].4s, v4.16b,  v19.16b \n"
                    "sdot           %[bd1].4s, v6.16b,  v19.16b \n"
                    "ldp            q3,  q4,  [%[xr], #96]      \n"
                    "ldp            q5,  q6,  [%[xi], #96]      \n"
                    "tbl            v16.16b, {%[wr0].16b}, v2.16b \n"
                    "tbl            v17.16b, {%[wi0].16b}, v2.16b \n"
                    "tbl            v18.16b, {%[wr1].16b}, v2.16b \n"
                    "tbl            v19.16b, {%[wi1].16b}, v2.16b \n"
                    "sdot           %[ac0].4s, v3.16b,  v16.16b \n"
                    "sdot           %[ad0].4s, v5.16b,  v16.16b \n"
                    "sdot           %[bc0].4s, v3.16b,  v17.16b \n"
                    "sdot           %[bd0].4s, v5.16b,  v17.16b \n"
                    "sdot           %[ac1].4s, v4.16b,  v18.16b \n"
                    "sdot           %[ad1].4s, v6.16b,  v18.16b \n"
                    "sdot           %[bc1].4s, v4.16b,  v19.16b \n"
                    "sdot           %[bd1].4s, v6.16b,  v19.16b \n"
                    : [ac0] "+w"(acc_ac0), [ad0] "+w"(acc_ad0), [bc0] "+w"(acc_bc0), [bd0] "+w"(acc_bd0),
                      [ac1] "+w"(acc_ac1), [ad1] "+w"(acc_ad1), [bc1] "+w"(acc_bc1), [bd1] "+w"(acc_bd1)
                    : [w] "r"(w_iter), [xr] "r"(xr), [xi] "r"(xi), [m0f] "w"(v_mask_0f_reg), [wr0] "w"(v_lut_wr_i0_reg),
                      [wi0] "w"(v_lut_wi_i0_reg), [wr1] "w"(v_lut_wr_i1_reg), [wi1] "w"(v_lut_wi_i1_reg)
                    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v16", "v17", "v18", "v19", "memory");
            }

            acc_ac0 = vaddq_s32(acc_ac0, acc_ac1);
            acc_ad0 = vaddq_s32(acc_ad0, acc_ad1);
            acc_bc0 = vaddq_s32(acc_bc0, acc_bc1);
            acc_bd0 = vaddq_s32(acc_bd0, acc_bd1);

            const float x_real = GGML_CPU_FP16_TO_FP32(x[i].d_real);
            const float x_imag = GGML_CPU_FP16_TO_FP32(x[i].d_imag);

            acc_ac_xr += x_real * (float) vaddvq_s32(acc_ac0);
            acc_bd_xi += x_imag * (float) vaddvq_s32(acc_bd0);
            acc_bc_xr += x_real * (float) vaddvq_s32(acc_bc0);
            acc_ad_xi += x_imag * (float) vaddvq_s32(acc_ad0);
        }

        const float coeff_w_real = GGML_CPU_FP16_TO_FP32(w[0].d_real);
        const float coeff_w_imag = GGML_CPU_FP16_TO_FP32(w[0].d_imag);

        sum_real_total = coeff_w_real * acc_ac_xr + coeff_w_imag * acc_bd_xi;
        sum_imag_total = coeff_w_imag * acc_bc_xr - coeff_w_real * acc_ad_xi;
    }

    ((ggml_bf16_t *) s)[0] = GGML_FP32_TO_BF16(sum_real_total);
    ((ggml_bf16_t *) s)[1] = GGML_FP32_TO_BF16(sum_imag_total);
#else
    ggml_vec_dot_ifairy_q16_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}
