#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "simd-mappings.h"

#include "../../ggml-cpu-impl.h"
#include "../../quants.h"
#include "quants-ifairy.h"

#include <assert.h>
#include <pthread.h>

typedef void (*ggml_vec_dot_ifairy_impl_t)(int                        n,
                                           float * GGML_RESTRICT      s,
                                           size_t                     bs,
                                           const void * GGML_RESTRICT vx,
                                           size_t                     bx,
                                           const void * GGML_RESTRICT vy,
                                           size_t                     by,
                                           int                        nrc);

#if defined(__ARM_NEON) && defined(__aarch64__)
static inline int32_t ggml_ifairy_dot_i8x16(int8x16_t a, int8x16_t b) {
    const int16x8_t prod_lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    const int16x8_t prod_hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));

    int32x4_t sum = vpaddlq_s16(prod_lo);
    sum           = vaddq_s32(sum, vpaddlq_s16(prod_hi));

    return vaddvq_s32(sum);
}

static inline void ggml_ifairy_accumulate_lanes_16(const int8_t * GGML_RESTRICT xr,
                                                   const int8_t * GGML_RESTRICT xi,
                                                   const int8x16_t              wr,
                                                   const int8x16_t              wi,
                                                   int32_t *                    sum_ac,
                                                   int32_t *                    sum_ad,
                                                   int32_t *                    sum_bc,
                                                   int32_t *                    sum_bd) {
    const int8x16_t vxr = vld1q_s8(xr);
    const int8x16_t vxi = vld1q_s8(xi);

    *sum_ac += ggml_ifairy_dot_i8x16(vxr, wr);
    *sum_ad += ggml_ifairy_dot_i8x16(vxi, wr);
    *sum_bc += ggml_ifairy_dot_i8x16(vxr, wi);
    *sum_bd += ggml_ifairy_dot_i8x16(vxi, wi);
}

static inline void ggml_ifairy_accumulate_block_64(const uint8_t * GGML_RESTRICT w_iter,
                                                   const int8_t * GGML_RESTRICT  xr,
                                                   const int8_t * GGML_RESTRICT  xi,
                                                   const int8x16_t               lut_wr_i0,
                                                   const int8x16_t               lut_wi_i0,
                                                   const int8x16_t               lut_wr_i1,
                                                   const int8x16_t               lut_wi_i1,
                                                   const uint8x16_t              mask_0f,
                                                   int32_t *                     sum_ac,
                                                   int32_t *                     sum_ad,
                                                   int32_t *                     sum_bc,
                                                   int32_t *                     sum_bd) {
    const uint8x16_t packed = vld1q_u8(w_iter);
    const uint8x16_t lo     = vandq_u8(packed, mask_0f);
    const uint8x16_t hi     = vshrq_n_u8(packed, 4);

    ggml_ifairy_accumulate_lanes_16(
        xr + 0, xi + 0, vqtbl1q_s8(lut_wr_i0, lo), vqtbl1q_s8(lut_wi_i0, lo), sum_ac, sum_ad, sum_bc, sum_bd);
    ggml_ifairy_accumulate_lanes_16(
        xr + 16, xi + 16, vqtbl1q_s8(lut_wr_i1, lo), vqtbl1q_s8(lut_wi_i1, lo), sum_ac, sum_ad, sum_bc, sum_bd);
    ggml_ifairy_accumulate_lanes_16(
        xr + 32, xi + 32, vqtbl1q_s8(lut_wr_i0, hi), vqtbl1q_s8(lut_wi_i0, hi), sum_ac, sum_ad, sum_bc, sum_bd);
    ggml_ifairy_accumulate_lanes_16(
        xr + 48, xi + 48, vqtbl1q_s8(lut_wr_i1, hi), vqtbl1q_s8(lut_wi_i1, hi), sum_ac, sum_ad, sum_bc, sum_bd);
}
#endif

void ggml_vec_dot_ifairy_q16_K_neon(int                        n,
                                    float * GGML_RESTRICT      s,
                                    size_t                     bs,
                                    const void * GGML_RESTRICT vx,
                                    size_t                     bx,
                                    const void * GGML_RESTRICT vy,
                                    size_t                     by,
                                    int                        nrc) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    assert(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);

    const block_ifairy * GGML_RESTRICT     w  = (const block_ifairy *) vx;
    const block_ifairy_q16 * GGML_RESTRICT x  = (const block_ifairy_q16 *) vy;
    const int                              nb = n / QK_IFAIRY;

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

    const uint8x16_t mask_0f   = vdupq_n_u8(0x0F);
    const int8x16_t  lut_wr_i0 = vld1q_s8(lut_real_data);
    const int8x16_t  lut_wi_i0 = vld1q_s8(lut_imag_data);
    const int8x16_t  lut_wr_i1 = vld1q_s8(lut_wr_idx1_data);
    const int8x16_t  lut_wi_i1 = vld1q_s8(lut_wi_idx1_data);

    float sum_real_total = 0.0f;
    float sum_imag_total = 0.0f;

    if (ggml_ifairy_vecdot_act_tensor_enabled()) {
        int32_t sum_ac_total = 0;
        int32_t sum_ad_total = 0;
        int32_t sum_bc_total = 0;
        int32_t sum_bd_total = 0;

        for (int i = 0; i < nb; ++i) {
            __builtin_prefetch(w + i + 1, 0, 1);
            __builtin_prefetch(x + i + 1, 0, 1);

            const uint8_t * GGML_RESTRICT w_ptr = w[i].qs;
            const int8_t * GGML_RESTRICT xr = (const int8_t *) x[i].x_real;
            const int8_t * GGML_RESTRICT xi = (const int8_t *) x[i].x_imag;

            int32_t sum_ac = 0;
            int32_t sum_ad = 0;
            int32_t sum_bc = 0;
            int32_t sum_bd = 0;

            for (int j = 0; j < QK_IFAIRY; j += 64) {
                ggml_ifairy_accumulate_block_64(
                    w_ptr + (j >> 2), xr + j, xi + j, lut_wr_i0, lut_wi_i0, lut_wr_i1, lut_wi_i1, mask_0f, &sum_ac,
                    &sum_ad, &sum_bc, &sum_bd);
            }

            sum_ac_total += sum_ac;
            sum_ad_total += sum_ad;
            sum_bc_total += sum_bc;
            sum_bd_total += sum_bd;
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

            const uint8_t * GGML_RESTRICT w_ptr = w[i].qs;
            const int8_t * GGML_RESTRICT  xr    = (const int8_t *) x[i].x_real;
            const int8_t * GGML_RESTRICT  xi    = (const int8_t *) x[i].x_imag;

            int32_t sum_ac = 0;
            int32_t sum_ad = 0;
            int32_t sum_bc = 0;
            int32_t sum_bd = 0;

            for (int j = 0; j < QK_IFAIRY; j += 64) {
                ggml_ifairy_accumulate_block_64(
                    w_ptr + (j >> 2), xr + j, xi + j, lut_wr_i0, lut_wi_i0, lut_wr_i1, lut_wi_i1, mask_0f, &sum_ac,
                    &sum_ad, &sum_bc, &sum_bd);
            }

            const float x_real = GGML_CPU_FP16_TO_FP32(x[i].d_real);
            const float x_imag = GGML_CPU_FP16_TO_FP32(x[i].d_imag);

            acc_ac_xr += x_real * (float) sum_ac;
            acc_bd_xi += x_imag * (float) sum_bd;
            acc_bc_xr += x_real * (float) sum_bc;
            acc_ad_xi += x_imag * (float) sum_ad;
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

static pthread_once_t             ggml_ifairy_vecdot_once = PTHREAD_ONCE_INIT;
static ggml_vec_dot_ifairy_impl_t ggml_ifairy_vecdot_impl = ggml_vec_dot_ifairy_q16_K_generic;

static void ggml_ifairy_vecdot_init(void) {
    ggml_ifairy_vecdot_impl = ggml_vec_dot_ifairy_q16_K_generic;

    if (ggml_vec_dot_ifairy_q16_K_dotprod_available() && ggml_cpu_has_dotprod()) {
        ggml_ifairy_vecdot_impl = ggml_vec_dot_ifairy_q16_K_dotprod;
    } else if (ggml_cpu_has_neon()) {
        ggml_ifairy_vecdot_impl = ggml_vec_dot_ifairy_q16_K_neon;
    }
}

void ggml_vec_dot_ifairy_q16_K(int                        n,
                               float * GGML_RESTRICT      s,
                               size_t                     bs,
                               const void * GGML_RESTRICT vx,
                               size_t                     bx,
                               const void * GGML_RESTRICT vy,
                               size_t                     by,
                               int                        nrc) {
    pthread_once(&ggml_ifairy_vecdot_once, ggml_ifairy_vecdot_init);
    ggml_ifairy_vecdot_impl(n, s, bs, vx, bx, vy, by, nrc);
}
