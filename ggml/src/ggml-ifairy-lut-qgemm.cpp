#define GGML_COMMON_DECL_CPP
#include "ggml-backend.h"
#include "ggml-common.h"
#include "ggml-ifairy-lut-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#ifndef GGML_FP16_TO_FP32
#    define GGML_FP16_TO_FP32 ggml_fp16_to_fp32
#endif

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>

#if defined(__ARM_NEON) && defined(__aarch64__)
#    include <arm_neon.h>
#endif

// Prefetch is enabled by default; set GGML_IFAIRY_LUT_PREFETCH=0 to disable for tuning.
// Note: env is read once per process (cached) to avoid per-op getenv overhead.
static inline bool ggml_ifairy_lut_prefetch_enabled(void) {
    static std::atomic<int> cached(-1);  // -1=unset, 0=disabled, 1=enabled
    int                     v = cached.load(std::memory_order_relaxed);
    if (v >= 0) {
        return v != 0;
    }
    const char * env = getenv("GGML_IFAIRY_LUT_PREFETCH");
    v                = (env && strcmp(env, "0") == 0) ? 0 : 1;
    cached.store(v, std::memory_order_relaxed);
    return v != 0;
}

// Prefetch distance in groups; defaults to 2. Set GGML_IFAIRY_LUT_PREFETCH_DIST=0 to disable distance-based prefetch.
// Note: env is read once per process (cached) to avoid per-op getenv overhead.
static inline int ggml_ifairy_lut_prefetch_dist(void) {
    static std::atomic<int> cached(-1);  // -1=unset, else the prefetch distance
    int                     v = cached.load(std::memory_order_relaxed);
    if (v >= 0) {
        return v;
    }

    const char * env = getenv("GGML_IFAIRY_LUT_PREFETCH_DIST");
    if (!env || env[0] == '\0') {
        v = 2;
    } else {
        char *     end = NULL;
        const long val = strtol(env, &end, 10);
        if (end == env) {
            v = 2;
        } else if (val <= 0) {
            v = 0;
        } else if (val > 16) {
            v = 16;
        } else {
            v = (int) val;
        }
    }

    cached.store(v, std::memory_order_relaxed);
    return v;
}


static void ggml_ifairy_lut_qgemm_ex_legacy(int             m,
                                            int             k,
                                            int             n,
                                            const void *    qweights,
                                            const uint8_t * indexes,
                                            const void *    lut,
                                            const void *    lut_scales,
                                            const void *    act,
                                            size_t          act_stride,
                                            float *         dst,
                                            size_t          dst_col_stride,
                                            size_t          dst_row_stride,
                                            bool            pack_bf16,
                                            bool            strict,
                                            bool            add) {
    if (!indexes || !dst || !qweights || !lut || !lut_scales) {
        return;
    }
    if (strict) {
        GGML_ASSERT(add == false);
    }

    const int    prefetch_dist   = ggml_ifairy_lut_prefetch_dist();
    const bool   prefetch        = ggml_ifairy_lut_prefetch_enabled() && prefetch_dist > 0;

    const int64_t K                = k;
    const int64_t blocks           = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups           = blocks * groups_per_block;

    const block_ifairy * w_blocks = (const block_ifairy *) qweights;


    for (int col = 0; col < n; ++col) {
            const int16_t * lut_base =
                (const int16_t *) ((const uint8_t *) lut +
                                   (size_t) col * (size_t) groups *
                                       (size_t) (k_ifairy_lut_channels * k_ifairy_lut_patterns) * sizeof(int16_t));
            const float *            scales = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2;

        for (int row = 0; row < m; ++row) {
            const block_ifairy * w_row   = w_blocks + (size_t) row * (size_t) blocks;
            const uint8_t *      idx_row = indexes + (size_t) row * (size_t) groups;

            const float coeff_w_real = GGML_FP16_TO_FP32(w_row[0].d_real);
            const float coeff_w_imag = GGML_FP16_TO_FP32(w_row[0].d_imag);
            
            float acc_ac_xr = 0.0f;
            float acc_ad_xi = 0.0f;
            float acc_bc_xr = 0.0f;
            float acc_bd_xi = 0.0f;

            float32x4_t accv = vdupq_n_f32(0.0f);  // {ac, ad, bc, bd}
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32x4_t isum0 = vdupq_n_s32(0);
                int32x4_t isum1 = vdupq_n_s32(0);

                const uint8_t * idx_blk = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int16_t * lut_blk = lut_base + (size_t) blk * (size_t) groups_per_block *
                                                         (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels);

                const size_t group_stride = (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels);
                for (size_t gi = 0; gi < groups_per_block; ++gi) {
                    const uint8_t pat0 = (uint8_t) (idx_blk[gi]);
                    const int16_t * grp0 = lut_blk + (size_t) gi * group_stride;
                    const int16_t * tbl0 = grp0 + (size_t) pat0 * k_ifairy_lut_channels;
                    const int16x4_t s0 = vld1_s16(tbl0);
                    isum0 = vaddw_s16(isum0, s0);
                }


                const float32x2_t srsi  = vld1_f32(scales + (size_t) blk * 2);
                const float32x4_t scv   = vcombine_f32(srsi, srsi);  // {sr, si, sr, si}
                const float32x4_t sumsf = vcvtq_f32_s32(vaddq_s32(isum0, isum1));
                accv                    = vmlaq_f32(accv, sumsf, scv);
            }

            acc_ac_xr = vgetq_lane_f32(accv, 0);
            acc_ad_xi = vgetq_lane_f32(accv, 1);
            acc_bc_xr = vgetq_lane_f32(accv, 2);
            acc_bd_xi = vgetq_lane_f32(accv, 3);

            const float out_r = coeff_w_real * acc_ac_xr + coeff_w_imag * acc_bd_xi;
            const float out_i = coeff_w_imag * acc_bc_xr - coeff_w_real * acc_ad_xi;

            if (!isfinite(out_r) || !isfinite(out_i)) {
                ggml_abort(__FILE__, __LINE__, "ifairy_lut_qgemm: non-finite output (row=%d col=%d acc_r=%f acc_i=%f)", row, col, out_r, out_i);
            }

            uint8_t * out_base = (uint8_t *) dst + (size_t) col * dst_col_stride + (size_t) row * dst_row_stride;
            if (pack_bf16) {
                ggml_bf16_t br                = GGML_FP32_TO_BF16(out_r);
                ggml_bf16_t bi                = GGML_FP32_TO_BF16(out_i);
                ((ggml_bf16_t *) out_base)[0] = br;
                ((ggml_bf16_t *) out_base)[1] = bi;
            } else {
                float * out_ptr = (float *) out_base;
                if (add) {
                    out_ptr[0] += out_r;
                    out_ptr[1] += out_i;
                } else {
                    out_ptr[0] = out_r;
                    out_ptr[1] = out_i;
                }
            }
        }
    }
}

void ggml_ifairy_lut_qgemm_ex(int             m,
                                int             k,
                                int             n,
                                const void *    qweights,
                                const uint8_t * indexes,
                                const void *    lut,
                                const void *    lut_scales,
                                const void *    act,
                                size_t          act_stride,
                                float *         dst,
                                size_t          dst_col_stride,
                                size_t          dst_row_stride,
                                bool            pack_bf16,
                                bool            strict,
                                bool            add) {
    const ggml_ifairy_lut_layout layout = ggml_ifairy_lut_layout_from_env(n);
    if (layout == GGML_IFAIRY_LUT_LAYOUT_LEGACY) {
        ggml_ifairy_lut_qgemm_ex_legacy(m, k, n, qweights, indexes, lut, lut_scales, act, act_stride, dst,
                                        dst_col_stride, dst_row_stride, pack_bf16, strict, add);
        return;
    }
}

void   ggml_ifairy_lut_mul_mat_scalar(int          m,
                                      int          k,
                                      int          n,
                                      const void * qweights,
                                      const void * act,
                                      size_t       act_stride,
                                      float *      dst){}