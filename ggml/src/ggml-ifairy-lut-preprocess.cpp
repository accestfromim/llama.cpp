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

#include <algorithm>

#if defined(__ARM_NEON) && defined(__aarch64__)
#    include <arm_neon.h>
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)
// wr(code) / wi(code) coefficients for all 64 3-weight patterns (direct 6-bit encoding).
// code -> (wr, wi): 0 -> (-1,0), 1 -> (1,0), 2 -> (0,-1), 3 -> (0,1)
static const int8_t k_ifairy_wr0[64] = { -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1,
                                            0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0,
                                            -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0 };
static const int8_t k_ifairy_wr1[64] = { -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                            -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                            -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                            -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
static const int8_t k_ifairy_wr2[64] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                            1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                                            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 };
static const int8_t k_ifairy_wi0[64] = { 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0,
                                            -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1,
                                            0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1 };
static const int8_t k_ifairy_wi1[64] = { 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1,
                                            0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1,
                                            0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1,
                                            0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1 };
static const int8_t k_ifairy_wi2[64] = { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                            1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 };
#endif
static void ggml_ifairy_lut_preprocess_legacy(int          m,
                                                int          k,
                                                int          n,
                                                const void * act,
                                                size_t       act_stride,
                                                void *       lut_scales,
                                                void *       lut_buf) {
    (void) m;  // rows unused in preprocess (per-column)
    if (!act || !lut_scales || !lut_buf) {
        return;
    }

    const int64_t K                = k;
    const int64_t blocks           = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups           = blocks * groups_per_block;

    for (int col = 0; col < n; ++col) {
        const uint8_t *          act_col_bytes = (const uint8_t *) act + (size_t) col * act_stride;
        const block_ifairy_q16 * act_blocks    = (const block_ifairy_q16 *) act_col_bytes;
        float *                  scales_out    = (float *) lut_scales + (size_t) col * (size_t) blocks * 2;

        // Layout: per-group, 64 patterns, interleaved 4 channels:
        //   tbl[(pat*4) + 0..3] = { sum_ac, sum_ad, sum_bc, sum_bd } (int16)
        int16_t * lut_out =
            (int16_t *) ((uint8_t *) lut_buf + (size_t) col * (size_t) groups *
                                                   (size_t) (k_ifairy_lut_channels * k_ifairy_lut_patterns) *
                                                    sizeof(int16_t));

        // per-block activation scales (shared by all groups in the block)
        for (int64_t blk = 0; blk < blocks; ++blk) {
            scales_out[blk * 2 + 0] = GGML_FP16_TO_FP32(act_blocks[blk].d_real);
            scales_out[blk * 2 + 1] = GGML_FP16_TO_FP32(act_blocks[blk].d_imag);
        }

        for (int64_t g = 0; g < groups; ++g) {
            const int64_t blk   = g / groups_per_block;
            const int64_t intra = g - blk * groups_per_block;

            const bool    tail     = intra == groups_per_block - 1;
            const int64_t base_off = tail ? (QK_K - 1) : intra * 3;
            const int64_t idx0     = blk * QK_K + base_off + 0;

            const int blk0 = (int) blk;
            const int off0 = (int) base_off;
            const int blk1 = (int) blk;
            const int blk2 = (int) blk;
            const int off1 = (int) (base_off + 1);
            const int off2 = (int) (base_off + 2);

            int8_t xr0 = 0;
            int8_t xi0 = 0;
            int8_t xr1 = 0;
            int8_t xi1 = 0;
            int8_t xr2 = 0;
            int8_t xi2 = 0;

            if (idx0 < K) {
                xr0 = (int8_t) act_blocks[blk0].x_real[off0];
                xi0 = (int8_t) act_blocks[blk0].x_imag[off0];
            }
            if (!tail) {
                xr1 = (int8_t) act_blocks[blk1].x_real[off1];
                xi1 = (int8_t) act_blocks[blk1].x_imag[off1];
                xr2 = (int8_t) act_blocks[blk2].x_real[off2];
                xi2 = (int8_t) act_blocks[blk2].x_imag[off2];
            }

            int16_t * tbl = lut_out + (size_t) g * (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels);

#if defined(__ARM_NEON) && defined(__aarch64__)
            const int8_t xr0_s8 = xr0;
            const int8_t xr1_s8 = xr1;
            const int8_t xr2_s8 = xr2;
            const int8_t xi0_s8 = xi0;
            const int8_t xi1_s8 = xi1;
            const int8_t xi2_s8 = xi2;

            for (int pat = 0; pat < 64; pat += 16) {
                const int8x16_t wr0 = vld1q_s8(k_ifairy_wr0 + pat);
                const int8x16_t wr1 = vld1q_s8(k_ifairy_wr1 + pat);
                const int8x16_t wr2 = vld1q_s8(k_ifairy_wr2 + pat);
                const int8x16_t wi0 = vld1q_s8(k_ifairy_wi0 + pat);
                const int8x16_t wi1 = vld1q_s8(k_ifairy_wi1 + pat);
                const int8x16_t wi2 = vld1q_s8(k_ifairy_wi2 + pat);

                int16x8_t ac0 = vmull_s8(vget_low_s8(wr0), vdup_n_s8(xr0_s8));
                ac0           = vmlal_s8(ac0, vget_low_s8(wr1), vdup_n_s8(xr1_s8));
                ac0           = vmlal_s8(ac0, vget_low_s8(wr2), vdup_n_s8(xr2_s8));

                int16x8_t ad0 = vmull_s8(vget_low_s8(wr0), vdup_n_s8(xi0_s8));
                ad0           = vmlal_s8(ad0, vget_low_s8(wr1), vdup_n_s8(xi1_s8));
                ad0           = vmlal_s8(ad0, vget_low_s8(wr2), vdup_n_s8(xi2_s8));

                int16x8_t bc0 = vmull_s8(vget_low_s8(wi0), vdup_n_s8(xr0_s8));
                bc0           = vmlal_s8(bc0, vget_low_s8(wi1), vdup_n_s8(xr1_s8));
                bc0           = vmlal_s8(bc0, vget_low_s8(wi2), vdup_n_s8(xr2_s8));

                int16x8_t bd0 = vmull_s8(vget_low_s8(wi0), vdup_n_s8(xi0_s8));
                bd0           = vmlal_s8(bd0, vget_low_s8(wi1), vdup_n_s8(xi1_s8));
                bd0           = vmlal_s8(bd0, vget_low_s8(wi2), vdup_n_s8(xi2_s8));

                int16x8_t ac1 = vmull_s8(vget_high_s8(wr0), vdup_n_s8(xr0_s8));
                ac1           = vmlal_s8(ac1, vget_high_s8(wr1), vdup_n_s8(xr1_s8));
                ac1           = vmlal_s8(ac1, vget_high_s8(wr2), vdup_n_s8(xr2_s8));

                int16x8_t ad1 = vmull_s8(vget_high_s8(wr0), vdup_n_s8(xi0_s8));
                ad1           = vmlal_s8(ad1, vget_high_s8(wr1), vdup_n_s8(xi1_s8));
                ad1           = vmlal_s8(ad1, vget_high_s8(wr2), vdup_n_s8(xi2_s8));

                int16x8_t bc1 = vmull_s8(vget_high_s8(wi0), vdup_n_s8(xr0_s8));
                bc1           = vmlal_s8(bc1, vget_high_s8(wi1), vdup_n_s8(xr1_s8));
                bc1           = vmlal_s8(bc1, vget_high_s8(wi2), vdup_n_s8(xr2_s8));

                int16x8_t bd1 = vmull_s8(vget_high_s8(wi0), vdup_n_s8(xi0_s8));
                bd1           = vmlal_s8(bd1, vget_high_s8(wi1), vdup_n_s8(xi1_s8));
                bd1           = vmlal_s8(bd1, vget_high_s8(wi2), vdup_n_s8(xi2_s8));

                int16x8x4_t out0;
                out0.val[0] = ac0;
                out0.val[1] = ad0;
                out0.val[2] = bc0;
                out0.val[3] = bd0;
                vst4q_s16(tbl + (size_t) pat * 4, out0);

                int16x8x4_t out1;
                out1.val[0] = ac1;
                out1.val[1] = ad1;
                out1.val[2] = bc1;
                out1.val[3] = bd1;
                vst4q_s16(tbl + (size_t) (pat + 8) * 4, out1);
            }
#endif
        }
    }
}



void ggml_ifairy_lut_preprocess_ex(int          m,
                                   int          k,
                                   int          n,
                                   const void * act,
                                   size_t       act_stride,
                                   void *       lut_scales,
                                   void *       lut_buf,
                                   int          ith,
                                   int          nth) {
    const ggml_ifairy_lut_layout layout = ggml_ifairy_lut_layout_from_env(n);
    if (layout == GGML_IFAIRY_LUT_LAYOUT_LEGACY) {
        if (ith == 0) {
            ggml_ifairy_lut_preprocess_legacy(m, k, n, act, act_stride, lut_scales, lut_buf);
        }
        return;
    }
}
