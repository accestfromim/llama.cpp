#include "ifairy-fuse.h"

#ifdef GGML_IFAIRY_LUT_CPU

#include "ggml-ifairy-lut-impl.h"
#include "ggml-ifairy-lut.h"
#include "quants.h"

#include <algorithm>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline float ggml_ifairy_wide_linear_lut_bias_at(const struct ggml_tensor * bias,
                                                         int64_t                    i0,
                                                         int64_t                    i1,
                                                         int64_t                    i2,
                                                         int64_t                    i3) {
    const char * ptr = (const char *) bias->data + (i0 % bias->ne[0]) * bias->nb[0] +
                       (i1 % bias->ne[1]) * bias->nb[1] + (i2 % bias->ne[2]) * bias->nb[2] +
                       (i3 % bias->ne[3]) * bias->nb[3];
    return *(const float *) ptr;
}

static bool ggml_ifairy_wide_linear_w2_have_packed_weights(const struct ggml_tensor * dst) {
    for (int i = 1; i <= 4; ++i) {
        const struct ggml_tensor * weight = dst->src[i];
        const struct ifairy_lut_extra * extra = weight ? (const struct ifairy_lut_extra *) weight->extra : nullptr;
        if (!extra || !extra->packed_w) {
            return false;
        }
    }
    return true;
}

size_t ggml_ifairy_wide_linear_w2_lut_wsize(const struct ggml_tensor * dst) {
    const struct ggml_tensor * x = dst ? dst->src[0] : nullptr;
    if (!x || x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || x->ne[0] % QK_IFAIRY != 0) {
        return 0;
    }

    const char * enabled_env = getenv("GGML_IFAIRY_LUT");
    if (!enabled_env || strcmp(enabled_env, "0") == 0) {
        return 0;
    }

    const size_t n             = (size_t) ggml_nrows(x);
    const size_t k             = (size_t) x->ne[0];
    const size_t m             = (size_t) dst->ne[0];
    const size_t act_blocks    = k / QK_IFAIRY;
    const size_t weight_blocks = k / QK_IFAIRY64;
    const size_t groups        = weight_blocks * QK_IFAIRY64_GROUPS_PER_BLOCK;

    GGML_ASSERT(n == 0 || act_blocks <= SIZE_MAX / n);
    const size_t q_elems = n * act_blocks;
    GGML_ASSERT(q_elems == 0 || sizeof(block_ifairy_q16) <= SIZE_MAX / q_elems);
    const size_t q_bytes = GGML_PAD(q_elems * sizeof(block_ifairy_q16), 64);

    GGML_ASSERT(n == 0 || groups <= SIZE_MAX / n);
    const size_t lut_groups = n * groups;
    GGML_ASSERT(lut_groups == 0 || k_ifairy_lut_group_bytes <= SIZE_MAX / lut_groups);
    const size_t lut_bytes = lut_groups * k_ifairy_lut_group_bytes;

    GGML_ASSERT(n == 0 || weight_blocks <= SIZE_MAX / n);
    const size_t scale_elems = n * weight_blocks;
    GGML_ASSERT(scale_elems <= SIZE_MAX / (2u * sizeof(float)));
    const size_t scale_bytes = scale_elems * 2u * sizeof(float);
    GGML_ASSERT(lut_bytes <= SIZE_MAX - scale_bytes);
    const size_t shared_bytes = GGML_PAD(lut_bytes + scale_bytes, 64);

    GGML_ASSERT(n == 0 || m <= SIZE_MAX / n);
    const size_t output_elems = n * m;
    GGML_ASSERT(output_elems <= SIZE_MAX / (8u * sizeof(float)));
    const size_t output_bytes = output_elems * 8u * sizeof(float);

    GGML_ASSERT(q_bytes <= SIZE_MAX - shared_bytes);
    GGML_ASSERT(q_bytes + shared_bytes <= SIZE_MAX - output_bytes);
    return q_bytes + shared_bytes + output_bytes;
}

static void ggml_ifairy_wide_linear_w2_flip_qimag(block_ifairy_q16 * q_x,
                                                   int64_t            blocks,
                                                   int64_t            ith,
                                                   int64_t            nth) {
    for (int64_t ib = ith; ib < blocks; ib += nth) {
        for (int j = 0; j < QK_IFAIRY; ++j) {
            q_x[ib].x_imag[j] = (uint8_t) (-(int8_t) q_x[ib].x_imag[j]);
        }
    }
}

bool ggml_compute_forward_ifairy_wide_linear_w2_lut(const struct ggml_compute_params * params,
                                                     struct ggml_tensor *                dst,
                                                     bool                                lut_c) {
    if (!ggml_ifairy_wide_linear_w2_have_packed_weights(dst)) {
        return false;
    }

    const struct ggml_tensor * x      = dst->src[0];
    const struct ggml_tensor * bias   = dst->src[5];
    const int64_t K                   = x->ne[0];
    const int64_t M                   = dst->ne[0];
    const int64_t N                   = ggml_nrows(x);
    const int64_t act_blocks          = K / QK_IFAIRY;
    const int64_t weight_blocks       = K / QK_IFAIRY64;
    const int64_t groups              = weight_blocks * QK_IFAIRY64_GROUPS_PER_BLOCK;
    const size_t  q_bytes             = GGML_PAD((size_t) N * (size_t) act_blocks * sizeof(block_ifairy_q16), 64);
    const size_t  lut_bytes           = (size_t) N * (size_t) groups * k_ifairy_lut_group_bytes;
    const size_t  scale_bytes         = (size_t) N * (size_t) weight_blocks * 2u * sizeof(float);
    const size_t  shared_bytes        = GGML_PAD(lut_bytes + scale_bytes, 64);
    const size_t  output_plane_bytes  = (size_t) M * (size_t) N * 2u * sizeof(float);
    const size_t  need                = ggml_ifairy_wide_linear_w2_lut_wsize(dst);

    if (!params->wdata || params->wsize < need) {
        return false;
    }

    block_ifairy_q16 * q_x    = (block_ifairy_q16 *) params->wdata;
    uint8_t *          shared = (uint8_t *) params->wdata + q_bytes;
    void *             lut    = shared;
    float *            scales = (float *) (shared + lut_bytes);
    float *            outputs[4];
    for (int i = 0; i < 4; ++i) {
        outputs[i] = (float *) (shared + shared_bytes + (size_t) i * output_plane_bytes);
    }

    void (*quantize_act)(const float * GGML_RESTRICT, void * GGML_RESTRICT, int64_t) =
        lut_c ? quantize_row_ifairy_q16_lut_c : quantize_row_ifairy_q16_tensor;
    for (int64_t ir = params->ith; ir < N; ir += params->nth) {
        quantize_act((const float *) ((const char *) x->data + ir * x->nb[1]), q_x + ir * act_blocks, K);
    }
    ggml_barrier(params->threadpool);

    const int64_t q_blocks_total = N * act_blocks;
    ggml_ifairy_wide_linear_w2_flip_qimag(q_x, q_blocks_total, params->ith, params->nth);
    ggml_barrier(params->threadpool);
    ggml_ifairy64_lut_preprocess_ex_lut16((int) M, (int) K, (int) N, q_x,
                                           (size_t) act_blocks * sizeof(block_ifairy_q16), scales, lut,
                                           params->ith, params->nth);
    ggml_barrier(params->threadpool);

    const int64_t tiles_total = (M + 15) / 16;
    const int64_t tile0       = (tiles_total * params->ith) / params->nth;
    const int64_t tile1       = (tiles_total * (params->ith + 1)) / params->nth;
    const int64_t row0        = tile0 * 16;
    const int64_t row1        = std::min<int64_t>(tile1 * 16, M);
    const int64_t nrows       = row1 - row0;
    const size_t  packed_tile_bytes = (size_t) weight_blocks * sizeof(ifairy64_lut_wtile_16);

    if (nrows > 0) {
        for (int branch = 0; branch < 2; ++branch) {
            const ifairy_lut_extra * extra = (const ifairy_lut_extra *) dst->src[branch + 1]->extra;
            const void * packed_w = (const uint8_t *) extra->packed_w + (size_t) tile0 * packed_tile_bytes;
            ggml_ifairy64_lut_qgemm_lut16((int) nrows, (int) K, (int) N, packed_w, lut, scales,
                                           outputs[branch] + row0 * 2, (size_t) M * 2u * sizeof(float),
                                           2u * sizeof(float), /*pack_bf16*/ false, /*add*/ false);
        }
    }
    ggml_barrier(params->threadpool);

    ggml_ifairy_wide_linear_w2_flip_qimag(q_x, q_blocks_total, params->ith, params->nth);
    ggml_barrier(params->threadpool);
    ggml_ifairy64_lut_preprocess_ex_lut16((int) M, (int) K, (int) N, q_x,
                                           (size_t) act_blocks * sizeof(block_ifairy_q16), scales, lut,
                                           params->ith, params->nth);
    ggml_barrier(params->threadpool);

    if (nrows > 0) {
        for (int branch = 2; branch < 4; ++branch) {
            const ifairy_lut_extra * extra = (const ifairy_lut_extra *) dst->src[branch + 1]->extra;
            const void * packed_w = (const uint8_t *) extra->packed_w + (size_t) tile0 * packed_tile_bytes;
            ggml_ifairy64_lut_qgemm_lut16((int) nrows, (int) K, (int) N, packed_w, lut, scales,
                                           outputs[branch] + row0 * 2, (size_t) M * 2u * sizeof(float),
                                           2u * sizeof(float), /*pack_bf16*/ false, /*add*/ false);
        }
    }
    ggml_barrier(params->threadpool);

    const int64_t total = M * N;
    const int64_t begin = (total * params->ith) / params->nth;
    const int64_t end   = (total * (params->ith + 1)) / params->nth;
    for (int64_t index = begin; index < end; ++index) {
        const int64_t row = index % M;
        const int64_t i1  = (index / M) % x->ne[1];
        const int64_t i2  = ((index / M) / x->ne[1]) % x->ne[2];
        const int64_t i3  = (index / M) / (x->ne[1] * x->ne[2]);
        float real = 0.0f;
        float imag = 0.0f;
        for (int branch = 0; branch < 4; ++branch) {
            real += outputs[branch][index * 2 + 0];
            imag += outputs[branch][index * 2 + 1];
        }
        if (bias) {
            real += ggml_ifairy_wide_linear_lut_bias_at(bias, row, i1, i2, i3);
            imag += ggml_ifairy_wide_linear_lut_bias_at(bias, row + M, i1, i2, i3);
        }
        ggml_bf16_t * out = (ggml_bf16_t *) ((char *) dst->data + (index / M) * dst->nb[1] + row * dst->nb[0]);
        out[0]             = GGML_FP32_TO_BF16(real);
        out[1]             = GGML_FP32_TO_BF16(imag);
    }
    return true;
}

#else

size_t ggml_ifairy_wide_linear_w2_lut_wsize(const struct ggml_tensor * dst) {
    (void) dst;
    return 0;
}

bool ggml_compute_forward_ifairy_wide_linear_w2_lut(const struct ggml_compute_params * params,
                                                     struct ggml_tensor *                dst,
                                                     bool                                lut_c) {
    (void) params;
    (void) dst;
    (void) lut_c;
    return false;
}

#endif
