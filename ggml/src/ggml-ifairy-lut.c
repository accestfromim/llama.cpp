// Copyright (c) 2024 The ggml authors
//
// SPDX-License-Identifier: MIT

#include "ggml-ifairy.h"

#include "ggml-impl.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#undef GGML_COMMON_DECL_C

#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

static bool ifairy_lut_initialized = false;
static bool ifairy_lut_env_checked = false;
static bool ifairy_lut_enabled = true;
static struct ggml_ifairy_tensor_extra * ifairy_tensor_extras = NULL;
static size_t ifairy_tensor_extras_index = 0;

static bool ggml_ifairy_select_tile_params(const int64_t m, const int64_t k, int * bm, int * bk) {
    GGML_ASSERT(bm != NULL && bk != NULL);

    if (m == 1536 && k == 4096) {
        *bm = 256;
        *bk = 128;
        return true;
    }

    if (m == 1536 && k == 1536) {
        *bm = 128;
        *bk = 64;
        return true;
    }

    if (m == 4096 && k == 1536) {
        *bm = 256;
        *bk = 128;
        return true;
    }

    return false;
}

static void ggml_ifairy_init_once(void) {
    if (ifairy_lut_initialized) {
        return;
    }

    if (!ifairy_lut_env_checked) {
        const char * disable_env = getenv("GGML_IFAIRY_ARM_LUT_DISABLE");
        ifairy_lut_enabled = (disable_env == NULL || disable_env[0] == '\0');
        ifairy_lut_env_checked = true;
    }

    if (!ifairy_lut_enabled) {
        ifairy_lut_initialized = true;
        return;
    }

    ifairy_tensor_extras = ggml_aligned_malloc(sizeof(struct ggml_ifairy_tensor_extra) * GGML_IFAIRY_MAX_NODES);
    GGML_ASSERT(ifairy_tensor_extras != NULL);
    memset(ifairy_tensor_extras, 0, sizeof(struct ggml_ifairy_tensor_extra) * GGML_IFAIRY_MAX_NODES);
    ifairy_lut_initialized = true;
}

void ggml_ifairy_lut_init(void) {
    ggml_ifairy_init_once();
}

void ggml_ifairy_lut_free(void) {
    if (!ifairy_lut_initialized) {
        return;
    }

    for (size_t i = 0; i < ifairy_tensor_extras_index; ++i) {
        if (ifairy_tensor_extras[i].scales) {
            ggml_aligned_free(ifairy_tensor_extras[i].scales, ifairy_tensor_extras[i].scales_bytes);
            ifairy_tensor_extras[i].scales = NULL;
        }
    }

    ggml_aligned_free(ifairy_tensor_extras, sizeof(struct ggml_ifairy_tensor_extra) * GGML_IFAIRY_MAX_NODES);
    ifairy_tensor_extras = NULL;
    ifairy_tensor_extras_index = 0;
    ifairy_lut_initialized = false;
}

static inline int8_t ggml_ifairy_clamp_s8(const float v) {
    const float clipped = v > 127.f ? 127.f : (v < -127.f ? -127.f : v);
    return (int8_t) lrintf(clipped);
}

static inline float ggml_ifairy_abs_f32(float v) {
    return v >= 0 ? v : -v;
}

static void ggml_ifairy_partial_max_reset(float * lut_scales) {
    lut_scales[0] = 0.0f;
    lut_scales[1] = 0.0f;
}

static void ggml_ifairy_per_tensor_quant(const block_ifairy_q16 * act_blocks, int64_t k, float * lut_scales_out, float * inv_scales_out) {
    ggml_ifairy_partial_max_reset(lut_scales_out);

    const int64_t n_blocks = k / QK_K;
    float max_r = 0.0f;
    float max_i = 0.0f;

    for (int64_t bi = 0; bi < n_blocks; ++bi) {
        const block_ifairy_q16 * blk = &act_blocks[bi];
        const float d_r = GGML_FP16_TO_FP32(blk->d_real);
        const float d_i = GGML_FP16_TO_FP32(blk->d_imag);

        for (int j = 0; j < QK_K; ++j) {
            const float vr = (float) blk->x_real[j] * d_r;
            const float vi = (float) blk->x_imag[j] * d_i;
            max_r = MAX(max_r, ggml_ifairy_abs_f32(vr));
            max_i = MAX(max_i, ggml_ifairy_abs_f32(vi));
        }
    }

    const float inv_r = max_r > 0.0f ? 127.0f / max_r : 0.0f;
    const float inv_i = max_i > 0.0f ? 127.0f / max_i : 0.0f;

    lut_scales_out[0] = inv_r;
    lut_scales_out[1] = inv_i;
    if (inv_scales_out) {
        inv_scales_out[0] = inv_r;
        inv_scales_out[1] = inv_i;
    }
}

static void ggml_ifairy_lut_ctor(const block_ifairy_q16 * act_blocks, int64_t k, const float * inv_scales, int8_t * qlut_r, int8_t * qlut_i) {
    const int64_t n_blocks = k / QK_K;

    memset(qlut_r, 0, k * 32);
    memset(qlut_i, 0, k * 32);

    const float inv_r = inv_scales[0];
    const float inv_i = inv_scales[1];

    // TODO: refine layout to match tbl_impl permutation; currently writes first k bytes sequentially and leaves padding zeroed.
    for (int64_t bi = 0; bi < n_blocks; ++bi) {
        const block_ifairy_q16 * blk = &act_blocks[bi];
        const float d_r = GGML_FP16_TO_FP32(blk->d_real);
        const float d_i = GGML_FP16_TO_FP32(blk->d_imag);

        const int64_t base = bi * QK_K;
        for (int j = 0; j < QK_K; ++j) {
            const float vr = (float) blk->x_real[j] * d_r * inv_r;
            const float vi = (float) blk->x_imag[j] * d_i * inv_i;
            qlut_r[base + j] = ggml_ifairy_clamp_s8(vr);
            qlut_i[base + j] = ggml_ifairy_clamp_s8(vi);
        }
    }
}

void ggml_ifairy_transform_tensor(struct ggml_tensor * tensor) {
#if defined(GGML_USE_OPENMP)
#pragma omp critical
#endif
    {
        if (tensor->type != GGML_TYPE_IFAIRY || tensor->extra != NULL) {
            return;
        }

        if (!ifairy_lut_enabled) {
            return;
        }

        ggml_ifairy_init_once();
        GGML_ASSERT(ifairy_tensor_extras_index < GGML_IFAIRY_MAX_NODES);

        const int64_t k = tensor->ne[0];
        const int64_t m = tensor->ne[1];
        GGML_ASSERT(k % QK_K == 0);

        int bm = (int) m;
        int bk = (int) QK_K;
        const bool shape_matched = ggml_ifairy_select_tile_params(m, k, &bm, &bk);
        GGML_UNUSED(shape_matched);

        const int n_tile_num = bm > 0 ? (int) (m / bm) : 1;

        const int64_t blocks_per_row = k / QK_K;
        const int64_t block_count = blocks_per_row * m;
        const size_t scales_bytes = (size_t) block_count * 2 * sizeof(float);

        float * scales = ggml_aligned_malloc(scales_bytes);
        GGML_ASSERT(scales != NULL);

        const block_ifairy * weights = (const block_ifairy *) tensor->data;
        for (int64_t i = 0; i < block_count; ++i) {
            scales[2 * i + 0] = GGML_FP16_TO_FP32(weights[i].d_real);
            scales[2 * i + 1] = GGML_FP16_TO_FP32(weights[i].d_imag);
        }

        const size_t row_size_bytes = (size_t) blocks_per_row * sizeof(block_ifairy);
        const size_t tile_stride = row_size_bytes * (size_t) bm;
        const size_t c_tile_size = (size_t) bm;

        struct ggml_ifairy_tensor_extra extra = {
            /* .lut_scales_size = */ 2,
            /* .n_tile_num      = */ n_tile_num > 0 ? n_tile_num : 1,
            /* .bk              = */ bk,
            /* .bm              = */ bm,
            /* .tile_stride     = */ tile_stride,
            /* .c_tile_size     = */ c_tile_size,
            /* .scales_bytes    = */ scales_bytes,
            /* .qweights        = */ (uint8_t *) tensor->data,
            /* .scales          = */ scales,
        };

        ifairy_tensor_extras[ifairy_tensor_extras_index] = extra;
        tensor->extra = ifairy_tensor_extras + ifairy_tensor_extras_index;
        ++ifairy_tensor_extras_index;
    }
}

bool ggml_ifairy_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    ggml_ifairy_init_once();

    if (!ifairy_lut_enabled) {
        return false;
    }

    if (src0->type != GGML_TYPE_IFAIRY || src1->type != GGML_TYPE_IFAIRY_Q16 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (src1->ne[1] > 1) {
        return false;
    }

    int bm = 0, bk = 0;
    if (!ggml_ifairy_select_tile_params(src1->ne[1], src1->ne[0], &bm, &bk)) {
        return false;
    }

    return true;
}

size_t ggml_ifairy_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    GGML_UNUSED(dst);
    if (!ggml_ifairy_can_mul_mat(src0, src1, dst)) {
        return 0;
    }

    const int64_t k   = src1->ne[0];
    const int64_t m   = src1->ne[1];
    int bm = 0, bk = 0;
    const bool ok = ggml_ifairy_select_tile_params(m, k, &bm, &bk);
    GGML_ASSERT(ok);
    GGML_UNUSED(bk);

    // QLUT layout: real + imag, each k/2*32 bytes per column
    const size_t qlut_bytes_per_col = (size_t) k * 32;
    const size_t lut_scales_bytes   = 2 * sizeof(float);
    size_t wsize = m * (qlut_bytes_per_col + lut_scales_bytes);

    // 64B align to match ggml allocator expectations
    wsize = GGML_PAD(wsize, 64);
    return wsize;
}

void ggml_ifairy_preprocessor(int m, int k, const void * B, void * lut_scales, void * qlut_real, void * qlut_imag) {
    GGML_UNUSED(m);
    GGML_ASSERT(k % QK_K == 0);

    const block_ifairy_q16 * act_blocks = (const block_ifairy_q16 *) B;
    float inv_scales[2] = {0.0f, 0.0f};

    ggml_ifairy_per_tensor_quant(act_blocks, k, (float *) lut_scales, inv_scales);
    ggml_ifairy_lut_ctor(act_blocks, k, inv_scales, (int8_t *) qlut_real, (int8_t *) qlut_imag);
}

static inline void ggml_ifairy_decode_weight_block(const block_ifairy * blk, int8_t * wr, int8_t * wi) {
    static const int8_t lut_wr[4] = { -1, 1, 0, 0 };
    static const int8_t lut_wi[4] = { 0, 0, -1, 1 };

    for (int j = 0; j < QK_K; ++j) {
        const int chunk    = j >> 6;          // 0..3 blocks of 64
        const int lane     = j & 0xF;         // 0..15 within each 16-lane stripe
        const int part     = (j >> 4) & 0x3;  // which 16-lane group inside the chunk
        const int byte_idx = (chunk << 4) + lane;
        const int bit_off  = part * 2;
        const uint8_t code = (blk->qs[byte_idx] >> bit_off) & 0x3;
        wr[j] = lut_wr[code];
        wi[j] = lut_wi[code];
    }
}

// Reference LUT matvec (single-column) using decoded QLUT and per-tensor scales.
// Output layout: dst[2*i + 0] = real, dst[2*i + 1] = imag for row i.
void ggml_ifairy_qgemm_lut_ref(const void * w, const int8_t * qlut_r, const int8_t * qlut_i, const float * lut_scales, int64_t k, int64_t m, float * dst) {
    const block_ifairy * w_blocks = (const block_ifairy *) w;
    GGML_ASSERT(k % QK_K == 0);

    const int64_t blocks_per_row = k / QK_K;
    const float inv_lut_r = lut_scales[0] != 0.0f ? 1.0f / lut_scales[0] : 0.0f;
    const float inv_lut_i = lut_scales[1] != 0.0f ? 1.0f / lut_scales[1] : 0.0f;

    int8_t wr_buf[QK_K];
    int8_t wi_buf[QK_K];

    for (int64_t row = 0; row < m; ++row) {
        float acc_rr = 0.0f;
        float acc_ii = 0.0f;
        float acc_ri = 0.0f;
        float acc_ir = 0.0f;

        const block_ifairy * row_w = w_blocks + row * blocks_per_row;
        const int8_t * row_qr = qlut_r;
        const int8_t * row_qi = qlut_i;

        for (int64_t b = 0; b < blocks_per_row; ++b) {
            ggml_ifairy_decode_weight_block(&row_w[b], wr_buf, wi_buf);

            for (int j = 0; j < QK_K; ++j) {
                const int8_t wr_v = wr_buf[j];
                const int8_t wi_v = wi_buf[j];
                const int8_t ar   = row_qr[j];
                const int8_t ai   = row_qi[j];

                acc_rr += (float) wr_v * (float) ar;
                acc_ii += (float) wi_v * (float) ai;
                acc_ri += (float) wr_v * (float) ai;
                acc_ir += (float) wi_v * (float) ar;
            }

            row_qr += QK_K;
            row_qi += QK_K;
        }

        const float w_r = GGML_FP16_TO_FP32(row_w[0].d_real);
        const float w_i = GGML_FP16_TO_FP32(row_w[0].d_imag);

        const float scale_wr_r = w_r * inv_lut_r; // matches ar scale (real)
        const float scale_wi_i = w_i * inv_lut_i; // matches ai scale (imag)
        const float scale_wi_r = w_i * inv_lut_r; // matches ar scale for wi
        const float scale_wr_i = w_r * inv_lut_i; // matches ai scale for wr

        const float out_real = scale_wr_r * acc_rr + scale_wi_i * acc_ii;
        const float out_imag = scale_wi_r * acc_ir - scale_wr_i * acc_ri;

        dst[2 * row + 0] = out_real;
        dst[2 * row + 1] = out_imag;
    }
}
