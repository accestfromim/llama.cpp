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
#include <inttypes.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static bool ifairy_lut_initialized = false;
static bool ifairy_lut_env_checked = false;
static bool ifairy_lut_enabled = true;
static bool ifairy_lut_debug = false;
static struct ggml_ifairy_tensor_extra * ifairy_tensor_extras = NULL;
static size_t ifairy_tensor_extras_index = 0;
static int ifairy_lut_debug_reports = 0;

#define GGML_IFAIRY_DEBUG(...) \
    do { \
        if (ifairy_lut_debug && ifairy_lut_debug_reports < 8) { \
            GGML_LOG_INFO(__VA_ARGS__); \
            ++ifairy_lut_debug_reports; \
        } \
    } while (0)

static size_t ggml_ifairy_act_q16_bytes(int64_t k, int64_t ncols) {
    if (ncols == 0) {
        return 0;
    }
    const size_t row_sz = ggml_row_size(GGML_TYPE_IFAIRY_Q16, k);
    return GGML_PAD(row_sz * (size_t) ncols, 64);
}

static size_t ggml_ifairy_qlut_bytes(int64_t k) {
    GGML_ASSERT((k % 2) == 0);
    return (size_t) (k / 2) * 32;
}

static size_t ggml_ifairy_packed_bytes(int64_t k) {
    return (size_t) (k / 2) * 4; // real/imag × even/odd, contiguous per pair
}

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
        const char * debug_env   = getenv("GGML_IFAIRY_ARM_LUT_DEBUG");
        ifairy_lut_enabled = (disable_env == NULL || disable_env[0] == '\0');
        ifairy_lut_debug   = (debug_env   != NULL &&  debug_env[0] != '\0');
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

#if defined(__ARM_NEON)
static inline void ggml_ifairy_quant8_pair(const int8_t * src, float scale, int16x8_t * even_q, int16x8_t * odd_q) {
    const int8x16_t v_all  = vld1q_s8(src);
    const int8x16_t v_even = vuzp1q_s8(v_all, v_all);
    const int8x16_t v_odd  = vuzp2q_s8(v_all, v_all);

    const int16x8_t v_even16 = vmovl_s8(vget_low_s8(v_even));
    const int16x8_t v_odd16  = vmovl_s8(vget_low_s8(v_odd));

    const float32x4_t e_f0 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_even16))), scale);
    const float32x4_t e_f1 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_even16))), scale);
    const float32x4_t o_f0 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_odd16))), scale);
    const float32x4_t o_f1 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_odd16))), scale);

    const int32x4_t e_i0 = vcvtnq_s32_f32(e_f0);
    const int32x4_t e_i1 = vcvtnq_s32_f32(e_f1);
    const int32x4_t o_i0 = vcvtnq_s32_f32(o_f0);
    const int32x4_t o_i1 = vcvtnq_s32_f32(o_f1);

    const int16x8_t e_q = vcombine_s16(vmovn_s32(e_i0), vmovn_s32(e_i1));
    const int16x8_t o_q = vcombine_s16(vmovn_s32(o_i0), vmovn_s32(o_i1));
    const int16x8_t min_q = vdupq_n_s16(-127);
    const int16x8_t max_q = vdupq_n_s16(127);

    *even_q = vmaxq_s16(min_q, vminq_s16(e_q, max_q));
    *odd_q  = vmaxq_s16(min_q, vminq_s16(o_q, max_q));
}
#endif

static inline void ggml_ifairy_fill_pair_tables(int8_t * table_base, int8_t even_val, int8_t odd_val) {
    // nibble tables are 16-byte each; we keep entries constant because weight tables will provide signs
    memset(table_base,         (uint8_t) even_val, 16);
    memset(table_base + 16,    (uint8_t) odd_val,  16);
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

#if defined(__ARM_NEON)
    float32x4_t max_r_q = vdupq_n_f32(0.0f);
    float32x4_t max_i_q = vdupq_n_f32(0.0f);
#endif

    for (int64_t bi = 0; bi < n_blocks; ++bi) {
        const block_ifairy_q16 * blk = &act_blocks[bi];
        const float d_r = GGML_FP16_TO_FP32(blk->d_real);
        const float d_i = GGML_FP16_TO_FP32(blk->d_imag);

#if defined(__ARM_NEON)
        const float32x4_t vd_r = vdupq_n_f32(d_r);
        const float32x4_t vd_i = vdupq_n_f32(d_i);

        for (int j = 0; j < QK_K; j += 16) {
            const int8x16_t vr_s8 = vld1q_s8((const int8_t *) blk->x_real + j);
            const int8x16_t vi_s8 = vld1q_s8((const int8_t *) blk->x_imag + j);

            const int16x8_t vr16_lo = vmovl_s8(vget_low_s8(vr_s8));
            const int16x8_t vr16_hi = vmovl_s8(vget_high_s8(vr_s8));
            const int16x8_t vi16_lo = vmovl_s8(vget_low_s8(vi_s8));
            const int16x8_t vi16_hi = vmovl_s8(vget_high_s8(vi_s8));

            float32x4_t vr_f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vr16_lo))), vd_r);
            float32x4_t vr_f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vr16_lo))), vd_r);
            float32x4_t vr_f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vr16_hi))), vd_r);
            float32x4_t vr_f3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vr16_hi))), vd_r);

            float32x4_t vi_f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vi16_lo))), vd_i);
            float32x4_t vi_f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vi16_lo))), vd_i);
            float32x4_t vi_f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vi16_hi))), vd_i);
            float32x4_t vi_f3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vi16_hi))), vd_i);

            max_r_q = vmaxq_f32(max_r_q, vabsq_f32(vr_f0));
            max_r_q = vmaxq_f32(max_r_q, vabsq_f32(vr_f1));
            max_r_q = vmaxq_f32(max_r_q, vabsq_f32(vr_f2));
            max_r_q = vmaxq_f32(max_r_q, vabsq_f32(vr_f3));

            max_i_q = vmaxq_f32(max_i_q, vabsq_f32(vi_f0));
            max_i_q = vmaxq_f32(max_i_q, vabsq_f32(vi_f1));
            max_i_q = vmaxq_f32(max_i_q, vabsq_f32(vi_f2));
            max_i_q = vmaxq_f32(max_i_q, vabsq_f32(vi_f3));
        }
#else
        for (int j = 0; j < QK_K; ++j) {
            const float vr = (float) ((int8_t) blk->x_real[j]) * d_r;
            const float vi = (float) ((int8_t) blk->x_imag[j]) * d_i;
            max_r = MAX(max_r, ggml_ifairy_abs_f32(vr));
            max_i = MAX(max_i, ggml_ifairy_abs_f32(vi));
        }
#endif
    }

#if defined(__ARM_NEON)
    max_r = vmaxvq_f32(max_r_q);
    max_i = vmaxvq_f32(max_i_q);
#endif

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
    const float inv_r = inv_scales[0];
    const float inv_i = inv_scales[1];

    const int64_t pairs_per_block = QK_K / 2;

#if defined(__ARM_NEON)
    for (int64_t bi = 0; bi < n_blocks; ++bi) {
        const block_ifairy_q16 * blk = &act_blocks[bi];
        const float scale_r = GGML_FP16_TO_FP32(blk->d_real) * inv_r;
        const float scale_i = GGML_FP16_TO_FP32(blk->d_imag) * inv_i;

        const int64_t block_pair_base = bi * pairs_per_block;
        for (int pair_chunk = 0; pair_chunk < pairs_per_block; pair_chunk += 8) {
            int16x8_t even_r_q, odd_r_q;
            int16x8_t even_i_q, odd_i_q;
            ggml_ifairy_quant8_pair((const int8_t *) blk->x_real + pair_chunk * 2, scale_r, &even_r_q, &odd_r_q);
            ggml_ifairy_quant8_pair((const int8_t *) blk->x_imag + pair_chunk * 2, scale_i, &even_i_q, &odd_i_q);

            int16_t even_r_arr[8], odd_r_arr[8], even_i_arr[8], odd_i_arr[8];
            vst1q_s16(even_r_arr, even_r_q);
            vst1q_s16(odd_r_arr,  odd_r_q);
            vst1q_s16(even_i_arr, even_i_q);
            vst1q_s16(odd_i_arr,  odd_i_q);

            for (int idx = 0; idx < 8; ++idx) {
                const int8_t qr_even = (int8_t) even_r_arr[idx];
                const int8_t qr_odd  = (int8_t) odd_r_arr[idx];
                const int8_t qi_even = (int8_t) even_i_arr[idx];
                const int8_t qi_odd  = (int8_t) odd_i_arr[idx];

                const size_t pair_index = (size_t) block_pair_base + (size_t) pair_chunk + (size_t) idx;
                const size_t pair_offset = pair_index * 32;

                ggml_ifairy_fill_pair_tables(qlut_r + pair_offset, qr_even, qr_odd);
                ggml_ifairy_fill_pair_tables(qlut_i + pair_offset, qi_even, qi_odd);
            }
        }
    }
#else
    for (int64_t bi = 0; bi < n_blocks; ++bi) {
        const block_ifairy_q16 * blk = &act_blocks[bi];
        const float scale_r = GGML_FP16_TO_FP32(blk->d_real) * inv_r;
        const float scale_i = GGML_FP16_TO_FP32(blk->d_imag) * inv_i;

        const int64_t block_pair_base = bi * pairs_per_block;
        for (int pair = 0; pair < pairs_per_block; ++pair) {
            const int j0 = pair * 2;
            const int j1 = j0 + 1;

            const float vr0 = (float) ((int8_t) blk->x_real[j0]) * scale_r;
            const float vr1 = (float) ((int8_t) blk->x_real[j1]) * scale_r;
            const float vi0 = (float) ((int8_t) blk->x_imag[j0]) * scale_i;
            const float vi1 = (float) ((int8_t) blk->x_imag[j1]) * scale_i;

            const int8_t qr0 = ggml_ifairy_clamp_s8(vr0);
            const int8_t qr1 = ggml_ifairy_clamp_s8(vr1);
            const int8_t qi0 = ggml_ifairy_clamp_s8(vi0);
            const int8_t qi1 = ggml_ifairy_clamp_s8(vi1);

            const int64_t pair_offset = (block_pair_base + pair) * 32;
            ggml_ifairy_fill_pair_tables(qlut_r + pair_offset, qr0, qr1);
            ggml_ifairy_fill_pair_tables(qlut_i + pair_offset, qi0, qi1);
        }
    }
#endif
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
        GGML_IFAIRY_DEBUG("ifairy_lut: disabled by env\n");
        return false;
    }

    const bool act_q16 = src1->type == GGML_TYPE_IFAIRY_Q16;
    const bool act_f32 = src1->type == GGML_TYPE_F32;
    if (src0->type != GGML_TYPE_IFAIRY || (!act_q16 && !act_f32) || dst->type != GGML_TYPE_F32) {
        GGML_IFAIRY_DEBUG("ifairy_lut: type mismatch src0=%s src1=%s dst=%s\n", ggml_type_name(src0->type), ggml_type_name(src1->type), ggml_type_name(dst->type));
        return false;
    }

    // only support single-column matvec for now
    if (src1->ne[1] != 1) {
        GGML_IFAIRY_DEBUG("ifairy_lut: skip because src1->ne[1]=%" PRId64 "\n", src1->ne[1]);
        return false;
    }

    if (src0->ne[0] != src1->ne[0]) {
        GGML_IFAIRY_DEBUG("ifairy_lut: skip because k mismatch src0=%" PRId64 " src1=%" PRId64 "\n", src0->ne[0], src1->ne[0]);
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t m = src0->ne[1];
    GGML_UNUSED(act_f32);

    if ((k % QK_K) != 0) {
        GGML_IFAIRY_DEBUG("ifairy_lut: skip because k %% QK_K != 0 (k=%" PRId64 ", QK_K=%d)\n", k, QK_K);
        return false;
    }

    int bm = 0, bk = 0;
    if (!ggml_ifairy_select_tile_params(m, k, &bm, &bk)) {
        GGML_IFAIRY_DEBUG("ifairy_lut: skip because shape (m=%" PRId64 ", k=%" PRId64 ") not supported\n", m, k);
        return false;
    }

    GGML_IFAIRY_DEBUG("ifairy_lut: enable LUT path for m=%" PRId64 " k=%" PRId64 " bm=%d bk=%d\n", m, k, bm, bk);

    return true;
}

size_t ggml_ifairy_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    GGML_UNUSED(dst);
    if (!ggml_ifairy_can_mul_mat(src0, src1, dst)) {
        return 0;
    }

    const int64_t k = src0->ne[0];
    const int64_t m = src0->ne[1];
    const int64_t n = src1->ne[1];
    const bool needs_q16 = src1->type == GGML_TYPE_F32;
    int bm = 0, bk = 0;
    const bool ok = ggml_ifairy_select_tile_params(m, k, &bm, &bk);
    GGML_ASSERT(ok);
    GGML_UNUSED(bk);

    GGML_ASSERT((k % 2) == 0);

    const size_t act_q16_bytes      = needs_q16 ? ggml_ifairy_act_q16_bytes(k, n) : 0;
    const size_t qlut_bytes_per_col = 2 * ggml_ifairy_qlut_bytes(k);
    const size_t packed_bytes       = ggml_ifairy_packed_bytes(k);
    const size_t lut_scales_bytes   = 2 * sizeof(float);

    size_t wsize = act_q16_bytes + n * (qlut_bytes_per_col + packed_bytes + lut_scales_bytes);

    // 64B align to match ggml allocator expectations
    wsize = GGML_PAD(wsize, 64);
    return wsize;
}

void ggml_ifairy_preprocessor(int m, int k, const void * B, void * lut_scales, void * qlut_real, void * qlut_imag) {
    GGML_UNUSED(m);
    GGML_ASSERT(k % QK_K == 0);

    const block_ifairy_q16 * act_blocks = (const block_ifairy_q16 *) B;
    const size_t qlut_bytes = ggml_ifairy_qlut_bytes(k);
    const int64_t pairs_total = k / 2;

    int8_t * ar_pack = (int8_t *) qlut_imag + qlut_bytes; // k bytes: [even0..7 | odd0..7] per 8 pairs
    int8_t * ai_pack = ar_pack + pairs_total * 2;         // k bytes

    float inv_scales[2] = {0.0f, 0.0f};

    ggml_ifairy_per_tensor_quant(act_blocks, k, (float *) lut_scales, inv_scales);
    ggml_ifairy_lut_ctor(act_blocks, k, inv_scales, (int8_t *) qlut_real, (int8_t *) qlut_imag);

    // compact view: store even/odd for 8 pairs into one 16B pack to avoid vcombine in the kernel
    for (int64_t pair = 0; pair < pairs_total; pair += 8) {
        const size_t pack_base_bytes = (size_t) pair * 2; // 16 bytes per 8 pairs
        for (int idx = 0; idx < 8; ++idx) {
            const size_t qbase = ((size_t) pair + (size_t) idx) * 32;
            const int8_t ar_even = ((int8_t *) qlut_real)[qbase + 0];
            const int8_t ar_odd  = ((int8_t *) qlut_real)[qbase + 16];
            const int8_t ai_even = ((int8_t *) qlut_imag)[qbase + 0];
            const int8_t ai_odd  = ((int8_t *) qlut_imag)[qbase + 16];

            const size_t pack_off = pack_base_bytes + (size_t) idx;
            ar_pack[pack_off]       = ar_even;
            ar_pack[pack_off + 8]   = ar_odd;
            ai_pack[pack_off]       = ai_even;
            ai_pack[pack_off + 8]   = ai_odd;
        }
    }
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

static inline uint8_t ggml_ifairy_weight_code(int8_t wr, int8_t wi) {
    if (wr == -1) {
        return 0;
    }
    if (wr == 1) {
        return 1;
    }
    if (wi == -1) {
        return 2;
    }
    return 3;
}

static inline int8_t ggml_ifairy_qlut_lookup(const int8_t * qlut, int64_t pair_index, bool is_odd, uint8_t code) {
    const int64_t base = pair_index * 32 + (is_odd ? 16 : 0);
    return qlut[base + (code & 0xF)];
}

// Reference LUT matvec (single-column) using decoded QLUT and per-tensor scales.
// Output layout: dst[2*i + 0] = real, dst[2*i + 1] = imag for row i.
void ggml_ifairy_qgemm_lut_ref_slice(const void * w, const int8_t * qlut_r, const int8_t * qlut_i, const float * lut_scales, int64_t k, int64_t row_start, int64_t row_end, float * dst) {
    const block_ifairy * w_blocks = (const block_ifairy *) w;
    GGML_ASSERT(k % QK_K == 0);

    const int64_t blocks_per_row = k / QK_K;
    const float inv_lut_r = lut_scales[0] != 0.0f ? 1.0f / lut_scales[0] : 0.0f;
    const float inv_lut_i = lut_scales[1] != 0.0f ? 1.0f / lut_scales[1] : 0.0f;
    const int64_t pairs_per_block = QK_K / 2;

    int8_t wr_buf[QK_K];
    int8_t wi_buf[QK_K];

    for (int64_t row = row_start; row < row_end; ++row) {
        float acc_rr = 0.0f;
        float acc_ii = 0.0f;
        float acc_ri = 0.0f;
        float acc_ir = 0.0f;

        const block_ifairy * row_w = w_blocks + row * blocks_per_row;

        for (int64_t b = 0; b < blocks_per_row; ++b) {
            ggml_ifairy_decode_weight_block(&row_w[b], wr_buf, wi_buf);

            const int64_t block_pair_base = b * pairs_per_block;
            for (int pair = 0; pair < pairs_per_block; ++pair) {
                const int j0 = pair * 2;
                const int j1 = j0 + 1;

                const int8_t wr0 = wr_buf[j0];
                const int8_t wi0 = wi_buf[j0];
                const int8_t wr1 = wr_buf[j1];
                const int8_t wi1 = wi_buf[j1];

                const uint8_t code0 = ggml_ifairy_weight_code(wr0, wi0);
                const uint8_t code1 = ggml_ifairy_weight_code(wr1, wi1);

                const int64_t pair_index = block_pair_base + pair;
                const int8_t ar0 = ggml_ifairy_qlut_lookup(qlut_r, pair_index, false, code0);
                const int8_t ar1 = ggml_ifairy_qlut_lookup(qlut_r, pair_index, true,  code1);
                const int8_t ai0 = ggml_ifairy_qlut_lookup(qlut_i, pair_index, false, code0);
                const int8_t ai1 = ggml_ifairy_qlut_lookup(qlut_i, pair_index, true,  code1);

                acc_rr += (float) wr0 * (float) ar0 + (float) wr1 * (float) ar1;
                acc_ii += (float) wi0 * (float) ai0 + (float) wi1 * (float) ai1;
                acc_ri += (float) wr0 * (float) ai0 + (float) wr1 * (float) ai1;
                acc_ir += (float) wi0 * (float) ar0 + (float) wi1 * (float) ar1;
            }
        }

        const float w_r = GGML_FP16_TO_FP32(row_w[0].d_real);
        const float w_i = GGML_FP16_TO_FP32(row_w[0].d_imag);

        const float scale_wr_r = w_r * inv_lut_r; // matches ar scale (real)
        const float scale_wi_i = w_i * inv_lut_i; // matches ai scale (imag)
        const float scale_wi_r = w_i * inv_lut_r; // matches ar scale for wi
        const float scale_wr_i = w_r * inv_lut_i; // matches ai scale for wr

        const float out_real = scale_wr_r * acc_rr + scale_wi_i * acc_ii;
        const float out_imag = scale_wi_r * acc_ir - scale_wr_i * acc_ri;

        float * dst_row = dst + row;
        ((ggml_bf16_t *) dst_row)[0] = GGML_FP32_TO_BF16(out_real);
        ((ggml_bf16_t *) dst_row)[1] = GGML_FP32_TO_BF16(out_imag);
    }
}

void ggml_ifairy_qgemm_lut_ref(const void * w, const int8_t * qlut_r, const int8_t * qlut_i, const float * lut_scales, int64_t k, int64_t m, float * dst) {
    ggml_ifairy_qgemm_lut_ref_slice(w, qlut_r, qlut_i, lut_scales, k, 0, m, dst);
}

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static const int8_t IFARY_WR_TBL[16] = { -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0 };
static const int8_t IFARY_WI_TBL[16] = { 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1 };

static inline void ggml_ifairy_unpack_block_codes(
        const block_ifairy * blk,
        int8_t * wr_packed,
        int8_t * wi_packed,
        const uint8x16_t mask2,
        const uint8x16_t idx_pack,
        const int8x16_t wr_tbl,
        const int8x16_t wi_tbl) {
    for (int chunk = 0; chunk < 4; ++chunk) {
        const uint8x16_t packed = vld1q_u8(blk->qs + chunk * 16);
        uint8x16_t shifted = packed;

        for (int part = 0; part < 4; ++part) {
            const uint8x16_t codes = vandq_u8(shifted, mask2);
            const int8x16_t wr_all = vqtbl1q_s8(wr_tbl, codes);
            const int8x16_t wi_all = vqtbl1q_s8(wi_tbl, codes);

            const int8x16_t wr_pack = vqtbl1q_s8(wr_all, idx_pack); // [even0..7 | odd0..7]
            const int8x16_t wi_pack = vqtbl1q_s8(wi_all, idx_pack);

            const size_t pair_base = (size_t) (chunk * 32 + part * 8) * 2; // 2 bytes per pair
            vst1q_s8(wr_packed + pair_base, wr_pack);
            vst1q_s8(wi_packed + pair_base, wi_pack);

            shifted = vshrq_n_u8(shifted, 2);
        }
    }
}

// NEON + DOTPROD matvec over packed activation tables ([even|odd] per 8 pairs)
void ggml_ifairy_qgemm_lut_neon_slice(const void * w, const int8_t * qlut_r, const int8_t * qlut_i, const float * lut_scales, int64_t k, int64_t row_start, int64_t row_end, float * dst) {
    GGML_ASSERT(k % QK_K == 0);
    const block_ifairy * w_blocks = (const block_ifairy *) w;

    GGML_UNUSED(qlut_r);

    const size_t qlut_bytes      = ggml_ifairy_qlut_bytes(k);
    const int64_t pairs_total    = k / 2;
    const int64_t pairs_per_block = QK_K / 2;
    const int64_t blocks_per_row = k / QK_K;

    const int8_t * ar_pack = qlut_i + qlut_bytes;           // k bytes, [even0..7 | odd0..7] per 8 pairs
    const int8_t * ai_pack = ar_pack + pairs_total * 2;     // k bytes, same layout for imag

    const float inv_lut_r = lut_scales[0] != 0.0f ? 1.0f / lut_scales[0] : 0.0f;
    const float inv_lut_i = lut_scales[1] != 0.0f ? 1.0f / lut_scales[1] : 0.0f;

    const uint8x16_t mask2 = vdupq_n_u8(0x3);
    const uint8x16_t idx_pack = { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 };
    const int8x16_t wr_tbl = vld1q_s8(IFARY_WR_TBL);
    const int8x16_t wi_tbl = vld1q_s8(IFARY_WI_TBL);

    for (int64_t row = row_start; row < row_end; ++row) {
        int32x4_t acc_rr0 = vdupq_n_s32(0);
        int32x4_t acc_ii0 = vdupq_n_s32(0);
        int32x4_t acc_ri0 = vdupq_n_s32(0);
        int32x4_t acc_ir0 = vdupq_n_s32(0);

        int32x4_t acc_rr1 = vdupq_n_s32(0);
        int32x4_t acc_ii1 = vdupq_n_s32(0);
        int32x4_t acc_ri1 = vdupq_n_s32(0);
        int32x4_t acc_ir1 = vdupq_n_s32(0);

        const block_ifairy * row_w = w_blocks + row * blocks_per_row;

        for (int64_t b = 0; b < blocks_per_row; ++b) {
            const size_t pair_base      = (size_t) b * pairs_per_block;
            const size_t pack_base_bytes = pair_base * 2;
            if (b + 1 < blocks_per_row) {
                __builtin_prefetch(row_w + b + 1, 0, 1);
            }
            __builtin_prefetch(ar_pack + pack_base_bytes + 64, 0, 1);
            __builtin_prefetch(ai_pack + pack_base_bytes + 64, 0, 1);

            for (int chunk = 0; chunk < 4; chunk += 2) {
                const uint8x16_t packed0 = vld1q_u8(row_w[b].qs + chunk * 16);
                const uint8x16_t packed1 = vld1q_u8(row_w[b].qs + (chunk + 1) * 16);

                uint8x16_t shift0 = packed0;
                uint8x16_t shift1 = packed1;

                const size_t base_bytes0 = pack_base_bytes + (size_t) chunk * 64;       // chunk*32 pairs * 2
                const size_t base_bytes1 = base_bytes0 + 64;                            // next 32 pairs

                __builtin_prefetch(ar_pack + base_bytes1 + 128, 0, 1);
                __builtin_prefetch(ai_pack + base_bytes1 + 128, 0, 1);

                for (int part = 0; part < 4; part += 2) {
                    // chunk0 part0
                    const size_t off_bytes0 = base_bytes0 + (size_t) part * 16;
                    const uint8x16_t codes0 = vandq_u8(shift0, mask2);
                    const int8x16_t wr_pack0 = vqtbl1q_s8(vqtbl1q_s8(wr_tbl, codes0), idx_pack);
                    const int8x16_t wi_pack0 = vqtbl1q_s8(vqtbl1q_s8(wi_tbl, codes0), idx_pack);
                    const int8x16_t ar_pack0 = vld1q_s8(ar_pack + off_bytes0);
                    const int8x16_t ai_pack0 = vld1q_s8(ai_pack + off_bytes0);
                    acc_rr0 = vdotq_s32(acc_rr0, wr_pack0, ar_pack0);
                    acc_ii0 = vdotq_s32(acc_ii0, wi_pack0, ai_pack0);
                    acc_ri0 = vdotq_s32(acc_ri0, wr_pack0, ai_pack0);
                    acc_ir0 = vdotq_s32(acc_ir0, wi_pack0, ar_pack0);
                    shift0 = vshrq_n_u8(shift0, 2);

                    // chunk0 part1
                    const size_t off_bytes1p = off_bytes0 + 16;
                    const uint8x16_t codes1 = vandq_u8(shift0, mask2);
                    const int8x16_t wr_pack1 = vqtbl1q_s8(vqtbl1q_s8(wr_tbl, codes1), idx_pack);
                    const int8x16_t wi_pack1 = vqtbl1q_s8(vqtbl1q_s8(wi_tbl, codes1), idx_pack);
                    const int8x16_t ar_pack1 = vld1q_s8(ar_pack + off_bytes1p);
                    const int8x16_t ai_pack1 = vld1q_s8(ai_pack + off_bytes1p);
                    acc_rr1 = vdotq_s32(acc_rr1, wr_pack1, ar_pack1);
                    acc_ii1 = vdotq_s32(acc_ii1, wi_pack1, ai_pack1);
                    acc_ri1 = vdotq_s32(acc_ri1, wr_pack1, ai_pack1);
                    acc_ir1 = vdotq_s32(acc_ir1, wi_pack1, ar_pack1);
                    shift0 = vshrq_n_u8(shift0, 2);

                    // chunk1 part0
                    const size_t off_bytes2 = base_bytes1 + (size_t) part * 16;
                    const uint8x16_t codes2 = vandq_u8(shift1, mask2);
                    const int8x16_t wr_pack2 = vqtbl1q_s8(vqtbl1q_s8(wr_tbl, codes2), idx_pack);
                    const int8x16_t wi_pack2 = vqtbl1q_s8(vqtbl1q_s8(wi_tbl, codes2), idx_pack);
                    const int8x16_t ar_pack2 = vld1q_s8(ar_pack + off_bytes2);
                    const int8x16_t ai_pack2 = vld1q_s8(ai_pack + off_bytes2);
                    acc_rr0 = vdotq_s32(acc_rr0, wr_pack2, ar_pack2);
                    acc_ii0 = vdotq_s32(acc_ii0, wi_pack2, ai_pack2);
                    acc_ri0 = vdotq_s32(acc_ri0, wr_pack2, ai_pack2);
                    acc_ir0 = vdotq_s32(acc_ir0, wi_pack2, ar_pack2);
                    shift1 = vshrq_n_u8(shift1, 2);

                    // chunk1 part1
                    const size_t off_bytes3 = off_bytes2 + 16;
                    const uint8x16_t codes3 = vandq_u8(shift1, mask2);
                    const int8x16_t wr_pack3 = vqtbl1q_s8(vqtbl1q_s8(wr_tbl, codes3), idx_pack);
                    const int8x16_t wi_pack3 = vqtbl1q_s8(vqtbl1q_s8(wi_tbl, codes3), idx_pack);
                    const int8x16_t ar_pack3 = vld1q_s8(ar_pack + off_bytes3);
                    const int8x16_t ai_pack3 = vld1q_s8(ai_pack + off_bytes3);
                    acc_rr1 = vdotq_s32(acc_rr1, wr_pack3, ar_pack3);
                    acc_ii1 = vdotq_s32(acc_ii1, wi_pack3, ai_pack3);
                    acc_ri1 = vdotq_s32(acc_ri1, wr_pack3, ai_pack3);
                    acc_ir1 = vdotq_s32(acc_ir1, wi_pack3, ar_pack3);
                    shift1 = vshrq_n_u8(shift1, 2);
                }
            }
        }

        const int32x4_t acc_rr_v = vaddq_s32(acc_rr0, acc_rr1);
        const int32x4_t acc_ii_v = vaddq_s32(acc_ii0, acc_ii1);
        const int32x4_t acc_ri_v = vaddq_s32(acc_ri0, acc_ri1);
        const int32x4_t acc_ir_v = vaddq_s32(acc_ir0, acc_ir1);

        const int32_t acc_rr = vaddvq_s32(acc_rr_v);
        const int32_t acc_ii = vaddvq_s32(acc_ii_v);
        const int32_t acc_ri = vaddvq_s32(acc_ri_v);
        const int32_t acc_ir = vaddvq_s32(acc_ir_v);

        const float w_r = GGML_FP16_TO_FP32(row_w[0].d_real);
        const float w_i = GGML_FP16_TO_FP32(row_w[0].d_imag);

        const float scale_wr_r = w_r * inv_lut_r;
        const float scale_wi_i = w_i * inv_lut_i;
        const float scale_wi_r = w_i * inv_lut_r;
        const float scale_wr_i = w_r * inv_lut_i;

        float * dst_row = dst + row;
        ((ggml_bf16_t *) dst_row)[0] = GGML_FP32_TO_BF16(scale_wr_r * (float) acc_rr + scale_wi_i * (float) acc_ii);
        ((ggml_bf16_t *) dst_row)[1] = GGML_FP32_TO_BF16(scale_wi_r * (float) acc_ir - scale_wr_i * (float) acc_ri);
    }
}

void ggml_ifairy_qgemm_lut_neon(const void * w, const int8_t * qlut_r, const int8_t * qlut_i, const float * lut_scales, int64_t k, int64_t m, float * dst) {
    ggml_ifairy_qgemm_lut_neon_slice(w, qlut_r, qlut_i, lut_scales, k, 0, m, dst);
}
#endif
