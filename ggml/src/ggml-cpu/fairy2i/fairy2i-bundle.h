#pragma once

#include "ggml-impl.h"
#include "ggml.h"

#include <stdint.h>

struct ggml_fairy2i_bundle_desc {
    const uint8_t *     codes;
    const ggml_fp16_t * scales;
    int64_t             logical_m;
    int64_t             logical_k;
    int64_t             k_blocks;
    int32_t             branches;
};

static inline bool ggml_fairy2i_is_bundle_op(const struct ggml_tensor * dst) {
    return dst && dst->src[1] && dst->src[1]->type == GGML_TYPE_FAIRY2I_BUNDLE_CODES;
}

static inline bool ggml_fairy2i_bundle_desc_init(const struct ggml_tensor *        dst,
                                                 struct ggml_fairy2i_bundle_desc * desc,
                                                 bool                              require_data) {
    if (!ggml_fairy2i_is_bundle_op(dst) || !desc || !dst->src[0] || !dst->src[2]) {
        return false;
    }

    const struct ggml_tensor * x        = dst->src[0];
    const struct ggml_tensor * codes    = dst->src[1];
    const struct ggml_tensor * scales   = dst->src[2];
    const int32_t              layout   = ggml_get_op_params_i32(dst, 0);
    const int64_t              m        = ggml_get_op_params_i32(dst, 1);
    const int64_t              k        = ggml_get_op_params_i32(dst, 2);
    const int32_t              branches = ggml_get_op_params_i32(dst, 3);

    if (layout != 1 || m <= 0 || k <= 0 || m % 64 != 0 || k % 64 != 0 || x->type != GGML_TYPE_F32 || x->ne[0] != k ||
        dst->ne[0] != m || scales->type != GGML_TYPE_F16 || (branches != 2 && branches != 4) ||
        (dst->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W1 ? branches != 2 : branches != 4)) {
        return false;
    }

    const int64_t physical_tiles = (m / 64) * (k / 64);
    if (codes->ne[0] != 16 || codes->ne[1] != branches || codes->ne[2] != 64 || codes->ne[3] != physical_tiles ||
        scales->ne[0] != 2 || scales->ne[1] != branches || scales->ne[2] != physical_tiles || scales->ne[3] != 1 ||
        !ggml_is_contiguous(codes) || !ggml_is_contiguous(scales) ||
        (require_data && (!codes->data || !scales->data))) {
        return false;
    }

    desc->codes     = (const uint8_t *) codes->data;
    desc->scales    = (const ggml_fp16_t *) scales->data;
    desc->logical_m = m;
    desc->logical_k = k;
    desc->k_blocks  = k / 64;
    desc->branches  = branches;
    return true;
}

static inline int64_t ggml_fairy2i_bundle_physical_tile(const struct ggml_fairy2i_bundle_desc * desc,
                                                        int64_t                                 global_m16,
                                                        int64_t                                 k_block) {
    return (global_m16 / 4) * desc->k_blocks + k_block;
}

static inline const uint8_t * ggml_fairy2i_bundle_codes_at(const struct ggml_fairy2i_bundle_desc * desc,
                                                           int64_t                                 global_m16,
                                                           int64_t                                 k_block,
                                                           int                                     branch) {
    const int64_t physical_tile = ggml_fairy2i_bundle_physical_tile(desc, global_m16, k_block);
    const int64_t slot_base     = (global_m16 % 4) * 16;
    return desc->codes + (((physical_tile * 64 + slot_base) * desc->branches + branch) * 16);
}

static inline const ggml_fp16_t * ggml_fairy2i_bundle_scales_at(const struct ggml_fairy2i_bundle_desc * desc,
                                                                int64_t                                 global_m16,
                                                                int64_t                                 k_block,
                                                                int                                     branch) {
    const int64_t physical_tile = ggml_fairy2i_bundle_physical_tile(desc, global_m16, k_block);
    return desc->scales + (physical_tile * desc->branches + branch) * 2;
}
