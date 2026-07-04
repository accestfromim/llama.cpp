#define GGML_COMMON_DECL_CPP
#include "../fairy2i-policy.h"
#include "ggml-backend.h"
#include "ggml-common.h"
#include "ggml-fairy2i-lut.h"
#include "ggml-fairy2i-lut-impl.h"
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

static inline size_t ggml_fairy2i_checked_mul_size(size_t a, size_t b) {
    GGML_ASSERT(a == 0 || b <= SIZE_MAX / a);
    return a * b;
}

static inline size_t ggml_fairy2i_checked_add_size(size_t a, size_t b) {
    GGML_ASSERT(a <= SIZE_MAX - b);
    return a + b;
}

struct ggml_fairy2i_lut_type_info {
    int64_t weight_block_k;
    int64_t act_block_k;
    int64_t groups_per_weight_block;
};

static bool ggml_fairy2i_lut_get_type_info(enum ggml_type type, struct ggml_fairy2i_lut_type_info * info) {
    if (!info) {
        return false;
    }

    switch (type) {
        case GGML_TYPE_FAIRY2I_TILE64_V2:
            info->weight_block_k         = QK_FAIRY2I_TILE64;
            info->act_block_k            = QK_FAIRY2I_ACT_Q16_64;
            info->groups_per_weight_block = QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK;
            return true;
        default:
            return false;
    }
}

static bool ggml_fairy2i_lut_is_fairy2i_type(enum ggml_type type) {
    return type == GGML_TYPE_FAIRY2I_TILE64_V2;
}

bool ggml_fairy2i_lut_can_mul_mat(const struct ggml_tensor * src0,
                                 const struct ggml_tensor * src1,
                                 const struct ggml_tensor * dst) {
    const struct ggml_fairy2i_lut_policy policy = ggml_fairy2i_lut_policy_from_env();
    if (!policy.lut_enabled) {
        if (policy.dbg) {
            GGML_LOG_WARN("fairy2i_lut: disabled by env GGML_FAIRY2I_LUT=0\n");
        }
        return false;
    }

#if !((defined(__aarch64__) && defined(__ARM_NEON)) || defined(__x86_64__) || defined(_M_X64))
    if (policy.dbg) {
        GGML_LOG_WARN("fairy2i_lut: disabled (requires aarch64+NEON or x86_64)\n");
    }
    return false;
#endif

    struct ggml_fairy2i_lut_type_info type_info;
    if (!ggml_fairy2i_lut_get_type_info(src0->type, &type_info) ||
        (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_FAIRY2I_ACT_Q16_64)) {
        if (policy.dbg) {
            GGML_LOG_WARN("fairy2i_lut: type mismatch src0=%s src1=%s dst=%s\n", ggml_type_name(src0->type),
                          ggml_type_name(src1->type), ggml_type_name(dst->type));
        }
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        if (policy.dbg) {
            GGML_LOG_WARN("fairy2i_lut: dst type not F32 (%s)\n", ggml_type_name(dst->type));
        }
        return false;
    }
    // require logical K aligned to block
    if (src0->ne[0] % type_info.weight_block_k != 0 || src0->ne[0] % type_info.act_block_k != 0 ||
        src1->ne[0] != src0->ne[0]) {
        if (policy.dbg) {
            GGML_LOG_WARN("fairy2i_lut: K misaligned type=%s K0=%lld K1=%lld weight_block_k=%lld act_block_k=%lld\n",
                          ggml_type_name(src0->type), (long long) src0->ne[0], (long long) src1->ne[0],
                          (long long) type_info.weight_block_k, (long long) type_info.act_block_k);
        }
        return false;
    }
    if (policy.dbg) {
        GGML_LOG_INFO("fairy2i_lut: can_mul_mat=true\n");
    }
    return true;
}

size_t ggml_fairy2i_lut_get_wsize(const struct ggml_tensor * src0,
                                 const struct ggml_tensor * src1,
                                 const struct ggml_tensor * dst,
                                 int                        n_threads) {
    if (!ggml_fairy2i_lut_can_mul_mat(src0, src1, dst)) {
        return 0;
    }
    (void) n_threads;

    struct ggml_fairy2i_lut_type_info type_info;
    GGML_ASSERT(ggml_fairy2i_lut_get_type_info(src0->type, &type_info));

    const int64_t K              = src0->ne[0];
    const int64_t N              = src1->ne[1];
    const int64_t weight_blocks  = K / type_info.weight_block_k;
    const int64_t act_blocks     = K / type_info.act_block_k;
    const int64_t groups         = weight_blocks * type_info.groups_per_weight_block;

    size_t quant_bytes = 0;
    if (src1->type == GGML_TYPE_F32) {
        const size_t q_elems = ggml_fairy2i_checked_mul_size((size_t) N, (size_t) act_blocks);
        quant_bytes          = GGML_PAD(ggml_fairy2i_checked_mul_size(q_elems, sizeof(block_fairy2i_q16)), 64);
    }

    const size_t lut_groups  = ggml_fairy2i_checked_mul_size((size_t) N, (size_t) groups);
    const size_t lut_bytes   = ggml_fairy2i_checked_mul_size(lut_groups, (size_t) k_fairy2i_lut_group_bytes);
    const size_t scale_bytes = ggml_fairy2i_checked_mul_size(
        ggml_fairy2i_checked_mul_size(ggml_fairy2i_checked_mul_size((size_t) N, (size_t) weight_blocks), 2u),
        sizeof(float));
    const size_t shared_bytes = GGML_PAD(ggml_fairy2i_checked_add_size(lut_bytes, scale_bytes), 64);

    return ggml_fairy2i_checked_add_size(quant_bytes, shared_bytes);
}
