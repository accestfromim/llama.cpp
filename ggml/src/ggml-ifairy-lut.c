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

static bool ifairy_lut_initialized = false;
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

void ggml_ifairy_transform_tensor(struct ggml_tensor * tensor) {
#if defined(GGML_USE_OPENMP)
#pragma omp critical
#endif
    {
        if (tensor->type != GGML_TYPE_IFAIRY || tensor->extra != NULL) {
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
