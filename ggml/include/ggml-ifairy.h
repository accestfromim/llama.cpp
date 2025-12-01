// Copyright (c) 2024 The ggml authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_IFAIRY_MAX_NODES 8192

struct ggml_ifairy_tensor_extra {
    int      lut_scales_size;
    int      n_tile_num;
    int      bk;
    int      bm;
    size_t   tile_stride;
    size_t   c_tile_size;
    size_t   scales_bytes;
    uint8_t * qweights;
    float   * scales; // pairs of {d_real, d_imag} per block
};

GGML_API void ggml_ifairy_lut_init(void);
GGML_API void ggml_ifairy_lut_free(void);
GGML_API void ggml_ifairy_transform_tensor(struct ggml_tensor * tensor);
GGML_API bool ggml_ifairy_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);
GGML_API size_t ggml_ifairy_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);
GGML_API void ggml_ifairy_preprocessor(int m, int k, const void * B, void * lut_scales, void * qlut_real, void * qlut_imag);
GGML_API void ggml_ifairy_qgemm_lut_ref(const void * w, const int8_t * qlut_r, const int8_t * qlut_i, const float * lut_scales, int64_t k, int64_t m, float * dst);
GGML_API void ggml_ifairy_qgemm_lut_ref_slice(const void * w, const int8_t * qlut_r, const int8_t * qlut_i, const float * lut_scales, int64_t k, int64_t row_start, int64_t row_end, float * dst);
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
GGML_API void ggml_ifairy_qgemm_lut_neon(const void * w, const int8_t * qlut_r, const int8_t * qlut_i, const float * lut_scales, int64_t k, int64_t m, float * dst);
GGML_API void ggml_ifairy_qgemm_lut_neon_slice(const void * w, const int8_t * qlut_r, const int8_t * qlut_i, const float * lut_scales, int64_t k, int64_t row_start, int64_t row_end, float * dst);
#endif

#ifdef __cplusplus
}
#endif
