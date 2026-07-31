#pragma once

#include "ggml-cpu-impl.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_fairy2i_wide_linear_w2_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst);

void ggml_fairy2i_wide_linear_w1_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst);

void ggml_fairy2i_wide_linear_w2_bundle_exact_compute(const struct ggml_compute_params * params,
                                                      struct ggml_tensor *               dst);

bool ggml_fairy2i_wide_linear_w2_compute_lut(const struct ggml_compute_params * params,
                                              struct ggml_tensor *                dst,
                                              bool                                lut_c);
size_t ggml_fairy2i_wide_linear_w2_lut_wsize(const struct ggml_tensor * dst);

bool ggml_fairy2i_wide_linear_w1_compute_lut(const struct ggml_compute_params * params,
                                              struct ggml_tensor *                dst,
                                              bool                                lut_c);
size_t ggml_fairy2i_wide_linear_w1_lut_wsize(const struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
