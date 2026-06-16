#pragma once

#include "ggml-cpu-impl.h"

#ifdef __cplusplus
extern "C" {
#endif

void ggml_compute_forward_ifairy_wide_linear_w2(const struct ggml_compute_params * params,
                                                 struct ggml_tensor *                dst);

bool   ggml_compute_forward_ifairy_wide_linear_w2_lut(const struct ggml_compute_params * params,
                                                       struct ggml_tensor *                dst,
                                                       bool                                lut_c);
size_t ggml_ifairy_wide_linear_w2_lut_wsize(const struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
