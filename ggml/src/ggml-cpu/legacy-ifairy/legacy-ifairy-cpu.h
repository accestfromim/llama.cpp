#pragma once

#include "ggml-cpu-impl.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

bool   ggml_legacy_ifairy_cpu_supports_op(const struct ggml_tensor * dst);
int    ggml_legacy_ifairy_cpu_n_tasks(const struct ggml_tensor * dst, int n_threads);
size_t ggml_legacy_ifairy_cpu_work_size(const struct ggml_tensor * dst, int n_tasks);
bool   ggml_legacy_ifairy_cpu_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst);

bool ggml_legacy_ifairy_cpu_compute_wide_linear_w2(
    const struct ggml_compute_params * params,
    struct ggml_tensor *                dst,
    bool                                use_lut,
    bool                                lut_c);

#ifdef __cplusplus
}
#endif
