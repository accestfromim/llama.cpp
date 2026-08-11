#pragma once

#include "ggml-cpu-impl.h"

#ifdef __cplusplus
extern "C" {
#endif

bool   ggml_row4_cpu_supports_op(const struct ggml_tensor * dst);
int    ggml_row4_cpu_n_tasks(const struct ggml_tensor * dst, int n_threads);
size_t ggml_row4_cpu_work_size(const struct ggml_tensor * dst, int n_tasks);
bool   ggml_row4_cpu_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
