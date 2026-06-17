#pragma once

#include "ggml-cpu-impl.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

bool   ggml_fairy2i_cpu_supports_op(const struct ggml_tensor * dst);
int    ggml_fairy2i_cpu_n_tasks(const struct ggml_tensor * dst, int n_threads);
size_t ggml_fairy2i_cpu_work_size(const struct ggml_tensor * dst, int n_tasks);
void   ggml_fairy2i_cpu_prepare_graph(const struct ggml_cgraph * cgraph);
void   ggml_fairy2i_cpu_free(void);
bool   ggml_fairy2i_cpu_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst);
bool   ggml_fairy2i_cpu_try_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
