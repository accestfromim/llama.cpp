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
void   ggml_legacy_ifairy_cpu_prepare_graph(const struct ggml_cgraph * cgraph);
bool   ggml_legacy_ifairy_cpu_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst);

bool ggml_legacy_ifairy_cpu_try_quantize_mul_mat_src1(
    const struct ggml_compute_params * params,
    const struct ggml_tensor *         src0,
    const struct ggml_tensor *         src1,
    enum ggml_type                     vec_dot_type,
    char *                             wdata,
    size_t                             nbw1,
    size_t                             nbw2,
    size_t                             nbw3);

#ifdef __cplusplus
}
#endif
