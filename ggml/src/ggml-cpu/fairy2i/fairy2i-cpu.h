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

// Reserved extension hook for a future CPU path that may consume
// GGML_OP_MUL_MAT with src0 GGML_TYPE_FAIRY2I_TILE64_V2, src1/dst F32,
// contiguous no-view tensors, and K aligned to QK_FAIRY2I_TILE64.
// The current Fairy2i CPU production path is GGML_OP_FAIRY2I_WIDE_LINEAR_W2,
// so this hook intentionally returns false until that MUL_MAT contract is
// implemented and tested.
bool   ggml_fairy2i_cpu_try_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
