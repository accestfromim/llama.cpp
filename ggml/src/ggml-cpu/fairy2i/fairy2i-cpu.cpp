#include "fairy2i-cpu.h"
#include "wide-linear.h"

#include "ggml.h"

static constexpr size_t GGML_FAIRY2I_CPU_CACHE_LINE = 64;

bool ggml_fairy2i_cpu_supports_op(const struct ggml_tensor * dst) {
    return dst != nullptr && dst->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W2;
}

int ggml_fairy2i_cpu_n_tasks(const struct ggml_tensor * dst, int n_threads) {
    if (!ggml_fairy2i_cpu_supports_op(dst)) {
        return 0;
    }
    return n_threads;
}

size_t ggml_fairy2i_cpu_work_size(const struct ggml_tensor * dst, int n_tasks) {
    GGML_UNUSED(n_tasks);

    if (!ggml_fairy2i_cpu_supports_op(dst)) {
        return 0;
    }

    const struct ggml_tensor * x = dst->src[0];
    GGML_ASSERT(x && x->type == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[0] % ggml_blck_size(GGML_TYPE_FAIRY2I_TILE64_V2) == 0);

    const size_t q_row_size = ggml_row_size(GGML_TYPE_FAIRY2I_ACT_Q16_64, x->ne[0]);
    const size_t q_bytes    = GGML_PAD((size_t) ggml_nrows(x) * q_row_size, GGML_FAIRY2I_CPU_CACHE_LINE);

#ifdef GGML_USE_FAIRY2I_CPU_LUT
    const size_t lut_bytes = ggml_fairy2i_wide_linear_w2_lut_wsize(dst);
    return q_bytes > lut_bytes ? q_bytes : lut_bytes;
#else
    return q_bytes;
#endif
}

bool ggml_fairy2i_cpu_compute_wide_linear_w2(
    const struct ggml_compute_params * params,
    struct ggml_tensor *                dst,
    bool                                use_lut,
    bool                                lut_c) {
    if (!ggml_fairy2i_cpu_supports_op(dst)) {
        return false;
    }

#ifdef GGML_USE_FAIRY2I_CPU_LUT
    if (use_lut && ggml_fairy2i_wide_linear_w2_compute_lut(params, dst, lut_c)) {
        return true;
    }
#else
    GGML_UNUSED(use_lut);
    GGML_UNUSED(lut_c);
#endif

    ggml_fairy2i_wide_linear_w2_compute(params, dst);
    return true;
}

bool ggml_fairy2i_cpu_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    return ggml_fairy2i_cpu_compute_wide_linear_w2(params, dst, false, false);
}

bool ggml_fairy2i_cpu_try_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    GGML_UNUSED(params);
    GGML_UNUSED(dst);
    return false;
}
