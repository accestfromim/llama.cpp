#include "legacy-ifairy-cpu.h"
#include "wide-linear.h"

#include "ggml.h"

static constexpr size_t GGML_LEGACY_IFAIRY_CPU_CACHE_LINE = 64;

bool ggml_legacy_ifairy_cpu_supports_op(const struct ggml_tensor * dst) {
    return dst != nullptr && dst->op == GGML_OP_IFAIRY_WIDE_LINEAR_W2;
}

int ggml_legacy_ifairy_cpu_n_tasks(const struct ggml_tensor * dst, int n_threads) {
    if (!ggml_legacy_ifairy_cpu_supports_op(dst)) {
        return 0;
    }
    return n_threads;
}

size_t ggml_legacy_ifairy_cpu_work_size(const struct ggml_tensor * dst, int n_tasks) {
    GGML_UNUSED(n_tasks);

    if (!ggml_legacy_ifairy_cpu_supports_op(dst)) {
        return 0;
    }

    const struct ggml_tensor * x = dst->src[0];
    GGML_ASSERT(x && x->type == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[0] % ggml_blck_size(GGML_TYPE_IFAIRY64) == 0);

    const size_t q_row_size = ggml_row_size(GGML_TYPE_IFAIRY64_Q16, x->ne[0]);
    return GGML_PAD((size_t) ggml_nrows(x) * q_row_size, GGML_LEGACY_IFAIRY_CPU_CACHE_LINE);
}

bool ggml_legacy_ifairy_cpu_compute_wide_linear_w2(
    const struct ggml_compute_params * params,
    struct ggml_tensor *                dst,
    bool                                use_lut,
    bool                                lut_c) {
    GGML_UNUSED(use_lut);
    GGML_UNUSED(lut_c);

    if (!ggml_legacy_ifairy_cpu_supports_op(dst)) {
        return false;
    }

    ggml_compute_forward_ifairy_wide_linear_w2(params, dst);
    return true;
}

bool ggml_legacy_ifairy_cpu_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    return ggml_legacy_ifairy_cpu_compute_wide_linear_w2(params, dst, false, false);
}
