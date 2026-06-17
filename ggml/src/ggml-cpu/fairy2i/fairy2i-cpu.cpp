#include "fairy2i-cpu.h"
#include "wide-linear.h"

#include "ggml.h"
#include "ggml-impl.h"

#ifdef GGML_USE_FAIRY2I_CPU_LUT
#    include "ggml-fairy2i-lut.h"
#endif

#include <string.h>

static constexpr size_t GGML_FAIRY2I_CPU_CACHE_LINE = 64;

#ifdef GGML_USE_FAIRY2I_CPU_LUT
enum ggml_fairy2i_lut_impl {
    GGML_FAIRY2I_LUT_IMPL_AUTO  = 0,
    GGML_FAIRY2I_LUT_IMPL_LUT16 = 1,
    GGML_FAIRY2I_LUT_IMPL_LUT_C = 2,
};

struct ggml_fairy2i_lut_config {
    bool                      dbg;
    bool                      lut_enabled;
    bool                      lut_explicit;
    enum ggml_fairy2i_lut_impl impl;
};

static enum ggml_fairy2i_lut_impl ggml_fairy2i_lut_impl_from_env(const char * env_name,
                                                                  bool         dbg,
                                                                  const char * log_prefix) {
    enum ggml_fairy2i_lut_impl impl = GGML_FAIRY2I_LUT_IMPL_AUTO;
    const char * impl_env           = getenv(env_name);
    if (impl_env && impl_env[0] != '\0' && strcmp(impl_env, "0") != 0 && strcmp(impl_env, "auto") != 0) {
        if (strcmp(impl_env, "lut16") == 0) {
            impl = GGML_FAIRY2I_LUT_IMPL_LUT16;
        } else if (strcmp(impl_env, "lut_c") == 0) {
            impl = GGML_FAIRY2I_LUT_IMPL_LUT_C;
        } else if (dbg) {
            GGML_LOG_WARN("%s: unknown %s=%s (expected auto|lut16|lut_c)\n", log_prefix, env_name, impl_env);
        }
    }
    return impl;
}

static struct ggml_fairy2i_lut_config ggml_fairy2i_lut_config_from_env(void) {
    struct ggml_fairy2i_lut_config cfg;

    cfg.dbg = ggml_fairy2i_env_enabled("GGML_FAIRY2I_LUT_DEBUG");

    const char * enabled_env = getenv("GGML_FAIRY2I_LUT");
    cfg.lut_enabled          = !(enabled_env && strcmp(enabled_env, "0") == 0);
    cfg.lut_explicit         = enabled_env && strcmp(enabled_env, "0") != 0;
    cfg.impl                 = ggml_fairy2i_lut_impl_from_env("GGML_FAIRY2I_LUT_IMPL", cfg.dbg, "fairy2i_lut");

    return cfg;
}
#endif

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

void ggml_fairy2i_cpu_prepare_graph(const struct ggml_cgraph * cgraph) {
#ifdef GGML_USE_FAIRY2I_CPU_LUT
    const struct ggml_fairy2i_lut_config cfg = ggml_fairy2i_lut_config_from_env();
    if (!cfg.lut_explicit) {
        return;
    }

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (!node || node->op != GGML_OP_FAIRY2I_WIDE_LINEAR_W2) {
            continue;
        }

        for (int src = 1; src <= 4; ++src) {
            struct ggml_tensor * weight = node->src[src];
            const struct fairy2i_lut_extra * extra = weight ? (const struct fairy2i_lut_extra *) weight->extra : nullptr;
            if (weight && (!extra || !extra->packed_w)) {
                ggml_fairy2i_lut_transform_tensor(weight, nullptr);
            }
        }
    }
#else
    GGML_UNUSED(cgraph);
#endif
}

void ggml_fairy2i_cpu_free(void) {
#ifdef GGML_USE_FAIRY2I_CPU_LUT
    ggml_fairy2i_lut_free();
#endif
}

static bool ggml_fairy2i_cpu_compute_wide_linear_w2(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (!ggml_fairy2i_cpu_supports_op(dst)) {
        return false;
    }

#ifdef GGML_USE_FAIRY2I_CPU_LUT
    const struct ggml_fairy2i_lut_config cfg = ggml_fairy2i_lut_config_from_env();
    const bool use_lut = cfg.lut_enabled && cfg.lut_explicit;
    const bool lut_c   = cfg.impl == GGML_FAIRY2I_LUT_IMPL_LUT_C;
    if (use_lut && ggml_fairy2i_wide_linear_w2_compute_lut(params, dst, lut_c)) {
        return true;
    }
#endif

    ggml_fairy2i_wide_linear_w2_compute(params, dst);
    return true;
}

bool ggml_fairy2i_cpu_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    return ggml_fairy2i_cpu_compute_wide_linear_w2(params, dst);
}

bool ggml_fairy2i_cpu_try_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    // See fairy2i-cpu.h for the reserved MUL_MAT contract. Returning false
    // keeps generic CPU matmul semantics unchanged until Fairy2i tile64 MUL_MAT
    // is implemented as an explicit CPU extension path.
    GGML_UNUSED(params);
    GGML_UNUSED(dst);
    return false;
}
