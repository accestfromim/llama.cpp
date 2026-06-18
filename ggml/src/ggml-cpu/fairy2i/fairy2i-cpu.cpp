#include "fairy2i-cpu.h"
#include "wide-linear.h"

#include "ggml.h"
#include "ggml-impl.h"

#if defined(__aarch64__) && defined(__ARM_NEON)
#    include "arm/fairy2i-quants.h"
#endif

#ifdef GGML_USE_FAIRY2I_CPU_LUT
#    include "ggml-fairy2i-lut.h"
#endif

#include <stdlib.h>
#include <string.h>

static constexpr size_t GGML_FAIRY2I_CPU_CACHE_LINE = 64;

enum ggml_fairy2i_lut_impl {
    GGML_FAIRY2I_LUT_IMPL_AUTO  = 0,
    GGML_FAIRY2I_LUT_IMPL_LUT16 = 1,
    GGML_FAIRY2I_LUT_IMPL_LUT_C = 2,
};

struct ggml_fairy2i_cpu_plan {
    bool                       use_lut;
    bool                       lut_c;
    size_t                     q_bytes;
    size_t                     lut_bytes;
    size_t                     work_size;
    enum ggml_fairy2i_lut_impl impl;
};

#ifdef GGML_USE_FAIRY2I_CPU_LUT

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

static bool ggml_fairy2i_test_require_lut(void) {
    return ggml_fairy2i_env_enabled("GGML_FAIRY2I_TEST_REQUIRE_LUT");
}
#endif

static bool ggml_fairy2i_cpu_debug_enabled(void) {
    const char * env = getenv("GGML_FAIRY2I_CPU_DEBUG");
    return env && strcmp(env, "0") != 0;
}

static bool ggml_fairy2i_test_force_scalar(void) {
    const char * env = getenv("GGML_FAIRY2I_TEST_FORCE_SCALAR");
    return env && strcmp(env, "0") != 0;
}

static const char * ggml_fairy2i_cpu_direct_path_name(void) {
    if (ggml_fairy2i_test_force_scalar()) {
        return "direct_scalar";
    }

#if defined(GGML_USE_FAIRY2I_CPU_AVX512) && defined(__AVX512F__) && defined(__AVX512BW__)
    return "direct_avx512";
#elif defined(__AVX2__)
    return "direct_avx2";
#elif defined(__aarch64__) && defined(__ARM_NEON)
    return ggml_fairy2i_tile64_w2_arm_path_name();
#else
    return "direct_scalar";
#endif
}

static bool ggml_fairy2i_cpu_lut_packed_weights_ready(const struct ggml_tensor * dst) {
#ifdef GGML_USE_FAIRY2I_CPU_LUT
    if (!dst) {
        return false;
    }
    for (int src = 1; src <= 4; ++src) {
        const struct ggml_tensor *       weight = dst->src[src];
        const struct fairy2i_lut_extra * extra  = weight ? (const struct fairy2i_lut_extra *) weight->extra : nullptr;
        if (!extra || !extra->packed_w) {
            return false;
        }
    }
    return true;
#else
    GGML_UNUSED(dst);
    return false;
#endif
}

static void ggml_fairy2i_cpu_debug_log_w2_once(const struct ggml_compute_params *   params,
                                               const struct ggml_tensor *           dst,
                                               const struct ggml_fairy2i_cpu_plan * plan,
                                               const char *                         path) {
    if (!ggml_fairy2i_cpu_debug_enabled() || (params && params->ith != 0)) {
        return;
    }

    static bool logged_direct_scalar  = false;
    static bool logged_direct_avx2    = false;
    static bool logged_direct_avx512  = false;
    static bool logged_direct_neon    = false;
    static bool logged_direct_dotprod = false;
    static bool logged_lut16          = false;
    static bool logged_lut_c          = false;
    static bool logged_unknown        = false;

    bool * logged = &logged_unknown;
    if (path && strcmp(path, "direct_scalar") == 0) {
        logged = &logged_direct_scalar;
    } else if (path && strcmp(path, "direct_avx2") == 0) {
        logged = &logged_direct_avx2;
    } else if (path && strcmp(path, "direct_avx512") == 0) {
        logged = &logged_direct_avx512;
    } else if (path && strcmp(path, "direct_neon") == 0) {
        logged = &logged_direct_neon;
    } else if (path && strcmp(path, "direct_dotprod") == 0) {
        logged = &logged_direct_dotprod;
    } else if (path && strcmp(path, "lut16") == 0) {
        logged = &logged_lut16;
    } else if (path && strcmp(path, "lut_c") == 0) {
        logged = &logged_lut_c;
    }
    if (*logged) {
        return;
    }
    *logged = true;

    const struct ggml_tensor * x = dst ? dst->src[0] : nullptr;
    GGML_LOG_INFO("fairy2i_w2: path=%s M=%lld N=%lld K=%lld nth=%d lut_packed=%d lut_wsize=%zu\n",
                  path ? path : "unknown", (long long) (dst ? dst->ne[0] : 0), (long long) (x ? ggml_nrows(x) : 0),
                  (long long) (x ? x->ne[0] : 0), params ? params->nth : 1,
                  ggml_fairy2i_cpu_lut_packed_weights_ready(dst) ? 1 : 0, plan ? plan->lut_bytes : 0);
}

static bool ggml_fairy2i_cpu_build_plan(const struct ggml_tensor * dst, int n_tasks, struct ggml_fairy2i_cpu_plan * plan) {
    GGML_UNUSED(n_tasks);

    if (!plan) {
        return false;
    }

    memset(plan, 0, sizeof(*plan));
    plan->impl = GGML_FAIRY2I_LUT_IMPL_AUTO;

    if (!ggml_fairy2i_cpu_supports_op(dst)) {
        return false;
    }

    const struct ggml_tensor * x = dst->src[0];
    GGML_ASSERT(x && x->type == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[0] % ggml_blck_size(GGML_TYPE_FAIRY2I_TILE64_V2) == 0);

    const size_t q_row_size = ggml_row_size(GGML_TYPE_FAIRY2I_ACT_Q16_64, x->ne[0]);
    plan->q_bytes           = GGML_PAD((size_t) ggml_nrows(x) * q_row_size, GGML_FAIRY2I_CPU_CACHE_LINE);
    plan->work_size         = plan->q_bytes;

#ifdef GGML_USE_FAIRY2I_CPU_LUT
    const struct ggml_fairy2i_lut_config cfg = ggml_fairy2i_lut_config_from_env();
    plan->use_lut                            = cfg.lut_enabled && cfg.lut_explicit;
    plan->lut_c                              = cfg.impl == GGML_FAIRY2I_LUT_IMPL_LUT_C;
    plan->impl                               = cfg.impl;
    if (plan->use_lut) {
        plan->lut_bytes = ggml_fairy2i_wide_linear_w2_lut_wsize(dst);
        if (plan->lut_bytes == 0) {
            plan->use_lut = false;
        } else if (plan->work_size < plan->lut_bytes) {
            plan->work_size = plan->lut_bytes;
        }
    }
#endif

    return true;
}

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
    struct ggml_fairy2i_cpu_plan plan;
    if (!ggml_fairy2i_cpu_build_plan(dst, n_tasks, &plan)) {
        return 0;
    }

    return plan.work_size;
}

void ggml_fairy2i_cpu_prepare_graph(const struct ggml_cgraph * cgraph) {
#ifdef GGML_USE_FAIRY2I_CPU_LUT
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];
        struct ggml_fairy2i_cpu_plan plan;
        if (!ggml_fairy2i_cpu_build_plan(node, 1, &plan) || !plan.use_lut) {
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
    struct ggml_fairy2i_cpu_plan plan;
    if (!ggml_fairy2i_cpu_build_plan(dst, params ? params->nth : 1, &plan)) {
        return false;
    }

#ifdef GGML_USE_FAIRY2I_CPU_LUT
    if (plan.use_lut) {
        if (ggml_fairy2i_wide_linear_w2_compute_lut(params, dst, plan.lut_c)) {
            ggml_fairy2i_cpu_debug_log_w2_once(params, dst, &plan, plan.lut_c ? "lut_c" : "lut16");
            return true;
        }
        if (ggml_fairy2i_test_require_lut()) {
            return false;
        }
    }
#endif

    ggml_fairy2i_wide_linear_w2_compute(params, dst);
    ggml_fairy2i_cpu_debug_log_w2_once(params, dst, &plan, ggml_fairy2i_cpu_direct_path_name());
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
