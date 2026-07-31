#include "fairy2i-cpu.h"

#include "fairy2i-bundle.h"
#include "fairy2i-policy.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "wide-linear.h"

#if defined(__aarch64__) && defined(__ARM_NEON)
#    include "arm/fairy2i-quants.h"
#endif

#include <stdlib.h>
#include <string.h>

static constexpr size_t GGML_FAIRY2I_CPU_CACHE_LINE = 64;

struct ggml_fairy2i_cpu_plan {
    bool                       use_lut;
    bool                       lut_c;
    bool                       exact_bundle;
    size_t                     q_bytes;
    size_t                     lut_bytes;
    size_t                     work_size;
    enum ggml_fairy2i_lut_impl impl;
};

static bool ggml_fairy2i_cpu_debug_enabled(void) {
    const char * env = getenv("GGML_FAIRY2I_CPU_DEBUG");
    return env && strcmp(env, "0") != 0;
}

static bool ggml_fairy2i_cpu_timing_enabled(void) {
    const char * env = getenv("GGML_FAIRY2I_CPU_TIMING");
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
    const int last_src = dst->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W1 ? 2 : 4;
    for (int src = 1; src <= last_src; ++src) {
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

static const char * ggml_fairy2i_cpu_op_log_name(const struct ggml_tensor * dst) {
    return dst && dst->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W1 ? "fairy2i_w1" : "fairy2i_w2";
}

static void ggml_fairy2i_cpu_debug_log_once(const struct ggml_compute_params *   params,
                                            const struct ggml_tensor *           dst,
                                            const struct ggml_fairy2i_cpu_plan * plan,
                                            const char *                         path) {
    if (!ggml_fairy2i_cpu_debug_enabled() || (params && params->ith != 0)) {
        return;
    }

    enum {
        PATH_SCALAR,
        PATH_AVX2,
        PATH_AVX512,
        PATH_NEON,
        PATH_DOTPROD,
        PATH_BF16_EXACT,
        PATH_LUT16,
        PATH_LUT_C,
        PATH_UNKNOWN,
        PATH_COUNT,
    };

    static bool logged[2][PATH_COUNT] = {};

    const int op_idx   = dst && dst->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W1 ? 0 : 1;
    int       path_idx = PATH_UNKNOWN;
    if (path && strcmp(path, "direct_scalar") == 0) {
        path_idx = PATH_SCALAR;
    } else if (path && strcmp(path, "direct_avx2") == 0) {
        path_idx = PATH_AVX2;
    } else if (path && strcmp(path, "direct_avx512") == 0) {
        path_idx = PATH_AVX512;
    } else if (path && strcmp(path, "direct_neon") == 0) {
        path_idx = PATH_NEON;
    } else if (path && strcmp(path, "direct_dotprod") == 0) {
        path_idx = PATH_DOTPROD;
    } else if (path && strcmp(path, "direct_bf16_exact") == 0) {
        path_idx = PATH_BF16_EXACT;
    } else if (path && strcmp(path, "lut16") == 0) {
        path_idx = PATH_LUT16;
    } else if (path && strcmp(path, "lut_c") == 0) {
        path_idx = PATH_LUT_C;
    }
    if (logged[op_idx][path_idx]) {
        return;
    }
    logged[op_idx][path_idx] = true;

    const struct ggml_tensor * x = dst ? dst->src[0] : nullptr;
    const bool                 bundle = ggml_fairy2i_is_bundle_op(dst);
    GGML_LOG_INFO("%s: path=%s M=%lld N=%lld K=%lld nth=%d layout=%s packed_w=%d branches=%d lut_wsize=%zu\n",
                  ggml_fairy2i_cpu_op_log_name(dst), path ? path : "unknown", (long long) (dst ? dst->ne[0] : 0),
                  (long long) (x ? ggml_nrows(x) : 0), (long long) (x ? x->ne[0] : 0), params ? params->nth : 1,
                  bundle ? "bundle_m64k64_v1" : "tile64_v2", ggml_fairy2i_cpu_lut_packed_weights_ready(dst) ? 1 : 0,
                  bundle ? ggml_get_op_params_i32(dst, 3) : (dst && dst->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W1 ? 2 : 4),
                  plan ? plan->lut_bytes : 0);
}

static int64_t ggml_fairy2i_cpu_timing_start_us(const struct ggml_compute_params * params) {
    return ggml_fairy2i_cpu_timing_enabled() && params && params->ith == 0 ? ggml_time_us() : 0;
}

static void ggml_fairy2i_cpu_timing_log(const struct ggml_compute_params * params,
                                        const struct ggml_tensor *         dst,
                                        const char *                       path,
                                        int64_t                            start_us) {
    if (start_us == 0 || !params || params->ith != 0) {
        return;
    }

    const struct ggml_tensor * x = dst ? dst->src[0] : nullptr;
    GGML_LOG_INFO("%s: timing path=%s us=%lld M=%lld N=%lld K=%lld nth=%d\n", ggml_fairy2i_cpu_op_log_name(dst),
                  path ? path : "unknown", (long long) (ggml_time_us() - start_us), (long long) (dst ? dst->ne[0] : 0),
                  (long long) (x ? ggml_nrows(x) : 0), (long long) (x ? x->ne[0] : 0), params->nth);
}

static bool ggml_fairy2i_cpu_build_plan(const struct ggml_tensor * dst, int n_tasks, struct ggml_fairy2i_cpu_plan * plan) {
    GGML_UNUSED(n_tasks);

    if (!plan) {
        return false;
    }

    memset(plan, 0, sizeof(*plan));
    plan->impl = GGML_FAIRY2I_LUT_IMPL_LUT16;

    if (!ggml_fairy2i_cpu_supports_op(dst)) {
        return false;
    }

    if (ggml_fairy2i_is_bundle_op(dst)) {
        struct ggml_fairy2i_bundle_desc bundle;
        if (!ggml_fairy2i_bundle_desc_init(dst, &bundle, false)) {
            return false;
        }
        plan->exact_bundle = bundle.scale_type == GGML_TYPE_BF16;
        if (plan->exact_bundle) {
            return true;
        }
    }

    const struct ggml_tensor * x = dst->src[0];
    GGML_ASSERT(x && x->type == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[0] % ggml_blck_size(GGML_TYPE_FAIRY2I_TILE64_V2) == 0);

    const size_t q_row_size = ggml_row_size(GGML_TYPE_FAIRY2I_ACT_Q16_64, x->ne[0]);
    plan->q_bytes           = GGML_PAD((size_t) ggml_nrows(x) * q_row_size, GGML_FAIRY2I_CPU_CACHE_LINE);
    plan->work_size         = plan->q_bytes;

#ifdef GGML_USE_FAIRY2I_CPU_LUT
    const struct ggml_fairy2i_lut_policy policy = ggml_fairy2i_lut_policy_from_env();
    plan->use_lut                               = policy.lut_enabled;
    plan->lut_c                                 = policy.impl == GGML_FAIRY2I_LUT_IMPL_LUT_C;
    plan->impl                                  = policy.impl;
    if (plan->use_lut) {
        plan->lut_bytes = dst->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W1 ? ggml_fairy2i_wide_linear_w1_lut_wsize(dst) :
                                                                      ggml_fairy2i_wide_linear_w2_lut_wsize(dst);
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
    if (!dst || (dst->op != GGML_OP_FAIRY2I_WIDE_LINEAR_W1 && dst->op != GGML_OP_FAIRY2I_WIDE_LINEAR_W2)) {
        return false;
    }
    if (!ggml_fairy2i_is_bundle_op(dst)) {
        return true;
    }

    struct ggml_fairy2i_bundle_desc bundle;
    if (!ggml_fairy2i_bundle_desc_init(dst, &bundle, false)) {
        return false;
    }
    if (bundle.scale_type == GGML_TYPE_BF16) {
        return dst->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W2;
    }

#ifdef GGML_USE_FAIRY2I_CPU_LUT
    const struct ggml_fairy2i_lut_policy policy = ggml_fairy2i_lut_policy_from_env();
    return policy.lut_enabled && policy.impl == GGML_FAIRY2I_LUT_IMPL_LUT16;
#else
    return false;
#endif
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

        if (ggml_fairy2i_is_bundle_op(node)) {
            continue;
        }

        const int last_src = node->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W1 ? 2 : 4;
        for (int src = 1; src <= last_src; ++src) {
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

static bool ggml_fairy2i_cpu_compute_wide_linear_w1(const struct ggml_compute_params * params,
                                                    struct ggml_tensor *               dst) {
    struct ggml_fairy2i_cpu_plan plan;
    if (!ggml_fairy2i_cpu_build_plan(dst, params ? params->nth : 1, &plan)) {
        return false;
    }

    const int64_t timing_start_us = ggml_fairy2i_cpu_timing_start_us(params);

#ifdef GGML_USE_FAIRY2I_CPU_LUT
    if (plan.use_lut) {
        if (ggml_fairy2i_wide_linear_w1_compute_lut(params, dst, plan.lut_c)) {
            const char * path = plan.lut_c ? "lut_c" : "lut16";
            ggml_fairy2i_cpu_debug_log_once(params, dst, &plan, path);
            ggml_fairy2i_cpu_timing_log(params, dst, path, timing_start_us);
            return true;
        }
        if (ggml_fairy2i_test_require_lut()) {
            return false;
        }
    }
#endif

    if (ggml_fairy2i_is_bundle_op(dst)) {
        if (!params || params->ith == 0) {
            GGML_LOG_ERROR(
                "fairy2i_w1: bundle_m64k64_v1 requires the CPU LUT16 path; check GGML_FAIRY2I_LUT and "
                "GGML_FAIRY2I_LUT_IMPL\n");
        }
        return false;
    }

    ggml_fairy2i_wide_linear_w1_compute(params, dst);
    const char * path = ggml_fairy2i_cpu_direct_path_name();
    ggml_fairy2i_cpu_debug_log_once(params, dst, &plan, path);
    ggml_fairy2i_cpu_timing_log(params, dst, path, timing_start_us);
    return true;
}

static bool ggml_fairy2i_cpu_compute_wide_linear_w2(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    struct ggml_fairy2i_cpu_plan plan;
    if (!ggml_fairy2i_cpu_build_plan(dst, params ? params->nth : 1, &plan)) {
        return false;
    }

    const int64_t timing_start_us = ggml_fairy2i_cpu_timing_start_us(params);

    if (plan.exact_bundle) {
        ggml_fairy2i_wide_linear_w2_bundle_exact_compute(params, dst);
        ggml_fairy2i_cpu_debug_log_once(params, dst, &plan, "direct_bf16_exact");
        ggml_fairy2i_cpu_timing_log(params, dst, "direct_bf16_exact", timing_start_us);
        return true;
    }

#ifdef GGML_USE_FAIRY2I_CPU_LUT
    if (plan.use_lut) {
        if (ggml_fairy2i_wide_linear_w2_compute_lut(params, dst, plan.lut_c)) {
            const char * path = plan.lut_c ? "lut_c" : "lut16";
            ggml_fairy2i_cpu_debug_log_once(params, dst, &plan, path);
            ggml_fairy2i_cpu_timing_log(params, dst, path, timing_start_us);
            return true;
        }
        if (ggml_fairy2i_test_require_lut()) {
            return false;
        }
    }
#endif

    if (ggml_fairy2i_is_bundle_op(dst)) {
        if (!params || params->ith == 0) {
            GGML_LOG_ERROR(
                "fairy2i_w2: bundle_m64k64_v1 requires the CPU LUT16 path; check GGML_FAIRY2I_LUT and "
                "GGML_FAIRY2I_LUT_IMPL\n");
        }
        return false;
    }

    ggml_fairy2i_wide_linear_w2_compute(params, dst);
    const char * path = ggml_fairy2i_cpu_direct_path_name();
    ggml_fairy2i_cpu_debug_log_once(params, dst, &plan, path);
    ggml_fairy2i_cpu_timing_log(params, dst, path, timing_start_us);
    return true;
}

bool ggml_fairy2i_cpu_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (dst && dst->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W1) {
        return ggml_fairy2i_cpu_compute_wide_linear_w1(params, dst);
    }
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
