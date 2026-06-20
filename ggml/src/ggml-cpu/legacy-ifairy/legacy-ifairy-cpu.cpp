#include "legacy-ifairy-cpu.h"
#include "wide-linear.h"

#include "ggml.h"
#include "quants.h"

#ifdef GGML_USE_LEGACY_IFAIRY_CPU_LUT
#    include "ggml-ifairy-lut.h"
#endif

#include <stdlib.h>
#include <string.h>

static constexpr size_t GGML_LEGACY_IFAIRY_CPU_CACHE_LINE = 64;

#ifdef GGML_USE_LEGACY_IFAIRY_CPU_LUT
enum ggml_legacy_ifairy_lut_impl {
    GGML_LEGACY_IFAIRY_LUT_IMPL_AUTO  = 0,
    GGML_LEGACY_IFAIRY_LUT_IMPL_LUT16 = 1,
    GGML_LEGACY_IFAIRY_LUT_IMPL_LUT_C = 2,
};

struct ggml_legacy_ifairy_lut_config {
    bool dbg;
    bool lut_enabled;
    enum ggml_legacy_ifairy_lut_impl impl;
};

static enum ggml_legacy_ifairy_lut_impl ggml_legacy_ifairy_lut_impl_from_env(const char * env_name, bool dbg) {
    enum ggml_legacy_ifairy_lut_impl impl = GGML_LEGACY_IFAIRY_LUT_IMPL_AUTO;
    const char * impl_env                 = getenv(env_name);
    if (impl_env && impl_env[0] != '\0' && strcmp(impl_env, "0") != 0 && strcmp(impl_env, "auto") != 0) {
        if (strcmp(impl_env, "lut16") == 0) {
            impl = GGML_LEGACY_IFAIRY_LUT_IMPL_LUT16;
        } else if (strcmp(impl_env, "lut_c") == 0) {
            impl = GGML_LEGACY_IFAIRY_LUT_IMPL_LUT_C;
        } else if (dbg) {
            GGML_LOG_WARN("legacy_ifairy_lut: unknown %s=%s (expected auto|lut16|lut_c)\n", env_name, impl_env);
        }
    }
    return impl;
}

static struct ggml_legacy_ifairy_lut_config ggml_legacy_ifairy_lut_config_from_env(void) {
    struct ggml_legacy_ifairy_lut_config cfg;

    cfg.dbg         = ggml_ifairy_env_enabled("GGML_IFAIRY_LUT_DEBUG");
    cfg.lut_enabled = ggml_ifairy_env_enabled("GGML_IFAIRY_LUT");
    cfg.impl        = ggml_legacy_ifairy_lut_impl_from_env("GGML_IFAIRY_LUT_IMPL", cfg.dbg);

    return cfg;
}
#endif

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
    const size_t q_bytes    = GGML_PAD((size_t) ggml_nrows(x) * q_row_size, GGML_LEGACY_IFAIRY_CPU_CACHE_LINE);

#ifdef GGML_USE_LEGACY_IFAIRY_CPU_LUT
    const size_t lut_bytes = ggml_ifairy_wide_linear_w2_lut_wsize(dst);
    return q_bytes > lut_bytes ? q_bytes : lut_bytes;
#else
    return q_bytes;
#endif
}

void ggml_legacy_ifairy_cpu_prepare_graph(const struct ggml_cgraph * cgraph) {
#ifdef GGML_USE_LEGACY_IFAIRY_CPU_LUT
    const struct ggml_legacy_ifairy_lut_config cfg = ggml_legacy_ifairy_lut_config_from_env();
    if (!cfg.lut_enabled) {
        return;
    }

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (!node) {
            continue;
        }

        if (node->op == GGML_OP_IFAIRY_WIDE_LINEAR_W2) {
            for (int src = 1; src <= 4; ++src) {
                struct ggml_tensor * weight = node->src[src];
                const struct ifairy_lut_extra * extra = weight ? (const struct ifairy_lut_extra *) weight->extra : nullptr;
                if (weight && (!extra || !extra->packed_w)) {
                    ggml_ifairy_lut_transform_tensor(weight, nullptr);
                }
            }
            continue;
        }

        if (node->op != GGML_OP_MUL_MAT) {
            continue;
        }

        struct ggml_tensor * src0 = node->src[0];
        struct ggml_tensor * src1 = node->src[1];
        if (!src0 || !src1 || !ggml_ifairy_lut_can_mul_mat(src0, src1, node)) {
            continue;
        }

        ggml_ifairy_lut_transform_tensor(src0, nullptr);
    }
#else
    GGML_UNUSED(cgraph);
#endif
}

bool ggml_legacy_ifairy_cpu_try_quantize_mul_mat_src1(
    const struct ggml_compute_params * params,
    const struct ggml_tensor *         src0,
    const struct ggml_tensor *         src1,
    enum ggml_type                     vec_dot_type,
    char *                             wdata,
    size_t                             nbw1,
    size_t                             nbw2,
    size_t                             nbw3) {
    if (!params || !src0 || !src1 || !wdata) {
        return false;
    }
    if (src0->type != GGML_TYPE_IFAIRY || vec_dot_type != GGML_TYPE_IFAIRY_Q16 || src1->type != GGML_TYPE_F32) {
        return false;
    }

    const char * env = getenv("GGML_IFAIRY_VEC_DOT_ACT_TENSOR");
    if (!env || strcmp(env, "0") == 0) {
        return false;
    }

    if (params->ith != 0) {
        return true;
    }

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];
    const size_t nb13 = src1->nb[3];

    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = 0; i11 < ne11; ++i11) {
                quantize_row_ifairy_q16_tensor((float *) ((char *) src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11),
                                               (void *) (wdata + i13 * nbw3 + i12 * nbw2 + i11 * nbw1), ne10);
            }
        }
    }

    return true;
}

static bool ggml_legacy_ifairy_cpu_compute_wide_linear_w2(
    const struct ggml_compute_params * params,
    struct ggml_tensor *                dst,
    bool                                use_lut,
    bool                                lut_c) {
    if (!ggml_legacy_ifairy_cpu_supports_op(dst)) {
        return false;
    }

#ifdef GGML_USE_LEGACY_IFAIRY_CPU_LUT
    if (use_lut && ggml_compute_forward_ifairy_wide_linear_w2_lut(params, dst, lut_c)) {
        return true;
    }
#else
    GGML_UNUSED(use_lut);
    GGML_UNUSED(lut_c);
#endif

    ggml_compute_forward_ifairy_wide_linear_w2(params, dst);
    return true;
}

bool ggml_legacy_ifairy_cpu_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
#ifdef GGML_USE_LEGACY_IFAIRY_CPU_LUT
    const struct ggml_legacy_ifairy_lut_config cfg = ggml_legacy_ifairy_lut_config_from_env();
    const bool use_lut = cfg.lut_enabled;
    const bool lut_c   = cfg.impl == GGML_LEGACY_IFAIRY_LUT_IMPL_LUT_C;
#else
    const bool use_lut = false;
    const bool lut_c   = false;
#endif
    return ggml_legacy_ifairy_cpu_compute_wide_linear_w2(params, dst, use_lut, lut_c);
}
