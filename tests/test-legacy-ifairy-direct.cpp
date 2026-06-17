// Legacy iFairy direct CPU coverage without LUT linkage.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

extern "C" {
#include "../ggml/src/ggml-common.h"
#include "../ggml/src/ggml-quants.h"

void ggml_vec_dot_ifairy_q16_K(int                        n,
                               float * GGML_RESTRICT      s,
                               size_t                     bs,
                               const void * GGML_RESTRICT vx,
                               size_t                     bx,
                               const void * GGML_RESTRICT vy,
                               size_t                     by,
                               int                        nrc);
void ggml_vec_dot_ifairy_q16_K_generic(int                        n,
                                       float * GGML_RESTRICT      s,
                                       size_t                     bs,
                                       const void * GGML_RESTRICT vx,
                                       size_t                     bx,
                                       const void * GGML_RESTRICT vy,
                                       size_t                     by,
                                       int                        nrc);
}

#ifndef GGML_FP16_TO_FP32
#    define GGML_FP16_TO_FP32 ggml_fp16_to_fp32
#endif
#ifndef GGML_FP32_TO_FP16
#    define GGML_FP32_TO_FP16 ggml_fp32_to_fp16
#endif
#ifndef GGML_BF16_TO_FP32
#    define GGML_BF16_TO_FP32 ggml_bf16_to_fp32
#endif
#ifndef GGML_FP32_TO_BF16
#    define GGML_FP32_TO_BF16 ggml_fp32_to_bf16
#endif

#undef NDEBUG
#include <assert.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void set_env_var(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    setenv(name, value ? value : "", 1);
#endif
}

static void unset_env_var(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

struct scoped_env_var {
    std::string name;
    std::string old_value;
    bool        had = false;

    scoped_env_var(const char * name_) : name(name_) {
        const char * v = getenv(name_);
        if (v) {
            had       = true;
            old_value = v;
        }
    }

    void set(const char * value) const { set_env_var(name.c_str(), value); }

    void unset() const { unset_env_var(name.c_str()); }

    ~scoped_env_var() {
        if (had) {
            set_env_var(name.c_str(), old_value.c_str());
        } else {
            unset_env_var(name.c_str());
        }
    }
};

static float pack_bf16_pair(float real, float imag) {
    ggml_bf16_t pair[2];
    pair[0] = GGML_FP32_TO_BF16(real);
    pair[1] = GGML_FP32_TO_BF16(imag);

    float out;
    memcpy(&out, pair, sizeof(out));
    return out;
}

static bool compare_packed_complex_outputs(const uint32_t * a, const uint32_t * b, size_t n, float max_error) {
    float  max_diff     = 0.0f;
    size_t max_diff_idx = 0;
    int    max_diff_ch  = 0;

    for (size_t i = 0; i < n; ++i) {
        ggml_bf16_t a_pair[2];
        ggml_bf16_t b_pair[2];
        memcpy(a_pair, a + i, sizeof(uint32_t));
        memcpy(b_pair, b + i, sizeof(uint32_t));

        const float ar = GGML_BF16_TO_FP32(a_pair[0]);
        const float ai = GGML_BF16_TO_FP32(a_pair[1]);
        const float br = GGML_BF16_TO_FP32(b_pair[0]);
        const float bi = GGML_BF16_TO_FP32(b_pair[1]);

        const float diff_r = fabsf(ar - br);
        const float diff_i = fabsf(ai - bi);
        if (diff_r > max_diff) {
            max_diff     = diff_r;
            max_diff_idx = i;
            max_diff_ch  = 0;
        }
        if (diff_i > max_diff) {
            max_diff     = diff_i;
            max_diff_idx = i;
            max_diff_ch  = 1;
        }
    }

    if (max_diff > max_error) {
        ggml_bf16_t a_pair[2];
        ggml_bf16_t b_pair[2];
        memcpy(a_pair, a + max_diff_idx, sizeof(uint32_t));
        memcpy(b_pair, b + max_diff_idx, sizeof(uint32_t));
        fprintf(stderr,
                "packed compare mismatch at index %zu channel=%s diff=%.6f "
                "(a=(%.6f, %.6f) b=(%.6f, %.6f))\n",
                max_diff_idx, max_diff_ch == 0 ? "real" : "imag", max_diff, GGML_BF16_TO_FP32(a_pair[0]),
                GGML_BF16_TO_FP32(a_pair[1]), GGML_BF16_TO_FP32(b_pair[0]), GGML_BF16_TO_FP32(b_pair[1]));
        return false;
    }

    printf("  packed compare max diff: %.6f (threshold %.6f)\n", max_diff, max_error);
    return true;
}

static void set_ifairy_code(block_ifairy & blk, int idx, uint8_t code) {
    assert(idx >= 0 && idx < QK_IFAIRY);
    const int chunk = idx / 64;
    const int part  = (idx >> 4) & 0x3;
    const int lane  = idx & 0x0f;

    uint8_t & packed = blk.qs[chunk * 16 + lane];
    packed &= (uint8_t) ~(0x3u << (2 * part));
    packed |= (uint8_t) ((code & 0x3u) << (2 * part));
}

static void set_ifairy64_code(block_ifairy64 & blk, int idx, uint8_t code) {
    assert(idx >= 0 && idx < QK_IFAIRY64);
    const int lane  = idx & 0x0f;
    const int part  = idx >> 4;
    const int shift = 2 * part;

    blk.qs[lane] = (uint8_t) ((blk.qs[lane] & ~(0x3u << shift)) | ((code & 0x3u) << shift));
}

static bool run_ifairy_vecdot_compare_case(int k, uint32_t seed, bool tensor_scale) {
    if (k <= 0 || (k % QK_IFAIRY) != 0) {
        fprintf(stderr, "invalid vecdot K: %d\n", k);
        return false;
    }

    const int nb   = k / QK_IFAIRY;
    const int rows = 3;

    std::mt19937                          rng(seed);
    std::uniform_int_distribution<int>    code_dist(0, 3);
    std::uniform_int_distribution<int>    act_dist(-42, 42);
    std::uniform_real_distribution<float> scale_dist(0.05f, 1.25f);

    std::vector<block_ifairy> w((size_t) rows * (size_t) nb);
    for (int r = 0; r < rows; ++r) {
        for (int ib = 0; ib < nb; ++ib) {
            block_ifairy blk{};
            blk.d_real = GGML_FP32_TO_FP16(1.0f);
            blk.d_imag = GGML_FP32_TO_FP16(1.0f);
            for (int j = 0; j < QK_IFAIRY; ++j) {
                set_ifairy_code(blk, j, (uint8_t) code_dist(rng));
            }
            w[(size_t) r * (size_t) nb + (size_t) ib] = blk;
        }
    }

    std::vector<block_ifairy_q16> x((size_t) nb);
    const float                   d_real_uniform = 0.125f;
    const float                   d_imag_uniform = 0.175f;
    for (int ib = 0; ib < nb; ++ib) {
        const float d_real = tensor_scale ? d_real_uniform : scale_dist(rng);
        const float d_imag = tensor_scale ? d_imag_uniform : scale_dist(rng);
        x[ib].d_real       = GGML_FP32_TO_FP16(d_real);
        x[ib].d_imag       = GGML_FP32_TO_FP16(d_imag);

        int8_t * xr = (int8_t *) x[ib].x_real;
        int8_t * xi = (int8_t *) x[ib].x_imag;
        for (int j = 0; j < QK_IFAIRY; ++j) {
            xr[j] = (int8_t) act_dist(rng);
            xi[j] = (int8_t) act_dist(rng);
        }
    }

    bool ok = true;
    for (int r = 0; r < rows; ++r) {
        alignas(4) uint32_t out_ref = 0;
        alignas(4) uint32_t out_opt = 0;

        ggml_vec_dot_ifairy_q16_K_generic(k, reinterpret_cast<float *>(&out_ref), 0,
                                          w.data() + (size_t) r * (size_t) nb, 0, x.data(), 0, 1);
        ggml_vec_dot_ifairy_q16_K(k, reinterpret_cast<float *>(&out_opt), 0, w.data() + (size_t) r * (size_t) nb, 0,
                                  x.data(), 0, 1);

        if (out_ref != out_opt) {
            const ggml_bf16_t rr{ (uint16_t) (out_ref & 0xffffu) };
            const ggml_bf16_t ri{ (uint16_t) (out_ref >> 16) };
            const ggml_bf16_t orr{ (uint16_t) (out_opt & 0xffffu) };
            const ggml_bf16_t ori{ (uint16_t) (out_opt >> 16) };
            fprintf(stderr,
                    "vecdot mismatch: k=%d tensor_scale=%d row=%d ref=(%.7g, %.7g) opt=(%.7g, %.7g)\n",
                    k, (int) tensor_scale, r, GGML_BF16_TO_FP32(rr), GGML_BF16_TO_FP32(ri), GGML_BF16_TO_FP32(orr),
                    GGML_BF16_TO_FP32(ori));
            ok = false;
        }
    }

    return ok;
}

static bool test_ifairy_vecdot_compare_mode(bool tensor_mode) {
    scoped_env_var env_vecdot("GGML_IFAIRY_VEC_DOT_ACT_TENSOR");
    if (tensor_mode) {
        env_vecdot.set("1");
    } else {
        env_vecdot.unset();
    }

    printf("\n=== legacy iFairy vecdot direct compare (%s) ===\n", tensor_mode ? "tensor-scale" : "per-block");

    bool ok = true;
    ok &= run_ifairy_vecdot_compare_case(QK_IFAIRY, 1001u, tensor_mode);
    ok &= run_ifairy_vecdot_compare_case(1536, 2027u, tensor_mode);
    printf("  vecdot compare (%s): %s\n", tensor_mode ? "tensor-scale" : "per-block", ok ? "PASS" : "FAIL");
    return ok;
}

static void fill_ifairy64_backend_weights(std::vector<block_ifairy64> & weights, int64_t m, int64_t k) {
    const int64_t blocks_per_row = k / QK_IFAIRY64;
    const float   w_scale        = 1.0f / 8.0f;

    weights.assign((size_t) m * (size_t) blocks_per_row, block_ifairy64{});
    for (int64_t r = 0; r < m; ++r) {
        for (int64_t b = 0; b < blocks_per_row; ++b) {
            block_ifairy64 blk{};
            blk.d_real = GGML_FP32_TO_FP16(w_scale);
            blk.d_imag = GGML_FP32_TO_FP16(w_scale);
            for (int j = 0; j < QK_IFAIRY64; ++j) {
                const int     k_idx = (int) (b * QK_IFAIRY64 + j);
                const uint8_t code  = (uint8_t) ((k_idx + 3 * (int) r + 1) & 0x3);
                set_ifairy64_code(blk, j, code);
            }
            weights[(size_t) r * (size_t) blocks_per_row + (size_t) b] = blk;
        }
    }
}

static void fill_ifairy_backend_act_f32(std::vector<float> & act_f32, int64_t n, int64_t k) {
    act_f32.assign((size_t) n * (size_t) k, 0.0f);
    for (int64_t c = 0; c < n; ++c) {
        for (int64_t k_idx = 0; k_idx < k; ++k_idx) {
            const float xr                                    = (float) (((k_idx + 7 * c) % 17) - 8) / 7.0f;
            const float xi                                    = (float) (((k_idx * 2 + 3 * c) % 15) - 7) / 6.0f;
            act_f32[(size_t) c * (size_t) k + (size_t) k_idx] = pack_bf16_pair(xr, xi);
        }
    }
}

static void fill_ifairy_backend_conj(std::vector<float> & x_conj, const std::vector<float> & x) {
    x_conj.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const ggml_bf16_t * in  = (const ggml_bf16_t *) &x[i];
        ggml_bf16_t *       out = (ggml_bf16_t *) &x_conj[i];
        out[0]                  = in[0];
        out[1]                  = GGML_FP32_TO_BF16(-GGML_BF16_TO_FP32(in[1]));
    }
}

static bool run_ifairy64_wide_linear_w2_backend(std::vector<uint32_t> &             packed_out,
                                                int64_t                             m,
                                                int64_t                             n,
                                                int64_t                             k,
                                                const std::vector<block_ifairy64> & u_s0_data,
                                                const std::vector<block_ifairy64> & u_s1_data,
                                                const std::vector<block_ifairy64> & w_s0_data,
                                                const std::vector<block_ifairy64> & w_s1_data,
                                                const std::vector<float> &          x_data,
                                                const std::vector<float> &          x_conj_data,
                                                const std::vector<float> *          bias_data,
                                                bool                                fused) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "failed to init CPU backend\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, 4);

    struct ggml_init_params params = {
        /*.mem_size   =*/128 * 1024 * 1024,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        fprintf(stderr, "failed to init ggml context\n");
        return false;
    }

    ggml_tensor * x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * x_conj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * u_s0   = ggml_new_tensor_2d(ctx, GGML_TYPE_IFAIRY64, k, m);
    ggml_tensor * u_s1   = ggml_new_tensor_2d(ctx, GGML_TYPE_IFAIRY64, k, m);
    ggml_tensor * w_s0   = ggml_new_tensor_2d(ctx, GGML_TYPE_IFAIRY64, k, m);
    ggml_tensor * w_s1   = ggml_new_tensor_2d(ctx, GGML_TYPE_IFAIRY64, k, m);
    ggml_tensor * bias   = bias_data ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * m) : nullptr;

    ggml_tensor * out = nullptr;
    if (fused) {
        out = ggml_ifairy_wide_linear_w2(ctx, x, u_s0, u_s1, w_s0, w_s1, bias);
    } else {
        ggml_tensor * u = ggml_ifairy_add(ctx, ggml_mul_mat(ctx, u_s0, x_conj), ggml_mul_mat(ctx, u_s1, x_conj));
        ggml_tensor * w = ggml_ifairy_add(ctx, ggml_mul_mat(ctx, w_s0, x), ggml_mul_mat(ctx, w_s1, x));
        out             = ggml_ifairy_add(ctx, u, w);
        if (bias) {
            out = ggml_ifairy_merge(ctx, ggml_add(ctx, ggml_ifairy_split(ctx, out), bias));
        }
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "failed to alloc backend buffer\n");
        return false;
    }

    ggml_backend_tensor_set(x, x_data.data(), 0, x_data.size() * sizeof(float));
    ggml_backend_tensor_set(x_conj, x_conj_data.data(), 0, x_conj_data.size() * sizeof(float));
    ggml_backend_tensor_set(u_s0, u_s0_data.data(), 0, u_s0_data.size() * sizeof(block_ifairy64));
    ggml_backend_tensor_set(u_s1, u_s1_data.data(), 0, u_s1_data.size() * sizeof(block_ifairy64));
    ggml_backend_tensor_set(w_s0, w_s0_data.data(), 0, w_s0_data.size() * sizeof(block_ifairy64));
    ggml_backend_tensor_set(w_s1, w_s1_data.data(), 0, w_s1_data.size() * sizeof(block_ifairy64));
    if (bias) {
        ggml_backend_tensor_set(bias, bias_data->data(), 0, bias_data->size() * sizeof(float));
    }

    const bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    if (ok) {
        std::vector<float> out_data((size_t) m * (size_t) n);
        ggml_backend_tensor_get(out, out_data.data(), 0, out_data.size() * sizeof(float));
        packed_out.resize(out_data.size());
        memcpy(packed_out.data(), out_data.data(), out_data.size() * sizeof(float));
    } else {
        fprintf(stderr, "legacy iFairy W2 graph compute failed\n");
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok;
}

static bool test_ifairy64_wide_linear_w2_direct() {
    printf("\n=== legacy iFairy64 W2 direct fuse smoke ===\n");

    scoped_env_var env_lut("GGML_IFAIRY_LUT");
    env_lut.set("0");

    const int64_t m = 7;
    const int64_t k = 2 * QK_IFAIRY64;

    std::vector<block_ifairy64> u_s0;
    std::vector<block_ifairy64> u_s1;
    std::vector<block_ifairy64> w_s0;
    std::vector<block_ifairy64> w_s1;
    fill_ifairy64_backend_weights(u_s0, m, k);
    fill_ifairy64_backend_weights(u_s1, m, k);
    fill_ifairy64_backend_weights(w_s0, m, k);
    fill_ifairy64_backend_weights(w_s1, m, k);

    for (size_t i = 0; i < u_s0.size(); ++i) {
        u_s0[i].d_real = GGML_FP32_TO_FP16(0.050f + 0.001f * (float) (i % 5));
        u_s0[i].d_imag = GGML_FP32_TO_FP16(0.040f + 0.001f * (float) (i % 3));
        u_s1[i].d_real = GGML_FP32_TO_FP16(0.035f + 0.001f * (float) (i % 7));
        u_s1[i].d_imag = GGML_FP32_TO_FP16(0.030f + 0.001f * (float) (i % 2));
        w_s0[i].d_real = GGML_FP32_TO_FP16(0.045f + 0.001f * (float) (i % 4));
        w_s0[i].d_imag = GGML_FP32_TO_FP16(0.025f + 0.001f * (float) (i % 6));
        w_s1[i].d_real = GGML_FP32_TO_FP16(0.055f + 0.001f * (float) (i % 3));
        w_s1[i].d_imag = GGML_FP32_TO_FP16(0.020f + 0.001f * (float) (i % 5));
    }

    std::vector<float> bias((size_t) 2 * (size_t) m);
    for (int64_t i = 0; i < 2 * m; ++i) {
        bias[(size_t) i] = (float) (i - m) / 32.0f;
    }

    const int64_t              n_cases[]    = { 1, 3 };
    const std::vector<float> * bias_cases[] = { nullptr, &bias };
    for (int64_t n : n_cases) {
        std::vector<float> x;
        std::vector<float> x_conj;
        fill_ifairy_backend_act_f32(x, n, k);
        fill_ifairy_backend_conj(x_conj, x);

        for (const std::vector<float> * bias_case : bias_cases) {
            std::vector<uint32_t> out_fused;
            std::vector<uint32_t> out_reference;
            if (!run_ifairy64_wide_linear_w2_backend(out_fused, m, n, k, u_s0, u_s1, w_s0, w_s1, x, x_conj,
                                                      bias_case, true) ||
                !run_ifairy64_wide_linear_w2_backend(out_reference, m, n, k, u_s0, u_s1, w_s0, w_s1, x, x_conj,
                                                      bias_case, false)) {
                return false;
            }

            if (!compare_packed_complex_outputs(out_fused.data(), out_reference.data(), out_reference.size(),
                                                1e-2f)) {
                fprintf(stderr, "W2 direct mismatch: N=%lld %s bias\n", (long long) n,
                        bias_case ? "with" : "without");
                return false;
            }
        }
    }

    return true;
}

int main() {
    ggml_cpu_init();

    printf("========================================\n");
    printf("Legacy iFairy Direct CPU Tests\n");
    printf("========================================\n");

    bool ok = true;
    ok &= test_ifairy_vecdot_compare_mode(false);
    ok &= test_ifairy_vecdot_compare_mode(true);
    ok &= test_ifairy64_wide_linear_w2_direct();

    printf("\nlegacy iFairy direct tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
