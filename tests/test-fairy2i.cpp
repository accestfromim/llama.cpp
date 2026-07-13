// Fairy2i CPU backend tests.

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

extern "C" {
#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"
#if defined(__aarch64__) && defined(__ARM_NEON)
bool ggml_fairy2i_tile64_w2_arm_neon_available(void);
bool ggml_fairy2i_tile64_w2_arm_dotprod_available(void);
void ggml_fairy2i_tile64_fuse_accumulate_block_four_neon(const block_fairy2i_tile64_v2 *  u0,
                                                         const block_fairy2i_tile64_v2 *  u1,
                                                         const block_fairy2i_tile64_v2 *  w0,
                                                         const block_fairy2i_tile64_v2 *  w1,
                                                         const block_fairy2i_act_q16_64 * x,
                                                         int32_t                          sums[4][4]);
void ggml_fairy2i_tile64_fuse_accumulate_block_four_dotprod(const block_fairy2i_tile64_v2 *  u0,
                                                            const block_fairy2i_tile64_v2 *  u1,
                                                            const block_fairy2i_tile64_v2 *  w0,
                                                            const block_fairy2i_tile64_v2 *  w1,
                                                            const block_fairy2i_act_q16_64 * x,
                                                            int32_t                          sums[4][4]);
bool ggml_fairy2i_tile64_fuse_accumulate_block_four_arm(const block_fairy2i_tile64_v2 *  u0,
                                                        const block_fairy2i_tile64_v2 *  u1,
                                                        const block_fairy2i_tile64_v2 *  w0,
                                                        const block_fairy2i_tile64_v2 *  w1,
                                                        const block_fairy2i_act_q16_64 * x,
                                                        int32_t                          sums[4][4]);
#endif
}

#if defined(GGML_USE_FAIRY2I_CPU_LUT)
#    include "../ggml/src/ggml-cpu/fairy2i/lut/ggml-fairy2i-lut.h"
void ggml_fairy2i_tile64_lut_quantize_block_q16_64_for_test(const float *              x,
                                                            block_fairy2i_act_q16_64 * y,
                                                            bool                       force_scalar);
#endif

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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

class scoped_env_var {
public:
    explicit scoped_env_var(const char * name) : name(name) {
        const char * value = getenv(name);
        if (value) {
            had_old = true;
            old     = value;
        }
    }

    ~scoped_env_var() {
        if (had_old) {
            set(old.c_str());
        } else {
            unset();
        }
    }

    void set(const char * value) {
#ifdef _WIN32
        _putenv_s(name, value ? value : "");
#else
        setenv(name, value ? value : "", 1);
#endif
    }

    void unset() {
#ifdef _WIN32
        _putenv_s(name, "");
#else
        unsetenv(name);
#endif
    }

private:
    const char * name;
    bool         had_old = false;
    std::string  old;
};

struct fairy2i_w2_case {
    int64_t M;
    int64_t N;
    int64_t K;
    bool    bias;
};

struct fairy2i_w2_data {
    std::vector<float>                   x;
    std::vector<float>                   bias;
    std::vector<block_fairy2i_tile64_v2> u_s0;
    std::vector<block_fairy2i_tile64_v2> u_s1;
    std::vector<block_fairy2i_tile64_v2> w_s0;
    std::vector<block_fairy2i_tile64_v2> w_s1;
};

static uint32_t pack_bf16_pair(float real, float imag) {
    ggml_bf16_t pair[2];
    pair[0] = GGML_FP32_TO_BF16(real);
    pair[1] = GGML_FP32_TO_BF16(imag);

    uint32_t word;
    memcpy(&word, pair, sizeof(word));
    return word;
}

static float pack_complex_bf16(float real, float imag) {
    const uint32_t word = pack_bf16_pair(real, imag);
    float          packed;
    memcpy(&packed, &word, sizeof(packed));
    return packed;
}

static void unpack_bf16_pair(uint32_t word, float & real, float & imag) {
    ggml_bf16_t pair[2];
    memcpy(pair, &word, sizeof(word));
    real = GGML_BF16_TO_FP32(pair[0]);
    imag = GGML_BF16_TO_FP32(pair[1]);
}

static void set_fairy2i_code(block_fairy2i_tile64_v2 & blk, int idx, uint8_t code) {
    const int lane = idx % 16;
    const int part = idx / 16;
    blk.qs[lane] &= (uint8_t) ~(0x3u << (2 * part));
    blk.qs[lane] |= (uint8_t) ((code & 0x3u) << (2 * part));
}

static uint8_t get_fairy2i_code(const block_fairy2i_tile64_v2 & blk, int idx) {
    const int lane = idx % 16;
    const int part = idx / 16;
    return (uint8_t) ((blk.qs[lane] >> (2 * part)) & 0x3u);
}

static void fill_fairy2i_weights(std::vector<block_fairy2i_tile64_v2> & weights, int64_t M, int64_t K, int salt) {
    const int64_t blocks = K / QK_FAIRY2I_TILE64;
    weights.assign((size_t) M * (size_t) blocks, block_fairy2i_tile64_v2{});

    for (int64_t row = 0; row < M; ++row) {
        for (int64_t ib = 0; ib < blocks; ++ib) {
            block_fairy2i_tile64_v2 blk{};
            blk.d_real = GGML_FP32_TO_FP16(0.021f + 0.003f * (float) ((row + ib + salt) % 7));
            blk.d_imag = GGML_FP32_TO_FP16(0.017f + 0.002f * (float) ((2 * row + ib + salt) % 5));
            for (int j = 0; j < QK_FAIRY2I_TILE64; ++j) {
                const uint8_t code = (uint8_t) ((j + 3 * row + 5 * ib + salt) & 0x3);
                set_fairy2i_code(blk, j, code);
            }
            weights[(size_t) row * (size_t) blocks + (size_t) ib] = blk;
        }
    }
}

static std::vector<float> make_fairy2i_input(int64_t N, int64_t K) {
    std::vector<float> x((size_t) N * (size_t) K);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            const float real = (float) (((k + 7 * n) % 19) - 9) / 8.0f;
            const float imag = (float) (((2 * k + 5 * n) % 23) - 11) / 9.0f;
            x[(size_t) n * (size_t) K + (size_t) k] = pack_complex_bf16(real, imag);
        }
    }
    return x;
}

static std::vector<float> make_fairy2i_bias(int64_t M) {
    std::vector<float> bias((size_t) 2 * (size_t) M);
    for (int64_t row = 0; row < M; ++row) {
        bias[(size_t) row]     = (float) (row - M / 2) / 32.0f;
        bias[(size_t) row + M] = (float) (M - 2 * row) / 40.0f;
    }
    return bias;
}

static fairy2i_w2_data make_fairy2i_w2_data(const fairy2i_w2_case & tc) {
    fairy2i_w2_data data;
    data.x = make_fairy2i_input(tc.N, tc.K);
    if (tc.bias) {
        data.bias = make_fairy2i_bias(tc.M);
    }
    fill_fairy2i_weights(data.u_s0, tc.M, tc.K, 1);
    fill_fairy2i_weights(data.u_s1, tc.M, tc.K, 5);
    fill_fairy2i_weights(data.w_s0, tc.M, tc.K, 9);
    fill_fairy2i_weights(data.w_s1, tc.M, tc.K, 13);
    return data;
}

static std::vector<float> make_fairy2i_lut_quantize_input(int salt) {
    static const float values[] = {
        0.0f,         1.0f,          -1.0f,          0.5f / 63.0f,    -0.5f / 63.0f,
        1.0f / 63.0f, -1.0f / 63.0f, 62.49f / 63.0f, -62.49f / 63.0f, 0.3125f,
        -0.6875f,     0.9375f,       -0.9375f,
    };

    std::vector<float> x(QK_FAIRY2I_TILE64);
    for (int j = 0; j < QK_FAIRY2I_TILE64; ++j) {
        const float real = values[(j + salt) % (int) (sizeof(values) / sizeof(values[0]))];
        const float imag = values[(3 * j + 5 * salt) % (int) (sizeof(values) / sizeof(values[0]))];
        x[(size_t) j]    = pack_complex_bf16(real, imag);
    }
    return x;
}

static bool compare_fairy2i_lut_quantize_block(const char *                     label,
                                               const block_fairy2i_act_q16_64 & fast,
                                               const block_fairy2i_act_q16_64 & scalar) {
    if (fast.d_real != scalar.d_real || fast.d_imag != scalar.d_imag) {
        fprintf(stderr, "%s scale mismatch d_real=%u/%u d_imag=%u/%u\n", label, (unsigned) fast.d_real,
                (unsigned) scalar.d_real, (unsigned) fast.d_imag, (unsigned) scalar.d_imag);
        return false;
    }

    for (int j = 0; j < QK_FAIRY2I_TILE64; ++j) {
        if (fast.x_real[j] != scalar.x_real[j] || fast.x_imag[j] != scalar.x_imag[j]) {
            fprintf(stderr, "%s value mismatch j=%d real=%d/%d imag=%d/%d\n", label, j,
                    (int) ((const int8_t *) fast.x_real)[j], (int) ((const int8_t *) scalar.x_real)[j],
                    (int) ((const int8_t *) fast.x_imag)[j], (int) ((const int8_t *) scalar.x_imag)[j]);
            return false;
        }
    }
    return true;
}

static bool test_fairy2i_lut_quantize_arm_neon() {
#if defined(GGML_USE_FAIRY2I_CPU_LUT) && defined(__aarch64__) && defined(__ARM_NEON)
    bool ok = true;
    for (int salt = 0; salt < 13; ++salt) {
        const std::vector<float> x = make_fairy2i_lut_quantize_input(salt);
        block_fairy2i_act_q16_64 fast{};
        block_fairy2i_act_q16_64 scalar{};
        ggml_fairy2i_tile64_lut_quantize_block_q16_64_for_test(x.data(), &scalar, true);
        ggml_fairy2i_tile64_lut_quantize_block_q16_64_for_test(x.data(), &fast, false);

        char label[96];
        snprintf(label, sizeof(label), "Fairy2i LUT ARM quantize salt=%d", salt);
        ok = compare_fairy2i_lut_quantize_block(label, fast, scalar) && ok;
    }
    printf("  Fairy2i LUT ARM quantize: %s\n", ok ? "PASS" : "FAIL");
    return ok;
#elif !defined(GGML_USE_FAIRY2I_CPU_LUT)
    printf("  Fairy2i LUT ARM quantize skipped: build lacks GGML_USE_FAIRY2I_CPU_LUT\n");
    return true;
#else
    printf("  Fairy2i LUT ARM quantize skipped: non-ARM build\n");
    return true;
#endif
}

#if defined(GGML_USE_FAIRY2I_CPU_LUT)
static std::vector<fairy2i_tile64_lut_wtile_16> make_fairy2i_lut_wtiles(int64_t M, int64_t K, int salt) {
    const int64_t                            tiles  = (M + 15) / 16;
    const int64_t                            blocks = K / QK_FAIRY2I_TILE64;
    std::vector<fairy2i_tile64_lut_wtile_16> wtiles((size_t) tiles * (size_t) blocks);

    for (int64_t tile = 0; tile < tiles; ++tile) {
        for (int64_t blk = 0; blk < blocks; ++blk) {
            fairy2i_tile64_lut_wtile_16 & wt = wtiles[(size_t) tile * (size_t) blocks + (size_t) blk];
            for (int lane = 0; lane < 16; ++lane) {
                wt.d_real[lane] = GGML_FP32_TO_FP16(0.013f + 0.001f * (float) ((lane + tile + blk + salt) % 9));
                wt.d_imag[lane] = GGML_FP32_TO_FP16(0.017f + 0.001f * (float) ((2 * lane + tile + blk + salt) % 7));
            }
            for (int byte_idx = 0; byte_idx < QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK / 2; ++byte_idx) {
                for (int lane = 0; lane < 16; ++lane) {
                    const uint8_t lo      = (uint8_t) ((byte_idx + 3 * lane + 5 * blk + salt) & 0x0f);
                    const uint8_t hi      = (uint8_t) ((7 * byte_idx + lane + 3 * tile + salt) & 0x0f);
                    wt.qs[byte_idx][lane] = (uint8_t) (lo | (hi << 4));
                }
            }
        }
    }

    return wtiles;
}

static void fill_fairy2i_lut_qgemm_inputs(std::vector<int8_t> & lut,
                                          std::vector<float> &  scales,
                                          int64_t               N,
                                          int64_t               K) {
    const int64_t blocks = K / QK_FAIRY2I_TILE64;
    const int64_t groups = blocks * QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK;
    lut.assign((size_t) N * (size_t) groups * (size_t) 64, 0);
    scales.assign((size_t) N * (size_t) blocks * 2u, 0.0f);

    for (int64_t col = 0; col < N; ++col) {
        for (int64_t blk = 0; blk < blocks; ++blk) {
            scales[((size_t) col * (size_t) blocks + (size_t) blk) * 2u + 0] =
                0.03125f + 0.00390625f * (float) ((col + blk) % 5);
            scales[((size_t) col * (size_t) blocks + (size_t) blk) * 2u + 1] =
                0.046875f + 0.00390625f * (float) ((2 * col + blk) % 5);
        }

        int8_t * lut_col = lut.data() + (size_t) col * (size_t) groups * (size_t) 64;
        for (int64_t group = 0; group < groups; ++group) {
            int8_t * tbl = lut_col + (size_t) group * 64u;
            for (int channel = 0; channel < 4; ++channel) {
                for (int entry = 0; entry < 16; ++entry) {
                    tbl[channel * 16 + entry] =
                        (int8_t) (((entry * (channel + 3) + 5 * group + 7 * col + channel) % 17) - 8);
                }
            }
        }
    }
}

static void fairy2i_lut_qgemm_lut16_scalar_ref(int64_t                                          M,
                                               int64_t                                          K,
                                               int64_t                                          N,
                                               const std::vector<fairy2i_tile64_lut_wtile_16> & wtiles,
                                               const std::vector<int8_t> &                      lut,
                                               const std::vector<float> &                       scales,
                                               std::vector<float> &                             dst,
                                               bool                                             negate_imag_scale,
                                               bool                                             add) {
    const int64_t blocks = K / QK_FAIRY2I_TILE64;
    const int64_t groups = blocks * QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK;
    if (!add) {
        std::fill(dst.begin(), dst.end(), 0.0f);
    }

    for (int64_t col = 0; col < N; ++col) {
        const int8_t * lut_col = lut.data() + (size_t) col * (size_t) groups * 64u;
        const float *  sc      = scales.data() + (size_t) col * (size_t) blocks * 2u;
        for (int64_t row = 0; row < M; ++row) {
            const int64_t tile = row >> 4;
            const int     lane = (int) (row & 15);
            float         real = add ? dst[((size_t) col * (size_t) M + (size_t) row) * 2u + 0] : 0.0f;
            float         imag = add ? dst[((size_t) col * (size_t) M + (size_t) row) * 2u + 1] : 0.0f;

            for (int64_t blk = 0; blk < blocks; ++blk) {
                const fairy2i_tile64_lut_wtile_16 & wt = wtiles[(size_t) tile * (size_t) blocks + (size_t) blk];
                const float                         lr = sc[blk * 2 + 0];
                const float                         li = negate_imag_scale ? -sc[blk * 2 + 1] : sc[blk * 2 + 1];
                const float                         wr = GGML_FP16_TO_FP32(wt.d_real[lane]);
                const float                         wi = GGML_FP16_TO_FP32(wt.d_imag[lane]);

                int sum_ac = 0;
                int sum_bd = 0;
                int sum_bc = 0;
                int sum_ad = 0;

                const int8_t * lut_blk = lut_col + (size_t) blk * (size_t) QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK * 64u;
                for (int byte_idx = 0; byte_idx < QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK / 2; ++byte_idx) {
                    const uint8_t  packed = wt.qs[byte_idx][lane];
                    const int8_t * tbl0   = lut_blk + (size_t) (byte_idx * 2) * 64u;
                    const int8_t * tbl1   = tbl0 + 64u;
                    const uint8_t  lo     = packed & 0x0fu;
                    const uint8_t  hi     = (packed >> 4) & 0x0fu;

                    sum_ac += (int) tbl0[0 * 16 + lo] + (int) tbl1[0 * 16 + hi];
                    sum_bd += (int) tbl0[1 * 16 + lo] + (int) tbl1[1 * 16 + hi];
                    sum_bc += (int) tbl0[2 * 16 + lo] + (int) tbl1[2 * 16 + hi];
                    sum_ad += (int) tbl0[3 * 16 + lo] + (int) tbl1[3 * 16 + hi];
                }

                real += (float) sum_ac * (lr * wr) + (float) sum_bd * (li * wi);
                imag += (float) sum_bc * (lr * wi) + (float) sum_ad * (li * wr);
            }

            dst[((size_t) col * (size_t) M + (size_t) row) * 2u + 0] = real;
            dst[((size_t) col * (size_t) M + (size_t) row) * 2u + 1] = imag;
        }
    }
}

static bool compare_f32_pairs(const char *               label,
                              const std::vector<float> & actual,
                              const std::vector<float> & expected) {
    if (actual.size() != expected.size()) {
        fprintf(stderr, "%s size mismatch: %zu vs %zu\n", label, actual.size(), expected.size());
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = fabsf(actual[i] - expected[i]);
        if (diff > 1e-5f) {
            fprintf(stderr, "%s mismatch i=%zu actual=%g expected=%g diff=%g\n", label, i, actual[i], expected[i],
                    diff);
            return false;
        }
    }
    return true;
}

static bool test_fairy2i_lut_qgemm_add() {
    const int64_t Ms[]      = { 1, 7, 16, 17, 23 };
    const int64_t Ks[]      = { 64, 128 };
    const int64_t N         = 2;
    bool          ok        = true;
    int           cases_run = 0;

    for (int64_t M : Ms) {
        for (int64_t K : Ks) {
            const std::vector<fairy2i_tile64_lut_wtile_16> w0 = make_fairy2i_lut_wtiles(M, K, 3);
            const std::vector<fairy2i_tile64_lut_wtile_16> w1 = make_fairy2i_lut_wtiles(M, K, 11);
            std::vector<int8_t>                            lut;
            std::vector<float>                             scales;
            fill_fairy2i_lut_qgemm_inputs(lut, scales, N, K);

            std::vector<float> expected((size_t) N * (size_t) M * 2u, 0.0f);
            fairy2i_lut_qgemm_lut16_scalar_ref(M, K, N, w0, lut, scales, expected, true, false);
            fairy2i_lut_qgemm_lut16_scalar_ref(M, K, N, w1, lut, scales, expected, false, true);

            std::vector<float> actual((size_t) N * (size_t) M * 2u, 0.0f);
            ggml_fairy2i_tile64_lut_qgemm_lut16((int) M, (int) K, (int) N, w0.data(), lut.data(), scales.data(),
                                                actual.data(), (size_t) M * 2u * sizeof(float), 2u * sizeof(float),
                                                false, true, false);
            ggml_fairy2i_tile64_lut_qgemm_lut16((int) M, (int) K, (int) N, w1.data(), lut.data(), scales.data(),
                                                actual.data(), (size_t) M * 2u * sizeof(float), 2u * sizeof(float),
                                                false, false, true);

            char label[128];
            snprintf(label, sizeof(label), "Fairy2i LUT qgemm add M=%lld K=%lld N=%lld", (long long) M, (long long) K,
                     (long long) N);
            ok = compare_f32_pairs(label, actual, expected) && ok;
            ++cases_run;
        }
    }

    printf("  Fairy2i LUT qgemm add: %d cases - %s\n", cases_run, ok ? "PASS" : "FAIL");
    return ok;
}
#else
static bool test_fairy2i_lut_qgemm_add() {
    printf("  Fairy2i LUT qgemm add skipped: build lacks GGML_USE_FAIRY2I_CPU_LUT\n");
    return true;
}
#endif

static int8_t fairy2i_quant_i8(float x) {
    const int q = (int) nearbyintf(x);
    if (q < -127) {
        return -127;
    }
    if (q > 127) {
        return 127;
    }
    return (int8_t) q;
}

static void quantize_fairy2i_act_q16_64(
    std::vector<block_fairy2i_act_q16_64> & out, const std::vector<float> & x, int64_t N, int64_t K) {
    const int64_t blocks = K / QK_FAIRY2I_ACT_Q16_64;
    out.assign((size_t) N * (size_t) blocks, block_fairy2i_act_q16_64{});

    for (int64_t n = 0; n < N; ++n) {
        const float * x_row = x.data() + (size_t) n * (size_t) K;
        for (int64_t ib = 0; ib < blocks; ++ib) {
            block_fairy2i_act_q16_64 & q = out[(size_t) n * (size_t) blocks + (size_t) ib];
            float max_real = 1e-5f;
            float max_imag = 1e-5f;
            for (int j = 0; j < QK_FAIRY2I_ACT_Q16_64; ++j) {
                const ggml_bf16_t * value = (const ggml_bf16_t *) (x_row + ib * QK_FAIRY2I_ACT_Q16_64 + j);
                max_real = fmaxf(max_real, fabsf(GGML_BF16_TO_FP32(value[0])));
                max_imag = fmaxf(max_imag, fabsf(GGML_BF16_TO_FP32(value[1])));
            }

            const float iscale_real = 127.0f / max_real;
            const float iscale_imag = 127.0f / max_imag;
            q.d_real = GGML_FP32_TO_FP16(1.0f / iscale_real);
            q.d_imag = GGML_FP32_TO_FP16(1.0f / iscale_imag);

            for (int j = 0; j < QK_FAIRY2I_ACT_Q16_64; ++j) {
                const ggml_bf16_t * value = (const ggml_bf16_t *) (x_row + ib * QK_FAIRY2I_ACT_Q16_64 + j);
                q.x_real[j] = (uint8_t) fairy2i_quant_i8(iscale_real * GGML_BF16_TO_FP32(value[0]));
                q.x_imag[j] = (uint8_t) fairy2i_quant_i8(iscale_imag * GGML_BF16_TO_FP32(value[1]));
            }
        }
    }
}

static void fairy2i_accumulate_block_scalar(
    const block_fairy2i_tile64_v2 & w, const block_fairy2i_act_q16_64 & x, int32_t sums[4]) {
    for (int j = 0; j < QK_FAIRY2I_TILE64; ++j) {
        const uint8_t code = get_fairy2i_code(w, j);
        const int     xr   = (int) ((const int8_t *) x.x_real)[j];
        const int     xi   = (int) ((const int8_t *) x.x_imag)[j];

        switch (code) {
            case 0:
                sums[0] -= xr;
                sums[1] -= xi;
                break;
            case 1:
                sums[0] += xr;
                sums[1] += xi;
                break;
            case 2:
                sums[2] -= xr;
                sums[3] -= xi;
                break;
            case 3:
                sums[2] += xr;
                sums[3] += xi;
                break;
        }
    }
}

static bool compare_fairy2i_sums(const char * label, const int32_t actual[4][4], const int32_t expected[4][4]) {
    for (int branch = 0; branch < 4; ++branch) {
        for (int channel = 0; channel < 4; ++channel) {
            if (actual[branch][channel] != expected[branch][channel]) {
                fprintf(stderr, "%s mismatch branch=%d channel=%d actual=%d expected=%d\n", label, branch, channel,
                        actual[branch][channel], expected[branch][channel]);
                return false;
            }
        }
    }
    return true;
}

static block_fairy2i_act_q16_64 make_fairy2i_test_act_block(int salt) {
    block_fairy2i_act_q16_64 x{};
    x.d_real = GGML_FP32_TO_FP16(0.03125f);
    x.d_imag = GGML_FP32_TO_FP16(0.046875f);
    for (int i = 0; i < QK_FAIRY2I_ACT_Q16_64; ++i) {
        int vr = 0;
        int vi = 0;
        switch (i & 7) {
            case 0:
                vr = 0;
                break;
            case 1:
                vr = 1;
                break;
            case 2:
                vr = -1;
                break;
            case 3:
                vr = 63;
                break;
            case 4:
                vr = -63;
                break;
            case 5:
                vr = 127;
                break;
            case 6:
                vr = -127;
                break;
            default:
                vr = ((i * 17 + salt) % 255) - 127;
                break;
        }
        switch ((i + salt) & 7) {
            case 0:
                vi = 0;
                break;
            case 1:
                vi = -1;
                break;
            case 2:
                vi = 1;
                break;
            case 3:
                vi = -63;
                break;
            case 4:
                vi = 63;
                break;
            case 5:
                vi = -127;
                break;
            case 6:
                vi = 127;
                break;
            default:
                vi = ((i * 29 + 3 * salt) % 255) - 127;
                break;
        }
        x.x_real[i] = (uint8_t) (int8_t) vr;
        x.x_imag[i] = (uint8_t) (int8_t) vi;
    }
    return x;
}

static block_fairy2i_tile64_v2 make_fairy2i_test_weight_block(int pattern, int salt) {
    block_fairy2i_tile64_v2 w{};
    w.d_real = GGML_FP32_TO_FP16(0.019f + 0.002f * (float) (salt & 3));
    w.d_imag = GGML_FP32_TO_FP16(0.023f + 0.003f * (float) ((salt + 1) & 3));
    for (int i = 0; i < QK_FAIRY2I_TILE64; ++i) {
        uint8_t code = 0;
        if (pattern < 4) {
            code = (uint8_t) pattern;
        } else if (pattern == 4) {
            code = (uint8_t) (i & 3);
        } else {
            code = (uint8_t) ((i * 5 + salt * 3 + (i >> 2)) & 3);
        }
        set_fairy2i_code(w, i, code);
    }
    return w;
}

static void fairy2i_accumulate_four_scalar(const block_fairy2i_tile64_v2 &  u0,
                                           const block_fairy2i_tile64_v2 &  u1,
                                           const block_fairy2i_tile64_v2 &  w0,
                                           const block_fairy2i_tile64_v2 &  w1,
                                           const block_fairy2i_act_q16_64 & x,
                                           int32_t                          sums[4][4]) {
    fairy2i_accumulate_block_scalar(u0, x, sums[0]);
    fairy2i_accumulate_block_scalar(u1, x, sums[1]);
    fairy2i_accumulate_block_scalar(w0, x, sums[2]);
    fairy2i_accumulate_block_scalar(w1, x, sums[3]);
}

static bool test_fairy2i_arm_accumulate_neon() {
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (!ggml_fairy2i_tile64_w2_arm_neon_available()) {
        printf("  Fairy2i ARM NEON accumulate skipped: runtime lacks NEON\n");
        return true;
    }

    bool           ok          = true;
    const bool     has_dotprod = ggml_fairy2i_tile64_w2_arm_dotprod_available() && ggml_cpu_has_dotprod();
    scoped_env_var env_disable_dotprod("GGML_FAIRY2I_TEST_DISABLE_ARM_DOTPROD");
    for (int pattern = 0; pattern < 12; ++pattern) {
        const block_fairy2i_act_q16_64 x  = make_fairy2i_test_act_block(pattern * 13 + 11);
        const block_fairy2i_tile64_v2  u0 = make_fairy2i_test_weight_block(pattern, 1);
        const block_fairy2i_tile64_v2  u1 = make_fairy2i_test_weight_block(pattern + 1, 5);
        const block_fairy2i_tile64_v2  w0 = make_fairy2i_test_weight_block(pattern + 2, 9);
        const block_fairy2i_tile64_v2  w1 = make_fairy2i_test_weight_block(pattern + 3, 13);

        int32_t expected[4][4]       = {};
        int32_t actual_neon[4][4]    = {};
        int32_t actual_arm[4][4]     = {};
        int32_t actual_no_dot[4][4]  = {};
        int32_t actual_dotprod[4][4] = {};
        fairy2i_accumulate_four_scalar(u0, u1, w0, w1, x, expected);
        ggml_fairy2i_tile64_fuse_accumulate_block_four_neon(&u0, &u1, &w0, &w1, &x, actual_neon);
        if (!ggml_fairy2i_tile64_fuse_accumulate_block_four_arm(&u0, &u1, &w0, &w1, &x, actual_arm)) {
            fprintf(stderr, "Fairy2i ARM dispatcher unexpectedly declined pattern=%d\n", pattern);
            ok = false;
        }

        env_disable_dotprod.set("1");
        if (!ggml_fairy2i_tile64_fuse_accumulate_block_four_arm(&u0, &u1, &w0, &w1, &x, actual_no_dot)) {
            fprintf(stderr, "Fairy2i ARM NEON fallback unexpectedly declined pattern=%d\n", pattern);
            ok = false;
        }
        env_disable_dotprod.unset();

        if (has_dotprod) {
            ggml_fairy2i_tile64_fuse_accumulate_block_four_dotprod(&u0, &u1, &w0, &w1, &x, actual_dotprod);
        }

        char label[96];
        snprintf(label, sizeof(label), "Fairy2i ARM NEON accumulate pattern=%d", pattern);
        ok = compare_fairy2i_sums(label, actual_neon, expected) && ok;
        snprintf(label, sizeof(label), "Fairy2i ARM dispatcher accumulate pattern=%d", pattern);
        ok = compare_fairy2i_sums(label, actual_arm, expected) && ok;
        snprintf(label, sizeof(label), "Fairy2i ARM NEON fallback accumulate pattern=%d", pattern);
        ok = compare_fairy2i_sums(label, actual_no_dot, expected) && ok;
        if (has_dotprod) {
            snprintf(label, sizeof(label), "Fairy2i ARM dotprod accumulate pattern=%d", pattern);
            ok = compare_fairy2i_sums(label, actual_dotprod, expected) && ok;
        }
    }

    printf("  Fairy2i ARM accumulate helpers%s: %s\n", has_dotprod ? " (NEON+dotprod)" : " (NEON)",
           ok ? "PASS" : "FAIL");
    return ok;
#else
    printf("  Fairy2i ARM NEON accumulate skipped: non-ARM build\n");
    return true;
#endif
}

static void fairy2i_apply_branch(
    const block_fairy2i_tile64_v2 & w, const block_fairy2i_act_q16_64 & x, const int32_t sums[4], bool conjugate_input,
    float & acc_real, float & acc_imag) {
    const float xr = GGML_FP16_TO_FP32(x.d_real);
    const float xi = GGML_FP16_TO_FP32(x.d_imag);
    const float wr = GGML_FP16_TO_FP32(w.d_real);
    const float wi = GGML_FP16_TO_FP32(w.d_imag);

    if (conjugate_input) {
        acc_real += (float) sums[0] * (xr * wr) + (float) sums[3] * (xi * wi);
        acc_imag += (float) sums[2] * (xr * wi) - (float) sums[1] * (xi * wr);
    } else {
        acc_real += (float) sums[0] * (xr * wr) - (float) sums[3] * (xi * wi);
        acc_imag += (float) sums[2] * (xr * wi) + (float) sums[1] * (xi * wr);
    }
}

static std::vector<uint32_t> fairy2i_w2_scalar_reference(const fairy2i_w2_case & tc, const fairy2i_w2_data & data) {
    const int64_t blocks = tc.K / QK_FAIRY2I_TILE64;
    std::vector<block_fairy2i_act_q16_64> q_x;
    quantize_fairy2i_act_q16_64(q_x, data.x, tc.N, tc.K);

    std::vector<uint32_t> out((size_t) tc.M * (size_t) tc.N);
    for (int64_t n = 0; n < tc.N; ++n) {
        for (int64_t row = 0; row < tc.M; ++row) {
            float real = 0.0f;
            float imag = 0.0f;
            for (int64_t ib = 0; ib < blocks; ++ib) {
                const block_fairy2i_act_q16_64 & x =
                    q_x[(size_t) n * (size_t) blocks + (size_t) ib];
                const size_t w_idx = (size_t) row * (size_t) blocks + (size_t) ib;

                int32_t sums[4][4] = {};
                fairy2i_accumulate_block_scalar(data.u_s0[w_idx], x, sums[0]);
                fairy2i_accumulate_block_scalar(data.u_s1[w_idx], x, sums[1]);
                fairy2i_accumulate_block_scalar(data.w_s0[w_idx], x, sums[2]);
                fairy2i_accumulate_block_scalar(data.w_s1[w_idx], x, sums[3]);

                fairy2i_apply_branch(data.u_s0[w_idx], x, sums[0], false, real, imag);
                fairy2i_apply_branch(data.u_s1[w_idx], x, sums[1], false, real, imag);
                fairy2i_apply_branch(data.w_s0[w_idx], x, sums[2], true, real, imag);
                fairy2i_apply_branch(data.w_s1[w_idx], x, sums[3], true, real, imag);
            }

            if (tc.bias) {
                real += data.bias[(size_t) row];
                imag += data.bias[(size_t) row + (size_t) tc.M];
            }

            out[(size_t) n * (size_t) tc.M + (size_t) row] = pack_bf16_pair(real, imag);
        }
    }
    return out;
}

static bool run_fairy2i_w2_backend(std::vector<uint32_t> & out,
                                   const fairy2i_w2_case & tc,
                                   const fairy2i_w2_data & data,
                                   bool                    lut_enabled,
                                   bool                    force_scalar) {
    scoped_env_var env_lut("GGML_FAIRY2I_LUT");
    scoped_env_var env_impl("GGML_FAIRY2I_LUT_IMPL");
    scoped_env_var env_force_scalar("GGML_FAIRY2I_TEST_FORCE_SCALAR");
    scoped_env_var env_require_lut("GGML_FAIRY2I_TEST_REQUIRE_LUT");
    env_lut.set(lut_enabled ? "1" : "0");
    env_impl.set("lut16");
    env_require_lut.set(lut_enabled ? "1" : "0");
    if (force_scalar) {
        env_force_scalar.set("1");
    } else {
        env_force_scalar.unset();
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "failed to initialize CPU backend\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, 4);

    struct ggml_init_params params = {
        /*.mem_size   =*/32 * 1024 * 1024,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        fprintf(stderr, "failed to initialize ggml context\n");
        return false;
    }

    ggml_tensor * x    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.K, tc.N);
    ggml_tensor * u_s0 = ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_TILE64_V2, tc.K, tc.M);
    ggml_tensor * u_s1 = ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_TILE64_V2, tc.K, tc.M);
    ggml_tensor * w_s0 = ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_TILE64_V2, tc.K, tc.M);
    ggml_tensor * w_s1 = ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_TILE64_V2, tc.K, tc.M);
    ggml_tensor * bias = tc.bias ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * tc.M) : nullptr;
    ggml_tensor * y    = ggml_fairy2i_wide_linear_w2(ctx, x, u_s0, u_s1, w_s0, w_s1, bias);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "failed to allocate backend buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_set(x, data.x.data(), 0, data.x.size() * sizeof(float));
    ggml_backend_tensor_set(u_s0, data.u_s0.data(), 0, data.u_s0.size() * sizeof(block_fairy2i_tile64_v2));
    ggml_backend_tensor_set(u_s1, data.u_s1.data(), 0, data.u_s1.size() * sizeof(block_fairy2i_tile64_v2));
    ggml_backend_tensor_set(w_s0, data.w_s0.data(), 0, data.w_s0.size() * sizeof(block_fairy2i_tile64_v2));
    ggml_backend_tensor_set(w_s1, data.w_s1.data(), 0, data.w_s1.size() * sizeof(block_fairy2i_tile64_v2));
    if (bias) {
        ggml_backend_tensor_set(bias, data.bias.data(), 0, data.bias.size() * sizeof(float));
    }

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Fairy2i W2 graph compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<float> out_f32((size_t) tc.M * (size_t) tc.N);
    ggml_backend_tensor_get(y, out_f32.data(), 0, out_f32.size() * sizeof(float));

    out.resize(out_f32.size());
    memcpy(out.data(), out_f32.data(), out_f32.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

static bool compare_exact(const char * label, const std::vector<uint32_t> & actual, const std::vector<uint32_t> & expected) {
    if (actual.size() != expected.size()) {
        fprintf(stderr, "%s size mismatch: %zu vs %zu\n", label, actual.size(), expected.size());
        return false;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            float ar, ai, er, ei;
            unpack_bf16_pair(actual[i], ar, ai);
            unpack_bf16_pair(expected[i], er, ei);
            fprintf(stderr,
                    "%s mismatch at index %zu: actual=0x%08x expected=0x%08x actual=(%.7g,%.7g) "
                    "expected=(%.7g,%.7g)\n",
                    label, i, actual[i], expected[i], ar, ai, er, ei);
            return false;
        }
    }
    return true;
}

static bool compare_packed_complex(
    const char * label, const std::vector<uint32_t> & actual, const std::vector<uint32_t> & expected, float max_error) {
    if (actual.size() != expected.size()) {
        fprintf(stderr, "%s size mismatch: %zu vs %zu\n", label, actual.size(), expected.size());
        return false;
    }

    float  max_diff = 0.0f;
    size_t max_idx  = 0;
    int    max_ch   = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        float ar, ai, er, ei;
        unpack_bf16_pair(actual[i], ar, ai);
        unpack_bf16_pair(expected[i], er, ei);
        const float dr = fabsf(ar - er);
        const float di = fabsf(ai - ei);
        if (dr > max_diff) {
            max_diff = dr;
            max_idx  = i;
            max_ch   = 0;
        }
        if (di > max_diff) {
            max_diff = di;
            max_idx  = i;
            max_ch   = 1;
        }
    }

    if (max_diff > max_error) {
        float ar, ai, er, ei;
        unpack_bf16_pair(actual[max_idx], ar, ai);
        unpack_bf16_pair(expected[max_idx], er, ei);
        fprintf(stderr,
                "%s max diff %.7g exceeds %.7g at index %zu channel=%s actual=(%.7g,%.7g) expected=(%.7g,%.7g)\n",
                label, max_diff, max_error, max_idx, max_ch == 0 ? "real" : "imag", ar, ai, er, ei);
        return false;
    }

    return true;
}

static ggml_backend_dev_t find_opencl_test_device() {
    ggml_backend_load_all();

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev      = ggml_backend_dev_get(i);
        const char *       name     = ggml_backend_dev_name(dev);
        ggml_backend_reg_t reg      = ggml_backend_dev_backend_reg(dev);
        const char *       reg_name = ggml_backend_reg_name(reg);

        if ((name && strcmp(name, "GPUOpenCL") == 0) || (reg_name && strcmp(reg_name, "OpenCL") == 0)) {
            return dev;
        }
    }

    return nullptr;
}

static ggml_backend_dev_t find_metal_test_device() {
    ggml_backend_load_all();

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev      = ggml_backend_dev_get(i);
        const char *       name     = ggml_backend_dev_name(dev);
        ggml_backend_reg_t reg      = ggml_backend_dev_backend_reg(dev);
        const char *       reg_name = ggml_backend_reg_name(reg);

        if ((name && strcmp(name, "Metal") == 0) || (reg_name && strcmp(reg_name, "Metal") == 0)) {
            return dev;
        }
    }

    return nullptr;
}

static bool run_fairy2i_w2_metal_backend(std::vector<uint32_t> & out,
                                         ggml_backend_dev_t      dev,
                                         const fairy2i_w2_case & tc,
                                         const fairy2i_w2_data & data) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
    if (!backend) {
        fprintf(stderr, "failed to initialize Metal backend\n");
        return false;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/4 * 1024 * 1024,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        fprintf(stderr, "failed to initialize ggml context for Fairy2i Metal W2\n");
        return false;
    }

    ggml_tensor * x    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.K, tc.N);
    ggml_tensor * u_s0 = ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_TILE64_V2, tc.K, tc.M);
    ggml_tensor * u_s1 = ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_TILE64_V2, tc.K, tc.M);
    ggml_tensor * w_s0 = ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_TILE64_V2, tc.K, tc.M);
    ggml_tensor * w_s1 = ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_TILE64_V2, tc.K, tc.M);
    ggml_tensor * bias = tc.bias ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * tc.M) : nullptr;
    ggml_tensor * y    = ggml_fairy2i_wide_linear_w2(ctx, x, u_s0, u_s1, w_s0, w_s1, bias);
    ggml_set_name(y, "fairy2i_metal_wide_linear_w2");

    if (!ggml_backend_supports_op(backend, y)) {
        fprintf(stderr, "Metal does not support Fairy2i W2 M=%lld N=%lld K=%lld\n", (long long) tc.M, (long long) tc.N,
                (long long) tc.K);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "failed to allocate Metal backend buffer for Fairy2i W2\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_set(x, data.x.data(), 0, data.x.size() * sizeof(float));
    ggml_backend_tensor_set(u_s0, data.u_s0.data(), 0, data.u_s0.size() * sizeof(block_fairy2i_tile64_v2));
    ggml_backend_tensor_set(u_s1, data.u_s1.data(), 0, data.u_s1.size() * sizeof(block_fairy2i_tile64_v2));
    ggml_backend_tensor_set(w_s0, data.w_s0.data(), 0, data.w_s0.size() * sizeof(block_fairy2i_tile64_v2));
    ggml_backend_tensor_set(w_s1, data.w_s1.data(), 0, data.w_s1.size() * sizeof(block_fairy2i_tile64_v2));
    if (bias) {
        ggml_backend_tensor_set(bias, data.bias.data(), 0, data.bias.size() * sizeof(float));
    }

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Fairy2i Metal W2 graph compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_synchronize(backend);

    std::vector<float> out_f32((size_t) tc.M * (size_t) tc.N);
    ggml_backend_tensor_get(y, out_f32.data(), 0, out_f32.size() * sizeof(float));

    out.resize(out_f32.size());
    memcpy(out.data(), out_f32.data(), out_f32.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

static bool test_fairy2i_metal_wide_linear_w2() {
    printf("\n=== Fairy2i Metal W2 tests ===\n");

    ggml_backend_dev_t dev = find_metal_test_device();
    if (!dev) {
        printf("Metal backend not found; skipping Fairy2i Metal W2 tests.\n");
        return true;
    }

    printf("Metal device: %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));

    const std::vector<fairy2i_w2_case> cases = {
        { 7,  3, 128, true },
        { 17, 1, 256, true },
    };

    struct metal_w2_mode {
        const char * label;
        const char * lut;
        const char * stream;
        const char * block_sum;
        const char * prefill_tile4x4;
        const char * prefill_tile8x4;
        const char * prefill_act_q8;
    };

    const metal_w2_mode modes[] = {
        { "Fairy2i Metal W2",                 nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
        { "Fairy2i Metal W2 LUT",             "1",     nullptr, nullptr, nullptr, nullptr, nullptr },
        { "Fairy2i Metal W2 LUT stream",      nullptr, "1",     nullptr, nullptr, nullptr, nullptr },
        { "Fairy2i Metal W2 block sum",       nullptr, nullptr, "1",     nullptr, nullptr, nullptr },
        { "Fairy2i Metal W2 prefill tile4x4", nullptr, nullptr, "1",     "1",     nullptr, nullptr },
        { "Fairy2i Metal W2 prefill tile8x4", nullptr, nullptr, "1",     nullptr, "1",     nullptr },
        { "Fairy2i Metal W2 prefill act q8",  nullptr, nullptr, nullptr, nullptr, nullptr, "1"     },
    };

    scoped_env_var lut_env("GGML_METAL_FAIRY2I_W2_LUT");
    scoped_env_var stream_env("GGML_METAL_FAIRY2I_W2_LUT_STREAM");
    scoped_env_var block_sum_env("GGML_METAL_FAIRY2I_W2_BLOCK_SUM");
    scoped_env_var prefill_tile4x4_env("GGML_METAL_FAIRY2I_W2_PREFILL_TILE4X4");
    scoped_env_var prefill_tile8x4_env("GGML_METAL_FAIRY2I_W2_PREFILL_TILE8X4");
    scoped_env_var prefill_act_q8_env("GGML_METAL_FAIRY2I_W2_PREFILL_ACT_Q8");

    for (const metal_w2_mode & mode : modes) {
        if (mode.lut) {
            lut_env.set(mode.lut);
        } else {
            lut_env.unset();
        }
        if (mode.stream) {
            stream_env.set(mode.stream);
        } else {
            stream_env.unset();
        }
        if (mode.block_sum) {
            block_sum_env.set(mode.block_sum);
        } else {
            block_sum_env.unset();
        }
        if (mode.prefill_tile4x4) {
            prefill_tile4x4_env.set(mode.prefill_tile4x4);
        } else {
            prefill_tile4x4_env.unset();
        }
        if (mode.prefill_tile8x4) {
            prefill_tile8x4_env.set(mode.prefill_tile8x4);
        } else {
            prefill_tile8x4_env.unset();
        }
        if (mode.prefill_act_q8) {
            prefill_act_q8_env.set(mode.prefill_act_q8);
        } else {
            prefill_act_q8_env.unset();
        }

        for (const fairy2i_w2_case & tc : cases) {
            const fairy2i_w2_data       data = make_fairy2i_w2_data(tc);
            const std::vector<uint32_t> ref  = fairy2i_w2_scalar_reference(tc, data);

            std::vector<uint32_t> metal;
            if (!run_fairy2i_w2_metal_backend(metal, dev, tc, data)) {
                return false;
            }

            if (!compare_packed_complex(mode.label, metal, ref, 1e-2f)) {
                return false;
            }
        }
    }

    return true;
}

static bool check_fairy2i_opencl_mul_mat_support(ggml_backend_dev_t dev,
                                                 const char *       label,
                                                 const char *       fairy_gate_value,
                                                 enum ggml_type     weight_type,
                                                 enum ggml_type     act_type,
                                                 bool               weight_view,
                                                 bool               act_view,
                                                 int64_t            k,
                                                 int64_t            m,
                                                 int64_t            n,
                                                 bool               expected) {
    scoped_env_var env_fairy("GGML_OPENCL_FAIRY2I");
    if (fairy_gate_value) {
        env_fairy.set(fairy_gate_value);
    } else {
        env_fairy.unset();
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/256 * 1024,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to init ggml context for Fairy2i OpenCL support %s\n", label);
        return false;
    }

    ggml_tensor * w_base = ggml_new_tensor_2d(ctx, weight_type, weight_view ? k * 2 : k, m);
    ggml_tensor * x_base = ggml_new_tensor_2d(ctx, act_type, act_view ? k * 2 : k, n);
    ggml_tensor * w      = weight_view ? ggml_view_2d(ctx, w_base, k, m, w_base->nb[1], 0) : w_base;
    ggml_tensor * x      = act_view ? ggml_view_2d(ctx, x_base, k, n, x_base->nb[1], 0) : x_base;
    ggml_tensor * out    = ggml_mul_mat(ctx, w, x);

    const bool supported = ggml_backend_dev_supports_op(dev, out);
    ggml_free(ctx);

    if (supported != expected) {
        fprintf(stderr, "Fairy2i OpenCL support case '%s' expected %s, got %s\n",
                label, expected ? "supported" : "unsupported", supported ? "supported" : "unsupported");
        return false;
    }

    printf("  %-28s : %s\n", label, supported ? "supported" : "unsupported");
    return true;
}

static std::vector<uint32_t> fairy2i_tile64_mul_mat_scalar_reference(
    int64_t M, int64_t N, int64_t K, const std::vector<block_fairy2i_tile64_v2> & weights, const std::vector<float> & x) {
    const int64_t blocks = K / QK_FAIRY2I_TILE64;

    std::vector<block_fairy2i_act_q16_64> q_x;
    quantize_fairy2i_act_q16_64(q_x, x, N, K);

    std::vector<uint32_t> out((size_t) M * (size_t) N);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t row = 0; row < M; ++row) {
            float real = 0.0f;
            float imag = 0.0f;
            for (int64_t ib = 0; ib < blocks; ++ib) {
                const block_fairy2i_act_q16_64 & x_blk = q_x[(size_t) n * (size_t) blocks + (size_t) ib];
                const block_fairy2i_tile64_v2 &  w_blk = weights[(size_t) row * (size_t) blocks + (size_t) ib];

                int32_t sums[4] = {};
                fairy2i_accumulate_block_scalar(w_blk, x_blk, sums);
                fairy2i_apply_branch(w_blk, x_blk, sums, true, real, imag);
            }

            out[(size_t) n * (size_t) M + (size_t) row] = pack_bf16_pair(real, imag);
        }
    }
    return out;
}

static bool run_fairy2i_tile64_mul_mat_opencl(std::vector<uint32_t> &                         out,
                                              ggml_backend_dev_t                              dev,
                                              int64_t                                         M,
                                              int64_t                                         N,
                                              int64_t                                         K,
                                              const std::vector<block_fairy2i_tile64_v2> &    weights,
                                              const std::vector<float> &                      x) {
    scoped_env_var env_fairy("GGML_OPENCL_FAIRY2I");
    env_fairy.set("1");

    ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
    if (!backend) {
        fprintf(stderr, "failed to initialize OpenCL backend\n");
        return false;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/4 * 1024 * 1024,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        fprintf(stderr, "failed to initialize ggml context for Fairy2i OpenCL matmul\n");
        return false;
    }

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_TILE64_V2, K, M);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * y = ggml_mul_mat(ctx, w, a);
    ggml_set_name(y, "fairy2i_opencl_tile64_mul_mat");

    if (!ggml_backend_supports_op(backend, y)) {
        fprintf(stderr, "%s does not support Fairy2i tile64 MUL_MAT M=%lld N=%lld K=%lld\n",
                ggml_backend_name(backend), (long long) M, (long long) N, (long long) K);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "failed to allocate OpenCL backend buffer for Fairy2i tile64 MUL_MAT\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_set(w, weights.data(), 0, weights.size() * sizeof(block_fairy2i_tile64_v2));
    ggml_backend_tensor_set(a, x.data(), 0, x.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Fairy2i OpenCL tile64 MUL_MAT failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_synchronize(backend);

    std::vector<float> out_f32((size_t) M * (size_t) N);
    ggml_backend_tensor_get(y, out_f32.data(), 0, out_f32.size() * sizeof(float));

    out.resize(out_f32.size());
    memcpy(out.data(), out_f32.data(), out_f32.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

static bool run_fairy2i_opencl_tile64_compare_case(ggml_backend_dev_t dev, int64_t M, int64_t N, int64_t K) {
    std::vector<block_fairy2i_tile64_v2> weights;
    fill_fairy2i_weights(weights, M, K, 17);
    const std::vector<float> x = make_fairy2i_input(N, K);

    const std::vector<uint32_t> ref = fairy2i_tile64_mul_mat_scalar_reference(M, N, K, weights, x);

    std::vector<uint32_t> opencl;
    if (!run_fairy2i_tile64_mul_mat_opencl(opencl, dev, M, N, K, weights, x)) {
        return false;
    }

    char label[128];
    snprintf(label, sizeof(label), "Fairy2i OpenCL tile64 M=%lld N=%lld K=%lld",
             (long long) M, (long long) N, (long long) K);
    return compare_packed_complex(label, opencl, ref, 1e-2f);
}

static bool test_fairy2i_opencl_tile64_mul_mat() {
    printf("\n=== Fairy2i OpenCL tile64 MUL_MAT tests ===\n");

    ggml_backend_dev_t dev = find_opencl_test_device();
    if (!dev) {
        printf("OpenCL backend not found; skipping Fairy2i OpenCL tile64 tests.\n");
        return true;
    }

    printf("OpenCL device: %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));

    int num_failed = 0;
    auto support = [&](const char *   label,
                       const char *   fairy_gate_value,
                       enum ggml_type weight_type,
                       enum ggml_type act_type,
                       bool           weight_view,
                       bool           act_view,
                       int64_t        k,
                       int64_t        m,
                       int64_t        n,
                       bool           expected) {
        if (!check_fairy2i_opencl_mul_mat_support(dev, label, fairy_gate_value, weight_type, act_type, weight_view,
                                                  act_view, k, m, n, expected)) {
            ++num_failed;
        }
    };

    support("env-unset",       nullptr, GGML_TYPE_FAIRY2I_TILE64_V2, GGML_TYPE_F32, false, false, 256, 7, 1, false);
    support("env-zero",        "0",     GGML_TYPE_FAIRY2I_TILE64_V2, GGML_TYPE_F32, false, false, 256, 7, 1, false);
    support("k64",             "1",     GGML_TYPE_FAIRY2I_TILE64_V2, GGML_TYPE_F32, false, false, 64,  7, 1, true);
    support("k128",            "1",     GGML_TYPE_FAIRY2I_TILE64_V2, GGML_TYPE_F32, false, false, 128, 7, 1, true);
    support("k192",            "1",     GGML_TYPE_FAIRY2I_TILE64_V2, GGML_TYPE_F32, false, false, 192, 7, 1, true);
    support("k256",            "1",     GGML_TYPE_FAIRY2I_TILE64_V2, GGML_TYPE_F32, false, false, 256, 7, 1, true);
    support("act-f16",         "1",     GGML_TYPE_FAIRY2I_TILE64_V2, GGML_TYPE_F16, false, false, 256, 7, 1, false);
    support("weight-view",     "1",     GGML_TYPE_FAIRY2I_TILE64_V2, GGML_TYPE_F32, true,  false, 256, 7, 1, false);
    support("activation-view", "1",     GGML_TYPE_FAIRY2I_TILE64_V2, GGML_TYPE_F32, false, true,  256, 7, 1, false);

    const int64_t Ms[] = { 7, 23 };
    const int64_t Ns[] = { 1, 4 };
    const int64_t Ks[] = { 64, 128, 192, 256 };
    for (int64_t M : Ms) {
        for (int64_t N : Ns) {
            for (int64_t K : Ks) {
                if (!run_fairy2i_opencl_tile64_compare_case(dev, M, N, K)) {
                    ++num_failed;
                }
            }
        }
    }

    if (num_failed == 0) {
        printf("Fairy2i OpenCL tile64 MUL_MAT tests PASSED!\n");
    } else {
        printf("%d Fairy2i OpenCL tile64 MUL_MAT test(s) FAILED\n", num_failed);
    }

    return num_failed == 0;
}

static bool test_fairy2i_wide_linear_w2_variants() {
    const int64_t Ms[] = { 1, 7, 16, 17, 23, 32, 33 };
    const int64_t Ks[] = { 64, 128, 192, 256, 320, 1024 };
    const int64_t Ns[] = { 1, 2, 4, 8 };

    int  cases_run = 0;
    bool ok        = true;
    const bool compare_scalar_default =
        ggml_cpu_has_avx2() != 0 || ggml_cpu_has_neon() != 0 || ggml_cpu_has_dotprod() != 0;

    for (int64_t M : Ms) {
        for (int64_t K : Ks) {
            for (int64_t N : Ns) {
                for (bool with_bias : { false, true }) {
                    const fairy2i_w2_case tc = { M, N, K, with_bias };
                    const fairy2i_w2_data data = make_fairy2i_w2_data(tc);
                    const std::vector<uint32_t> ref = fairy2i_w2_scalar_reference(tc, data);

                    std::vector<uint32_t> direct;
                    std::vector<uint32_t> direct_scalar;
                    std::vector<uint32_t> lut;
                    if (!run_fairy2i_w2_backend(direct, tc, data, false, false)) {
                        return false;
                    }
                    if (compare_scalar_default && !run_fairy2i_w2_backend(direct_scalar, tc, data, false, true)) {
                        return false;
                    }
#if defined(GGML_USE_FAIRY2I_CPU_LUT)
                    if (!run_fairy2i_w2_backend(lut, tc, data, true, false)) {
                        return false;
                    }
#endif

                    char label[160];
                    snprintf(label, sizeof(label), "reference vs direct M=%lld N=%lld K=%lld bias=%d",
                             (long long) M, (long long) N, (long long) K, (int) with_bias);
                    ok = compare_exact(label, direct, ref) && ok;

                    if (compare_scalar_default) {
                        snprintf(label, sizeof(label), "forced scalar vs default M=%lld N=%lld K=%lld bias=%d",
                                 (long long) M, (long long) N, (long long) K, (int) with_bias);
                        ok = compare_exact(label, direct_scalar, direct) && ok;
                    }

#if defined(GGML_USE_FAIRY2I_CPU_LUT)
                    snprintf(label, sizeof(label), "direct vs LUT M=%lld N=%lld K=%lld bias=%d",
                             (long long) M, (long long) N, (long long) K, (int) with_bias);
                    ok = compare_packed_complex(label, lut, direct, 1e-2f) && ok;
#endif
                    ++cases_run;
                }
            }
        }
    }

    if (!compare_scalar_default) {
        printf("  Fairy2i W2 scalar/default fast-path compare skipped: CPU backend lacks AVX2/NEON/dotprod\n");
    }
#if !defined(GGML_USE_FAIRY2I_CPU_LUT)
    printf("  Fairy2i W2 LUT compare skipped: build lacks GGML_USE_FAIRY2I_CPU_LUT\n");
#endif
    printf("  Fairy2i W2 variant matrix: %d cases - %s\n", cases_run, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    ggml_cpu_init();

    printf("========================================\n");
    printf("Fairy2i CPU Unit Tests\n");
    printf("========================================\n");

    int num_failed = 0;
    if (!test_fairy2i_arm_accumulate_neon()) {
        fprintf(stderr, "Fairy2i ARM NEON accumulate helper FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_lut_quantize_arm_neon()) {
        fprintf(stderr, "Fairy2i LUT ARM quantize FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_lut_qgemm_add()) {
        fprintf(stderr, "Fairy2i LUT qgemm add FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_wide_linear_w2_variants()) {
        fprintf(stderr, "Fairy2i W2 variant matrix FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_metal_wide_linear_w2()) {
        fprintf(stderr, "Fairy2i Metal W2 FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_opencl_tile64_mul_mat()) {
        fprintf(stderr, "Fairy2i OpenCL tile64 MUL_MAT FAILED\n");
        ++num_failed;
    }

    printf("\n========================================\n");
    if (num_failed == 0) {
        printf("All tests PASSED!\n");
    } else {
        printf("%d test(s) FAILED\n", num_failed);
    }
    printf("========================================\n");

    return num_failed == 0 ? 0 : 1;
}
