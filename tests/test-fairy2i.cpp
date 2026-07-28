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
void ggml_fairy2i_tile64_fuse_accumulate_block_two_neon(const block_fairy2i_tile64_v2 *  u0,
                                                        const block_fairy2i_tile64_v2 *  w0,
                                                        const block_fairy2i_act_q16_64 * x,
                                                        int32_t                          sums[2][4]);
void ggml_fairy2i_tile64_fuse_accumulate_block_four_dotprod(const block_fairy2i_tile64_v2 *  u0,
                                                            const block_fairy2i_tile64_v2 *  u1,
                                                            const block_fairy2i_tile64_v2 *  w0,
                                                            const block_fairy2i_tile64_v2 *  w1,
                                                            const block_fairy2i_act_q16_64 * x,
                                                            int32_t                          sums[4][4]);
void ggml_fairy2i_tile64_fuse_accumulate_block_two_dotprod(const block_fairy2i_tile64_v2 *  u0,
                                                           const block_fairy2i_tile64_v2 *  w0,
                                                           const block_fairy2i_act_q16_64 * x,
                                                           int32_t                          sums[2][4]);
bool ggml_fairy2i_tile64_fuse_accumulate_block_four_arm(const block_fairy2i_tile64_v2 *  u0,
                                                        const block_fairy2i_tile64_v2 *  u1,
                                                        const block_fairy2i_tile64_v2 *  w0,
                                                        const block_fairy2i_tile64_v2 *  w1,
                                                        const block_fairy2i_act_q16_64 * x,
                                                        int32_t                          sums[4][4]);
bool ggml_fairy2i_tile64_fuse_accumulate_block_two_arm(const block_fairy2i_tile64_v2 *  u0,
                                                       const block_fairy2i_tile64_v2 *  w0,
                                                       const block_fairy2i_act_q16_64 * x,
                                                       int32_t                          sums[2][4]);
#endif
}

#if defined(GGML_USE_FAIRY2I_CPU_LUT)
#    include "../ggml/src/ggml-cpu/fairy2i/lut/ggml-fairy2i-lut.h"
void ggml_fairy2i_tile64_lut_quantize_block_q16_64_for_test(const float *              x,
                                                            block_fairy2i_act_q16_64 * y,
                                                            bool                       force_scalar);
int  ggml_fairy2i_wide_linear_w1_dynamic_tiles_hits_for_test(bool reset);
int  ggml_fairy2i_wide_linear_w1_dynamic_tiles_last_batch_for_test(void);
int  ggml_fairy2i_wide_linear_w2_dynamic_tiles_hits_for_test(bool reset);
int  ggml_fairy2i_wide_linear_w2_dynamic_tiles_last_batch_for_test(void);
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

struct fairy2i_w1_data {
    std::vector<float>                   x;
    std::vector<float>                   bias;
    std::vector<block_fairy2i_tile64_v2> u_s0;
    std::vector<block_fairy2i_tile64_v2> w_s0;
};

struct fairy2i_bundle_data {
    std::vector<uint8_t>     codes;
    std::vector<ggml_fp16_t> scales;
    int                      branches;
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

static void make_fairy2i_weights_bundle_compatible(std::vector<block_fairy2i_tile64_v2> & weights,
                                                   int64_t                                M,
                                                   int64_t                                K) {
    const int64_t blocks = K / QK_FAIRY2I_TILE64;
    for (int64_t mb = 0; mb < M / 64; ++mb) {
        for (int64_t kb = 0; kb < blocks; ++kb) {
            const block_fairy2i_tile64_v2 & scale_source = weights[(size_t) (mb * 64) * blocks + (size_t) kb];
            for (int64_t lane = 0; lane < 64; ++lane) {
                block_fairy2i_tile64_v2 & block = weights[(size_t) (mb * 64 + lane) * blocks + (size_t) kb];
                block.d_real                    = scale_source.d_real;
                block.d_imag                    = scale_source.d_imag;
            }
        }
    }
}

static fairy2i_bundle_data pack_fairy2i_bundle(
    const std::vector<const std::vector<block_fairy2i_tile64_v2> *> & branch_weights,
    int64_t                                                           M,
    int64_t                                                           K) {
    fairy2i_bundle_data result;
    result.branches          = (int) branch_weights.size();
    const int64_t blocks     = K / QK_FAIRY2I_TILE64;
    const int64_t tile_count = (M / 64) * blocks;
    result.codes.resize((size_t) tile_count * 64u * (size_t) result.branches * 16u);
    result.scales.resize((size_t) tile_count * (size_t) result.branches * 2u);

    for (int64_t mb = 0; mb < M / 64; ++mb) {
        for (int64_t kb = 0; kb < blocks; ++kb) {
            const int64_t physical_tile = mb * blocks + kb;
            for (int branch = 0; branch < result.branches; ++branch) {
                const auto &                    weights = *branch_weights[(size_t) branch];
                const block_fairy2i_tile64_v2 & scale_source =
                    weights[(size_t) (mb * 64) * (size_t) blocks + (size_t) kb];
                result.scales[((size_t) physical_tile * (size_t) result.branches + (size_t) branch) * 2u + 0] =
                    scale_source.d_real;
                result.scales[((size_t) physical_tile * (size_t) result.branches + (size_t) branch) * 2u + 1] =
                    scale_source.d_imag;

                for (int m16 = 0; m16 < 4; ++m16) {
                    for (int q4 = 0; q4 < 16; ++q4) {
                        const int64_t slot = q4 + 16 * m16;
                        for (int lane = 0; lane < 16; ++lane) {
                            const int64_t                   row = mb * 64 + m16 * 16 + lane;
                            const block_fairy2i_tile64_v2 & block =
                                weights[(size_t) row * (size_t) blocks + (size_t) kb];
                            const uint8_t p0 = (uint8_t) (get_fairy2i_code(block, q4 * 4 + 0) |
                                                          (get_fairy2i_code(block, q4 * 4 + 1) << 2));
                            const uint8_t p1 = (uint8_t) (get_fairy2i_code(block, q4 * 4 + 2) |
                                                          (get_fairy2i_code(block, q4 * 4 + 3) << 2));
                            const size_t  offset =
                                ((((size_t) physical_tile * 64u + (size_t) slot) * (size_t) result.branches +
                                  (size_t) branch) *
                                     16u +
                                 (size_t) lane);
                            result.codes[offset] = (uint8_t) (p0 | (p1 << 4));
                        }
                    }
                }
            }
        }
    }
    return result;
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

static fairy2i_w1_data make_fairy2i_w1_data(const fairy2i_w2_case & tc) {
    fairy2i_w1_data data;
    data.x = make_fairy2i_input(tc.N, tc.K);
    if (tc.bias) {
        data.bias = make_fairy2i_bias(tc.M);
    }
    fill_fairy2i_weights(data.u_s0, tc.M, tc.K, 1);
    fill_fairy2i_weights(data.w_s0, tc.M, tc.K, 9);
    return data;
}

static void share_fairy2i_w64_scales(std::vector<block_fairy2i_tile64_v2> & weights, int64_t M, int64_t K) {
    const int64_t blocks = K / QK_FAIRY2I_TILE64;
    for (int64_t row = 0; row < M; ++row) {
        const int64_t scale_row = (row / QK_FAIRY2I_TILE64) * QK_FAIRY2I_TILE64;
        for (int64_t ib = 0; ib < blocks; ++ib) {
            block_fairy2i_tile64_v2 &       dst = weights[(size_t) row * (size_t) blocks + (size_t) ib];
            const block_fairy2i_tile64_v2 & src = weights[(size_t) scale_row * (size_t) blocks + (size_t) ib];
            dst.d_real                          = src.d_real;
            dst.d_imag                          = src.d_imag;
        }
    }
}

static void share_fairy2i_w64_scales(fairy2i_w2_data & data, const fairy2i_w2_case & tc) {
    share_fairy2i_w64_scales(data.u_s0, tc.M, tc.K);
    share_fairy2i_w64_scales(data.u_s1, tc.M, tc.K);
    share_fairy2i_w64_scales(data.w_s0, tc.M, tc.K);
    share_fairy2i_w64_scales(data.w_s1, tc.M, tc.K);
}

static void share_fairy2i_w64_scales(fairy2i_w1_data & data, const fairy2i_w2_case & tc) {
    share_fairy2i_w64_scales(data.u_s0, tc.M, tc.K);
    share_fairy2i_w64_scales(data.w_s0, tc.M, tc.K);
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

static void mark_bytes(std::vector<uint8_t> & touched, size_t offset, size_t size) {
    std::fill(touched.begin() + offset, touched.begin() + offset + size, (uint8_t) 1);
}

static bool check_untouched_bytes(const char *                 label,
                                  const std::vector<uint8_t> & data,
                                  const std::vector<uint8_t> & touched,
                                  uint8_t                      sentinel) {
    for (size_t i = 0; i < data.size(); ++i) {
        if (!touched[i] && data[i] != sentinel) {
            fprintf(stderr, "%s sentinel mismatch byte=%zu actual=0x%02x expected=0x%02x\n", label, i,
                    (unsigned) data[i], (unsigned) sentinel);
            return false;
        }
    }
    return true;
}

static float fairy2i_pair_guard_seed_real(int64_t col, int64_t row) {
    return 0.125f + 0.015625f * (float) ((3 * col + row) % 11);
}

static float fairy2i_pair_guard_seed_imag(int64_t col, int64_t row) {
    return -0.1875f + 0.015625f * (float) ((5 * col + 2 * row) % 13);
}

static bool compare_fairy2i_pair_strided_f32(const char *                 label,
                                             const std::vector<uint8_t> & actual,
                                             const std::vector<float> &   expected,
                                             int64_t                      M,
                                             int64_t                      N,
                                             size_t                       dst_col_stride,
                                             size_t                       dst_row_stride) {
    for (int64_t col = 0; col < N; ++col) {
        for (int64_t row = 0; row < M; ++row) {
            const size_t  off = (size_t) col * dst_col_stride + (size_t) row * dst_row_stride;
            const float * out = (const float *) (actual.data() + off);
            const size_t  idx = ((size_t) col * (size_t) M + (size_t) row) * 2u;
            const float   dr  = fabsf(out[0] - expected[idx + 0]);
            const float   di  = fabsf(out[1] - expected[idx + 1]);
            if (dr > 1e-5f || di > 1e-5f) {
                fprintf(stderr,
                        "%s mismatch col=%lld row=%lld actual=(%.7g,%.7g) expected=(%.7g,%.7g) diff=(%.7g,%.7g)\n",
                        label, (long long) col, (long long) row, out[0], out[1], expected[idx + 0], expected[idx + 1],
                        dr, di);
                return false;
            }
        }
    }
    return true;
}

static bool compare_fairy2i_pair_strided_bf16(const char *                 label,
                                              const std::vector<uint8_t> & actual,
                                              const std::vector<float> &   expected,
                                              int64_t                      M,
                                              int64_t                      N,
                                              size_t                       dst_col_stride,
                                              size_t                       dst_row_stride) {
    for (int64_t col = 0; col < N; ++col) {
        for (int64_t row = 0; row < M; ++row) {
            const size_t off = (size_t) col * dst_col_stride + (size_t) row * dst_row_stride;
            ggml_bf16_t  pair[2];
            memcpy(pair, actual.data() + off, sizeof(pair));
            const float  ar  = GGML_BF16_TO_FP32(pair[0]);
            const float  ai  = GGML_BF16_TO_FP32(pair[1]);
            const size_t idx = ((size_t) col * (size_t) M + (size_t) row) * 2u;
            const float  dr  = fabsf(ar - expected[idx + 0]);
            const float  di  = fabsf(ai - expected[idx + 1]);
            if (dr > 2e-2f || di > 2e-2f) {
                fprintf(stderr,
                        "%s bf16 mismatch col=%lld row=%lld actual=(%.7g,%.7g) expected=(%.7g,%.7g) "
                        "diff=(%.7g,%.7g)\n",
                        label, (long long) col, (long long) row, ar, ai, expected[idx + 0], expected[idx + 1], dr, di);
                return false;
            }
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

static bool test_fairy2i_lut_qgemm_pair_extreme_same_lane() {
#    if defined(__aarch64__) && defined(__ARM_NEON)
    const int64_t M                  = 17;
    const int64_t K                  = 64;
    const int64_t N                  = 2;
    const int64_t blocks             = K / QK_FAIRY2I_TILE64;
    const int64_t groups             = blocks * QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK;
    const uint8_t selected_lut_index = 7;
    const size_t  dst_size           = (size_t) N * (size_t) M * 2u;
    bool          ok                 = true;
    int           cases_run          = 0;

    for (const int8_t lut_value : { (int8_t) 126, (int8_t) -126 }) {
        std::vector<fairy2i_tile64_lut_wtile_16> wt0 = make_fairy2i_lut_wtiles(M, K, 23);
        std::vector<fairy2i_tile64_lut_wtile_16> wt1 = make_fairy2i_lut_wtiles(M, K, 31);

        for (fairy2i_tile64_lut_wtile_16 & wt : wt0) {
            for (int lane = 0; lane < 16; ++lane) {
                wt.d_real[lane] = GGML_FP32_TO_FP16(1.0f);
                wt.d_imag[lane] = GGML_FP32_TO_FP16(0.0f);
            }
            for (int byte_idx = 0; byte_idx < QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK / 2; ++byte_idx) {
                for (int lane = 0; lane < 16; ++lane) {
                    wt.qs[byte_idx][lane] = (uint8_t) (selected_lut_index | (selected_lut_index << 4));
                }
            }
        }
        for (fairy2i_tile64_lut_wtile_16 & wt : wt1) {
            for (int lane = 0; lane < 16; ++lane) {
                wt.d_real[lane] = GGML_FP32_TO_FP16(1.0f);
                wt.d_imag[lane] = GGML_FP32_TO_FP16(0.0f);
            }
            for (int byte_idx = 0; byte_idx < QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK / 2; ++byte_idx) {
                for (int lane = 0; lane < 16; ++lane) {
                    wt.qs[byte_idx][lane] = (uint8_t) (selected_lut_index | (selected_lut_index << 4));
                }
            }
        }

        std::vector<int8_t> lut((size_t) N * (size_t) groups * 64u, 0);
        for (int64_t col = 0; col < N; ++col) {
            int8_t * lut_col = lut.data() + (size_t) col * (size_t) groups * 64u;
            for (int64_t group = 0; group < groups; ++group) {
                int8_t * tbl = lut_col + (size_t) group * 64u;
                for (int channel = 0; channel < 4; ++channel) {
                    tbl[channel * 16 + selected_lut_index] = lut_value;
                }
            }
        }

        std::vector<float> scales((size_t) N * (size_t) blocks * 2u, 1.0f);
        std::vector<float> expected(dst_size, 0.0f);
        std::vector<float> actual(dst_size, 0.0f);
        for (size_t i = 0; i < dst_size; ++i) {
            const float seed = (i & 1u) ? -0.5f : 0.25f;
            expected[i]      = seed;
            actual[i]        = seed;
        }

        fairy2i_lut_qgemm_lut16_scalar_ref(M, K, N, wt0, lut, scales, expected, false, true);
        fairy2i_lut_qgemm_lut16_scalar_ref(M, K, N, wt1, lut, scales, expected, false, true);

        ggml_fairy2i_tile64_lut_qgemm_pair_lut16((int) M, (int) K, (int) N, wt0.data(), wt1.data(), lut.data(),
                                                 scales.data(), actual.data(), (size_t) M * 2u * sizeof(float),
                                                 2u * sizeof(float), false, false, true);

        char label[160];
        snprintf(label, sizeof(label), "Fairy2i LUT qgemm pair same-lane value=%d M=%lld K=%lld N=%lld",
                 (int) lut_value, (long long) M, (long long) K, (long long) N);
        ok = compare_f32_pairs(label, actual, expected) && ok;
        ++cases_run;
    }

    printf("  Fairy2i LUT qgemm pair same-lane extremes: %d cases - %s\n", cases_run, ok ? "PASS" : "FAIL");
    return ok;
#    else
    printf("  Fairy2i LUT qgemm pair same-lane extremes skipped: non-ARM build\n");
    return true;
#    endif
}

static bool test_fairy2i_lut_qgemm_pair_layout_guardrails() {
    const int64_t Ms[]      = { 1, 7, 16, 17, 32, 33 };
    const int64_t Ks[]      = { 64, 128 };
    const int64_t N         = 2;
    const uint8_t sentinel  = 0xa5;
    bool          ok        = true;
    int           cases_run = 0;

    struct pair_case {
        bool pack_bf16;
        bool negate_imag_scale;
        bool add;
        int  salt0;
        int  salt1;
    };

    const pair_case cases[] = {
        { false, true,  false, 41, 47 },
        { false, false, true,  53, 59 },
        { true,  true,  false, 61, 67 },
        { true,  false, true,  71, 73 },
    };

    for (int64_t M : Ms) {
        for (int64_t K : Ks) {
            std::vector<int8_t> lut;
            std::vector<float>  scales;
            fill_fairy2i_lut_qgemm_inputs(lut, scales, N, K);

            for (const pair_case & tc : cases) {
                const std::vector<fairy2i_tile64_lut_wtile_16> w0 = make_fairy2i_lut_wtiles(M, K, tc.salt0);
                const std::vector<fairy2i_tile64_lut_wtile_16> w1 = make_fairy2i_lut_wtiles(M, K, tc.salt1);
                std::vector<float>                             pair_out((size_t) N * (size_t) M * 2u, 0.0f);
                fairy2i_lut_qgemm_lut16_scalar_ref(M, K, N, w0, lut, scales, pair_out, tc.negate_imag_scale, false);
                fairy2i_lut_qgemm_lut16_scalar_ref(M, K, N, w1, lut, scales, pair_out, tc.negate_imag_scale, true);

                const size_t         elem_size      = tc.pack_bf16 ? sizeof(ggml_bf16_t) : sizeof(float);
                const size_t         output_size    = 2u * elem_size;
                const size_t         dst_row_stride = 4u * elem_size;
                const size_t         dst_col_stride = ((size_t) M + 3u) * dst_row_stride;
                std::vector<uint8_t> actual((size_t) N * dst_col_stride, sentinel);
                std::vector<uint8_t> touched(actual.size(), 0);
                std::vector<float>   expected((size_t) N * (size_t) M * 2u, 0.0f);

                for (int64_t col = 0; col < N; ++col) {
                    for (int64_t row = 0; row < M; ++row) {
                        const size_t off = (size_t) col * dst_col_stride + (size_t) row * dst_row_stride;
                        const size_t idx = ((size_t) col * (size_t) M + (size_t) row) * 2u;
                        mark_bytes(touched, off, output_size);
                        if (tc.add) {
                            const float seed_r = fairy2i_pair_guard_seed_real(col, row);
                            const float seed_i = fairy2i_pair_guard_seed_imag(col, row);
                            if (tc.pack_bf16) {
                                ggml_bf16_t pair[2] = {
                                    GGML_FP32_TO_BF16(seed_r),
                                    GGML_FP32_TO_BF16(seed_i),
                                };
                                memcpy(actual.data() + off, pair, sizeof(pair));
                                expected[idx + 0] = GGML_BF16_TO_FP32(pair[0]);
                                expected[idx + 1] = GGML_BF16_TO_FP32(pair[1]);
                            } else {
                                float pair[2] = { seed_r, seed_i };
                                memcpy(actual.data() + off, pair, sizeof(pair));
                                expected[idx + 0] = seed_r;
                                expected[idx + 1] = seed_i;
                            }
                        }
                    }
                }
                for (size_t i = 0; i < pair_out.size(); ++i) {
                    expected[i] += pair_out[i];
                }

                ggml_fairy2i_tile64_lut_qgemm_pair_lut16((int) M, (int) K, (int) N, w0.data(), w1.data(), lut.data(),
                                                         scales.data(), (float *) actual.data(), dst_col_stride,
                                                         dst_row_stride, tc.pack_bf16, tc.negate_imag_scale, tc.add);

                char label[192];
                snprintf(label, sizeof(label),
                         "Fairy2i LUT qgemm pair layout M=%lld K=%lld N=%lld pack_bf16=%d negate=%d add=%d",
                         (long long) M, (long long) K, (long long) N, tc.pack_bf16 ? 1 : 0,
                         tc.negate_imag_scale ? 1 : 0, tc.add ? 1 : 0);
                if (tc.pack_bf16) {
                    ok = compare_fairy2i_pair_strided_bf16(label, actual, expected, M, N, dst_col_stride,
                                                           dst_row_stride) &&
                         ok;
                } else {
                    ok = compare_fairy2i_pair_strided_f32(label, actual, expected, M, N, dst_col_stride,
                                                          dst_row_stride) &&
                         ok;
                }
                ok = check_untouched_bytes(label, actual, touched, sentinel) && ok;
                ++cases_run;
            }
        }
    }

    printf("  Fairy2i LUT qgemm pair layout guardrails: %d cases - %s\n", cases_run, ok ? "PASS" : "FAIL");
    return ok;
}
#else
static bool test_fairy2i_lut_qgemm_add() {
    printf("  Fairy2i LUT qgemm add skipped: build lacks GGML_USE_FAIRY2I_CPU_LUT\n");
    return true;
}

static bool test_fairy2i_lut_qgemm_pair_extreme_same_lane() {
    printf("  Fairy2i LUT qgemm pair same-lane extremes skipped: build lacks GGML_USE_FAIRY2I_CPU_LUT\n");
    return true;
}

static bool test_fairy2i_lut_qgemm_pair_layout_guardrails() {
    printf("  Fairy2i LUT qgemm pair layout guardrails skipped: build lacks GGML_USE_FAIRY2I_CPU_LUT\n");
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

static bool compare_fairy2i_sums_two(const char * label, const int32_t actual[2][4], const int32_t expected[2][4]) {
    for (int branch = 0; branch < 2; ++branch) {
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

static void fairy2i_accumulate_two_scalar(const block_fairy2i_tile64_v2 &  u0,
                                          const block_fairy2i_tile64_v2 &  w0,
                                          const block_fairy2i_act_q16_64 & x,
                                          int32_t                          sums[2][4]) {
    fairy2i_accumulate_block_scalar(u0, x, sums[0]);
    fairy2i_accumulate_block_scalar(w0, x, sums[1]);
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
        int32_t expected_two[2][4]       = {};
        int32_t actual_two_neon[2][4]    = {};
        int32_t actual_two_arm[2][4]     = {};
        int32_t actual_two_no_dot[2][4]  = {};
        int32_t actual_two_dotprod[2][4] = {};
        fairy2i_accumulate_four_scalar(u0, u1, w0, w1, x, expected);
        fairy2i_accumulate_two_scalar(u0, w0, x, expected_two);
        ggml_fairy2i_tile64_fuse_accumulate_block_four_neon(&u0, &u1, &w0, &w1, &x, actual_neon);
        ggml_fairy2i_tile64_fuse_accumulate_block_two_neon(&u0, &w0, &x, actual_two_neon);
        if (!ggml_fairy2i_tile64_fuse_accumulate_block_four_arm(&u0, &u1, &w0, &w1, &x, actual_arm)) {
            fprintf(stderr, "Fairy2i ARM dispatcher unexpectedly declined pattern=%d\n", pattern);
            ok = false;
        }
        if (!ggml_fairy2i_tile64_fuse_accumulate_block_two_arm(&u0, &w0, &x, actual_two_arm)) {
            fprintf(stderr, "Fairy2i ARM W1 dispatcher unexpectedly declined pattern=%d\n", pattern);
            ok = false;
        }

        env_disable_dotprod.set("1");
        if (!ggml_fairy2i_tile64_fuse_accumulate_block_four_arm(&u0, &u1, &w0, &w1, &x, actual_no_dot)) {
            fprintf(stderr, "Fairy2i ARM NEON fallback unexpectedly declined pattern=%d\n", pattern);
            ok = false;
        }
        if (!ggml_fairy2i_tile64_fuse_accumulate_block_two_arm(&u0, &w0, &x, actual_two_no_dot)) {
            fprintf(stderr, "Fairy2i ARM W1 NEON fallback unexpectedly declined pattern=%d\n", pattern);
            ok = false;
        }
        env_disable_dotprod.unset();

        if (has_dotprod) {
            ggml_fairy2i_tile64_fuse_accumulate_block_four_dotprod(&u0, &u1, &w0, &w1, &x, actual_dotprod);
            ggml_fairy2i_tile64_fuse_accumulate_block_two_dotprod(&u0, &w0, &x, actual_two_dotprod);
        }

        char label[96];
        snprintf(label, sizeof(label), "Fairy2i ARM NEON accumulate pattern=%d", pattern);
        ok = compare_fairy2i_sums(label, actual_neon, expected) && ok;
        snprintf(label, sizeof(label), "Fairy2i ARM W1 NEON accumulate pattern=%d", pattern);
        ok = compare_fairy2i_sums_two(label, actual_two_neon, expected_two) && ok;
        snprintf(label, sizeof(label), "Fairy2i ARM dispatcher accumulate pattern=%d", pattern);
        ok = compare_fairy2i_sums(label, actual_arm, expected) && ok;
        snprintf(label, sizeof(label), "Fairy2i ARM W1 dispatcher accumulate pattern=%d", pattern);
        ok = compare_fairy2i_sums_two(label, actual_two_arm, expected_two) && ok;
        snprintf(label, sizeof(label), "Fairy2i ARM NEON fallback accumulate pattern=%d", pattern);
        ok = compare_fairy2i_sums(label, actual_no_dot, expected) && ok;
        snprintf(label, sizeof(label), "Fairy2i ARM W1 NEON fallback accumulate pattern=%d", pattern);
        ok = compare_fairy2i_sums_two(label, actual_two_no_dot, expected_two) && ok;
        if (has_dotprod) {
            snprintf(label, sizeof(label), "Fairy2i ARM dotprod accumulate pattern=%d", pattern);
            ok = compare_fairy2i_sums(label, actual_dotprod, expected) && ok;
            snprintf(label, sizeof(label), "Fairy2i ARM W1 dotprod accumulate pattern=%d", pattern);
            ok = compare_fairy2i_sums_two(label, actual_two_dotprod, expected_two) && ok;
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

static std::vector<uint32_t> fairy2i_w1_scalar_reference(const fairy2i_w2_case & tc, const fairy2i_w1_data & data) {
    const int64_t                         blocks = tc.K / QK_FAIRY2I_TILE64;
    std::vector<block_fairy2i_act_q16_64> q_x;
    quantize_fairy2i_act_q16_64(q_x, data.x, tc.N, tc.K);

    std::vector<uint32_t> out((size_t) tc.M * (size_t) tc.N);
    for (int64_t n = 0; n < tc.N; ++n) {
        for (int64_t row = 0; row < tc.M; ++row) {
            float real = 0.0f;
            float imag = 0.0f;
            for (int64_t ib = 0; ib < blocks; ++ib) {
                const block_fairy2i_act_q16_64 & x     = q_x[(size_t) n * (size_t) blocks + (size_t) ib];
                const size_t                     w_idx = (size_t) row * (size_t) blocks + (size_t) ib;

                int32_t sums[2][4] = {};
                fairy2i_accumulate_block_scalar(data.u_s0[w_idx], x, sums[0]);
                fairy2i_accumulate_block_scalar(data.w_s0[w_idx], x, sums[1]);

                fairy2i_apply_branch(data.u_s0[w_idx], x, sums[0], false, real, imag);
                fairy2i_apply_branch(data.w_s0[w_idx], x, sums[1], true, real, imag);
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

static bool run_fairy2i_w1_backend(std::vector<uint32_t> & out,
                                   const fairy2i_w2_case & tc,
                                   const fairy2i_w1_data & data,
                                   const char *            lut_env,
                                   const char *            lut_impl_env,
                                   bool                    require_lut,
                                   bool                    force_scalar,
                                   int                     n_threads              = 4,
                                   const char *            dynamic_tiles_env      = nullptr,
                                   const char *            dynamic_tile_batch_env = nullptr) {
    scoped_env_var env_lut("GGML_FAIRY2I_LUT");
    scoped_env_var env_impl("GGML_FAIRY2I_LUT_IMPL");
    scoped_env_var env_force_scalar("GGML_FAIRY2I_TEST_FORCE_SCALAR");
    scoped_env_var env_require_lut("GGML_FAIRY2I_TEST_REQUIRE_LUT");
    scoped_env_var env_dynamic_tiles("GGML_FAIRY2I_W1_DYNAMIC_TILES");
    scoped_env_var env_dynamic_tile_batch("GGML_FAIRY2I_W1_DYNAMIC_TILE_BATCH");
    if (lut_env) {
        env_lut.set(lut_env);
    } else {
        env_lut.unset();
    }
    if (lut_impl_env) {
        env_impl.set(lut_impl_env);
    } else {
        env_impl.unset();
    }
    env_require_lut.set(require_lut ? "1" : "0");
    if (force_scalar) {
        env_force_scalar.set("1");
    } else {
        env_force_scalar.unset();
    }
    if (dynamic_tiles_env) {
        env_dynamic_tiles.set(dynamic_tiles_env);
    } else {
        env_dynamic_tiles.unset();
    }
    if (dynamic_tile_batch_env) {
        env_dynamic_tile_batch.set(dynamic_tile_batch_env);
    } else {
        env_dynamic_tile_batch.unset();
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "failed to initialize CPU backend\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, n_threads);

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
    ggml_tensor * w_s0 = ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_TILE64_V2, tc.K, tc.M);
    ggml_tensor * bias = tc.bias ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * tc.M) : nullptr;
    ggml_tensor * y    = ggml_fairy2i_wide_linear_w1(ctx, x, u_s0, w_s0, bias);

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
    ggml_backend_tensor_set(w_s0, data.w_s0.data(), 0, data.w_s0.size() * sizeof(block_fairy2i_tile64_v2));
    if (bias) {
        ggml_backend_tensor_set(bias, data.bias.data(), 0, data.bias.size() * sizeof(float));
    }

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Fairy2i W1 graph compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<float> out_f32((size_t) tc.M * (size_t) tc.N);
    ggml_backend_tensor_get(y, out_f32.data(), 0, out_f32.size() * sizeof(float));

    out.resize(out_f32.size());
    for (size_t i = 0; i < out_f32.size(); ++i) {
        memcpy(&out[i], &out_f32[i], sizeof(uint32_t));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

static bool run_fairy2i_w2_backend(std::vector<uint32_t> & out,
                                   const fairy2i_w2_case & tc,
                                   const fairy2i_w2_data & data,
                                   const char *            lut_env,
                                   const char *            lut_impl_env,
                                   bool                    require_lut,
                                   bool                    force_scalar,
                                   int                     n_threads              = 4,
                                   const char *            dynamic_tiles_env      = nullptr,
                                   const char *            dynamic_tile_batch_env = nullptr) {
    scoped_env_var env_lut("GGML_FAIRY2I_LUT");
    scoped_env_var env_impl("GGML_FAIRY2I_LUT_IMPL");
    scoped_env_var env_force_scalar("GGML_FAIRY2I_TEST_FORCE_SCALAR");
    scoped_env_var env_require_lut("GGML_FAIRY2I_TEST_REQUIRE_LUT");
    scoped_env_var env_dynamic_tiles("GGML_FAIRY2I_W2_DYNAMIC_TILES");
    scoped_env_var env_dynamic_tile_batch("GGML_FAIRY2I_W2_DYNAMIC_TILE_BATCH");
    if (lut_env) {
        env_lut.set(lut_env);
    } else {
        env_lut.unset();
    }
    if (lut_impl_env) {
        env_impl.set(lut_impl_env);
    } else {
        env_impl.unset();
    }
    env_require_lut.set(require_lut ? "1" : "0");
    if (force_scalar) {
        env_force_scalar.set("1");
    } else {
        env_force_scalar.unset();
    }
    if (dynamic_tiles_env) {
        env_dynamic_tiles.set(dynamic_tiles_env);
    } else {
        env_dynamic_tiles.unset();
    }
    if (dynamic_tile_batch_env) {
        env_dynamic_tile_batch.set(dynamic_tile_batch_env);
    } else {
        env_dynamic_tile_batch.unset();
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "failed to initialize CPU backend\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, n_threads);

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

static bool run_fairy2i_bundle_backend(std::vector<uint32_t> &     out,
                                       const fairy2i_w2_case &     tc,
                                       const std::vector<float> &  x_data,
                                       const std::vector<float> &  bias_data,
                                       const fairy2i_bundle_data & bundle_data,
                                       bool                        is_w1,
                                       const char *                lut_env,
                                       const char *                lut_impl_env,
                                       bool                        force_scalar,
                                       int                         n_threads,
                                       const char *                dynamic_tiles_env      = nullptr,
                                       const char *                dynamic_tile_batch_env = nullptr,
                                       bool                        expect_success         = true) {
    scoped_env_var env_lut("GGML_FAIRY2I_LUT");
    scoped_env_var env_impl("GGML_FAIRY2I_LUT_IMPL");
    scoped_env_var env_force_scalar("GGML_FAIRY2I_TEST_FORCE_SCALAR");
    scoped_env_var env_require_lut("GGML_FAIRY2I_TEST_REQUIRE_LUT");
    scoped_env_var env_dynamic_tiles(is_w1 ? "GGML_FAIRY2I_W1_DYNAMIC_TILES" : "GGML_FAIRY2I_W2_DYNAMIC_TILES");
    scoped_env_var env_dynamic_tile_batch(is_w1 ? "GGML_FAIRY2I_W1_DYNAMIC_TILE_BATCH" :
                                                  "GGML_FAIRY2I_W2_DYNAMIC_TILE_BATCH");
    if (lut_env) {
        env_lut.set(lut_env);
    } else {
        env_lut.unset();
    }
    if (lut_impl_env) {
        env_impl.set(lut_impl_env);
    } else {
        env_impl.unset();
    }
    if (force_scalar) {
        env_force_scalar.set("1");
    } else {
        env_force_scalar.unset();
    }
    env_require_lut.set(expect_success ? "1" : "0");
    if (dynamic_tiles_env) {
        env_dynamic_tiles.set(dynamic_tiles_env);
    } else {
        env_dynamic_tiles.unset();
    }
    if (dynamic_tile_batch_env) {
        env_dynamic_tile_batch.set(dynamic_tile_batch_env);
    } else {
        env_dynamic_tile_batch.unset();
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "failed to initialize CPU backend for Fairy2i bundle\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, n_threads);

    struct ggml_init_params params = {
        /*.mem_size   =*/32 * 1024 * 1024,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return false;
    }

    const int64_t physical_tiles = (tc.M / 64) * (tc.K / 64);
    ggml_tensor * x              = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.K, tc.N);
    ggml_tensor * codes =
        ggml_new_tensor_4d(ctx, GGML_TYPE_FAIRY2I_BUNDLE_CODES, 16, bundle_data.branches, 64, physical_tiles);
    ggml_tensor * scales = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 2, bundle_data.branches, physical_tiles);
    ggml_tensor * bias   = tc.bias ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * tc.M) : nullptr;
    ggml_tensor * y      = is_w1 ? ggml_fairy2i_wide_linear_w1_bundle(ctx, x, codes, scales, bias, tc.M, tc.K) :
                                   ggml_fairy2i_wide_linear_w2_bundle(ctx, x, codes, scales, bias, tc.M, tc.K);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_set(x, x_data.data(), 0, x_data.size() * sizeof(float));
    ggml_backend_tensor_set(codes, bundle_data.codes.data(), 0, bundle_data.codes.size());
    ggml_backend_tensor_set(scales, bundle_data.scales.data(), 0, bundle_data.scales.size() * sizeof(ggml_fp16_t));
    if (bias) {
        ggml_backend_tensor_set(bias, bias_data.data(), 0, bias_data.size() * sizeof(float));
    }

    bool              ok     = codes->extra == nullptr && scales->extra == nullptr;
    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    ok                       = (expect_success ? status == GGML_STATUS_SUCCESS : status != GGML_STATUS_SUCCESS) && ok;
    ok                       = codes->extra == nullptr && scales->extra == nullptr && ok;
    if (expect_success && status == GGML_STATUS_SUCCESS) {
        std::vector<float> out_f32((size_t) tc.M * (size_t) tc.N);
        ggml_backend_tensor_get(y, out_f32.data(), 0, out_f32.size() * sizeof(float));
        out.resize(out_f32.size());
        memcpy(out.data(), out_f32.data(), out_f32.size() * sizeof(float));
    } else if (expect_success) {
        fprintf(stderr, "Fairy2i bundle graph compute failed: %s\n", ggml_status_to_string(status));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok;
}

static bool compare_exact(const char *                  label,
                          const std::vector<uint32_t> & actual,
                          const std::vector<uint32_t> & expected);
static bool compare_packed_complex(const char *                  label,
                                   const std::vector<uint32_t> & actual,
                                   const std::vector<uint32_t> & expected,
                                   float                         max_error);

static bool check_fairy2i_w1_dynamic_tiles_hit(const char * label, bool expected) {
#if defined(GGML_USE_FAIRY2I_CPU_LUT)
    const int hits = ggml_fairy2i_wide_linear_w1_dynamic_tiles_hits_for_test(false);
    if ((hits > 0) != expected) {
        fprintf(stderr, "%s dynamic hit mismatch: hits=%d expected=%d\n", label, hits, expected ? 1 : 0);
        return false;
    }
    return true;
#else
    (void) label;
    (void) expected;
    return true;
#endif
}

static bool check_fairy2i_w1_dynamic_tiles_last_batch(const char * label, int expected) {
#if defined(GGML_USE_FAIRY2I_CPU_LUT)
    const int batch = ggml_fairy2i_wide_linear_w1_dynamic_tiles_last_batch_for_test();
    if (batch != expected) {
        fprintf(stderr, "%s dynamic batch mismatch: batch=%d expected=%d\n", label, batch, expected);
        return false;
    }
    return true;
#else
    (void) label;
    (void) expected;
    return true;
#endif
}

static bool check_fairy2i_w2_dynamic_tiles_hit(const char * label, bool expected) {
#if defined(GGML_USE_FAIRY2I_CPU_LUT)
    const int hits = ggml_fairy2i_wide_linear_w2_dynamic_tiles_hits_for_test(false);
    if ((hits > 0) != expected) {
        fprintf(stderr, "%s dynamic hit mismatch: hits=%d expected=%d\n", label, hits, expected ? 1 : 0);
        return false;
    }
    return true;
#else
    (void) label;
    (void) expected;
    return true;
#endif
}

static bool check_fairy2i_w2_dynamic_tiles_last_batch(const char * label, int expected) {
#if defined(GGML_USE_FAIRY2I_CPU_LUT)
    const int batch = ggml_fairy2i_wide_linear_w2_dynamic_tiles_last_batch_for_test();
    if (batch != expected) {
        fprintf(stderr, "%s dynamic batch mismatch: batch=%d expected=%d\n", label, batch, expected);
        return false;
    }
    return true;
#else
    (void) label;
    (void) expected;
    return true;
#endif
}

static bool test_fairy2i_wide_linear_w1_lut_dynamic_tiles() {
#if defined(GGML_USE_FAIRY2I_CPU_LUT)
    const int     threads[]     = { 1, 2, 3, 4, 6, 8 };
    const int64_t Ms[]          = { 15, 16, 17, 31, 32, 33, 65, 127 };
    const int64_t fallback_Ns[] = { 2, 3 };
    const int64_t Ks[]          = { 256, 512 };
    const char *  batches[]     = { nullptr, "1", "2", "4" };

    int  dynamic_cases  = 0;
    int  fallback_cases = 0;
    bool ok             = true;

    {
        scoped_env_var        outer_dynamic_tiles("GGML_FAIRY2I_W1_DYNAMIC_TILES");
        const fairy2i_w2_case tc   = { 65, 1, 512, true };
        const fairy2i_w1_data data = make_fairy2i_w1_data(tc);
        std::vector<uint32_t> direct;
        std::vector<uint32_t> lut_static;
        std::vector<uint32_t> lut_default_off;
        std::vector<uint32_t> lut_default_batch;
        std::vector<uint32_t> lut_batch_one;
        std::vector<uint32_t> lut_batch_invalid;
        static const char *   invalid_batches[] = { "3", "2x" };
        static const int      n_threads         = 6;
        static const char *   label_default_off = "W1 LUT dynamic tiles default-off with polluted external environment";
        static const char *   label_default     = "W1 LUT dynamic tiles default batch";
        static const char *   label_invalid     = "W1 LUT dynamic tiles invalid batch fallback";

        if (!run_fairy2i_w1_backend(direct, tc, data, "0", "lut16", false, false, n_threads)) {
            return false;
        }
        if (!run_fairy2i_w1_backend(lut_static, tc, data, "1", "lut16", true, false, n_threads, "0")) {
            return false;
        }

        outer_dynamic_tiles.set("1");
        ggml_fairy2i_wide_linear_w1_dynamic_tiles_hits_for_test(true);
        if (!run_fairy2i_w1_backend(lut_default_off, tc, data, "1", "lut16", true, false, n_threads)) {
            return false;
        }
        ok = check_fairy2i_w1_dynamic_tiles_hit(label_default_off, false) && ok;
        ok = compare_exact(label_default_off, lut_default_off, lut_static) && ok;

        ggml_fairy2i_wide_linear_w1_dynamic_tiles_hits_for_test(true);
        if (!run_fairy2i_w1_backend(lut_default_batch, tc, data, "1", "lut16", true, false, n_threads, "1")) {
            return false;
        }
        ok = check_fairy2i_w1_dynamic_tiles_hit(label_default, true) && ok;
        ok = check_fairy2i_w1_dynamic_tiles_last_batch(label_default, 2) && ok;
        ok = compare_exact(label_default, lut_default_batch, lut_static) && ok;
        ok = compare_packed_complex("W1 dynamic LUT semantic check vs direct", lut_default_batch, direct, 1e-2f) && ok;

        ggml_fairy2i_wide_linear_w1_dynamic_tiles_hits_for_test(true);
        if (!run_fairy2i_w1_backend(lut_batch_one, tc, data, "1", "lut16", true, false, n_threads, "1", "1")) {
            return false;
        }
        ok = check_fairy2i_w1_dynamic_tiles_hit("W1 LUT dynamic tiles batch=1 control", true) && ok;
        ok = check_fairy2i_w1_dynamic_tiles_last_batch("W1 LUT dynamic tiles batch=1 control", 1) && ok;

        for (const char * invalid_batch : invalid_batches) {
            ggml_fairy2i_wide_linear_w1_dynamic_tiles_hits_for_test(true);
            if (!run_fairy2i_w1_backend(lut_batch_invalid, tc, data, "1", "lut16", true, false, n_threads, "1",
                                        invalid_batch)) {
                return false;
            }
            ok = check_fairy2i_w1_dynamic_tiles_hit(label_invalid, true) && ok;
            ok = check_fairy2i_w1_dynamic_tiles_last_batch(label_invalid, 1) && ok;
            ok = compare_exact(label_invalid, lut_batch_invalid, lut_batch_one) && ok;
        }
    }

    for (int n_threads : threads) {
        for (int64_t M : Ms) {
            for (int64_t K : Ks) {
                for (bool with_bias : { false, true }) {
                    const fairy2i_w2_case tc   = { M, 1, K, with_bias };
                    const fairy2i_w1_data data = make_fairy2i_w1_data(tc);
                    char                  label[192];

                    std::vector<uint32_t> lut_static;
                    std::vector<uint32_t> lut_dynamic;
                    ggml_fairy2i_wide_linear_w1_dynamic_tiles_hits_for_test(true);
                    if (!run_fairy2i_w1_backend(lut_static, tc, data, "1", "lut16", true, false, n_threads, "0")) {
                        return false;
                    }
                    snprintf(label, sizeof(label), "W1 LUT dynamic tiles disabled M=%lld N=1 K=%lld bias=%d threads=%d",
                             (long long) M, (long long) K, (int) with_bias, n_threads);
                    ok = check_fairy2i_w1_dynamic_tiles_hit(label, false) && ok;

                    for (const char * batch : batches) {
                        ggml_fairy2i_wide_linear_w1_dynamic_tiles_hits_for_test(true);
                        if (!run_fairy2i_w1_backend(lut_dynamic, tc, data, "1", "lut16", true, false, n_threads, "1",
                                                    batch)) {
                            return false;
                        }
                        snprintf(label, sizeof(label),
                                 "W1 LUT dynamic tiles hit M=%lld N=1 K=%lld bias=%d threads=%d batch=%s",
                                 (long long) M, (long long) K, (int) with_bias, n_threads, batch ? batch : "default");
                        ok                       = check_fairy2i_w1_dynamic_tiles_hit(label, true) && ok;
                        const int expected_batch = !batch                           ? 2 :
                                                   batch && strcmp(batch, "2") == 0 ? 2 :
                                                   batch && strcmp(batch, "4") == 0 ? 4 :
                                                                                      1;
                        ok = check_fairy2i_w1_dynamic_tiles_last_batch(label, expected_batch) && ok;

                        snprintf(label, sizeof(label),
                                 "W1 LUT dynamic tiles M=%lld N=1 K=%lld bias=%d threads=%d batch=%s", (long long) M,
                                 (long long) K, (int) with_bias, n_threads, batch ? batch : "default");
                        ok = compare_exact(label, lut_dynamic, lut_static) && ok;
                        ++dynamic_cases;
                    }

                    for (int64_t N : fallback_Ns) {
                        const fairy2i_w2_case fallback_tc   = { M, N, K, with_bias };
                        const fairy2i_w1_data fallback_data = make_fairy2i_w1_data(fallback_tc);

                        std::vector<uint32_t> fallback_static;
                        std::vector<uint32_t> fallback_gate_on;
                        ggml_fairy2i_wide_linear_w1_dynamic_tiles_hits_for_test(true);
                        if (!run_fairy2i_w1_backend(fallback_static, fallback_tc, fallback_data, "1", "lut16", true,
                                                    false, n_threads, "0")) {
                            return false;
                        }
                        snprintf(label, sizeof(label),
                                 "W1 LUT dynamic tiles fallback disabled M=%lld N=%lld K=%lld bias=%d threads=%d",
                                 (long long) M, (long long) N, (long long) K, (int) with_bias, n_threads);
                        ok = check_fairy2i_w1_dynamic_tiles_hit(label, false) && ok;

                        ggml_fairy2i_wide_linear_w1_dynamic_tiles_hits_for_test(true);
                        if (!run_fairy2i_w1_backend(fallback_gate_on, fallback_tc, fallback_data, "1", "lut16", true,
                                                    false, n_threads, "1", "4")) {
                            return false;
                        }
                        snprintf(label, sizeof(label),
                                 "W1 LUT dynamic tiles fallback hit M=%lld N=%lld K=%lld bias=%d threads=%d batch=4",
                                 (long long) M, (long long) N, (long long) K, (int) with_bias, n_threads);
                        ok = check_fairy2i_w1_dynamic_tiles_hit(label, false) && ok;

                        snprintf(label, sizeof(label),
                                 "W1 LUT dynamic tiles fallback/static M=%lld N=%lld K=%lld bias=%d threads=%d batch=4",
                                 (long long) M, (long long) N, (long long) K, (int) with_bias, n_threads);
                        ok = compare_exact(label, fallback_gate_on, fallback_static) && ok;
                        ++fallback_cases;
                    }
                }
            }
        }
    }

    printf("  Fairy2i W1 LUT dynamic tiles: %d N=1 cases, %d N>1 fallback cases - %s\n", dynamic_cases, fallback_cases,
           ok ? "PASS" : "FAIL");
    return ok;
#else
    printf("  Fairy2i W1 LUT dynamic tiles skipped: build lacks GGML_USE_FAIRY2I_CPU_LUT\n");
    return true;
#endif
}

static bool test_fairy2i_wide_linear_w2_lut_dynamic_tiles() {
#if defined(GGML_USE_FAIRY2I_CPU_LUT)
    const int     threads[]     = { 1, 2, 3, 4, 6, 8, 10, 12 };
    const int64_t Ms[]          = { 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255 };
    const int64_t fallback_Ns[] = { 2, 3 };
    const int64_t Ks[]          = { 256, 512 };
    const char *  batches[]     = { nullptr, "1", "2", "4" };

    int  dynamic_cases  = 0;
    int  fallback_cases = 0;
    bool ok             = true;

    {
        scoped_env_var        outer_dynamic_tiles("GGML_FAIRY2I_W2_DYNAMIC_TILES");
        const fairy2i_w2_case tc   = { 65, 1, 512, true };
        const fairy2i_w2_data data = make_fairy2i_w2_data(tc);
        std::vector<uint32_t> lut_static;
        std::vector<uint32_t> lut_default_on;
        std::vector<uint32_t> lut_batch_one;
        std::vector<uint32_t> lut_batch_invalid;
        static const char *   invalid_batches[] = { "3", "2x" };
        static const int      n_threads         = 6;
        static const char * label_default = "W2 LUT dynamic tiles helper default-on with polluted external environment";
        static const char * label_invalid = "W2 LUT dynamic tiles invalid batch fallback";

        if (!run_fairy2i_w2_backend(lut_static, tc, data, "1", "lut16", true, false, n_threads, "0")) {
            return false;
        }

        outer_dynamic_tiles.set("0");
        ggml_fairy2i_wide_linear_w2_dynamic_tiles_hits_for_test(true);
        if (!run_fairy2i_w2_backend(lut_default_on, tc, data, "1", "lut16", true, false, n_threads)) {
            return false;
        }
        ok = check_fairy2i_w2_dynamic_tiles_hit(label_default, true) && ok;
        ok = check_fairy2i_w2_dynamic_tiles_last_batch(label_default, 2) && ok;
        ok = compare_exact(label_default, lut_default_on, lut_static) && ok;

        ggml_fairy2i_wide_linear_w2_dynamic_tiles_hits_for_test(true);
        if (!run_fairy2i_w2_backend(lut_batch_one, tc, data, "1", "lut16", true, false, n_threads, "1", "1")) {
            return false;
        }
        ok = check_fairy2i_w2_dynamic_tiles_hit("W2 LUT dynamic tiles batch=1 control", true) && ok;
        ok = check_fairy2i_w2_dynamic_tiles_last_batch("W2 LUT dynamic tiles batch=1 control", 1) && ok;

        for (const char * invalid_batch : invalid_batches) {
            ggml_fairy2i_wide_linear_w2_dynamic_tiles_hits_for_test(true);
            if (!run_fairy2i_w2_backend(lut_batch_invalid, tc, data, "1", "lut16", true, false, n_threads, "1",
                                        invalid_batch)) {
                return false;
            }
            ok = check_fairy2i_w2_dynamic_tiles_hit(label_invalid, true) && ok;
            ok = check_fairy2i_w2_dynamic_tiles_last_batch(label_invalid, 1) && ok;
            ok = compare_exact(label_invalid, lut_batch_invalid, lut_batch_one) && ok;
        }
    }

    for (int n_threads : threads) {
        for (int64_t M : Ms) {
            for (int64_t K : Ks) {
                for (bool with_bias : { false, true }) {
                    const fairy2i_w2_case tc   = { M, 1, K, with_bias };
                    const fairy2i_w2_data data = make_fairy2i_w2_data(tc);
                    char                  label[192];

                    std::vector<uint32_t> lut_static;
                    std::vector<uint32_t> lut_dynamic;
                    ggml_fairy2i_wide_linear_w2_dynamic_tiles_hits_for_test(true);
                    if (!run_fairy2i_w2_backend(lut_static, tc, data, "1", "lut16", true, false, n_threads, "0")) {
                        return false;
                    }
                    snprintf(label, sizeof(label), "W2 LUT dynamic tiles disabled M=%lld N=1 K=%lld bias=%d threads=%d",
                             (long long) M, (long long) K, (int) with_bias, n_threads);
                    ok = check_fairy2i_w2_dynamic_tiles_hit(label, false) && ok;

                    for (const char * batch : batches) {
                        ggml_fairy2i_wide_linear_w2_dynamic_tiles_hits_for_test(true);
                        if (!run_fairy2i_w2_backend(lut_dynamic, tc, data, "1", "lut16", true, false, n_threads, "1",
                                                    batch)) {
                            return false;
                        }
                        snprintf(label, sizeof(label),
                                 "W2 LUT dynamic tiles hit M=%lld N=1 K=%lld bias=%d threads=%d batch=%s",
                                 (long long) M, (long long) K, (int) with_bias, n_threads, batch ? batch : "default");
                        ok                       = check_fairy2i_w2_dynamic_tiles_hit(label, true) && ok;
                        const int expected_batch = !batch                           ? 2 :
                                                   batch && strcmp(batch, "2") == 0 ? 2 :
                                                   batch && strcmp(batch, "4") == 0 ? 4 :
                                                                                      1;
                        ok = check_fairy2i_w2_dynamic_tiles_last_batch(label, expected_batch) && ok;

                        snprintf(label, sizeof(label),
                                 "W2 LUT dynamic tiles M=%lld N=1 K=%lld bias=%d threads=%d batch=%s", (long long) M,
                                 (long long) K, (int) with_bias, n_threads, batch ? batch : "default");
                        ok = compare_exact(label, lut_dynamic, lut_static) && ok;
                        ++dynamic_cases;
                    }

                    for (int64_t N : fallback_Ns) {
                        const fairy2i_w2_case fallback_tc   = { M, N, K, with_bias };
                        const fairy2i_w2_data fallback_data = make_fairy2i_w2_data(fallback_tc);

                        std::vector<uint32_t> fallback_static;
                        std::vector<uint32_t> fallback_gate_on;
                        ggml_fairy2i_wide_linear_w2_dynamic_tiles_hits_for_test(true);
                        if (!run_fairy2i_w2_backend(fallback_static, fallback_tc, fallback_data, "1", "lut16", true,
                                                    false, n_threads, "0")) {
                            return false;
                        }
                        snprintf(label, sizeof(label),
                                 "W2 LUT dynamic tiles fallback disabled M=%lld N=%lld K=%lld bias=%d threads=%d",
                                 (long long) M, (long long) N, (long long) K, (int) with_bias, n_threads);
                        ok = check_fairy2i_w2_dynamic_tiles_hit(label, false) && ok;

                        ggml_fairy2i_wide_linear_w2_dynamic_tiles_hits_for_test(true);
                        if (!run_fairy2i_w2_backend(fallback_gate_on, fallback_tc, fallback_data, "1", "lut16", true,
                                                    false, n_threads, "1", "4")) {
                            return false;
                        }
                        snprintf(label, sizeof(label),
                                 "W2 LUT dynamic tiles fallback hit M=%lld N=%lld K=%lld bias=%d threads=%d batch=4",
                                 (long long) M, (long long) N, (long long) K, (int) with_bias, n_threads);
                        ok = check_fairy2i_w2_dynamic_tiles_hit(label, false) && ok;

                        snprintf(label, sizeof(label),
                                 "W2 LUT dynamic tiles fallback/static M=%lld N=%lld K=%lld bias=%d threads=%d batch=4",
                                 (long long) M, (long long) N, (long long) K, (int) with_bias, n_threads);
                        ok = compare_exact(label, fallback_gate_on, fallback_static) && ok;
                        ++fallback_cases;
                    }
                }
            }
        }
    }

    printf("  Fairy2i W2 LUT dynamic tiles: %d N=1 cases, %d N>1 fallback cases - %s\n", dynamic_cases, fallback_cases,
           ok ? "PASS" : "FAIL");
    return ok;
#else
    printf("  Fairy2i W2 LUT dynamic tiles skipped: build lacks GGML_USE_FAIRY2I_CPU_LUT\n");
    return true;
#endif
}

static bool compare_exact(const char *                  label,
                          const std::vector<uint32_t> & actual,
                          const std::vector<uint32_t> & expected) {
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

static bool test_fairy2i_bundle_generic_mul_mat_rejected() {
    printf("\n=== Fairy2i Bundle generic MUL_MAT rejection ===\n");

    struct ggml_init_params params = {
        /*.mem_size   =*/256 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "failed to initialize context for Bundle generic MUL_MAT support tests\n");
        return false;
    }

    ggml_tensor * codes = ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_BUNDLE_CODES, 16, 64);
    ggml_tensor * x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 1);
    ggml_tensor * mm    = ggml_mul_mat(ctx, codes, x);

    ggml_tensor * expert_codes = ggml_new_tensor_3d(ctx, GGML_TYPE_FAIRY2I_BUNDLE_CODES, 16, 64, 2);
    ggml_tensor * expert_x     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 1, 1);
    ggml_tensor * expert_ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
    ggml_tensor * mm_id        = ggml_mul_mat_id(ctx, expert_codes, expert_x, expert_ids);

    ggml_tensor * bundle_codes  = ggml_new_tensor_4d(ctx, GGML_TYPE_FAIRY2I_BUNDLE_CODES, 16, 4, 64, 2);
    ggml_tensor * bundle_scales = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 2, 4, 2);
    ggml_tensor * bundle_x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 1);
    ggml_tensor * bundle_w2 =
        ggml_fairy2i_wide_linear_w2_bundle(ctx, bundle_x, bundle_codes, bundle_scales, nullptr, 64, 128);

    bool               ok      = true;
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        fprintf(stderr, "CPU backend device not found for Bundle generic MUL_MAT support tests\n");
        ok = false;
    } else if (ggml_backend_dev_supports_op(cpu_dev, mm) || ggml_backend_dev_supports_op(cpu_dev, mm_id)) {
        fprintf(stderr, "CPU backend must reject Bundle codes in generic MUL_MAT and MUL_MAT_ID\n");
        ok = false;
    }

#if defined(GGML_USE_FAIRY2I_CPU_LUT)
    if (cpu_dev) {
        scoped_env_var lut_enabled("GGML_FAIRY2I_LUT");
        scoped_env_var lut_impl("GGML_FAIRY2I_LUT_IMPL");
        lut_enabled.set("0");
        if (ggml_backend_dev_supports_op(cpu_dev, bundle_w2)) {
            fprintf(stderr, "CPU backend must reject Bundle W2 when GGML_FAIRY2I_LUT=0\n");
            ok = false;
        }
        lut_enabled.set("1");
        lut_impl.set("lut16");
        if (!ggml_backend_dev_supports_op(cpu_dev, bundle_w2)) {
            fprintf(stderr, "CPU backend must support Bundle W2 with the LUT16 path enabled\n");
            ok = false;
        }
        lut_impl.set("lut_c");
        if (ggml_backend_dev_supports_op(cpu_dev, bundle_w2)) {
            fprintf(stderr, "CPU backend must reject Bundle W2 with GGML_FAIRY2I_LUT_IMPL=lut_c\n");
            ok = false;
        }
    }
#else
    if (cpu_dev && ggml_backend_dev_supports_op(cpu_dev, bundle_w2)) {
        fprintf(stderr, "CPU backend without LUT16 must reject the dedicated Bundle W2 op\n");
        ok = false;
    }
#endif

    ggml_backend_dev_t metal_dev = find_metal_test_device();
    if (metal_dev) {
        if (ggml_backend_dev_supports_op(metal_dev, mm) || ggml_backend_dev_supports_op(metal_dev, mm_id)) {
            fprintf(stderr, "Metal backend must reject Bundle codes in generic MUL_MAT and MUL_MAT_ID\n");
            ok = false;
        }
        if (!ggml_backend_dev_supports_op(metal_dev, bundle_w2)) {
            fprintf(stderr, "Metal backend must continue to support the dedicated Bundle W2 op\n");
            ok = false;
        }
    } else {
        printf("Metal backend not found; skipping Metal capability assertions.\n");
    }

    ggml_free(ctx);
    printf("  Bundle generic matmul rejection: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool run_fairy2i_metal_backend(std::vector<uint32_t> &                      out,
                                      ggml_backend_dev_t                           dev,
                                      const fairy2i_w2_case &                      tc,
                                      const std::vector<float> &                   input,
                                      const std::vector<block_fairy2i_tile64_v2> & u_s0_data,
                                      const std::vector<block_fairy2i_tile64_v2> * u_s1_data,
                                      const std::vector<block_fairy2i_tile64_v2> & w_s0_data,
                                      const std::vector<block_fairy2i_tile64_v2> * w_s1_data,
                                      const std::vector<float> &                   bias_data,
                                      const fairy2i_bundle_data *                  bundle_data = nullptr) {
    const bool is_bundle = bundle_data != nullptr;
    const bool is_w1     = is_bundle ? bundle_data->branches == 2 : u_s1_data == nullptr && w_s1_data == nullptr;
    if (!is_w1 && (!u_s1_data || !w_s1_data)) {
        fprintf(stderr, "invalid Fairy2i Metal test weight set\n");
        return false;
    }
    if (is_bundle && bundle_data->branches != 2 && bundle_data->branches != 4) {
        fprintf(stderr, "invalid Fairy2i Metal bundle branch count\n");
        return false;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        fprintf(stderr, "failed to initialize Metal backend\n");
        return false;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/4 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        fprintf(stderr, "failed to initialize ggml context for Fairy2i Metal\n");
        return false;
    }

    const int64_t physical_tiles = (tc.M / 64) * (tc.K / 64);
    ggml_tensor * x              = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.K, tc.N);
    ggml_tensor * codes = is_bundle ? ggml_new_tensor_4d(ctx, GGML_TYPE_FAIRY2I_BUNDLE_CODES, 16, bundle_data->branches,
                                                         64, physical_tiles) :
                                      nullptr;
    ggml_tensor * scales =
        is_bundle ? ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 2, bundle_data->branches, physical_tiles) : nullptr;
    ggml_tensor * u_s0 = is_bundle ? nullptr : ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_TILE64_V2, tc.K, tc.M);
    ggml_tensor * u_s1 =
        is_bundle || is_w1 ? nullptr : ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_TILE64_V2, tc.K, tc.M);
    ggml_tensor * w_s0 = is_bundle ? nullptr : ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_TILE64_V2, tc.K, tc.M);
    ggml_tensor * w_s1 =
        is_bundle || is_w1 ? nullptr : ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_TILE64_V2, tc.K, tc.M);
    ggml_tensor * bias = tc.bias ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * tc.M) : nullptr;
    ggml_tensor * y    = is_bundle ?
                             (is_w1 ? ggml_fairy2i_wide_linear_w1_bundle(ctx, x, codes, scales, bias, tc.M, tc.K) :
                                      ggml_fairy2i_wide_linear_w2_bundle(ctx, x, codes, scales, bias, tc.M, tc.K)) :
                             ggml_fairy2i_wide_linear_w2(ctx, x, u_s0, u_s1, w_s0, w_s1, bias);
    ggml_set_name(y, is_bundle ? (is_w1 ? "fairy2i_metal_bundle_w1" : "fairy2i_metal_bundle_w2") :
                     is_w1     ? "fairy2i_metal_wide_linear_w1" :
                                 "fairy2i_metal_wide_linear_w2");

    if (!ggml_backend_supports_op(backend, y)) {
        fprintf(stderr, "Metal does not support Fairy2i W%d M=%lld N=%lld K=%lld\n", is_w1 ? 1 : 2, (long long) tc.M,
                (long long) tc.N, (long long) tc.K);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "failed to allocate Metal backend buffer for Fairy2i W%d\n", is_w1 ? 1 : 2);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    if (is_bundle) {
        ggml_backend_tensor_set(codes, bundle_data->codes.data(), 0, bundle_data->codes.size());
        ggml_backend_tensor_set(scales, bundle_data->scales.data(), 0,
                                bundle_data->scales.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(u_s0, u_s0_data.data(), 0, u_s0_data.size() * sizeof(block_fairy2i_tile64_v2));
        ggml_backend_tensor_set(w_s0, w_s0_data.data(), 0, w_s0_data.size() * sizeof(block_fairy2i_tile64_v2));
    }
    if (!is_bundle && !is_w1) {
        ggml_backend_tensor_set(u_s1, u_s1_data->data(), 0, u_s1_data->size() * sizeof(block_fairy2i_tile64_v2));
        ggml_backend_tensor_set(w_s1, w_s1_data->data(), 0, w_s1_data->size() * sizeof(block_fairy2i_tile64_v2));
    }
    if (bias) {
        ggml_backend_tensor_set(bias, bias_data.data(), 0, bias_data.size() * sizeof(float));
    }

    const bool        direct_bundle = !is_bundle || (codes->extra == nullptr && scales->extra == nullptr);
    const ggml_status status        = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Fairy2i Metal W%d graph compute failed: %s\n", is_w1 ? 1 : 2, ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_synchronize(backend);
    if (!direct_bundle || (is_bundle && (codes->extra != nullptr || scales->extra != nullptr))) {
        fprintf(stderr, "Fairy2i Metal bundle unexpectedly allocated a repacked weight copy\n");
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

static bool test_fairy2i_metal_wide_linear() {
    printf("\n=== Fairy2i Metal W1/W2 tests ===\n");

    ggml_backend_dev_t dev = find_metal_test_device();
    if (!dev) {
        printf("Metal backend not found; skipping Fairy2i Metal tests.\n");
        return true;
    }

    printf("Metal device: %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));

    const std::vector<fairy2i_w2_case> w2_cases = {
        { 7,  3, 128, true },
        { 17, 1, 256, true },
    };
    for (const fairy2i_w2_case & tc : w2_cases) {
        fairy2i_w2_data data = make_fairy2i_w2_data(tc);
        share_fairy2i_w64_scales(data, tc);
        const std::vector<uint32_t> ref = fairy2i_w2_scalar_reference(tc, data);

        std::vector<uint32_t> metal;
        if (!run_fairy2i_metal_backend(metal, dev, tc, data.x, data.u_s0, &data.u_s1, data.w_s0, &data.w_s1,
                                       data.bias) ||
            !compare_packed_complex("Fairy2i Metal W2", metal, ref, 1e-2f)) {
            return false;
        }
    }

    const std::vector<fairy2i_w2_case> w1_cases = {
        { 9,  3, 128, true  },
        { 16, 1, 256, false },
    };
    for (const fairy2i_w2_case & tc : w1_cases) {
        fairy2i_w1_data data = make_fairy2i_w1_data(tc);
        share_fairy2i_w64_scales(data, tc);
        const std::vector<uint32_t> ref = fairy2i_w1_scalar_reference(tc, data);

        std::vector<uint32_t> metal;
        if (!run_fairy2i_metal_backend(metal, dev, tc, data.x, data.u_s0, nullptr, data.w_s0, nullptr, data.bias) ||
            !compare_packed_complex("Fairy2i Metal W1", metal, ref, 1e-2f)) {
            return false;
        }
    }

    const std::vector<fairy2i_w2_case> bundle_w1_cases = {
        { 128, 1,  256, false },
        { 64,  1,  128, true  },
        { 64,  16, 128, true  },
        { 64,  17, 128, true  },
    };
    for (const fairy2i_w2_case & tc : bundle_w1_cases) {
        fairy2i_w1_data data = make_fairy2i_w1_data(tc);
        make_fairy2i_weights_bundle_compatible(data.u_s0, tc.M, tc.K);
        make_fairy2i_weights_bundle_compatible(data.w_s0, tc.M, tc.K);
        const fairy2i_bundle_data   bundle = pack_fairy2i_bundle({ &data.u_s0, &data.w_s0 }, tc.M, tc.K);
        const std::vector<uint32_t> ref    = fairy2i_w1_scalar_reference(tc, data);

        std::vector<uint32_t> metal;
        if (!run_fairy2i_metal_backend(metal, dev, tc, data.x, data.u_s0, nullptr, data.w_s0, nullptr, data.bias,
                                       &bundle) ||
            !compare_packed_complex("Fairy2i Metal bundle W1", metal, ref, 1e-2f)) {
            return false;
        }
    }

    const std::vector<fairy2i_w2_case> bundle_w2_cases = {
        { 128, 1,  256, false },
        { 64,  1,  128, true  },
        { 64,  16, 128, true  },
        { 64,  17, 128, true  },
    };
    for (const fairy2i_w2_case & tc : bundle_w2_cases) {
        fairy2i_w2_data data = make_fairy2i_w2_data(tc);
        make_fairy2i_weights_bundle_compatible(data.u_s0, tc.M, tc.K);
        make_fairy2i_weights_bundle_compatible(data.u_s1, tc.M, tc.K);
        make_fairy2i_weights_bundle_compatible(data.w_s0, tc.M, tc.K);
        make_fairy2i_weights_bundle_compatible(data.w_s1, tc.M, tc.K);
        const fairy2i_bundle_data bundle =
            pack_fairy2i_bundle({ &data.u_s0, &data.u_s1, &data.w_s0, &data.w_s1 }, tc.M, tc.K);
        const std::vector<uint32_t> ref = fairy2i_w2_scalar_reference(tc, data);

        std::vector<uint32_t> metal;
        if (!run_fairy2i_metal_backend(metal, dev, tc, data.x, data.u_s0, &data.u_s1, data.w_s0, &data.w_s1, data.bias,
                                       &bundle) ||
            !compare_packed_complex("Fairy2i Metal bundle W2", metal, ref, 1e-2f)) {
            return false;
        }
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
                    std::vector<uint32_t> lut_default;
                    if (!run_fairy2i_w2_backend(direct, tc, data, "0", "lut16", false, false)) {
                        return false;
                    }
                    if (compare_scalar_default &&
                        !run_fairy2i_w2_backend(direct_scalar, tc, data, "0", "lut16", false, true)) {
                        return false;
                    }
#if defined(GGML_USE_FAIRY2I_CPU_LUT)
                    if (!run_fairy2i_w2_backend(lut, tc, data, "1", "lut16", true, false)) {
                        return false;
                    }
                    if (!run_fairy2i_w2_backend(lut_default, tc, data, nullptr, nullptr, true, false)) {
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

                    snprintf(label, sizeof(label), "explicit LUT vs default LUT M=%lld N=%lld K=%lld bias=%d",
                             (long long) M, (long long) N, (long long) K, (int) with_bias);
                    ok = compare_exact(label, lut_default, lut) && ok;
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

static bool test_fairy2i_wide_linear_w1_variants() {
    const int64_t Ms[] = { 1, 7, 16, 17, 23, 32, 33 };
    const int64_t Ks[] = { 64, 128, 192, 256, 320, 1024 };
    const int64_t Ns[] = { 1, 2, 4, 8 };

    int        cases_run = 0;
    bool       ok        = true;
    const bool compare_scalar_default =
        ggml_cpu_has_avx2() != 0 || ggml_cpu_has_neon() != 0 || ggml_cpu_has_dotprod() != 0;

    for (int64_t M : Ms) {
        for (int64_t K : Ks) {
            for (int64_t N : Ns) {
                for (bool with_bias : { false, true }) {
                    const fairy2i_w2_case       tc   = { M, N, K, with_bias };
                    const fairy2i_w1_data       data = make_fairy2i_w1_data(tc);
                    const std::vector<uint32_t> ref  = fairy2i_w1_scalar_reference(tc, data);

                    std::vector<uint32_t> direct;
                    std::vector<uint32_t> direct_scalar;
                    std::vector<uint32_t> lut;
                    std::vector<uint32_t> lut_default;
                    if (!run_fairy2i_w1_backend(direct, tc, data, "0", "lut16", false, false)) {
                        return false;
                    }
                    if (compare_scalar_default &&
                        !run_fairy2i_w1_backend(direct_scalar, tc, data, "0", "lut16", false, true)) {
                        return false;
                    }
#if defined(GGML_USE_FAIRY2I_CPU_LUT)
                    if (!run_fairy2i_w1_backend(lut, tc, data, "1", "lut16", true, false)) {
                        return false;
                    }
                    if (!run_fairy2i_w1_backend(lut_default, tc, data, nullptr, nullptr, true, false)) {
                        return false;
                    }
#endif

                    char label[160];
                    snprintf(label, sizeof(label), "W1 reference vs direct M=%lld N=%lld K=%lld bias=%d", (long long) M,
                             (long long) N, (long long) K, (int) with_bias);
                    ok = compare_exact(label, direct, ref) && ok;

                    if (compare_scalar_default) {
                        snprintf(label, sizeof(label), "W1 forced scalar vs default M=%lld N=%lld K=%lld bias=%d",
                                 (long long) M, (long long) N, (long long) K, (int) with_bias);
                        ok = compare_exact(label, direct_scalar, direct) && ok;
                    }

#if defined(GGML_USE_FAIRY2I_CPU_LUT)
                    snprintf(label, sizeof(label), "W1 direct vs LUT M=%lld N=%lld K=%lld bias=%d", (long long) M,
                             (long long) N, (long long) K, (int) with_bias);
                    ok = compare_packed_complex(label, lut, direct, 1e-2f) && ok;

                    snprintf(label, sizeof(label), "W1 explicit LUT vs default LUT M=%lld N=%lld K=%lld bias=%d",
                             (long long) M, (long long) N, (long long) K, (int) with_bias);
                    ok = compare_exact(label, lut_default, lut) && ok;
#endif
                    ++cases_run;
                }
            }
        }
    }

    if (!compare_scalar_default) {
        printf("  Fairy2i W1 scalar/default fast-path compare skipped: CPU backend lacks AVX2/NEON/dotprod\n");
    }
#if !defined(GGML_USE_FAIRY2I_CPU_LUT)
    printf("  Fairy2i W1 LUT compare skipped: build lacks GGML_USE_FAIRY2I_CPU_LUT\n");
#endif
    printf("  Fairy2i W1 variant matrix: %d cases - %s\n", cases_run, ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_fairy2i_bundle_lut() {
#if defined(GGML_USE_FAIRY2I_CPU_LUT)
    bool          ok         = true;
    int           cases_run  = 0;
    const int64_t k_values[] = { 64, 128, 256, 1024 };
    const int     threads[]  = { 1, 2, 4, 8 };

    for (int i = 0; i < 4; ++i) {
        const fairy2i_w2_case tc = { 128, i == 1 ? 3 : 1, k_values[i], (i & 1) != 0 };

        fairy2i_w1_data w1_data = make_fairy2i_w1_data(tc);
        make_fairy2i_weights_bundle_compatible(w1_data.u_s0, tc.M, tc.K);
        make_fairy2i_weights_bundle_compatible(w1_data.w_s0, tc.M, tc.K);
        const fairy2i_bundle_data w1_bundle = pack_fairy2i_bundle({ &w1_data.u_s0, &w1_data.w_s0 }, tc.M, tc.K);
        std::vector<uint32_t>     old_w1;
        std::vector<uint32_t>     bundle_w1;
        if (!run_fairy2i_w1_backend(old_w1, tc, w1_data, "1", "lut16", true, false, threads[i], "0") ||
            !run_fairy2i_bundle_backend(bundle_w1, tc, w1_data.x, w1_data.bias, w1_bundle, true, "1", "lut16", false,
                                        threads[i], "0")) {
            return false;
        }
        char label[160];
        snprintf(label, sizeof(label), "W1 old/bundle LUT bit exact M=%lld N=%lld K=%lld bias=%d threads=%d",
                 (long long) tc.M, (long long) tc.N, (long long) tc.K, (int) tc.bias, threads[i]);
        ok = compare_exact(label, bundle_w1, old_w1) && ok;
        ++cases_run;

        fairy2i_w2_data w2_data = make_fairy2i_w2_data(tc);
        make_fairy2i_weights_bundle_compatible(w2_data.u_s0, tc.M, tc.K);
        make_fairy2i_weights_bundle_compatible(w2_data.u_s1, tc.M, tc.K);
        make_fairy2i_weights_bundle_compatible(w2_data.w_s0, tc.M, tc.K);
        make_fairy2i_weights_bundle_compatible(w2_data.w_s1, tc.M, tc.K);
        const fairy2i_bundle_data w2_bundle =
            pack_fairy2i_bundle({ &w2_data.u_s0, &w2_data.u_s1, &w2_data.w_s0, &w2_data.w_s1 }, tc.M, tc.K);
        std::vector<uint32_t> old_w2;
        std::vector<uint32_t> bundle_w2;
        if (!run_fairy2i_w2_backend(old_w2, tc, w2_data, "1", "lut16", true, false, threads[i], "0") ||
            !run_fairy2i_bundle_backend(bundle_w2, tc, w2_data.x, w2_data.bias, w2_bundle, false, "1", "lut16", false,
                                        threads[i], "0")) {
            return false;
        }
        snprintf(label, sizeof(label), "W2 old/bundle LUT bit exact M=%lld N=%lld K=%lld bias=%d threads=%d",
                 (long long) tc.M, (long long) tc.N, (long long) tc.K, (int) tc.bias, threads[i]);
        ok = compare_exact(label, bundle_w2, old_w2) && ok;
        ++cases_run;

        if (tc.K == 256) {
            std::vector<uint32_t> scalar_w1;
            std::vector<uint32_t> scalar_w2;
            if (!run_fairy2i_bundle_backend(scalar_w1, tc, w1_data.x, w1_data.bias, w1_bundle, true, "1", "lut16", true,
                                            threads[i], "0") ||
                !run_fairy2i_bundle_backend(scalar_w2, tc, w2_data.x, w2_data.bias, w2_bundle, false, "1", "lut16",
                                            true, threads[i], "0")) {
                return false;
            }
            ok = compare_packed_complex("W1 bundle scalar/ISA", scalar_w1, bundle_w1, 1e-2f) && ok;
            ok = compare_packed_complex("W2 bundle scalar/ISA", scalar_w2, bundle_w2, 1e-2f) && ok;
        }
    }

    const fairy2i_w2_case dynamic_tc = { 256, 1, 256, true };
    fairy2i_w1_data       dynamic_w1 = make_fairy2i_w1_data(dynamic_tc);
    make_fairy2i_weights_bundle_compatible(dynamic_w1.u_s0, dynamic_tc.M, dynamic_tc.K);
    make_fairy2i_weights_bundle_compatible(dynamic_w1.w_s0, dynamic_tc.M, dynamic_tc.K);
    const fairy2i_bundle_data dynamic_w1_bundle =
        pack_fairy2i_bundle({ &dynamic_w1.u_s0, &dynamic_w1.w_s0 }, dynamic_tc.M, dynamic_tc.K);
    std::vector<uint32_t> dynamic_w1_static;
    if (!run_fairy2i_bundle_backend(dynamic_w1_static, dynamic_tc, dynamic_w1.x, dynamic_w1.bias, dynamic_w1_bundle,
                                    true, "1", "lut16", false, 8, "0")) {
        return false;
    }
    std::vector<uint32_t> dynamic_w1_default;
    ggml_fairy2i_wide_linear_w1_dynamic_tiles_hits_for_test(true);
    if (!run_fairy2i_bundle_backend(dynamic_w1_default, dynamic_tc, dynamic_w1.x, dynamic_w1.bias, dynamic_w1_bundle,
                                    true, "1", "lut16", false, 8)) {
        return false;
    }
    ok = check_fairy2i_w1_dynamic_tiles_hit("W1 bundle dynamic default", true) && ok;
    ok = check_fairy2i_w1_dynamic_tiles_last_batch("W1 bundle dynamic default", 4) && ok;
    ok = compare_exact("W1 bundle dynamic default/static", dynamic_w1_default, dynamic_w1_static) && ok;
    ++cases_run;
    for (const char * batch : { "1", "2", "4" }) {
        std::vector<uint32_t> output;
        if (!run_fairy2i_bundle_backend(output, dynamic_tc, dynamic_w1.x, dynamic_w1.bias, dynamic_w1_bundle, true, "1",
                                        "lut16", false, 8, "1", batch)) {
            return false;
        }
        ok = compare_exact("W1 bundle dynamic/static", output, dynamic_w1_static) && ok;
        ++cases_run;
    }

    fairy2i_w2_data dynamic_w2 = make_fairy2i_w2_data(dynamic_tc);
    make_fairy2i_weights_bundle_compatible(dynamic_w2.u_s0, dynamic_tc.M, dynamic_tc.K);
    make_fairy2i_weights_bundle_compatible(dynamic_w2.u_s1, dynamic_tc.M, dynamic_tc.K);
    make_fairy2i_weights_bundle_compatible(dynamic_w2.w_s0, dynamic_tc.M, dynamic_tc.K);
    make_fairy2i_weights_bundle_compatible(dynamic_w2.w_s1, dynamic_tc.M, dynamic_tc.K);
    const fairy2i_bundle_data dynamic_w2_bundle = pack_fairy2i_bundle(
        { &dynamic_w2.u_s0, &dynamic_w2.u_s1, &dynamic_w2.w_s0, &dynamic_w2.w_s1 }, dynamic_tc.M, dynamic_tc.K);
    std::vector<uint32_t> dynamic_w2_static;
    if (!run_fairy2i_bundle_backend(dynamic_w2_static, dynamic_tc, dynamic_w2.x, dynamic_w2.bias, dynamic_w2_bundle,
                                    false, "1", "lut16", false, 8, "0")) {
        return false;
    }
    for (const char * batch : { "1", "2", "4" }) {
        std::vector<uint32_t> output;
        if (!run_fairy2i_bundle_backend(output, dynamic_tc, dynamic_w2.x, dynamic_w2.bias, dynamic_w2_bundle, false,
                                        "1", "lut16", false, 8, "1", batch)) {
            return false;
        }
        ok = compare_exact("W2 bundle dynamic/static", output, dynamic_w2_static) && ok;
        ++cases_run;
    }

    printf("  Fairy2i bundle LUT16: %d execution cases - %s\n", cases_run, ok ? "PASS" : "FAIL");
    return ok;
#else
    printf("  Fairy2i bundle LUT16 skipped: build lacks GGML_USE_FAIRY2I_CPU_LUT\n");
    return true;
#endif
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
    if (!test_fairy2i_lut_qgemm_pair_extreme_same_lane()) {
        fprintf(stderr, "Fairy2i LUT qgemm pair same-lane extremes FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_lut_qgemm_pair_layout_guardrails()) {
        fprintf(stderr, "Fairy2i LUT qgemm pair layout guardrails FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_wide_linear_w1_variants()) {
        fprintf(stderr, "Fairy2i W1 variant matrix FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_wide_linear_w2_variants()) {
        fprintf(stderr, "Fairy2i W2 variant matrix FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_wide_linear_w1_lut_dynamic_tiles()) {
        fprintf(stderr, "Fairy2i W1 LUT dynamic tiles FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_wide_linear_w2_lut_dynamic_tiles()) {
        fprintf(stderr, "Fairy2i W2 LUT dynamic tiles FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_metal_wide_linear()) {
        fprintf(stderr, "Fairy2i Metal W1/W2 FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_bundle_generic_mul_mat_rejected()) {
        fprintf(stderr, "Fairy2i Bundle generic MUL_MAT rejection FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_bundle_lut()) {
        fprintf(stderr, "Fairy2i bundle LUT16 FAILED\n");
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
