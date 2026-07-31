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
    std::vector<ggml_bf16_t> scales_bf16;
    enum ggml_type           scale_type = GGML_TYPE_F16;
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

static uint16_t bf16_bits(float value) {
    return GGML_FP32_TO_BF16(value).bits;
}

static float bf16_round(float value) {
    return GGML_BF16_TO_FP32(GGML_FP32_TO_BF16(value));
}

static float bf16_from_bits(uint16_t bits) {
    const uint32_t value_bits = (uint32_t) bits << 16;
    float          value;
    memcpy(&value, &value_bits, sizeof(value));
    return value;
}

static bool test_fairy2i_bf16_rne_bits() {
    struct rne_case {
        const char * label;
        float        value;
        uint16_t     expected;
    };

    const rne_case cases[] = {
        { "+zero",               0.0f,                  0x0000 },
        { "-zero",               -0.0f,                 0x8000 },
        { "tie-even-lower",      1.0f + 0x1p-8f,        0x3f80 },
        { "tie-even-upper",      1.0f + 3.0f * 0x1p-8f, 0x3f82 },
        { "negative-tie-even",   -1.0f - 0x1p-8f,       0xbf80 },
        { "minimum-subnormal",   0x1p-133f,             0x0001 },
        { "subnormal-tie-zero",  0x1p-134f,             0x0000 },
        { "subnormal-tie-even",  3.0f * 0x1p-134f,      0x0002 },
        { "maximum-finite-bf16", 0x1.fep127f,           0x7f7f },
        { "negative-max-bf16",   -0x1.fep127f,          0xff7f },
    };

    bool ok = true;
    for (const rne_case & tc : cases) {
        const uint16_t actual = bf16_bits(tc.value);
        if (actual != tc.expected) {
            fprintf(stderr, "BF16 RNE %s mismatch: actual=0x%04x expected=0x%04x\n", tc.label, (unsigned) actual,
                    (unsigned) tc.expected);
            ok = false;
        }
    }

    printf("  Fairy2i BF16 RNE bit cases: %zu cases - %s\n", sizeof(cases) / sizeof(cases[0]), ok ? "PASS" : "FAIL");
    return ok;
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

static void set_fairy2i_bundle_code(fairy2i_bundle_data & bundle,
                                    int64_t               M,
                                    int64_t               K,
                                    int64_t               row,
                                    int64_t               col,
                                    int                   branch,
                                    uint8_t               code) {
    const int64_t k_blocks      = K / QK_FAIRY2I_TILE64;
    const int64_t physical_tile = (row / 64) * k_blocks + col / 64;
    const int64_t slot          = ((row % 64) / 16) * 16 + (col % 64) / 4;
    const int64_t lane          = row % 16;
    const int     q             = (int) (col % 4);
    const size_t  offset =
        (((size_t) physical_tile * 64u + (size_t) slot) * (size_t) bundle.branches + (size_t) branch) * 16u +
        (size_t) lane;

    GGML_ASSERT(row >= 0 && row < M && col >= 0 && col < K);
    GGML_ASSERT(branch >= 0 && branch < bundle.branches);
    bundle.codes[offset] =
        (uint8_t) ((bundle.codes[offset] & (uint8_t) ~(0x3u << (2 * q))) | ((code & 0x3u) << (2 * q)));
}

static fairy2i_bundle_data make_fairy2i_exact_bundle(int64_t M, int64_t K) {
    fairy2i_bundle_data bundle;
    bundle.branches   = 4;
    bundle.scale_type = GGML_TYPE_BF16;

    const int64_t physical_tiles = (M / 64) * (K / 64);
    bundle.codes.assign((size_t) physical_tiles * 64u * 4u * 16u, 0);
    bundle.scales_bf16.resize((size_t) physical_tiles * 4u * 2u);

    for (int64_t tile = 0; tile < physical_tiles; ++tile) {
        ggml_bf16_t * scales = bundle.scales_bf16.data() + (size_t) tile * 8u;
        scales[0]            = GGML_FP32_TO_BF16(1.0f + 0x1p-8f + 0x1p-20f);
        scales[1]            = GGML_FP32_TO_BF16(0.5f);
        scales[2]            = GGML_FP32_TO_BF16(0x1p-8f);
        scales[3]            = GGML_FP32_TO_BF16(0x1p-9f);
        scales[4]            = GGML_FP32_TO_BF16(0x1p-7f);
        scales[5]            = GGML_FP32_TO_BF16(0x1p-8f);
        scales[6]            = GGML_FP32_TO_BF16(0.0f);
        scales[7]            = GGML_FP32_TO_BF16(0.0f);
    }

    for (int64_t row = 0; row < M; ++row) {
        for (int branch = 0; branch < 4; ++branch) {
            set_fairy2i_bundle_code(bundle, M, K, row, 0, branch, 1);
            set_fairy2i_bundle_code(bundle, M, K, row, 1, branch, 3);
        }
    }

    return bundle;
}

static fairy2i_bundle_data make_fairy2i_exact_empty_bundle(int64_t M, int64_t K) {
    fairy2i_bundle_data bundle;
    bundle.branches   = 4;
    bundle.scale_type = GGML_TYPE_BF16;

    const int64_t physical_tiles = (M / 64) * (K / 64);
    bundle.codes.assign((size_t) physical_tiles * 64u * 4u * 16u, 0);
    bundle.scales_bf16.assign((size_t) physical_tiles * 8u, GGML_FP32_TO_BF16(0.0f));
    return bundle;
}

static void set_fairy2i_exact_scale(fairy2i_bundle_data & bundle, size_t index, float value) {
    GGML_ASSERT(bundle.scale_type == GGML_TYPE_BF16);
    GGML_ASSERT(index < bundle.scales_bf16.size());
    bundle.scales_bf16[index] = GGML_FP32_TO_BF16(value);
}

static void set_fairy2i_exact_tile_scales(fairy2i_bundle_data & bundle, size_t tile, const float scales[8]) {
    for (size_t i = 0; i < 8; ++i) {
        set_fairy2i_exact_scale(bundle, tile * 8u + i, scales[i]);
    }
}

static fairy2i_bundle_data make_fairy2i_exact_mixed_bundle(int64_t M, int64_t K) {
    fairy2i_bundle_data bundle;
    bundle.branches   = 4;
    bundle.scale_type = GGML_TYPE_BF16;

    const int64_t physical_tiles = (M / 64) * (K / 64);
    bundle.codes.assign((size_t) physical_tiles * 64u * 4u * 16u, 0);
    bundle.scales_bf16.resize((size_t) physical_tiles * 8u);

    static const float base_scales[8] = {
        0x1.02p-5f, 0x1.82p-6f, 0x1.04p-6f, 0x1.84p-7f, 0x1.06p-7f, 0x1.86p-8f, 0x1.08p-8f, 0x1.88p-9f,
    };
    for (int64_t tile = 0; tile < physical_tiles; ++tile) {
        const float tile_scale = 1.0f + 0.25f * (float) tile;
        for (int i = 0; i < 8; ++i) {
            bundle.scales_bf16[(size_t) tile * 8u + (size_t) i] = GGML_FP32_TO_BF16(base_scales[i] * tile_scale);
        }
    }

    for (int64_t row = 0; row < M; ++row) {
        for (int64_t col = 0; col < K; ++col) {
            const uint8_t pattern = (uint8_t) (((row & 1) << 7) | (col & 127));
            for (int branch = 0; branch < 4; ++branch) {
                set_fairy2i_bundle_code(bundle, M, K, row, col, branch, (uint8_t) ((pattern >> (2 * branch)) & 3u));
            }
        }
    }

    return bundle;
}

static fairy2i_bundle_data make_fairy2i_exact_subnormal_bundle(int64_t M, int64_t K) {
    fairy2i_bundle_data bundle;
    bundle.branches   = 4;
    bundle.scale_type = GGML_TYPE_BF16;

    const int64_t physical_tiles = (M / 64) * (K / 64);
    bundle.codes.assign((size_t) physical_tiles * 64u * 4u * 16u, 0);
    bundle.scales_bf16.assign((size_t) physical_tiles * 4u * 2u, GGML_FP32_TO_BF16(0.0f));

    for (int64_t tile = 0; tile < physical_tiles; ++tile) {
        ggml_bf16_t * scales = bundle.scales_bf16.data() + (size_t) tile * 8u;
        scales[0]            = GGML_FP32_TO_BF16(0x1p-126f);  // U0 real = BF16 0x0080.
        scales[2]            = GGML_FP32_TO_BF16(0x1p-133f);  // U1 real = BF16 0x0001.
    }

    for (int64_t row = 0; row < M; ++row) {
        for (int branch = 0; branch < 4; ++branch) {
            set_fairy2i_bundle_code(bundle, M, K, row, 0, branch, 1);
        }
    }

    return bundle;
}

static fairy2i_bundle_data make_fairy2i_exact_pure_subnormal_bundle(int64_t M, int64_t K) {
    fairy2i_bundle_data bundle;
    bundle.branches   = 4;
    bundle.scale_type = GGML_TYPE_BF16;

    const int64_t physical_tiles = (M / 64) * (K / 64);
    bundle.codes.assign((size_t) physical_tiles * 64u * 4u * 16u, 0);
    bundle.scales_bf16.assign((size_t) physical_tiles * 4u * 2u, GGML_FP32_TO_BF16(0.0f));

    for (int64_t tile = 0; tile < physical_tiles; ++tile) {
        bundle.scales_bf16[(size_t) tile * 8u] = GGML_FP32_TO_BF16(0x1p-133f);
    }
    for (int64_t row = 0; row < M; ++row) {
        set_fairy2i_bundle_code(bundle, M, K, row, 0, 0, 1);
    }
    return bundle;
}

struct fairy2i_exact_coeff_bits {
    uint16_t rr;
    uint16_t ri;
    uint16_t ir;
    uint16_t ii;
};

struct fairy2i_exact_stage_bits {
    uint16_t real;
    uint16_t imag;
};

static uint16_t add_fairy2i_exact_bf16_bits(uint16_t a, uint16_t b) {
    return bf16_bits(bf16_round(bf16_from_bits(a) + bf16_from_bits(b)));
}

static void build_fairy2i_exact_qstage_bits(const float scales[8], fairy2i_exact_stage_bits qstage[4][4]) {
    for (int branch = 0; branch < 4; ++branch) {
        const float real  = scales[2 * branch + 0];
        const float imag  = scales[2 * branch + 1];
        qstage[branch][0] = { bf16_bits(-real), 0 };
        qstage[branch][1] = { bf16_bits(real), 0 };
        qstage[branch][2] = { 0, bf16_bits(-imag) };
        qstage[branch][3] = { 0, bf16_bits(imag) };
    }
}

static fairy2i_exact_coeff_bits reconstruct_fairy2i_exact_coeff_from_qstage_bits(
    const uint8_t                  code[4],
    const fairy2i_exact_stage_bits qstage[4][4]) {
    const fairy2i_exact_stage_bits u = {
        add_fairy2i_exact_bf16_bits(qstage[0][code[0]].real, qstage[1][code[1]].real),
        add_fairy2i_exact_bf16_bits(qstage[0][code[0]].imag, qstage[1][code[1]].imag),
    };
    const fairy2i_exact_stage_bits w = {
        add_fairy2i_exact_bf16_bits(qstage[2][code[2]].real, qstage[3][code[3]].real),
        add_fairy2i_exact_bf16_bits(qstage[2][code[2]].imag, qstage[3][code[3]].imag),
    };
    return {
        add_fairy2i_exact_bf16_bits(u.real, w.real),
        add_fairy2i_exact_bf16_bits((uint16_t) (u.imag ^ 0x8000u), w.imag),
        add_fairy2i_exact_bf16_bits(u.imag, w.imag),
        add_fairy2i_exact_bf16_bits(u.real, (uint16_t) (w.real ^ 0x8000u)),
    };
}

static fairy2i_exact_coeff_bits reconstruct_fairy2i_exact_coeff_bits(const uint8_t code[4], const float scales[8]) {
    float real[4] = {};
    float imag[4] = {};
    for (int branch = 0; branch < 4; ++branch) {
        const bool  is_real = (code[branch] & 2u) == 0;
        const float scale   = scales[2 * branch + (is_real ? 0 : 1)];
        const float stage   = bf16_round((code[branch] & 1u) != 0 ? scale : -scale);
        if (is_real) {
            real[branch] = stage;
        } else {
            imag[branch] = stage;
        }
    }

    const float u_real = bf16_round(real[0] + real[1]);
    const float u_imag = bf16_round(imag[0] + imag[1]);
    const float w_real = bf16_round(real[2] + real[3]);
    const float w_imag = bf16_round(imag[2] + imag[3]);
    return {
        bf16_bits(bf16_round(u_real + w_real)),
        bf16_bits(bf16_round(-u_imag + w_imag)),
        bf16_bits(bf16_round(u_imag + w_imag)),
        bf16_bits(bf16_round(u_real - w_real)),
    };
}

static uint32_t fairy2i_exact_bf16_product_metric(uint16_t value) {
    const uint32_t abs_value = (uint32_t) value & 0x7fffu;
    if (abs_value == 0) {
        return 255u;
    }
    const uint32_t exponent = abs_value >> 7;
    return exponent == 0 || exponent == 0xffu ? 0u : exponent;
}

static uint32_t fairy2i_exact_coefficient_metric_bound(const fairy2i_exact_stage_bits qstage[4][4]) {
    uint32_t min_stage_metric   = 255u;
    uint32_t max_stage_exponent = 0u;
    for (int branch = 0; branch < 4; ++branch) {
        const uint16_t stage_bits[2] = {
            qstage[branch][0].real,
            qstage[branch][2].imag,
        };
        for (uint16_t bits : stage_bits) {
            const uint32_t metric = fairy2i_exact_bf16_product_metric(bits);
            min_stage_metric      = std::min(min_stage_metric, metric);
            max_stage_exponent    = std::max(max_stage_exponent, metric == 255u ? 0u : metric);
        }
    }

    if (min_stage_metric == 255u) {
        return 255u;
    }
    if (min_stage_metric <= 14u || max_stage_exponent >= 253u) {
        return 0u;
    }
    return min_stage_metric - 14u;
}

static bool fairy2i_exact_product_metrics_require_software(uint32_t lhs, uint32_t rhs) {
    const bool lhs_nonzero = lhs != 255u;
    const bool rhs_nonzero = rhs != 255u;
    return lhs == 0u || rhs == 0u || (lhs_nonzero && rhs_nonzero && lhs + rhs < 142u);
}

static bool test_fairy2i_exact_coefficient_metric_bound() {
    struct bound_profile {
        const char * label;
        float        scales[8];
        uint32_t     expected_bound;
    };

    const bound_profile profiles[] = {
        { "all-zero",                             {},                                                                                   255u },
        { "signed-zero",                          { 0.0f, -0.0f, -0.0f, 0.0f, 0.0f, -0.0f, -0.0f, 0.0f },                               255u },
        { "round-to-zero",                        { 0x1p-134f, -0x1p-134f, 0x1p-135f, -0x1p-135f, 0x1p-136f, -0x1p-136f, 0.0f, -0.0f }, 255u },
        { "minimum-subnormal",                    { 0x1p-133f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },                              0u   },
        { "minimum-normal",
         { bf16_from_bits(0x0080), bf16_from_bits(0x8080), bf16_from_bits(0x0081), bf16_from_bits(0x8081),
            bf16_from_bits(0x00ff), bf16_from_bits(0x80ff), bf16_from_bits(0x0080), bf16_from_bits(0x8080) },
         0u                                                                                                                                  },
        { "exponent-14-boundary",
         { bf16_from_bits(14u << 7), bf16_from_bits((14u << 7) | 1u), bf16_from_bits(14u << 7),
            bf16_from_bits((14u << 7) | 2u), bf16_from_bits(14u << 7), bf16_from_bits((14u << 7) | 3u),
            bf16_from_bits(14u << 7), bf16_from_bits((14u << 7) | 4u) },
         0u                                                                                                                                  },
        { "exponent-15-boundary",
         { bf16_from_bits(15u << 7), bf16_from_bits((15u << 7) | 1u), bf16_from_bits(15u << 7),
            bf16_from_bits((15u << 7) | 2u), bf16_from_bits(15u << 7), bf16_from_bits((15u << 7) | 3u),
            bf16_from_bits(15u << 7), bf16_from_bits((15u << 7) | 4u) },
         1u                                                                                                                                  },
        { "mixed-exponents-with-zero",
         { 0.0f, bf16_from_bits(20u << 7), bf16_from_bits((40u << 7) | 7u), -0.0f, bf16_from_bits((80u << 7) | 31u),
            bf16_from_bits((160u << 7) | 63u), bf16_from_bits(200u << 7), bf16_from_bits((252u << 7) | 127u) },
         6u                                                                                                                                  },
        { "real-model-exponents-116-through-122",
         { bf16_from_bits((116u << 7) | 1u), bf16_from_bits((117u << 7) | 3u), bf16_from_bits((118u << 7) | 7u),
            bf16_from_bits((119u << 7) | 15u), bf16_from_bits((120u << 7) | 31u), bf16_from_bits((121u << 7) | 63u),
            bf16_from_bits((122u << 7) | 95u), bf16_from_bits((122u << 7) | 127u) },
         102u                                                                                                                                },
        { "RNE-ties-and-non-BF16-F32",
         { 1.0f + 0x1p-8f, 1.0f + 3.0f * 0x1p-8f, 0.5f + 0x1p-9f, 0.5f + 3.0f * 0x1p-9f, 0.25f + 0x1p-10f + 0x1p-20f,
            0.25f + 3.0f * 0x1p-10f - 0x1p-21f, 0.125f + 0x1p-11f, 0.125f + 3.0f * 0x1p-11f },
         110u                                                                                                                                },
        { "opposite-sign-cancellation",
         { bf16_from_bits(0x3f80), bf16_from_bits(0xbf7f), bf16_from_bits(0x3f01), bf16_from_bits(0xbf00),
            bf16_from_bits(0x3e81), bf16_from_bits(0xbe80), bf16_from_bits(0x3e01), bf16_from_bits(0xbe00) },
         110u                                                                                                                                },
        { "exponent-252-max-tail",
         { bf16_from_bits(0x7e7f), bf16_from_bits(0xfe7f), bf16_from_bits(0x7e7f), bf16_from_bits(0xfe7f),
            bf16_from_bits(0x7e7f), bf16_from_bits(0xfe7f), bf16_from_bits(0x7e7f), bf16_from_bits(0xfe7f) },
         238u                                                                                                                                },
        { "exponent-253-overflow-guard",
         { bf16_from_bits(0x7e80), bf16_from_bits(0xfe80), 1.0f, -1.0f, 0.5f, -0.5f, 0.25f, -0.25f },
         0u                                                                                                                                  },
        { "exponent-254-overflow-guard",
         { bf16_from_bits(0x7f00), bf16_from_bits(0xff00), 1.0f, -1.0f, 0.5f, -0.5f, 0.25f, -0.25f },
         0u                                                                                                                                  },
        { "maximum-finite",
         { bf16_from_bits(0x7f7f), bf16_from_bits(0xff7f), bf16_from_bits(0x7f7f), bf16_from_bits(0xff7f),
            bf16_from_bits(0x7f7f), bf16_from_bits(0xff7f), bf16_from_bits(0x7f7f), bf16_from_bits(0xff7f) },
         0u                                                                                                                                  },
        { "infinity",                             { bf16_from_bits(0x7f80), 1.0f, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f },  0u   },
        { "NaN",                                  { bf16_from_bits(0x7fc1), 1.0f, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f },  0u   },
    };

    bool ok = true;
    for (const bound_profile & profile : profiles) {
        fairy2i_exact_stage_bits qstage[4][4];
        build_fairy2i_exact_qstage_bits(profile.scales, qstage);
        const uint32_t bound = fairy2i_exact_coefficient_metric_bound(qstage);
        if (bound != profile.expected_bound) {
            fprintf(stderr, "Exact coefficient metric bound mismatch for %s: got=%u expected=%u\n", profile.label,
                    bound, profile.expected_bound);
            ok = false;
        }

        const bool replays_with_zero_activation = fairy2i_exact_product_metrics_require_software(bound, 255u);
        if (replays_with_zero_activation != (profile.expected_bound == 0u)) {
            fprintf(stderr, "Exact coefficient metric replay mismatch for %s: bound=%u replay=%d\n", profile.label,
                    bound, (int) replays_with_zero_activation);
            ok = false;
        }

        for (unsigned pattern = 0; pattern < 256; ++pattern) {
            const uint8_t code[4] = {
                (uint8_t) (pattern & 3u),
                (uint8_t) ((pattern >> 2) & 3u),
                (uint8_t) ((pattern >> 4) & 3u),
                (uint8_t) ((pattern >> 6) & 3u),
            };
            const fairy2i_exact_coeff_bits coeff = reconstruct_fairy2i_exact_coeff_from_qstage_bits(code, qstage);
            const uint16_t                 components[4] = { coeff.rr, coeff.ri, coeff.ir, coeff.ii };
            for (unsigned component = 0; component < 4; ++component) {
                const uint16_t actual_bits = components[component];
                if ((actual_bits & 0x7fffu) == 0) {
                    continue;
                }
                const uint32_t actual_metric = fairy2i_exact_bf16_product_metric(actual_bits);
                if (bound > actual_metric) {
                    fprintf(stderr,
                            "Exact coefficient metric bound exceeded actual for %s pattern=0x%02x component=%u: "
                            "bound=%u actual=0x%04x metric=%u\n",
                            profile.label, pattern, component, bound, (unsigned) actual_bits, actual_metric);
                    ok = false;
                }
            }
        }
    }

    printf("  Fairy2i exact coefficient metric bound: 256 patterns x %zu profiles - %s\n",
           sizeof(profiles) / sizeof(profiles[0]), ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_fairy2i_exact_coefficient_lut_patterns() {
    static const float normal_scales[8] = {
        1.0f + 0x1p-8f + 0x1p-20f, 0.5f, 0x1p-8f, 0x1p-9f, 0x1p-7f, 0x1p-8f, 0.0f, 0.0f,
    };
    static const float subnormal_scales[8] = {
        0x1p-126f, 0x1p-133f, 0x1p-132f, 0x1p-131f, 0x1p-130f, 0x1p-129f, 0x1p-128f, 0x1p-127f,
    };
    static const float signed_zero_scales[8] = {
        0.0f, -0.0f, -0.0f, 0.0f, 0.0f, 0.0f, -0.0f, -0.0f,
    };
    static const float tie_scales[8] = {
        1.0f + 0x1p-8f,   1.0f + 3.0f * 0x1p-8f,   0.5f + 0x1p-9f,    0.5f + 3.0f * 0x1p-9f,
        0.25f + 0x1p-10f, 0.25f + 3.0f * 0x1p-10f, 0.125f + 0x1p-11f, 0.125f + 3.0f * 0x1p-11f,
    };
    static const float * scale_profiles[] = { normal_scales, subnormal_scales, signed_zero_scales, tie_scales };

    bool ok = true;
    for (const float * scales : scale_profiles) {
        fairy2i_exact_stage_bits qstage[4][4];
        build_fairy2i_exact_qstage_bits(scales, qstage);
        if (scales == signed_zero_scales &&
            (qstage[0][0].real != 0x8000 || qstage[0][0].imag != 0x0000 || qstage[0][1].real != 0x0000 ||
             qstage[0][1].imag != 0x0000 || qstage[0][2].real != 0x0000 || qstage[0][2].imag != 0x0000 ||
             qstage[0][3].real != 0x0000 || qstage[0][3].imag != 0x8000)) {
            fprintf(stderr, "Exact coefficient qstage signed-zero mismatch\n");
            ok = false;
        }

        fairy2i_exact_coeff_bits lut[256];
        for (unsigned pattern = 0; pattern < 256; ++pattern) {
            const uint8_t code[4] = {
                (uint8_t) (pattern & 3u),
                (uint8_t) ((pattern >> 2) & 3u),
                (uint8_t) ((pattern >> 4) & 3u),
                (uint8_t) ((pattern >> 6) & 3u),
            };
            lut[pattern] = reconstruct_fairy2i_exact_coeff_from_qstage_bits(code, qstage);
        }
        if (scales == subnormal_scales &&
            (lut[0x55].rr != 0x00aa || lut[0x55].ri != 0x0000 || lut[0x55].ir != 0x0000 || lut[0x55].ii != 0x005a ||
             lut[0xff].rr != 0x0000 || lut[0xff].ri != 0x004b || lut[0xff].ir != 0x0055 || lut[0xff].ii != 0x0000)) {
            fprintf(stderr, "Exact coefficient LUT subnormal reconstruction mismatch\n");
            ok = false;
        }

        for (unsigned u0 = 0; u0 < 4; ++u0) {
            for (unsigned u1 = 0; u1 < 4; ++u1) {
                for (unsigned w0 = 0; w0 < 4; ++w0) {
                    for (unsigned w1 = 0; w1 < 4; ++w1) {
                        const uint8_t code[4] = {
                            (uint8_t) u0,
                            (uint8_t) u1,
                            (uint8_t) w0,
                            (uint8_t) w1,
                        };
                        const unsigned                 pattern = u0 | (u1 << 2) | (w0 << 4) | (w1 << 6);
                        const fairy2i_exact_coeff_bits direct  = reconstruct_fairy2i_exact_coeff_bits(code, scales);
                        if (memcmp(&lut[pattern], &direct, sizeof(direct)) != 0) {
                            fprintf(stderr, "Exact coefficient LUT mismatch at pattern=0x%02x\n", pattern);
                            ok = false;
                        }
                    }
                }
            }
        }
    }

    printf("  Fairy2i exact coefficient qstage/UW: 256 patterns x %zu profiles - %s\n",
           sizeof(scale_profiles) / sizeof(scale_profiles[0]), ok ? "PASS" : "FAIL");
    return ok;
}

static uint32_t transpose_fairy2i_exact_branch_code_bytes(uint32_t branch_bytes) {
    uint32_t swap = (branch_bytes ^ (branch_bytes >> 12)) & UINT32_C(0x0000f0f0);
    branch_bytes ^= swap ^ (swap << 12);
    swap = (branch_bytes ^ (branch_bytes >> 6)) & UINT32_C(0x00cc00cc);
    return branch_bytes ^ swap ^ (swap << 6);
}

static uint32_t transpose_fairy2i_exact_branch_code_bytes_reference(uint32_t branch_bytes) {
    uint32_t lut_indices = 0;
    for (unsigned part = 0; part < 4; ++part) {
        for (unsigned branch = 0; branch < 4; ++branch) {
            lut_indices |= ((branch_bytes >> (8 * branch + 2 * part)) & 3u) << (8 * part + 2 * branch);
        }
    }
    return lut_indices;
}

static bool test_fairy2i_exact_branch_code_transpose() {
    static const uint32_t samples[] = {
        UINT32_C(0x00000000), UINT32_C(0xffffffff), UINT32_C(0xe4e4e4e4),
        UINT32_C(0x1b1b1b1b), UINT32_C(0x01234567), UINT32_C(0x89abcdef),
    };
    static const uint32_t fill_profiles[] = {
        UINT32_C(0x00000000),
        UINT32_C(0xffffffff),
        UINT32_C(0x6c93b1e4),
    };

    bool ok = true;
    for (uint32_t branch_bytes : samples) {
        if (transpose_fairy2i_exact_branch_code_bytes(branch_bytes) !=
            transpose_fairy2i_exact_branch_code_bytes_reference(branch_bytes)) {
            fprintf(stderr, "Exact branch-code SWAR transpose mismatch for bytes=0x%08x\n", branch_bytes);
            ok = false;
        }
    }

    for (unsigned part = 0; part < 4; ++part) {
        const uint32_t part_mask = UINT32_C(0x03030303) << (2 * part);
        for (uint32_t fill : fill_profiles) {
            for (unsigned pattern = 0; pattern < 256; ++pattern) {
                uint32_t branch_bytes = fill & ~part_mask;
                for (unsigned branch = 0; branch < 4; ++branch) {
                    branch_bytes |= ((pattern >> (2 * branch)) & 3u) << (8 * branch + 2 * part);
                }
                if (transpose_fairy2i_exact_branch_code_bytes(branch_bytes) !=
                    transpose_fairy2i_exact_branch_code_bytes_reference(branch_bytes)) {
                    fprintf(stderr,
                            "Exact branch-code SWAR transpose mismatch at part=%u pattern=0x%02x bytes=0x%08x\n", part,
                            pattern, branch_bytes);
                    ok = false;
                }
            }
        }
    }

    printf("  Fairy2i exact branch-code SWAR transpose: 4 parts x 256 patterns x %zu fills - %s\n",
           sizeof(fill_profiles) / sizeof(fill_profiles[0]), ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_fairy2i_exact_coefficient_stage_bits() {
    const float u0_real = bf16_round(1.0f + 0x1p-8f + 0x1p-20f);
    const float u1_real = bf16_round(0x1p-8f);
    const float w0_real = bf16_round(0x1p-7f);
    const float w1_real = bf16_round(0.0f);
    const float u_real  = bf16_round(u0_real + u1_real);
    const float w_real  = bf16_round(w0_real + w1_real);
    const float a11     = bf16_round(w_real + u_real);
    const float a22     = bf16_round(-w_real + u_real);

    const float u0_imag = bf16_round(0.5f);
    const float u1_imag = bf16_round(0x1p-9f);
    const float w0_imag = bf16_round(0x1p-8f);
    const float w1_imag = bf16_round(0.0f);
    const float u_imag  = bf16_round(u0_imag + u1_imag);
    const float w_imag  = bf16_round(w0_imag + w1_imag);
    const float a12     = bf16_round(w_imag - u_imag);
    const float a21     = bf16_round(w_imag + u_imag);

    struct stage_case {
        const char * label;
        float        value;
        uint16_t     expected;
    };

    const stage_case cases[] = {
        { "F32 scale to BF16", u0_real, 0x3f81 },
        { "U real merge",      u_real,  0x3f82 },
        { "W real merge",      w_real,  0x3c00 },
        { "A11",               a11,     0x3f83 },
        { "A22",               a22,     0x3f81 },
        { "U imag merge",      u_imag,  0x3f00 },
        { "W imag merge",      w_imag,  0x3b80 },
        { "A12",               a12,     0xbefe },
        { "A21",               a21,     0x3f01 },
    };

    bool ok = true;
    for (const stage_case & tc : cases) {
        const uint16_t actual = bf16_bits(tc.value);
        if (actual != tc.expected) {
            fprintf(stderr, "Exact coefficient %s mismatch: actual=0x%04x expected=0x%04x\n", tc.label,
                    (unsigned) actual, (unsigned) tc.expected);
            ok = false;
        }
    }

    printf("  Fairy2i exact coefficient stage bits: %zu stages - %s\n", sizeof(cases) / sizeof(cases[0]),
           ok ? "PASS" : "FAIL");
    return ok;
}

static bool compare_bf16_value_bits(const char *               label,
                                    const std::vector<float> & actual,
                                    const std::vector<float> & expected) {
    if (actual.size() != expected.size()) {
        fprintf(stderr, "%s size mismatch: %zu vs %zu\n", label, actual.size(), expected.size());
        return false;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        const uint16_t actual_bits   = bf16_bits(actual[i]);
        const uint16_t expected_bits = bf16_bits(expected[i]);
        if (actual_bits != expected_bits || actual[i] != bf16_round(actual[i])) {
            fprintf(stderr,
                    "%s mismatch i=%zu actual=%g/0x%04x expected=%g/0x%04x (actual must be BF16-expanded F32)\n", label,
                    i, actual[i], (unsigned) actual_bits, expected[i], (unsigned) expected_bits);
            return false;
        }
    }
    return true;
}

static uint16_t bf16_ordered_rank(uint16_t bits) {
    return (bits & 0x8000U) != 0 ? (uint16_t) ~bits : (uint16_t) (bits ^ 0x8000U);
}

static bool compare_bf16_value_bits_one_step(const char *               label,
                                             const std::vector<float> & actual,
                                             const std::vector<float> & expected) {
    if (actual.size() != expected.size()) {
        fprintf(stderr, "%s size mismatch: %zu vs %zu\n", label, actual.size(), expected.size());
        return false;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        const uint16_t actual_bits   = bf16_bits(actual[i]);
        const uint16_t expected_bits = bf16_bits(expected[i]);
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || actual[i] != bf16_round(actual[i]) ||
            expected[i] != bf16_round(expected[i])) {
            fprintf(stderr, "%s non-finite or non-BF16-expanded value i=%zu actual=%g/0x%04x expected=%g/0x%04x\n",
                    label, i, actual[i], (unsigned) actual_bits, expected[i], (unsigned) expected_bits);
            return false;
        }

        const uint16_t actual_rank   = bf16_ordered_rank(actual_bits);
        const uint16_t expected_rank = bf16_ordered_rank(expected_bits);
        const uint32_t distance =
            actual_rank > expected_rank ? actual_rank - expected_rank : expected_rank - actual_rank;
        if (distance > 1U) {
            fprintf(stderr, "%s exceeds one BF16 step i=%zu actual=%g/0x%04x expected=%g/0x%04x distance=%u\n", label,
                    i, actual[i], (unsigned) actual_bits, expected[i], (unsigned) expected_bits, (unsigned) distance);
            return false;
        }
    }
    return true;
}

static ggml_backend_dev_t find_metal_test_device();
static bool               compare_exact(const char *                  label,
                                        const std::vector<uint32_t> & actual,
                                        const std::vector<uint32_t> & expected);

static std::vector<float> fairy2i_exact_rms_norm_oracle(const std::vector<float> & x,
                                                        const std::vector<float> & weight,
                                                        int64_t                    n_dims,
                                                        int64_t                    n_rows,
                                                        float                      eps) {
    std::vector<float> out(x.size());
    for (int64_t row = 0; row < n_rows; ++row) {
        const float * src = x.data() + (size_t) row * (size_t) n_dims;
        float         sum = 0.0f;
        for (int64_t i = 0; i < n_dims; ++i) {
            sum = fmaf(src[i], src[i], sum);
        }
        const float inv_rms = 1.0f / sqrtf(sum / (float) n_dims + eps);
        for (int64_t i = 0; i < n_dims; ++i) {
            const float normalized                           = bf16_round(src[i] * inv_rms);
            const float weight_bf16                          = bf16_round(weight[(size_t) i]);
            out[(size_t) row * (size_t) n_dims + (size_t) i] = bf16_round(normalized * weight_bf16);
        }
    }
    return out;
}

static bool run_fairy2i_exact_rms_norm_backend(std::vector<float> &       out,
                                               ggml_backend_t             backend,
                                               const std::vector<float> & x_data,
                                               const std::vector<float> & weight_data,
                                               int64_t                    n_dims,
                                               int64_t                    n_rows,
                                               float                      eps) {
    struct ggml_init_params params = {
        /*.mem_size   =*/256 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_dims, n_rows);
    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_dims);
    ggml_tensor * y      = ggml_fairy2i_rms_norm_exact(ctx, x, weight, eps);
    if (!ggml_backend_supports_op(backend, y)) {
        ggml_free(ctx);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(x, x_data.data(), 0, x_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status == GGML_STATUS_SUCCESS) {
        out.resize(x_data.size());
        ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return status == GGML_STATUS_SUCCESS;
}

static bool test_fairy2i_exact_rms_norm() {
    ggml_backend_dev_t metal_dev = find_metal_test_device();
    bool               ok        = true;

    auto run_case = [&](const char * label, const std::vector<float> & x, const std::vector<float> & weight,
                        int64_t n_dims, int64_t n_rows, float eps, bool metal_allow_one_bf16_step) {
        const std::vector<float> expected = fairy2i_exact_rms_norm_oracle(x, weight, n_dims, n_rows, eps);
        ggml_backend_t           cpu      = ggml_backend_cpu_init();
        if (!cpu) {
            return false;
        }
        ggml_backend_cpu_set_n_threads(cpu, 3);

        std::vector<float> actual;
        std::string        cpu_label = std::string(label) + " CPU";
        bool               case_ok = run_fairy2i_exact_rms_norm_backend(actual, cpu, x, weight, n_dims, n_rows, eps) &&
                                     compare_bf16_value_bits(cpu_label.c_str(), actual, expected);
        ggml_backend_free(cpu);

        if (metal_dev) {
            ggml_backend_t     metal = ggml_backend_dev_init(metal_dev, nullptr);
            std::vector<float> metal_out;
            std::string        metal_label = std::string(label) + " Metal";
            if (metal && run_fairy2i_exact_rms_norm_backend(metal_out, metal, x, weight, n_dims, n_rows, eps)) {
                case_ok = (metal_allow_one_bf16_step ?
                               compare_bf16_value_bits_one_step(metal_label.c_str(), metal_out, expected) :
                               compare_bf16_value_bits(metal_label.c_str(), metal_out, expected)) &&
                          case_ok;
            } else {
                case_ok = false;
            }
            if (metal) {
                ggml_backend_free(metal);
            }
        }
        return case_ok;
    };

    {
        const int64_t      n_dims = 8;
        const int64_t      n_rows = 3;
        std::vector<float> x((size_t) n_dims * (size_t) n_rows);
        for (int64_t row = 0; row < n_rows; ++row) {
            for (int64_t i = 0; i < n_dims; ++i) {
                const float value = (float) (((5 * i + 3 * row) % 17) - 8) / (float) (4 + row);
                x[(size_t) row * (size_t) n_dims + (size_t) i] = bf16_round(value);
            }
        }
        std::vector<float> weight((size_t) n_dims);
        for (int64_t i = 0; i < n_dims; ++i) {
            weight[(size_t) i] = 0.75f + (float) i * 0.0625f;
        }
        weight[0] = 1.0f + 0x1p-8f + 0x1p-20f;
        weight[7] = -0.5f - 0x1p-9f - 0x1p-20f;
        ok        = run_case("Fairy2i exact RMSNorm base", x, weight, n_dims, n_rows, 1.0e-5f, false) && ok;
    }

    {
        constexpr int64_t  n_dims = 5120;
        std::vector<float> x((size_t) n_dims, bf16_round(6.1875f));
        std::vector<float> weight((size_t) n_dims, 1.0f);
        x[0] = bf16_round(4768.0f);
        ok   = run_case("Fairy2i exact RMSNorm F32 division", x, weight, n_dims, 1, 1.0e-5f, true) && ok;
    }

    {
        constexpr int64_t  n_dims    = 32768;
        constexpr int64_t  n_rows    = 2;
        constexpr float    values[]  = { 0.5f, -0.5f, 1.0f, -1.0f, 1.5f, -1.5f, 2.0f, -2.0f };
        constexpr float    weights[] = { 0.5f, 0.75f, 1.0f, -1.25f };
        std::vector<float> x((size_t) n_dims * (size_t) n_rows);
        std::vector<float> weight((size_t) n_dims);
        for (int64_t row = 0; row < n_rows; ++row) {
            for (int64_t i = 0; i < n_dims; ++i) {
                x[(size_t) row * (size_t) n_dims + (size_t) i] =
                    values[((size_t) i + 3U * (size_t) row) % (sizeof(values) / sizeof(values[0]))];
            }
        }
        for (int64_t i = 0; i < n_dims; ++i) {
            weight[(size_t) i] = weights[(size_t) i % (sizeof(weights) / sizeof(weights[0]))];
        }
        ok = run_case("Fairy2i exact RMSNorm large normal SIMD reduction", x, weight, n_dims, n_rows, 1.0e-5f, false) &&
             ok;
    }

    {
        constexpr int64_t  n_dims = 8;
        std::vector<float> x((size_t) n_dims, bf16_from_bits(0x0001));
        std::vector<float> weight((size_t) n_dims, 1.0f);
        ok = run_case("Fairy2i exact RMSNorm subnormal", x, weight, n_dims, 1, 1.0f, false) && ok;
    }

    {
        constexpr int64_t  n_dims = 8;
        // BF16 0x1f80 is 2^-64, whose square is a non-zero F32
        // subnormal. This forces the Metal bit-domain accumulation path.
        std::vector<float> x((size_t) n_dims, bf16_from_bits(0x1f80));
        std::vector<float> weight((size_t) n_dims, 1.0f);
        ok = run_case("Fairy2i exact RMSNorm subnormal square", x, weight, n_dims, 1, 0.0f, false) && ok;
    }

    {
        constexpr int64_t  n_dims = 64;
        std::vector<float> x((size_t) n_dims, 0.0f);
        std::vector<float> weight((size_t) n_dims, 1.0f);
        x[31] = bf16_from_bits(0x1f80);
        ok    = run_case("Fairy2i exact RMSNorm remote-lane fallback", x, weight, n_dims, 1, 0.0f, false) && ok;
    }

    {
        constexpr int64_t  n_dims = 8;
        std::vector<float> x((size_t) n_dims, 0.0f);
        std::vector<float> weight((size_t) n_dims, 1.0f);
        x[0] = bf16_from_bits(0x1f80);
        // E=52,f=0 squares to an exact RNE zero. The current accumulation
        // step must nevertheless preserve the preceding subnormal accumulator.
        x[1] = bf16_from_bits(0x1a00);
        ok   = run_case("Fairy2i exact RMSNorm mixed subnormal accumulator", x, weight, n_dims, 1, 0.0f, false) && ok;
    }

    {
        constexpr int64_t  n_dims = 3912;
        std::vector<float> x((size_t) n_dims, bf16_from_bits(0x1d03));
        std::vector<float> weight((size_t) n_dims, 1.0f);
        // The prefix reaches F32 acc 0x008001e7. The last square is exactly
        // half a minimum-subnormal ULP and must tie-to-even to 0x008001e8.
        x.back() = bf16_from_bits(0x1a00);
        ok =
            run_case("Fairy2i exact RMSNorm E52 tie after normal accumulator", x, weight, n_dims, 1, 0.0f, false) && ok;
    }

    {
        constexpr int64_t  n_dims = 8;
        std::vector<float> x((size_t) n_dims, 1.0f);
        std::vector<float> weight((size_t) n_dims, 1.0f);
        weight[0] = bf16_from_bits(0x0001);
        ok        = run_case("Fairy2i exact RMSNorm subnormal weight", x, weight, n_dims, 1, 0.0f, false) && ok;
    }

    printf("  Fairy2i exact RMSNorm BF16 boundaries/large normal CPU%s: %s\n", metal_dev ? "+Metal" : "",
           ok ? "PASS" : "FAIL");
    return ok;
}

static std::vector<float> fairy2i_exact_rope_oracle(const std::vector<float> &   x,
                                                    const std::vector<int32_t> & positions,
                                                    const std::vector<float> *   freq_factors,
                                                    int64_t                      n_dims_total,
                                                    int64_t                      n_heads,
                                                    int64_t                      n_tokens,
                                                    int64_t                      n_batches,
                                                    int64_t                      n_dims_rope,
                                                    float                        freq_base) {
    std::vector<float> out(x.size());
    const float        theta_scale = powf(freq_base, -2.0f / (float) n_dims_rope);

    for (int64_t batch = 0; batch < n_batches; ++batch) {
        for (int64_t token = 0; token < n_tokens; ++token) {
            for (int64_t head = 0; head < n_heads; ++head) {
                const size_t row_base =
                    (((size_t) batch * (size_t) n_tokens + (size_t) token) * (size_t) n_heads + (size_t) head) *
                    (size_t) n_dims_total;
                float theta = (float) positions[(size_t) token];
                for (int64_t i0 = 0; i0 < n_dims_rope; i0 += 2) {
                    const int64_t ic            = i0 / 2;
                    const float   freq_factor   = freq_factors ? (*freq_factors)[(size_t) ic] : 1.0f;
                    const float   cos_bf16      = bf16_round(cosf(theta / freq_factor));
                    const float   sin_bf16      = bf16_round(sinf(theta / freq_factor));
                    const float   x0            = bf16_round(x[row_base + (size_t) ic]);
                    const float   x1            = bf16_round(x[row_base + (size_t) ic + (size_t) n_dims_rope / 2u]);
                    const float   x0_cos        = bf16_round(x0 * cos_bf16);
                    const float   x1_sin        = bf16_round(x1 * sin_bf16);
                    const float   x0_sin        = bf16_round(x0 * sin_bf16);
                    const float   x1_cos        = bf16_round(x1 * cos_bf16);
                    out[row_base + (size_t) ic] = bf16_round(x0_cos - x1_sin);
                    out[row_base + (size_t) ic + (size_t) n_dims_rope / 2u] = bf16_round(x0_sin + x1_cos);
                    theta *= theta_scale;
                }
                for (int64_t i = n_dims_rope; i < n_dims_total; ++i) {
                    out[row_base + (size_t) i] = bf16_round(x[row_base + (size_t) i]);
                }
            }
        }
    }
    return out;
}

static bool run_fairy2i_exact_rope_backend(std::vector<float> &         out,
                                           ggml_backend_t               backend,
                                           const std::vector<float> &   x_data,
                                           const std::vector<int32_t> & positions,
                                           const std::vector<float> *   freq_factors,
                                           int64_t                      n_dims_total,
                                           int64_t                      n_heads,
                                           int64_t                      n_tokens,
                                           int64_t                      n_batches,
                                           int64_t                      n_dims_rope,
                                           float                        freq_base) {
    struct ggml_init_params params = {
        /*.mem_size   =*/256 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * x   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_dims_total, n_heads, n_tokens, n_batches);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_tensor * factors =
        freq_factors ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) freq_factors->size()) : nullptr;
    ggml_tensor * y = ggml_fairy2i_rope_ext_exact(ctx, x, pos, factors, (int) n_dims_rope, GGML_ROPE_TYPE_NEOX, 4096,
                                                  freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    if (!ggml_backend_supports_op(backend, y)) {
        ggml_free(ctx);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(x, x_data.data(), 0, x_data.size() * sizeof(float));
    ggml_backend_tensor_set(pos, positions.data(), 0, positions.size() * sizeof(int32_t));
    if (factors) {
        ggml_backend_tensor_set(factors, freq_factors->data(), 0, freq_factors->size() * sizeof(float));
    }
    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status == GGML_STATUS_SUCCESS) {
        out.resize(x_data.size());
        ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return status == GGML_STATUS_SUCCESS;
}

static bool test_fairy2i_exact_rope() {
    const int64_t n_dims_total = 8;
    const int64_t n_dims_rope  = 4;
    const int64_t n_heads      = 5;
    const int64_t n_tokens     = 2;
    const int64_t n_batches    = 2;
    const float   freq_base    = 10000.0f;

    std::vector<float> x((size_t) n_dims_total * (size_t) n_heads * (size_t) n_tokens * (size_t) n_batches);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = bf16_round((float) ((int) (7 * i % 29) - 14) / 8.0f);
    }
    x[0]                                    = bf16_from_bits(0x0001);
    x[(size_t) n_dims_rope / 2u]            = 0.0f;
    const std::vector<int32_t> positions    = { 0, 7 };
    const std::vector<float>   expected     = fairy2i_exact_rope_oracle(x, positions, nullptr, n_dims_total, n_heads,
                                                                        n_tokens, n_batches, n_dims_rope, freq_base);
    const std::vector<float>   freq_factors = { 1.25f, 0.75f };
    const std::vector<float>   expected_factors = fairy2i_exact_rope_oracle(
        x, positions, &freq_factors, n_dims_total, n_heads, n_tokens, n_batches, n_dims_rope, freq_base);

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, 3);

    std::vector<float> actual;
    bool ok = run_fairy2i_exact_rope_backend(actual, backend, x, positions, nullptr, n_dims_total, n_heads, n_tokens,
                                             n_batches, n_dims_rope, freq_base) &&
              compare_bf16_value_bits("Fairy2i exact RoPE CPU", actual, expected);
    std::vector<float> actual_factors;
    ok = run_fairy2i_exact_rope_backend(actual_factors, backend, x, positions, &freq_factors, n_dims_total, n_heads,
                                        n_tokens, n_batches, n_dims_rope, freq_base) &&
         compare_bf16_value_bits("Fairy2i exact partial RoPE CPU with minimal factors", actual_factors,
                                 expected_factors) &&
         ok;
    ggml_backend_free(backend);

    ggml_backend_dev_t metal_dev = find_metal_test_device();
    if (metal_dev) {
        ggml_backend_t     metal = ggml_backend_dev_init(metal_dev, nullptr);
        std::vector<float> metal_out;
        ok = metal &&
             run_fairy2i_exact_rope_backend(metal_out, metal, x, positions, nullptr, n_dims_total, n_heads, n_tokens,
                                            n_batches, n_dims_rope, freq_base) &&
             compare_bf16_value_bits("Fairy2i exact RoPE Metal", metal_out, expected) && ok;
        std::vector<float> metal_factors;
        ok = metal &&
             run_fairy2i_exact_rope_backend(metal_factors, metal, x, positions, &freq_factors, n_dims_total, n_heads,
                                            n_tokens, n_batches, n_dims_rope, freq_base) &&
             compare_bf16_value_bits("Fairy2i exact partial RoPE Metal with minimal factors", metal_factors,
                                     expected_factors) &&
             ok;
        if (metal) {
            ggml_backend_free(metal);
        }
    }

    printf("  Fairy2i exact RoPE BF16 boundaries CPU%s: %s\n", metal_dev ? "+Metal" : "", ok ? "PASS" : "FAIL");
    return ok;
}

static std::vector<float> fairy2i_exact_flash_attn_oracle(const std::vector<float> &       q,
                                                          const std::vector<ggml_bf16_t> & k,
                                                          const std::vector<ggml_bf16_t> & v,
                                                          const std::vector<float> &       mask,
                                                          int64_t                          n_queries,
                                                          int64_t                          n_kv,
                                                          int64_t                          n_dims,
                                                          float                            scale) {
    std::vector<float> out((size_t) n_queries * (size_t) n_dims);
    std::vector<float> logits((size_t) n_kv);
    std::vector<float> probabilities((size_t) n_kv);

    for (int64_t query = 0; query < n_queries; ++query) {
        float row_max = -INFINITY;
        float row_sum = 0.0f;
        for (int64_t key = 0; key < n_kv; ++key) {
            const float mask_value = mask[(size_t) query * (size_t) n_kv + (size_t) key];
            if (mask_value == -INFINITY) {
                logits[(size_t) key] = -INFINITY;
                continue;
            }

            float qk = 0.0f;
            for (int64_t d = 0; d < n_dims; ++d) {
                const float q_bf16 = bf16_round(q[(size_t) query * (size_t) n_dims + (size_t) d]);
                const float k_bf16 = GGML_BF16_TO_FP32(k[(size_t) key * (size_t) n_dims + (size_t) d]);
                qk                 = fmaf(q_bf16, k_bf16, qk);
            }

            const float scaled   = bf16_round(qk * scale);
            const float logit    = bf16_round(scaled + mask_value);
            logits[(size_t) key] = logit;

            const float next_max = fmaxf(row_max, logit);
            row_sum              = row_sum * expf(row_max - next_max) + expf(logit - next_max);
            row_max              = next_max;
        }

        for (int64_t key = 0; key < n_kv; ++key) {
            probabilities[(size_t) key] =
                logits[(size_t) key] == -INFINITY ? 0.0f : bf16_round(expf(logits[(size_t) key] - row_max) / row_sum);
        }

        for (int64_t d = 0; d < n_dims; ++d) {
            float value = 0.0f;
            for (int64_t key = 0; key < n_kv; ++key) {
                const float v_bf16 = GGML_BF16_TO_FP32(v[(size_t) key * (size_t) n_dims + (size_t) d]);
                value              = fmaf(probabilities[(size_t) key], v_bf16, value);
            }
            out[(size_t) query * (size_t) n_dims + (size_t) d] = bf16_round(value);
        }
    }
    return out;
}

static std::vector<float> fairy2i_exact_flash_attn_gqa_oracle(const std::vector<float> &       q,
                                                              const std::vector<ggml_bf16_t> & k,
                                                              const std::vector<ggml_bf16_t> & v,
                                                              const std::vector<float> &       mask,
                                                              int64_t                          n_queries,
                                                              int64_t                          n_kv,
                                                              int64_t                          n_dims,
                                                              int64_t                          n_q_heads,
                                                              int64_t                          n_kv_heads,
                                                              float                            scale) {
    GGML_ASSERT(n_q_heads % n_kv_heads == 0);

    const int64_t      gqa_ratio = n_q_heads / n_kv_heads;
    std::vector<float> out((size_t) n_dims * (size_t) n_q_heads * (size_t) n_queries);
    for (int64_t q_head = 0; q_head < n_q_heads; ++q_head) {
        const int64_t            kv_head = q_head / gqa_ratio;
        std::vector<float>       q_slice(q.begin() + (size_t) q_head * (size_t) n_queries * (size_t) n_dims,
                                         q.begin() + (size_t) (q_head + 1) * (size_t) n_queries * (size_t) n_dims);
        std::vector<ggml_bf16_t> k_slice(k.begin() + (size_t) kv_head * (size_t) n_kv * (size_t) n_dims,
                                         k.begin() + (size_t) (kv_head + 1) * (size_t) n_kv * (size_t) n_dims);
        std::vector<ggml_bf16_t> v_slice(v.begin() + (size_t) kv_head * (size_t) n_kv * (size_t) n_dims,
                                         v.begin() + (size_t) (kv_head + 1) * (size_t) n_kv * (size_t) n_dims);
        const std::vector<float> head_out =
            fairy2i_exact_flash_attn_oracle(q_slice, k_slice, v_slice, mask, n_queries, n_kv, n_dims, scale);

        for (int64_t query = 0; query < n_queries; ++query) {
            for (int64_t d = 0; d < n_dims; ++d) {
                out[(size_t) d + (size_t) n_dims * ((size_t) q_head + (size_t) n_q_heads * (size_t) query)] =
                    head_out[(size_t) query * (size_t) n_dims + (size_t) d];
            }
        }
    }
    return out;
}

static bool run_fairy2i_exact_flash_attn_metal(std::vector<float> &             out,
                                               ggml_backend_dev_t               dev,
                                               const std::vector<float> &       q_data,
                                               const std::vector<ggml_bf16_t> & k_data,
                                               const std::vector<ggml_bf16_t> & v_data,
                                               const std::vector<float> &       mask_data,
                                               int64_t                          n_queries,
                                               int64_t                          n_kv,
                                               int64_t                          n_dims,
                                               float                            scale,
                                               int64_t                          n_q_heads  = 1,
                                               int64_t                          n_kv_heads = 1) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        return false;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/512 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_dims, n_queries, n_q_heads, 1);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, n_dims, n_kv, n_kv_heads, 1);
    ggml_tensor * v    = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, n_dims, n_kv, n_kv_heads, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, GGML_KQ_MASK_PAD, 1, 1);
    ggml_tensor * y    = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(y, GGML_PREC_F32);
    ggml_flash_attn_ext_set_fairy2i_exact(y, true);

    if (!ggml_backend_supports_op(backend, y)) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status == GGML_STATUS_SUCCESS) {
        out.resize((size_t) n_queries * (size_t) n_q_heads * (size_t) n_dims);
        ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return status == GGML_STATUS_SUCCESS;
}

static bool test_fairy2i_exact_flash_attn() {
    ggml_backend_dev_t dev = find_metal_test_device();
    if (!dev) {
        printf("  Fairy2i exact Flash Attention skipped: Metal unavailable\n");
        return true;
    }

    constexpr int64_t n_kv  = 32;
    constexpr float   scale = 0.125f;
    bool              ok    = true;

    for (int64_t n_dims : { 40, 64, 128 }) {
        for (int64_t n_queries : { 1, 8 }) {
            std::vector<float>       q((size_t) n_queries * (size_t) n_dims, 0.0f);
            std::vector<ggml_bf16_t> k((size_t) n_kv * (size_t) n_dims, GGML_FP32_TO_BF16(0.0f));
            std::vector<ggml_bf16_t> v((size_t) n_kv * (size_t) n_dims);
            std::vector<float>       mask((size_t) GGML_KQ_MASK_PAD * n_kv, -INFINITY);

            for (int64_t query = 0; query < n_queries; ++query) {
                q[(size_t) query * (size_t) n_dims] = bf16_round(1.0f + (float) query * 0.125f);
                for (int64_t key = 0; key < n_kv; ++key) {
                    const bool visible = n_queries == 1 || key <= query;
                    if (visible) {
                        const float tie_offset = key == 0 ? 0x1p-20f : 0.0f;
                        mask[(size_t) query * n_kv + (size_t) key] =
                            (key == 0 ? 1.0f + 0x1p-8f : (float) (key % 5) * 0.125f) + tie_offset;
                    }
                }
            }

            for (int64_t key = 0; key < n_kv; ++key) {
                k[(size_t) key * (size_t) n_dims] = GGML_FP32_TO_BF16((float) ((key % 7) - 3) * 0.25f);
                for (int64_t d = 0; d < n_dims; ++d) {
                    const float value                              = (float) (((3 * key + d) % 17) - 8) * 0.0625f;
                    v[(size_t) key * (size_t) n_dims + (size_t) d] = GGML_FP32_TO_BF16(value);
                }
            }

            const std::vector<float> expected =
                fairy2i_exact_flash_attn_oracle(q, k, v, mask, n_queries, n_kv, n_dims, scale);
            std::vector<float> actual;
            char               label[128];
            snprintf(label, sizeof(label), "Fairy2i exact Flash Attention Metal D=%lld N=%lld", (long long) n_dims,
                     (long long) n_queries);
            ok = run_fairy2i_exact_flash_attn_metal(actual, dev, q, k, v, mask, n_queries, n_kv, n_dims, scale) &&
                 compare_bf16_value_bits(label, actual, expected) && ok;
        }
    }

    {
        constexpr int64_t        n_dims     = 128;
        constexpr int64_t        n_queries  = 3;
        constexpr int64_t        n_q_heads  = 4;
        constexpr int64_t        n_kv_heads = 2;
        std::vector<float>       q((size_t) n_dims * n_queries * n_q_heads, 0.0f);
        std::vector<ggml_bf16_t> k((size_t) n_dims * n_kv * n_kv_heads, GGML_FP32_TO_BF16(0.0f));
        std::vector<ggml_bf16_t> v((size_t) n_dims * n_kv * n_kv_heads, GGML_FP32_TO_BF16(0.0f));
        std::vector<float>       mask((size_t) GGML_KQ_MASK_PAD * n_kv, -INFINITY);

        for (int64_t query = 0; query < n_queries; ++query) {
            mask[(size_t) query * n_kv]     = 0.0f;
            mask[(size_t) query * n_kv + 1] = 0.0f;
        }
        for (int64_t head = 0; head < n_q_heads; ++head) {
            for (int64_t query = 0; query < n_queries; ++query) {
                q[(size_t) n_dims * ((size_t) query + (size_t) n_queries * (size_t) head)] =
                    bf16_round(0.5f + 0.5f * (float) query);
            }
        }

        k[(size_t) n_dims]                           = GGML_FP32_TO_BF16(1.0f);
        k[(size_t) n_dims * n_kv]                    = GGML_FP32_TO_BF16(1.0f);
        v[0]                                         = GGML_FP32_TO_BF16(1.0f);
        v[(size_t) n_dims + 1]                       = GGML_FP32_TO_BF16(2.0f);
        v[(size_t) n_dims * n_kv]                    = GGML_FP32_TO_BF16(-1.0f);
        v[(size_t) n_dims * ((size_t) n_kv + 1) + 1] = GGML_FP32_TO_BF16(-2.0f);

        const std::vector<float> expected =
            fairy2i_exact_flash_attn_gqa_oracle(q, k, v, mask, n_queries, n_kv, n_dims, n_q_heads, n_kv_heads, 1.0f);
        std::vector<float> actual;
        ok = run_fairy2i_exact_flash_attn_metal(actual, dev, q, k, v, mask, n_queries, n_kv, n_dims, 1.0f, n_q_heads,
                                                n_kv_heads) &&
             compare_bf16_value_bits("Fairy2i exact Flash Attention Metal GQA 4:2", actual, expected) && ok;
    }

    {
        constexpr int64_t        n_dims     = 128;
        constexpr int64_t        n_queries  = 1;
        constexpr int64_t        n_q_heads  = 10;
        constexpr int64_t        n_kv_heads = 2;
        std::vector<float>       q((size_t) n_dims * n_queries * n_q_heads);
        std::vector<ggml_bf16_t> k((size_t) n_dims * n_kv * n_kv_heads);
        std::vector<ggml_bf16_t> v((size_t) n_dims * n_kv * n_kv_heads);
        std::vector<float>       mask((size_t) GGML_KQ_MASK_PAD * n_kv, -INFINITY);

        std::fill_n(mask.begin(), n_kv, 0.0f);
        for (int64_t head = 0; head < n_q_heads; ++head) {
            for (int64_t d = 0; d < n_dims; ++d) {
                q[(size_t) d + (size_t) n_dims * (size_t) head] =
                    bf16_round((float) (((7 * head + 3 * d) % 19) - 9) * 0.0625f);
            }
        }
        for (int64_t head = 0; head < n_kv_heads; ++head) {
            for (int64_t key = 0; key < n_kv; ++key) {
                for (int64_t d = 0; d < n_dims; ++d) {
                    const size_t index = (size_t) d + (size_t) n_dims * ((size_t) key + (size_t) n_kv * (size_t) head);
                    k[index]           = GGML_FP32_TO_BF16((float) (((5 * head + 11 * key + d) % 23) - 11) * 0.03125f);
                    v[index] = GGML_FP32_TO_BF16((float) (((13 * head + 3 * key + 5 * d) % 29) - 14) * 0.03125f);
                }
            }
        }

        const std::vector<float> expected =
            fairy2i_exact_flash_attn_gqa_oracle(q, k, v, mask, n_queries, n_kv, n_dims, n_q_heads, n_kv_heads, scale);
        std::vector<float> actual;
        ok = run_fairy2i_exact_flash_attn_metal(actual, dev, q, k, v, mask, n_queries, n_kv, n_dims, scale, n_q_heads,
                                                n_kv_heads) &&
             compare_bf16_value_bits("Fairy2i exact Flash Attention Metal decode GQA 10:2", actual, expected) && ok;
    }

    {
        constexpr int64_t        n_dims     = 128;
        constexpr int64_t        n_queries  = 1;
        constexpr int64_t        n_kv_long  = 512;
        constexpr int64_t        n_q_heads  = 40;
        constexpr int64_t        n_kv_heads = 8;
        std::vector<float>       q((size_t) n_dims * n_queries * n_q_heads);
        std::vector<ggml_bf16_t> k((size_t) n_dims * n_kv_long * n_kv_heads);
        std::vector<ggml_bf16_t> v((size_t) n_dims * n_kv_long * n_kv_heads);
        std::vector<float>       mask((size_t) GGML_KQ_MASK_PAD * n_kv_long, -INFINITY);

        std::fill_n(mask.begin(), n_kv_long, 0.0f);
        for (int64_t head = 0; head < n_q_heads; ++head) {
            for (int64_t d = 0; d < n_dims; ++d) {
                q[(size_t) d + (size_t) n_dims * (size_t) head] =
                    bf16_round((float) (((5 * head + 7 * d) % 31) - 15) * 0.03125f);
            }
        }
        for (int64_t head = 0; head < n_kv_heads; ++head) {
            for (int64_t key = 0; key < n_kv_long; ++key) {
                for (int64_t d = 0; d < n_dims; ++d) {
                    const size_t index =
                        (size_t) d + (size_t) n_dims * ((size_t) key + (size_t) n_kv_long * (size_t) head);
                    k[index] = GGML_FP32_TO_BF16((float) (((17 * head + 3 * key + 11 * d) % 37) - 18) * 0.015625f);
                    v[index] = GGML_FP32_TO_BF16((float) (((7 * head + 13 * key + 5 * d) % 41) - 20) * 0.015625f);
                }
            }
        }

        const std::vector<float> expected = fairy2i_exact_flash_attn_gqa_oracle(q, k, v, mask, n_queries, n_kv_long,
                                                                                n_dims, n_q_heads, n_kv_heads, scale);
        std::vector<float>       actual;
        ok = run_fairy2i_exact_flash_attn_metal(actual, dev, q, k, v, mask, n_queries, n_kv_long, n_dims, scale,
                                                n_q_heads, n_kv_heads) &&
             compare_bf16_value_bits("Fairy2i exact Flash Attention Metal long-K decode GQA 40:8", actual, expected) &&
             ok;
    }

    {
        constexpr int64_t        n_dims     = 128;
        constexpr int64_t        n_queries  = 1;
        constexpr int64_t        n_kv_long  = 24 * 1024;
        constexpr int64_t        n_q_heads  = 40;
        constexpr int64_t        n_kv_heads = 8;
        std::vector<float>       q((size_t) n_dims * n_queries * n_q_heads, 0.0f);
        std::vector<ggml_bf16_t> k((size_t) n_dims * n_kv_long * n_kv_heads, GGML_FP32_TO_BF16(0.0f));
        std::vector<ggml_bf16_t> v((size_t) n_dims * n_kv_long * n_kv_heads, GGML_FP32_TO_BF16(0.5f));
        std::vector<float>       mask((size_t) GGML_KQ_MASK_PAD * n_kv_long, 0.0f);

        const float probability    = bf16_round(1.0f / (float) n_kv_long);
        float       expected_value = 0.0f;
        for (int64_t key = 0; key < n_kv_long; ++key) {
            expected_value = fmaf(probability, 0.5f, expected_value);
        }
        expected_value = bf16_round(expected_value);
        const std::vector<float> expected((size_t) n_dims * n_q_heads, expected_value);

        std::vector<float> actual;
        ok = run_fairy2i_exact_flash_attn_metal(actual, dev, q, k, v, mask, n_queries, n_kv_long, n_dims, scale,
                                                n_q_heads, n_kv_heads) &&
             compare_bf16_value_bits("Fairy2i exact Flash Attention Metal 24K decode GQA 40:8", actual, expected) && ok;
    }

    {
        constexpr int64_t        n_dims    = 64;
        constexpr int64_t        n_queries = 1;
        std::vector<float>       q((size_t) n_queries * n_dims, 0.0f);
        std::vector<ggml_bf16_t> k((size_t) n_kv * n_dims, GGML_FP32_TO_BF16(0.0f));
        std::vector<ggml_bf16_t> v((size_t) n_kv * n_dims, GGML_FP32_TO_BF16(0.0f));
        std::vector<float>       mask((size_t) GGML_KQ_MASK_PAD * n_kv, -INFINITY);
        mask[0]   = 0.0f;
        v[0].bits = 0x0001;

        const std::vector<float> expected =
            fairy2i_exact_flash_attn_oracle(q, k, v, mask, n_queries, n_kv, n_dims, scale);
        std::vector<float> actual;
        ok = run_fairy2i_exact_flash_attn_metal(actual, dev, q, k, v, mask, n_queries, n_kv, n_dims, scale) &&
             compare_bf16_value_bits("Fairy2i exact Flash Attention subnormal P*V", actual, expected) && ok;
    }

    {
        constexpr int64_t        n_dims    = 64;
        constexpr int64_t        n_queries = 1;
        constexpr int64_t        n_kv_long = 4096;
        std::vector<float>       q((size_t) n_queries * n_dims, 0.0f);
        std::vector<ggml_bf16_t> k((size_t) n_kv_long * n_dims, GGML_FP32_TO_BF16(0.0f));
        std::vector<ggml_bf16_t> v((size_t) n_kv_long * n_dims, GGML_FP32_TO_BF16(0.0f));
        std::vector<float>       mask((size_t) GGML_KQ_MASK_PAD * n_kv_long, 0.0f);
        mask[0]   = -80.0f;
        v[0].bits = 0x7f00;

        const std::vector<float> expected =
            fairy2i_exact_flash_attn_oracle(q, k, v, mask, n_queries, n_kv_long, n_dims, scale);
        std::vector<float> actual;
        ok = run_fairy2i_exact_flash_attn_metal(actual, dev, q, k, v, mask, n_queries, n_kv_long, n_dims, scale) &&
             compare_bf16_value_bits("Fairy2i exact Flash Attention row_sum subnormal probability", actual, expected) &&
             ok;
    }

    {
        constexpr int64_t        n_dims    = 64;
        constexpr int64_t        n_queries = 1;
        std::vector<float>       q((size_t) n_queries * n_dims, bf16_from_bits(0x7f7f));
        std::vector<ggml_bf16_t> k((size_t) n_kv * n_dims, ggml_bf16_t{ 0x7f7f });
        std::vector<ggml_bf16_t> v((size_t) n_kv * n_dims, GGML_FP32_TO_BF16(0.0f));
        std::vector<float>       mask((size_t) GGML_KQ_MASK_PAD * n_kv, -INFINITY);
        mask[0] = 0.0f;
        std::fill_n(k.begin(), n_dims, GGML_FP32_TO_BF16(0.0f));
        for (int64_t d = 0; d < n_dims; ++d) {
            v[(size_t) d] = GGML_FP32_TO_BF16((float) (d % 7) * 0.125f);
        }

        const std::vector<float> expected =
            fairy2i_exact_flash_attn_oracle(q, k, v, mask, n_queries, n_kv, n_dims, scale);
        std::vector<float> actual;
        ok = run_fairy2i_exact_flash_attn_metal(actual, dev, q, k, v, mask, n_queries, n_kv, n_dims, scale) &&
             compare_bf16_value_bits("Fairy2i exact Flash Attention all-masked overflow blocks", actual, expected) &&
             ok;
    }

    printf("  Fairy2i exact Flash Attention D=40/64/128 N=1/8 + subnormal probabilities: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static std::vector<float> fairy2i_exact_cpu_attn_oracle(const std::vector<float> &       q,
                                                        const std::vector<ggml_bf16_t> & k,
                                                        const std::vector<ggml_bf16_t> & v,
                                                        const std::vector<float> &       mask,
                                                        int64_t                          n_dims,
                                                        int64_t                          n_queries,
                                                        int64_t                          n_q_heads,
                                                        int64_t                          n_kv,
                                                        int64_t                          n_kv_heads,
                                                        float                            scale) {
    std::vector<float> out((size_t) n_dims * n_queries * n_q_heads);
    std::vector<float> logits((size_t) n_kv);
    std::vector<float> probabilities((size_t) n_kv);
    const int64_t      gqa_ratio = n_q_heads / n_kv_heads;

    for (int64_t head = 0; head < n_q_heads; ++head) {
        const int64_t kv_head = head / gqa_ratio;
        for (int64_t query = 0; query < n_queries; ++query) {
            float row_max = -INFINITY;
            float row_sum = 0.0f;
            for (int64_t key = 0; key < n_kv; ++key) {
                const float mask_value = mask[(size_t) key + (size_t) n_kv * (size_t) query];
                if (mask_value == -INFINITY) {
                    logits[(size_t) key] = -INFINITY;
                    continue;
                }

                float dot = 0.0f;
                for (int64_t d = 0; d < n_dims; ++d) {
                    const size_t q_index =
                        (size_t) d + (size_t) n_dims * ((size_t) query + (size_t) n_queries * (size_t) head);
                    const size_t k_index =
                        (size_t) d + (size_t) n_dims * ((size_t) key + (size_t) n_kv * (size_t) kv_head);
                    dot = fmaf(bf16_round(q[q_index]), GGML_BF16_TO_FP32(k[k_index]), dot);
                }
                const float scaled   = bf16_round(dot * scale);
                const float logit    = bf16_round(scaled + mask_value);
                logits[(size_t) key] = logit;

                if (logit > -INFINITY) {
                    if (row_sum == 0.0f) {
                        row_max = logit;
                        row_sum = 1.0f;
                    } else {
                        const float next_max = fmaxf(row_max, logit);
                        row_sum              = row_sum * expf(row_max - next_max) + expf(logit - next_max);
                        row_max              = next_max;
                    }
                }
            }

            for (int64_t key = 0; key < n_kv; ++key) {
                probabilities[(size_t) key] = logits[(size_t) key] == -INFINITY || row_sum == 0.0f ?
                                                  0.0f :
                                                  bf16_round(expf(logits[(size_t) key] - row_max) / row_sum);
            }
            for (int64_t d = 0; d < n_dims; ++d) {
                float value = 0.0f;
                for (int64_t key = 0; key < n_kv; ++key) {
                    const size_t v_index =
                        (size_t) key + (size_t) n_kv * ((size_t) d + (size_t) n_dims * (size_t) kv_head);
                    value = fmaf(probabilities[(size_t) key], GGML_BF16_TO_FP32(v[v_index]), value);
                }
                const size_t out_index =
                    (size_t) d + (size_t) n_dims * ((size_t) query + (size_t) n_queries * (size_t) head);
                out[out_index] = bf16_round(value);
            }
        }
    }
    return out;
}

static bool test_fairy2i_exact_cpu_attention() {
    constexpr int64_t n_dims     = 128;
    constexpr int64_t n_queries  = 3;
    constexpr int64_t n_q_heads  = 4;
    constexpr int64_t n_kv       = 32;
    constexpr int64_t n_kv_heads = 2;
    constexpr float   scale      = 0.125f;

    std::vector<float>       q((size_t) n_dims * n_queries * n_q_heads);
    std::vector<ggml_bf16_t> k((size_t) n_dims * n_kv * n_kv_heads);
    std::vector<ggml_bf16_t> v((size_t) n_kv * n_dims * n_kv_heads);
    std::vector<float>       mask((size_t) n_kv * n_queries, 0.0f);
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = bf16_round((float) ((int) (13 * i % 41) - 20) * 0.0625f);
    }
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = GGML_FP32_TO_BF16((float) ((int) (7 * i % 31) - 15) * 0.0625f);
    }
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = GGML_FP32_TO_BF16((float) ((int) (11 * i % 37) - 18) * 0.03125f);
    }
    for (int64_t key = 2; key < n_kv; ++key) {
        mask[(size_t) key] = -INFINITY;
    }

    // Directed F32-accumulator crossing:
    // ((0 + 2^24) + 1) - 2^24 rounds to 0 in sequential F32, while a
    // double accumulator would retain 1. With key 1 fixed at logit 0 and
    // V(key 0)=1, V(key 1)=0, the required output is exactly BF16 0.5.
    std::fill_n(q.begin(), n_dims, 0.0f);
    q[0] = q[1] = q[2] = 1.0f;
    for (int64_t key = 0; key < 2; ++key) {
        for (int64_t d = 0; d < n_dims; ++d) {
            k[(size_t) d + (size_t) n_dims * (size_t) key] = GGML_FP32_TO_BF16(0.0f);
        }
    }
    k[0].bits = 0x4b80;
    k[1].bits = 0x3f80;
    k[2].bits = 0xcb80;
    for (int64_t key = 0; key < n_kv; ++key) {
        v[(size_t) key] = GGML_FP32_TO_BF16(0.0f);
    }
    v[0].bits = 0x3f80;

    // Keep a separate subnormal Q/K/V fixture in the second KV head.
    const size_t q_subnormal = (size_t) n_dims * (1 + n_queries * 2);
    const size_t k_subnormal = (size_t) n_dims * (n_kv);
    const size_t v_subnormal = (size_t) n_kv * (n_dims);
    q[q_subnormal]           = bf16_from_bits(0x0001);
    k[k_subnormal].bits      = 0x7f00;
    v[v_subnormal].bits      = 0x0001;

    const std::vector<float> expected =
        fairy2i_exact_cpu_attn_oracle(q, k, v, mask, n_dims, n_queries, n_q_heads, n_kv, n_kv_heads, scale);
    if (bf16_bits(expected[0]) != 0x3f00) {
        fprintf(stderr,
                "Fairy2i exact CPU Attention directed F32 accumulator fixture mismatch: "
                "actual=0x%04x expected=0x3f00\n",
                (unsigned) bf16_bits(expected[0]));
        return false;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/512 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    ggml_tensor * q_tensor    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_dims, n_queries, n_q_heads, 1);
    ggml_tensor * k_tensor    = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, n_dims, n_kv, n_kv_heads, 1);
    ggml_tensor * v_tensor    = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, n_kv, n_dims, n_kv_heads, 1);
    ggml_tensor * mask_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_queries, 1, 1);
    ggml_tensor * out_tensor  = ggml_fairy2i_attn_exact_cpu(ctx, q_tensor, k_tensor, v_tensor, mask_tensor, scale);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu || !ggml_backend_supports_op(cpu, out_tensor)) {
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_tensor);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buf) {
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(q_tensor, q.data(), 0, q.size() * sizeof(float));
    ggml_backend_tensor_set(k_tensor, k.data(), 0, k.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(v_tensor, v.data(), 0, v.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(mask_tensor, mask.data(), 0, mask.size() * sizeof(float));
    bool               ok = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    std::vector<float> actual(expected.size());
    if (ok) {
        ggml_backend_tensor_get(out_tensor, actual.data(), 0, actual.size() * sizeof(float));
        ok = compare_bf16_value_bits("Fairy2i exact CPU Attention GQA/subnormal", actual, expected);
    }

    // Every masked QK product overflows, while the sole visible key has a zero
    // K vector. Masked keys must be skipped before QK so they cannot introduce
    // +INF + -INF -> NaN into the probability or P*V stages.
    std::vector<float>       overflow_q(q.size(), bf16_from_bits(0x7f7f));
    std::vector<ggml_bf16_t> overflow_k(k.size(), ggml_bf16_t{ 0x7f7f });
    std::vector<ggml_bf16_t> overflow_v(v.size(), GGML_FP32_TO_BF16(0.0f));
    std::vector<float>       overflow_mask(mask.size(), -INFINITY);
    for (int64_t query = 0; query < n_queries; ++query) {
        overflow_mask[(size_t) query * n_kv] = 0.0f;
    }
    for (int64_t kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
        for (int64_t d = 0; d < n_dims; ++d) {
            overflow_k[(size_t) d + (size_t) n_dims * n_kv * kv_head] = GGML_FP32_TO_BF16(0.0f);
            overflow_v[(size_t) n_kv * ((size_t) d + (size_t) n_dims * kv_head)] =
                GGML_FP32_TO_BF16((float) (d % 7 + 1) * 0.125f);
        }
    }

    const std::vector<float> overflow_expected = fairy2i_exact_cpu_attn_oracle(
        overflow_q, overflow_k, overflow_v, overflow_mask, n_dims, n_queries, n_q_heads, n_kv, n_kv_heads, scale);
    if (bf16_bits(overflow_expected[0]) != 0x3e00) {
        fprintf(stderr,
                "Fairy2i exact CPU Attention masked QK overflow fixture mismatch: "
                "actual=0x%04x expected=0x3e00\n",
                (unsigned) bf16_bits(overflow_expected[0]));
        ok = false;
    }

    ggml_backend_tensor_set(q_tensor, overflow_q.data(), 0, overflow_q.size() * sizeof(float));
    ggml_backend_tensor_set(k_tensor, overflow_k.data(), 0, overflow_k.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(v_tensor, overflow_v.data(), 0, overflow_v.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(mask_tensor, overflow_mask.data(), 0, overflow_mask.size() * sizeof(float));
    const bool         overflow_compute_ok = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    std::vector<float> overflow_actual(overflow_expected.size());
    if (overflow_compute_ok) {
        ggml_backend_tensor_get(out_tensor, overflow_actual.data(), 0, overflow_actual.size() * sizeof(float));
        ok = compare_bf16_value_bits("Fairy2i exact CPU Attention masked QK overflow", overflow_actual,
                                     overflow_expected) &&
             ok;
    } else {
        ok = false;
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(cpu);
    ggml_free(ctx);

    printf("  Fairy2i exact CPU Attention F32 accumulation/GQA/masked overflow: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

struct fairy2i_exact_boundary_outputs {
    std::vector<float>       silu;
    std::vector<float>       mul;
    std::vector<float>       direct_mul;
    std::vector<ggml_bf16_t> packed;
    std::vector<uint32_t>    residual;
    std::vector<float>       logits;
};

static bool run_fairy2i_exact_boundary_backend(fairy2i_exact_boundary_outputs & out,
                                               ggml_backend_t                   backend,
                                               const std::vector<float> &       gate_data,
                                               const std::vector<float> &       up_data,
                                               const std::vector<uint32_t> &    residual_a_data,
                                               const std::vector<uint32_t> &    residual_b_data,
                                               const std::vector<uint32_t> &    logits_data) {
    struct ggml_init_params params = {
        /*.mem_size   =*/512 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * gate       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, gate_data.size());
    ggml_tensor * up         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, up_data.size());
    ggml_tensor * residual_a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, residual_a_data.size());
    ggml_tensor * residual_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, residual_b_data.size());
    ggml_tensor * logits_in  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, logits_data.size());

    ggml_tensor * silu       = ggml_fairy2i_silu_exact(ctx, gate);
    ggml_tensor * mul        = ggml_fairy2i_mul_exact(ctx, silu, up);
    ggml_tensor * direct_mul = ggml_fairy2i_mul_exact(ctx, gate, up);
    ggml_tensor * packed     = ggml_fairy2i_pack_bf16_exact(ctx, gate);
    ggml_tensor * residual   = ggml_complex_add(ctx, residual_a, residual_b);
    ggml_tensor * logits     = ggml_complex_split(ctx, logits_in);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, silu);
    ggml_build_forward_expand(gf, mul);
    ggml_build_forward_expand(gf, direct_mul);
    ggml_build_forward_expand(gf, packed);
    ggml_build_forward_expand(gf, residual);
    ggml_build_forward_expand(gf, logits);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(gate, gate_data.data(), 0, gate_data.size() * sizeof(float));
    ggml_backend_tensor_set(up, up_data.data(), 0, up_data.size() * sizeof(float));
    ggml_backend_tensor_set(residual_a, residual_a_data.data(), 0, residual_a_data.size() * sizeof(uint32_t));
    ggml_backend_tensor_set(residual_b, residual_b_data.data(), 0, residual_b_data.size() * sizeof(uint32_t));
    ggml_backend_tensor_set(logits_in, logits_data.data(), 0, logits_data.size() * sizeof(uint32_t));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status == GGML_STATUS_SUCCESS) {
        out.silu.resize(gate_data.size());
        out.mul.resize(gate_data.size());
        out.direct_mul.resize(gate_data.size());
        out.packed.resize(gate_data.size());
        out.residual.resize(residual_a_data.size());
        out.logits.resize(logits_data.size() * 2u);
        ggml_backend_tensor_get(silu, out.silu.data(), 0, out.silu.size() * sizeof(float));
        ggml_backend_tensor_get(mul, out.mul.data(), 0, out.mul.size() * sizeof(float));
        ggml_backend_tensor_get(direct_mul, out.direct_mul.data(), 0, out.direct_mul.size() * sizeof(float));
        ggml_backend_tensor_get(packed, out.packed.data(), 0, out.packed.size() * sizeof(ggml_bf16_t));
        ggml_backend_tensor_get(residual, out.residual.data(), 0, out.residual.size() * sizeof(uint32_t));
        ggml_backend_tensor_get(logits, out.logits.data(), 0, out.logits.size() * sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return status == GGML_STATUS_SUCCESS;
}

static bool test_fairy2i_exact_mlp_residual_logits() {
    const std::vector<float> gate_values = {
        0.0f,
        -0.0f,
        1.0f,
        -1.0f,
        2.0f,
        -2.0f,
        4.0f,
        -4.0f,
        0.5f,
        bf16_from_bits(0x0002),
        bf16_from_bits(0x0003),
        -90.0f,
        bf16_from_bits(0x7f7f),
    };
    const std::vector<float> up_values = {
        1.0f, 2.0f, -0.5f, 0.25f, 2.0f, -1.0f, 0.125f, -2.0f, 4.0f, bf16_from_bits(0x7f00), 1.0f, 1.0f, 2.0f,
    };
    std::vector<float>       gate;
    std::vector<float>       up;
    std::vector<float>       expected_silu;
    std::vector<float>       expected_mul;
    std::vector<float>       expected_direct_mul;
    std::vector<ggml_bf16_t> expected_packed;
    for (size_t i = 0; i < gate_values.size(); ++i) {
        const float gate_bf16 = bf16_round(gate_values[i]);
        const float up_bf16   = bf16_round(up_values[i]);
        const float exp_value = expf(gate_bf16);
        const float silu_f32 =
            signbit(gate_bf16) ? gate_bf16 * exp_value / (1.0f + exp_value) : gate_bf16 / (1.0f + expf(-gate_bf16));
        const float silu_bf16 = bf16_round(silu_f32);
        gate.push_back(gate_bf16);
        up.push_back(up_bf16);
        expected_silu.push_back(silu_bf16);
        expected_mul.push_back(bf16_round(silu_bf16 * up_bf16));
        expected_direct_mul.push_back(bf16_round(gate_bf16 * up_bf16));
        expected_packed.push_back(GGML_FP32_TO_BF16(gate_bf16));
    }

    auto compare_packed = [&](const char * label, const std::vector<ggml_bf16_t> & actual) {
        if (actual.size() != expected_packed.size()) {
            fprintf(stderr, "%s size mismatch: %zu vs %zu\n", label, actual.size(), expected_packed.size());
            return false;
        }
        for (size_t i = 0; i < actual.size(); ++i) {
            if (actual[i].bits != expected_packed[i].bits) {
                fprintf(stderr, "%s mismatch i=%zu actual=0x%04x expected=0x%04x\n", label, i,
                        (unsigned) actual[i].bits, (unsigned) expected_packed[i].bits);
                return false;
            }
        }
        return true;
    };

    const std::vector<uint32_t> residual_a = {
        pack_bf16_pair(1.0f, 1.0f + 0x1p-7f),
        pack_bf16_pair(-1.0f, -1.0f - 0x1p-7f),
        pack_bf16_pair(0.0f, -0.0f),
        pack_bf16_pair(0x1p-126f, -0x1p-126f),
    };
    const std::vector<uint32_t> residual_b = {
        pack_bf16_pair(0x1p-8f, 0x1p-8f),
        pack_bf16_pair(-0x1p-8f, -0x1p-8f),
        pack_bf16_pair(-0.0f, 0.0f),
        pack_bf16_pair(0x1p-133f, -0x1p-133f),
    };
    std::vector<uint32_t> expected_residual(residual_a.size());
    for (size_t i = 0; i < residual_a.size(); ++i) {
        float ar;
        float ai;
        float br;
        float bi;
        unpack_bf16_pair(residual_a[i], ar, ai);
        unpack_bf16_pair(residual_b[i], br, bi);
        expected_residual[i] = pack_bf16_pair(ar + br, ai + bi);
    }

    const std::vector<uint32_t> logits_in = {
        pack_bf16_pair(1.0f, -1.0f),
        pack_bf16_pair(0.0f, -0.0f),
        pack_bf16_pair(0x1.fep127f, -0x1.fep127f),
        pack_bf16_pair(0x1p-133f, -0x1p-133f),
    };
    std::vector<float> expected_logits(logits_in.size() * 2u);
    for (size_t i = 0; i < logits_in.size(); ++i) {
        unpack_bf16_pair(logits_in[i], expected_logits[i], expected_logits[i + logits_in.size()]);
    }

    bool           ok  = true;
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        return false;
    }
    fairy2i_exact_boundary_outputs cpu_out;
    ok = run_fairy2i_exact_boundary_backend(cpu_out, cpu, gate, up, residual_a, residual_b, logits_in) &&
         compare_bf16_value_bits("Fairy2i exact SiLU CPU", cpu_out.silu, expected_silu) &&
         compare_bf16_value_bits("Fairy2i exact MLP multiply CPU", cpu_out.mul, expected_mul) &&
         compare_bf16_value_bits("Fairy2i exact direct multiply CPU", cpu_out.direct_mul, expected_direct_mul) &&
         compare_packed("Fairy2i exact BF16 pack CPU", cpu_out.packed) &&
         compare_exact("Fairy2i exact residual CPU", cpu_out.residual, expected_residual) &&
         compare_bf16_value_bits("Fairy2i exact logits widen CPU", cpu_out.logits, expected_logits) && ok;
    ggml_backend_free(cpu);

    ggml_backend_dev_t metal_dev = find_metal_test_device();
    if (metal_dev) {
        ggml_backend_t                 metal = ggml_backend_dev_init(metal_dev, nullptr);
        fairy2i_exact_boundary_outputs metal_out;
        ok =
            metal &&
            run_fairy2i_exact_boundary_backend(metal_out, metal, gate, up, residual_a, residual_b, logits_in) &&
            compare_bf16_value_bits("Fairy2i exact SiLU Metal", metal_out.silu, expected_silu) &&
            compare_bf16_value_bits("Fairy2i exact MLP multiply Metal", metal_out.mul, expected_mul) &&
            compare_bf16_value_bits("Fairy2i exact direct multiply Metal", metal_out.direct_mul, expected_direct_mul) &&
            compare_packed("Fairy2i exact BF16 pack Metal", metal_out.packed) &&
            compare_exact("Fairy2i exact residual Metal", metal_out.residual, expected_residual) &&
            compare_bf16_value_bits("Fairy2i exact logits widen Metal", metal_out.logits, expected_logits) && ok;
        if (metal) {
            ggml_backend_free(metal);
        }
    }

    printf("  Fairy2i exact MLP/residual/logits CPU%s: %s\n", metal_dev ? "+Metal" : "", ok ? "PASS" : "FAIL");
    return ok;
}

static bool run_fairy2i_bf16_set_rows_backend(const char * label, ggml_backend_t backend) {
    constexpr int64_t n_dims = 5;
    constexpr int64_t n_rows = 3;

    struct ggml_init_params params = {
        /*.mem_size   =*/256 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_dims, n_rows);
    ggml_tensor * src   = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_dims, 1);
    ggml_tensor * index = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
    ggml_tensor * out   = ggml_set_rows(ctx, cache, src, index);
    if (!ggml_backend_supports_op(backend, out)) {
        ggml_free(ctx);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }

    std::vector<ggml_bf16_t> cache_data((size_t) n_dims * n_rows);
    for (size_t i = 0; i < cache_data.size(); ++i) {
        cache_data[i].bits = (uint16_t) (0x3f80U + i);
    }
    std::vector<ggml_bf16_t> src_data((size_t) n_dims);
    const uint16_t           payloads[n_dims] = { 0x0001, 0x8001, 0x0000, 0x8000, 0x7fc1 };
    for (int64_t i = 0; i < n_dims; ++i) {
        src_data[(size_t) i].bits = payloads[i];
    }
    const int64_t row_index = 2;

    ggml_backend_tensor_set(cache, cache_data.data(), 0, cache_data.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(index, &row_index, 0, sizeof(row_index));

    bool                     ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    std::vector<ggml_bf16_t> actual(cache_data.size());
    if (ok) {
        ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(ggml_bf16_t));
        for (int64_t row = 0; row < n_rows && ok; ++row) {
            for (int64_t i = 0; i < n_dims; ++i) {
                const uint16_t expected =
                    row == row_index ? payloads[i] : cache_data[(size_t) row * n_dims + (size_t) i].bits;
                const uint16_t got = actual[(size_t) row * n_dims + (size_t) i].bits;
                if (got != expected) {
                    fprintf(stderr, "%s mismatch row=%lld i=%lld actual=0x%04x expected=0x%04x\n", label,
                            (long long) row, (long long) i, (unsigned) got, (unsigned) expected);
                    ok = false;
                    break;
                }
            }
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

static bool test_fairy2i_exact_bf16_set_rows() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        return false;
    }
    bool ok = run_fairy2i_bf16_set_rows_backend("Fairy2i exact BF16 SET_ROWS CPU", cpu);
    ggml_backend_free(cpu);

    ggml_backend_dev_t metal_dev = find_metal_test_device();
    if (metal_dev) {
        ggml_backend_t metal = ggml_backend_dev_init(metal_dev, nullptr);
        ok = metal && run_fairy2i_bf16_set_rows_backend("Fairy2i exact BF16 SET_ROWS Metal", metal) && ok;
        if (metal) {
            ggml_backend_free(metal);
        }
    }

    printf("  Fairy2i exact BF16 SET_ROWS raw payload CPU%s: %s\n", metal_dev ? "+Metal" : "", ok ? "PASS" : "FAIL");
    return ok;
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
    ggml_tensor * scales = ggml_new_tensor_3d(ctx, bundle_data.scale_type, 2, bundle_data.branches, physical_tiles);
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
    if (bundle_data.scale_type == GGML_TYPE_BF16) {
        ggml_backend_tensor_set(scales, bundle_data.scales_bf16.data(), 0,
                                bundle_data.scales_bf16.size() * sizeof(ggml_bf16_t));
    } else {
        ggml_backend_tensor_set(scales, bundle_data.scales.data(), 0, bundle_data.scales.size() * sizeof(ggml_fp16_t));
    }
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

static bool compare_packed_complex_nmse(const char *                  label,
                                        const std::vector<uint32_t> & actual,
                                        const std::vector<uint32_t> & expected,
                                        double                        max_nmse,
                                        double                        max_error) {
    if (actual.size() != expected.size()) {
        fprintf(stderr, "%s size mismatch: %zu vs %zu\n", label, actual.size(), expected.size());
        return false;
    }

    double squared_error    = 0.0;
    double reference_energy = 0.0;
    double max_diff         = 0.0;
    size_t max_idx          = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        float ar, ai, er, ei;
        unpack_bf16_pair(actual[i], ar, ai);
        unpack_bf16_pair(expected[i], er, ei);
        const double dr = (double) ar - (double) er;
        const double di = (double) ai - (double) ei;
        squared_error += dr * dr + di * di;
        reference_energy += (double) er * (double) er + (double) ei * (double) ei;
        if (fabs(dr) > max_diff) {
            max_diff = fabs(dr);
            max_idx  = i;
        }
        if (fabs(di) > max_diff) {
            max_diff = fabs(di);
            max_idx  = i;
        }
    }

    const double nmse = reference_energy > 0.0 ? squared_error / reference_energy : squared_error;
    if (nmse > max_nmse || max_diff > max_error) {
        float ar, ai, er, ei;
        unpack_bf16_pair(actual[max_idx], ar, ai);
        unpack_bf16_pair(expected[max_idx], er, ei);
        fprintf(stderr,
                "%s mismatch: NMSE=%.9g (max %.9g), max_abs=%.9g (max %.9g), index=%zu "
                "actual=(%.7g,%.7g) expected=(%.7g,%.7g)\n",
                label, nmse, max_nmse, max_diff, max_error, max_idx, ar, ai, er, ei);
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
    ggml_tensor * exact_scales = ggml_new_tensor_3d(ctx, GGML_TYPE_BF16, 2, 4, 2);
    ggml_tensor * exact_w2 =
        ggml_fairy2i_wide_linear_w2_bundle(ctx, bundle_x, bundle_codes, exact_scales, nullptr, 64, 128);
    ggml_tensor * padded_x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 129, 2);
    ggml_tensor * strided_x = ggml_view_2d(ctx, padded_x, 128, 2, 129 * sizeof(float), 0);
    ggml_tensor * strided_exact_w2 =
        ggml_fairy2i_wide_linear_w2_bundle(ctx, strided_x, bundle_codes, exact_scales, nullptr, 64, 128);

    bool               ok      = true;
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        fprintf(stderr, "CPU backend device not found for Bundle generic MUL_MAT support tests\n");
        ok = false;
    } else if (ggml_backend_dev_supports_op(cpu_dev, mm) || ggml_backend_dev_supports_op(cpu_dev, mm_id)) {
        fprintf(stderr, "CPU backend must reject Bundle codes in generic MUL_MAT and MUL_MAT_ID\n");
        ok = false;
    } else if (!ggml_backend_dev_supports_op(cpu_dev, exact_w2)) {
        fprintf(stderr, "CPU backend must support contiguous exact Bundle W2\n");
        ok = false;
    } else if (ggml_backend_dev_supports_op(cpu_dev, strided_exact_w2)) {
        fprintf(stderr, "CPU backend must reject exact Bundle W2 with a noncontiguous activation view\n");
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
        is_bundle ? ggml_new_tensor_3d(ctx, bundle_data->scale_type, 2, bundle_data->branches, physical_tiles) :
                    nullptr;
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
        if (bundle_data->scale_type == GGML_TYPE_BF16) {
            ggml_backend_tensor_set(scales, bundle_data->scales_bf16.data(), 0,
                                    bundle_data->scales_bf16.size() * sizeof(ggml_bf16_t));
        } else {
            ggml_backend_tensor_set(scales, bundle_data->scales.data(), 0,
                                    bundle_data->scales.size() * sizeof(ggml_fp16_t));
        }
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

static bool test_fairy2i_exact_w2_packed_bits() {
    printf("\n=== Fairy2i exact W2 packed BF16 bits ===\n");

    scoped_env_var strict_staged_reconstruction("GGML_FAIRY2I_STRICT_STAGED_RECONSTRUCTION");
    strict_staged_reconstruction.unset();

    struct exact_case {
        int64_t N;
        bool    bias;
        int     subnormal;
    };

    const exact_case cases[] = {
        { 1,  false, false },
        { 1,  true,  false },
        { 4,  false, false },
        { 4,  true,  false },
        { 1,  false, true  },
        { 4,  false, true  },
        { 1,  true,  true  },
        { 4,  true,  true  },
        { 1,  false, 2     },
        { 4,  false, 2     },
        { 1,  false, 3     },
        { 4,  false, 3     },
        { 16, false, 2     },
        { 16, false, 3     },
        { 16, true,  true  },
    };

    const uint16_t base_bits[4][2] = {
        { 0x3f83, 0x0000 },
        { 0x0000, 0x3f81 },
        { 0x0000, 0x3f01 },
        { 0xbefe, 0x0000 },
    };
    const uint16_t biased_bits[4][2] = {
        { 0x3fa3, 0xbe00 },
        { 0x3e80, 0x3f62 },
        { 0x3e80, 0x3ec2 },
        { 0xbe7c, 0xbe00 },
    };

    bool               ok        = true;
    int                cases_run = 0;
    ggml_backend_dev_t metal_dev = find_metal_test_device();

    {
        const fairy2i_w2_case tc     = { 64, 1, 64, false };
        fairy2i_bundle_data   bundle = make_fairy2i_exact_empty_bundle(tc.M, tc.K);
        bundle.scales_bf16[0].bits   = 0x3f81;
        bundle.scales_bf16[2].bits   = 0x3b80;
        bundle.scales_bf16[4].bits   = 0x3b80;
        for (int64_t row = 0; row < tc.M; ++row) {
            set_fairy2i_bundle_code(bundle, tc.M, tc.K, row, 0, 0, 1);
            set_fairy2i_bundle_code(bundle, tc.M, tc.K, row, 0, 1, 1);
            set_fairy2i_bundle_code(bundle, tc.M, tc.K, row, 0, 2, 0);
            set_fairy2i_bundle_code(bundle, tc.M, tc.K, row, 0, 3, 1);
        }

        std::vector<float> input((size_t) tc.K, pack_complex_bf16(0.0f, 0.0f));
        input[0] = pack_complex_bf16(1.0f, 0.0f);
        const std::vector<float>    bias;
        const std::vector<uint32_t> expected((size_t) tc.M, UINT32_C(0x00003f82));
        std::vector<uint32_t>       cpu;
        if (!run_fairy2i_bundle_backend(cpu, tc, input, bias, bundle, false, "0", "lut16", false, 4) ||
            !compare_exact("Fairy2i exact W2 staged BF16 tie CPU", cpu, expected)) {
            ok = false;
        }

        if (metal_dev) {
            const std::vector<block_fairy2i_tile64_v2> unused;
            std::vector<uint32_t>                      metal;
            if (!run_fairy2i_metal_backend(metal, metal_dev, tc, input, unused, &unused, unused, &unused, bias,
                                           &bundle) ||
                !compare_exact("Fairy2i exact W2 staged BF16 tie Metal", metal, expected)) {
                ok = false;
            }
        }
        ++cases_run;
    }

    for (const exact_case & exact_tc : cases) {
        const fairy2i_w2_case     tc     = { 64, exact_tc.N, 64, exact_tc.bias };
        const fairy2i_bundle_data bundle = exact_tc.subnormal >= 2 ?
                                               make_fairy2i_exact_pure_subnormal_bundle(tc.M, tc.K) :
                                           exact_tc.subnormal == 1 ? make_fairy2i_exact_subnormal_bundle(tc.M, tc.K) :
                                                                     make_fairy2i_exact_bundle(tc.M, tc.K);

        std::vector<float> input((size_t) tc.N * (size_t) tc.K, pack_complex_bf16(0.0f, 0.0f));
        if (exact_tc.subnormal) {
            for (int64_t n = 0; n < tc.N; ++n) {
                input[(size_t) n * (size_t) tc.K] =
                    exact_tc.subnormal == 3 ? pack_complex_bf16(0x1p127f, 0.0f) : pack_complex_bf16(1.0f, 0.0f);
            }
        } else {
            input[0] = pack_complex_bf16(1.0f, 0.0f);
        }
        if (!exact_tc.subnormal && tc.N == 4) {
            input[(size_t) tc.K + 0]     = pack_complex_bf16(0.0f, 1.0f);
            input[(size_t) 2 * tc.K + 1] = pack_complex_bf16(1.0f, 0.0f);
            input[(size_t) 3 * tc.K + 1] = pack_complex_bf16(0.0f, 1.0f);
        }

        std::vector<float> bias;
        if (tc.bias) {
            bias.resize((size_t) 2 * (size_t) tc.M);
            std::fill(bias.begin(), bias.begin() + tc.M, exact_tc.subnormal ? 0x1p-133f : 0.25f);
            std::fill(bias.begin() + tc.M, bias.end(), exact_tc.subnormal ? -0x1p-133f : -0.125f);
        }

        std::vector<uint32_t> expected((size_t) tc.M * (size_t) tc.N);
        for (int64_t n = 0; n < tc.N; ++n) {
            if (exact_tc.subnormal) {
                const uint32_t pair = exact_tc.subnormal == 2 ? UINT32_C(0x00000001) :
                                      exact_tc.subnormal == 3 ? UINT32_C(0x00003c80) :
                                      tc.bias                 ? UINT32_C(0x80010082) :
                                                                UINT32_C(0x00000081);
                std::fill(expected.begin() + (size_t) n * (size_t) tc.M,
                          expected.begin() + (size_t) (n + 1) * (size_t) tc.M, pair);
                continue;
            }
            const uint16_t (*bits)[2] = tc.bias ? biased_bits : base_bits;
            const uint32_t pair       = (uint32_t) bits[n][0] | ((uint32_t) bits[n][1] << 16);
            std::fill(expected.begin() + (size_t) n * (size_t) tc.M,
                      expected.begin() + (size_t) (n + 1) * (size_t) tc.M, pair);
        }

        std::vector<uint32_t> cpu;
        if (!run_fairy2i_bundle_backend(cpu, tc, input, bias, bundle, false, "0", "lut16", false, 4) ||
            !compare_exact("Fairy2i exact W2 CPU packed bits", cpu, expected)) {
            ok = false;
        }

        if (metal_dev) {
            const std::vector<block_fairy2i_tile64_v2> unused;
            std::vector<uint32_t>                      metal;
            if (!run_fairy2i_metal_backend(metal, metal_dev, tc, input, unused, &unused, unused, &unused, bias,
                                           &bundle) ||
                !compare_exact("Fairy2i exact W2 Metal packed bits", metal, expected)) {
                ok = false;
            }
        }
        ++cases_run;
    }

    auto run_m64_bit_exact_case = [&](const char * label, const fairy2i_w2_case & tc, const std::vector<float> & input,
                                      const std::vector<float> & bias, const fairy2i_bundle_data & bundle,
                                      std::vector<uint32_t> * cpu_out = nullptr) {
        std::vector<uint32_t> cpu;
        bool case_ok = run_fairy2i_bundle_backend(cpu, tc, input, bias, bundle, false, "0", "lut16", false, 4);
        if (cpu_out) {
            *cpu_out = cpu;
        }
        if (case_ok && metal_dev) {
            const std::vector<block_fairy2i_tile64_v2> unused;
            std::vector<uint32_t>                      metal;
            case_ok = run_fairy2i_metal_backend(metal, metal_dev, tc, input, unused, &unused, unused, &unused, bias,
                                                &bundle) &&
                      compare_exact(label, metal, cpu);
        }
        ++cases_run;
        return case_ok;
    };

    struct m64_pattern_case {
        int64_t M;
        int64_t K;
        int64_t active_k;
        int     pattern_base;
    };

    const m64_pattern_case pattern_cases[] = {
        { 192, 64,  15, 0   },
        { 64,  128, 64, 192 },
    };
    for (const m64_pattern_case & pattern_tc : pattern_cases) {
        const fairy2i_w2_case tc        = { pattern_tc.M, 1, pattern_tc.K, false };
        fairy2i_bundle_data   bundle    = make_fairy2i_exact_empty_bundle(tc.M, tc.K);
        static const float    scales[8] = {
            1.0f + 0x1p-8f, 0.75f, 0.5f + 3.0f * 0x1p-9f, 0.375f, 0.25f, 0.1875f, 0.125f, 0.09375f,
        };
        for (size_t tile = 0; tile < bundle.scales_bf16.size() / 8u; ++tile) {
            set_fairy2i_exact_tile_scales(bundle, tile, scales);
        }
        for (int64_t row = 0; row < tc.M; ++row) {
            const unsigned pattern = (unsigned) (pattern_tc.pattern_base + row);
            for (int branch = 0; branch < 4; ++branch) {
                set_fairy2i_bundle_code(bundle, tc.M, tc.K, row, pattern_tc.active_k, branch,
                                        (uint8_t) ((pattern >> (2 * branch)) & 3u));
            }
        }

        std::vector<float> input((size_t) tc.K, pack_complex_bf16(0.0f, 0.0f));
        input[(size_t) pattern_tc.active_k] = pack_complex_bf16(1.0f, -0.5f);
        char label[160];
        snprintf(label, sizeof(label), "Fairy2i exact M64 pattern map M=%lld K=%lld active_k=%lld", (long long) tc.M,
                 (long long) tc.K, (long long) pattern_tc.active_k);
        ok = run_m64_bit_exact_case(label, tc, input, {}, bundle) && ok;
    }

    for (bool with_bias : { false, true }) {
        // Exercise one complete four-K64 group plus a fifth-block tail. No-bias selects
        // the function-constant kernel while bias uses the dynamic-argument variant.
        const fairy2i_w2_case tc        = { 64, 1, 320, with_bias };
        fairy2i_bundle_data   bundle    = make_fairy2i_exact_empty_bundle(tc.M, tc.K);
        static const float    scales[8] = {
            1.0f + 0x1p-8f, 0.75f, 0.5f + 3.0f * 0x1p-9f, 0.375f, 0.25f, 0.1875f, 0.125f, 0.09375f,
        };
        for (size_t tile = 0; tile < bundle.scales_bf16.size() / 8u; ++tile) {
            set_fairy2i_exact_tile_scales(bundle, tile, scales);
        }

        const int64_t active_k[] = { 0, 63, 64, 127, 128, 191, 192, 255, 256, 319 };
        const size_t  n_active   = sizeof(active_k) / sizeof(active_k[0]);
        for (int64_t row = 0; row < tc.M; ++row) {
            for (size_t i = 0; i < n_active; ++i) {
                const uint8_t code = (uint8_t) ((row + (int64_t) i) & 3);
                set_fairy2i_bundle_code(bundle, tc.M, tc.K, row, active_k[i], (int) (i & 3u), code);
            }
        }

        std::vector<float> input((size_t) tc.K, pack_complex_bf16(0.0f, 0.0f));
        for (size_t i = 0; i < n_active; ++i) {
            input[(size_t) active_k[i]] =
                pack_complex_bf16((i & 1u) == 0 ? 1.0f : -0.5f, (i & 2u) == 0 ? 0.25f : -0.75f);
        }
        std::vector<float> bias;
        if (with_bias) {
            bias.resize((size_t) 2 * (size_t) tc.M);
            std::fill(bias.begin(), bias.begin() + tc.M, 0x1.008p-8f);
            std::fill(bias.begin() + tc.M, bias.end(), -0x1.008p-9f);
        }

        char label[160];
        snprintf(label, sizeof(label), "Fairy2i exact M64 grouped K64 tail K=320 bias=%d", (int) with_bias);
        ok = run_m64_bit_exact_case(label, tc, input, bias, bundle) && ok;
    }

    {
        const fairy2i_w2_case tc         = { 128, 1, 256, true };
        fairy2i_bundle_data   bundle     = make_fairy2i_exact_empty_bundle(tc.M, tc.K);
        const int64_t         active_k[] = { 15, 16, 31, 32, 47, 48, 63, 64, 127, 128, 191, 192, 255 };
        const size_t          n_active   = sizeof(active_k) / sizeof(active_k[0]);

        for (size_t tile = 0; tile < bundle.scales_bf16.size() / 8u; ++tile) {
            set_fairy2i_exact_scale(bundle, tile * 8u, 1.0f + (float) tile * 0.125f);
        }
        for (int64_t row = 0; row < tc.M; ++row) {
            for (size_t i = 0; i < n_active; ++i) {
                set_fairy2i_bundle_code(bundle, tc.M, tc.K, row, active_k[i], 0, 2);
            }
            set_fairy2i_bundle_code(bundle, tc.M, tc.K, row, active_k[(size_t) row % n_active], 0, 1);
        }

        std::vector<float> input((size_t) tc.K, pack_complex_bf16(0.0f, 0.0f));
        for (size_t i = 0; i < n_active; ++i) {
            input[(size_t) active_k[i]] = pack_complex_bf16(1.0f, 0.0f);
        }
        std::vector<float> bias((size_t) 2 * (size_t) tc.M, 0.0f);
        bias[0] = 0x1.008p-8f;

        std::vector<uint32_t> cpu;
        const bool            case_ok =
            run_m64_bit_exact_case("Fairy2i exact M64 Kseg/physical-tile/full-F32-bias", tc, input, bias, bundle, &cpu);
        if (!case_ok || cpu.empty() || (cpu[0] & 0xffffU) != 0x3f81U) {
            fprintf(stderr, "Fairy2i exact M64 full-F32 bias oracle mismatch: got=0x%04x expected=0x3f81\n",
                    cpu.empty() ? 0U : cpu[0] & 0xffffU);
            ok = false;
        }
    }

    {
        // The only subnormal coefficient is in an unused LUT pattern. The conservative physical-tile
        // metric must still route the complete M64 group through logical-K accumulation: native K16
        // partial reduction would produce +1, while the canonical sequence rounds to +0.
        const fairy2i_w2_case tc     = { 64, 1, 320, false };
        fairy2i_bundle_data   bundle = make_fairy2i_exact_empty_bundle(tc.M, tc.K);
        for (size_t tile = 0; tile < bundle.scales_bf16.size() / 8u; ++tile) {
            set_fairy2i_exact_scale(bundle, tile * 8u + 0u, 1.0f);
            set_fairy2i_exact_scale(bundle, tile * 8u + 1u, 0x1p-133f);
        }
        for (int64_t row = 0; row < tc.M; ++row) {
            set_fairy2i_bundle_code(bundle, tc.M, tc.K, row, 63, 0, 1);
            set_fairy2i_bundle_code(bundle, tc.M, tc.K, row, 64, 0, 1);
        }

        std::vector<float> input((size_t) tc.K, pack_complex_bf16(0.0f, 0.0f));
        input[63] = pack_complex_bf16(0x1p24f, 0.0f);
        input[64] = pack_complex_bf16(1.0f, 0.0f);
        input[80] = pack_complex_bf16(0x1p24f, 0.0f);

        std::vector<uint32_t> cpu;
        const bool case_ok = run_m64_bit_exact_case("Fairy2i exact M64 unused-subnormal-LUT conservative fallback", tc,
                                                    input, {}, bundle, &cpu);
        if (!case_ok || cpu.size() != (size_t) tc.M) {
            ok = false;
        }
        for (int64_t row = 0; row < tc.M && cpu.size() == (size_t) tc.M; ++row) {
            if (cpu[(size_t) row] != 0U) {
                fprintf(
                    stderr,
                    "Fairy2i exact M64 unused-danger-pattern oracle mismatch row=%lld got=0x%08x expected=0x00000000\n",
                    (long long) row, cpu[(size_t) row]);
                ok = false;
                break;
            }
        }
    }

    {
        // A minimum BF16 subnormal activation has metric zero even against a normal coefficient.
        // Metal must use the bit-domain fallback whether danger appears in the first group and must
        // survive a safe fifth-block tail, or appears only in that tail.
        const int64_t active_ks[] = { 0, 256 };
        for (const int64_t active_k : active_ks) {
            for (const bool with_bias : { false, true }) {
                const fairy2i_w2_case tc     = { 64, 1, 320, with_bias };
                fairy2i_bundle_data   bundle = make_fairy2i_exact_empty_bundle(tc.M, tc.K);
                set_fairy2i_exact_scale(bundle, (size_t) (active_k / 64) * 8u, 1.0f);
                for (int64_t row = 0; row < tc.M; ++row) {
                    set_fairy2i_bundle_code(bundle, tc.M, tc.K, row, active_k, 0, 1);
                }

                std::vector<float> input((size_t) tc.K, pack_complex_bf16(0.0f, 0.0f));
                input[(size_t) active_k] = pack_complex_bf16(0x1p-133f, 0.0f);
                std::vector<float> bias;
                if (with_bias) {
                    bias.resize((size_t) 2 * (size_t) tc.M, 0.0f);
                }

                char label[160];
                snprintf(label, sizeof(label), "Fairy2i exact M64 subnormal-activation fallback active_k=%lld bias=%d",
                         (long long) active_k, (int) with_bias);
                std::vector<uint32_t> cpu;
                const bool            case_ok = run_m64_bit_exact_case(label, tc, input, bias, bundle, &cpu);
                if (!case_ok || cpu.size() != (size_t) tc.M) {
                    ok = false;
                }
                for (int64_t row = 0; row < tc.M && cpu.size() == (size_t) tc.M; ++row) {
                    if (cpu[(size_t) row] != UINT32_C(0x00000001)) {
                        fprintf(stderr,
                                "Fairy2i exact M64 subnormal-activation oracle mismatch active_k=%lld bias=%d "
                                "row=%lld got=0x%08x expected=0x00000001\n",
                                (long long) active_k, (int) with_bias, (long long) row, cpu[(size_t) row]);
                        ok = false;
                        break;
                    }
                }
            }
        }
    }

    {
        const fairy2i_w2_case tc     = { 64, 1, 320, false };
        fairy2i_bundle_data   bundle = make_fairy2i_exact_empty_bundle(tc.M, tc.K);
        for (size_t tile = 0; tile < bundle.scales_bf16.size() / 8u; ++tile) {
            set_fairy2i_exact_scale(bundle, tile * 8u + 0u, 1.0f);
            set_fairy2i_exact_scale(bundle, tile * 8u + 1u, 0x1p-133f);
            set_fairy2i_exact_scale(bundle, tile * 8u + 3u, 0x1p-133f);
        }

        for (int64_t row = 0; row < tc.M; ++row) {
            set_fairy2i_bundle_code(bundle, tc.M, tc.K, row, 63, 0, 1);
            set_fairy2i_bundle_code(bundle, tc.M, tc.K, row, 64, 0, 1);
            set_fairy2i_bundle_code(bundle, tc.M, tc.K, row, 80, 0, row == 63 ? 0 : 1);
            set_fairy2i_bundle_code(bundle, tc.M, tc.K, row, 96, 0, 3);
            set_fairy2i_bundle_code(bundle, tc.M, tc.K, row, 96, 1, row == 63 ? 3 : 2);
        }

        std::vector<float> input((size_t) tc.K, pack_complex_bf16(0.0f, 0.0f));
        input[63] = pack_complex_bf16(0x1p24f, 0.0f);
        input[64] = pack_complex_bf16(1.0f, 0.0f);
        input[80] = pack_complex_bf16(0x1p24f, 0.0f);
        input[96] = pack_complex_bf16(0.0f, -1.0f);

        std::vector<uint32_t> cpu;
        const bool            case_ok =
            run_m64_bit_exact_case("Fairy2i exact M64 single-danger-row fallback", tc, input, {}, bundle, &cpu);
        if (!case_ok || cpu.size() != (size_t) tc.M || cpu[63] != UINT32_C(0x00000002)) {
            fprintf(stderr, "Fairy2i exact M64 dangerous-row oracle mismatch: got=0x%08x expected=0x00000002\n",
                    cpu.size() == (size_t) tc.M ? cpu[63] : 0U);
            ok = false;
        }
        for (int64_t row = 0; row < 63 && cpu.size() == (size_t) tc.M; ++row) {
            if (cpu[(size_t) row] != UINT32_C(0x00004c00)) {
                fprintf(stderr, "Fairy2i exact M64 safe-row oracle mismatch row=%lld got=0x%08x expected=0x00004c00\n",
                        (long long) row, cpu[(size_t) row]);
                ok = false;
                break;
            }
        }
    }

    struct exact_prefill_case {
        int64_t M;
        int64_t N;
        int64_t K;
    };

    const exact_prefill_case exact_prefill_cases[] = {
        { 128, 1,  128  },
        { 128, 3,  128  },
        { 128, 16, 128  },
        { 128, 17, 128  },
        { 128, 31, 128  },
        { 128, 32, 128  },
        { 128, 33, 128  },
        { 128, 63, 128  },
        { 128, 64, 128  },
        { 128, 65, 128  },
        { 64,  32, 64   },
        { 192, 33, 192  },
        { 128, 17, 320  },
        { 64,  3,  1024 },
    };
    for (const exact_prefill_case & prefill_tc : exact_prefill_cases) {
        for (bool with_bias : { false, true }) {
            const fairy2i_w2_case tc     = { prefill_tc.M, prefill_tc.N, prefill_tc.K, with_bias };
            fairy2i_bundle_data   bundle = make_fairy2i_exact_mixed_bundle(tc.M, tc.K);
            std::vector<float>    input((size_t) tc.N * (size_t) tc.K);
            for (int64_t col = 0; col < tc.N; ++col) {
                for (int64_t k = 0; k < tc.K; ++k) {
                    const int   real_mag = 1 + 2 * (int) ((5 * k + 3 * col) & 7);
                    const int   imag_mag = 1 + 2 * (int) ((7 * k + 5 * col + 1) & 7);
                    const float real     = ((k + col) & 1 ? -real_mag : real_mag) * (1.0f / 128.0f);
                    const float imag     = ((k + 2 * col) & 2 ? -imag_mag : imag_mag) * (1.0f / 128.0f);
                    input[(size_t) col * (size_t) tc.K + (size_t) k] = pack_complex_bf16(real, imag);
                }
            }
            if (tc.K > 192) {
                const int64_t blocks       = tc.K / 64;
                const int64_t danger_block = tc.K == 320 ? 4 : blocks - 1;
                for (int64_t row_tile = 0; row_tile < tc.M / 64; ++row_tile) {
                    const size_t scale_base = (size_t) (row_tile * blocks + danger_block) * 8u;
                    set_fairy2i_exact_scale(bundle, scale_base, 0x1p-133f);
                }
                input[(size_t) danger_block * 64u + 63u] = pack_complex_bf16(bf16_from_bits(0x0001), 0.0f);
            }

            std::vector<float> bias;
            if (with_bias) {
                bias.resize((size_t) 2 * (size_t) tc.M);
                for (int64_t row = 0; row < tc.M; ++row) {
                    bias[(size_t) row]                 = (float) ((row % 7) - 3) * (1.0f / 256.0f);
                    bias[(size_t) tc.M + (size_t) row] = (float) ((row % 5) - 2) * (1.0f / 256.0f);
                }
            }

            std::vector<uint32_t> cpu;
            if (!run_fairy2i_bundle_backend(cpu, tc, input, bias, bundle, false, "0", "lut16", false, 4)) {
                ok = false;
                continue;
            }

            if (metal_dev) {
                const std::vector<block_fairy2i_tile64_v2> unused;
                std::vector<uint32_t>                      metal;
                char                                       label[160];
                snprintf(label, sizeof(label), "Fairy2i exact W2 mixed LUT M=%lld N=%lld K=%lld bias=%d",
                         (long long) tc.M, (long long) tc.N, (long long) tc.K, (int) tc.bias);
                if (!run_fairy2i_metal_backend(metal, metal_dev, tc, input, unused, &unused, unused, &unused, bias,
                                               &bundle) ||
                    !compare_packed_complex_nmse(label, metal, cpu, 1e-6, 1e-2)) {
                    ok = false;
                }
            }
            ++cases_run;
        }
    }

    if (!metal_dev) {
        printf("  Metal backend not found; exact W2 packed-bit Metal checks skipped.\n");
    }
    printf("  Fairy2i exact W2 packed bits: %d CPU%s cases - %s\n", cases_run, metal_dev ? "+Metal" : "",
           ok ? "PASS" : "FAIL");
    return ok;
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

    const char * require_metal = getenv("LLAMA_FAIRY2I_REQUIRE_METAL_TESTS");
    if (require_metal && strcmp(require_metal, "0") != 0 && !find_metal_test_device()) {
        fprintf(stderr,
                "LLAMA_FAIRY2I_REQUIRE_METAL_TESTS is set, but the Metal backend/device (including runtime "
                "library compilation) is unavailable\n");
        return 1;
    }

    printf("========================================\n");
    printf("Fairy2i CPU Unit Tests\n");
    printf("========================================\n");

    int num_failed = 0;
    if (!test_fairy2i_bf16_rne_bits()) {
        fprintf(stderr, "Fairy2i BF16 RNE bit tests FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_exact_coefficient_metric_bound()) {
        fprintf(stderr, "Fairy2i exact coefficient metric bound tests FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_exact_coefficient_lut_patterns()) {
        fprintf(stderr, "Fairy2i exact coefficient LUT tests FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_exact_branch_code_transpose()) {
        fprintf(stderr, "Fairy2i exact branch-code SWAR transpose tests FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_exact_coefficient_stage_bits()) {
        fprintf(stderr, "Fairy2i exact coefficient stage tests FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_exact_rms_norm()) {
        fprintf(stderr, "Fairy2i exact RMSNorm FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_exact_rope()) {
        fprintf(stderr, "Fairy2i exact RoPE FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_exact_flash_attn()) {
        fprintf(stderr, "Fairy2i exact Flash Attention FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_exact_cpu_attention()) {
        fprintf(stderr, "Fairy2i exact CPU Attention FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_exact_mlp_residual_logits()) {
        fprintf(stderr, "Fairy2i exact MLP/residual/logits FAILED\n");
        ++num_failed;
    }
    if (!test_fairy2i_exact_bf16_set_rows()) {
        fprintf(stderr, "Fairy2i exact BF16 SET_ROWS FAILED\n");
        ++num_failed;
    }
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
    if (!test_fairy2i_exact_w2_packed_bits()) {
        fprintf(stderr, "Fairy2i exact W2 packed BF16 bits FAILED\n");
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
