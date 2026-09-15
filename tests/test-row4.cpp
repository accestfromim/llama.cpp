// Qwen3 Row4/W8A8 format and CPU operator tests.
//
// The reference implementation in this file is intentionally self-contained:
// it does not call production Row4 packing, decoding, activation quantization,
// or linear helpers. This prevents a shared implementation bug from making the
// test and the kernel agree.

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int64_t ROW4_TILE_K = 128;
constexpr int64_t ROW4_TILE_O = 16;
constexpr int64_t ROW4_PAIR2_TILE_K = 256;
constexpr int64_t ROW4_PAIR2_TILE_O = 32;

// Canonical uv_axis_v1 codebook, indexed as [code][row within an O4 group].
constexpr int8_t ROW4_CODEBOOK[16][4] = {
    { 2,  0,  0,  0  },
    { 0,  0,  0,  -2 },
    { 1,  -1, 1,  -1 },
    { 1,  1,  -1, -1 },
    { 0,  0,  0,  2  },
    { -2, 0,  0,  0  },
    { -1, -1, 1,  1  },
    { -1, 1,  -1, 1  },
    { 1,  1,  1,  1  },
    { -1, 1,  1,  -1 },
    { 0,  0,  2,  0  },
    { 0,  2,  0,  0  },
    { 1,  -1, -1, 1  },
    { -1, -1, -1, -1 },
    { 0,  -2, 0,  0  },
    { 0,  0,  -2, 0  },
};

static uint32_t f32_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float f32_from_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

// Independent IEEE-754 binary32 -> BF16 round-to-nearest-even conversion.
static uint16_t oracle_bf16_bits(float value) {
    const uint32_t bits = f32_bits(value);
    const uint32_t abs  = bits & 0x7fffffffu;
    if (abs > 0x7f800000u) {
        // Preserve the payload high bits and force a quiet non-zero NaN.
        return (uint16_t) ((bits >> 16) | 0x0040u);
    }
    const uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t) (rounded >> 16);
}

static float oracle_bf16_from_bits(uint16_t bits) {
    return f32_from_bits((uint32_t) bits << 16);
}

static float oracle_bf16_round(float value) {
    return oracle_bf16_from_bits(oracle_bf16_bits(value));
}

static int32_t oracle_round_away(float value) {
    const float magnitude = floorf(fabsf(value) + 0.5f);
    return (int32_t) copysignf(magnitude, value);
}

struct quantized_token {
    std::vector<int8_t> values;
    float               scale;
};

static quantized_token oracle_quantize_token(const float * input, int64_t k) {
    quantized_token result;
    result.values.resize((size_t) k);

    float amax = 0.0f;
    for (int64_t ik = 0; ik < k; ++ik) {
        amax = std::max(amax, fabsf(oracle_bf16_round(input[ik])));
    }
    result.scale = std::max(amax / 127.0f, 1.0e-8f);

    for (int64_t ik = 0; ik < k; ++ik) {
        const float   value        = oracle_bf16_round(input[ik]);
        const int32_t q            = oracle_round_away(value / result.scale);
        result.values[(size_t) ik] = (int8_t) std::max(-127, std::min(127, q));
    }
    return result;
}

static size_t row4_offset(int64_t o, int64_t k, int64_t logical_k) {
    const int64_t ot    = o / ROW4_TILE_O;
    const int64_t group = (o % ROW4_TILE_O) / 4;
    const int64_t kt    = k / ROW4_TILE_K;
    const int64_t inner = k % ROW4_TILE_K;
    const int64_t split = inner / 16;
    const int64_t lane  = inner % 8;
    const int64_t kts   = logical_k / ROW4_TILE_K;
    return (size_t) (((((ot * kts + kt) * 4 + group) * 8 + split) * 8) + lane);
}

static std::vector<uint8_t> pack_row4_codes(const std::vector<uint8_t> & logical_codes, int64_t o, int64_t k) {
    std::vector<uint8_t> packed((size_t) o * (size_t) k / 8, 0);
    for (int64_t group = 0; group < o / 4; ++group) {
        for (int64_t ik = 0; ik < k; ++ik) {
            const uint8_t code   = logical_codes[(size_t) group * (size_t) k + (size_t) ik];
            const size_t  offset = row4_offset(group * 4, ik, k);
            if ((ik % 16) < 8) {
                packed[offset] = (uint8_t) ((packed[offset] & 0xf0u) | code);
            } else {
                packed[offset] = (uint8_t) ((packed[offset] & 0x0fu) | (uint8_t) (code << 4));
            }
        }
    }
    return packed;
}

static uint8_t unpack_row4_code(const std::vector<uint8_t> & packed, int64_t o, int64_t k, int64_t logical_k) {
    const uint8_t byte = packed[row4_offset(o, k, logical_k)];
    return (k % 16) < 8 ? byte & 0x0fu : byte >> 4;
}

static size_t row4_pair2_offset(int64_t o, int64_t k, int64_t logical_k) {
    const int64_t output_block = o / ROW4_PAIR2_TILE_O;
    const int64_t group        = (o % ROW4_PAIR2_TILE_O) / 4;
    const int64_t k_pair       = k / ROW4_PAIR2_TILE_K;
    const int64_t k_half       = (k % ROW4_PAIR2_TILE_K) / ROW4_TILE_K;
    const int64_t inner        = k % ROW4_TILE_K;
    const int64_t split        = inner / 16;
    const int64_t j            = inner % 8;
    const int64_t lane         = split * 4 + j / 2;
    const int64_t byte         = 2 * k_half + j % 2;
    const int64_t k_pairs      = logical_k / ROW4_PAIR2_TILE_K;
    return (size_t) (((((output_block * k_pairs + k_pair) * 8 + group) * 32 + lane) * 4) + byte);
}

static std::vector<uint8_t> pack_row4_pair2_codes(const std::vector<uint8_t> & logical_codes, int64_t o, int64_t k) {
    std::vector<uint8_t> packed((size_t) o * (size_t) k / 8, 0);
    for (int64_t group = 0; group < o / 4; ++group) {
        for (int64_t ik = 0; ik < k; ++ik) {
            const uint8_t code   = logical_codes[(size_t) group * (size_t) k + (size_t) ik];
            const size_t  offset = row4_pair2_offset(group * 4, ik, k);
            if ((ik % 16) < 8) {
                packed[offset] = (uint8_t) ((packed[offset] & 0xf0u) | code);
            } else {
                packed[offset] = (uint8_t) ((packed[offset] & 0x0fu) | (uint8_t) (code << 4));
            }
        }
    }
    return packed;
}

static uint8_t unpack_row4_pair2_code(const std::vector<uint8_t> & packed, int64_t o, int64_t k, int64_t logical_k) {
    const uint8_t byte = packed[row4_pair2_offset(o, k, logical_k)];
    return (k % 16) < 8 ? byte & 0x0fu : byte >> 4;
}

static int8_t oracle_row4_weight(const std::vector<uint8_t> & packed, int64_t o, int64_t k, int64_t logical_k) {
    return ROW4_CODEBOOK[unpack_row4_code(packed, o, k, logical_k)][o % 4];
}

static size_t w8_offset(int64_t o, int64_t k, int64_t logical_k) {
    const int64_t ot  = o / ROW4_TILE_O;
    const int64_t row = o % ROW4_TILE_O;
    const int64_t kt  = k / ROW4_TILE_K;
    const int64_t ik  = k % ROW4_TILE_K;
    const int64_t kts = logical_k / ROW4_TILE_K;
    return (size_t) ((((ot * kts + kt) * ROW4_TILE_O + row) * ROW4_TILE_K) + ik);
}

static std::vector<float> oracle_row4_linear(const std::vector<float> &    input,
                                             const std::vector<uint8_t> &  codes,
                                             const std::vector<uint16_t> & scale_bits,
                                             int64_t                       o,
                                             int64_t                       k,
                                             int64_t                       tokens) {
    std::vector<float> output((size_t) o * (size_t) tokens);
    for (int64_t token = 0; token < tokens; ++token) {
        const quantized_token qx = oracle_quantize_token(input.data() + token * k, k);
        for (int64_t row = 0; row < o; ++row) {
            int32_t accumulator = 0;
            for (int64_t ik = 0; ik < k; ++ik) {
                accumulator += (int32_t) oracle_row4_weight(codes, row, ik, k) * qx.values[(size_t) ik];
            }
            const float activation_scaled = (float) accumulator * qx.scale;
            const float scaled            = activation_scaled * oracle_bf16_from_bits(scale_bits[(size_t) row]);
            output[(size_t) token * (size_t) o + (size_t) row] = oracle_bf16_round(scaled);
        }
    }
    return output;
}

static std::vector<float> oracle_w8a8_linear(const std::vector<float> &  input,
                                             const std::vector<int8_t> & codes,
                                             const std::vector<float> &  scales,
                                             int64_t                     o,
                                             int64_t                     k,
                                             int64_t                     tokens) {
    std::vector<float> output((size_t) o * (size_t) tokens);
    for (int64_t token = 0; token < tokens; ++token) {
        const quantized_token qx = oracle_quantize_token(input.data() + token * k, k);
        for (int64_t row = 0; row < o; ++row) {
            int32_t accumulator = 0;
            for (int64_t ik = 0; ik < k; ++ik) {
                accumulator += (int32_t) codes[w8_offset(row, ik, k)] * qx.values[(size_t) ik];
            }
            const float activation_scaled = (float) accumulator * qx.scale;
            output[(size_t) token * (size_t) o + (size_t) row] =
                oracle_bf16_round(activation_scaled * scales[(size_t) row]);
        }
    }
    return output;
}

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

static bool compare_exact(const char * label, const std::vector<float> & actual, const std::vector<float> & expected) {
    if (actual.size() != expected.size()) {
        fprintf(stderr, "%s size mismatch: actual=%zu expected=%zu\n", label, actual.size(), expected.size());
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (f32_bits(actual[i]) != f32_bits(expected[i])) {
            fprintf(stderr, "%s mismatch at %zu: actual=%g (0x%08x) expected=%g (0x%08x)\n", label, i, actual[i],
                    f32_bits(actual[i]), expected[i], f32_bits(expected[i]));
            return false;
        }
    }
    return true;
}

static bool test_codebook() {
    bool ok = true;
    for (uint8_t code = 0; code < 16; ++code) {
        const int    axis_u     = code & 3;
        const int    axis_v     = code >> 2;
        const int    ur         = axis_u == 0 ? 1 : axis_u == 1 ? -1 : 0;
        const int    ui         = axis_u == 2 ? 1 : axis_u == 3 ? -1 : 0;
        const int    vr         = axis_v == 0 ? 1 : axis_v == 1 ? -1 : 0;
        const int    vi         = axis_v == 2 ? 1 : axis_v == 3 ? -1 : 0;
        const int8_t decoded[4] = {
            (int8_t) (ur + vr),
            (int8_t) (-ui + vi),
            (int8_t) (ui + vi),
            (int8_t) (ur - vr),
        };
        if (memcmp(decoded, ROW4_CODEBOOK[code], sizeof(decoded)) != 0) {
            fprintf(stderr, "Row4 codebook mismatch at code=0x%x\n", code);
            ok = false;
        }
    }
    printf("  Row4 canonical codebook: 16 codes - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_split8_layout() {
    bool ok = true;

    std::vector<uint8_t> logical(4 * ROW4_TILE_K, 0);
    for (int k = 0; k < 16; ++k) {
        logical[(size_t) k] = (uint8_t) k;
    }
    const std::vector<uint8_t> packed      = pack_row4_codes(logical, 16, ROW4_TILE_K);
    const uint8_t              expected[8] = { 0x80, 0x91, 0xa2, 0xb3, 0xc4, 0xd5, 0xe6, 0xf7 };
    if (memcmp(packed.data(), expected, sizeof(expected)) != 0) {
        fprintf(stderr, "Row4 split8 known vector mismatch\n");
        ok = false;
    }

    constexpr int64_t O = 32;
    constexpr int64_t K = 256;
    logical.assign((size_t) (O / 4) * K, 0);
    for (int64_t group = 0; group < O / 4; ++group) {
        for (int64_t k = 0; k < K; ++k) {
            logical[(size_t) group * K + k] = (uint8_t) ((11 * group + 7 * k) & 15);
        }
    }
    const std::vector<uint8_t> boundary_packed = pack_row4_codes(logical, O, K);
    if (boundary_packed.size() != (size_t) O * K / 8) {
        fprintf(stderr, "Row4 payload size mismatch\n");
        ok = false;
    }
    for (int64_t group = 0; group < O / 4; ++group) {
        for (int64_t k : { 0, 7, 8, 15, 16, 127, 128, 255 }) {
            const uint8_t actual        = unpack_row4_code(boundary_packed, group * 4, k, K);
            const uint8_t expected_code = logical[(size_t) group * K + k];
            if (actual != expected_code) {
                fprintf(stderr, "Row4 boundary mismatch group=%lld k=%lld actual=%u expected=%u\n", (long long) group,
                        (long long) k, actual, expected_code);
                ok = false;
            }
        }
    }

    printf("  Row4 M16K128 split8 layout: known vector and boundaries - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_pair2_layout() {
    constexpr int64_t O = 64;
    constexpr int64_t K = 512;

    std::vector<uint8_t> logical((size_t) (O / 4) * K);
    for (int64_t group = 0; group < O / 4; ++group) {
        for (int64_t k = 0; k < K; ++k) {
            logical[(size_t) group * K + k] = (uint8_t) ((13 * group + 7 * k + k / 128) & 15);
        }
    }

    const std::vector<uint8_t> v1    = pack_row4_codes(logical, O, K);
    const std::vector<uint8_t> pair2 = pack_row4_pair2_codes(logical, O, K);
    bool                       ok    = pair2.size() == (size_t) O * K / 8 && pair2.size() == v1.size();

    for (int64_t group = 0; group < O / 4; ++group) {
        for (int64_t k = 0; k < K; ++k) {
            const uint8_t expected = logical[(size_t) group * K + k];
            const uint8_t actual   = unpack_row4_pair2_code(pair2, group * 4, k, K);
            if (actual != expected) {
                fprintf(stderr, "Row4 pair2 mismatch group=%lld k=%lld actual=%u expected=%u\n", (long long) group,
                        (long long) k, actual, expected);
                ok = false;
                break;
            }
        }
    }

    // Pair2 is a pure permutation of two adjacent v1 K128 tiles.  Every lane
    // stores the two v1 bytes for the first half followed by the two bytes for
    // the second half.
    for (int64_t ob = 0; ob < O / ROW4_PAIR2_TILE_O; ++ob) {
        for (int64_t kp = 0; kp < K / ROW4_PAIR2_TILE_K; ++kp) {
            for (int64_t group = 0; group < 8; ++group) {
                for (int64_t lane = 0; lane < 32; ++lane) {
                    const int64_t output_group = ob * 8 + group;
                    const int64_t split        = lane / 4;
                    const int64_t j            = (lane % 4) * 2;
                    const size_t  pair_base    = row4_pair2_offset(output_group * 4, kp * 256 + split * 16 + j, K);
                    for (int64_t half = 0; half < 2; ++half) {
                        for (int64_t byte = 0; byte < 2; ++byte) {
                            const int64_t k         = kp * 256 + half * 128 + split * 16 + j + byte;
                            const size_t  v1_offset = row4_offset(output_group * 4, k, K);
                            if (pair2[pair_base + 2 * half + byte] != v1[v1_offset]) {
                                fprintf(stderr,
                                        "Row4 pair2 permutation mismatch ob=%lld kp=%lld group=%lld lane=%lld "
                                        "half=%lld byte=%lld\n",
                                        (long long) ob, (long long) kp, (long long) group, (long long) lane,
                                        (long long) half, (long long) byte);
                                ok = false;
                            }
                        }
                    }
                }
            }
        }
    }

    printf("  Row4 M32K256 pair2 layout: exhaustive roundtrip and v1 permutation - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_bf16_and_rounding() {
    struct bf16_case {
        const char * label;
        float        value;
        uint16_t     expected;
    };

    const bf16_case bf16_cases[] = {
        { "+zero",              0.0f,                  0x0000 },
        { "-zero",              -0.0f,                 0x8000 },
        { "tie-even-lower",     1.0f + 0x1p-8f,        0x3f80 },
        { "tie-even-upper",     1.0f + 3.0f * 0x1p-8f, 0x3f82 },
        { "negative-tie",       -1.0f - 0x1p-8f,       0xbf80 },
        { "minimum-subnormal",  0x1p-133f,             0x0001 },
        { "subnormal-tie-zero", 0x1p-134f,             0x0000 },
    };
    bool ok = true;
    for (const bf16_case & tc : bf16_cases) {
        if (oracle_bf16_bits(tc.value) != tc.expected) {
            fprintf(stderr, "BF16 %s mismatch: actual=0x%04x expected=0x%04x\n", tc.label, oracle_bf16_bits(tc.value),
                    tc.expected);
            ok = false;
        }
    }

    const float   round_inputs[]  = { -127.0f, -126.5f, -2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 126.5f, 127.0f };
    const int32_t round_outputs[] = { -127, -127, -3, -2, -1, 1, 2, 3, 127, 127 };
    for (size_t i = 0; i < sizeof(round_inputs) / sizeof(round_inputs[0]); ++i) {
        if (oracle_round_away(round_inputs[i]) != round_outputs[i]) {
            fprintf(stderr, "half-away mismatch at %g\n", round_inputs[i]);
            ok = false;
        }
    }

    const uint16_t signed_scale_bits[] = { 0x3d80, 0xbd80, 0x0000, 0x8000, 0x0001, 0x8001 };
    for (uint16_t bits : signed_scale_bits) {
        if (oracle_bf16_bits(oracle_bf16_from_bits(bits)) != bits) {
            fprintf(stderr, "signed BF16 scale payload changed: 0x%04x\n", bits);
            ok = false;
        }
    }

    printf("  BF16 RNE, half-away, and signed scales - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_activation_profile() {
    bool                           ok    = true;
    std::array<float, ROW4_TILE_K> zero  = {};
    const quantized_token          qzero = oracle_quantize_token(zero.data(), ROW4_TILE_K);
    if (f32_bits(qzero.scale) != f32_bits(1.0e-8f) ||
        std::any_of(qzero.values.begin(), qzero.values.end(), [](int8_t value) { return value != 0; })) {
        fprintf(stderr, "zero activation scale floor mismatch\n");
        ok = false;
    }

    std::array<float, ROW4_TILE_K> a = {};
    std::array<float, ROW4_TILE_K> b = {};
    for (int64_t k = 0; k < ROW4_TILE_K; ++k) {
        // Keep the pair away from signed zero, whose sign bit is intentionally
        // preserved by BF16 and would make the two rounded payloads differ.
        const float base = 4.0f + (float) ((k % 31) - 15) / 8.0f;
        a[(size_t) k]    = base + 0x1p-20f;
        b[(size_t) k]    = base - 0x1p-20f;
        if (oracle_bf16_bits(a[(size_t) k]) != oracle_bf16_bits(b[(size_t) k])) {
            fprintf(stderr, "test construction failed: BF16 pair differs at k=%lld\n", (long long) k);
            return false;
        }
    }
    const quantized_token qa = oracle_quantize_token(a.data(), ROW4_TILE_K);
    const quantized_token qb = oracle_quantize_token(b.data(), ROW4_TILE_K);
    if (f32_bits(qa.scale) != f32_bits(qb.scale) || qa.values != qb.values) {
        fprintf(stderr, "activation profile depended on discarded F32 low bits\n");
        ok = false;
    }

    printf("  Activation BF16/A8 profile and scale floor - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_int32_extremes() {
    bool              ok       = true;
    constexpr int64_t K        = 12288;
    int32_t           row4_sum = 0;
    int32_t           cancel   = 0;
    for (int64_t k = 0; k < K; ++k) {
        row4_sum += 2 * 127;
        cancel += (k & 1) ? -2 * 127 : 2 * 127;
    }
    if (row4_sum != 3121152 || cancel != 0) {
        fprintf(stderr, "Row4 INT32 extreme mismatch: sum=%d cancel=%d\n", row4_sum, cancel);
        ok = false;
    }

    int32_t w8_sum = 0;
    for (int64_t k = 0; k < 4096; ++k) {
        w8_sum += 127 * 127;
    }
    if (w8_sum != 66064384) {
        fprintf(stderr, "W8 INT32 extreme mismatch: %d\n", w8_sum);
        ok = false;
    }

    printf("  INT32 Row4/W8 extremes and cancellation - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static std::vector<float> make_input(int64_t k, int64_t tokens) {
    std::vector<float> input((size_t) k * (size_t) tokens);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t ik = 0; ik < k; ++ik) {
            const int   value                                = (int) ((37 * ik + 53 * token + 11) % 255) - 127;
            const float low                                  = ((ik + token) & 1) ? 0x1p-20f : -0x1p-20f;
            input[(size_t) token * (size_t) k + (size_t) ik] = (float) value / 16.0f + low;
        }
    }
    return input;
}

constexpr int8_t ROW4_LUT16_ACTIVATION_EXTREMES[] = { -127, -1, 0, 1, 127 };

static int row4_lut16_activation_index(int64_t k, int64_t token = 0) {
    constexpr int count = sizeof(ROW4_LUT16_ACTIVATION_EXTREMES) / sizeof(ROW4_LUT16_ACTIVATION_EXTREMES[0]);
    return (int) ((k + k / 16 + 3 * token) % count);
}

static uint8_t row4_lut16_exhaustive_code(int64_t group, int64_t k) {
    return (uint8_t) ((5 * group + 3 * k + 7 * (k / 16) + 11 * (k / 128)) & 15);
}

static std::vector<float> make_row4_lut16_exhaustive_input(int64_t k, int64_t tokens) {
    std::vector<float> input((size_t) k * (size_t) tokens);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t ik = 0; ik < k; ++ik) {
            input[(size_t) token * (size_t) k + (size_t) ik] =
                (float) ROW4_LUT16_ACTIVATION_EXTREMES[row4_lut16_activation_index(ik, token)];
        }
    }
    return input;
}

static std::vector<uint8_t> make_row4_lut16_exhaustive_logical(int64_t o, int64_t k) {
    std::vector<uint8_t> logical((size_t) (o / 4) * (size_t) k);
    for (int64_t group = 0; group < o / 4; ++group) {
        for (int64_t ik = 0; ik < k; ++ik) {
            logical[(size_t) group * (size_t) k + (size_t) ik] = row4_lut16_exhaustive_code(group, ik);
        }
    }
    return logical;
}

static bool test_row4_lut16_oracle() {
    bool ok = true;

    // Reconstruct each four-row LUT entry from the canonical uv axes instead
    // of copying the Metal table.  The algebraic oracle must agree with the
    // canonical codebook for every code and signed A8 extreme used below.
    for (uint8_t code = 0; code < 16; ++code) {
        const int axis_u = code & 3;
        const int axis_v = code >> 2;
        const int ur     = axis_u == 0 ? 1 : axis_u == 1 ? -1 : 0;
        const int ui     = axis_u == 2 ? 1 : axis_u == 3 ? -1 : 0;
        const int vr     = axis_v == 0 ? 1 : axis_v == 1 ? -1 : 0;
        const int vi     = axis_v == 2 ? 1 : axis_v == 3 ? -1 : 0;
        for (int8_t activation : ROW4_LUT16_ACTIVATION_EXTREMES) {
            const int u_real        = ur * (int) activation;
            const int u_imag        = ui * (int) activation;
            const int v_real        = vr * (int) activation;
            const int v_imag        = vi * (int) activation;
            const int oracle_rows[] = {
                u_real + v_real,
                -u_imag + v_imag,
                u_imag + v_imag,
                u_real - v_real,
            };
            for (int row = 0; row < 4; ++row) {
                const int   row_i32 = (int) ROW4_CODEBOOK[code][row] * (int) activation;
                const float row_f32 = (float) ROW4_CODEBOOK[code][row] * (float) activation;
                if (oracle_rows[row] != row_i32 || row_f32 != (float) row_i32) {
                    fprintf(stderr, "Row4 LUT16 oracle mismatch code=%u activation=%d row=%d\n", code, (int) activation,
                            row);
                    ok = false;
                }
            }
        }
    }

    // This one physical Pair2 tile covers both nibble positions, both K128
    // halves, and both O4s assigned to one SIMDgroup.  Across O96, every
    // code/activation combination occurs in every one of those dimensions.
    constexpr int64_t          O       = 96;
    constexpr int64_t          K       = 256;
    const std::vector<float>   input   = make_row4_lut16_exhaustive_input(K, 1);
    const quantized_token      qx      = oracle_quantize_token(input.data(), K);
    const std::vector<uint8_t> logical = make_row4_lut16_exhaustive_logical(O, K);
    const std::vector<uint8_t> v1      = pack_row4_codes(logical, O, K);
    const std::vector<uint8_t> pair2   = pack_row4_pair2_codes(logical, O, K);
    bool seen[16][sizeof(ROW4_LUT16_ACTIVATION_EXTREMES) / sizeof(ROW4_LUT16_ACTIVATION_EXTREMES[0])][2][2][2] = {};

    if (f32_bits(qx.scale) != f32_bits(1.0f)) {
        fprintf(stderr, "Row4 LUT16 exhaustive activation scale is not exactly one\n");
        ok = false;
    }
    for (int64_t group = 0; group < O / 4; ++group) {
        for (int64_t ik = 0; ik < K; ++ik) {
            const int     activation_index = row4_lut16_activation_index(ik);
            const uint8_t code             = logical[(size_t) group * K + (size_t) ik];
            seen[code][activation_index][(ik % 16) >= 8][ik >= ROW4_TILE_K][group & 1] = true;
            if (qx.values[(size_t) ik] != ROW4_LUT16_ACTIVATION_EXTREMES[activation_index] ||
                unpack_row4_code(v1, group * 4, ik, K) != code ||
                unpack_row4_pair2_code(pair2, group * 4, ik, K) != code) {
                fprintf(stderr, "Row4 LUT16 exhaustive fixture mismatch group=%lld k=%lld\n", (long long) group,
                        (long long) ik);
                ok = false;
            }
        }
    }
    for (int code = 0; code < 16; ++code) {
        for (size_t activation = 0;
             activation < sizeof(ROW4_LUT16_ACTIVATION_EXTREMES) / sizeof(ROW4_LUT16_ACTIVATION_EXTREMES[0]);
             ++activation) {
            for (int nibble = 0; nibble < 2; ++nibble) {
                for (int half = 0; half < 2; ++half) {
                    for (int dual_o4 = 0; dual_o4 < 2; ++dual_o4) {
                        if (!seen[code][activation][nibble][half][dual_o4]) {
                            fprintf(stderr,
                                    "Row4 LUT16 fixture lacks code=%d activation=%d nibble=%d half=%d dual_o4=%d\n",
                                    code, (int) activation, nibble, half, dual_o4);
                            ok = false;
                        }
                    }
                }
            }
        }
    }

    printf("  Row4 LUT16 independent uv-axis oracle and Pair2 coverage - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static std::vector<uint16_t> make_row4_scales(int64_t o) {
    const uint16_t        profiles[] = { 0x3d80, 0xbd80, 0x3f80, 0xbf80, 0x0000, 0x8000, 0x0001, 0x8001 };
    std::vector<uint16_t> scales((size_t) o);
    for (int64_t row = 0; row < o; ++row) {
        scales[(size_t) row] = profiles[row % (int64_t) (sizeof(profiles) / sizeof(profiles[0]))];
    }
    return scales;
}

static std::vector<uint8_t> make_row4_codes(int64_t o, int64_t k) {
    std::vector<uint8_t> logical((size_t) (o / 4) * (size_t) k);
    for (int64_t group = 0; group < o / 4; ++group) {
        for (int64_t ik = 0; ik < k; ++ik) {
            logical[(size_t) group * (size_t) k + (size_t) ik] =
                (uint8_t) ((5 * group + 3 * ik + 7 * (ik / 128) + 11 * (ik / 256)) & 15);
        }
    }
    return pack_row4_codes(logical, o, k);
}

static std::vector<uint8_t> make_row4_pair2_codes(int64_t o, int64_t k) {
    std::vector<uint8_t> logical((size_t) (o / 4) * (size_t) k);
    for (int64_t group = 0; group < o / 4; ++group) {
        for (int64_t ik = 0; ik < k; ++ik) {
            logical[(size_t) group * (size_t) k + (size_t) ik] =
                (uint8_t) ((5 * group + 3 * ik + 7 * (ik / 128) + 11 * (ik / 256)) & 15);
        }
    }
    return pack_row4_pair2_codes(logical, o, k);
}

static std::vector<int8_t> make_w8_codes(int64_t o, int64_t k) {
    std::vector<int8_t> codes((size_t) o * (size_t) k);
    for (int64_t row = 0; row < o; ++row) {
        for (int64_t ik = 0; ik < k; ++ik) {
            codes[w8_offset(row, ik, k)] = (int8_t) (((17 * row + 29 * ik + 3) % 255) - 127);
        }
    }
    return codes;
}

static std::vector<float> make_w8_scales(int64_t o) {
    const float        profiles[] = { 0.0f, -0.0f, 0.000125f, -0.000125f, 0.03125f, -0.0625f, 1.0f };
    std::vector<float> scales((size_t) o);
    for (int64_t row = 0; row < o; ++row) {
        scales[(size_t) row] = profiles[row % (int64_t) (sizeof(profiles) / sizeof(profiles[0]))];
    }
    return scales;
}

enum class linear_kind {
    row4,
    row4_pair2,
    w8a8,
};

static bool cpu_path_available(const char * path, int64_t tokens) {
    scoped_env_var force("GGML_ROW4_TEST_FORCE_PATH");
    force.set(path);

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        return false;
    }
    const ggml_init_params params = {
        /*.mem_size   =*/256 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor * x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, tokens);
    ggml_tensor * codes  = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES, 64, 4, 1, 8);
    ggml_tensor * scales = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, 128);
    ggml_tensor * result = ggml_row4_linear(ctx, x, codes, scales, 128, 128);
    const bool    ok     = ggml_backend_supports_op(backend, result);

    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok;
}

static bool run_operator_backend(std::vector<float> &          output,
                                 linear_kind                   kind,
                                 const std::vector<float> &    input,
                                 const std::vector<uint8_t> &  row4_codes,
                                 const std::vector<uint16_t> & row4_scales,
                                 const std::vector<int8_t> &   w8_codes,
                                 const std::vector<float> &    w8_scales,
                                 int64_t                       o,
                                 int64_t                       k,
                                 int64_t                       tokens,
                                 const char *                  force_path,
                                 bool                          debug_marker     = false,
                                 ggml_backend_t                backend_override = nullptr,
                                 bool                          rewrite_codes    = false) {
    scoped_env_var force("GGML_ROW4_TEST_FORCE_PATH");
    scoped_env_var debug("GGML_ROW4_CPU_DEBUG");
    if (force_path) {
        force.set(force_path);
    } else {
        force.unset();
    }
    if (debug_marker) {
        debug.set("1");
    } else {
        debug.unset();
    }

    const bool     owns_backend = backend_override == nullptr;
    ggml_backend_t backend      = owns_backend ? ggml_backend_cpu_init() : backend_override;
    if (!backend) {
        fprintf(stderr, "failed to initialize CPU backend\n");
        return false;
    }
    if (owns_backend) {
        ggml_backend_cpu_set_n_threads(backend, 4);
    }

    const ggml_init_params params = {
        /*.mem_size   =*/8 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        if (owns_backend) {
            ggml_backend_free(backend);
        }
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, tokens);
    ggml_tensor * codes;
    ggml_tensor * scales;
    ggml_tensor * result;
    if (kind == linear_kind::row4) {
        codes  = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES, 64, 4, k / 128, o / 16);
        scales = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, o);
        result = ggml_row4_linear(ctx, x, codes, scales, o, k);
    } else if (kind == linear_kind::row4_pair2) {
        codes  = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES_PAIR2, 128, 8, k / 256, o / 32);
        scales = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, o);
        result = ggml_row4_linear(ctx, x, codes, scales, o, k);
    } else {
        codes  = ggml_new_tensor_4d(ctx, GGML_TYPE_I8, 128, 16, k / 128, o / 16);
        scales = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, o);
        result = ggml_w8a8_linear(ctx, x, codes, scales, o, k);
    }

    const bool is_row4 = kind != linear_kind::w8a8;
    int32_t    op_params[3];
    memcpy(op_params, result->op_params, sizeof(op_params));
    const int expected_layout = kind == linear_kind::row4_pair2 ? 2 : 1;
    bool      ok = result->type == GGML_TYPE_F32 && result->ne[0] == o && result->ne[1] == tokens &&
                   result->op == (is_row4 ? GGML_OP_ROW4_LINEAR : GGML_OP_W8A8_LINEAR) && result->src[0] == x &&
                   result->src[1] == codes && result->src[2] == scales && op_params[0] == expected_layout &&
                   op_params[1] == o && op_params[2] == k && ggml_backend_supports_op(backend, result);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        if (owns_backend) {
            ggml_backend_free(backend);
        }
        return false;
    }

    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    if (is_row4) {
        ggml_backend_tensor_set(codes, row4_codes.data(), 0, row4_codes.size());
        ggml_backend_tensor_set(scales, row4_scales.data(), 0, row4_scales.size() * sizeof(uint16_t));
    } else {
        ggml_backend_tensor_set(codes, w8_codes.data(), 0, w8_codes.size());
        ggml_backend_tensor_set(scales, w8_scales.data(), 0, w8_scales.size() * sizeof(float));
    }

    if (rewrite_codes) {
        // Reuse the same graph and scratch after updating the compressed
        // weights. Every numeric INT4 destination must be overwritten.
        GGML_ASSERT(is_row4);
        const std::vector<uint8_t> poison(row4_codes.size(), 0xa5U);
        ggml_backend_tensor_set(codes, poison.data(), 0, poison.size());
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS && ok;
        ggml_backend_tensor_set(codes, row4_codes.data(), 0, row4_codes.size());
    }

    ok                       = codes->extra == nullptr && scales->extra == nullptr && ok;
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s path=%s graph compute failed: %s\n", is_row4 ? "Row4" : "W8A8",
                force_path ? force_path : "default", ggml_status_to_string(status));
        ok = false;
    } else {
        output.resize((size_t) o * (size_t) tokens);
        ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
    }
    ok = codes->extra == nullptr && scales->extra == nullptr && ok;

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    if (owns_backend) {
        ggml_backend_free(backend);
    }
    return ok;
}

static ggml_backend_dev_t find_metal_device();

struct pair2_copy_log_marker {
    std::atomic<bool> hit = false;
};

static void pair2_copy_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    pair2_copy_log_marker * state = static_cast<pair2_copy_log_marker *>(user_data);
    if (strstr(text, "kernel_cpy_row4_codes_pair2_row4_codes_pair2")) {
        state->hit.store(true, std::memory_order_relaxed);
    }
}

static bool run_pair2_raw_copy_backend(ggml_backend_t backend, const char * label, bool require_pipeline_marker) {
    const ggml_init_params params = {
        /*.mem_size   =*/256 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * src  = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES_PAIR2, 128, 8, 2, 3);
    ggml_tensor * dst  = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES_PAIR2, 128, 8, 2, 3);
    ggml_tensor * copy = ggml_cpy(ctx, src, dst);
    ggml_set_output(copy);

    bool          ok    = ggml_backend_supports_op(backend, copy);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, copy);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }

    std::vector<uint8_t> expected(ggml_nbytes(src));
    std::vector<uint8_t> sentinel(expected.size(), 0xa5U);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = (uint8_t) ((73 * i + 19) & 0xffU);
    }
    ggml_backend_tensor_set(src, expected.data(), 0, expected.size());
    ggml_backend_tensor_set(dst, sentinel.data(), 0, sentinel.size());

    pair2_copy_log_marker marker;
    if (require_pipeline_marker) {
        ggml_log_set(pair2_copy_log_callback, &marker);
    }
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (require_pipeline_marker) {
        ggml_log_set(nullptr, nullptr);
    }

    std::vector<uint8_t> actual(expected.size());
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(copy, actual.data(), 0, actual.size());
    } else {
        fprintf(stderr, "%s Pair2 raw CPY graph compute failed: %s\n", label, ggml_status_to_string(status));
        ok = false;
    }
    if (actual != expected) {
        const auto   mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
        const size_t offset   = (size_t) std::distance(actual.begin(), mismatch.first);
        fprintf(stderr, "%s Pair2 raw CPY byte mismatch at %zu\n", label, offset);
        ok = false;
    }
    if (require_pipeline_marker && !marker.hit.load(std::memory_order_relaxed)) {
        fprintf(stderr, "%s Pair2 raw CPY did not hit the lazy Metal Pair2 pipeline\n", label);
        ok = false;
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

static bool test_opaque_type_isolation() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "failed to initialize CPU backend for opaque type tests\n");
        return false;
    }

    const ggml_init_params params = {
        /*.mem_size   =*/1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor * codes_a = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES, 64, 4, 1, 8);
    ggml_tensor * codes_b = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES, 64, 4, 1, 8);
    ggml_tensor * f32     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 4, 1, 8);

    ggml_tensor * pair2_a   = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES_PAIR2, 128, 8, 1, 1);
    ggml_tensor * pair2_b   = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES_PAIR2, 128, 8, 1, 1);
    ggml_tensor * pair2_f32 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 8, 1, 1);

    ggml_tensor * raw_copy          = ggml_cpy(ctx, codes_a, codes_b);
    ggml_tensor * row4_to_f32       = ggml_cpy(ctx, codes_a, f32);
    ggml_tensor * f32_to_row4       = ggml_cpy(ctx, f32, codes_b);
    ggml_tensor * add               = ggml_add(ctx, codes_a, codes_b);
    ggml_tensor * row4_rhs          = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 1, 1, 8);
    ggml_tensor * generic_row4      = ggml_mul_mat(ctx, codes_a, row4_rhs);
    ggml_tensor * indices           = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 1, 1, 8);
    ggml_tensor * get_rows          = ggml_get_rows(ctx, codes_a, indices);

    ggml_tensor * pair2_raw_copy    = ggml_cpy(ctx, pair2_a, pair2_b);
    ggml_tensor * pair2_to_f32      = ggml_cpy(ctx, pair2_a, pair2_f32);
    ggml_tensor * f32_to_pair2      = ggml_cpy(ctx, pair2_f32, pair2_b);
    ggml_tensor * pair2_add         = ggml_add(ctx, pair2_a, pair2_b);
    ggml_tensor * pair2_rhs         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
    ggml_tensor * pair2_scales      = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, 32);
    ggml_tensor * pair2_linear      = ggml_row4_linear(ctx, pair2_rhs, pair2_a, pair2_scales, 32, 256);
    ggml_tensor * generic_pair2_rhs = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 1, 1, 8);
    ggml_tensor * generic_pair2     = ggml_mul_mat(ctx, pair2_a, generic_pair2_rhs);
    ggml_tensor * pair2_indices     = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 1, 1, 1);
    ggml_tensor * pair2_get_rows    = ggml_get_rows(ctx, pair2_a, pair2_indices);

    // Pair2 B1 stages the full activation row in Metal threadgroup memory.
    // This valid opaque layout deliberately exceeds all supported Apple GPU
    // threadgroup-memory limits and therefore must not be advertised.
    constexpr int64_t oversized_pair2_k = 1 << 20;
    ggml_tensor *     oversized_pair2_codes =
        ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES_PAIR2, 128, 8, oversized_pair2_k / 256, 1);
    ggml_tensor * oversized_pair2_rhs    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, oversized_pair2_k, 1);
    ggml_tensor * oversized_pair2_scales = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, 32);
    ggml_tensor * oversized_pair2_linear = ggml_row4_linear(ctx, oversized_pair2_rhs, oversized_pair2_codes,
                                                            oversized_pair2_scales, 32, oversized_pair2_k);

    auto generic_ops_are_isolated = [&](ggml_backend_t candidate) {
        return !ggml_backend_supports_op(candidate, row4_to_f32) && !ggml_backend_supports_op(candidate, f32_to_row4) &&
               !ggml_backend_supports_op(candidate, add) && !ggml_backend_supports_op(candidate, generic_row4) &&
               !ggml_backend_supports_op(candidate, get_rows);
    };

    bool ok = ggml_backend_supports_op(backend, raw_copy) && generic_ops_are_isolated(backend);
    if (!ok) {
        fprintf(stderr, "CPU ROW4_CODES escaped its dedicated operator contract\n");
    }
    const bool pair2_cpu_ok =
        ggml_backend_supports_op(backend, pair2_raw_copy) && !ggml_backend_supports_op(backend, pair2_to_f32) &&
        !ggml_backend_supports_op(backend, f32_to_pair2) && !ggml_backend_supports_op(backend, pair2_add) &&
        !ggml_backend_supports_op(backend, generic_pair2) && !ggml_backend_supports_op(backend, pair2_get_rows) &&
        !ggml_backend_supports_op(backend, pair2_linear);
    if (!pair2_cpu_ok) {
        fprintf(stderr, "CPU ROW4_CODES_PAIR2 violated its opaque Metal-only contract\n");
        ok = false;
    }
    ok = run_pair2_raw_copy_backend(backend, "CPU", false) && ok;

    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const bool         device_ok =
            !ggml_backend_dev_supports_op(dev, row4_to_f32) && !ggml_backend_dev_supports_op(dev, f32_to_row4) &&
            !ggml_backend_dev_supports_op(dev, add) && !ggml_backend_dev_supports_op(dev, generic_row4) &&
            !ggml_backend_dev_supports_op(dev, get_rows) && !ggml_backend_dev_supports_op(dev, pair2_to_f32) &&
            !ggml_backend_dev_supports_op(dev, f32_to_pair2) && !ggml_backend_dev_supports_op(dev, pair2_add) &&
            !ggml_backend_dev_supports_op(dev, generic_pair2) && !ggml_backend_dev_supports_op(dev, pair2_get_rows);
        if (!device_ok) {
            fprintf(stderr, "%s device advertised a generic ROW4_CODES operation\n", ggml_backend_dev_name(dev));
            ok = false;
        }
    }

    ggml_backend_dev_t metal_dev = find_metal_device();
    if (metal_dev) {
        ggml_backend_t metal    = ggml_backend_dev_init(metal_dev, nullptr);
        const bool     metal_ok =
            metal && generic_ops_are_isolated(metal) && ggml_backend_supports_op(metal, pair2_raw_copy) &&
            !ggml_backend_supports_op(metal, pair2_to_f32) && !ggml_backend_supports_op(metal, f32_to_pair2) &&
            !ggml_backend_supports_op(metal, pair2_add) && !ggml_backend_supports_op(metal, generic_pair2) &&
            !ggml_backend_supports_op(metal, pair2_get_rows) && ggml_backend_supports_op(metal, pair2_linear) &&
            !ggml_backend_supports_op(metal, oversized_pair2_linear);
        if (!metal_ok) {
            fprintf(stderr, "Metal Row4 opaque/dedicated operator contract mismatch\n");
            ok = false;
        }
        if (metal) {
            ok = run_pair2_raw_copy_backend(metal, "Metal", true) && ok;
        }
        if (metal) {
            ggml_backend_free(metal);
        }
    } else {
        printf("  Metal opaque ROW4_CODES generic-op isolation: SKIP (Metal backend unavailable)\n");
    }

    ggml_free(ctx);
    ggml_backend_free(backend);
    printf("  CPU%s opaque ROW4_CODES isolation and Pair2 raw CPY execution - %s\n", metal_dev ? "/Metal" : "",
           ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_cpu_operator_matrix() {
    constexpr int64_t           O           = 128;
    constexpr int64_t           K           = 128;
    const std::vector<uint8_t>  row4_codes  = make_row4_codes(O, K);
    const std::vector<uint16_t> row4_scales = make_row4_scales(O);
    const std::vector<int8_t>   w8_codes    = make_w8_codes(O, K);
    const std::vector<float>    w8_scales   = make_w8_scales(O);

    const bool    dotprod_available = cpu_path_available("dotprod", 1);
    const bool    i8mm_available    = cpu_path_available("i8mm", 2);
    bool          ok                = true;
    const int64_t token_counts[]    = { 1, 2, 8, 9, 31, 32, 33, 63, 64, 65, 512 };
    for (int64_t tokens : token_counts) {
        const std::vector<float> input    = make_input(K, tokens);
        const std::vector<float> expected = oracle_row4_linear(input, row4_codes, row4_scales, O, K, tokens);
        std::vector<float>       actual;
        if (!run_operator_backend(actual, linear_kind::row4, input, row4_codes, row4_scales, {}, {}, O, K, tokens,
                                  "scalar") ||
            !compare_exact(("Row4 scalar B=" + std::to_string(tokens)).c_str(), actual, expected)) {
            ok = false;
        }
    }

    // These forced-path executions are path assertions: a requested but
    // unavailable implementation must fail instead of silently falling back.
    if (dotprod_available) {
        const std::vector<float> input    = make_input(K, 1);
        const std::vector<float> expected = oracle_row4_linear(input, row4_codes, row4_scales, O, K, 1);
        std::vector<float>       actual;
        if (!run_operator_backend(actual, linear_kind::row4, input, row4_codes, row4_scales, {}, {}, O, K, 1, "dotprod",
                                  true) ||
            !compare_exact("Row4 dotprod B=1", actual, expected)) {
            ok = false;
        }
    } else {
        printf("  Row4 dotprod forced path: SKIP (CPU/build lacks dotprod)\n");
    }

    if (i8mm_available) {
        for (int64_t tokens : { 2, 8 }) {
            const std::vector<float> input    = make_input(K, tokens);
            const std::vector<float> expected = oracle_row4_linear(input, row4_codes, row4_scales, O, K, tokens);
            std::vector<float>       actual;
            if (!run_operator_backend(actual, linear_kind::row4, input, row4_codes, row4_scales, {}, {}, O, K, tokens,
                                      "i8mm", true) ||
                !compare_exact(("Row4 i8mm B=" + std::to_string(tokens)).c_str(), actual, expected)) {
                ok = false;
            }
        }
    } else {
        printf("  Row4 i8mm forced path: SKIP (CPU/build lacks i8mm)\n");
    }

    for (int64_t tokens : { 1, 2, 8 }) {
        const std::vector<float> input    = make_input(K, tokens);
        const std::vector<float> expected = oracle_w8a8_linear(input, w8_codes, w8_scales, O, K, tokens);
        std::vector<float>       actual;
        if (!run_operator_backend(actual, linear_kind::w8a8, input, {}, {}, w8_codes, w8_scales, O, K, tokens,
                                  "scalar") ||
            !compare_exact(("W8A8 scalar B=" + std::to_string(tokens)).c_str(), actual, expected)) {
            ok = false;
        }
        if (dotprod_available && tokens == 1) {
            if (!run_operator_backend(actual, linear_kind::w8a8, input, {}, {}, w8_codes, w8_scales, O, K, tokens,
                                      "dotprod", true) ||
                !compare_exact("W8A8 dotprod B=1", actual, expected)) {
                ok = false;
            }
        }
        if (i8mm_available && tokens != 1) {
            if (!run_operator_backend(actual, linear_kind::w8a8, input, {}, {}, w8_codes, w8_scales, O, K, tokens,
                                      "i8mm", true) ||
                !compare_exact(("W8A8 i8mm B=" + std::to_string(tokens)).c_str(), actual, expected)) {
                ok = false;
            }
        }
    }

    // Cross a K-tile boundary in every production path. This catches code
    // which accidentally treats ne[2] as a flat byte stride.
    {
        constexpr int64_t           K2      = 256;
        const std::vector<uint8_t>  codes2  = make_row4_codes(O, K2);
        const std::vector<uint16_t> scales2 = make_row4_scales(O);
        const std::vector<int8_t>   w8c2    = make_w8_codes(O, K2);
        const std::vector<float>    w8s2    = make_w8_scales(O);
        for (int64_t tokens : { 1, 8, 9 }) {
            const std::vector<float> input = make_input(K2, tokens);
            std::vector<float>       actual;
            const std::vector<float> expected_row4 = oracle_row4_linear(input, codes2, scales2, O, K2, tokens);
            const char * row4_path = tokens == 1 && dotprod_available ? "dotprod" : i8mm_available ? "i8mm" : "scalar";
            if (!run_operator_backend(actual, linear_kind::row4, input, codes2, scales2, {}, {}, O, K2, tokens,
                                      row4_path) ||
                !compare_exact(("Row4 K=256 B=" + std::to_string(tokens)).c_str(), actual, expected_row4)) {
                ok = false;
            }

            const std::vector<float> expected_w8 = oracle_w8a8_linear(input, w8c2, w8s2, O, K2, tokens);
            const char * w8_path = tokens == 1 && dotprod_available ? "dotprod" : i8mm_available ? "i8mm" : "scalar";
            if (!run_operator_backend(actual, linear_kind::w8a8, input, {}, {}, w8c2, w8s2, O, K2, tokens, w8_path) ||
                !compare_exact(("W8A8 K=256 B=" + std::to_string(tokens)).c_str(), actual, expected_w8)) {
                ok = false;
            }
        }
    }

    // Exercise the real Row4 down-projection K and the exact maximum Row4
    // accumulator in production code. Group 0 is all +2; group 1 alternates
    // +2/-2 and must cancel exactly.
    {
        constexpr int64_t    KMAX = 12288;
        std::vector<uint8_t> logical((size_t) (O / 4) * KMAX, 0);
        for (int64_t k = 0; k < KMAX; ++k) {
            logical[(size_t) KMAX + (size_t) k] = (uint8_t) ((k & 1) ? 5 : 0);
        }
        const std::vector<uint8_t> codes_max = pack_row4_codes(logical, O, KMAX);
        std::vector<uint16_t>      scales_max((size_t) O, oracle_bf16_bits(1.0f));
        std::vector<float>         input((size_t) KMAX, 1.0f);
        const std::vector<float>   expected = oracle_row4_linear(input, codes_max, scales_max, O, KMAX, 1);
        std::vector<float>         actual;
        const char *               path = dotprod_available ? "dotprod" : "scalar";
        if (!run_operator_backend(actual, linear_kind::row4, input, codes_max, scales_max, {}, {}, O, KMAX, 1, path) ||
            !compare_exact("Row4 K=12288 maximum/cancellation", actual, expected) ||
            f32_bits(expected[0]) != f32_bits(24576.0f) || f32_bits(expected[4]) != f32_bits(0.0f)) {
            fprintf(stderr, "Row4 production maximum/cancellation guard failed\n");
            ok = false;
        }
    }

    // lm_head K=4096 maximum W8 accumulator. The accumulator itself is
    // 66,064,384; rescaling happens only after the exact INT32 reduction.
    {
        constexpr int64_t        KMAX = 4096;
        std::vector<int8_t>      codes_max((size_t) O * KMAX, 127);
        std::vector<float>       scales_max((size_t) O, 1.0f);
        std::vector<float>       input((size_t) KMAX, 1.0f);
        const std::vector<float> expected = oracle_w8a8_linear(input, codes_max, scales_max, O, KMAX, 1);
        std::vector<float>       actual;
        const char *             path = dotprod_available ? "dotprod" : "scalar";
        if (!run_operator_backend(actual, linear_kind::w8a8, input, {}, {}, codes_max, scales_max, O, KMAX, 1, path) ||
            !compare_exact("W8A8 K=4096 maximum accumulator", actual, expected)) {
            ok = false;
        }
    }

    printf("  CPU Row4/W8A8 oracle and forced path matrix - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static ggml_backend_dev_t find_metal_device() {
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

struct gate_up_fusion_log_marker {
    std::atomic<bool> hit                     = false;
    std::atomic<bool> m5_tensorops_bypass_hit = false;
};

struct m5_tensorops_path_log_marker {
    std::atomic<uint32_t> tile_mask              = 0;
    std::atomic<bool>     pair2_device_preexpand = false;
    std::atomic<bool>     pair2_cooperative_n64  = false;
    std::atomic<bool>     pair2_cooperative_n128 = false;
};

static void gate_up_fusion_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    constexpr const char * marker = "fuse Row4 gate_up + VIEW + QAT SILU_EXACT + VIEW + QAT MUL_EXACT + ROW4_LINEAR";
    if (strstr(text, marker)) {
        static_cast<gate_up_fusion_log_marker *>(user_data)->hit.store(true, std::memory_order_relaxed);
    }
    constexpr const char * m5_marker = "bypass Row4 gate-up producer fusion for M5 MPP TensorOps";
    if (strstr(text, m5_marker)) {
        static_cast<gate_up_fusion_log_marker *>(user_data)->m5_tensorops_bypass_hit.store(true,
                                                                                           std::memory_order_relaxed);
    }
}

static void m5_tensorops_path_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    auto * marker = static_cast<m5_tensorops_path_log_marker *>(user_data);
    if (strstr(text, "layout=m32k256_pair2_split8_v2") && strstr(text, "cooperative-store")) {
        if (strstr(text, "M64N64 BK512 device-preexpand")) {
            marker->pair2_cooperative_n64.store(true, std::memory_order_relaxed);
        }
        if (strstr(text, "M64N128 BK512 device-preexpand")) {
            marker->pair2_cooperative_n128.store(true, std::memory_order_relaxed);
        }
    }
    uint32_t bit = 0;
    if (strstr(text, "M5 MPP TensorOps exact A8/I4/I32 M32N128 BK128 device-preexpand")) {
        bit = 1u << 4;
    } else if (strstr(text, "M5 MPP TensorOps exact A8/I4/I32 M8N128 multi-decode")) {
        bit = 1u << 5;
    } else if (strstr(text, "M5 MPP TensorOps exact A8/I4/I32 M16N64 SG4 multi-decode")) {
        bit = 1u << 6;
    } else if (strstr(text, "M5 MPP TensorOps exact A8/I4/I32 M32N128")) {
        bit = 1u << 0;
    } else if (strstr(text, "M5 MPP TensorOps exact A8/I4/I32 M64N64")) {
        bit = 1u << 1;
    } else if (strstr(text, "M5 MPP TensorOps exact A8/I4/I32 M128N32")) {
        bit = 1u << 2;
    } else if (strstr(text, "M5 MPP TensorOps exact A8/I4/I32 M256N32")) {
        bit = 1u << 3;
    } else if (strstr(text, "M5 MPP TensorOps exact A8/I8/I32 M8N16 direct-device")) {
        bit = 1u << 7;
    } else if (strstr(text, "M5 MPP TensorOps exact A8/I8/I32 M16N16 direct-device")) {
        bit = 1u << 8;
    } else if (strstr(text, "M5 MPP TensorOps exact A8/I8/I32 M32N16 direct-device")) {
        bit = 1u << 9;
    } else if (strstr(text, "B2 shared-weight")) {
        bit = 1u << 10;
    } else if (strstr(text, "B4 shared-weight")) {
        bit = 1u << 11;
    }
    if (bit != 0) {
        marker->tile_mask.fetch_or(bit, std::memory_order_relaxed);
        if (strstr(text, "layout=m32k256_pair2_split8_v2") && strstr(text, "act_rows=512 ") &&
            strstr(text, "M32N128 BK128 device-preexpand") && strstr(text, "blocked-O128-K32")) {
            marker->pair2_device_preexpand.store(true, std::memory_order_relaxed);
        }
    }
}

static bool run_row4_swiglu_backend(std::vector<float> &          output,
                                    ggml_backend_t                backend,
                                    const std::vector<float> &    input,
                                    const std::vector<uint8_t> &  codes,
                                    const std::vector<uint16_t> & scales,
                                    int64_t                       tokens,
                                    bool                          qat,
                                    bool *                        gate_up_fusion_hit = nullptr) {
    constexpr int64_t K         = 128;
    constexpr int64_t N_FF      = 128;
    constexpr int64_t GATE_UP_O = 2 * N_FF;

    const ggml_init_params params = {
        /*.mem_size   =*/2 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, tokens);
    ggml_tensor * row4_codes =
        ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES, 64, 4, K / ROW4_TILE_K, GATE_UP_O / ROW4_TILE_O);
    ggml_tensor * row4_scales = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, GATE_UP_O);
    ggml_tensor * gate_up     = ggml_row4_linear(ctx, x, row4_codes, row4_scales, GATE_UP_O, K);
    ggml_tensor * gate        = ggml_view_2d(ctx, gate_up, N_FF, tokens, gate_up->nb[1], 0);
    ggml_tensor * up          = ggml_view_2d(ctx, gate_up, N_FF, tokens, gate_up->nb[1], N_FF * sizeof(float));
    ggml_tensor * silu        = ggml_fairy2i_silu_exact(ctx, gate);
    ggml_fairy2i_exact_set_qat(silu, qat);
    ggml_tensor * swiglu = ggml_fairy2i_mul_exact(ctx, silu, up);
    ggml_fairy2i_exact_set_qat(swiglu, qat);

    bool          ok    = ggml_backend_supports_op(backend, gate_up) && ggml_backend_supports_op(backend, silu) &&
                          ggml_backend_supports_op(backend, swiglu);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, swiglu);

    int silu_idx = -1;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (ggml_graph_node(graph, i) == silu) {
            silu_idx = i;
            break;
        }
    }
    ok = silu_idx >= 0 && silu_idx + 2 < ggml_graph_n_nodes(graph) && ggml_graph_node(graph, silu_idx + 1) == up &&
         ggml_graph_node(graph, silu_idx + 2) == swiglu && ok;
    if (!ok) {
        fprintf(stderr, "Row4 SwiGLU test graph does not contain SILU, VIEW(up), MUL adjacency\n");
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(row4_codes, codes.data(), 0, codes.size());
    ggml_backend_tensor_set(row4_scales, scales.data(), 0, scales.size() * sizeof(uint16_t));

    gate_up_fusion_log_marker marker;
    if (gate_up_fusion_hit != nullptr) {
        ggml_log_set(gate_up_fusion_log_callback, &marker);
    }
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (gate_up_fusion_hit != nullptr) {
        *gate_up_fusion_hit = marker.hit.load(std::memory_order_relaxed);
        ggml_log_set(nullptr, nullptr);
    }
    if (status == GGML_STATUS_SUCCESS) {
        output.resize((size_t) N_FF * (size_t) tokens);
        ggml_backend_tensor_get(swiglu, output.data(), 0, output.size() * sizeof(float));
    } else {
        fprintf(stderr, "Row4 SwiGLU graph compute failed: %s\n", ggml_status_to_string(status));
        ok = false;
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return status == GGML_STATUS_SUCCESS && ok;
}

static std::vector<float> oracle_row4_qat_swiglu_down(const std::vector<float> &    input,
                                                      const std::vector<uint8_t> &  gate_up_codes,
                                                      const std::vector<uint16_t> & gate_up_scales,
                                                      const std::vector<uint8_t> &  down_codes,
                                                      const std::vector<uint16_t> & down_scales,
                                                      int64_t                       input_k,
                                                      int64_t                       n_ff,
                                                      int64_t                       down_o,
                                                      int64_t                       tokens) {
    const std::vector<float> gate_up =
        oracle_row4_linear(input, gate_up_codes, gate_up_scales, 2 * n_ff, input_k, tokens);
    std::vector<float> swiglu((size_t) n_ff * (size_t) tokens);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t col = 0; col < n_ff; ++col) {
            const float gate = gate_up[(size_t) token * (size_t) (2 * n_ff) + (size_t) col];
            const float up   = gate_up[(size_t) token * (size_t) (2 * n_ff) + (size_t) n_ff + (size_t) col];
            float       silu;
            if (signbit(gate)) {
                const float exp_value = expf(gate);
                silu                  = gate * exp_value / (1.0f + exp_value);
            } else {
                silu = gate / (1.0f + expf(-gate));
            }
            const float silu_bf16                                 = oracle_bf16_round(silu);
            swiglu[(size_t) token * (size_t) n_ff + (size_t) col] = oracle_bf16_round(silu_bf16 * up);
        }
    }
    return oracle_row4_linear(swiglu, down_codes, down_scales, down_o, n_ff, tokens);
}

static bool run_row4_swiglu_down_backend(std::vector<float> &          output,
                                         ggml_backend_t                backend,
                                         const std::vector<float> &    input,
                                         const std::vector<uint8_t> &  gate_up_codes_data,
                                         const std::vector<uint16_t> & gate_up_scales_data,
                                         const std::vector<uint8_t> &  down_codes_data,
                                         const std::vector<uint16_t> & down_scales_data,
                                         int64_t                       tokens,
                                         bool                          qat,
                                         bool                          mark_mul_output,
                                         bool                          w8_down                 = false,
                                         bool *                        gate_up_fusion_hit      = nullptr,
                                         bool                          mark_gate_up_output     = false,
                                         bool                          pair2                   = false,
                                         bool *                        m5_tensorops_bypass_hit = nullptr) {
    const int64_t INPUT_K   = pair2 ? 256 : 128;
    const int64_t N_FF      = pair2 ? 256 : 128;
    const int64_t GATE_UP_O = 2 * N_FF;
    const int64_t DOWN_O    = N_FF;

    const ggml_init_params params = {
        /*.mem_size   =*/4 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, INPUT_K, tokens);
    ggml_tensor * gate_up_codes =
        pair2 ? ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES_PAIR2, 128, 8, INPUT_K / ROW4_PAIR2_TILE_K,
                                   GATE_UP_O / ROW4_PAIR2_TILE_O) :
                ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES, 64, 4, INPUT_K / ROW4_TILE_K, GATE_UP_O / ROW4_TILE_O);
    ggml_tensor * gate_up_scales = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, GATE_UP_O);
    ggml_tensor * gate_up        = ggml_row4_linear(ctx, x, gate_up_codes, gate_up_scales, GATE_UP_O, INPUT_K);
    if (mark_gate_up_output) {
        ggml_set_output(gate_up);
    }
    ggml_tensor * gate           = ggml_view_2d(ctx, gate_up, N_FF, tokens, gate_up->nb[1], 0);
    ggml_tensor * up             = ggml_view_2d(ctx, gate_up, N_FF, tokens, gate_up->nb[1], N_FF * sizeof(float));
    ggml_tensor * silu           = ggml_fairy2i_silu_exact(ctx, gate);
    ggml_fairy2i_exact_set_qat(silu, qat);
    ggml_tensor * mul = ggml_fairy2i_mul_exact(ctx, silu, up);
    ggml_fairy2i_exact_set_qat(mul, qat);
    if (mark_mul_output) {
        ggml_set_output(mul);
    }

    ggml_tensor *       down_codes;
    ggml_tensor *       down_scales;
    ggml_tensor *       down;
    std::vector<int8_t> w8_codes_data;
    std::vector<float>  w8_scales_data;
    if (w8_down) {
        down_codes     = ggml_new_tensor_4d(ctx, GGML_TYPE_I8, 128, 16, N_FF / ROW4_TILE_K, DOWN_O / ROW4_TILE_O);
        down_scales    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, DOWN_O);
        down           = ggml_w8a8_linear(ctx, mul, down_codes, down_scales, DOWN_O, N_FF);
        w8_codes_data  = make_w8_codes(DOWN_O, N_FF);
        w8_scales_data = make_w8_scales(DOWN_O);
    } else {
        down_codes = pair2 ?
                         ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES_PAIR2, 128, 8, N_FF / ROW4_PAIR2_TILE_K,
                                            DOWN_O / ROW4_PAIR2_TILE_O) :
                         ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES, 64, 4, N_FF / ROW4_TILE_K, DOWN_O / ROW4_TILE_O);
        down_scales = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, DOWN_O);
        down        = ggml_row4_linear(ctx, mul, down_codes, down_scales, DOWN_O, N_FF);
    }

    bool          ok    = ggml_backend_supports_op(backend, gate_up) && ggml_backend_supports_op(backend, silu) &&
                          ggml_backend_supports_op(backend, mul) && ggml_backend_supports_op(backend, down);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, down);

    int silu_idx = -1;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (ggml_graph_node(graph, i) == silu) {
            silu_idx = i;
            break;
        }
    }
    ok = silu_idx >= 0 && silu_idx + 3 < ggml_graph_n_nodes(graph) && ggml_graph_node(graph, silu_idx + 1) == up &&
         ggml_graph_node(graph, silu_idx + 2) == mul && ggml_graph_node(graph, silu_idx + 3) == down && ok;
    if (!ok) {
        fprintf(stderr, "Row4 SwiGLU-down test graph does not contain SILU, VIEW(up), MUL, ROW4 adjacency\n");
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(gate_up_codes, gate_up_codes_data.data(), 0, gate_up_codes_data.size());
    ggml_backend_tensor_set(gate_up_scales, gate_up_scales_data.data(), 0,
                            gate_up_scales_data.size() * sizeof(uint16_t));
    if (w8_down) {
        ggml_backend_tensor_set(down_codes, w8_codes_data.data(), 0, w8_codes_data.size());
        ggml_backend_tensor_set(down_scales, w8_scales_data.data(), 0, w8_scales_data.size() * sizeof(float));
    } else {
        ggml_backend_tensor_set(down_codes, down_codes_data.data(), 0, down_codes_data.size());
        ggml_backend_tensor_set(down_scales, down_scales_data.data(), 0, down_scales_data.size() * sizeof(uint16_t));
    }

    gate_up_fusion_log_marker marker;
    if (gate_up_fusion_hit != nullptr || m5_tensorops_bypass_hit != nullptr) {
        ggml_log_set(gate_up_fusion_log_callback, &marker);
    }
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (gate_up_fusion_hit != nullptr || m5_tensorops_bypass_hit != nullptr) {
        if (gate_up_fusion_hit != nullptr) {
            *gate_up_fusion_hit = marker.hit.load(std::memory_order_relaxed);
        }
        if (m5_tensorops_bypass_hit != nullptr) {
            *m5_tensorops_bypass_hit = marker.m5_tensorops_bypass_hit.load(std::memory_order_relaxed);
        }
        ggml_log_set(nullptr, nullptr);
    }
    if (status == GGML_STATUS_SUCCESS) {
        output.resize((size_t) DOWN_O * (size_t) tokens);
        ggml_backend_tensor_get(down, output.data(), 0, output.size() * sizeof(float));
    } else {
        fprintf(stderr, "Row4 SwiGLU-down graph compute failed: %s\n", ggml_status_to_string(status));
        ok = false;
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return status == GGML_STATUS_SUCCESS && ok;
}

enum class rms_row4_gate : uint8_t {
    normal,
    output,
    extra_use,
    split,
    strided,
    repeated_weight,
    nonqat,
    aliased_weight
};

static void rms_row4_fusion_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    if (strstr(text, "fuse QAT FAIRY2I_RMS_NORM_EXACT + ROW4_LINEAR activation quantization")) {
        static_cast<std::atomic<bool> *>(user_data)->store(true, std::memory_order_relaxed);
    }
}

static bool run_rms_row4_backend(ggml_backend_t       backend,
                                 int64_t              k,
                                 int64_t              tokens,
                                 bool                 pair2,
                                 rms_row4_gate        gate,
                                 std::vector<float> & norm,
                                 std::vector<float> & output,
                                 bool &               fusion_hit) {
    constexpr int64_t      O      = 128;
    const ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context *         ctx    = ggml_init(params);
    if (!ctx) {
        return false;
    }
    const bool    strided = gate == rms_row4_gate::strided;
    ggml_tensor * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k + (strided ? 4 : 0), tokens);
    ggml_tensor * x       = strided ? ggml_view_2d(ctx, storage, k, tokens, storage->nb[1], 0) : storage;
    ggml_tensor * weight  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, gate == rms_row4_gate::repeated_weight ? 2 : 1);
    ggml_tensor * rms     = ggml_fairy2i_rms_norm_exact(ctx, x, weight, 1.0e-6f);
    ggml_fairy2i_exact_set_qat(rms, gate != rms_row4_gate::nonqat);
    if (gate == rms_row4_gate::output) {
        ggml_set_output(rms);
    }
    ggml_tensor * codes  = pair2 ? ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES_PAIR2, 128, 8, k / 256, O / 32) :
                                   ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES, 64, 4, k / 128, O / 16);
    ggml_tensor * scales = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, O);
    ggml_tensor * linear = ggml_row4_linear(ctx, rms, codes, scales, O, k);
    ggml_cgraph * graph  = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, linear);
    ggml_tensor * extra = gate == rms_row4_gate::extra_use ? ggml_add(ctx, rms, rms) : nullptr;
    if (extra) {
        ggml_build_forward_expand(graph, extra);
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }
    if (gate == rms_row4_gate::aliased_weight) {
        // The RMS weight remains live while the linear's INT4 scratch is
        // otherwise reusable. Quantization may fuse, but expansion must wait.
        const size_t output_pad = (ggml_nbytes(linear) + 63) & ~size_t(63);
        const size_t quant_end  = output_pad + tokens * (k + sizeof(float));
        const size_t offset     = quant_end + (64 - ((uintptr_t) linear->data + quant_end) % 64) % 64;
        void *       address    = (char *) linear->data + offset;
        weight->buffer          = nullptr;
        weight->data            = nullptr;
        if (ggml_backend_tensor_alloc(buffer, weight, address) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            return false;
        }
    }
    std::vector<float> input = make_input(k, tokens);
    // Distinct rows, zero rows, signed zero and tiny rows exercise the row
    // offsets, scale floor and BF16 boundaries through the real fused graph.
    for (int64_t row = 0; row < tokens; ++row) {
        for (int64_t col = 0; col < k; ++col) {
            if (row % 17 == 0) {
                input[(size_t) row * k + col] = col % 2 ? -0.0f : 0.0f;
            } else if (row % 17 == 1) {
                input[(size_t) row * k + col] *= 0x1p-30f;
            }
        }
        ggml_backend_tensor_set(storage, input.data() + row * k, row * storage->nb[1], k * sizeof(float));
    }
    std::vector<float> weights((size_t) ggml_nelements(weight));
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = (float) ((int) (i % 67) - 33) / 32.0f + ((i & 1) ? 0x1p-18f : -0x1p-18f);
    }
    const auto packed     = pair2 ? make_row4_pair2_codes(O, k) : make_row4_codes(O, k);
    const auto scale_bits = make_row4_scales(O);
    ggml_backend_tensor_set(weight, weights.data(), 0, weights.size() * sizeof(float));
    ggml_backend_tensor_set(codes, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(scales, scale_bits.data(), 0, scale_bits.size() * sizeof(uint16_t));
    std::atomic<bool> marker{ false };
    ggml_log_set(rms_row4_fusion_log_callback, &marker);
    ggml_status status;
    if (gate == rms_row4_gate::split) {
        ggml_cgraph * first  = ggml_new_graph(ctx);
        ggml_cgraph * second = ggml_new_graph(ctx);
        ggml_build_forward_expand(first, rms);
        ggml_graph_add_node(second, linear);
        status = ggml_backend_graph_compute(backend, first);
        if (status == GGML_STATUS_SUCCESS) {
            status = ggml_backend_graph_compute(backend, second);
        }
    } else {
        status = ggml_backend_graph_compute(backend, graph);
    }
    ggml_log_set(nullptr, nullptr);
    fusion_hit = marker.load(std::memory_order_relaxed);
    bool ok    = status == GGML_STATUS_SUCCESS;
    if (ok) {
        norm.resize((size_t) k * tokens);
        output.resize((size_t) O * tokens);
        ggml_backend_tensor_get(rms, norm.data(), 0, norm.size() * sizeof(float));
        ggml_backend_tensor_get(linear, output.data(), 0, output.size() * sizeof(float));
        if (extra) {
            std::vector<float> actual(norm.size());
            std::vector<float> expected(norm.size());
            ggml_backend_tensor_get(extra, actual.data(), 0, actual.size() * sizeof(float));
            for (size_t i = 0; i < norm.size(); ++i) {
                expected[i] = norm[i] + norm[i];
            }
            ok = compare_exact("RMS Row4 extra consumer", actual, expected) && ok;
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

static bool test_metal_rms_row4_fusion() {
    ggml_backend_dev_t dev = find_metal_device();
    if (!dev) {
        const char * required = getenv("LLAMA_ROW4_REQUIRE_METAL_TESTS");
        printf("  Metal RMSNorm/Row4 fusion: SKIP (Metal backend unavailable)\n");
        return !required || strcmp(required, "0") == 0;
    }
    scoped_env_var disable("GGML_METAL_FUSION_DISABLE");
    scoped_env_var debug("GGML_METAL_FUSION_DEBUG");
    disable.set("1");
    ggml_backend_t unfused = ggml_backend_dev_init(dev, nullptr);
    disable.unset();
    debug.set("2");
    ggml_backend_t fused = ggml_backend_dev_init(dev, nullptr);
    if (!unfused || !fused) {
        ggml_backend_free(unfused);
        ggml_backend_free(fused);
        return false;
    }
    const char * strict     = getenv("LLAMA_ROW4_REQUIRE_M5_TENSOROPS_TESTS");
    const bool   require_m5 = strict && strcmp(strict, "0") != 0;

    struct test_case {
        int64_t       k, tokens;
        bool          pair2;
        rms_row4_gate gate;
    };

    const test_case cases[] = {
        { 4096, 1,    true,  rms_row4_gate::normal          },
        { 4096, 16,   true,  rms_row4_gate::normal          },
        { 4096, 31,   true,  rms_row4_gate::normal          },
        { 4096, 32,   true,  rms_row4_gate::normal          },
        { 4096, 33,   true,  rms_row4_gate::normal          },
        { 4096, 64,   true,  rms_row4_gate::normal          },
        { 4096, 128,  true,  rms_row4_gate::normal          },
        { 4096, 256,  true,  rms_row4_gate::normal          },
        { 4096, 512,  true,  rms_row4_gate::normal          },
        { 4096, 512,  true,  rms_row4_gate::aliased_weight  },
        { 4096, 544,  true,  rms_row4_gate::normal          },
        { 4096, 2048, true,  rms_row4_gate::normal          },
        { 4096, 32,   false, rms_row4_gate::normal          },
        { 512,  32,   true,  rms_row4_gate::normal          },
        { 4096, 32,   true,  rms_row4_gate::output          },
        { 4096, 32,   true,  rms_row4_gate::extra_use       },
        { 4096, 32,   true,  rms_row4_gate::split           },
        { 4096, 32,   true,  rms_row4_gate::strided         },
        { 4096, 32,   true,  rms_row4_gate::repeated_weight },
        { 4096, 32,   true,  rms_row4_gate::nonqat          },
    };
    bool ok = true;
    for (const auto & c : cases) {
        if (c.gate == rms_row4_gate::aliased_weight && !require_m5) {
            continue;
        }

        std::vector<float> norm_ref;
        std::vector<float> linear_ref;
        std::vector<float> norm;
        std::vector<float> linear;
        bool               unfused_hit = false;
        bool               fused_hit   = false;
        const std::string  label       = "RMS/Row4 B=" + std::to_string(c.tokens) + " K=" + std::to_string(c.k) +
                                         " gate=" + std::to_string((int) c.gate);
        const bool         ran =
            run_rms_row4_backend(unfused, c.k, c.tokens, c.pair2, c.gate, norm_ref, linear_ref, unfused_hit) &&
            run_rms_row4_backend(fused, c.k, c.tokens, c.pair2, c.gate, norm, linear, fused_hit);
        ok                  = ran && compare_exact((label + " RMS carriers").c_str(), norm, norm_ref) &&
                              compare_exact((label + " linear").c_str(), linear, linear_ref) && ok;
        const bool eligible = (c.gate == rms_row4_gate::normal || c.gate == rms_row4_gate::aliased_weight) &&
                              c.k == 4096 && (c.tokens == 1 || c.tokens % 32 == 0);
        if (unfused_hit || (fused_hit && !eligible) || (eligible && (require_m5 || c.tokens == 1) && !fused_hit)) {
            fprintf(stderr, "%s unexpected fusion path: disabled=%d enabled=%d eligible=%d\n", label.c_str(),
                    unfused_hit, fused_hit, eligible);
            ok = false;
        }
        // Independent Row4 oracle on small cases; larger tiles also compare
        // every RMS and linear output against the standalone Metal graph.
        if (ran && c.tokens <= 33) {
            const auto expected =
                oracle_row4_linear(norm_ref, make_row4_codes(128, c.k), make_row4_scales(128), 128, c.k, c.tokens);
            ok = compare_exact((label + " Row4 oracle").c_str(), linear, expected) && ok;
        }
    }
    ggml_backend_free(unfused);
    ggml_backend_free(fused);
    printf("  Metal RMSNorm/Row4 fusion: carriers, A8 oracle, tile/tail/graph gates - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static void row4_lookahead_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    if (strstr(text, "Row4 M5 preexpand lookahead")) {
        static_cast<std::atomic<bool> *>(user_data)->store(true, std::memory_order_relaxed);
    }
}

static bool test_metal_row4_preexpand_lookahead() {
    const char * strict = getenv("LLAMA_ROW4_REQUIRE_M5_TENSOROPS_TESTS");
    if (!strict || strcmp(strict, "0") == 0) {
        printf("  Metal Row4 preexpand lookahead: SKIP (strict M5 suite only)\n");
        return true;
    }
    ggml_backend_dev_t dev = find_metal_device();
    if (!dev) {
        return false;
    }
    scoped_env_var concurrency("GGML_METAL_CONCURRENCY_DISABLE");
    scoped_env_var debug("GGML_METAL_FUSION_DEBUG");
    debug.set("2");
    bool ok = true;
    enum class mode : uint8_t { concurrent, serial, alias, split };
    for (const auto test_mode : { mode::concurrent, mode::serial, mode::alias, mode::split }) {
        if (test_mode == mode::serial) {
            concurrency.set("1");
        } else {
            concurrency.unset();
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            return false;
        }
        for (int64_t tokens : { 512, 544 }) {
            constexpr int64_t      K       = 256;
            constexpr int64_t      O       = 128;
            const ggml_init_params params  = { 1024 * 1024, nullptr, true };
            ggml_context *         ctx     = ggml_init(params);
            ggml_tensor *          x       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, tokens);
            ggml_tensor *          codes1  = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES_PAIR2, 128, 8, 1, K / 32);
            ggml_tensor *          scales1 = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, K);
            ggml_tensor *          first   = ggml_row4_linear(ctx, x, codes1, scales1, K, K);
            ggml_tensor *          codes2  = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES_PAIR2, 128, 8, 1, O / 32);
            ggml_tensor *          scales2 = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, O);
            ggml_tensor *          second  = ggml_row4_linear(ctx, first, codes2, scales2, O, K);
            ggml_tensor *          live    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
            ggml_tensor *          saved   = ggml_dup(ctx, live);
            ggml_cgraph *          graph   = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, first);
            ggml_build_forward_expand(graph, saved);
            ggml_build_forward_expand(graph, second);
            ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buffer) {
                ggml_free(ctx);
                ggml_backend_free(backend);
                return false;
            }
            if (test_mode == mode::alias) {
                // A live value occupies the future INT4 region until saved is
                // computed. Looking only at the two linear nodes misses it.
                const size_t output_pad = (ggml_nbytes(second) + 63) & ~size_t(63);
                const size_t quant_end  = output_pad + tokens * (K + sizeof(float));
                const size_t offset     = quant_end + (64 - ((uintptr_t) second->data + quant_end) % 64) % 64;
                void *       address    = (char *) second->data + offset;
                live->buffer            = nullptr;
                live->data              = nullptr;
                ok                      = ggml_backend_tensor_alloc(buffer, live, address) == GGML_STATUS_SUCCESS && ok;
            }
            const auto input          = make_input(K, tokens);
            const auto packed1        = make_row4_pair2_codes(K, K);
            auto       packed2        = make_row4_pair2_codes(O, K);
            auto       oracle_codes2  = make_row4_codes(O, K);
            const auto scale1         = make_row4_scales(K);
            const auto scale2         = make_row4_scales(O);
            const auto live_input     = make_input(K, 1);
            const auto first_expected = oracle_row4_linear(input, make_row4_codes(K, K), scale1, K, K, tokens);
            ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
            ggml_backend_tensor_set(codes1, packed1.data(), 0, packed1.size());
            ggml_backend_tensor_set(scales1, scale1.data(), 0, ggml_nbytes(scales1));
            ggml_backend_tensor_set(scales2, scale2.data(), 0, ggml_nbytes(scales2));
            for (int iteration = 0; iteration < 2; ++iteration) {
                // A new graph compute must re-expand updated weights, even
                // when tensor addresses and scratch are unchanged.
                if (iteration) {
                    for (auto & value : packed2) {
                        value ^= 0xff;
                    }
                    for (auto & value : oracle_codes2) {
                        value ^= 0xff;
                    }
                }
                ggml_backend_tensor_set(codes2, packed2.data(), 0, packed2.size());
                ggml_backend_tensor_set(live, live_input.data(), 0, ggml_nbytes(live));
                std::atomic<bool> hit{ false };
                ggml_log_set(row4_lookahead_log_callback, &hit);
                ggml_status status;
                if (test_mode == mode::split) {
                    ggml_cgraph * a = ggml_new_graph(ctx);
                    ggml_cgraph * b = ggml_new_graph(ctx);
                    ggml_build_forward_expand(a, first);
                    ggml_build_forward_expand(a, saved);
                    ggml_graph_add_node(b, second);
                    status = ggml_backend_graph_compute(backend, a);
                    if (status == GGML_STATUS_SUCCESS) {
                        status = ggml_backend_graph_compute(backend, b);
                    }
                } else {
                    status = ggml_backend_graph_compute(backend, graph);
                }
                ggml_log_set(nullptr, nullptr);
                const bool expected_hit = test_mode == mode::concurrent;
                if (status != GGML_STATUS_SUCCESS || hit.load(std::memory_order_relaxed) != expected_hit) {
                    fprintf(stderr, "Row4 lookahead B=%lld mode=%d iteration=%d status=%d hit=%d expected=%d\n",
                            (long long) tokens, (int) test_mode, iteration, (int) status,
                            hit.load(std::memory_order_relaxed), expected_hit);
                    ok = false;
                }
                std::vector<float> actual((size_t) O * tokens);
                std::vector<float> saved_actual((size_t) K);
                ggml_backend_tensor_get(second, actual.data(), 0, ggml_nbytes(second));
                ggml_backend_tensor_get(saved, saved_actual.data(), 0, ggml_nbytes(saved));
                const auto expected = oracle_row4_linear(first_expected, oracle_codes2, scale2, O, K, tokens);
                ok                  = compare_exact("Row4 lookahead output", actual, expected) &&
                                      compare_exact("Row4 lookahead live scratch", saved_actual, live_input) && ok;
            }
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
        ggml_backend_free(backend);
    }
    printf("  Metal Row4 preexpand lookahead: exact output, live scratch, serial/split, updated weights - %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_metal_row4_swiglu_fusion() {
    ggml_backend_dev_t dev = find_metal_device();
    if (!dev) {
        const char * required = getenv("LLAMA_ROW4_REQUIRE_METAL_TESTS");
        if (required && strcmp(required, "0") != 0) {
            fprintf(stderr, "Row4 SwiGLU fusion test requires Metal, but no Metal device is available\n");
            return false;
        }
        printf("  Metal Row4 QAT SwiGLU fusion: SKIP (Metal backend unavailable)\n");
        return true;
    }

    constexpr int64_t           K         = 128;
    constexpr int64_t           GATE_UP_O = 256;
    const std::vector<uint8_t>  codes     = make_row4_codes(GATE_UP_O, K);
    const std::vector<uint16_t> scales    = make_row4_scales(GATE_UP_O);

    scoped_env_var fusion_disable("GGML_METAL_FUSION_DISABLE");
    scoped_env_var fusion_debug("GGML_METAL_FUSION_DEBUG");
    bool           ok = true;

    for (int64_t tokens : { 1, 3, 32 }) {
        const std::vector<float> input = make_input(K, tokens);

        ggml_backend_t cpu = ggml_backend_cpu_init();
        if (!cpu) {
            return false;
        }
        ggml_backend_cpu_set_n_threads(cpu, 4);
        std::vector<float> expected;
        ok = run_row4_swiglu_backend(expected, cpu, input, codes, scales, tokens, true) && ok;
        ggml_backend_free(cpu);

        fusion_disable.set("1");
        fusion_debug.unset();
        ggml_backend_t     metal_unfused = ggml_backend_dev_init(dev, nullptr);
        std::vector<float> unfused;
        ok = metal_unfused && run_row4_swiglu_backend(unfused, metal_unfused, input, codes, scales, tokens, true) &&
             compare_exact(("Row4 QAT SwiGLU unfused B=" + std::to_string(tokens)).c_str(), unfused, expected) && ok;
        if (metal_unfused) {
            ggml_backend_free(metal_unfused);
        }

        fusion_disable.unset();
        fusion_debug.set("2");
        ggml_backend_t     metal_fused = ggml_backend_dev_init(dev, nullptr);
        std::vector<float> fused;
        bool               gate_up_fusion_hit = false;
        ok = metal_fused &&
             run_row4_swiglu_backend(fused, metal_fused, input, codes, scales, tokens, true, &gate_up_fusion_hit) &&
             compare_exact(("Row4 QAT SwiGLU fused B=" + std::to_string(tokens)).c_str(), fused, expected) &&
             compare_exact(("Row4 QAT SwiGLU fused/unfused B=" + std::to_string(tokens)).c_str(), fused, unfused) && ok;
        if (gate_up_fusion_hit) {
            fprintf(stderr, "Row4 gate-up producer fusion ran without an adjacent down projection at B=%lld\n",
                    (long long) tokens);
            ok = false;
        }
        if (metal_fused) {
            ggml_backend_free(metal_fused);
        }
    }

    // The specialized route must not change the non-QAT exact implementation.
    {
        constexpr int64_t        TOKENS = 3;
        const std::vector<float> input  = make_input(K, TOKENS);

        fusion_disable.set("1");
        fusion_debug.unset();
        ggml_backend_t     metal_unfused = ggml_backend_dev_init(dev, nullptr);
        std::vector<float> unfused;
        ok =
            metal_unfused && run_row4_swiglu_backend(unfused, metal_unfused, input, codes, scales, TOKENS, false) && ok;
        if (metal_unfused) {
            ggml_backend_free(metal_unfused);
        }

        fusion_disable.unset();
        ggml_backend_t     metal_nonqat = ggml_backend_dev_init(dev, nullptr);
        std::vector<float> nonqat;
        ok = metal_nonqat && run_row4_swiglu_backend(nonqat, metal_nonqat, input, codes, scales, TOKENS, false) &&
             compare_exact("Row4 non-QAT SwiGLU fusion gate", nonqat, unfused) && ok;
        if (metal_nonqat) {
            ggml_backend_free(metal_nonqat);
        }
    }

    printf("  Metal Row4 QAT SwiGLU fusion/non-QAT gate - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_metal_row4_swiglu_down_fusion() {
    ggml_backend_dev_t dev = find_metal_device();
    if (!dev) {
        const char * required = getenv("LLAMA_ROW4_REQUIRE_METAL_TESTS");
        if (required && strcmp(required, "0") != 0) {
            fprintf(stderr, "Row4 SwiGLU-down fusion test requires Metal, but no Metal device is available\n");
            return false;
        }
        printf("  Metal Row4 QAT SwiGLU-down packed fusion: SKIP (Metal backend unavailable)\n");
        return true;
    }

    constexpr int64_t           INPUT_K        = 128;
    constexpr int64_t           N_FF           = 128;
    constexpr int64_t           GATE_UP_O      = 2 * N_FF;
    constexpr int64_t           DOWN_O         = 128;
    const std::vector<uint8_t>  gate_up_codes  = make_row4_codes(GATE_UP_O, INPUT_K);
    const std::vector<uint16_t> gate_up_scales = make_row4_scales(GATE_UP_O);
    const std::vector<uint8_t>  down_codes     = make_row4_codes(DOWN_O, N_FF);
    const std::vector<uint16_t> down_scales    = make_row4_scales(DOWN_O);

    scoped_env_var fusion_disable("GGML_METAL_FUSION_DISABLE");
    scoped_env_var fusion_debug("GGML_METAL_FUSION_DEBUG");
    bool           ok = true;

    for (int64_t tokens : { 1, 3, 8, 9, 32, 64, 128, 256, 512 }) {
        const std::vector<float> input  = make_input(INPUT_K, tokens);
        const std::vector<float> oracle = oracle_row4_qat_swiglu_down(input, gate_up_codes, gate_up_scales, down_codes,
                                                                      down_scales, INPUT_K, N_FF, DOWN_O, tokens);

        ggml_backend_t cpu = ggml_backend_cpu_init();
        if (!cpu) {
            return false;
        }
        ggml_backend_cpu_set_n_threads(cpu, 4);
        std::vector<float> expected;
        ok = run_row4_swiglu_down_backend(expected, cpu, input, gate_up_codes, gate_up_scales, down_codes, down_scales,
                                          tokens, true, false) &&
             compare_exact(("Row4 QAT SwiGLU-down CPU/oracle B=" + std::to_string(tokens)).c_str(), expected, oracle) &&
             ok;
        ggml_backend_free(cpu);

        fusion_disable.set("1");
        fusion_debug.unset();
        ggml_backend_t     metal_unfused = ggml_backend_dev_init(dev, nullptr);
        std::vector<float> unfused;
        ok = metal_unfused &&
             run_row4_swiglu_down_backend(unfused, metal_unfused, input, gate_up_codes, gate_up_scales, down_codes,
                                          down_scales, tokens, true, false) &&
             compare_exact(("Row4 QAT SwiGLU-down unfused/oracle B=" + std::to_string(tokens)).c_str(), unfused,
                           oracle) &&
             ok;
        if (metal_unfused) {
            ggml_backend_free(metal_unfused);
        }

        fusion_disable.unset();
        fusion_debug.set("2");
        ggml_backend_t     metal_fused = ggml_backend_dev_init(dev, nullptr);
        std::vector<float> fused;
        bool               gate_up_fusion_hit      = false;
        bool               m5_tensorops_bypass_hit = false;
        ok =
            metal_fused &&
            run_row4_swiglu_down_backend(fused, metal_fused, input, gate_up_codes, gate_up_scales, down_codes,
                                         down_scales, tokens, true, false, false, &gate_up_fusion_hit, false, false,
                                         &m5_tensorops_bypass_hit) &&
            compare_exact(("Row4 QAT SwiGLU-down fused/oracle B=" + std::to_string(tokens)).c_str(), fused, oracle) &&
            compare_exact(("Row4 QAT SwiGLU-down fused/unfused B=" + std::to_string(tokens)).c_str(), fused, unfused) &&
            ok;
        const bool expect_gate_up_fusion = tokens > 8 && tokens % 32 == 0;
        const bool optimized_route_hit   = gate_up_fusion_hit != m5_tensorops_bypass_hit;
        if (optimized_route_hit != expect_gate_up_fusion) {
            fprintf(stderr,
                    "Row4 gate-up optimized route mismatch B=%lld: fusion=%d M5-TensorOps-bypass=%d expected=%d\n",
                    (long long) tokens, (int) gate_up_fusion_hit, (int) m5_tensorops_bypass_hit,
                    (int) expect_gate_up_fusion);
            ok = false;
        }
        if (metal_fused) {
            ggml_backend_free(metal_fused);
        }
    }

    // Schema-v2 must support both the ordinary chain and the prefill
    // gate/up producer fusion without converting the pair2 weights back to
    // v1. The oracle deliberately consumes a separately packed v1 copy.
    {
        constexpr int64_t           PAIR_INPUT_K         = 256;
        constexpr int64_t           PAIR_N_FF            = 256;
        constexpr int64_t           PAIR_GATE_UP_O       = 2 * PAIR_N_FF;
        constexpr int64_t           PAIR_DOWN_O          = PAIR_N_FF;
        const std::vector<uint8_t>  gate_up_v1           = make_row4_codes(PAIR_GATE_UP_O, PAIR_INPUT_K);
        const std::vector<uint8_t>  gate_up_pair2        = make_row4_pair2_codes(PAIR_GATE_UP_O, PAIR_INPUT_K);
        const std::vector<uint16_t> gate_up_pair2_scales = make_row4_scales(PAIR_GATE_UP_O);
        const std::vector<uint8_t>  down_v1              = make_row4_codes(PAIR_DOWN_O, PAIR_N_FF);
        const std::vector<uint8_t>  down_pair2           = make_row4_pair2_codes(PAIR_DOWN_O, PAIR_N_FF);
        const std::vector<uint16_t> down_pair2_scales    = make_row4_scales(PAIR_DOWN_O);

        for (int64_t tokens : { 1, 9, 32 }) {
            const std::vector<float> input = make_input(PAIR_INPUT_K, tokens);
            const std::vector<float> oracle =
                oracle_row4_qat_swiglu_down(input, gate_up_v1, gate_up_pair2_scales, down_v1, down_pair2_scales,
                                            PAIR_INPUT_K, PAIR_N_FF, PAIR_DOWN_O, tokens);

            fusion_disable.set("1");
            fusion_debug.unset();
            ggml_backend_t     metal_unfused = ggml_backend_dev_init(dev, nullptr);
            std::vector<float> unfused;
            ok = metal_unfused &&
                 run_row4_swiglu_down_backend(unfused, metal_unfused, input, gate_up_pair2, gate_up_pair2_scales,
                                              down_pair2, down_pair2_scales, tokens, true, false, false, nullptr, false,
                                              true) &&
                 compare_exact(("Row4 pair2 QAT SwiGLU-down unfused/oracle B=" + std::to_string(tokens)).c_str(),
                               unfused, oracle) &&
                 ok;
            if (metal_unfused) {
                ggml_backend_free(metal_unfused);
            }

            fusion_disable.unset();
            fusion_debug.set("2");
            ggml_backend_t     metal_fused = ggml_backend_dev_init(dev, nullptr);
            std::vector<float> fused;
            bool               gate_up_fusion_hit = false;
            bool               m5_tensorops_bypass_hit = false;
            ok = metal_fused &&
                 run_row4_swiglu_down_backend(fused, metal_fused, input, gate_up_pair2, gate_up_pair2_scales,
                                              down_pair2, down_pair2_scales, tokens, true, false, false,
                                              &gate_up_fusion_hit, false, true, &m5_tensorops_bypass_hit) &&
                 compare_exact(("Row4 pair2 QAT SwiGLU-down fused/oracle B=" + std::to_string(tokens)).c_str(), fused,
                               oracle) &&
                 compare_exact(("Row4 pair2 QAT SwiGLU-down fused/unfused B=" + std::to_string(tokens)).c_str(), fused,
                               unfused) &&
                 ok;
            const bool expect_gate_up_fusion = tokens > 8 && tokens % 32 == 0 && !m5_tensorops_bypass_hit;
            if (gate_up_fusion_hit != expect_gate_up_fusion) {
                fprintf(stderr, "Row4 pair2 gate-up producer fusion hit mismatch B=%lld: actual=%d expected=%d\n",
                        (long long) tokens, (int) gate_up_fusion_hit, (int) expect_gate_up_fusion);
                ok = false;
            }
            if (metal_fused) {
                ggml_backend_free(metal_fused);
            }
        }
    }

    // A requested gate-up result must preserve the F32 materialization. The
    // legacy SiLU-start fusion may still consume the later QAT/down chain.
    {
        constexpr int64_t        TOKENS = 32;
        const std::vector<float> input  = make_input(INPUT_K, TOKENS);

        fusion_disable.set("1");
        fusion_debug.unset();
        ggml_backend_t     metal_unfused = ggml_backend_dev_init(dev, nullptr);
        std::vector<float> unfused;
        ok = metal_unfused &&
             run_row4_swiglu_down_backend(unfused, metal_unfused, input, gate_up_codes, gate_up_scales, down_codes,
                                          down_scales, TOKENS, true, false) &&
             ok;
        if (metal_unfused) {
            ggml_backend_free(metal_unfused);
        }

        fusion_disable.unset();
        fusion_debug.set("2");
        ggml_backend_t     metal_gate_up_output = ggml_backend_dev_init(dev, nullptr);
        std::vector<float> gate_up_output;
        bool               gate_up_fusion_hit = false;
        ok = metal_gate_up_output &&
             run_row4_swiglu_down_backend(gate_up_output, metal_gate_up_output, input, gate_up_codes, gate_up_scales,
                                          down_codes, down_scales, TOKENS, true, false, false, &gate_up_fusion_hit,
                                          true) &&
             compare_exact("Row4 QAT gate-up-output producer fusion gate", gate_up_output, unfused) && ok;
        if (gate_up_fusion_hit) {
            fprintf(stderr, "Row4 gate-up producer fusion ignored requested gate-up output\n");
            ok = false;
        }
        if (metal_gate_up_output) {
            ggml_backend_free(metal_gate_up_output);
        }
    }

    // An explicitly requested MUL output must block the four-node handoff and
    // leave the existing three-node F32-carrier fusion as the fallback.
    {
        constexpr int64_t        TOKENS = 32;
        const std::vector<float> input  = make_input(INPUT_K, TOKENS);

        fusion_disable.set("1");
        fusion_debug.unset();
        ggml_backend_t     metal_unfused = ggml_backend_dev_init(dev, nullptr);
        std::vector<float> unfused;
        ok = metal_unfused &&
             run_row4_swiglu_down_backend(unfused, metal_unfused, input, gate_up_codes, gate_up_scales, down_codes,
                                          down_scales, TOKENS, true, false) &&
             ok;
        if (metal_unfused) {
            ggml_backend_free(metal_unfused);
        }

        fusion_disable.unset();
        fusion_debug.set("2");
        ggml_backend_t     metal_mul_output = ggml_backend_dev_init(dev, nullptr);
        std::vector<float> mul_output;
        bool               gate_up_fusion_hit = false;
        ok = metal_mul_output &&
             run_row4_swiglu_down_backend(mul_output, metal_mul_output, input, gate_up_codes, gate_up_scales,
                                          down_codes, down_scales, TOKENS, true, true, false, &gate_up_fusion_hit) &&
             compare_exact("Row4 QAT SwiGLU-down MUL-output fusion gate", mul_output, unfused) && ok;
        if (gate_up_fusion_hit) {
            fprintf(stderr, "Row4 gate-up producer fusion ignored requested MUL output\n");
            ok = false;
        }
        if (metal_mul_output) {
            ggml_backend_free(metal_mul_output);
        }
    }

    // A W8A8 consumer has the same activation shape but is outside this
    // Row4-only packed handoff. It must retain the three-node producer fusion
    // and the ordinary W8A8 activation quantizer.
    {
        constexpr int64_t        TOKENS = 32;
        const std::vector<float> input  = make_input(INPUT_K, TOKENS);

        fusion_disable.set("1");
        fusion_debug.unset();
        ggml_backend_t     metal_unfused = ggml_backend_dev_init(dev, nullptr);
        std::vector<float> unfused;
        ok = metal_unfused &&
             run_row4_swiglu_down_backend(unfused, metal_unfused, input, gate_up_codes, gate_up_scales, down_codes,
                                          down_scales, TOKENS, true, false, true) &&
             ok;
        if (metal_unfused) {
            ggml_backend_free(metal_unfused);
        }

        fusion_disable.unset();
        fusion_debug.set("2");
        ggml_backend_t     metal_w8_down = ggml_backend_dev_init(dev, nullptr);
        std::vector<float> w8_down;
        bool               gate_up_fusion_hit = false;
        ok = metal_w8_down &&
             run_row4_swiglu_down_backend(w8_down, metal_w8_down, input, gate_up_codes, gate_up_scales, down_codes,
                                          down_scales, TOKENS, true, false, true, &gate_up_fusion_hit) &&
             compare_exact("Row4 QAT SwiGLU W8A8-down fusion gate", w8_down, unfused) && ok;
        if (gate_up_fusion_hit) {
            fprintf(stderr, "Row4 gate-up producer fusion accepted a W8A8 down projection\n");
            ok = false;
        }
        if (metal_w8_down) {
            ggml_backend_free(metal_w8_down);
        }
    }

    // Non-QAT exact operations must not enter either specialized QAT route.
    {
        constexpr int64_t        TOKENS = 32;
        const std::vector<float> input  = make_input(INPUT_K, TOKENS);

        fusion_disable.set("1");
        fusion_debug.unset();
        ggml_backend_t     metal_unfused = ggml_backend_dev_init(dev, nullptr);
        std::vector<float> unfused;
        ok = metal_unfused &&
             run_row4_swiglu_down_backend(unfused, metal_unfused, input, gate_up_codes, gate_up_scales, down_codes,
                                          down_scales, TOKENS, false, false) &&
             ok;
        if (metal_unfused) {
            ggml_backend_free(metal_unfused);
        }

        fusion_disable.unset();
        fusion_debug.set("2");
        ggml_backend_t     metal_nonqat = ggml_backend_dev_init(dev, nullptr);
        std::vector<float> nonqat;
        bool               gate_up_fusion_hit = false;
        ok = metal_nonqat &&
             run_row4_swiglu_down_backend(nonqat, metal_nonqat, input, gate_up_codes, gate_up_scales, down_codes,
                                          down_scales, TOKENS, false, false, false, &gate_up_fusion_hit) &&
             compare_exact("Row4 non-QAT SwiGLU-down fusion gate", nonqat, unfused) && ok;
        if (gate_up_fusion_hit) {
            fprintf(stderr, "Row4 gate-up producer fusion accepted non-QAT elementwise operations\n");
            ok = false;
        }
        if (metal_nonqat) {
            ggml_backend_free(metal_nonqat);
        }
    }

    printf("  Metal Row4 QAT SwiGLU-down packed fusion/gates - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

void swiglu_a8_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    if (strstr(text, "fuse Row4 QAT SwiGLU + A8 quantization K12288")) {
        static_cast<std::atomic<bool> *>(user_data)->store(true, std::memory_order_relaxed);
    }
}

bool test_metal_row4_swiglu_a8_fusion() {
    const char * strict = getenv("LLAMA_ROW4_REQUIRE_M5_TENSOROPS_TESTS");
    if (!strict || strcmp(strict, "0") == 0) {
        printf("  Metal Row4 SwiGLU/A8 fusion: SKIP (strict M5 suite only)\n");
        return true;
    }
    ggml_backend_dev_t dev = find_metal_device();
    if (!dev) {
        return false;
    }
    scoped_env_var disable("GGML_METAL_FUSION_DISABLE");
    scoped_env_var debug("GGML_METAL_FUSION_DEBUG");
    disable.set("1");
    ggml_backend_t reference = ggml_backend_dev_init(dev, nullptr);
    disable.unset();
    debug.set("2");
    ggml_backend_t fused = ggml_backend_dev_init(dev, nullptr);
    if (!reference || !fused) {
        if (reference) {
            ggml_backend_free(reference);
        }
        if (fused) {
            ggml_backend_free(fused);
        }
        return false;
    }
    bool ok = true;
    enum class mode : uint8_t { normal, output, extra_use, split, alias };
    const std::pair<int64_t, mode> cases[] = {
        { 128, mode::normal    },
        { 512, mode::normal    },
        { 544, mode::normal    },
        { 512, mode::output    },
        { 512, mode::extra_use },
        { 512, mode::split     },
        { 512, mode::alias     },
    };
    for (const auto & test_case : cases) {
        constexpr int64_t      N_FF      = 12288;
        constexpr int64_t      O         = 128;
        const int64_t          tokens    = test_case.first;
        const mode             test_mode = test_case.second;
        const int64_t          k         = test_mode == mode::alias ? 4096 : 256;
        const ggml_init_params params    = { 2 * 1024 * 1024, nullptr, true };
        ggml_context *         ctx       = ggml_init(params);
        ggml_tensor *          x         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, tokens);
        ggml_tensor * codes1  = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES_PAIR2, 128, 8, k / 256, 2 * N_FF / 32);
        ggml_tensor * scales1 = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, 2 * N_FF);
        ggml_tensor * gate_up = ggml_row4_linear(ctx, x, codes1, scales1, 2 * N_FF, k);
        ggml_tensor * gate    = ggml_view_2d(ctx, gate_up, N_FF, tokens, gate_up->nb[1], 0);
        ggml_tensor * up      = ggml_view_2d(ctx, gate_up, N_FF, tokens, gate_up->nb[1], N_FF * sizeof(float));
        ggml_tensor * silu    = ggml_fairy2i_silu_exact(ctx, gate);
        ggml_fairy2i_exact_set_qat(silu, true);
        ggml_tensor * mul = ggml_fairy2i_mul_exact(ctx, silu, up);
        ggml_fairy2i_exact_set_qat(mul, true);
        if (test_mode == mode::output) {
            ggml_set_output(mul);
        }
        ggml_tensor * codes2  = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES_PAIR2, 128, 8, N_FF / 256, O / 32);
        ggml_tensor * scales2 = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, O);
        ggml_tensor * down    = ggml_row4_linear(ctx, mul, codes2, scales2, O, N_FF);
        ggml_tensor * extra   = test_mode == mode::extra_use ? ggml_dup(ctx, mul) : nullptr;
        ggml_cgraph * graph   = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, down);
        if (extra) {
            ggml_build_forward_expand(graph, extra);
        }
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, fused);
        if (!buffer) {
            ggml_free(ctx);
            ok = false;
            break;
        }
        if (test_mode == mode::alias) {
            // The ordinary packed handoff can reuse dead gate/up INT4, but
            // the new fusion deliberately rejects overlap with its backing
            // allocation. Its standalone fallback must still be exact.
            const size_t end     = ((ggml_nbytes(gate_up) + 63) & ~size_t(63)) + tokens * (k + sizeof(float));
            const size_t offset  = end + (64 - ((uintptr_t) gate_up->data + end) % 64) % 64;
            void *       address = (char *) gate_up->data + offset;
            mul->buffer          = nullptr;
            mul->data            = nullptr;
            ok                   = ggml_backend_tensor_alloc(buffer, mul, address) == GGML_STATUS_SUCCESS && ok;
        }
        auto input = make_input(k, tokens);
        for (int64_t row = 0; row < tokens; ++row) {
            for (int64_t col = 0; col < k; ++col) {
                if (row % 17 == 0) {
                    input[row * k + col] = col % 2 ? -0.0f : 0.0f;
                }
                if (row % 17 == 1) {
                    input[row * k + col] *= 0x1p-30f;
                }
            }
        }
        const auto packed1 = make_row4_pair2_codes(2 * N_FF, k);
        const auto scale1  = make_row4_scales(2 * N_FF);
        const auto packed2 = make_row4_pair2_codes(O, N_FF);
        const auto scale2  = make_row4_scales(O);
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(codes1, packed1.data(), 0, packed1.size());
        ggml_backend_tensor_set(scales1, scale1.data(), 0, ggml_nbytes(scales1));
        ggml_backend_tensor_set(codes2, packed2.data(), 0, packed2.size());
        ggml_backend_tensor_set(scales2, scale2.data(), 0, ggml_nbytes(scales2));
        std::vector<float> expected_mul(N_FF * tokens);
        std::vector<float> expected_output(O * tokens);
        for (bool expected : { true, false }) {
            std::atomic<bool> hit{ false };
            ggml_log_set(swiglu_a8_log_callback, &hit);
            ggml_status status;
            if (!expected && test_mode == mode::split) {
                ggml_cgraph * a = ggml_new_graph(ctx);
                ggml_cgraph * b = ggml_new_graph(ctx);
                ggml_build_forward_expand(a, mul);
                ggml_graph_add_node(b, down);
                status = ggml_backend_graph_compute(fused, a);
                if (status == GGML_STATUS_SUCCESS) {
                    status = ggml_backend_graph_compute(fused, b);
                }
            } else {
                status = ggml_backend_graph_compute(expected ? reference : fused, graph);
            }
            ggml_log_set(nullptr, nullptr);
            const bool expected_hit = !expected && test_mode == mode::normal && tokens >= 512;
            if (status != GGML_STATUS_SUCCESS || hit.load() != expected_hit) {
                fprintf(stderr, "SwiGLU/A8 B=%lld mode=%d reference=%d status=%d hit=%d expected=%d\n",
                        (long long) tokens, (int) test_mode, expected, (int) status, hit.load(), expected_hit);
                ok = false;
            }
            std::vector<float> output(O * tokens);
            ggml_backend_tensor_get(down, output.data(), 0, ggml_nbytes(down));
            if (expected) {
                ggml_backend_tensor_get(mul, expected_mul.data(), 0, ggml_nbytes(mul));
                expected_output = std::move(output);
            } else {
                ok = compare_exact("SwiGLU/A8 down output", output, expected_output) && ok;
                if (expected_hit) {
                    std::vector<int8_t> actual_q(N_FF * tokens);
                    std::vector<float>  actual_scales(tokens);
                    std::vector<float>  expected_scales(tokens);
                    ggml_backend_tensor_get(mul, actual_q.data(), 0, actual_q.size());
                    ggml_backend_tensor_get(mul, actual_scales.data(), actual_q.size(), tokens * sizeof(float));
                    for (int64_t row = 0; row < tokens; ++row) {
                        const auto q         = oracle_quantize_token(expected_mul.data() + row * N_FF, N_FF);
                        expected_scales[row] = q.scale;
                        if (memcmp(q.values.data(), actual_q.data() + row * N_FF, N_FF) != 0) {
                            fprintf(stderr, "SwiGLU/A8 integer mismatch B=%lld row=%lld\n", (long long) tokens,
                                    (long long) row);
                            ok = false;
                            break;
                        }
                    }
                    ok = compare_exact("SwiGLU/A8 scales", actual_scales, expected_scales) && ok;
                } else if (test_mode == mode::output || test_mode == mode::extra_use || test_mode == mode::split) {
                    std::vector<float> actual_mul(N_FF * tokens);
                    ggml_backend_tensor_get(extra ? extra : mul, actual_mul.data(), 0, ggml_nbytes(mul));
                    ok = compare_exact("SwiGLU/A8 observable MUL", actual_mul, expected_mul) && ok;
                }
            }
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    ggml_backend_free(fused);
    ggml_backend_free(reference);
    printf("  Metal Row4 SwiGLU/A8 fusion: integer oracle, scales, output/extra/split/alias gates - %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

struct row4_residual_outputs {
    std::vector<float> row4;
    std::vector<float> add;
};

struct row4_decode_production_log_marker {
    std::atomic<bool> hit = false;
};

static void row4_decode_production_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    row4_decode_production_log_marker * state = static_cast<row4_decode_production_log_marker *>(user_data);
    if (strstr(text, "ROW4 Metal W1A8 path: decode") && strstr(text, "layout=m32k256_pair2_split8_v2") &&
        strstr(text, "pipeline=lut16-const-rows-f32")) {
        state->hit.store(true, std::memory_order_relaxed);
    }
}

static bool test_metal_row4_pair2_decode_production() {
    ggml_backend_dev_t dev = find_metal_device();
    if (!dev) {
        const char * required = getenv("LLAMA_ROW4_REQUIRE_METAL_TESTS");
        if (required && strcmp(required, "0") != 0) {
            fprintf(stderr, "Pair2 production decode test requires Metal, but no Metal device is available\n");
            return false;
        }
        printf("  Metal Row4 Pair2 production decode: SKIP (Metal backend unavailable)\n");
        return true;
    }

    ggml_backend_t metal = ggml_backend_dev_init(dev, nullptr);
    if (!metal) {
        fprintf(stderr, "failed to initialize Metal backend for Pair2 production decode test\n");
        return false;
    }

    constexpr int64_t O  = 96;
    bool              ok = true;
    for (int64_t k : { 256, 4096, 12288 }) {
        const bool                 exhaustive = k == 256;
        const std::vector<float>   input      = exhaustive ? make_row4_lut16_exhaustive_input(k, 1) : make_input(k, 1);
        const std::vector<uint8_t> logical =
            exhaustive ? make_row4_lut16_exhaustive_logical(O, k) : std::vector<uint8_t>();
        const std::vector<uint8_t> v1_codes = exhaustive ? pack_row4_codes(logical, O, k) : make_row4_codes(O, k);
        const std::vector<uint8_t> pair2_codes =
            exhaustive ? pack_row4_pair2_codes(logical, O, k) : make_row4_pair2_codes(O, k);
        const std::vector<uint16_t> scales =
            exhaustive ? std::vector<uint16_t>((size_t) O, oracle_bf16_bits(1.0f)) : make_row4_scales(O);
        const std::vector<float> expected = oracle_row4_linear(input, v1_codes, scales, O, k, 1);

        row4_decode_production_log_marker production_marker;
        ggml_log_set(row4_decode_production_log_callback, &production_marker);
        std::vector<float> production;
        const bool production_run = run_operator_backend(production, linear_kind::row4_pair2, input, pair2_codes,
                                                         scales, {}, {}, O, k, 1, nullptr, false, metal);
        ggml_log_set(nullptr, nullptr);
        const bool production_hit = production_marker.hit.load(std::memory_order_relaxed);
        if (!production_hit) {
            fprintf(stderr, "Row4 Pair2 production decode marker missing for K=%lld\n", (long long) k);
        }
        ok = production_run &&
             compare_exact(("Row4 Pair2 production decode K=" + std::to_string(k)).c_str(), production, expected) &&
             production_hit && ok;
    }

    {
        constexpr int64_t    K = 12288;
        std::vector<uint8_t> logical((size_t) (O / 4) * K, 0);
        for (int64_t ik = 0; ik < K; ++ik) {
            logical[(size_t) K + (size_t) ik] = (uint8_t) ((ik & 1) ? 5 : 0);
        }
        const std::vector<float>    input((size_t) K, 1.0f);
        const std::vector<uint8_t>  v1_codes    = pack_row4_codes(logical, O, K);
        const std::vector<uint8_t>  pair2_codes = pack_row4_pair2_codes(logical, O, K);
        const std::vector<uint16_t> scales((size_t) O, oracle_bf16_bits(1.0f));
        const std::vector<float>    expected = oracle_row4_linear(input, v1_codes, scales, O, K, 1);
        std::vector<float>          actual;
        if (!run_operator_backend(actual, linear_kind::row4_pair2, input, pair2_codes, scales, {}, {}, O, K, 1, nullptr,
                                  false, metal) ||
            !compare_exact("Row4 Pair2 production K=12288 maximum/cancellation", actual, expected) ||
            f32_bits(expected[0]) != f32_bits(24576.0f) || f32_bits(expected[4]) != f32_bits(0.0f)) {
            fprintf(stderr, "Row4 Pair2 production maximum/cancellation guard failed\n");
            ok = false;
        }
    }

    ggml_backend_free(metal);
    printf("  Metal Row4 Pair2 production decode exact matrix - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

struct row4_residual_case {
    int64_t tokens         = 1;
    bool    qat            = true;
    bool    reverse        = false;
    bool    row4_output    = false;
    bool    extra_consumer = false;
    bool    graph_gap      = false;
};

struct row4_residual_fusion_log_marker {
    std::atomic<bool> hit               = false;
    bool              require_pair2     = false;
    const char *      required_pipeline = nullptr;
};

static void row4_residual_fusion_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    constexpr const char *            marker = "fuse Row4 decode staged epilogue + QAT COMPLEX_ADD residual";
    row4_residual_fusion_log_marker * state  = static_cast<row4_residual_fusion_log_marker *>(user_data);
    if (strstr(text, marker) && (!state->require_pair2 || strstr(text, "layout=m32k256_pair2_split8_v2")) &&
        (!state->required_pipeline || strstr(text, state->required_pipeline))) {
        state->hit.store(true, std::memory_order_relaxed);
    }
}

static bool run_row4_residual_backend(row4_residual_outputs &       output,
                                      ggml_backend_t                backend,
                                      const std::vector<float> &    input,
                                      const std::vector<uint8_t> &  codes_data,
                                      const std::vector<uint16_t> & scales_data,
                                      const std::vector<float> &    residual_data,
                                      int64_t                       o,
                                      int64_t                       k,
                                      const row4_residual_case &    test_case,
                                      bool                          pair2               = false,
                                      bool *                        residual_fusion_hit = nullptr,
                                      const char *                  required_pipeline   = nullptr) {
    const ggml_init_params params = {
        /*.mem_size   =*/4 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, test_case.tokens);
    ggml_tensor * codes =
        pair2 ?
            ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES_PAIR2, 128, 8, k / ROW4_PAIR2_TILE_K, o / ROW4_PAIR2_TILE_O) :
            ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES, 64, 4, k / ROW4_TILE_K, o / ROW4_TILE_O);
    ggml_tensor * scales   = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, o);
    ggml_tensor * row4     = ggml_row4_linear(ctx, x, codes, scales, o, k);
    ggml_tensor * residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, o, test_case.tokens);
    ggml_tensor * add =
        test_case.reverse ? ggml_complex_add(ctx, residual, row4) : ggml_complex_add(ctx, row4, residual);
    ggml_complex_add_set_qat(add, test_case.qat);
    if (test_case.row4_output) {
        ggml_set_output(row4);
    }

    ggml_tensor * extra = test_case.extra_consumer ? ggml_dup(ctx, row4) : nullptr;
    ggml_tensor * gap   = test_case.graph_gap ? ggml_dup(ctx, residual) : nullptr;
    ggml_cgraph * graph = ggml_new_graph(ctx);
    if (gap) {
        ggml_build_forward_expand(graph, row4);
        ggml_build_forward_expand(graph, gap);
    }
    ggml_build_forward_expand(graph, add);
    if (extra) {
        ggml_build_forward_expand(graph, extra);
    }

    bool ok = ggml_backend_supports_op(backend, row4) && ggml_backend_supports_op(backend, add) &&
              (!extra || ggml_backend_supports_op(backend, extra)) && (!gap || ggml_backend_supports_op(backend, gap));
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(codes, codes_data.data(), 0, codes_data.size());
    ggml_backend_tensor_set(scales, scales_data.data(), 0, scales_data.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(residual, residual_data.data(), 0, residual_data.size() * sizeof(float));

    row4_residual_fusion_log_marker marker;
    marker.require_pair2     = pair2;
    marker.required_pipeline = required_pipeline;
    if (residual_fusion_hit) {
        ggml_log_set(row4_residual_fusion_log_callback, &marker);
    }
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (residual_fusion_hit) {
        *residual_fusion_hit = marker.hit.load(std::memory_order_relaxed);
        ggml_log_set(nullptr, nullptr);
    }
    if (status == GGML_STATUS_SUCCESS) {
        output.row4.resize((size_t) o * (size_t) test_case.tokens);
        output.add.resize((size_t) o * (size_t) test_case.tokens);
        ggml_backend_tensor_get(row4, output.row4.data(), 0, output.row4.size() * sizeof(float));
        ggml_backend_tensor_get(add, output.add.data(), 0, output.add.size() * sizeof(float));
    } else {
        fprintf(stderr, "Row4 residual graph compute failed: %s\n", ggml_status_to_string(status));
        ok = false;
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return status == GGML_STATUS_SUCCESS && ok;
}

static std::vector<float> make_residual_carriers(const std::vector<float> & row4) {
    std::vector<float> residual(row4.size());
    for (size_t i = 0; i < row4.size(); ++i) {
        const uint16_t row4_imag = (uint16_t) (f32_bits(row4[i]) >> 16);
        uint16_t       real;
        uint16_t       imag;
        switch (i % 4) {
            case 0:
                real = 0x8000u;
                imag = row4_imag ^ 0x8000u;
                break;
            case 1:
                real = 0x0000u;
                imag = ((row4_imag & 0x7f80u) >= 0x0400u && (row4_imag & 0x7f80u) < 0x7f80u) ? 0x0001u : 0x0000u;
                break;
            case 2:
                real = 0x7f7fu;
                imag = 0xff7fu;
                break;
            default:
                real = 0x0000u;
                imag = 0x8000u;
                break;
        }
        residual[i] = f32_from_bits((uint32_t) real | ((uint32_t) imag << 16));
    }
    return residual;
}

static std::vector<uint16_t> make_row4_residual_scales(int64_t o) {
    constexpr uint16_t    profiles[] = { 0x3d80u, 0xbd80u, 0x3f80u, 0xbf80u };
    std::vector<uint16_t> scales((size_t) o);
    for (int64_t row = 0; row < o; ++row) {
        scales[(size_t) row] = profiles[row % 4];
    }
    return scales;
}

void row4_prefill_residual_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    if (strstr(text, "fuse Row4 prefill MPP + QAT COMPLEX_ADD residual")) {
        static_cast<std::atomic<bool> *>(user_data)->store(true, std::memory_order_relaxed);
    }
}

bool test_metal_row4_prefill_residual() {
    const char * strict = getenv("LLAMA_ROW4_REQUIRE_M5_TENSOROPS_TESTS");
    if (!strict || strcmp(strict, "0") == 0) {
        printf("  Metal Row4 prefill residual: SKIP (strict M5 suite only)\n");
        return true;
    }
    ggml_backend_dev_t dev = find_metal_device();
    if (!dev) {
        return false;
    }
    scoped_env_var disable("GGML_METAL_FUSION_DISABLE");
    scoped_env_var debug("GGML_METAL_FUSION_DEBUG");
    disable.set("1");
    ggml_backend_t reference = ggml_backend_dev_init(dev, nullptr);
    disable.unset();
    debug.set("2");
    ggml_backend_t fused = ggml_backend_dev_init(dev, nullptr);
    if (!reference || !fused) {
        if (reference) {
            ggml_backend_free(reference);
        }
        if (fused) {
            ggml_backend_free(fused);
        }
        return false;
    }
    bool ok = true;
    enum class mode : uint8_t { normal, output, extra, reverse, non_qat, split, scratch, inplace, ffn, packed_alias };

    struct test_case {
        int64_t k;
        int64_t rows;
        mode    kind;
    };

    std::vector<test_case> cases;
    for (int64_t k : { 4096, 12288 }) {
        for (int64_t rows : { 128, 512, 544, 2048 }) {
            cases.push_back({ k, rows, mode::normal });
        }
        for (mode kind :
             { mode::output, mode::extra, mode::reverse, mode::non_qat, mode::split, mode::scratch, mode::inplace }) {
            cases.push_back({ k, 512, kind });
        }
    }
    cases.push_back({ 12288, 512, mode::ffn });
    cases.push_back({ 12288, 512, mode::packed_alias });
    for (const auto & tc : cases) {
        constexpr int64_t      O          = 4096;
        const bool             ffn        = tc.kind == mode::ffn || tc.kind == mode::packed_alias;
        const int64_t          input_k    = ffn ? 4096 : tc.k;
        const ggml_init_params params     = { 2 * 1024 * 1024, nullptr, true };
        ggml_context *         ctx        = ggml_init(params);
        ggml_tensor *          x          = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_k, tc.rows);
        ggml_tensor *          activation = x;
        ggml_tensor *          gc         = nullptr;
        ggml_tensor *          gs         = nullptr;
        if (ffn) {
            gc = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES_PAIR2, 128, 8, input_k / 256, 2 * tc.k / 32);
            gs = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, 2 * tc.k);
            ggml_tensor * gate_up = ggml_row4_linear(ctx, x, gc, gs, 2 * tc.k, input_k);
            ggml_tensor * gate    = ggml_view_2d(ctx, gate_up, tc.k, tc.rows, gate_up->nb[1], 0);
            ggml_tensor * up      = ggml_view_2d(ctx, gate_up, tc.k, tc.rows, gate_up->nb[1], tc.k * sizeof(float));
            ggml_tensor * silu    = ggml_fairy2i_silu_exact(ctx, gate);
            ggml_fairy2i_exact_set_qat(silu, true);
            activation = ggml_fairy2i_mul_exact(ctx, silu, up);
            ggml_fairy2i_exact_set_qat(activation, true);
        }
        ggml_tensor * codes    = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES_PAIR2, 128, 8, tc.k / 256, O / 32);
        ggml_tensor * scales   = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, O);
        ggml_tensor * linear   = ggml_row4_linear(ctx, activation, codes, scales, O, tc.k);
        ggml_tensor * residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, O, tc.rows);
        ggml_tensor * add      = tc.kind == mode::reverse ? ggml_complex_add(ctx, residual, linear) :
                                                            ggml_complex_add(ctx, linear, residual);
        ggml_complex_add_set_qat(add, tc.kind != mode::non_qat);
        if (tc.kind == mode::output) {
            ggml_set_output(linear);
        }
        ggml_tensor * extra = tc.kind == mode::extra ? ggml_dup(ctx, linear) : nullptr;
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, add);
        if (extra) {
            ggml_build_forward_expand(graph, extra);
        }
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, fused);
        if (!buffer) {
            ggml_free(ctx);
            ok = false;
            break;
        }
        if (tc.kind == mode::scratch || tc.kind == mode::inplace || tc.kind == mode::packed_alias) {
            void * address = (char *) linear->data + ((ggml_nbytes(linear) + 63) & ~size_t(63));
            if (tc.kind == mode::inplace) {
                address = residual->data;
            } else if (tc.kind == mode::packed_alias) {
                address = activation->data;
            }
            add->buffer = nullptr;
            add->data   = nullptr;
            ok          = ggml_backend_tensor_alloc(buffer, add, address) == GGML_STATUS_SUCCESS && ok;
        }
        auto input = make_input(input_k, tc.rows);
        for (int64_t row = 0; row < tc.rows; row += 17) {
            for (int64_t col = 0; col < input_k; ++col) {
                input[row * input_k + col] = col % 2 ? -0.0f : 0.0f;
            }
        }
        const auto packed = make_row4_pair2_codes(O, tc.k);
        const auto scale  = make_row4_scales(O);
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(codes, packed.data(), 0, packed.size());
        ggml_backend_tensor_set(scales, scale.data(), 0, ggml_nbytes(scales));
        if (ffn) {
            const auto pc = make_row4_pair2_codes(2 * tc.k, input_k);
            const auto ps = make_row4_scales(2 * tc.k);
            ggml_backend_tensor_set(gc, pc.data(), 0, pc.size());
            ggml_backend_tensor_set(gs, ps.data(), 0, ggml_nbytes(gs));
        }
        constexpr uint16_t profiles[] = { 0,      0x8000, 1,      0x8001, 0x7f,   0x807f, 0x80,   0x8080,
                                          0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1, 0xffc1 };
        std::vector<float> residual_data(O * tc.rows);
        for (size_t i = 0; i < residual_data.size(); ++i) {
            residual_data[i] = f32_from_bits((uint32_t) profiles[i % 16] | ((uint32_t) profiles[(i / 16) % 16] << 16));
        }
        std::vector<float> expected(O * tc.rows);
        std::vector<float> expected_linear(O * tc.rows);
        for (bool ref : { true, false }) {
            ggml_backend_tensor_set(residual, residual_data.data(), 0, ggml_nbytes(residual));
            std::atomic<bool> hit{ false };
            ggml_log_set(row4_prefill_residual_log_callback, &hit);
            ggml_backend_t backend = ref ? reference : fused;
            ggml_status    status;
            if (!ref && tc.kind == mode::split) {
                ggml_cgraph * a = ggml_new_graph(ctx);
                ggml_cgraph * b = ggml_new_graph(ctx);
                ggml_build_forward_expand(a, linear);
                ggml_graph_add_node(b, add);
                status = ggml_backend_graph_compute(backend, a);
                if (status == GGML_STATUS_SUCCESS) {
                    status = ggml_backend_graph_compute(backend, b);
                }
            } else {
                status = ggml_backend_graph_compute(backend, graph);
            }
            ggml_log_set(nullptr, nullptr);
            const bool expected_hit = !ref && tc.rows >= 512 && tc.rows % 64 == 0 &&
                                      (tc.kind == mode::normal || tc.kind == mode::inplace || tc.kind == mode::ffn);
            if (status != GGML_STATUS_SUCCESS || hit.load() != expected_hit) {
                fprintf(stderr, "prefill residual K=%lld B=%lld mode=%d ref=%d status=%d hit=%d expected=%d\n",
                        (long long) tc.k, (long long) tc.rows, (int) tc.kind, ref, (int) status, hit.load(),
                        expected_hit);
                ok = false;
            }
            std::vector<float> actual(O * tc.rows);
            ggml_backend_tensor_get(add, actual.data(), 0, ggml_nbytes(add));
            if (ref) {
                expected = std::move(actual);
            } else {
                ok = compare_exact("prefill residual", actual, expected) && ok;
            }
            if (tc.kind == mode::output || extra) {
                std::vector<float> visible(O * tc.rows);
                ggml_backend_tensor_get(extra ? extra : linear, visible.data(), 0, ggml_nbytes(linear));
                if (ref) {
                    expected_linear = std::move(visible);
                } else {
                    ok = compare_exact("prefill observable projection", visible, expected_linear) && ok;
                }
            }
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    ggml_backend_free(fused);
    ggml_backend_free(reference);
    printf(
        "  Metal Row4 prefill residual: BF16 edges, output/extra/reverse/non-QAT/split/scratch/inplace/FFN gates - "
        "%s\n",
        ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_metal_row4_decode_residual_fusion() {
    ggml_backend_dev_t dev = find_metal_device();
    if (!dev) {
        const char * required = getenv("LLAMA_ROW4_REQUIRE_METAL_TESTS");
        if (required && strcmp(required, "0") != 0) {
            fprintf(stderr, "Row4 decode residual fusion test requires Metal, but no Metal device is available\n");
            return false;
        }
        printf("  Metal Row4 decode QAT residual fusion: SKIP (Metal backend unavailable)\n");
        return true;
    }

    constexpr int64_t O = 128;
    scoped_env_var    fusion_disable("GGML_METAL_FUSION_DISABLE");
    scoped_env_var    fusion_debug("GGML_METAL_FUSION_DEBUG");
    bool              ok = true;

    for (int64_t k : { 4096, 12288 }) {
        const std::vector<float>    input       = make_input(k, 1);
        const std::vector<uint8_t>  codes       = make_row4_codes(O, k);
        const std::vector<uint8_t>  pair2_codes = make_row4_pair2_codes(O, k);
        const std::vector<uint16_t> scales      = make_row4_residual_scales(O);
        const std::vector<float>    row4_oracle = oracle_row4_linear(input, codes, scales, O, k, 1);
        const std::vector<float>    residual    = make_residual_carriers(row4_oracle);
        const row4_residual_case    test_case;

        ggml_backend_t cpu = ggml_backend_cpu_init();
        if (!cpu) {
            return false;
        }
        ggml_backend_cpu_set_n_threads(cpu, 4);
        row4_residual_outputs expected;
        ok = run_row4_residual_backend(expected, cpu, input, codes, scales, residual, O, k, test_case) &&
             compare_exact(("Row4 residual CPU/oracle K=" + std::to_string(k)).c_str(), expected.row4, row4_oracle) &&
             ok;
        ggml_backend_free(cpu);

        fusion_disable.set("1");
        fusion_debug.unset();
        ggml_backend_t        metal_unfused = ggml_backend_dev_init(dev, nullptr);
        row4_residual_outputs unfused;
        ok =
            metal_unfused &&
            run_row4_residual_backend(unfused, metal_unfused, input, codes, scales, residual, O, k, test_case) &&
            compare_exact(("Row4 residual unfused row4 K=" + std::to_string(k)).c_str(), unfused.row4, expected.row4) &&
            compare_exact(("Row4 residual unfused add K=" + std::to_string(k)).c_str(), unfused.add, expected.add) &&
            ok;
        if (metal_unfused) {
            ggml_backend_free(metal_unfused);
        }

        fusion_disable.unset();
        fusion_debug.set("2");
        ggml_backend_t        metal_fused = ggml_backend_dev_init(dev, nullptr);
        row4_residual_outputs fused;
        ok = metal_fused &&
             run_row4_residual_backend(fused, metal_fused, input, codes, scales, residual, O, k, test_case) &&
             compare_exact(("Row4 residual fused row4 K=" + std::to_string(k)).c_str(), fused.row4, expected.row4) &&
             compare_exact(("Row4 residual fused add K=" + std::to_string(k)).c_str(), fused.add, expected.add) &&
             compare_exact(("Row4 residual fused/unfused K=" + std::to_string(k)).c_str(), fused.add, unfused.add) &&
             ok;
        if (metal_fused) {
            ggml_backend_free(metal_fused);
        }

        fusion_disable.set("1");
        fusion_debug.unset();
        ggml_backend_t        pair2_unfused_backend = ggml_backend_dev_init(dev, nullptr);
        row4_residual_outputs pair2_unfused;
        ok = pair2_unfused_backend &&
             run_row4_residual_backend(pair2_unfused, pair2_unfused_backend, input, pair2_codes, scales, residual, O, k,
                                       test_case, true) &&
             compare_exact(("Row4 pair2 residual unfused row4 K=" + std::to_string(k)).c_str(), pair2_unfused.row4,
                           expected.row4) &&
             compare_exact(("Row4 pair2 residual unfused add K=" + std::to_string(k)).c_str(), pair2_unfused.add,
                           expected.add) &&
             ok;
        if (pair2_unfused_backend) {
            ggml_backend_free(pair2_unfused_backend);
        }

        fusion_disable.unset();
        fusion_debug.set("2");
        ggml_backend_t        pair2_fused_backend = ggml_backend_dev_init(dev, nullptr);
        row4_residual_outputs pair2_fused;
        bool                  pair2_fusion_hit = false;
        ok = pair2_fused_backend &&
             run_row4_residual_backend(pair2_fused, pair2_fused_backend, input, pair2_codes, scales, residual, O, k,
                                       test_case, true, &pair2_fusion_hit, "pipeline=lut16-const-rows-f32") &&
             compare_exact(("Row4 pair2 production residual fused row4 K=" + std::to_string(k)).c_str(),
                           pair2_fused.row4, expected.row4) &&
             compare_exact(("Row4 pair2 production residual fused add K=" + std::to_string(k)).c_str(), pair2_fused.add,
                           expected.add) &&
             compare_exact(("Row4 pair2 production residual fused/unfused K=" + std::to_string(k)).c_str(),
                           pair2_fused.add, pair2_unfused.add) &&
             ok;
        if (!pair2_fusion_hit) {
            fprintf(stderr, "Row4 pair2 production residual fusion marker missing for K=%lld\n", (long long) k);
            ok = false;
        }
        if (pair2_fused_backend) {
            ggml_backend_free(pair2_fused_backend);
        }
    }

    // K256 is a correctness fixture, not a production fused shape.  It must
    // remain on the standalone Pair2 ROW4 + QAT COMPLEX_ADD path.
    {
        constexpr int64_t           fallback_o           = 32;
        constexpr int64_t           fallback_k           = 256;
        const std::vector<float>    fallback_input       = make_input(fallback_k, 1);
        const std::vector<uint8_t>  fallback_v1_codes    = make_row4_codes(fallback_o, fallback_k);
        const std::vector<uint8_t>  fallback_pair2_codes = make_row4_pair2_codes(fallback_o, fallback_k);
        const std::vector<uint16_t> fallback_scales      = make_row4_residual_scales(fallback_o);
        const std::vector<float>    fallback_row4 =
            oracle_row4_linear(fallback_input, fallback_v1_codes, fallback_scales, fallback_o, fallback_k, 1);
        const std::vector<float> fallback_residual = make_residual_carriers(fallback_row4);
        const row4_residual_case fallback_case;

        fusion_disable.set("1");
        fusion_debug.unset();
        ggml_backend_t        fallback_unfused_backend = ggml_backend_dev_init(dev, nullptr);
        row4_residual_outputs fallback_unfused;
        ok = fallback_unfused_backend &&
             run_row4_residual_backend(fallback_unfused, fallback_unfused_backend, fallback_input, fallback_pair2_codes,
                                       fallback_scales, fallback_residual, fallback_o, fallback_k, fallback_case,
                                       true) &&
             ok;
        if (fallback_unfused_backend) {
            ggml_backend_free(fallback_unfused_backend);
        }

        fusion_disable.unset();
        fusion_debug.set("2");
        ggml_backend_t        fallback_backend = ggml_backend_dev_init(dev, nullptr);
        row4_residual_outputs fallback_output;
        bool                  fallback_fusion_hit = false;
        ok = fallback_backend &&
             run_row4_residual_backend(fallback_output, fallback_backend, fallback_input, fallback_pair2_codes,
                                       fallback_scales, fallback_residual, fallback_o, fallback_k, fallback_case, true,
                                       &fallback_fusion_hit) &&
             compare_exact("Row4 pair2 residual K256 fallback row4", fallback_output.row4, fallback_row4) &&
             compare_exact("Row4 pair2 residual K256 fallback/unfused", fallback_output.add, fallback_unfused.add) &&
             ok;
        if (fallback_fusion_hit) {
            fprintf(stderr, "Row4 pair2 K256 unexpectedly entered the production-only residual fusion\n");
            ok = false;
        }
        if (fallback_backend) {
            ggml_backend_free(fallback_backend);
        }
    }

    // Every condition below must retain the ordinary two-dispatch path.
    const int64_t                         k          = 4096;
    const std::vector<uint8_t>            codes      = make_row4_codes(O, k);
    const std::vector<uint16_t>           scales     = make_row4_residual_scales(O);
    const std::vector<float>              input_1    = make_input(k, 1);
    const std::vector<float>              row4_1     = oracle_row4_linear(input_1, codes, scales, O, k, 1);
    const std::vector<float>              residual_1 = make_residual_carriers(row4_1);
    const std::vector<row4_residual_case> gates      = {
        { 9, true,  false, false, false, false },
        { 1, false, false, false, false, false },
        { 1, true,  true,  false, false, false },
        { 1, true,  false, true,  false, false },
        { 1, true,  false, false, true,  false },
        { 1, true,  false, false, false, true  },
    };
    for (size_t i = 0; i < gates.size(); ++i) {
        const row4_residual_case & test_case = gates[i];
        const std::vector<float>   input     = test_case.tokens == 1 ? input_1 : make_input(k, test_case.tokens);
        const std::vector<float>   residual =
            test_case.tokens == 1 ?
                residual_1 :
                make_residual_carriers(oracle_row4_linear(input, codes, scales, O, k, test_case.tokens));

        fusion_disable.set("1");
        fusion_debug.unset();
        ggml_backend_t        metal_unfused = ggml_backend_dev_init(dev, nullptr);
        row4_residual_outputs unfused;
        ok = metal_unfused &&
             run_row4_residual_backend(unfused, metal_unfused, input, codes, scales, residual, O, k, test_case) && ok;
        if (metal_unfused) {
            ggml_backend_free(metal_unfused);
        }

        fusion_disable.unset();
        fusion_debug.set("2");
        ggml_backend_t        metal_gated = ggml_backend_dev_init(dev, nullptr);
        row4_residual_outputs gated;
        ok = metal_gated &&
             run_row4_residual_backend(gated, metal_gated, input, codes, scales, residual, O, k, test_case) &&
             compare_exact(("Row4 residual negative gate " + std::to_string(i)).c_str(), gated.add, unfused.add) && ok;
        if (metal_gated) {
            ggml_backend_free(metal_gated);
        }
    }

    printf("  Metal Row4 decode QAT residual fusion/gates - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_metal_operator_matrix() {
    const char *       require_m5_value     = getenv("LLAMA_ROW4_REQUIRE_M5_TENSOROPS_TESTS");
    const bool         require_m5_tensorops = require_m5_value && strcmp(require_m5_value, "0") != 0;
    ggml_backend_dev_t dev = find_metal_device();
    if (!dev) {
        const char * require_metal = getenv("LLAMA_ROW4_REQUIRE_METAL_TESTS");
        if ((require_metal && strcmp(require_metal, "0") != 0) || require_m5_tensorops) {
            fprintf(stderr, "a required Metal Row4 test was requested, but no Metal device is available\n");
            return false;
        }
        printf("  Metal Row4/W8A8 dispatch boundaries: SKIP (Metal backend unavailable)\n");
        return true;
    }

    ggml_backend_t metal = ggml_backend_dev_init(dev, nullptr);
    if (!metal) {
        fprintf(stderr, "failed to initialize Metal backend\n");
        return false;
    }

    constexpr int64_t           O           = 128;
    constexpr int64_t           K           = 128;
    const std::vector<uint8_t>  row4_codes  = make_row4_codes(O, K);
    const std::vector<uint16_t> row4_scales = make_row4_scales(O);
    const std::vector<int8_t>   w8_codes    = make_w8_codes(O, K);
    const std::vector<float>    w8_scales   = make_w8_scales(O);

    m5_tensorops_path_log_marker m5_marker;
    if (require_m5_tensorops) {
        ggml_log_set(m5_tensorops_path_log_callback, &m5_marker);
    }

    bool ok = true;
    for (int64_t tokens : { 1, 2, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 64, 96, 128, 256, 512 }) {
        const std::vector<float> input = make_input(K, tokens);
        std::vector<float>       actual;
        const std::vector<float> expected_row4 = oracle_row4_linear(input, row4_codes, row4_scales, O, K, tokens);
        if (!run_operator_backend(actual, linear_kind::row4, input, row4_codes, row4_scales, {}, {}, O, K, tokens,
                                  nullptr, false, metal) ||
            !compare_exact(("Row4 Metal B=" + std::to_string(tokens)).c_str(), actual, expected_row4)) {
            ok = false;
        }

        const std::vector<float> expected_w8 = oracle_w8a8_linear(input, w8_codes, w8_scales, O, K, tokens);
        if (!run_operator_backend(actual, linear_kind::w8a8, input, {}, {}, w8_codes, w8_scales, O, K, tokens, nullptr,
                                  false, metal) ||
            !compare_exact(("W8A8 Metal B=" + std::to_string(tokens)).c_str(), actual, expected_w8)) {
            ok = false;
        }
    }
    // Minimum Pair2 output block: exercises O32 decode, reduced small-batch,
    // and the non-M64 generic prefill kernel at B9.
    {
        constexpr int64_t           pair2_o        = 32;
        constexpr int64_t           pair2_k        = 256;
        const std::vector<uint8_t>  pair2_v1_codes = make_row4_codes(pair2_o, pair2_k);
        const std::vector<uint8_t>  pair2_codes    = make_row4_pair2_codes(pair2_o, pair2_k);
        const std::vector<uint16_t> pair2_scales   = make_row4_scales(pair2_o);
        for (int64_t tokens : { 1, 2, 9 }) {
            const std::vector<float> input = make_input(pair2_k, tokens);
            const std::vector<float> expected =
                oracle_row4_linear(input, pair2_v1_codes, pair2_scales, pair2_o, pair2_k, tokens);
            std::vector<float> actual;
            if (!run_operator_backend(actual, linear_kind::row4_pair2, input, pair2_codes, pair2_scales, {}, {},
                                      pair2_o, pair2_k, tokens, nullptr, false, metal) ||
                !compare_exact(("Row4 pair2 Metal O32 K256 B=" + std::to_string(tokens)).c_str(), actual, expected)) {
                ok = false;
            }
        }
    }

    // Pair2 is a Metal-only physical layout for the same logical Row4
    // weights. Cover every dispatch boundary using K=256, the minimum legal
    // pair2 K, and compare against the independent v1 oracle.
    {
        constexpr int64_t          KPAIR       = 256;
        const std::vector<uint8_t> v1_codes    = make_row4_codes(O, KPAIR);
        const std::vector<uint8_t> pair2_codes = make_row4_pair2_codes(O, KPAIR);
        for (int64_t tokens : { 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 96, 128, 256, 512 }) {
            const std::vector<float> input    = make_input(KPAIR, tokens);
            const std::vector<float> expected = oracle_row4_linear(input, v1_codes, row4_scales, O, KPAIR, tokens);
            std::vector<float>       actual;
            if (!run_operator_backend(actual, linear_kind::row4_pair2, input, pair2_codes, row4_scales, {}, {}, O,
                                      KPAIR, tokens, nullptr, false, metal) ||
                !compare_exact(("Row4 pair2 Metal B=" + std::to_string(tokens)).c_str(), actual, expected)) {
                ok = false;
            }
        }
    }

    // Non-power-of-two Pair2 dimensions cross both output and K blocks
    // and exercise a non-512 prefill batch.
    {
        constexpr int64_t           pair2_o     = 1152;
        constexpr int64_t           pair2_k     = 768;
        constexpr int64_t           tokens      = 544;
        const std::vector<uint8_t>  v1_codes    = make_row4_codes(pair2_o, pair2_k);
        const std::vector<uint8_t>  pair2_codes = make_row4_pair2_codes(pair2_o, pair2_k);
        const std::vector<uint16_t> scales      = make_row4_scales(pair2_o);
        const std::vector<float>    input       = make_input(pair2_k, tokens);
        const std::vector<float>    expected    = oracle_row4_linear(input, v1_codes, scales, pair2_o, pair2_k, tokens);
        std::vector<float>          actual;
        if (!run_operator_backend(actual, linear_kind::row4_pair2, input, pair2_codes, scales, {}, {}, pair2_o, pair2_k,
                                  tokens, nullptr, false, metal) ||
            !compare_exact("Row4 pair2 Metal O1152 K768 B544", actual, expected)) {
            ok = false;
        }
    }

    // Blocked expansion and cooperative M64 stores: minimum O, non-power-
    // of-two O/K, a partial group of token tiles, and the wide N128 path.
    // Random codes distinguish output tiles; an odd token period also
    // exposes row-tile permutations without a huge oracle.
    const int64_t preexpanded_shapes[][3] = {
        { 128,   512,  512 },
        { 384,   768,  544 },
        { 1152,  1024, 576 },
        { 16384, 512,  512 },
    };
    for (const auto & shape : preexpanded_shapes) {
        const int64_t        o      = shape[0];
        const int64_t        k      = shape[1];
        const int64_t        tokens = shape[2];
        constexpr int64_t    period = 17;
        std::vector<uint8_t> logical((size_t) (o / 4) * k);
        uint32_t             state = 42;
        for (uint8_t & code : logical) {
            state = state * 1664525u + 1013904223u;
            code  = (uint8_t) (state >> 28);
        }
        const auto         v1_codes         = pack_row4_codes(logical, o, k);
        const auto         pair2_codes      = pack_row4_pair2_codes(logical, o, k);
        const auto         scales           = make_row4_scales(o);
        const auto         input_pattern    = make_input(k, period);
        const auto         expected_pattern = oracle_row4_linear(input_pattern, v1_codes, scales, o, k, period);
        std::vector<float> input((size_t) tokens * k);
        std::vector<float> expected((size_t) tokens * o);
        std::vector<float> actual;
        for (int64_t token = 0; token < tokens; ++token) {
            std::copy_n(input_pattern.data() + (token % period) * k, k, input.data() + token * k);
            std::copy_n(expected_pattern.data() + (token % period) * o, o, expected.data() + token * o);
        }
        const std::string label = "Row4 pair2 preexpanded O=" + std::to_string(o) + " K=" + std::to_string(k) +
                                  " B=" + std::to_string(tokens);
        if (!run_operator_backend(actual, linear_kind::row4_pair2, input, pair2_codes, scales, {}, {}, o, k, tokens,
                                  nullptr, false, metal, true) ||
            !compare_exact(label.c_str(), actual, expected)) {
            ok = false;
        }
    }

    // Real-K prefill cases. Row4 covers all 96 K tiles of ffn_down and
    // includes both the maximum sum and cancellation. W8 crosses four K1024
    // segments so an implementation cannot accidentally use one inexact F32
    // accumulator for the whole lm_head reduction.
    {
        constexpr int64_t    KROW4  = 12288;
        constexpr int64_t    TOKENS = 9;
        std::vector<uint8_t> logical((size_t) (O / 4) * KROW4, 0);
        for (int64_t k = 0; k < KROW4; ++k) {
            logical[(size_t) KROW4 + (size_t) k] = (uint8_t) ((k & 1) ? 5 : 0);
        }
        const std::vector<uint8_t>  codes       = pack_row4_codes(logical, O, KROW4);
        const std::vector<uint8_t>  pair2_codes = pack_row4_pair2_codes(logical, O, KROW4);
        const std::vector<uint16_t> scales((size_t) O, oracle_bf16_bits(1.0f));
        const std::vector<float>    input((size_t) KROW4 * TOKENS, 1.0f);
        const std::vector<float>    expected = oracle_row4_linear(input, codes, scales, O, KROW4, TOKENS);
        std::vector<float>          actual;
        if (!run_operator_backend(actual, linear_kind::row4, input, codes, scales, {}, {}, O, KROW4, TOKENS, nullptr,
                                  false, metal) ||
            !compare_exact("Row4 Metal K=12288 B=9 maximum/cancellation", actual, expected)) {
            ok = false;
        }
        if (!run_operator_backend(actual, linear_kind::row4_pair2, input, pair2_codes, scales, {}, {}, O, KROW4, TOKENS,
                                  nullptr, false, metal) ||
            !compare_exact("Row4 pair2 Metal K=12288 B=9 maximum/cancellation", actual, expected)) {
            ok = false;
        }
    }

    {
        constexpr int64_t   KW8    = 4096;
        constexpr int64_t   TOKENS = 9;
        std::vector<int8_t> codes((size_t) O * KW8, 127);
        for (int64_t k = 0; k < KW8; ++k) {
            codes[w8_offset(1, k, KW8)] = (int8_t) (((k / 1024) & 1) ? -127 : 127);
        }
        const std::vector<float> scales((size_t) O, 1.0f);
        const std::vector<float> input((size_t) KW8 * TOKENS, 1.0f);
        const std::vector<float> expected = oracle_w8a8_linear(input, codes, scales, O, KW8, TOKENS);
        std::vector<float>       actual;
        if (!run_operator_backend(actual, linear_kind::w8a8, input, {}, {}, codes, scales, O, KW8, TOKENS, nullptr,
                                  false, metal) ||
            !compare_exact("W8A8 Metal K=4096 B=9 segmented maximum/cancellation", actual, expected)) {
            ok = false;
        }
    }

    if (require_m5_tensorops) {
        ggml_log_set(nullptr, nullptr);
        constexpr uint32_t all_m5_tiles = (1u << 12) - 1u;
        const uint32_t     hit_mask     = m5_marker.tile_mask.load(std::memory_order_relaxed);
        if (hit_mask != all_m5_tiles) {
            fprintf(stderr, "M5 TensorOps tile marker mismatch: actual=0x%x expected=0x%x\n", hit_mask, all_m5_tiles);
            ok = false;
        }
        if (!m5_marker.pair2_device_preexpand.load(std::memory_order_relaxed)) {
            fprintf(stderr, "M5 TensorOps Pair2 B512 device-preexpand coalesced-expand marker was not observed\n");
            ok = false;
        }
        if (!m5_marker.pair2_cooperative_n64.load(std::memory_order_relaxed) ||
            !m5_marker.pair2_cooperative_n128.load(std::memory_order_relaxed)) {
            fprintf(stderr, "M5 TensorOps Pair2 M64 cooperative-store markers were not observed\n");
            ok = false;
        }
    }

    ggml_backend_free(metal);
    printf("  Metal Row4/W8A8 decode/small-batch/prefill boundaries - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool compare_exact_matrix(const char *               label,
                                 const std::vector<float> & actual,
                                 const std::vector<float> & expected,
                                 int64_t                    rows) {
    if (actual.size() != expected.size()) {
        fprintf(stderr, "%s size mismatch: actual=%zu expected=%zu\n", label, actual.size(), expected.size());
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (f32_bits(actual[i]) != f32_bits(expected[i])) {
            fprintf(stderr, "%s mismatch token=%zu row=%zu: actual=%g (0x%08x) expected=%g (0x%08x)\n", label,
                    i / (size_t) rows, i % (size_t) rows, actual[i], f32_bits(actual[i]), expected[i],
                    f32_bits(expected[i]));
            return false;
        }
    }
    return true;
}

static std::vector<float> repeat_two_token_pattern(const std::vector<float> & two_tokens, int64_t k, int64_t tokens) {
    std::vector<float> repeated((size_t) k * (size_t) tokens);
    for (int64_t token = 0; token < tokens; ++token) {
        memcpy(repeated.data() + token * k, two_tokens.data() + (token % 2) * k, (size_t) k * sizeof(float));
    }
    return repeated;
}

static std::vector<float> repeat_two_token_output(const std::vector<float> & two_outputs,
                                                  int64_t                    rows,
                                                  int64_t                    tokens) {
    std::vector<float> repeated((size_t) rows * (size_t) tokens);
    for (int64_t token = 0; token < tokens; ++token) {
        memcpy(repeated.data() + token * rows, two_outputs.data() + (token % 2) * rows, (size_t) rows * sizeof(float));
    }
    return repeated;
}

static bool test_metal_real_shape_matrix() {
    const char * enabled = getenv("LLAMA_ROW4_REAL_SHAPE_TESTS");
    if (!enabled || strcmp(enabled, "0") == 0) {
        printf("  Metal Row4 real-shape exact matrix: SKIP (set LLAMA_ROW4_REAL_SHAPE_TESTS=1)\n");
        return true;
    }

    ggml_backend_dev_t dev = find_metal_device();
    if (!dev) {
        fprintf(stderr, "real-shape Row4 tests requested, but Metal is unavailable\n");
        return false;
    }
    ggml_backend_t metal = ggml_backend_dev_init(dev, nullptr);
    if (!metal) {
        fprintf(stderr, "failed to initialize Metal for real-shape Row4 tests\n");
        return false;
    }

    struct row4_shape {
        const char * label;
        int64_t      o;
        int64_t      k;
    };

    const row4_shape shapes[] = {
        { "qkv",     6144,  4096  },
        { "o",       4096,  4096  },
        { "gate_up", 24576, 4096  },
        { "down",    4096,  12288 },
    };

    bool ok = true;
    for (const row4_shape & shape : shapes) {
        const std::vector<uint8_t> codes       = make_row4_codes(shape.o, shape.k);
        const std::vector<uint8_t> pair2_codes = make_row4_pair2_codes(shape.o, shape.k);
        std::vector<uint16_t>      scales((size_t) shape.o);
        for (int64_t row = 0; row < shape.o; ++row) {
            const float scale    = (row & 1) ? -0.03125f * (float) (1 + row % 3) : 0.03125f * (float) (1 + row % 3);
            scales[(size_t) row] = oracle_bf16_bits(scale);
        }

        const std::vector<float> input_two    = make_input(shape.k, 2);
        const std::vector<float> expected_two = oracle_row4_linear(input_two, codes, scales, shape.o, shape.k, 2);
        for (int64_t tokens : { 1, 2, 3, 4, 7, 8, 9, 11, 12, 15, 16, 17, 24, 31, 32, 33, 64, 96, 128, 256, 512 }) {
            const std::vector<float> input    = repeat_two_token_pattern(input_two, shape.k, tokens);
            const std::vector<float> expected = repeat_two_token_output(expected_two, shape.o, tokens);
            std::vector<float>       actual;
            const std::string label = std::string("Row4 Metal real ") + shape.label + " O=" + std::to_string(shape.o) +
                                      " K=" + std::to_string(shape.k) + " B=" + std::to_string(tokens);
            if (!run_operator_backend(actual, linear_kind::row4, input, codes, scales, {}, {}, shape.o, shape.k, tokens,
                                      nullptr, false, metal) ||
                !compare_exact_matrix(label.c_str(), actual, expected, shape.o)) {
                ok = false;
            }
            const std::string pair2_label = std::string("Row4 pair2 Metal real ") + shape.label +
                                            " O=" + std::to_string(shape.o) + " K=" + std::to_string(shape.k) +
                                            " B=" + std::to_string(tokens);
            if (!run_operator_backend(actual, linear_kind::row4_pair2, input, pair2_codes, scales, {}, {}, shape.o,
                                      shape.k, tokens, nullptr, false, metal) ||
                !compare_exact_matrix(pair2_label.c_str(), actual, expected, shape.o)) {
                ok = false;
            }
        }
    }

    {
        constexpr int64_t         O     = 256;
        constexpr int64_t         K     = 4096;
        const std::vector<int8_t> codes = make_w8_codes(O, K);
        std::vector<float>        scales((size_t) O);
        for (int64_t row = 0; row < O; ++row) {
            scales[(size_t) row] =
                (row & 1) ? -0.00390625f * (float) (1 + row % 3) : 0.00390625f * (float) (1 + row % 3);
        }
        const std::vector<float> input_two    = make_input(K, 2);
        const std::vector<float> expected_two = oracle_w8a8_linear(input_two, codes, scales, O, K, 2);
        for (int64_t tokens : { 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33 }) {
            const std::vector<float> input    = repeat_two_token_pattern(input_two, K, tokens);
            const std::vector<float> expected = repeat_two_token_output(expected_two, O, tokens);
            std::vector<float>       actual;
            const std::string        label = "W8A8 Metal real O=256 K=4096 B=" + std::to_string(tokens);
            if (!run_operator_backend(actual, linear_kind::w8a8, input, {}, {}, codes, scales, O, K, tokens, nullptr,
                                      false, metal) ||
                !compare_exact_matrix(label.c_str(), actual, expected, O)) {
                ok = false;
            }
        }

        const char * full_lm_head = getenv("LLAMA_ROW4_FULL_LM_HEAD_TESTS");
        if (full_lm_head && strcmp(full_lm_head, "0") != 0) {
            constexpr int64_t   FULL_O = 151936;
            std::vector<int8_t> full_codes((size_t) FULL_O * K);
            std::vector<float>  full_scales((size_t) FULL_O);
            for (int64_t row = 0; row < FULL_O; ++row) {
                const int64_t source_row  = row % O;
                full_scales[(size_t) row] = scales[(size_t) source_row];
                for (int64_t kt = 0; kt < K / ROW4_TILE_K; ++kt) {
                    memcpy(full_codes.data() + w8_offset(row, kt * ROW4_TILE_K, K),
                           codes.data() + w8_offset(source_row, kt * ROW4_TILE_K, K),
                           (size_t) ROW4_TILE_K * sizeof(int8_t));
                }
            }

            for (int64_t tokens : { 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33 }) {
                const std::vector<float> input = repeat_two_token_pattern(input_two, K, tokens);
                std::vector<float>       expected((size_t) FULL_O * tokens);
                for (int64_t token = 0; token < tokens; ++token) {
                    for (int64_t row = 0; row < FULL_O; ++row) {
                        expected[(size_t) token * FULL_O + row] =
                            expected_two[(size_t) (token % 2) * O + (size_t) (row % O)];
                    }
                }

                std::vector<float> actual;
                const std::string label =
                    "W8A8 Metal full lm_head O=151936 K=4096 B=" + std::to_string(tokens);
                if (!run_operator_backend(actual, linear_kind::w8a8, input, {}, {}, full_codes, full_scales, FULL_O, K,
                                          tokens, nullptr, false, metal) ||
                    !compare_exact_matrix(label.c_str(), actual, expected, FULL_O)) {
                    ok = false;
                }
            }
        } else {
            printf("  Metal full lm_head exact test: SKIP (set LLAMA_ROW4_FULL_LM_HEAD_TESTS=1)\n");
        }
    }

    ggml_backend_free(metal);
    printf("  Metal Row4/W8A8 real-shape exact matrix - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_fused_boundaries() {
    bool              ok                 = true;
    constexpr int64_t K                  = 128;
    const int64_t     projection_sizes[] = { 128, 256, 128 };
    int64_t           total_o            = 0;
    for (int64_t size : projection_sizes) {
        total_o += size;
    }

    std::vector<uint8_t> fused_logical((size_t) (total_o / 4) * K);
    std::vector<uint8_t> concatenated;
    int64_t              group_base = 0;
    for (size_t projection = 0; projection < sizeof(projection_sizes) / sizeof(projection_sizes[0]); ++projection) {
        const int64_t        projection_o = projection_sizes[projection];
        std::vector<uint8_t> separate((size_t) (projection_o / 4) * K);
        for (int64_t group = 0; group < projection_o / 4; ++group) {
            for (int64_t k = 0; k < K; ++k) {
                const uint8_t code               = (uint8_t) ((projection * 7 + group * 3 + k) & 15);
                separate[(size_t) group * K + k] = code;
                fused_logical[(size_t) (group_base + group) * K + k] = code;
            }
        }
        const std::vector<uint8_t> packed = pack_row4_codes(separate, projection_o, K);
        concatenated.insert(concatenated.end(), packed.begin(), packed.end());
        group_base += projection_o / 4;
    }
    const std::vector<uint8_t> fused = pack_row4_codes(fused_logical, total_o, K);
    if (fused != concatenated) {
        fprintf(stderr, "QKV/gate-up projection boundary crossed an M16 tile\n");
        ok = false;
    }

    printf("  Fused projection tile boundaries - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace

namespace {

struct rms_rope_marker {
    std::atomic<bool> hit{ false };
    std::atomic<bool> kv{ false };
    std::atomic<bool> bf16{ false };
};

void rms_rope_log(enum ggml_log_level level, const char * text, void * user) {
    (void) level;
    auto & m = *static_cast<rms_rope_marker *>(user);
    if (strstr(text, "fuse QAT RMS128 + RoPE")) {
        m.hit.store(true, std::memory_order_relaxed);
        if (strstr(text, "kv-store=1")) {
            m.kv.store(true, std::memory_order_relaxed);
        }
        if (strstr(text, "bf16-weight=1")) {
            m.bf16.store(true, std::memory_order_relaxed);
        }
    }
}

enum class rms_rope_gate : uint8_t {
    normal,
    rms_output,
    rope_output,
    extra_rms,
    extra_rope,
    split,
    nonqat,
    partial,
    cast_output,
    aliased_cache,
    aliased_freq,
    repeated_weight
};

struct rms_rope_case {
    int64_t       heads, tokens;
    bool          bf16, kv, strided, edge, freq;
    rms_rope_gate gate;
};

bool run_rms_rope_backend(ggml_backend_t        backend,
                          const rms_rope_case & c,
                          std::vector<float> &  output,
                          rms_rope_marker &     marker) {
    const ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context *         ctx    = ggml_init(params);
    ggml_tensor * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128 * c.heads + (c.strided ? 64 : 0), c.tokens);
    ggml_tensor * x       = ggml_view_3d(ctx, storage, 128, c.heads, c.tokens, 128 * sizeof(float), storage->nb[1], 0);
    ggml_tensor * weight  = ggml_new_tensor_2d(ctx, c.bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F32,
                                               c.gate == rms_rope_gate::repeated_weight ? 64 : 128,
                                               c.gate == rms_rope_gate::repeated_weight ? 2 : 1);
    ggml_tensor * wf      = c.bf16 ? ggml_cast(ctx, weight, GGML_TYPE_F32) : weight;
    if (c.gate == rms_rope_gate::cast_output) {
        ggml_set_output(wf);
    }
    ggml_tensor * rms = ggml_fairy2i_rms_norm_exact(ctx, x, wf, 1.0e-6f);
    ggml_fairy2i_exact_set_qat(rms, c.gate != rms_rope_gate::nonqat);
    if (c.gate == rms_rope_gate::rms_output) {
        ggml_set_output(rms);
    }
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.tokens);
    ggml_tensor * pos_view  = ggml_view_1d(ctx, positions, c.tokens, 0);
    ggml_tensor * freq      = c.freq ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64) : nullptr;
    if (c.gate == rms_rope_gate::aliased_freq) {
        freq = ggml_view_1d(ctx, x, 64, 0);
    }
    ggml_tensor * rope = ggml_fairy2i_rope_ext_exact(ctx, rms, pos_view, freq,
                                                     c.gate == rms_rope_gate::partial ? 64 : 128, GGML_ROPE_TYPE_NEOX,
                                                     32768, 1000000.0f, 1.0f, c.freq ? 0.5f : 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_fairy2i_exact_set_qat(rope, c.gate != rms_rope_gate::nonqat);
    if (c.gate == rms_rope_gate::rope_output) {
        ggml_set_output(rope);
    }
    ggml_tensor * rows    = c.kv ? ggml_reshape_2d(ctx, rope, 128 * c.heads, c.tokens) : nullptr;
    ggml_tensor * cache   = c.kv ? ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 128 * c.heads, c.tokens + 17) : nullptr;
    ggml_tensor * indices = c.kv ? ggml_new_tensor_1d(ctx, GGML_TYPE_I64, c.tokens) : nullptr;
    ggml_tensor * result =
        c.kv ? ggml_set_rows_bf16_carrier(ctx, cache, rows, indices, GGML_SET_ROWS_BF16_CARRIER_ROWS) : rope;
    ggml_tensor * extra = nullptr;
    if (c.gate == rms_rope_gate::extra_rms) {
        extra = ggml_add(ctx, rms, rms);
    } else if (c.gate == rms_rope_gate::extra_rope) {
        extra = ggml_add(ctx, rope, rope);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    if (extra) {
        ggml_build_forward_expand(graph, extra);
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }
    if (c.gate == rms_rope_gate::aliased_cache) {
        // The original stages finish reading X before SET_ROWS overwrites it.
        // A fused direct K store would race those reads and must fall back.
        GGML_ASSERT(cache && ggml_nbytes(cache) <= ggml_nbytes(storage));
        cache->buffer = nullptr;
        cache->data   = nullptr;
        if (ggml_backend_tensor_alloc(buffer, cache, storage->data) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            return false;
        }
        result->buffer = nullptr;
        result->data   = nullptr;
        if (ggml_backend_view_init(result) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            return false;
        }
    }
    std::vector<int32_t> pos(c.tokens);
    for (int64_t i = 0; i < c.tokens; i++) {
        pos[i] = (int32_t) (i * 17 - 64);
    }
    ggml_backend_tensor_set(positions, pos.data(), 0, ggml_nbytes(positions));
    if (freq) {
        std::vector<float> factors(64);
        for (size_t i = 0; i < 64; i++) {
            factors[i] = 1.0f + (float) (i % 7) / 8.0f;
        }
        ggml_backend_tensor_set(freq, factors.data(), 0, ggml_nbytes(freq));
    }
    if (cache) {
        std::vector<uint16_t> zero(ggml_nelements(cache), 0x1234);
        ggml_backend_tensor_set(cache, zero.data(), 0, ggml_nbytes(cache));
        std::vector<int64_t> ids(c.tokens);
        for (int64_t i = 0; i < c.tokens; i++) {
            ids[i] = c.tokens - i + 8;
        }
        ggml_backend_tensor_set(indices, ids.data(), 0, ggml_nbytes(indices));
    }
    bool ok = true;
    output.clear();
    for (int pass = 0; pass < 2; pass++) {
        std::vector<float> input(ggml_nelements(storage), 0);
        const uint16_t     special[] = { 0,      0x8000, 1,      0x8001, 0x007f, 0x0080, 0x3f80, 0xbf80,
                                         0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1, 0xffc1, 0x7f81, 0xff81 };
        for (int64_t t = 0; t < c.tokens; t++) {
            for (int64_t h = 0; h < c.heads; h++) {
                for (int i = 0; i < 128; i++) {
                    const size_t offset = (size_t) t * storage->ne[0] + h * 128 + i;
                    const int    v      = (int) ((offset * 97 + 31 + pass * 13) % 4093) - 2046;
                    float        f      = (float) v / 1024.0f + ((i & 1) ? 0x1p-17f : -0x1p-17f);
                    if (c.edge) {
                        if (t % 9 == 0) {
                            f = (i & 1) ? -0.0f : 0.0f;
                        }
                        if (t % 9 == 1) {
                            f *= 0x1p-80f;
                        }
                        if (t % 9 == 2) {
                            f *= 0x1p60f;
                        }
                        if (t % 9 == 3) {
                            f = oracle_bf16_from_bits(special[(i + pass) % 16]);
                        }
                        if (t % 9 == 4) {
                            f = oracle_bf16_from_bits((uint16_t) (52 * 128 + (i % 128)));
                        }
                    }
                    input[offset] = f;
                }
            }
        }
        ggml_backend_tensor_set(storage, input.data(), 0, ggml_nbytes(storage));
        std::vector<float>    weights(128);
        std::vector<uint16_t> bits(128);
        for (int i = 0; i < 128; i++) {
            weights[i] = (float) (((i * 11 + pass) % 67) - 33) / 32.0f + ((i & 1) ? 0x1p-18f : -0x1p-18f);
            if (c.edge && pass == 1) {
                weights[i] = oracle_bf16_from_bits(special[(i + 5) % 16]);
            }
            bits[i] = oracle_bf16_bits(weights[i]);
        }
        ggml_backend_tensor_set(weight, c.bf16 ? static_cast<const void *>(bits.data()) : weights.data(), 0,
                                ggml_nbytes(weight));
        ggml_log_set(rms_rope_log, &marker);
        ggml_status status;
        if (c.gate == rms_rope_gate::split) {
            ggml_cgraph * first = ggml_new_graph(ctx);
            ggml_build_forward_expand(first, rms);
            ggml_cgraph * second = ggml_new_graph(ctx);
            ggml_graph_add_node(second, rope);
            if (rows) {
                ggml_graph_add_node(second, rows);
            }
            if (c.kv) {
                ggml_graph_add_node(second, result);
            }
            status = ggml_backend_graph_compute(backend, first);
            if (status == GGML_STATUS_SUCCESS) {
                status = ggml_backend_graph_compute(backend, second);
            }
        } else {
            status = ggml_backend_graph_compute(backend, graph);
        }
        ggml_log_set(nullptr, nullptr);
        ok &= status == GGML_STATUS_SUCCESS;
        if (!ok) {
            break;
        }
        auto append = [&](ggml_tensor * tensor) {
            const size_t offset = output.size();
            output.resize(offset + ggml_nelements(tensor));
            if (tensor->type == GGML_TYPE_BF16) {
                std::vector<uint16_t> values(ggml_nelements(tensor));
                ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
                for (size_t i = 0; i < values.size(); i++) {
                    output[offset + i] = oracle_bf16_from_bits(values[i]);
                }
            } else {
                ggml_backend_tensor_get(tensor, output.data() + offset, 0, ggml_nbytes(tensor));
            }
        };
        append(result);
        if (c.gate == rms_rope_gate::cast_output) {
            append(wf);
        }
        if (extra) {
            append(extra);
        }
        if (c.gate == rms_rope_gate::rms_output) {
            append(rms);
        }
        if (c.gate == rms_rope_gate::rope_output && c.kv) {
            append(rope);
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

bool test_metal_rms_rope_fusion() {
    ggml_backend_dev_t dev = find_metal_device();
    if (!dev) {
        const char * required = getenv("LLAMA_ROW4_REQUIRE_METAL_TESTS");
        return !required || strcmp(required, "0") == 0;
    }
    scoped_env_var disable("GGML_METAL_FUSION_DISABLE");
    scoped_env_var debug("GGML_METAL_FUSION_DEBUG");
    disable.set("1");
    ggml_backend_t baseline = ggml_backend_dev_init(dev, nullptr);
    disable.unset();
    debug.set("2");
    ggml_backend_t candidate = ggml_backend_dev_init(dev, nullptr);
    if (!baseline || !candidate) {
        ggml_backend_free(baseline);
        ggml_backend_free(candidate);
        return false;
    }
    const char *        strict   = getenv("LLAMA_ROW4_REQUIRE_M5_TENSOROPS_TESTS");
    const bool          required = strict && strcmp(strict, "0") != 0;
    const rms_rope_case cases[]  = {
        { 32, 512, false, false, true,  false, false, rms_rope_gate::normal          },
        { 8,  512, true,  true,  true,  false, false, rms_rope_gate::normal          },
        { 32, 33,  true,  false, true,  true,  true,  rms_rope_gate::normal          },
        { 8,  64,  false, true,  true,  true,  true,  rms_rope_gate::normal          },
        { 8,  31,  true,  true,  false, false, false, rms_rope_gate::normal          },
        { 8,  1,   false, false, false, false, false, rms_rope_gate::normal          },
        { 8,  32,  true,  true,  false, false, false, rms_rope_gate::rms_output      },
        { 8,  32,  true,  true,  false, false, false, rms_rope_gate::rope_output     },
        { 8,  32,  true,  true,  false, false, false, rms_rope_gate::extra_rms       },
        { 8,  32,  true,  true,  false, false, false, rms_rope_gate::extra_rope      },
        { 8,  32,  true,  true,  false, false, false, rms_rope_gate::split           },
        { 8,  32,  true,  true,  false, false, false, rms_rope_gate::nonqat          },
        { 8,  32,  true,  true,  false, false, false, rms_rope_gate::partial         },
        { 8,  32,  true,  true,  false, false, false, rms_rope_gate::cast_output     },
        { 8,  32,  true,  true,  false, false, false, rms_rope_gate::aliased_cache   },
        { 32, 33,  true,  false, true,  false, true,  rms_rope_gate::aliased_freq    },
        { 8,  64,  true,  true,  true,  false, true,  rms_rope_gate::aliased_freq    },
        { 8,  32,  true,  true,  false, false, false, rms_rope_gate::repeated_weight },
        { 32, 32,  false, false, false, false, false, rms_rope_gate::repeated_weight },
    };
    bool ok = true;
    for (const auto & c : cases) {
        std::vector<float> expected;
        std::vector<float> actual;
        rms_rope_marker    old_marker;
        rms_rope_marker    marker;
        const bool         ran  = run_rms_rope_backend(baseline, c, expected, old_marker) &&
                                  run_rms_rope_backend(candidate, c, actual, marker);
        const std::string  name = "RMS/RoPE h=" + std::to_string(c.heads) + " t=" + std::to_string(c.tokens) +
                                  " kv=" + std::to_string(c.kv) + " gate=" + std::to_string((int) c.gate);
        const bool         eligible =
            c.tokens >= 32 && (c.gate == rms_rope_gate::normal || c.gate == rms_rope_gate::rope_output ||
                               c.gate == rms_rope_gate::extra_rope || c.gate == rms_rope_gate::cast_output ||
                               c.gate == rms_rope_gate::aliased_freq);
        const bool kv    = eligible && c.kv &&
                           (c.gate == rms_rope_gate::normal || c.gate == rms_rope_gate::cast_output ||
                            c.gate == rms_rope_gate::aliased_freq);
        const bool exact = ran && compare_exact(name.c_str(), actual, expected);
        if (!exact || old_marker.hit || (marker.hit && !eligible) || (marker.kv && !kv) ||
            (required && (marker.hit != eligible || marker.kv != kv ||
                          (eligible && c.bf16 && c.gate != rms_rope_gate::cast_output && !marker.bf16)))) {
            fprintf(stderr, "%s failed exact=%d hit=%d eligible=%d kv=%d expected-kv=%d bf16=%d\n", name.c_str(), exact,
                    marker.hit.load(), eligible, marker.kv.load(), kv, marker.bf16.load());
            ok = false;
        }
    }
    ggml_backend_free(baseline);
    ggml_backend_free(candidate);
    printf("  Metal RMS128/RoPE/KV fusion: exact boundaries, graph gates, updated weights - %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace

namespace {

struct flash3_case {
    int  queries;
    int  cache;
    int  ratio;
    int  sequences;
    int  kv_sequences;
    int  mask_heads;
    bool strided;
    bool bias;
    bool softcap;
    bool sinks;
};

void flash3_gqa_log(enum ggml_log_level level, const char * text, void * user) {
    (void) level;
    if (strstr(text, "Flash3 GQA2 reuse:")) {
        static_cast<std::atomic<bool> *>(user)->store(true, std::memory_order_relaxed);
    }
}

bool run_flash3_backend(ggml_backend_t       backend,
                        const flash3_case &  c,
                        std::vector<float> & output,
                        std::atomic<bool> &  marker) {
    constexpr int  d        = 128;
    constexpr int  kv_heads = 2;
    const int      heads    = kv_heads * c.ratio;
    const int      stride   = d + (c.strided ? 8 : 0);
    ggml_context * ctx      = ggml_init({ 4 * 1024 * 1024, nullptr, true });
    ggml_tensor *  q_base   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, stride, heads, c.queries, c.sequences);
    ggml_tensor *  q =
        ggml_view_4d(ctx, q_base, d, c.queries, heads, c.sequences, q_base->nb[2], q_base->nb[1], q_base->nb[3], 0);
    ggml_tensor * k_base = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, stride, kv_heads, c.cache + 64, c.kv_sequences);
    ggml_tensor * v_base = ggml_dup_tensor(ctx, k_base);
    ggml_tensor * k =
        ggml_view_4d(ctx, k_base, d, c.cache, kv_heads, c.kv_sequences, k_base->nb[2], k_base->nb[1], k_base->nb[3], 0);
    ggml_tensor * v =
        ggml_view_4d(ctx, v_base, d, c.cache, kv_heads, c.kv_sequences, v_base->nb[2], v_base->nb[1], v_base->nb[3], 0);
    ggml_tensor * mask =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.cache, GGML_PAD(c.queries, 64), c.mask_heads, c.sequences);
    ggml_tensor * result = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / std::sqrt(float(d)), c.bias ? 0.125f : 0.0f,
                                               c.softcap ? 4.0f : 0.0f);
    ggml_flash_attn_ext_set_fairy2i_flash3(result, true);
    if (c.sinks) {
        result->src[4] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    if (!ggml_backend_supports_op(backend, result)) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }
    std::vector<float>       queries(ggml_nelements(q_base));
    std::vector<ggml_bf16_t> keys(ggml_nelements(k_base));
    std::vector<ggml_bf16_t> values(keys.size());
    std::vector<ggml_fp16_t> masks(ggml_nelements(mask));
    for (size_t i = 0; i < queries.size(); ++i) {
        queries[i] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(std::sin(float(i % 977) * 0.19f)));
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i]   = ggml_fp32_to_bf16(std::cos(float(i % 733) * 0.07f));
        values[i] = ggml_fp32_to_bf16(std::sin(float(i % 613) * 0.11f));
    }
    for (size_t i = 0; i < masks.size(); ++i) {
        const int   pos        = int(i % c.cache);
        const int   query      = int((i / c.cache) % mask->ne[1]);
        // Causal future blocks, a fully masked interior C64 block, and
        // finite mask offsets exercise the unchanged online softmax order.
        const bool  masked     = pos > c.cache - c.queries + query || (pos >= 128 && pos < 192);
        const float mask_value = pos % 17 == 0 ? -0.25f : 0.0f;
        masks[i]               = ggml_fp32_to_fp16(masked ? -INFINITY : mask_value);
    }
    ggml_backend_tensor_set(q_base, queries.data(), 0, ggml_nbytes(q_base));
    ggml_backend_tensor_set(k_base, keys.data(), 0, ggml_nbytes(k_base));
    ggml_backend_tensor_set(v_base, values.data(), 0, ggml_nbytes(v_base));
    ggml_backend_tensor_set(mask, masks.data(), 0, ggml_nbytes(mask));
    if (c.sinks) {
        const std::vector<float> sink_values(heads, -0.5f);
        ggml_backend_tensor_set(result->src[4], sink_values.data(), 0, ggml_nbytes(result->src[4]));
    }
    ggml_log_set(flash3_gqa_log, &marker);
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(backend);
    ggml_log_set(nullptr, nullptr);
    if (ok) {
        output.resize(ggml_nelements(result));
        ggml_backend_tensor_get(result, output.data(), 0, ggml_nbytes(result));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

bool test_metal_flash3_gqa_reuse() {
    ggml_backend_dev_t dev = find_metal_device();
    if (!dev) {
        const char * required = getenv("LLAMA_ROW4_REQUIRE_METAL_TESTS");
        return !required || strcmp(required, "0") == 0;
    }
    scoped_env_var disable("GGML_METAL_FUSION_DISABLE");
    scoped_env_var debug("GGML_METAL_FUSION_DEBUG");
    disable.set("1");
    ggml_backend_t baseline = ggml_backend_dev_init(dev, nullptr);
    disable.unset();
    debug.set("2");
    ggml_backend_t candidate = ggml_backend_dev_init(dev, nullptr);
    if (!baseline || !candidate) {
        ggml_backend_free(baseline);
        ggml_backend_free(candidate);
        return false;
    }
    const char *   strict     = getenv("LLAMA_ROW4_REQUIRE_M5_TENSOROPS_TESTS");
    const bool     required   = strict && strcmp(strict, "0") != 0;
    ggml_context * probe_ctx  = ggml_init({ 1024 * 1024, nullptr, true });
    ggml_tensor *  probe_q    = ggml_new_tensor_3d(probe_ctx, GGML_TYPE_F32, 128, 512, 8);
    ggml_tensor *  probe_k    = ggml_new_tensor_3d(probe_ctx, GGML_TYPE_BF16, 128, 1024, 2);
    ggml_tensor *  probe_v    = ggml_dup_tensor(probe_ctx, probe_k);
    ggml_tensor *  probe_mask = ggml_new_tensor_2d(probe_ctx, GGML_TYPE_F16, 1024, 512);
    ggml_tensor *  probe = ggml_flash_attn_ext(probe_ctx, probe_q, probe_k, probe_v, probe_mask, 0.125f, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_fairy2i_flash3(probe, true);
    const bool supported = ggml_backend_supports_op(candidate, probe);
    ggml_free(probe_ctx);
    if (!supported) {
        ggml_backend_free(baseline);
        ggml_backend_free(candidate);
        printf("  Metal BF16 Flash3 GQA reuse: SKIP (BF16 attention unavailable)\n");
        return !required;
    }
    const flash3_case cases[] = {
        { 512,  1024, 4, 1, 1, 1, false, false, false, false },
        { 513,  2048, 4, 1, 1, 1, true,  false, false, false },
        { 1024, 2048, 4, 2, 2, 1, true,  false, false, false },
        { 2048, 2048, 4, 1, 1, 1, false, false, false, false },
        { 512,  512,  4, 1, 1, 1, false, false, false, false },
        { 511,  1024, 4, 1, 1, 1, false, false, false, false },
        { 33,   1024, 4, 1, 1, 1, true,  false, false, false },
        { 1,    1024, 4, 1, 1, 1, false, false, false, false },
        { 512,  1024, 2, 1, 1, 1, false, false, false, false },
        { 512,  1024, 4, 1, 1, 2, false, false, false, false },
        { 512,  1024, 4, 1, 1, 1, false, true,  false, false },
        { 512,  1024, 4, 1, 1, 1, false, false, true,  false },
        { 512,  1024, 4, 1, 1, 1, false, false, false, true  },
    };
    bool ok = true;
    for (const auto & c : cases) {
        std::vector<float> expected;
        std::vector<float> actual;
        std::atomic<bool>  baseline_hit{ false };
        std::atomic<bool>  candidate_hit{ false };
        const bool         ran = run_flash3_backend(baseline, c, expected, baseline_hit) &&
                                 run_flash3_backend(candidate, c, actual, candidate_hit);
        const bool eligible    = c.queries >= 512 && c.cache >= 1024 && c.ratio == 4 && c.sequences == c.kv_sequences &&
                                 c.mask_heads == 1 && !c.bias && !c.softcap && !c.sinks;
        const std::string label = "Flash3 GQA reuse Q=" + std::to_string(c.queries) + " KV=" + std::to_string(c.cache);
        ok                      = ran && compare_exact(label.c_str(), actual, expected) && ok;
        if (baseline_hit || (candidate_hit && !eligible) || (required && eligible && !candidate_hit)) {
            fprintf(stderr, "%s unexpected reuse path: baseline=%d candidate=%d eligible=%d\n", label.c_str(),
                    baseline_hit.load(), candidate_hit.load(), eligible);
            ok = false;
        }
    }
    ggml_backend_free(baseline);
    ggml_backend_free(candidate);
    printf("  Metal BF16 Flash3 GQA reuse: exact carriers, strides, tails, masks, graph gates - %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace

namespace {

struct cache_test_marker {
    std::atomic<int> builds{ 0 };
    std::atomic<int> uses{ 0 };
};

void cache_test_log(enum ggml_log_level level, const char * text, void * user) {
    (void) level;
    auto & marker = *static_cast<cache_test_marker *>(user);
    if (strstr(text, "Row4 persistent INT4 cache build:")) {
        ++marker.builds;
    }
    if (strstr(text, "Row4 persistent INT4 cache reuse:")) {
        ++marker.uses;
    }
}

bool test_row4_cache_updates() {
    const char * strict = getenv("LLAMA_ROW4_REQUIRE_M5_TENSOROPS_TESTS");
    if (!strict || strcmp(strict, "0") == 0) {
        printf("  Metal Row4 persistent INT4 cache: SKIP (strict M5 suite only)\n");
        return true;
    }
    ggml_backend_load_all();
    ggml_backend_dev_t dev = find_metal_device();
    if (!dev) {
        return false;
    }
    scoped_env_var enable("GGML_METAL_ROW4_INT4_CACHE");
    scoped_env_var debug("GGML_METAL_FUSION_DEBUG");
    enable.set("0");
    ggml_backend_t baseline = ggml_backend_dev_init(dev, nullptr);
    enable.set("1");
    debug.set("2");
    ggml_backend_t candidate = ggml_backend_dev_init(dev, nullptr);
    if (!baseline || !candidate) {
        return false;
    }
    bool          ok = true;
    constexpr int o  = 128;
    constexpr int k  = 512;
    constexpr int b  = 512;
    for (int lifetime = 0; lifetime < 3 && ok; ++lifetime) {
        ggml_context *        weights = ggml_init({ 1024 * 1024, nullptr, true });
        ggml_tensor *         codes = ggml_new_tensor_4d(weights, GGML_TYPE_ROW4_CODES_PAIR2, 128, 8, k / 256, o / 32);
        ggml_backend_buffer_t wb    = ggml_backend_alloc_ctx_tensors(weights, candidate);
        ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_context * ctx    = ggml_init({ 4 * 1024 * 1024, nullptr, true });
        ggml_tensor *  x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, b);
        ggml_tensor *  sw     = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, o);
        ggml_tensor *  fresh  = ggml_dup_tensor(ctx, codes);
        ggml_tensor *  copy   = ggml_cpy(ctx, fresh, codes);
        ggml_tensor *  alias  = ggml_view_1d(ctx, codes, 128, 128);
        ggml_tensor *  result = ggml_row4_linear(ctx, x, codes, sw, o, k);
        ggml_cgraph *  normal = ggml_new_graph(ctx);
        ggml_build_forward_expand(normal, result);
        ggml_cgraph * writer = ggml_new_graph(ctx);
        ggml_build_forward_expand(writer, copy);
        ggml_build_forward_expand(writer, result);
        ggml_cgraph * writer_only = ggml_new_graph(ctx);
        ggml_build_forward_expand(writer_only, copy);
        ggml_backend_buffer_t buf    = ggml_backend_alloc_ctx_tensors(ctx, candidate);
        auto                  packed = make_row4_pair2_codes(o, k);
        for (auto & value : packed) {
            value ^= uint8_t(lifetime * 37);
        }
        auto       input  = make_input(k, b);
        const auto scales = make_row4_scales(o);
        ggml_backend_tensor_set(codes, packed.data(), 0, packed.size());
        ggml_backend_tensor_set(fresh, packed.data(), 0, packed.size());
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(sw, scales.data(), 0, ggml_nbytes(sw));
        for (int step = 0; step < 12 && ok; ++step) {
            const bool graph_write    = step == 7;
            const bool cache_fallback = graph_write || step == 10;
            if (step == 10 || step == 11) {
                ggml_backend_buffer_set_usage(
                    wb, step == 10 ? GGML_BACKEND_BUFFER_USAGE_ANY : GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
            if (step == 2) {
                for (int i = 0; i < 256; ++i) {
                    packed[i] ^= 0xA5;
                }
                ggml_backend_tensor_set(codes, packed.data(), 0, 256);
            } else if (step == 3) {
                ggml_backend_tensor_memset(alias, 0x37, 16, 64);
            } else if (step == 4) {
                ggml_backend_tensor_set_async(candidate, codes, packed.data() + 192, 192, 24);
                ggml_backend_synchronize(candidate);
            } else if (step == 5) {
                ggml_backend_buffer_clear(wb, 0);
            } else if (step == 6) {
                input[0] += 1.25f;
                ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
            } else if (step == 8) {
                // A separate graph with caching disabled must invalidate a
                // shared weight's existing entry even without any Row4 op.
                for (auto & value : packed) {
                    value ^= 0x6F;
                }
                ggml_backend_tensor_set(fresh, packed.data(), 0, packed.size());
                ok = ggml_backend_graph_compute(baseline, writer_only) == GGML_STATUS_SUCCESS && ok;
                ggml_backend_synchronize(baseline);
            }
            ggml_cgraph *      graph = graph_write ? writer : normal;
            std::vector<float> expected(ggml_nelements(result));
            std::vector<float> actual(expected.size());
            cache_test_marker  marker;
            ok = ggml_backend_graph_compute(baseline, graph) == GGML_STATUS_SUCCESS && ok;
            ggml_backend_tensor_get(result, expected.data(), 0, ggml_nbytes(result));
            ggml_log_set(cache_test_log, &marker);
            ok = ggml_backend_graph_compute(candidate, graph) == GGML_STATUS_SUCCESS && ok;
            ggml_backend_synchronize(candidate);
            ggml_log_set(nullptr, nullptr);
            ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
            const bool        expect_build = step == 0 || (step >= 2 && step <= 5) || step == 8;
            const std::string label =
                "Row4 INT4 cache lifetime=" + std::to_string(lifetime) + " step=" + std::to_string(step);
            ok = compare_exact(label.c_str(), actual, expected) && ok;
            if (marker.uses != (cache_fallback ? 0 : 1) || marker.builds != (expect_build ? 1 : 0)) {
                fprintf(stderr, "%s unexpected cache builds=%d uses=%d\n", label.c_str(), marker.builds.load(),
                        marker.uses.load());
                ok = false;
            }
            printf("%s builds=%d uses=%d exact=%d\n", label.c_str(), marker.builds.load(), marker.uses.load(), ok);
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_buffer_free(wb);
        ggml_free(weights);
    }
    ggml_backend_free(candidate);
    ggml_backend_free(baseline);
    printf("  Metal Row4 persistent INT4 cache: warm reuse, writes, aliases, graph gates, lifetimes - %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    ggml_cpu_init();

    printf("========================================\n");
    printf("Qwen3 Row4/W8A8 Unit Tests\n");
    printf("========================================\n");

    int failed = 0;
    failed += !test_codebook();
    failed += !test_split8_layout();
    failed += !test_pair2_layout();
    failed += !test_bf16_and_rounding();
    failed += !test_activation_profile();
    failed += !test_row4_lut16_oracle();
    failed += !test_int32_extremes();
    failed += !test_fused_boundaries();
    failed += !test_opaque_type_isolation();
    failed += !test_cpu_operator_matrix();
    failed += !test_metal_row4_pair2_decode_production();
    failed += !test_metal_operator_matrix();
    failed += !test_metal_rms_row4_fusion();
    failed += !test_metal_rms_rope_fusion();
    failed += !test_metal_flash3_gqa_reuse();
    failed += !test_row4_cache_updates();
    failed += !test_metal_row4_preexpand_lookahead();
    failed += !test_metal_row4_swiglu_fusion();
    failed += !test_metal_row4_swiglu_down_fusion();
    failed += !test_metal_row4_swiglu_a8_fusion();
    failed += !test_metal_row4_decode_residual_fusion();
    failed += !test_metal_row4_prefill_residual();
    failed += !test_metal_real_shape_matrix();

    printf("========================================\n");
    printf("%s (%d failed)\n", failed == 0 ? "All Row4 tests PASSED" : "Row4 tests FAILED", failed);
    printf("========================================\n");
    return failed == 0 ? 0 : 1;
}
