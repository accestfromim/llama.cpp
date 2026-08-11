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
            logical[(size_t) group * (size_t) k + (size_t) ik] = (uint8_t) ((5 * group + 3 * ik) & 15);
        }
    }
    return pack_row4_codes(logical, o, k);
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
                                 ggml_backend_t                backend_override = nullptr) {
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
    } else {
        codes  = ggml_new_tensor_4d(ctx, GGML_TYPE_I8, 128, 16, k / 128, o / 16);
        scales = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, o);
        result = ggml_w8a8_linear(ctx, x, codes, scales, o, k);
    }

    int32_t op_params[3];
    memcpy(op_params, result->op_params, sizeof(op_params));
    bool ok = result->type == GGML_TYPE_F32 && result->ne[0] == o && result->ne[1] == tokens &&
              result->op == (kind == linear_kind::row4 ? GGML_OP_ROW4_LINEAR : GGML_OP_W8A8_LINEAR) &&
              result->src[0] == x && result->src[1] == codes && result->src[2] == scales && op_params[0] == 1 &&
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
    if (kind == linear_kind::row4) {
        ggml_backend_tensor_set(codes, row4_codes.data(), 0, row4_codes.size());
        ggml_backend_tensor_set(scales, row4_scales.data(), 0, row4_scales.size() * sizeof(uint16_t));
    } else {
        ggml_backend_tensor_set(codes, w8_codes.data(), 0, w8_codes.size());
        ggml_backend_tensor_set(scales, w8_scales.data(), 0, w8_scales.size() * sizeof(float));
    }

    ok                       = codes->extra == nullptr && scales->extra == nullptr && ok;
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s path=%s graph compute failed: %s\n", kind == linear_kind::row4 ? "Row4" : "W8A8",
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

    ggml_tensor * raw_copy          = ggml_cpy(ctx, codes_a, codes_b);
    ggml_tensor * row4_to_f32       = ggml_cpy(ctx, codes_a, f32);
    ggml_tensor * f32_to_row4       = ggml_cpy(ctx, f32, codes_b);
    ggml_tensor * add               = ggml_add(ctx, codes_a, codes_b);
    ggml_tensor * row4_rhs          = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 1, 1, 8);
    ggml_tensor * generic_row4      = ggml_mul_mat(ctx, codes_a, row4_rhs);
    ggml_tensor * indices           = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 1, 1, 8);
    ggml_tensor * get_rows          = ggml_get_rows(ctx, codes_a, indices);

    auto generic_ops_are_isolated = [&](ggml_backend_t candidate) {
        return !ggml_backend_supports_op(candidate, row4_to_f32) && !ggml_backend_supports_op(candidate, f32_to_row4) &&
               !ggml_backend_supports_op(candidate, add) && !ggml_backend_supports_op(candidate, generic_row4) &&
               !ggml_backend_supports_op(candidate, get_rows);
    };

    bool ok = ggml_backend_supports_op(backend, raw_copy) && generic_ops_are_isolated(backend);
    if (!ok) {
        fprintf(stderr, "CPU ROW4_CODES escaped its dedicated operator contract\n");
    }

    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const bool         device_ok =
            !ggml_backend_dev_supports_op(dev, row4_to_f32) && !ggml_backend_dev_supports_op(dev, f32_to_row4) &&
            !ggml_backend_dev_supports_op(dev, add) && !ggml_backend_dev_supports_op(dev, generic_row4) &&
            !ggml_backend_dev_supports_op(dev, get_rows);
        if (!device_ok) {
            fprintf(stderr, "%s device advertised a generic ROW4_CODES operation\n", ggml_backend_dev_name(dev));
            ok = false;
        }
    }

    ggml_backend_dev_t metal_dev = find_metal_device();
    if (metal_dev) {
        ggml_backend_t metal    = ggml_backend_dev_init(metal_dev, nullptr);
        const bool     metal_ok = metal && generic_ops_are_isolated(metal);
        if (!metal_ok) {
            fprintf(stderr, "Metal ROW4_CODES escaped its dedicated operator contract\n");
            ok = false;
        }
        if (metal) {
            ggml_backend_free(metal);
        }
    } else {
        printf("  Metal opaque ROW4_CODES generic-op isolation: SKIP (Metal backend unavailable)\n");
    }

    ggml_free(ctx);
    ggml_backend_free(backend);
    printf("  CPU%s opaque ROW4_CODES generic-op isolation - %s\n", metal_dev ? "/Metal" : "", ok ? "PASS" : "FAIL");
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
    std::atomic<bool> hit = false;
};

static void gate_up_fusion_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    constexpr const char * marker = "fuse Row4 gate_up + VIEW + QAT SILU_EXACT + VIEW + QAT MUL_EXACT + ROW4_LINEAR";
    if (strstr(text, marker)) {
        static_cast<gate_up_fusion_log_marker *>(user_data)->hit.store(true, std::memory_order_relaxed);
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
                                         bool                          w8_down             = false,
                                         bool *                        gate_up_fusion_hit  = nullptr,
                                         bool                          mark_gate_up_output = false) {
    constexpr int64_t INPUT_K   = 128;
    constexpr int64_t N_FF      = 128;
    constexpr int64_t GATE_UP_O = 2 * N_FF;
    constexpr int64_t DOWN_O    = 128;

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
        down_codes  = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES, 64, 4, N_FF / ROW4_TILE_K, DOWN_O / ROW4_TILE_O);
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
    if (gate_up_fusion_hit != nullptr) {
        ggml_log_set(gate_up_fusion_log_callback, &marker);
    }
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (gate_up_fusion_hit != nullptr) {
        *gate_up_fusion_hit = marker.hit.load(std::memory_order_relaxed);
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

    for (int64_t tokens : { 1, 3, 8, 9, 32, 64 }) {
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
        bool               gate_up_fusion_hit = false;
        ok =
            metal_fused &&
            run_row4_swiglu_down_backend(fused, metal_fused, input, gate_up_codes, gate_up_scales, down_codes,
                                         down_scales, tokens, true, false, false, &gate_up_fusion_hit) &&
            compare_exact(("Row4 QAT SwiGLU-down fused/oracle B=" + std::to_string(tokens)).c_str(), fused, oracle) &&
            compare_exact(("Row4 QAT SwiGLU-down fused/unfused B=" + std::to_string(tokens)).c_str(), fused, unfused) &&
            ok;
        const bool expect_gate_up_fusion = tokens > 8 && tokens % 32 == 0;
        if (gate_up_fusion_hit != expect_gate_up_fusion) {
            fprintf(stderr, "Row4 gate-up producer fusion hit mismatch B=%lld: actual=%d expected=%d\n",
                    (long long) tokens, (int) gate_up_fusion_hit, (int) expect_gate_up_fusion);
            ok = false;
        }
        if (metal_fused) {
            ggml_backend_free(metal_fused);
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

struct row4_residual_outputs {
    std::vector<float> row4;
    std::vector<float> add;
};

struct row4_residual_case {
    int64_t tokens         = 1;
    bool    qat            = true;
    bool    reverse        = false;
    bool    row4_output    = false;
    bool    extra_consumer = false;
    bool    graph_gap      = false;
};

static bool run_row4_residual_backend(row4_residual_outputs &       output,
                                      ggml_backend_t                backend,
                                      const std::vector<float> &    input,
                                      const std::vector<uint8_t> &  codes_data,
                                      const std::vector<uint16_t> & scales_data,
                                      const std::vector<float> &    residual_data,
                                      int64_t                       o,
                                      int64_t                       k,
                                      const row4_residual_case &    test_case) {
    const ggml_init_params params = {
        /*.mem_size   =*/4 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * x        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, test_case.tokens);
    ggml_tensor * codes    = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES, 64, 4, k / ROW4_TILE_K, o / ROW4_TILE_O);
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

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
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
    ggml_backend_dev_t dev = find_metal_device();
    if (!dev) {
        const char * required = getenv("LLAMA_ROW4_REQUIRE_METAL_TESTS");
        if (required && strcmp(required, "0") != 0) {
            fprintf(stderr, "LLAMA_ROW4_REQUIRE_METAL_TESTS is set, but no Metal device is available\n");
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

    bool ok = true;
    for (int64_t tokens : { 1, 2, 8, 9, 16, 17, 31, 32, 33, 64, 96, 128 }) {
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
        const std::vector<uint8_t>  codes = pack_row4_codes(logical, O, KROW4);
        const std::vector<uint16_t> scales((size_t) O, oracle_bf16_bits(1.0f));
        const std::vector<float>    input((size_t) KROW4 * TOKENS, 1.0f);
        const std::vector<float>    expected = oracle_row4_linear(input, codes, scales, O, KROW4, TOKENS);
        std::vector<float>          actual;
        if (!run_operator_backend(actual, linear_kind::row4, input, codes, scales, {}, {}, O, KROW4, TOKENS, nullptr,
                                  false, metal) ||
            !compare_exact("Row4 Metal K=12288 B=9 maximum/cancellation", actual, expected)) {
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
        bool         packed_n32_boundaries;
    };

    const row4_shape shapes[] = {
        { "qkv",     6144,  4096,  true  },
        { "o",       4096,  4096,  false },
        { "gate_up", 24576, 4096,  false },
        { "down",    4096,  12288, true  },
    };

    bool ok = true;
    for (const row4_shape & shape : shapes) {
        const std::vector<uint8_t> codes = make_row4_codes(shape.o, shape.k);
        std::vector<uint16_t>      scales((size_t) shape.o);
        for (int64_t row = 0; row < shape.o; ++row) {
            const float scale    = (row & 1) ? -0.03125f * (float) (1 + row % 3) : 0.03125f * (float) (1 + row % 3);
            scales[(size_t) row] = oracle_bf16_bits(scale);
        }

        const std::vector<float> input_two    = make_input(shape.k, 2);
        const std::vector<float> expected_two = oracle_row4_linear(input_two, codes, scales, shape.o, shape.k, 2);
        for (int64_t tokens : { 1, 9, 16, 17, 31, 32, 33, 64, 96, 128 }) {
            if (!shape.packed_n32_boundaries && tokens != 1 && tokens != 9 && tokens != 32) {
                continue;
            }
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
        for (int64_t tokens : { 9, 32 }) {
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

            for (int64_t tokens : { 1, 32 }) {
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

int main() {
    ggml_cpu_init();

    printf("========================================\n");
    printf("Qwen3 Row4/W8A8 Unit Tests\n");
    printf("========================================\n");

    int failed = 0;
    failed += !test_codebook();
    failed += !test_split8_layout();
    failed += !test_bf16_and_rounding();
    failed += !test_activation_profile();
    failed += !test_int32_extremes();
    failed += !test_fused_boundaries();
    failed += !test_opaque_type_isolation();
    failed += !test_cpu_operator_matrix();
    failed += !test_metal_row4_swiglu_fusion();
    failed += !test_metal_row4_swiglu_down_fusion();
    failed += !test_metal_row4_decode_residual_fusion();
    failed += !test_metal_operator_matrix();
    failed += !test_metal_real_shape_matrix();

    printf("========================================\n");
    printf("%s (%d failed)\n", failed == 0 ? "All Row4 tests PASSED" : "Row4 tests FAILED", failed);
    printf("========================================\n");
    return failed == 0 ? 0 : 1;
}
