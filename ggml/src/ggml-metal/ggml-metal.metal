#define GGML_COMMON_DECL_METAL
#define GGML_COMMON_IMPL_METAL
#if defined(GGML_METAL_EMBED_LIBRARY)
__embed_ggml-common.h__
#else
#include "ggml-common.h"
#endif
#include "ggml-metal-impl.h"

#include <metal_stdlib>

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal/ggml-metal.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal/ggml-metal.metal
//
#if __METAL_VERSION__ < 310 && defined(GGML_METAL_HAS_BF16)
#undef GGML_METAL_HAS_BF16
#endif

#if defined(GGML_METAL_HAS_BF16)
typedef matrix<bfloat, 4, 4> bfloat4x4;
#endif

constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant static float kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

static inline int best_index_int8(int n, constant float * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static inline float e8m0_to_fp32(uint8_t x) {
    uint32_t bits;

    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }

    return as_type<float>(bits);
}

static inline float fairy2i_bf16_to_f32(ushort h) {
    return as_type<float>(((uint) h) << 16);
}

static inline ushort fairy2i_f32_bits_to_bf16_rne(uint bits) {
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return (ushort) ((bits >> 16) | 64U);
    }
    return (ushort) ((bits + (0x7fffU + ((bits >> 16) & 1U))) >> 16);
}

static inline ushort fairy2i_f32_to_bf16(float x) {
    return fairy2i_f32_bits_to_bf16_rne(as_type<uint>(x));
}

static inline ushort4 fairy2i_f32_to_bf16(float4 x) {
    const uint4 bits = as_type<uint4>(x);
    const uint4 rounded =
        (bits + (uint4(0x7fffU) + ((bits >> uint4(16)) & uint4(1)))) >> uint4(16);
    const uint4 quiet_nan = (bits >> uint4(16)) | uint4(64U);
    return ushort4(select(rounded, quiet_nan, (bits & uint4(0x7fffffffU)) > uint4(0x7f800000U)));
}

static inline float fairy2i_round_to_bf16_f32(float x) {
    return fairy2i_bf16_to_f32(fairy2i_f32_to_bf16(x));
}

static inline float4 fairy2i_round_to_bf16_f32(float4 x) {
    const uint4 bits = as_type<uint4>(x);
    const uint4 rounded =
        (bits + (uint4(0x7fffU) + ((bits >> uint4(16)) & uint4(1)))) & uint4(0xffff0000U);
    const uint4 quiet_nan = (bits & uint4(0xffff0000U)) | uint4(0x00400000U);
    return as_type<float4>(
        select(rounded, quiet_nan, (bits & uint4(0x7fffffffU)) > uint4(0x7f800000U)));
}

static inline uint fairy2i_shift_right_jam(uint value, uint shift) {
    if (shift == 0) {
        return value;
    }
    if (shift >= 32) {
        return value != 0 ? 1U : 0U;
    }
    const uint mask = (1U << shift) - 1U;
    return (value >> shift) | ((value & mask) != 0 ? 1U : 0U);
}

static inline uint fairy2i_shift_right_rne(uint value, uint shift) {
    if (shift == 0) {
        return value;
    }
    if (shift >= 32) {
        return 0;
    }

    const uint result = value >> shift;
    const uint halfway = 1U << (shift - 1);
    const uint lost   = value & ((1U << shift) - 1U);
    return result + (lost > halfway || (lost == halfway && (result & 1U) != 0) ? 1U : 0U);
}

static inline ulong fairy2i_shift_right_rne_u64(ulong value, uint shift) {
    if (shift == 0) {
        return value;
    }
    if (shift >= 64) {
        return 0;
    }

    const ulong result  = value >> shift;
    const ulong halfway = (ulong) 1 << (shift - 1);
    const ulong lost    = value & (((ulong) 1 << shift) - 1);
    return result + (lost > halfway || (lost == halfway && (result & 1UL) != 0) ? 1UL : 0UL);
}

// Returns the exponent of the least-significant bit in a finite non-zero BF16 value, encoded so
// `metric - 134` is the binary exponent. Zero has no product quantum and maps to 255. BF16
// subnormals and non-finite values map to zero so the caller can force the software path even when
// the other operand has a very large exponent.
static inline uint fairy2i_bf16_product_metric(ushort value) {
    const uint abs_value = (uint) value & 0x7fffU;
    if (abs_value == 0) {
        return 255U;
    }
    const uint exponent = abs_value >> 7;
    if (exponent == 0 || exponent == 0xffU) {
        return 0U;
    }
    return exponent;
}

static inline bool fairy2i_product_metrics_require_software(uint lhs, uint rhs) {
    const bool lhs_nonzero = lhs != 255U;
    const bool rhs_nonzero = rhs != 255U;
    return lhs == 0U || rhs == 0U ||
           (lhs_nonzero && rhs_nonzero && lhs + rhs < 142U);
}

// A non-zero BF16 add can lower its smallest input exponent field by at most seven.
// The U/W and Aij hierarchy crosses two add boundaries. Stages that can underflow or
// overflow the hierarchy conservatively force software replay.
static inline uint fairy2i_bf16_two_add_coefficient_metric_bound(uint min_stage_metric, uint max_stage_exponent) {
    if (min_stage_metric == 255U) {
        return 255U;
    }
    if (min_stage_metric <= 14U || max_stage_exponent >= 253U) {
        return 0U;
    }
    return min_stage_metric - 14U;
}

// Bundle W1 has one BF16 add boundary between its signed U/W stages and Aij.
static inline uint fairy2i_bf16_one_add_coefficient_metric_bound(uint min_stage_metric, uint max_stage_exponent) {
    if (min_stage_metric == 255U) {
        return 255U;
    }
    if (min_stage_metric <= 7U || max_stage_exponent >= 254U) {
        return 0U;
    }
    return min_stage_metric - 7U;
}

// BF16 inputs contain at most eight significant bits, so their product is represented exactly by
// F32 unless it must be rounded into the F32 subnormal range. Build the result from payload bits to
// avoid Apple GPU F32/BF16 flush-to-zero behavior.
static inline uint fairy2i_mul_bf16_to_f32_bits_rne(ushort a, ushort b) {
    const uint sign_a = (uint) a >> 15;
    const uint sign_b = (uint) b >> 15;
    const uint abs_a  = (uint) a & 0x7fffU;
    const uint abs_b  = (uint) b & 0x7fffU;
    const uint exp_a  = abs_a >> 7;
    const uint exp_b  = abs_b >> 7;
    const uint frac_a = abs_a & 0x7fU;
    const uint frac_b = abs_b & 0x7fU;
    const uint sign   = (sign_a ^ sign_b) << 31;

    if (exp_a == 0xffU || exp_b == 0xffU) {
        const bool nan_a = exp_a == 0xffU && frac_a != 0;
        const bool nan_b = exp_b == 0xffU && frac_b != 0;
        if (nan_a || nan_b || ((abs_a == 0x7f80U || abs_b == 0x7f80U) && (abs_a == 0 || abs_b == 0))) {
            return 0x7fc00000U;
        }
        return sign | 0x7f800000U;
    }
    if (abs_a == 0 || abs_b == 0) {
        return sign;
    }

    const uint sig_a = exp_a == 0 ? frac_a : 0x80U | frac_a;
    const uint sig_b = exp_b == 0 ? frac_b : 0x80U | frac_b;
    const int  e_a   = exp_a == 0 ? -133 : (int) exp_a - 134;
    const int  e_b   = exp_b == 0 ? -133 : (int) exp_b - 134;
    const uint product = sig_a * sig_b;
    const int  msb     = 31 - (int) clz(product);
    const int  result_e = e_a + e_b + msb;

    if (result_e > 127) {
        return sign | 0x7f800000U;
    }
    if (result_e >= -126) {
        const uint significand = product << (23 - msb);
        return sign | ((uint) (result_e + 127) << 23) | (significand & 0x7fffffU);
    }

    const int subnormal_shift = e_a + e_b + 149;
    uint      fraction;
    if (subnormal_shift >= 0) {
        fraction = product << subnormal_shift;
    } else {
        fraction = fairy2i_shift_right_rne(product, (uint) -subnormal_shift);
    }
    if (fraction >= 0x800000U) {
        return sign | 0x00800000U;
    }
    return sign | fraction;
}

static inline uint fairy2i_mul_f32_bits_rne(uint a, uint b) {
    const uint sign_a = a >> 31;
    const uint sign_b = b >> 31;
    const uint abs_a  = a & 0x7fffffffU;
    const uint abs_b  = b & 0x7fffffffU;
    const uint exp_a  = abs_a >> 23;
    const uint exp_b  = abs_b >> 23;
    const uint frac_a = abs_a & 0x7fffffU;
    const uint frac_b = abs_b & 0x7fffffU;
    const uint sign   = (sign_a ^ sign_b) << 31;

    if (exp_a == 0xffU || exp_b == 0xffU) {
        const bool nan_a = exp_a == 0xffU && frac_a != 0;
        const bool nan_b = exp_b == 0xffU && frac_b != 0;
        if (nan_a || nan_b || ((exp_a == 0xffU || exp_b == 0xffU) && (abs_a == 0 || abs_b == 0))) {
            return 0x7fc00000U;
        }
        return sign | 0x7f800000U;
    }
    if (abs_a == 0 || abs_b == 0) {
        return sign;
    }

    const ulong sig_a = exp_a == 0 ? (ulong) frac_a : (ulong) (0x800000U | frac_a);
    const ulong sig_b = exp_b == 0 ? (ulong) frac_b : (ulong) (0x800000U | frac_b);
    const int e_a = exp_a == 0 ? -149 : (int) exp_a - 150;
    const int e_b = exp_b == 0 ? -149 : (int) exp_b - 150;
    const ulong product = sig_a * sig_b;
    const uint product_hi = (uint) (product >> 32);
    const int msb = product_hi != 0 ? 32 + 31 - (int) clz(product_hi) :
                                      31 - (int) clz((uint) product);
    int result_e = e_a + e_b + msb;

    if (result_e > 127) {
        return sign | 0x7f800000U;
    }
    if (result_e >= -126) {
        ulong significand = msb > 23 ? fairy2i_shift_right_rne_u64(product, (uint) (msb - 23)) :
                                       product << (23 - msb);
        if (significand >= 0x1000000UL) {
            significand >>= 1;
            ++result_e;
            if (result_e > 127) {
                return sign | 0x7f800000U;
            }
        }
        return sign | ((uint) (result_e + 127) << 23) | ((uint) significand & 0x7fffffU);
    }

    const int subnormal_shift = e_a + e_b + 149;
    ulong fraction = subnormal_shift >= 0 ? product << subnormal_shift :
                                            fairy2i_shift_right_rne_u64(product, (uint) -subnormal_shift);
    if (fraction >= 0x800000UL) {
        return sign | 0x00800000U;
    }
    return sign | (uint) fraction;
}

static inline uint fairy2i_mul_f32_bits_rne_ftz_safe(uint a, uint b) {
    const uint abs_a = a & 0x7fffffffU;
    const uint abs_b = b & 0x7fffffffU;
    const uint exp_a = abs_a >> 23;
    const uint exp_b = abs_b >> 23;
    if (exp_a == 0xffU || exp_b == 0xffU) {
        return fairy2i_mul_f32_bits_rne(a, b);
    }
    if (abs_a == 0U || abs_b == 0U) {
        return ((a ^ b) >> 31) << 31;
    }

    if (exp_a > 0U && exp_b > 0U && exp_a + exp_b >= 128U) {
        return as_type<uint>(as_type<float>(a) * as_type<float>(b));
    }
    return fairy2i_mul_f32_bits_rne(a, b);
}

// Divide a finite F32 value by a positive integer with IEEE-754 RNE semantics.
// Metal fast-math may lower even a normal division to reciprocal-multiply (and
// flush subnormals), so RMSNorm cannot use native `/` for its canonical mean.
static inline uint fairy2i_div_f32_by_positive_int_bits_rne(uint value, uint divisor) {
    const uint sign      = value & 0x80000000U;
    const uint abs_value = value & 0x7fffffffU;
    const uint exponent  = abs_value >> 23;
    const uint fraction  = abs_value & 0x7fffffU;

    if (divisor == 0) {
        return abs_value == 0 ? 0x7fc00000U : sign | 0x7f800000U;
    }
    if (exponent == 0xffU) {
        return fraction == 0 ? value : value | 0x00400000U;
    }
    if (abs_value == 0) {
        return value;
    }

    const uint significand = exponent == 0 ? fraction : 0x800000U | fraction;
    const int  value_e     = exponent == 0 ? -149 : (int) exponent - 150;
    const int  sig_msb     = 31 - (int) clz(significand);
    const int  divisor_msb = 31 - (int) clz(divisor);

    int ratio_e = sig_msb - divisor_msb;
    if (ratio_e >= 0) {
        if ((ulong) significand < ((ulong) divisor << (uint) ratio_e)) {
            --ratio_e;
        }
    } else if (((ulong) significand << (uint) -ratio_e) < (ulong) divisor) {
        --ratio_e;
    }

    int result_e = value_e + ratio_e;
    if (result_e >= -126) {
        const uint  shift     = (uint) (23 - ratio_e);
        const ulong numerator = (ulong) significand << shift;
        ulong       quotient  = numerator / divisor;
        const ulong remainder = numerator % divisor;
        const ulong twice_remainder = remainder << 1;
        if (twice_remainder > divisor ||
            (twice_remainder == divisor && (quotient & 1UL) != 0)) {
            ++quotient;
        }
        if (quotient >= 0x1000000UL) {
            quotient >>= 1;
            ++result_e;
        }
        if (result_e > 127) {
            return sign | 0x7f800000U;
        }
        return sign | ((uint) (result_e + 127) << 23) | ((uint) quotient & 0x7fffffU);
    }

    const uint  shift     = exponent == 0 ? 0U : exponent - 1U;
    const ulong numerator = (ulong) significand << shift;
    ulong       quotient  = numerator / divisor;
    const ulong remainder = numerator % divisor;
    const ulong twice_remainder = remainder << 1;
    if (twice_remainder > divisor ||
        (twice_remainder == divisor && (quotient & 1UL) != 0)) {
        ++quotient;
    }
    return sign | (quotient >= 0x800000UL ? 0x00800000U : (uint) quotient);
}

// Exact reciprocal for a positive normal F32. RMSNorm applies this to the
// rounded result of precise::sqrt; that range always produces a normal finite
// reciprocal for finite BF16 inputs.
static inline uint fairy2i_reciprocal_positive_normal_f32_bits_rne(uint value) {
    const uint exponent    = (value & 0x7fffffffU) >> 23;
    const uint fraction    = value & 0x7fffffU;
    const uint significand = 0x800000U | fraction;

    if (exponent == 0xffU) {
        return fraction == 0 ? 0U : value | 0x00400000U;
    }
    if (exponent == 0U) {
        return 0x7f800000U;
    }

    const int ratio_e = significand > 0x800000U ? -1 : 0;
    int       result_e = 127 - (int) exponent + ratio_e;
    const ulong numerator = (ulong) 0x800000U << (uint) (23 - ratio_e);
    ulong       quotient  = numerator / significand;
    const ulong remainder = numerator % significand;
    const ulong twice_remainder = remainder << 1;
    if (twice_remainder > significand ||
        (twice_remainder == significand && (quotient & 1UL) != 0)) {
        ++quotient;
    }
    if (quotient >= 0x1000000UL) {
        quotient >>= 1;
        ++result_e;
    }
    return ((uint) (result_e + 127) << 23) | ((uint) quotient & 0x7fffffU);
}

static inline ushort fairy2i_add_bf16_bits_rne(ushort a, ushort b) {
    const uint sign_a = (uint) a >> 15;
    const uint sign_b = (uint) b >> 15;
    const uint abs_a = (uint) a & 0x7fffU;
    const uint abs_b = (uint) b & 0x7fffU;
    const uint exp_a_raw = abs_a >> 7;
    const uint exp_b_raw = abs_b >> 7;
    const uint frac_a = abs_a & 0x7fU;
    const uint frac_b = abs_b & 0x7fU;

    // Normal BF16 values with exponent >= 8 cannot cancel to an F32
    // subnormal, so the native F32 add is exact and substantially cheaper.
    if (exp_a_raw >= 8U && exp_a_raw < 0xffU &&
        exp_b_raw >= 8U && exp_b_raw < 0xffU) {
        return fairy2i_f32_to_bf16(fairy2i_bf16_to_f32(a) + fairy2i_bf16_to_f32(b));
    }

    if (exp_a_raw == 0xffU || exp_b_raw == 0xffU) {
        const bool nan_a = exp_a_raw == 0xffU && frac_a != 0;
        const bool nan_b = exp_b_raw == 0xffU && frac_b != 0;
        if (nan_a) {
            return (ushort) ((uint) a | 0x40U);
        }
        if (nan_b) {
            return (ushort) ((uint) b | 0x40U);
        }
        if (sign_a != sign_b && exp_a_raw == 0xffU && exp_b_raw == 0xffU) {
            return (ushort) 0x7fc0U;
        }
        return exp_a_raw == 0xffU ? a : b;
    }

    if (abs_a == 0 && abs_b == 0) {
        return (ushort) ((sign_a & sign_b) << 15);
    }
    if (abs_a == 0) {
        return b;
    }
    if (abs_b == 0) {
        return a;
    }

    uint sig_a = exp_a_raw == 0 ? frac_a : 0x80U | frac_a;
    uint sig_b = exp_b_raw == 0 ? frac_b : 0x80U | frac_b;
    int exp_a = exp_a_raw == 0 ? -133 : (int) exp_a_raw - 134;
    int exp_b = exp_b_raw == 0 ? -133 : (int) exp_b_raw - 134;
    uint work_sign_a = sign_a;
    uint work_sign_b = sign_b;

    while (sig_a < 0x80U) {
        sig_a <<= 1;
        --exp_a;
    }
    while (sig_b < 0x80U) {
        sig_b <<= 1;
        --exp_b;
    }

    if (exp_a < exp_b || (exp_a == exp_b && sig_a < sig_b)) {
        uint tmp_sig = sig_a;
        sig_a = sig_b;
        sig_b = tmp_sig;
        int tmp_exp = exp_a;
        exp_a = exp_b;
        exp_b = tmp_exp;
        const uint tmp_sign = work_sign_a;
        work_sign_a = work_sign_b;
        work_sign_b = tmp_sign;
    }

    const uint result_sign = work_sign_a;
    uint ext_a = sig_a << 3;
    uint ext_b = fairy2i_shift_right_jam(sig_b << 3, (uint) (exp_a - exp_b));
    uint ext;
    int result_exp = exp_a;

    if (work_sign_a == work_sign_b) {
        ext = ext_a + ext_b;
        if (ext >= (0x100U << 3)) {
            ext = fairy2i_shift_right_jam(ext, 1);
            ++result_exp;
        }
    } else {
        ext = ext_a - ext_b;
        if (ext == 0) {
            return (ushort) 0;
        }
        while (ext < (0x80U << 3)) {
            ext <<= 1;
            --result_exp;
        }
    }

    bool subnormal = false;
    if (result_exp < -133) {
        ext = fairy2i_shift_right_jam(ext, (uint) (-133 - result_exp));
        result_exp = -133;
        subnormal = true;
    }

    uint rounded_sig = ext >> 3;
    const uint remainder = ext & 7U;
    if (remainder > 4U || (remainder == 4U && (rounded_sig & 1U) != 0)) {
        ++rounded_sig;
    }

    if (rounded_sig >= 0x100U) {
        rounded_sig >>= 1;
        ++result_exp;
        subnormal = false;
    }

    if (subnormal) {
        return (ushort) ((result_sign << 15) | min(rounded_sig, 0x80U));
    }

    const int result_exp_raw = result_exp + 134;
    if (result_exp_raw >= 0xff) {
        return (ushort) ((result_sign << 15) | 0x7f80U);
    }
    return (ushort) ((result_sign << 15) | ((uint) result_exp_raw << 7) | (rounded_sig & 0x7fU));
}

static inline uint fairy2i_add_f32_bits_rne(uint a, uint b) {
    const uint sign_a    = a >> 31;
    const uint sign_b    = b >> 31;
    const uint abs_a     = a & 0x7fffffffU;
    const uint abs_b     = b & 0x7fffffffU;
    const uint exp_a_raw = abs_a >> 23;
    const uint exp_b_raw = abs_b >> 23;
    const uint frac_a    = abs_a & 0x7fffffU;
    const uint frac_b    = abs_b & 0x7fffffU;

    // With both exponents at least 24, cancellation cannot produce an F32 subnormal:
    // the smallest possible difference is one ULP at exponent 24, exactly 2^-126.
    if (exp_a_raw >= 24U && exp_a_raw < 0xffU &&
        exp_b_raw >= 24U && exp_b_raw < 0xffU) {
        return as_type<uint>(as_type<float>(a) + as_type<float>(b));
    }

    if (exp_a_raw == 0xffU || exp_b_raw == 0xffU) {
        const bool nan_a = exp_a_raw == 0xffU && frac_a != 0;
        const bool nan_b = exp_b_raw == 0xffU && frac_b != 0;
        if (nan_a) {
            return a | 0x00400000U;
        }
        if (nan_b) {
            return b | 0x00400000U;
        }
        if (sign_a != sign_b && exp_a_raw == 0xffU && exp_b_raw == 0xffU) {
            return 0x7fc00000U;
        }
        return exp_a_raw == 0xffU ? a : b;
    }

    if (abs_a == 0 && abs_b == 0) {
        return (sign_a & sign_b) << 31;
    }
    if (abs_a == 0) {
        return b;
    }
    if (abs_b == 0) {
        return a;
    }

    uint sig_a = exp_a_raw == 0 ? frac_a : 0x800000U | frac_a;
    uint sig_b = exp_b_raw == 0 ? frac_b : 0x800000U | frac_b;
    int  exp_a = exp_a_raw == 0 ? -149 : (int) exp_a_raw - 150;
    int  exp_b = exp_b_raw == 0 ? -149 : (int) exp_b_raw - 150;
    uint work_sign_a = sign_a;
    uint work_sign_b = sign_b;

    while (sig_a < 0x800000U) {
        sig_a <<= 1;
        --exp_a;
    }
    while (sig_b < 0x800000U) {
        sig_b <<= 1;
        --exp_b;
    }

    if (exp_a < exp_b || (exp_a == exp_b && sig_a < sig_b)) {
        uint tmp_sig = sig_a;
        sig_a = sig_b;
        sig_b = tmp_sig;
        int tmp_exp = exp_a;
        exp_a = exp_b;
        exp_b = tmp_exp;
        const uint tmp_sign = work_sign_a;
        work_sign_a = work_sign_b;
        work_sign_b = tmp_sign;
    }

    const uint result_sign = work_sign_a;
    uint ext_a = sig_a << 3;
    uint ext_b = fairy2i_shift_right_jam(sig_b << 3, (uint) (exp_a - exp_b));
    uint ext;
    int result_exp = exp_a;

    if (work_sign_a == work_sign_b) {
        ext = ext_a + ext_b;
        if (ext >= (0x1000000U << 3)) {
            ext = fairy2i_shift_right_jam(ext, 1);
            ++result_exp;
        }
    } else {
        ext = ext_a - ext_b;
        if (ext == 0) {
            return 0;
        }
        while (ext < (0x800000U << 3)) {
            ext <<= 1;
            --result_exp;
        }
    }

    bool subnormal = false;
    if (result_exp < -149) {
        ext = fairy2i_shift_right_jam(ext, (uint) (-149 - result_exp));
        result_exp = -149;
        subnormal = true;
    }

    uint rounded_sig = ext >> 3;
    const uint remainder = ext & 7U;
    if (remainder > 4U || (remainder == 4U && (rounded_sig & 1U) != 0)) {
        ++rounded_sig;
    }

    if (rounded_sig >= 0x1000000U) {
        rounded_sig >>= 1;
        ++result_exp;
        subnormal = false;
    }

    if (subnormal) {
        return (result_sign << 31) | min(rounded_sig, 0x800000U);
    }

    const int result_exp_raw = result_exp + 150;
    if (result_exp_raw >= 0xff) {
        return (result_sign << 31) | 0x7f800000U;
    }
    return (result_sign << 31) | ((uint) result_exp_raw << 23) | (rounded_sig & 0x7fffffU);
}

// Exact fused BF16*BF16+F32. Keeping the unrounded 16-bit BF16 product until after it is aligned
// with the accumulator avoids the double-rounding error at the bottom of the F32 subnormal range.
static inline uint fairy2i_fma_bf16_bf16_f32_bits_rne(ushort a, ushort b, uint acc) {
    const uint sign_a = (uint) a >> 15;
    const uint sign_b = (uint) b >> 15;
    const uint abs_a  = (uint) a & 0x7fffU;
    const uint abs_b  = (uint) b & 0x7fffU;
    const uint exp_a  = abs_a >> 7;
    const uint exp_b  = abs_b >> 7;
    const uint frac_a = abs_a & 0x7fU;
    const uint frac_b = abs_b & 0x7fU;
    const uint product_sign = sign_a ^ sign_b;

    const uint acc_sign = acc >> 31;
    const uint acc_abs  = acc & 0x7fffffffU;
    const uint acc_exp  = acc_abs >> 23;
    const uint acc_frac = acc_abs & 0x7fffffU;

    const bool nan_a   = exp_a == 0xffU && frac_a != 0;
    const bool nan_b   = exp_b == 0xffU && frac_b != 0;
    const bool nan_acc = acc_exp == 0xffU && acc_frac != 0;
    if (nan_a || nan_b || nan_acc) {
        return 0x7fc00000U;
    }

    const bool product_inf = exp_a == 0xffU || exp_b == 0xffU;
    const bool product_zero = abs_a == 0 || abs_b == 0;
    if (product_inf) {
        if (product_zero || (acc_exp == 0xffU && product_sign != acc_sign)) {
            return 0x7fc00000U;
        }
        return (product_sign << 31) | 0x7f800000U;
    }
    if (acc_exp == 0xffU) {
        return acc;
    }
    if (product_zero) {
        return fairy2i_add_f32_bits_rne(acc, product_sign << 31);
    }

    uint product_sig = (exp_a == 0 ? frac_a : 0x80U | frac_a) *
                       (exp_b == 0 ? frac_b : 0x80U | frac_b);
    int product_exp = (exp_a == 0 ? -133 : (int) exp_a - 134) +
                      (exp_b == 0 ? -133 : (int) exp_b - 134);
    while (product_sig < 0x800000U) {
        product_sig <<= 1;
        --product_exp;
    }

    if (acc_abs == 0) {
        uint ext = product_sig << 3;
        int result_exp = product_exp;
        if (result_exp < -149) {
            ext = fairy2i_shift_right_jam(ext, (uint) (-149 - result_exp));
            result_exp = -149;
        }
        uint rounded_sig = ext >> 3;
        const uint remainder = ext & 7U;
        if (remainder > 4U || (remainder == 4U && (rounded_sig & 1U) != 0)) {
            ++rounded_sig;
        }
        if (rounded_sig >= 0x1000000U) {
            rounded_sig >>= 1;
            ++result_exp;
        }
        if (result_exp == -149 && rounded_sig < 0x800000U) {
            return (product_sign << 31) | rounded_sig;
        }
        const int result_exp_raw = result_exp + 150;
        if (result_exp_raw >= 0xff) {
            return (product_sign << 31) | 0x7f800000U;
        }
        return (product_sign << 31) | ((uint) result_exp_raw << 23) | (rounded_sig & 0x7fffffU);
    }

    uint acc_sig = acc_exp == 0 ? acc_frac : 0x800000U | acc_frac;
    int  acc_lsb_exp = acc_exp == 0 ? -149 : (int) acc_exp - 150;
    while (acc_sig < 0x800000U) {
        acc_sig <<= 1;
        --acc_lsb_exp;
    }

    uint large_sig  = product_sig;
    int  large_exp  = product_exp;
    uint large_sign = product_sign;
    uint small_sig  = acc_sig;
    int  small_exp  = acc_lsb_exp;
    uint small_sign = acc_sign;
    if (large_exp < small_exp || (large_exp == small_exp && large_sig < small_sig)) {
        const uint tmp_sig = large_sig;
        large_sig = small_sig;
        small_sig = tmp_sig;
        const int tmp_exp = large_exp;
        large_exp = small_exp;
        small_exp = tmp_exp;
        const uint tmp_sign = large_sign;
        large_sign = small_sign;
        small_sign = tmp_sign;
    }

    uint ext = large_sig << 3;
    const uint small_ext = fairy2i_shift_right_jam(small_sig << 3, (uint) (large_exp - small_exp));
    int result_exp = large_exp;
    if (large_sign == small_sign) {
        ext += small_ext;
        if (ext >= (0x1000000U << 3)) {
            ext = fairy2i_shift_right_jam(ext, 1);
            ++result_exp;
        }
    } else {
        ext -= small_ext;
        if (ext == 0) {
            return 0;
        }
        while (ext < (0x800000U << 3)) {
            ext <<= 1;
            --result_exp;
        }
    }

    if (result_exp < -149) {
        ext = fairy2i_shift_right_jam(ext, (uint) (-149 - result_exp));
        result_exp = -149;
    }
    uint rounded_sig = ext >> 3;
    const uint remainder = ext & 7U;
    if (remainder > 4U || (remainder == 4U && (rounded_sig & 1U) != 0)) {
        ++rounded_sig;
    }
    if (rounded_sig >= 0x1000000U) {
        rounded_sig >>= 1;
        ++result_exp;
    }
    if (result_exp == -149 && rounded_sig < 0x800000U) {
        return (large_sign << 31) | rounded_sig;
    }
    const int result_exp_raw = result_exp + 150;
    if (result_exp_raw >= 0xff) {
        return (large_sign << 31) | 0x7f800000U;
    }
    return (large_sign << 31) | ((uint) result_exp_raw << 23) | (rounded_sig & 0x7fffffU);
}

static inline ushort fairy2i_add_f32_bias_to_bf16_bits_rne(float acc, float bias) {
    const uint sum_bits = fairy2i_add_f32_bits_rne(as_type<uint>(acc), as_type<uint>(bias));
    return fairy2i_f32_to_bf16(as_type<float>(sum_bits));
}

static inline ushort fairy2i_add_f32_bits_f32_bias_to_bf16_bits_rne(uint acc_bits, float bias) {
    const uint sum_bits = fairy2i_add_f32_bits_rne(acc_bits, as_type<uint>(bias));
    return fairy2i_f32_to_bf16(as_type<float>(sum_bits));
}

constant int FC_fairy2i_bundle_w1_prefill_act_rows [[function_constant(FC_FAIRY2I_BUNDLE_W1_PREFILL + 0)]];

static inline uint fairy2i_pack_bf16_pair(float real, float imag) {
    return ((uint) fairy2i_f32_to_bf16(real)) | (((uint) fairy2i_f32_to_bf16(imag)) << 16);
}

// Qwen3 ROW4/W8A8 numeric boundary:
//   F32 -> BF16 RNE -> per-token S8 (half-away) -> I32 dot
//       -> row scale -> BF16 RNE -> F32 carrier.
// The packed weights are deployment layouts and must never be routed through a
// generic MUL_MAT dequantizer.
kernel void kernel_row4_quantize_activation_i8(
        constant ggml_metal_kargs_row_quant_linear & args [[buffer(0)]],
        device const float * x                            [[buffer(1)]],
        device char * act_q                               [[buffer(2)]],
        device float * act_scales                         [[buffer(3)]],
        threadgroup float * simd_maxima                   [[threadgroup(0)]],
        uint3 tgpig                                       [[threadgroup_position_in_grid]],
        uint tid                                          [[thread_index_in_threadgroup]]) {
    const uint token      = tgpig.x;
    const uint simd_lane  = tid & 31U;
    const uint simd_group = tid >> 5;
    const ulong row_base  = (ulong) token * (ulong) args.k;

    float thread_max = 0.0f;
    for (uint i = tid; i < (uint) args.k; i += 256U) {
        const float xb = fairy2i_round_to_bf16_f32(x[row_base + i]);
        thread_max = max(thread_max, fabs(xb));
    }

    const float group_max = simd_max(thread_max);
    if (simd_lane == 0U) {
        simd_maxima[simd_group] = group_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0U) {
        float value = simd_lane < 8U ? simd_maxima[simd_lane] : 0.0f;
        value = simd_max(value);
        if (simd_lane == 0U) {
            const uint scale_bits = fairy2i_div_f32_by_positive_int_bits_rne(as_type<uint>(value), 127U);
            simd_maxima[0] = max(as_type<float>(scale_bits), 1.0e-8f);
            act_scales[token] = simd_maxima[0];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float sx = simd_maxima[0];
    for (uint i = tid; i < (uint) args.k; i += 256U) {
        const float xb        = fairy2i_round_to_bf16_f32(x[row_base + i]);
        const float magnitude = floor(fabs(precise::divide(xb, sx)) + 0.5f);
        int q = (int) magnitude;
        q = xb < 0.0f ? -q : q;
        act_q[row_base + i] = (char) clamp(q, -127, 127);
    }
}


kernel void kernel_row4_transpose_activation_i8_kmajor_half(
        constant ggml_metal_kargs_row_quant_linear & args [[buffer(0)]],
        device const char * act_q                         [[buffer(1)]],
        device half * act_h                               [[buffer(2)]],
        threadgroup half * tile                           [[threadgroup(0)]],
        uint3 tgpig                                       [[threadgroup_position_in_grid]],
        uint tid                                          [[thread_index_in_threadgroup]]) {
    constexpr uint tile_size   = 32U;
    constexpr uint tile_stride = 33U;
    constexpr uint n_threads   = 256U;
    const uint k_base          = tgpig.x * tile_size;
    const uint token_base      = tgpig.y * tile_size;

    for (uint index = tid; index < tile_size * tile_size; index += n_threads) {
        const uint token = index >> 5;
        const uint kk    = index & 31U;
        tile[token * tile_stride + kk] =
            (half) ((int) act_q[(ulong) (token_base + token) * (ulong) args.k + (ulong) (k_base + kk)]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = tid; index < tile_size * tile_size; index += n_threads) {
        const uint kk    = index >> 5;
        const uint token = index & 31U;
        act_h[(ulong) (k_base + kk) * (ulong) args.reserved + (ulong) (token_base + token)] =
            tile[token * tile_stride + kk];
    }
}

static inline int row4_weight_at(
        device const uchar * codes,
        int k_tiles,
        int output,
        int k) {
    const int output_tile = output >> 4;
    const int output_lane = output & 15;
    const int group       = output_lane >> 2;
    const int row         = output_lane & 3;
    const int k_tile      = k >> 7;
    const int k_in_tile   = k & 127;
    const int split       = k_in_tile >> 4;
    const int j           = k_in_tile & 7;
    const ulong offset =
        ((((ulong) output_tile * (ulong) k_tiles + (ulong) k_tile) * 4UL + (ulong) group) * 64UL) +
        (ulong) split * 8UL + (ulong) j;
    const uchar packed = codes[offset];
    const uint code = (k_in_tile & 8) == 0 ? (uint) (packed & 0x0fU) : (uint) (packed >> 4);

    const uint u_axis = code & 3U;
    const uint v_axis = code >> 2;
    const int ur = (int) (u_axis == 0U) - (int) (u_axis == 1U);
    const int ui = (int) (u_axis == 2U) - (int) (u_axis == 3U);
    const int vr = (int) (v_axis == 0U) - (int) (v_axis == 1U);
    const int vi = (int) (v_axis == 2U) - (int) (v_axis == 3U);

    switch (row) {
        case 0: return ur + vr;
        case 1: return -ui + vi;
        case 2: return ui + vi;
        default: return ur - vr;
    }
}

constexpr constant static half4 k_row4_prefill_codebook[16] = {
    half4( 2.0h,  0.0h,  0.0h,  0.0h),
    half4( 0.0h,  0.0h,  0.0h, -2.0h),
    half4( 1.0h, -1.0h,  1.0h, -1.0h),
    half4( 1.0h,  1.0h, -1.0h, -1.0h),
    half4( 0.0h,  0.0h,  0.0h,  2.0h),
    half4(-2.0h,  0.0h,  0.0h,  0.0h),
    half4(-1.0h, -1.0h,  1.0h,  1.0h),
    half4(-1.0h,  1.0h, -1.0h,  1.0h),
    half4( 1.0h,  1.0h,  1.0h,  1.0h),
    half4(-1.0h,  1.0h,  1.0h, -1.0h),
    half4( 0.0h,  0.0h,  2.0h,  0.0h),
    half4( 0.0h,  2.0h,  0.0h,  0.0h),
    half4( 1.0h, -1.0h, -1.0h,  1.0h),
    half4(-1.0h, -1.0h, -1.0h, -1.0h),
    half4( 0.0h, -2.0h,  0.0h,  0.0h),
    half4( 0.0h,  0.0h, -2.0h,  0.0h),
};

static inline half4 row4_decode4(uint code) {
    return k_row4_prefill_codebook[code];
}

static inline int w8a8_weight_at(
        device const char * codes,
        int k_tiles,
        int output,
        int k) {
    const int output_tile = output >> 4;
    const int output_lane = output & 15;
    const int k_tile      = k >> 7;
    const int k_in_tile   = k & 127;
    const ulong offset =
        ((((ulong) output_tile * (ulong) k_tiles + (ulong) k_tile) * 16UL + (ulong) output_lane) * 128UL) +
        (ulong) k_in_tile;
    return (int) codes[offset];
}

static inline float row4_finish_i32(int acc, float sx, ushort row_scale) {
    const uint activation_scaled = fairy2i_mul_f32_bits_rne_ftz_safe(as_type<uint>((float) acc), as_type<uint>(sx));
    const uint value = fairy2i_mul_f32_bits_rne_ftz_safe(activation_scaled, (uint) row_scale << 16);
    return fairy2i_bf16_to_f32(fairy2i_f32_bits_to_bf16_rne(value));
}

static inline float w8a8_finish_i32(int acc, float sx, float row_scale) {
    const uint activation_scaled = fairy2i_mul_f32_bits_rne_ftz_safe(as_type<uint>((float) acc), as_type<uint>(sx));
    const uint value = fairy2i_mul_f32_bits_rne_ftz_safe(activation_scaled, as_type<uint>(row_scale));
    return fairy2i_bf16_to_f32(fairy2i_f32_bits_to_bf16_rne(value));
}

static inline void row4_accumulate_basis_branchless(
        uint code,
        int activation,
        thread int & u_real,
        thread int & u_imag,
        thread int & v_real,
        thread int & v_imag) {
    const uint u_axis = code & 3U;
    const uint v_axis = code >> 2;
    const uint activation_bits = as_type<uint>(activation);
    const uint u_sign          = u_axis & 1U;
    const uint v_sign          = v_axis & 1U;
    const uint u_signed_bits   = (activation_bits ^ (0U - u_sign)) + u_sign;
    const uint v_signed_bits   = (activation_bits ^ (0U - v_sign)) + v_sign;
    const uint u_imag_mask     = 0U - (u_axis >> 1);
    const uint v_imag_mask     = 0U - (v_axis >> 1);

    u_real += as_type<int>(u_signed_bits & ~u_imag_mask);
    u_imag += as_type<int>(u_signed_bits & u_imag_mask);
    v_real += as_type<int>(v_signed_bits & ~v_imag_mask);
    v_imag += as_type<int>(v_signed_bits & v_imag_mask);
}

static inline void row4_accumulate_packed4_branchless(
        uchar4 packed,
        char4 activation_low,
        char4 activation_high,
        thread int & u_real,
        thread int & u_imag,
        thread int & v_real,
        thread int & v_imag) {
    row4_accumulate_basis_branchless(
        (uint) (packed.x & 0x0fU), (int) activation_low.x, u_real, u_imag, v_real, v_imag);
    row4_accumulate_basis_branchless(
        (uint) (packed.y & 0x0fU), (int) activation_low.y, u_real, u_imag, v_real, v_imag);
    row4_accumulate_basis_branchless(
        (uint) (packed.z & 0x0fU), (int) activation_low.z, u_real, u_imag, v_real, v_imag);
    row4_accumulate_basis_branchless(
        (uint) (packed.w & 0x0fU), (int) activation_low.w, u_real, u_imag, v_real, v_imag);
    row4_accumulate_basis_branchless(
        (uint) (packed.x >> 4), (int) activation_high.x, u_real, u_imag, v_real, v_imag);
    row4_accumulate_basis_branchless(
        (uint) (packed.y >> 4), (int) activation_high.y, u_real, u_imag, v_real, v_imag);
    row4_accumulate_basis_branchless(
        (uint) (packed.z >> 4), (int) activation_high.z, u_real, u_imag, v_real, v_imag);
    row4_accumulate_basis_branchless(
        (uint) (packed.w >> 4), (int) activation_high.w, u_real, u_imag, v_real, v_imag);
}


static inline int row4_segmented_sum16(int value, uint segment_lane) {
    int shuffled = simd_shuffle_down(value, 8U);
    value += segment_lane < 8U ? shuffled : 0;
    shuffled = simd_shuffle_down(value, 4U);
    value += segment_lane < 4U ? shuffled : 0;
    shuffled = simd_shuffle_down(value, 2U);
    value += segment_lane < 2U ? shuffled : 0;
    shuffled = simd_shuffle_down(value, 1U);
    value += segment_lane < 1U ? shuffled : 0;
    return value;
}

static inline void row4_linear_reduced(
        constant ggml_metal_kargs_row_quant_linear & args,
        device const uchar * codes,
        device const ushort * scales,
        device const char * act_q,
        device const float * act_scales,
        device float * dst,
        threadgroup int * partials,
        uint output_tile,
        uint token,
        uint tid) {
    constexpr uint lanes_per_row = 8U;
    const uint output_lane = tid / lanes_per_row;
    const uint dot_lane    = tid & (lanes_per_row - 1U);
    const int  output      = (int) output_tile * 16 + (int) output_lane;
    const int  k_tiles     = args.k / 128;
    const ulong act_base   = (ulong) token * (ulong) args.k;

    int acc = 0;
    for (int k = (int) dot_lane; k < args.k; k += (int) lanes_per_row) {
        acc += row4_weight_at(codes, k_tiles, output, k) * (int) act_q[act_base + (ulong) k];
    }
    partials[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (dot_lane == 0U) {
        int sum = partials[tid];
        for (uint lane = 1U; lane < lanes_per_row; ++lane) {
            sum += partials[tid + lane];
        }
        dst[(ulong) token * (ulong) args.m + (ulong) output] =
            row4_finish_i32(sum, act_scales[token], scales[output]);
    }
}

static inline void w8a8_linear_reduced(
        constant ggml_metal_kargs_row_quant_linear & args,
        device const char * codes,
        device const float * scales,
        device const char * act_q,
        device const float * act_scales,
        device float * dst,
        threadgroup int * partials,
        uint output_tile,
        uint token,
        uint tid) {
    constexpr uint lanes_per_row = 8U;
    const uint output_lane = tid / lanes_per_row;
    const uint dot_lane    = tid & (lanes_per_row - 1U);
    const int  output      = (int) output_tile * 16 + (int) output_lane;
    const int  k_tiles     = args.k / 128;
    const ulong act_base   = (ulong) token * (ulong) args.k;

    int acc = 0;
    for (int k = (int) dot_lane; k < args.k; k += (int) lanes_per_row) {
        acc += w8a8_weight_at(codes, k_tiles, output, k) * (int) act_q[act_base + (ulong) k];
    }
    partials[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (dot_lane == 0U) {
        int sum = partials[tid];
        for (uint lane = 1U; lane < lanes_per_row; ++lane) {
            sum += partials[tid + lane];
        }
        dst[(ulong) token * (ulong) args.m + (ulong) output] =
            w8a8_finish_i32(sum, act_scales[token], scales[output]);
    }
}

#define ROW4_REDUCED_KERNEL(NAME, TOKEN_EXPR)                                                        \
kernel void NAME(                                                                                   \
        constant ggml_metal_kargs_row_quant_linear & args [[buffer(0)]],                            \
        device const uchar * codes                        [[buffer(1)]],                            \
        device const ushort * scales                      [[buffer(2)]],                            \
        device const char * act_q                         [[buffer(3)]],                            \
        device const float * act_scales                   [[buffer(4)]],                            \
        device float * dst                                [[buffer(5)]],                            \
        threadgroup int * partials                        [[threadgroup(0)]],                       \
        uint3 tgpig                                       [[threadgroup_position_in_grid]],         \
        uint tid                                          [[thread_index_in_threadgroup]]) {        \
    row4_linear_reduced(args, codes, scales, act_q, act_scales, dst, partials, tgpig.x, TOKEN_EXPR, tid); \
}

#define W8A8_REDUCED_KERNEL(NAME, TOKEN_EXPR)                                                        \
kernel void NAME(                                                                                   \
        constant ggml_metal_kargs_row_quant_linear & args [[buffer(0)]],                            \
        device const char * codes                         [[buffer(1)]],                            \
        device const float * scales                       [[buffer(2)]],                            \
        device const char * act_q                         [[buffer(3)]],                            \
        device const float * act_scales                   [[buffer(4)]],                            \
        device float * dst                                [[buffer(5)]],                            \
        threadgroup int * partials                        [[threadgroup(0)]],                       \
        uint3 tgpig                                       [[threadgroup_position_in_grid]],         \
        uint tid                                          [[thread_index_in_threadgroup]]) {        \
    w8a8_linear_reduced(args, codes, scales, act_q, act_scales, dst, partials, tgpig.x, TOKEN_EXPR, tid); \
}



kernel void kernel_row4_w1a8_decode_o32_segmented16(
        constant ggml_metal_kargs_row_quant_linear & args [[buffer(0)]],
        device const uchar * codes                        [[buffer(1)]],
        device const ushort * scales                      [[buffer(2)]],
        device const char * act_q                         [[buffer(3)]],
        device const float * act_scales                   [[buffer(4)]],
        device float * dst                                [[buffer(5)]],
        uint3 tgpig                                       [[threadgroup_position_in_grid]],
        uint simd_lane                                    [[thread_index_in_simdgroup]],
        uint output_group                                 [[simdgroup_index_in_threadgroup]]) {
    constexpr int output_tiles_per_tg = 2;
    constexpr int groups_per_tile     = 4;
    constexpr int rows_per_group      = 4;
    constexpr int k_tile              = 128;
    constexpr int code_tile_bytes     = 256;

    const uint lane_half   = simd_lane >> 4;
    const uint local_lane  = simd_lane & 15U;
    const uint split       = local_lane >> 1;
    const uint jblock      = (local_lane & 1U) * 4U;
    const int output_tile  = (int) tgpig.x * output_tiles_per_tg + (int) (output_group >> 1);
    const int group_pair   = (int) (output_group & 1U) * 2;
    const int output_group_in_tile = group_pair + (int) lane_half;
    const int k_tiles      = args.k / k_tile;
    device const uchar * code_ptr = codes +
        (ulong) output_tile * (ulong) k_tiles * (ulong) code_tile_bytes +
        (ulong) output_group_in_tile * 64UL + (ulong) split * 8UL + (ulong) jblock;
    device const char * act_ptr = act_q + (ulong) split * 16UL + (ulong) jblock;

    int u_real = 0;
    int u_imag = 0;
    int v_real = 0;
    int v_imag = 0;
    for (int kt = 0; kt < k_tiles; ++kt) {
        const uint packed_bits = *((device const uint *) code_ptr);
        uint activation_low_bits = 0U;
        uint activation_high_bits = 0U;
        if (lane_half == 0U) {
            activation_low_bits = as_type<uint>(*((device const char4 *) act_ptr));
            activation_high_bits = as_type<uint>(*((device const char4 *) (act_ptr + 8)));
        }
        activation_low_bits = simd_shuffle(activation_low_bits, local_lane);
        activation_high_bits = simd_shuffle(activation_high_bits, local_lane);

        row4_accumulate_packed4_branchless(
            as_type<uchar4>(packed_bits),
            as_type<char4>(activation_low_bits),
            as_type<char4>(activation_high_bits),
            u_real,
            u_imag,
            v_real,
            v_imag);
        code_ptr += code_tile_bytes;
        act_ptr += k_tile;
    }

    u_real = row4_segmented_sum16(u_real, local_lane);
    u_imag = row4_segmented_sum16(u_imag, local_lane);
    v_real = row4_segmented_sum16(v_real, local_lane);
    v_imag = row4_segmented_sum16(v_imag, local_lane);
    if (local_lane == 0U) {
        const int output = output_tile * groups_per_tile * rows_per_group + output_group_in_tile * rows_per_group;
        const float sx = act_scales[0];
        dst[output + 0] = row4_finish_i32(u_real + v_real, sx, scales[output + 0]);
        dst[output + 1] = row4_finish_i32(-u_imag + v_imag, sx, scales[output + 1]);
        dst[output + 2] = row4_finish_i32(u_imag + v_imag, sx, scales[output + 2]);
        dst[output + 3] = row4_finish_i32(u_real - v_real, sx, scales[output + 3]);
    }
}


kernel void kernel_row4_w1a8_decode_o32_o4_staged_act(
        constant ggml_metal_kargs_row_quant_linear & args [[buffer(0)]],
        device const uchar * codes                        [[buffer(1)]],
        device const ushort * scales                      [[buffer(2)]],
        device const char * act_q                         [[buffer(3)]],
        device const float * act_scales                   [[buffer(4)]],
        device float * dst                                [[buffer(5)]],
        threadgroup char * act_tg                         [[threadgroup(0)]],
        uint3 tgpig                                       [[threadgroup_position_in_grid]],
        uint tid                                          [[thread_index_in_threadgroup]],
        uint simd_lane                                    [[thread_index_in_simdgroup]],
        uint output_group                                 [[simdgroup_index_in_threadgroup]]) {
    constexpr int output_tiles_per_tg = 2;
    constexpr int groups_per_tile     = 4;
    constexpr int rows_per_group      = 4;
    constexpr int k_tile              = 128;
    constexpr int code_tile_bytes     = 256;
    constexpr int copy_bytes          = 16;
    constexpr int threads_per_tg      = 256;

    for (int byte_offset = (int) tid * copy_bytes; byte_offset < args.k;
         byte_offset += threads_per_tg * copy_bytes) {
        *((threadgroup uint4 *) (act_tg + byte_offset)) = *((device const uint4 *) (act_q + byte_offset));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint split = simd_lane >> 2;
    const uint j     = (simd_lane & 3U) * 2U;
    const int output_tile = (int) tgpig.x * output_tiles_per_tg + (int) (output_group >> 2);
    const int output_group_in_tile = (int) (output_group & 3U);
    const int k_tiles = args.k / k_tile;
    device const uchar * code_ptr = codes +
        (ulong) output_tile * (ulong) k_tiles * (ulong) code_tile_bytes +
        (ulong) output_group_in_tile * 64UL + (ulong) simd_lane * 2UL;
    threadgroup const char * act_ptr = act_tg + (ulong) split * 16UL + (ulong) j;

    int u_real = 0;
    int u_imag = 0;
    int v_real = 0;
    int v_imag = 0;
    for (int kt = 0; kt < k_tiles; ++kt) {
        const uchar2 packed = *((device const uchar2 *) code_ptr);
        const char2 activation_low = *((threadgroup const char2 *) act_ptr);
        const char2 activation_high = *((threadgroup const char2 *) (act_ptr + 8));
        row4_accumulate_basis_branchless(
            (uint) (packed.x & 0x0fU), (int) activation_low.x, u_real, u_imag, v_real, v_imag);
        row4_accumulate_basis_branchless(
            (uint) (packed.y & 0x0fU), (int) activation_low.y, u_real, u_imag, v_real, v_imag);
        row4_accumulate_basis_branchless(
            (uint) (packed.x >> 4), (int) activation_high.x, u_real, u_imag, v_real, v_imag);
        row4_accumulate_basis_branchless(
            (uint) (packed.y >> 4), (int) activation_high.y, u_real, u_imag, v_real, v_imag);
        code_ptr += code_tile_bytes;
        act_ptr += k_tile;
    }

    const int sum_u_real = simd_sum(u_real);
    const int sum_u_imag = simd_sum(u_imag);
    const int sum_v_real = simd_sum(v_real);
    const int sum_v_imag = simd_sum(v_imag);
    if (simd_lane == 0U) {
        const int output = output_tile * groups_per_tile * rows_per_group + output_group_in_tile * rows_per_group;
        const float sx = act_scales[0];
        dst[output + 0] = row4_finish_i32(sum_u_real + sum_v_real, sx, scales[output + 0]);
        dst[output + 1] = row4_finish_i32(-sum_u_imag + sum_v_imag, sx, scales[output + 1]);
        dst[output + 2] = row4_finish_i32(sum_u_imag + sum_v_imag, sx, scales[output + 2]);
        dst[output + 3] = row4_finish_i32(sum_u_real - sum_v_real, sx, scales[output + 3]);
    }
}


ROW4_REDUCED_KERNEL(kernel_row4_w1a8_small_batch, tgpig.y)


static inline void w8a8_accumulate_char4(
        char4 weights,
        char4 activation,
        thread int & acc) {
    acc += (int) weights.x * (int) activation.x;
    acc += (int) weights.y * (int) activation.y;
    acc += (int) weights.z * (int) activation.z;
    acc += (int) weights.w * (int) activation.w;
}


kernel void kernel_row8_w8a8_decode_o64_rows8(
        constant ggml_metal_kargs_row_quant_linear & args [[buffer(0)]],
        device const char * codes                         [[buffer(1)]],
        device const float * scales                       [[buffer(2)]],
        device const char * act_q                         [[buffer(3)]],
        device const float * act_scales                   [[buffer(4)]],
        device float * dst                                [[buffer(5)]],
        uint3 tgpig                                       [[threadgroup_position_in_grid]],
        uint simd_lane                                    [[thread_index_in_simdgroup]],
        uint output_group                                 [[simdgroup_index_in_threadgroup]]) {
    constexpr int output_tiles_per_tg = 4;
    constexpr int rows_per_tile       = 16;
    constexpr int rows_per_group      = 8;
    constexpr int k_tile              = 128;
    constexpr int code_tile_bytes     = rows_per_tile * k_tile;

    const int output_tile = (int) tgpig.x * output_tiles_per_tg + (int) (output_group >> 1);
    const int row0        = (int) (output_group & 1U) * rows_per_group;
    const int k_tiles     = args.k / k_tile;
    device const char * weight_ptr = codes +
        (ulong) output_tile * (ulong) k_tiles * (ulong) code_tile_bytes +
        (ulong) row0 * (ulong) k_tile + (ulong) simd_lane * 4UL;
    device const char * act_ptr = act_q + simd_lane * 4U;

    int acc0 = 0;
    int acc1 = 0;
    int acc2 = 0;
    int acc3 = 0;
    int acc4 = 0;
    int acc5 = 0;
    int acc6 = 0;
    int acc7 = 0;
    for (int kt = 0; kt < k_tiles; ++kt) {
        const char4 activation = *((device const char4 *) act_ptr);
        const char4 weights0 = *((device const char4 *) (weight_ptr + 0 * k_tile));
        w8a8_accumulate_char4(weights0, activation, acc0);
        const char4 weights1 = *((device const char4 *) (weight_ptr + 1 * k_tile));
        w8a8_accumulate_char4(weights1, activation, acc1);
        const char4 weights2 = *((device const char4 *) (weight_ptr + 2 * k_tile));
        w8a8_accumulate_char4(weights2, activation, acc2);
        const char4 weights3 = *((device const char4 *) (weight_ptr + 3 * k_tile));
        w8a8_accumulate_char4(weights3, activation, acc3);
        const char4 weights4 = *((device const char4 *) (weight_ptr + 4 * k_tile));
        w8a8_accumulate_char4(weights4, activation, acc4);
        const char4 weights5 = *((device const char4 *) (weight_ptr + 5 * k_tile));
        w8a8_accumulate_char4(weights5, activation, acc5);
        const char4 weights6 = *((device const char4 *) (weight_ptr + 6 * k_tile));
        w8a8_accumulate_char4(weights6, activation, acc6);
        const char4 weights7 = *((device const char4 *) (weight_ptr + 7 * k_tile));
        w8a8_accumulate_char4(weights7, activation, acc7);
        weight_ptr += code_tile_bytes;
        act_ptr += k_tile;
    }

    const int sum0 = simd_sum(acc0);
    const int sum1 = simd_sum(acc1);
    const int sum2 = simd_sum(acc2);
    const int sum3 = simd_sum(acc3);
    const int sum4 = simd_sum(acc4);
    const int sum5 = simd_sum(acc5);
    const int sum6 = simd_sum(acc6);
    const int sum7 = simd_sum(acc7);
    if (simd_lane == 0U) {
        const int output = output_tile * rows_per_tile + row0;
        const float sx = act_scales[0];
        dst[output + 0] = w8a8_finish_i32(sum0, sx, scales[output + 0]);
        dst[output + 1] = w8a8_finish_i32(sum1, sx, scales[output + 1]);
        dst[output + 2] = w8a8_finish_i32(sum2, sx, scales[output + 2]);
        dst[output + 3] = w8a8_finish_i32(sum3, sx, scales[output + 3]);
        dst[output + 4] = w8a8_finish_i32(sum4, sx, scales[output + 4]);
        dst[output + 5] = w8a8_finish_i32(sum5, sx, scales[output + 5]);
        dst[output + 6] = w8a8_finish_i32(sum6, sx, scales[output + 6]);
        dst[output + 7] = w8a8_finish_i32(sum7, sx, scales[output + 7]);
    }
}

kernel void kernel_row8_w8a8_decode_o128_rows16(
        constant ggml_metal_kargs_row_quant_linear & args [[buffer(0)]],
        device const char * codes                         [[buffer(1)]],
        device const float * scales                       [[buffer(2)]],
        device const char * act_q                         [[buffer(3)]],
        device const float * act_scales                   [[buffer(4)]],
        device float * dst                                [[buffer(5)]],
        uint3 tgpig                                       [[threadgroup_position_in_grid]],
        uint simd_lane                                    [[thread_index_in_simdgroup]],
        uint output_group                                 [[simdgroup_index_in_threadgroup]]) {
    constexpr int output_tiles_per_tg = 8;
    constexpr int rows_per_tile       = 16;
    constexpr int k_tile              = 128;
    constexpr int code_tile_bytes     = rows_per_tile * k_tile;

    const int output_tile = (int) tgpig.x * output_tiles_per_tg + (int) output_group;
    const int k_tiles     = args.k / k_tile;
    device const char * weight_ptr = codes +
        (ulong) output_tile * (ulong) k_tiles * (ulong) code_tile_bytes + (ulong) simd_lane * 4UL;
    device const char * act_ptr = act_q + simd_lane * 4U;

    int acc0  = 0;
    int acc1  = 0;
    int acc2  = 0;
    int acc3  = 0;
    int acc4  = 0;
    int acc5  = 0;
    int acc6  = 0;
    int acc7  = 0;
    int acc8  = 0;
    int acc9  = 0;
    int acc10 = 0;
    int acc11 = 0;
    int acc12 = 0;
    int acc13 = 0;
    int acc14 = 0;
    int acc15 = 0;
    for (int kt = 0; kt < k_tiles; ++kt) {
        const char4 activation = *((device const char4 *) act_ptr);
        w8a8_accumulate_char4(*((device const char4 *) (weight_ptr +  0 * k_tile)), activation, acc0);
        w8a8_accumulate_char4(*((device const char4 *) (weight_ptr +  1 * k_tile)), activation, acc1);
        w8a8_accumulate_char4(*((device const char4 *) (weight_ptr +  2 * k_tile)), activation, acc2);
        w8a8_accumulate_char4(*((device const char4 *) (weight_ptr +  3 * k_tile)), activation, acc3);
        w8a8_accumulate_char4(*((device const char4 *) (weight_ptr +  4 * k_tile)), activation, acc4);
        w8a8_accumulate_char4(*((device const char4 *) (weight_ptr +  5 * k_tile)), activation, acc5);
        w8a8_accumulate_char4(*((device const char4 *) (weight_ptr +  6 * k_tile)), activation, acc6);
        w8a8_accumulate_char4(*((device const char4 *) (weight_ptr +  7 * k_tile)), activation, acc7);
        w8a8_accumulate_char4(*((device const char4 *) (weight_ptr +  8 * k_tile)), activation, acc8);
        w8a8_accumulate_char4(*((device const char4 *) (weight_ptr +  9 * k_tile)), activation, acc9);
        w8a8_accumulate_char4(*((device const char4 *) (weight_ptr + 10 * k_tile)), activation, acc10);
        w8a8_accumulate_char4(*((device const char4 *) (weight_ptr + 11 * k_tile)), activation, acc11);
        w8a8_accumulate_char4(*((device const char4 *) (weight_ptr + 12 * k_tile)), activation, acc12);
        w8a8_accumulate_char4(*((device const char4 *) (weight_ptr + 13 * k_tile)), activation, acc13);
        w8a8_accumulate_char4(*((device const char4 *) (weight_ptr + 14 * k_tile)), activation, acc14);
        w8a8_accumulate_char4(*((device const char4 *) (weight_ptr + 15 * k_tile)), activation, acc15);
        weight_ptr += code_tile_bytes;
        act_ptr += k_tile;
    }

    acc0  = simd_sum(acc0);
    acc1  = simd_sum(acc1);
    acc2  = simd_sum(acc2);
    acc3  = simd_sum(acc3);
    acc4  = simd_sum(acc4);
    acc5  = simd_sum(acc5);
    acc6  = simd_sum(acc6);
    acc7  = simd_sum(acc7);
    acc8  = simd_sum(acc8);
    acc9  = simd_sum(acc9);
    acc10 = simd_sum(acc10);
    acc11 = simd_sum(acc11);
    acc12 = simd_sum(acc12);
    acc13 = simd_sum(acc13);
    acc14 = simd_sum(acc14);
    acc15 = simd_sum(acc15);
    if (simd_lane == 0U) {
        const int output = output_tile * rows_per_tile;
        const float sx = act_scales[0];
        dst[output +  0] = w8a8_finish_i32(acc0,  sx, scales[output +  0]);
        dst[output +  1] = w8a8_finish_i32(acc1,  sx, scales[output +  1]);
        dst[output +  2] = w8a8_finish_i32(acc2,  sx, scales[output +  2]);
        dst[output +  3] = w8a8_finish_i32(acc3,  sx, scales[output +  3]);
        dst[output +  4] = w8a8_finish_i32(acc4,  sx, scales[output +  4]);
        dst[output +  5] = w8a8_finish_i32(acc5,  sx, scales[output +  5]);
        dst[output +  6] = w8a8_finish_i32(acc6,  sx, scales[output +  6]);
        dst[output +  7] = w8a8_finish_i32(acc7,  sx, scales[output +  7]);
        dst[output +  8] = w8a8_finish_i32(acc8,  sx, scales[output +  8]);
        dst[output +  9] = w8a8_finish_i32(acc9,  sx, scales[output +  9]);
        dst[output + 10] = w8a8_finish_i32(acc10, sx, scales[output + 10]);
        dst[output + 11] = w8a8_finish_i32(acc11, sx, scales[output + 11]);
        dst[output + 12] = w8a8_finish_i32(acc12, sx, scales[output + 12]);
        dst[output + 13] = w8a8_finish_i32(acc13, sx, scales[output + 13]);
        dst[output + 14] = w8a8_finish_i32(acc14, sx, scales[output + 14]);
        dst[output + 15] = w8a8_finish_i32(acc15, sx, scales[output + 15]);
    }
}


W8A8_REDUCED_KERNEL(kernel_row8_w8a8_small_batch, tgpig.y)

template<int col_tile>
static inline void row4_w1a8_prefill_impl(
        constant ggml_metal_kargs_row_quant_linear & args,
        device const uchar * codes,
        device const ushort * scales,
        device const char * act_q,
        device const float * act_scales,
        device float * dst,
        threadgroup half * weight_tile,
        threadgroup half * act_tile,
        threadgroup float * out_tile,
        uint3 tgpig,
        uint tid,
        uint sgitg) {
    constexpr int row_tile   = 16;
    constexpr int k_tile     = 32;
    constexpr int n_threads  = col_tile * 8;
    const int row_base       = (int) tgpig.x * row_tile;
    const int col_base       = (int) tgpig.y * col_tile;
    const int k_tiles        = args.k / 128;
    const int row_block      = (int) (sgitg & 1U);
    const int col_block      = (int) (sgitg >> 1);

    simdgroup_half8x8  weights;
    simdgroup_half8x8  activations;
    simdgroup_float8x8 accum = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (int k_base = 0; k_base < args.k; k_base += k_tile) {
        const int canonical_k_tile = k_base >> 7;
        const int split_base       = (k_base & 127) >> 4;
        for (uint packed_index = tid; packed_index < 64U; packed_index += n_threads) {
            const uint group       = packed_index >> 4;
            const uint byte        = packed_index & 15U;
            const uint local_split = byte >> 3;
            const uint j           = byte & 7U;
            const ulong code_offset =
                ((((ulong) tgpig.x * (ulong) k_tiles + (ulong) canonical_k_tile) * 4UL + (ulong) group) * 64UL) +
                (ulong) (split_base + (int) local_split) * 8UL + (ulong) j;
            const uchar packed = codes[code_offset];
            const uint k_low   = local_split * 16U + j;
            const uint k_high  = k_low + 8U;

            *((threadgroup half4 *) (weight_tile + k_low * row_tile + group * 4U)) =
                row4_decode4((uint) (packed & 0x0fU));
            *((threadgroup half4 *) (weight_tile + k_high * row_tile + group * 4U)) =
                row4_decode4((uint) (packed >> 4));
        }

        for (uint index = tid; index < (uint) (col_tile * k_tile); index += n_threads) {
            const uint col   = index / (uint) k_tile;
            const uint kk    = index % (uint) k_tile;
            const uint token = (uint) col_base + col;
            act_tile[index] = token < (uint) args.act_rows ?
                (half) ((int) act_q[(ulong) token * (ulong) args.k + (ulong) k_base + (ulong) kk]) : (half) 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int ik = 0; ik < k_tile; ik += 8) {
            simdgroup_load(weights, weight_tile + ik * row_tile + row_block * 8, row_tile, 0, true);
            simdgroup_load(activations, act_tile + col_block * 8 * k_tile + ik, k_tile, 0, true);
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(accum, weights, activations, accum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(accum, out_tile + col_block * 8 * row_tile + row_block * 8, row_tile, 0, true);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = tid; index < (uint) (col_tile * row_tile); index += n_threads) {
        const uint col       = index / (uint) row_tile;
        const uint row       = index % (uint) row_tile;
        const uint output    = (uint) row_base + row;
        const uint token     = (uint) col_base + col;
        if (token < (uint) args.act_rows) {
            const int exact_acc = (int) out_tile[index];
            dst[(ulong) token * (ulong) args.m + (ulong) output] =
                row4_finish_i32(exact_acc, act_scales[token], scales[output]);
        }
    }
}

kernel void kernel_row4_w1a8_prefill(
        constant ggml_metal_kargs_row_quant_linear & args [[buffer(0)]],
        device const uchar * codes                        [[buffer(1)]],
        device const ushort * scales                      [[buffer(2)]],
        device const char * act_q                         [[buffer(3)]],
        device const float * act_scales                   [[buffer(4)]],
        device float * dst                                [[buffer(5)]],
        threadgroup half * weight_tile                    [[threadgroup(0)]],
        threadgroup half * act_tile                       [[threadgroup(1)]],
        threadgroup float * out_tile                      [[threadgroup(2)]],
        uint3 tgpig                                       [[threadgroup_position_in_grid]],
        uint tid                                          [[thread_index_in_threadgroup]],
        uint sgitg                                        [[simdgroup_index_in_threadgroup]]) {
    row4_w1a8_prefill_impl<16>(
        args, codes, scales, act_q, act_scales, dst, weight_tile, act_tile, out_tile, tgpig, tid, sgitg);
}










static inline void row4_finish_prefill_m64n32_simd_stripe(
        constant ggml_metal_kargs_row_quant_linear & args,
        device const ushort * scales,
        device const float * act_scales,
        device float * dst,
        threadgroup float * out_tile,
        int row_base,
        int col_base,
        int col_block,
        int row_half,
        uint simd_lane,
        uint sgitg) {
    constexpr uint matrix_elems = 64U;
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint item = 0U; item < 2U; ++item) {
        const uint local  = simd_lane + item * 32U;
        const uint col    = local >> 3;
        const uint row    = local & 7U;
        const uint output = (uint) row_base + sgitg * 16U + (uint) row_half * 8U + row;
        const uint token  = (uint) col_base + (uint) col_block * 8U + col;
        if (token < (uint) args.act_rows) {
            const int exact_acc = (int) out_tile[sgitg * matrix_elems + local];
            dst[(ulong) token * (ulong) args.m + (ulong) output] =
                row4_finish_i32(exact_acc, act_scales[token], scales[output]);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}




kernel void kernel_row4_w1a8_prefill_m64n32_ilp4_native_layout_dualw(
        constant ggml_metal_kargs_row_quant_linear & args [[buffer(0)]],
        device const uchar * codes                        [[buffer(1)]],
        device const ushort * scales                      [[buffer(2)]],
        device const char * act_q                         [[buffer(3)]],
        device const float * act_scales                   [[buffer(4)]],
        device float * dst                                [[buffer(5)]],
        threadgroup half * weight_tile                    [[threadgroup(0)]],
        threadgroup half * act_tile                       [[threadgroup(1)]],
        threadgroup float * out_tile                      [[threadgroup(2)]],
        uint3 tgpig                                       [[threadgroup_position_in_grid]],
        uint tid                                          [[thread_index_in_threadgroup]],
        uint sgitg                                        [[simdgroup_index_in_threadgroup]]) {
    constexpr int row_tile  = 64;
    constexpr int col_tile  = 32;
    constexpr int k_tile    = 32;
    constexpr int n_threads = 128;
    const int row_base      = (int) tgpig.x * row_tile;
    const int col_base      = (int) tgpig.y * col_tile;
    const int k_tiles       = args.k / 128;
    const int row_block     = (int) sgitg * 16;
    const uint simd_lane    = tid & 31U;

    simdgroup_half8x8  weights0;
    simdgroup_half8x8  weights1;
    simdgroup_half8x8  activations0;
    simdgroup_half8x8  activations1;
    simdgroup_half8x8  activations2;
    simdgroup_half8x8  activations3;
    simdgroup_float8x8 accum00 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 accum01 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 accum02 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 accum03 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 accum10 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 accum11 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 accum12 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 accum13 = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (int k_base = 0; k_base < args.k; k_base += k_tile) {
        const int canonical_k_tile = k_base >> 7;
        const int split_base       = (k_base & 127) >> 4;
        for (uint work = tid; work < 64U; work += n_threads) {
            const uint output_quarter = work >> 4;
            const uint tile_work      = work & 15U;
            const uint group          = tile_work >> 2;
            const uint local_split    = (tile_work >> 1) & 1U;
            const uint jblock         = tile_work & 1U;
            const uint output_tile    = (uint) tgpig.x * 4U + output_quarter;
            const ulong code_offset =
                ((((ulong) output_tile * (ulong) k_tiles + (ulong) canonical_k_tile) * 4UL + (ulong) group) * 64UL) +
                (ulong) (split_base + (int) local_split) * 8UL + (ulong) jblock * 4UL;
            const uchar4 packed = *((device const uchar4 *) (codes + code_offset));

            const half4 low0 = row4_decode4((uint) (packed.x & 0x0fU));
            const half4 low1 = row4_decode4((uint) (packed.y & 0x0fU));
            const half4 low2 = row4_decode4((uint) (packed.z & 0x0fU));
            const half4 low3 = row4_decode4((uint) (packed.w & 0x0fU));
            const half4 high0 = row4_decode4((uint) (packed.x >> 4));
            const half4 high1 = row4_decode4((uint) (packed.y >> 4));
            const half4 high2 = row4_decode4((uint) (packed.z >> 4));
            const half4 high3 = row4_decode4((uint) (packed.w >> 4));
            const uint output_base = (output_quarter * 4U + group) * 4U;
            const uint k_low       = local_split * 16U + jblock * 4U;
            const uint k_high      = k_low + 8U;

            *((threadgroup half4 *) (weight_tile + (output_base + 0U) * k_tile + k_low)) =
                half4(low0.x, low1.x, low2.x, low3.x);
            *((threadgroup half4 *) (weight_tile + (output_base + 1U) * k_tile + k_low)) =
                half4(low0.y, low1.y, low2.y, low3.y);
            *((threadgroup half4 *) (weight_tile + (output_base + 2U) * k_tile + k_low)) =
                half4(low0.z, low1.z, low2.z, low3.z);
            *((threadgroup half4 *) (weight_tile + (output_base + 3U) * k_tile + k_low)) =
                half4(low0.w, low1.w, low2.w, low3.w);
            *((threadgroup half4 *) (weight_tile + (output_base + 0U) * k_tile + k_high)) =
                half4(high0.x, high1.x, high2.x, high3.x);
            *((threadgroup half4 *) (weight_tile + (output_base + 1U) * k_tile + k_high)) =
                half4(high0.y, high1.y, high2.y, high3.y);
            *((threadgroup half4 *) (weight_tile + (output_base + 2U) * k_tile + k_high)) =
                half4(high0.z, high1.z, high2.z, high3.z);
            *((threadgroup half4 *) (weight_tile + (output_base + 3U) * k_tile + k_high)) =
                half4(high0.w, high1.w, high2.w, high3.w);
        }

        for (uint index = tid; index < (uint) (col_tile * k_tile); index += n_threads) {
            const uint col   = index / (uint) k_tile;
            const uint kk    = index % (uint) k_tile;
            const uint token = (uint) col_base + col;
            act_tile[kk * (uint) col_tile + col] = token < (uint) args.act_rows ?
                (half) ((int) act_q[(ulong) token * (ulong) args.k + (ulong) k_base + (ulong) kk]) : (half) 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int ik = 0; ik < k_tile; ik += 8) {
            simdgroup_load(weights0, weight_tile + row_block * k_tile + ik, k_tile);
            simdgroup_load(weights1, weight_tile + (row_block + 8) * k_tile + ik, k_tile);
            simdgroup_load(activations0, act_tile + ik * col_tile, col_tile);
            simdgroup_load(activations1, act_tile + ik * col_tile + 8, col_tile);
            simdgroup_load(activations2, act_tile + ik * col_tile + 16, col_tile);
            simdgroup_load(activations3, act_tile + ik * col_tile + 24, col_tile);
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(accum00, weights0, activations0, accum00);
            simdgroup_multiply_accumulate(accum01, weights0, activations1, accum01);
            simdgroup_multiply_accumulate(accum02, weights0, activations2, accum02);
            simdgroup_multiply_accumulate(accum03, weights0, activations3, accum03);
            simdgroup_multiply_accumulate(accum10, weights1, activations0, accum10);
            simdgroup_multiply_accumulate(accum11, weights1, activations1, accum11);
            simdgroup_multiply_accumulate(accum12, weights1, activations2, accum12);
            simdgroup_multiply_accumulate(accum13, weights1, activations3, accum13);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float * simd_out = out_tile + sgitg * 64U;
    simdgroup_store(accum00, simd_out, 8, 0, true);
    row4_finish_prefill_m64n32_simd_stripe(
        args, scales, act_scales, dst, out_tile, row_base, col_base, 0, 0, simd_lane, sgitg);
    simdgroup_store(accum10, simd_out, 8, 0, true);
    row4_finish_prefill_m64n32_simd_stripe(
        args, scales, act_scales, dst, out_tile, row_base, col_base, 0, 1, simd_lane, sgitg);

    simdgroup_store(accum01, simd_out, 8, 0, true);
    row4_finish_prefill_m64n32_simd_stripe(
        args, scales, act_scales, dst, out_tile, row_base, col_base, 1, 0, simd_lane, sgitg);
    simdgroup_store(accum11, simd_out, 8, 0, true);
    row4_finish_prefill_m64n32_simd_stripe(
        args, scales, act_scales, dst, out_tile, row_base, col_base, 1, 1, simd_lane, sgitg);

    simdgroup_store(accum02, simd_out, 8, 0, true);
    row4_finish_prefill_m64n32_simd_stripe(
        args, scales, act_scales, dst, out_tile, row_base, col_base, 2, 0, simd_lane, sgitg);
    simdgroup_store(accum12, simd_out, 8, 0, true);
    row4_finish_prefill_m64n32_simd_stripe(
        args, scales, act_scales, dst, out_tile, row_base, col_base, 2, 1, simd_lane, sgitg);

    simdgroup_store(accum03, simd_out, 8, 0, true);
    row4_finish_prefill_m64n32_simd_stripe(
        args, scales, act_scales, dst, out_tile, row_base, col_base, 3, 0, simd_lane, sgitg);
    simdgroup_store(accum13, simd_out, 8, 0, true);
    row4_finish_prefill_m64n32_simd_stripe(
        args, scales, act_scales, dst, out_tile, row_base, col_base, 3, 1, simd_lane, sgitg);
}

kernel void kernel_row4_w1a8_prefill_m64n32_ilp4_native_layout_dualw_direct_act(
        constant ggml_metal_kargs_row_quant_linear & args [[buffer(0)]],
        device const uchar * codes                        [[buffer(1)]],
        device const ushort * scales                      [[buffer(2)]],
        device const float * act_scales                   [[buffer(4)]],
        device float * dst                                [[buffer(5)]],
        device const half * act_h                         [[buffer(6)]],
        threadgroup half * weight_tile                    [[threadgroup(0)]],
        threadgroup float * out_tile                      [[threadgroup(2)]],
        uint3 tgpig                                       [[threadgroup_position_in_grid]],
        uint tid                                          [[thread_index_in_threadgroup]],
        uint sgitg                                        [[simdgroup_index_in_threadgroup]]) {
    constexpr int row_tile  = 64;
    constexpr int col_tile  = 32;
    constexpr int k_tile    = 32;
    constexpr int n_threads = 128;
    const int row_base      = (int) tgpig.x * row_tile;
    const int col_base      = (int) tgpig.y * col_tile;
    const int k_tiles       = args.k / 128;
    const int row_block     = (int) sgitg * 16;
    const uint simd_lane    = tid & 31U;

    simdgroup_half8x8  weights0;
    simdgroup_half8x8  weights1;
    simdgroup_half8x8  activations0;
    simdgroup_half8x8  activations1;
    simdgroup_half8x8  activations2;
    simdgroup_half8x8  activations3;
    simdgroup_float8x8 accum00 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 accum01 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 accum02 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 accum03 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 accum10 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 accum11 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 accum12 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 accum13 = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (int k_base = 0; k_base < args.k; k_base += k_tile) {
        const int canonical_k_tile = k_base >> 7;
        const int split_base       = (k_base & 127) >> 4;
        for (uint work = tid; work < 64U; work += n_threads) {
            const uint output_quarter = work >> 4;
            const uint tile_work      = work & 15U;
            const uint group          = tile_work >> 2;
            const uint local_split    = (tile_work >> 1) & 1U;
            const uint jblock         = tile_work & 1U;
            const uint output_tile    = (uint) tgpig.x * 4U + output_quarter;
            const ulong code_offset =
                ((((ulong) output_tile * (ulong) k_tiles + (ulong) canonical_k_tile) * 4UL + (ulong) group) * 64UL) +
                (ulong) (split_base + (int) local_split) * 8UL + (ulong) jblock * 4UL;
            const uchar4 packed = *((device const uchar4 *) (codes + code_offset));

            const half4 low0 = row4_decode4((uint) (packed.x & 0x0fU));
            const half4 low1 = row4_decode4((uint) (packed.y & 0x0fU));
            const half4 low2 = row4_decode4((uint) (packed.z & 0x0fU));
            const half4 low3 = row4_decode4((uint) (packed.w & 0x0fU));
            const half4 high0 = row4_decode4((uint) (packed.x >> 4));
            const half4 high1 = row4_decode4((uint) (packed.y >> 4));
            const half4 high2 = row4_decode4((uint) (packed.z >> 4));
            const half4 high3 = row4_decode4((uint) (packed.w >> 4));
            const uint output_base = (output_quarter * 4U + group) * 4U;
            const uint k_low       = local_split * 16U + jblock * 4U;
            const uint k_high      = k_low + 8U;

            *((threadgroup half4 *) (weight_tile + (output_base + 0U) * k_tile + k_low)) =
                half4(low0.x, low1.x, low2.x, low3.x);
            *((threadgroup half4 *) (weight_tile + (output_base + 1U) * k_tile + k_low)) =
                half4(low0.y, low1.y, low2.y, low3.y);
            *((threadgroup half4 *) (weight_tile + (output_base + 2U) * k_tile + k_low)) =
                half4(low0.z, low1.z, low2.z, low3.z);
            *((threadgroup half4 *) (weight_tile + (output_base + 3U) * k_tile + k_low)) =
                half4(low0.w, low1.w, low2.w, low3.w);
            *((threadgroup half4 *) (weight_tile + (output_base + 0U) * k_tile + k_high)) =
                half4(high0.x, high1.x, high2.x, high3.x);
            *((threadgroup half4 *) (weight_tile + (output_base + 1U) * k_tile + k_high)) =
                half4(high0.y, high1.y, high2.y, high3.y);
            *((threadgroup half4 *) (weight_tile + (output_base + 2U) * k_tile + k_high)) =
                half4(high0.z, high1.z, high2.z, high3.z);
            *((threadgroup half4 *) (weight_tile + (output_base + 3U) * k_tile + k_high)) =
                half4(high0.w, high1.w, high2.w, high3.w);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int ik = 0; ik < k_tile; ik += 8) {
            simdgroup_load(weights0, weight_tile + row_block * k_tile + ik, k_tile);
            simdgroup_load(weights1, weight_tile + (row_block + 8) * k_tile + ik, k_tile);
            const int act_base = (k_base + ik) * args.reserved + col_base;
            simdgroup_load(activations0, act_h + act_base, args.reserved);
            simdgroup_load(activations1, act_h + act_base + 8, args.reserved);
            simdgroup_load(activations2, act_h + act_base + 16, args.reserved);
            simdgroup_load(activations3, act_h + act_base + 24, args.reserved);
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(accum00, weights0, activations0, accum00);
            simdgroup_multiply_accumulate(accum01, weights0, activations1, accum01);
            simdgroup_multiply_accumulate(accum02, weights0, activations2, accum02);
            simdgroup_multiply_accumulate(accum03, weights0, activations3, accum03);
            simdgroup_multiply_accumulate(accum10, weights1, activations0, accum10);
            simdgroup_multiply_accumulate(accum11, weights1, activations1, accum11);
            simdgroup_multiply_accumulate(accum12, weights1, activations2, accum12);
            simdgroup_multiply_accumulate(accum13, weights1, activations3, accum13);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float * simd_out = out_tile + sgitg * 64U;
    simdgroup_store(accum00, simd_out, 8, 0, true);
    row4_finish_prefill_m64n32_simd_stripe(
        args, scales, act_scales, dst, out_tile, row_base, col_base, 0, 0, simd_lane, sgitg);
    simdgroup_store(accum10, simd_out, 8, 0, true);
    row4_finish_prefill_m64n32_simd_stripe(
        args, scales, act_scales, dst, out_tile, row_base, col_base, 0, 1, simd_lane, sgitg);

    simdgroup_store(accum01, simd_out, 8, 0, true);
    row4_finish_prefill_m64n32_simd_stripe(
        args, scales, act_scales, dst, out_tile, row_base, col_base, 1, 0, simd_lane, sgitg);
    simdgroup_store(accum11, simd_out, 8, 0, true);
    row4_finish_prefill_m64n32_simd_stripe(
        args, scales, act_scales, dst, out_tile, row_base, col_base, 1, 1, simd_lane, sgitg);

    simdgroup_store(accum02, simd_out, 8, 0, true);
    row4_finish_prefill_m64n32_simd_stripe(
        args, scales, act_scales, dst, out_tile, row_base, col_base, 2, 0, simd_lane, sgitg);
    simdgroup_store(accum12, simd_out, 8, 0, true);
    row4_finish_prefill_m64n32_simd_stripe(
        args, scales, act_scales, dst, out_tile, row_base, col_base, 2, 1, simd_lane, sgitg);

    simdgroup_store(accum03, simd_out, 8, 0, true);
    row4_finish_prefill_m64n32_simd_stripe(
        args, scales, act_scales, dst, out_tile, row_base, col_base, 3, 0, simd_lane, sgitg);
    simdgroup_store(accum13, simd_out, 8, 0, true);
    row4_finish_prefill_m64n32_simd_stripe(
        args, scales, act_scales, dst, out_tile, row_base, col_base, 3, 1, simd_lane, sgitg);
}








kernel void kernel_row8_w8a8_prefill(
        constant ggml_metal_kargs_row_quant_linear & args [[buffer(0)]],
        device const char * codes                         [[buffer(1)]],
        device const float * scales                       [[buffer(2)]],
        device const char * act_q                         [[buffer(3)]],
        device const float * act_scales                   [[buffer(4)]],
        device float * dst                                [[buffer(5)]],
        threadgroup half * weight_tile                    [[threadgroup(0)]],
        threadgroup half * act_tile                       [[threadgroup(1)]],
        threadgroup float * out_tile                      [[threadgroup(2)]],
        uint3 tgpig                                       [[threadgroup_position_in_grid]],
        uint tid                                          [[thread_index_in_threadgroup]],
        uint sgitg                                        [[simdgroup_index_in_threadgroup]]) {
    constexpr int row_tile    = 16;
    constexpr int col_tile    = 16;
    constexpr int k_tile      = 32;
    constexpr int k_segment   = 1024;
    constexpr int n_threads   = 128;
    const int row_base        = (int) tgpig.x * row_tile;
    const int col_base        = (int) tgpig.y * col_tile;
    const int k_tiles         = args.k / 128;
    const int row_block       = (int) (sgitg & 1U);
    const int col_block       = (int) (sgitg >> 1);
    int exact_acc0            = 0;
    int exact_acc1            = 0;

    for (int segment_base = 0; segment_base < args.k; segment_base += k_segment) {
        simdgroup_half8x8  weights;
        simdgroup_half8x8  activations;
        simdgroup_float8x8 segment_accum = make_filled_simdgroup_matrix<float, 8>(0.0f);
        const int segment_end = min(segment_base + k_segment, args.k);

        for (int k_base = segment_base; k_base < segment_end; k_base += k_tile) {
            for (uint index = tid; index < (uint) (row_tile * k_tile); index += n_threads) {
                const int row = (int) index / k_tile;
                const int ik  = (int) index % k_tile;
                weight_tile[index] =
                    (half) w8a8_weight_at(codes, k_tiles, row_base + row, k_base + ik);

                const int kk    = (int) index / col_tile;
                const int col   = (int) index % col_tile;
                const int token = col_base + col;
                act_tile[index] = token < args.act_rows ?
                    (half) ((int) act_q[(ulong) token * (ulong) args.k + (ulong) (k_base + kk)]) : (half) 0.0h;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int ik = 0; ik < k_tile; ik += 8) {
                simdgroup_load(weights, weight_tile + row_block * 8 * k_tile + ik, k_tile);
                simdgroup_load(activations, act_tile + ik * col_tile + col_block * 8, col_tile);
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_multiply_accumulate(segment_accum, weights, activations, segment_accum);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_store(segment_accum, out_tile + sgitg * 64U, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        exact_acc0 += (int) out_tile[tid];
        exact_acc1 += (int) out_tile[tid + n_threads];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint indices[2] = { tid, tid + n_threads };
    const int accumulators[2] = { exact_acc0, exact_acc1 };
    for (int item = 0; item < 2; ++item) {
        const uint index     = indices[item];
        const uint matrix    = index >> 6;
        const uint cell      = index & 63U;
        const uint row       = (matrix & 1U) * 8U + cell / 8U;
        const uint col       = (matrix >> 1) * 8U + cell % 8U;
        const uint output    = (uint) row_base + row;
        const uint token     = (uint) col_base + col;
        if (token < (uint) args.act_rows) {
            dst[(ulong) token * (ulong) args.m + (ulong) output] =
                w8a8_finish_i32(accumulators[item], act_scales[token], scales[output]);
        }
    }
}

#undef ROW4_REDUCED_KERNEL
#undef W8A8_REDUCED_KERNEL

kernel void kernel_fairy2i_act_half_64_stage_bf16(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const char * x                             [[buffer(1)]],
        device half * act_h                               [[buffer(2)]],
        uint2 tgpig                                       [[threadgroup_position_in_grid]],
        uint2 tiitg                                       [[thread_position_in_threadgroup]]) {
    const uint block = tgpig.x;
    const uint act_row = tgpig.y;
    const uint lid = tiitg.x;
    const int i1 = (int) act_row % args.x_ne1;
    const int i2 = ((int) act_row / args.x_ne1) % args.x_ne2;
    const int i3 = (int) act_row / (args.x_ne1 * args.x_ne2);
    const int j = (int) lid;
    const int k_idx = (int) block * QK_FAIRY2I_ACT_Q16_64 + j;

    const uint pair = *((device const uint *) (x + (ulong) i1 * args.x_nb1 + (ulong) i2 * args.x_nb2 +
                                               (ulong) i3 * args.x_nb3 + (ulong) k_idx * args.x_nb0));
    const int blocks = args.k / QK_FAIRY2I_ACT_Q16_64;
    const int act_index = (int) act_row * blocks + (int) block;
    const int h_base = act_index * (2 * QK_FAIRY2I_ACT_Q16_64);

    act_h[h_base + j] = (half) fairy2i_bf16_to_f32((ushort) (pair & 0xffffU));
    act_h[h_base + QK_FAIRY2I_ACT_Q16_64 + j] = (half) fairy2i_bf16_to_f32((ushort) (pair >> 16));
}

#if defined(GGML_METAL_HAS_BF16)
kernel void kernel_fairy2i_act_bfloat_64_stage_bf16_exact(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const char * x                             [[buffer(1)]],
        device ushort * act_b                             [[buffer(2)]],
        device uint * block_metrics                       [[buffer(3)]],
        threadgroup uint * simd_metrics                   [[threadgroup(0)]],
        uint2 tgpig                                       [[threadgroup_position_in_grid]],
        uint2 tiitg                                       [[thread_position_in_threadgroup]]) {
    const uint block = tgpig.x;
    const uint act_row = tgpig.y;
    const uint lid = tiitg.x;
    const int i1 = (int) act_row % args.x_ne1;
    const int i2 = ((int) act_row / args.x_ne1) % args.x_ne2;
    const int i3 = (int) act_row / (args.x_ne1 * args.x_ne2);
    const int j = (int) lid;
    const int k_idx = (int) block * QK_FAIRY2I_ACT_Q16_64 + j;

    const uint pair = *((device const uint *) (x + (ulong) i1 * args.x_nb1 + (ulong) i2 * args.x_nb2 +
                                               (ulong) i3 * args.x_nb3 + (ulong) k_idx * args.x_nb0));
    const int blocks = args.k / QK_FAIRY2I_ACT_Q16_64;
    const int act_index = (int) act_row * blocks + (int) block;
    const int b_base = act_index * (2 * QK_FAIRY2I_ACT_Q16_64);

    // The graph's F32-tagged complex value already contains two BF16 payloads. Preserve the payload bits
    // verbatim in the bfloat staging buffer instead of introducing a numeric conversion.
    act_b[b_base + j] = (ushort) (pair & 0xffffU);
    act_b[b_base + QK_FAIRY2I_ACT_Q16_64 + j] = (ushort) (pair >> 16);

    const uint thread_metric = min(
        fairy2i_bf16_product_metric((ushort) (pair & 0xffffU)),
        fairy2i_bf16_product_metric((ushort) (pair >> 16)));
    const uint simd_lane = lid & 31U;
    const uint simd_group = lid >> 5;
    const uint simd_metric = simd_min(thread_metric);
    if (simd_lane == 0U) {
        simd_metrics[simd_group] = simd_metric;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0U) {
        block_metrics[act_index] = min(simd_metrics[0], simd_metrics[1]);
    }
}

kernel void kernel_fairy2i_act_bfloat_64_metric_bf16_exact(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const char * x                             [[buffer(1)]],
        device uint * block_metrics                       [[buffer(2)]],
        threadgroup uint * simd_metrics                   [[threadgroup(0)]],
        uint2 tgpig                                       [[threadgroup_position_in_grid]],
        uint2 tiitg                                       [[thread_position_in_threadgroup]]) {
    const uint block = tgpig.x;
    const uint lid = tiitg.x;
    const uint simd_lane = lid & 31U;
    const uint simd_group = lid >> 5;
    const int k_idx = (int) block * QK_FAIRY2I_ACT_Q16_64 + (int) lid;
    const uint pair = *((device const uint *) (x + (ulong) k_idx * args.x_nb0));
    const uint thread_metric = min(
        fairy2i_bf16_product_metric((ushort) (pair & 0xffffU)),
        fairy2i_bf16_product_metric((ushort) (pair >> 16)));
    const uint simd_metric = simd_min(thread_metric);
    if (simd_lane == 0U) {
        simd_metrics[simd_group] = simd_metric;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0U) {
        block_metrics[block] = min(simd_metrics[0], simd_metrics[1]);
    }
}

#endif

kernel void kernel_fairy2i_act_half_64_stage_bf16_kmajor(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const char * x                             [[buffer(1)]],
        device half * act_h                               [[buffer(2)]],
        uint2 tgpig                                       [[threadgroup_position_in_grid]],
        uint2 tiitg                                       [[thread_position_in_threadgroup]]) {
    const uint block = tgpig.x;
    const uint act_row = tgpig.y;
    const uint j = tiitg.x;
    const int i1 = (int) act_row % args.x_ne1;
    const int i2 = ((int) act_row / args.x_ne1) % args.x_ne2;
    const int i3 = (int) act_row / (args.x_ne1 * args.x_ne2);
    const int k_idx = (int) block * QK_FAIRY2I_ACT_Q16_64 + (int) j;

    const uint pair = *((device const uint *) (x + (ulong) i1 * args.x_nb1 + (ulong) i2 * args.x_nb2 +
                                               (ulong) i3 * args.x_nb3 + (ulong) k_idx * args.x_nb0));
    const ulong plane_size = (ulong) QK_FAIRY2I_ACT_Q16_64 * (ulong) FC_fairy2i_bundle_w1_prefill_act_rows;
    const ulong block_base = (ulong) block * 2 * plane_size;
    const ulong kmajor_index = (ulong) j * (ulong) FC_fairy2i_bundle_w1_prefill_act_rows + (ulong) act_row;

    act_h[block_base + kmajor_index] = (half) fairy2i_bf16_to_f32((ushort) (pair & 0xffffU));
    act_h[block_base + plane_size + kmajor_index] = (half) fairy2i_bf16_to_f32((ushort) (pair >> 16));
}

#if defined(GGML_METAL_HAS_BF16)
kernel void kernel_fairy2i_act_bfloat_64_stage_bf16_kmajor_exact(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const char * x                             [[buffer(1)]],
        device ushort * act_b                             [[buffer(2)]],
        device uint * block_metrics                       [[buffer(3)]],
        threadgroup uint * simd_metrics                   [[threadgroup(0)]],
        uint2 tgpig                                       [[threadgroup_position_in_grid]],
        uint2 tiitg                                       [[thread_position_in_threadgroup]]) {
    const uint block = tgpig.x;
    const uint act_row = tgpig.y;
    const uint j = tiitg.x;
    const int i1 = (int) act_row % args.x_ne1;
    const int i2 = ((int) act_row / args.x_ne1) % args.x_ne2;
    const int i3 = (int) act_row / (args.x_ne1 * args.x_ne2);
    const int k_idx = (int) block * QK_FAIRY2I_ACT_Q16_64 + (int) j;

    const uint pair = *((device const uint *) (x + (ulong) i1 * args.x_nb1 + (ulong) i2 * args.x_nb2 +
                                               (ulong) i3 * args.x_nb3 + (ulong) k_idx * args.x_nb0));
    const ulong plane_size = (ulong) QK_FAIRY2I_ACT_Q16_64 * (ulong) FC_fairy2i_bundle_w1_prefill_act_rows;
    const ulong block_base = (ulong) block * 2 * plane_size;
    const ulong kmajor_index = (ulong) j * (ulong) FC_fairy2i_bundle_w1_prefill_act_rows + (ulong) act_row;

    act_b[block_base + kmajor_index] = (ushort) (pair & 0xffffU);
    act_b[block_base + plane_size + kmajor_index] = (ushort) (pair >> 16);

    const uint thread_metric = min(
        fairy2i_bf16_product_metric((ushort) (pair & 0xffffU)),
        fairy2i_bf16_product_metric((ushort) (pair >> 16)));
    const uint simd_lane = j & 31U;
    const uint simd_group = j >> 5;
    const uint simd_metric = simd_min(thread_metric);
    if (simd_lane == 0U) {
        simd_metrics[simd_group] = simd_metric;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j == 0U) {
        const int blocks = args.k / QK_FAIRY2I_ACT_Q16_64;
        block_metrics[(int) act_row * blocks + (int) block] = min(simd_metrics[0], simd_metrics[1]);
    }
}
#endif

static inline float fairy2i_sum_float4(float4 v) {
    return v.x + v.y + v.z + v.w;
}

static inline float fairy2i_sum_float2(float2 v) {
    return v.x + v.y;
}

static inline half4 fairy2i_mma_coeff_w1_codes_scaled_half(uint2 code, half2 wr, half2 wi) {
    const bool2 positive = (code & uint2(1)) == uint2(1);
    const bool2 is_real = (code & uint2(2)) == uint2(0);
    const half2 scale = select(wi, wr, is_real);
    const half2 signed_scale = select(-scale, scale, positive);
    const half2 real_coeff = select(half2(0.0h), signed_scale, is_real);
    const half2 imag_coeff = select(half2(0.0h), signed_scale, !is_real);

    return half4(
        real_coeff.x + real_coeff.y,
        -imag_coeff.x + imag_coeff.y,
        imag_coeff.x + imag_coeff.y,
        real_coeff.x - real_coeff.y);
}

static inline ushort2 fairy2i_w1_signed_stage_bf16_exact_bits(
        uint code,
        ushort scale_real,
        ushort scale_imag) {
    const ushort scale = (code & 2U) == 0U ? scale_real : scale_imag;
    const ushort stage = (ushort) ((uint) scale ^ ((code & 1U) != 0U ? 0x0000U : 0x8000U));
    return (code & 2U) == 0U ? ushort2(stage, 0) : ushort2(0, stage);
}

static inline ushort4 fairy2i_mma_coeff_w1_codes_scaled_bf16_exact_bits(
        uint2 code,
        ushort2 scale_real,
        ushort2 scale_imag) {
    const ushort2 u = fairy2i_w1_signed_stage_bf16_exact_bits(
        code.x, scale_real.x, scale_imag.x);
    const ushort2 w = fairy2i_w1_signed_stage_bf16_exact_bits(
        code.y, scale_real.y, scale_imag.y);

    // U/W are already BF16 signed stages. A11/A12/A21/A22 are four separate
    // BF16 RNE result boundaries, matching the training-side reconstruction.
    return ushort4(
        fairy2i_add_bf16_bits_rne(u.x, w.x),
        fairy2i_add_bf16_bits_rne((ushort) ((uint) u.y ^ 0x8000U), w.y),
        fairy2i_add_bf16_bits_rne(u.y, w.y),
        fairy2i_add_bf16_bits_rne(u.x, (ushort) ((uint) w.x ^ 0x8000U)));
}

static inline void fairy2i_build_bundle_w1_coeff_lut_bf16_exact(
        threadgroup ushort4 * coeff_lut,
        ushort2 scale_real,
        ushort2 scale_imag,
        uint tid,
        uint n_threads) {
    for (uint pattern = tid; pattern < 16U; pattern += n_threads) {
        coeff_lut[pattern] = fairy2i_mma_coeff_w1_codes_scaled_bf16_exact_bits(
            uint2(pattern & 3U, (pattern >> 2) & 3U), scale_real, scale_imag);
    }
}

static inline ushort4 fairy2i_bundle_w1_coeff_at_bf16_exact_bits(
        device const uchar * codes,
        device const ushort * scales,
        int blocks,
        int row,
        int k) {
    const int physical_tile = (row / QK_FAIRY2I_TILE64) * blocks + k / QK_FAIRY2I_TILE64;
    const int slot = ((row & 63) >> 4) * 16 + ((k & 63) >> 2);
    const int lane = row & 15;
    const int q = k & 3;
    const ulong code_base =
        ((((ulong) physical_tile * 64 + (ulong) slot) * 2) * 16) + (ulong) lane;
    const uint2 code = (uint2(
        (uint) codes[code_base],
        (uint) codes[code_base + 16]) >> uint2(2 * q)) & uint2(3);
    const ushort4 scale = *((device const ushort4 *) (scales + (ulong) physical_tile * 4UL));
    return fairy2i_mma_coeff_w1_codes_scaled_bf16_exact_bits(
        code, scale.xz, scale.yw);
}

static inline float4 fairy2i_mma_coeff_w2_codes_scaled(uint4 code, float4 wr, float4 wi) {
    const float4 sign = select(float4(-1.0f), float4(1.0f), (code & uint4(1)) == uint4(1));
    const bool4 is_real = (code & uint4(2)) == uint4(0);
    const float4 real_coeff = select(float4(0.0f), sign * wr, is_real);
    const float4 imag_coeff = select(float4(0.0f), sign * wi, !is_real);
    const float u_real = real_coeff.x + real_coeff.y;
    const float w_real = real_coeff.z + real_coeff.w;
    const float u_imag = imag_coeff.x + imag_coeff.y;
    const float w_imag = imag_coeff.z + imag_coeff.w;

    return float4(u_real + w_real, -u_imag + w_imag, u_imag + w_imag, u_real - w_real);
}

static inline ushort4 fairy2i_reconstruct_w2_coeff_from_uw_bf16_exact_bits(ushort2 u, ushort2 w) {
    // The final A11..A22 reconstruction is a distinct BF16 result boundary.
    return ushort4(
        fairy2i_add_bf16_bits_rne(u.x, w.x),
        fairy2i_add_bf16_bits_rne((ushort) ((uint) u.y ^ 0x8000U), w.y),
        fairy2i_add_bf16_bits_rne(u.y, w.y),
        fairy2i_add_bf16_bits_rne(u.x, (ushort) ((uint) w.x ^ 0x8000U)));
}

static inline ushort4 fairy2i_reconstruct_w2_coeff_from_stage_bf16_exact_bits(ushort4 real, ushort4 imag) {
    // The stage merge (U/W) and final A11..A22 reconstruction are separate BF16 result boundaries.
    // Keep them on payload bits so Metal's F32 flush-to-zero behavior cannot erase BF16 subnormals.
    const ushort2 u = ushort2(
        fairy2i_add_bf16_bits_rne(real.x, real.y),
        fairy2i_add_bf16_bits_rne(imag.x, imag.y));
    const ushort2 w = ushort2(
        fairy2i_add_bf16_bits_rne(real.z, real.w),
        fairy2i_add_bf16_bits_rne(imag.z, imag.w));

    return fairy2i_reconstruct_w2_coeff_from_uw_bf16_exact_bits(u, w);
}

static inline ushort4 fairy2i_reconstruct_w2_coeff_from_stage_collapsed_f32_native_bits(
        ushort4 real,
        ushort4 imag) {
    // Production tiles contain finite, normal BF16 scales. Widen them exactly,
    // collapse U/W/Aij with native F32 adds, and round only the final Aij values
    // to BF16. Exceptional tiles are discarded and replayed by the software path.
    const float4 real_f = as_type<float4>(uint4(real) << uint4(16));
    const float4 imag_f = as_type<float4>(uint4(imag) << uint4(16));
    const float  u_real = real_f.x + real_f.y;
    const float  u_imag = imag_f.x + imag_f.y;
    const float  w_real = real_f.z + real_f.w;
    const float  w_imag = imag_f.z + imag_f.w;

    return fairy2i_f32_to_bf16(
        float4(u_real + w_real, -u_imag + w_imag, u_imag + w_imag, u_real - w_real));
}

static inline ushort4 fairy2i_reconstruct_w2_coeff_from_stage_collapsed_f32_software_bits(
        ushort4 real,
        ushort4 imag) {
    // The stored stage values are BF16, but all U/W/Aij additions stay in F32.
    // The replay path keeps the additions in the bit domain so Metal FTZ cannot
    // erase valid F32/BF16 subnormals.
    const uint u_real = fairy2i_add_f32_bits_rne((uint) real.x << 16, (uint) real.y << 16);
    const uint u_imag = fairy2i_add_f32_bits_rne((uint) imag.x << 16, (uint) imag.y << 16);
    const uint w_real = fairy2i_add_f32_bits_rne((uint) real.z << 16, (uint) real.w << 16);
    const uint w_imag = fairy2i_add_f32_bits_rne((uint) imag.z << 16, (uint) imag.w << 16);

    return ushort4(
        fairy2i_f32_to_bf16(as_type<float>(fairy2i_add_f32_bits_rne(u_real, w_real))),
        fairy2i_f32_to_bf16(as_type<float>(fairy2i_add_f32_bits_rne(u_imag ^ 0x80000000U, w_imag))),
        fairy2i_f32_to_bf16(as_type<float>(fairy2i_add_f32_bits_rne(u_imag, w_imag))),
        fairy2i_f32_to_bf16(as_type<float>(fairy2i_add_f32_bits_rne(u_real, w_real ^ 0x80000000U))));
}

static inline ushort4 fairy2i_mma_coeff_w2_codes_scaled_bf16_exact_bits(
        uint4 code,
        ushort4 scale_real,
        ushort4 scale_imag) {
    // The converter stores the positive, forward-visible BF16 scale payloads.
    // Apply the code sign by flipping the payload sign bit without another conversion.
    const bool4 is_real  = (code & uint4(2)) == uint4(0);
    const bool4 positive = (code & uint4(1)) == uint4(1);
    ushort4      real     = ushort4(0);
    ushort4      imag     = ushort4(0);

    FOR_UNROLL (int branch = 0; branch < 4; ++branch) {
        const ushort magnitude = is_real[branch] ? scale_real[branch] : scale_imag[branch];
        const ushort stage = (ushort) ((uint) magnitude ^ (positive[branch] ? 0x0000U : 0x8000U));
        if (is_real[branch]) {
            real[branch] = stage;
        } else {
            imag[branch] = stage;
        }
    }

    return fairy2i_reconstruct_w2_coeff_from_stage_bf16_exact_bits(real, imag);
}

static inline ushort4 fairy2i_mma_coeff_w2_codes_scaled_bf16_collapsed_f32_bits(
        uint4 code,
        ushort4 scale_real,
        ushort4 scale_imag,
        bool software_replay) {
    if (!software_replay) {
        const float4 coeff = fairy2i_mma_coeff_w2_codes_scaled(
            code,
            as_type<float4>(uint4(scale_real) << uint4(16)),
            as_type<float4>(uint4(scale_imag) << uint4(16)));
        return fairy2i_f32_to_bf16(coeff);
    }

    const bool4 is_real  = (code & uint4(2)) == uint4(0);
    const bool4 positive = (code & uint4(1)) == uint4(1);
    ushort4      real     = ushort4(0);
    ushort4      imag     = ushort4(0);

    FOR_UNROLL (int branch = 0; branch < 4; ++branch) {
        const ushort magnitude = is_real[branch] ? scale_real[branch] : scale_imag[branch];
        const ushort stage = (ushort) ((uint) magnitude ^ (positive[branch] ? 0x0000U : 0x8000U));
        if (is_real[branch]) {
            real[branch] = stage;
        } else {
            imag[branch] = stage;
        }
    }

    return software_replay ?
               fairy2i_reconstruct_w2_coeff_from_stage_collapsed_f32_software_bits(real, imag) :
               fairy2i_reconstruct_w2_coeff_from_stage_collapsed_f32_native_bits(real, imag);
}

static inline uint fairy2i_bundle_w2_coeff_lut_index(uint4 code) {
    return code.x | (code.y << 2) | (code.z << 4) | (code.w << 6);
}

static inline void fairy2i_build_bundle_w2_coeff_lut_bf16_exact(
        threadgroup ushort4 * coeff_lut,
        ushort4 scale_real,
        ushort4 scale_imag,
        uint tid,
        uint n_threads,
        bool strict_staged_reconstruction) {
    for (uint pattern = tid; pattern < 256; pattern += n_threads) {
        const uint4 code = uint4(
            pattern & 3U,
            (pattern >> 2) & 3U,
            (pattern >> 4) & 3U,
            (pattern >> 6) & 3U);
        coeff_lut[pattern] = strict_staged_reconstruction ?
                                 fairy2i_mma_coeff_w2_codes_scaled_bf16_exact_bits(
                                     code, scale_real, scale_imag) :
                                 fairy2i_mma_coeff_w2_codes_scaled_bf16_collapsed_f32_bits(
                                     code, scale_real, scale_imag, false);
    }
}

static inline ushort4 fairy2i_bundle_w2_coeff_at_bf16_exact_bits(
        device const uchar * codes,
        device const ushort * scales,
        int blocks,
        int row,
        int k) {
    const int physical_tile = (row / QK_FAIRY2I_TILE64) * blocks + k / QK_FAIRY2I_TILE64;
    const int slot = ((row & 63) >> 4) * 16 + ((k & 63) >> 2);
    const int lane = row & 15;
    const int q = k & 3;
    const ulong code_base =
        ((((ulong) physical_tile * 64 + (ulong) slot) * 4) * 16) + (ulong) lane;
    const uint4 code = (uint4(
        (uint) codes[code_base],
        (uint) codes[code_base + 16],
        (uint) codes[code_base + 32],
        (uint) codes[code_base + 48]) >> uint4(2 * q)) & uint4(3);
    const int scale_base = physical_tile * 8;
    const ushort4 scale01 = *((device const ushort4 *) (scales + scale_base));
    const ushort4 scale23 = *((device const ushort4 *) (scales + scale_base + 4));
    const ushort4 scale_real = ushort4(scale01.x, scale01.z, scale23.x, scale23.z);
    const ushort4 scale_imag = ushort4(scale01.y, scale01.w, scale23.y, scale23.w);
    return fairy2i_mma_coeff_w2_codes_scaled_bf16_exact_bits(code, scale_real, scale_imag);
}

static inline ushort4 fairy2i_bundle_w2_coeff_at_bf16_collapsed_f32_bits(
        device const uchar * codes,
        device const ushort * scales,
        int blocks,
        int row,
        int k) {
    const int physical_tile = (row / QK_FAIRY2I_TILE64) * blocks + k / QK_FAIRY2I_TILE64;
    const int slot = ((row & 63) >> 4) * 16 + ((k & 63) >> 2);
    const int lane = row & 15;
    const int q = k & 3;
    const ulong code_base =
        ((((ulong) physical_tile * 64 + (ulong) slot) * 4) * 16) + (ulong) lane;
    const uint4 code = (uint4(
        (uint) codes[code_base],
        (uint) codes[code_base + 16],
        (uint) codes[code_base + 32],
        (uint) codes[code_base + 48]) >> uint4(2 * q)) & uint4(3);
    const int scale_base = physical_tile * 8;
    const ushort4 scale01 = *((device const ushort4 *) (scales + scale_base));
    const ushort4 scale23 = *((device const ushort4 *) (scales + scale_base + 4));
    const ushort4 scale_real = ushort4(scale01.x, scale01.z, scale23.x, scale23.z);
    const ushort4 scale_imag = ushort4(scale01.y, scale01.w, scale23.y, scale23.w);
    return fairy2i_mma_coeff_w2_codes_scaled_bf16_collapsed_f32_bits(
        code, scale_real, scale_imag, true);
}

static inline uint fairy2i_accumulate_bf16_product_f32_bits_rne(uint acc_bits, ushort lhs, ushort rhs) {
    return fairy2i_fma_bf16_bf16_f32_bits_rne(lhs, rhs, acc_bits);
}

static inline half2 fairy2i_load_staged_half_activation_pair(device const half * act_h, int col, int wb, int blocks, int k) {
    const int act_index = col * blocks + wb;
    const int h_base = act_index * (2 * QK_FAIRY2I_ACT_Q16_64);
    return half2(act_h[h_base + k], act_h[h_base + QK_FAIRY2I_ACT_Q16_64 + k]);
}

#if defined(GGML_METAL_HAS_BF16)
static inline void fairy2i_load_staged_bfloat_activation_pair(
        device const bfloat * act_b,
        int col,
        int wb,
        int blocks,
        int k,
        thread bfloat & real,
        thread bfloat & imag) {
    const int act_index = col * blocks + wb;
    const int b_base = act_index * (2 * QK_FAIRY2I_ACT_Q16_64);
    real = act_b[b_base + k];
    imag = act_b[b_base + QK_FAIRY2I_ACT_Q16_64 + k];
}
#endif

static inline void fairy2i_accumulate_tile_weight4_decode_f32_reg(
        uint4 packed,
        float4 wr,
        float4 wi,
        float4 act_real,
        float4 act_imag,
        int out_idx,
        thread float * acc_real,
        thread float * acc_imag) {
    float4 sum_ac = float4(0.0f);
    float4 sum_ad = float4(0.0f);
    float4 sum_bc = float4(0.0f);
    float4 sum_bd = float4(0.0f);

    FOR_UNROLL (int part = 0; part < 4; ++part) {
        const float xr = act_real[part];
        const float xi = act_imag[part];

        const uint4 code = (packed >> (uint) (2 * part)) & uint4(3);
        const float4 sign = select(float4(-1.0f), float4(1.0f), (code & uint4(1)) == uint4(1));
        const bool4 is_imag = (code & uint4(2)) == uint4(2);
        const bool4 is_real = !is_imag;

        const float4 signed_xr = float4(xr) * sign;
        const float4 signed_xi = float4(xi) * sign;

        sum_ac += select(float4(0.0f), signed_xr, is_real);
        sum_ad += select(float4(0.0f), signed_xi, is_real);
        sum_bc += select(float4(0.0f), signed_xr, is_imag);
        sum_bd += select(float4(0.0f), signed_xi, is_imag);
    }

    const float4 bd_sign = float4(-1.0f, -1.0f, 1.0f, 1.0f);
    const float4 ad_sign = float4(1.0f, 1.0f, -1.0f, -1.0f);

    acc_real[out_idx] += fairy2i_sum_float4(wr * sum_ac + bd_sign * wi * sum_bd);
    acc_imag[out_idx] += fairy2i_sum_float4(wi * sum_bc + ad_sign * wr * sum_ad);
}

static inline void fairy2i_accumulate_tile_weight4_decode_bf16_exact_f32_reg(
        uint4 packed,
        threadgroup const ushort2 * qstage_table,
        threadgroup const ushort2 * uw_table,
        bool strict_staged_reconstruction,
        float4 act_real,
        float4 act_imag,
        int out_idx,
        thread float * acc_real,
        thread float * acc_imag,
        thread uint & min_coeff_metric) {
    FOR_UNROLL (int part = 0; part < 4; ++part) {
        const uint4 code = (packed >> (uint) (2 * part)) & uint4(3);
        ushort4 coeff_bits;
        if (strict_staged_reconstruction) {
            const uint u_pattern = code.x | (code.y << 2);
            const uint w_pattern = code.z | (code.w << 2);
            coeff_bits = fairy2i_reconstruct_w2_coeff_from_uw_bf16_exact_bits(
                uw_table[u_pattern], uw_table[16 + w_pattern]);
        } else {
            const ushort2 u0 = qstage_table[code.x];
            const ushort2 u1 = qstage_table[4 + code.y];
            const ushort2 w0 = qstage_table[8 + code.z];
            const ushort2 w1 = qstage_table[12 + code.w];
            coeff_bits = fairy2i_reconstruct_w2_coeff_from_stage_collapsed_f32_native_bits(
                ushort4(u0.x, u1.x, w0.x, w1.x),
                ushort4(u0.y, u1.y, w0.y, w1.y));
        }
        min_coeff_metric = min(min_coeff_metric, fairy2i_bf16_product_metric(coeff_bits.x));
        min_coeff_metric = min(min_coeff_metric, fairy2i_bf16_product_metric(coeff_bits.y));
        min_coeff_metric = min(min_coeff_metric, fairy2i_bf16_product_metric(coeff_bits.z));
        min_coeff_metric = min(min_coeff_metric, fairy2i_bf16_product_metric(coeff_bits.w));
        const float4 coeff = float4(
            fairy2i_bf16_to_f32(coeff_bits.x),
            fairy2i_bf16_to_f32(coeff_bits.y),
            fairy2i_bf16_to_f32(coeff_bits.z),
            fairy2i_bf16_to_f32(coeff_bits.w));
        const float xr = act_real[part];
        const float xi = act_imag[part];

        acc_real[out_idx] += coeff.x * xr + coeff.y * xi;
        acc_imag[out_idx] += coeff.z * xr + coeff.w * xi;
    }
}

static inline void fairy2i_accumulate_tile_weight2_decode_f32_reg(
        uint2 packed,
        float2 wr,
        float2 wi,
        float4 act_real,
        float4 act_imag,
        int out_idx,
        thread float * acc_real,
        thread float * acc_imag) {
    float2 sum_ac = float2(0.0f);
    float2 sum_ad = float2(0.0f);
    float2 sum_bc = float2(0.0f);
    float2 sum_bd = float2(0.0f);

    FOR_UNROLL (int part = 0; part < 4; ++part) {
        const float xr = act_real[part];
        const float xi = act_imag[part];

        const uint2 code = (packed >> (uint) (2 * part)) & uint2(3);
        const float2 sign = select(float2(-1.0f), float2(1.0f), (code & uint2(1)) == uint2(1));
        const bool2 is_imag = (code & uint2(2)) == uint2(2);
        const bool2 is_real = !is_imag;

        const float2 signed_xr = float2(xr) * sign;
        const float2 signed_xi = float2(xi) * sign;

        sum_ac += select(float2(0.0f), signed_xr, is_real);
        sum_ad += select(float2(0.0f), signed_xi, is_real);
        sum_bc += select(float2(0.0f), signed_xr, is_imag);
        sum_bd += select(float2(0.0f), signed_xi, is_imag);
    }

    const float2 bd_sign = float2(-1.0f, 1.0f);
    const float2 ad_sign = float2(1.0f, -1.0f);

    acc_real[out_idx] += fairy2i_sum_float2(wr * sum_ac + bd_sign * wi * sum_bd);
    acc_imag[out_idx] += fairy2i_sum_float2(wi * sum_bc + ad_sign * wr * sum_ad);
}

constant int FC_fairy2i_w2_decode_blocks  [[function_constant(FC_FAIRY2I_W2_DECODE + 0)]];
constant int FC_fairy2i_w2_decode_x_nb0   [[function_constant(FC_FAIRY2I_W2_DECODE + 1)]];
constant int FC_fairy2i_w2_decode_dst_nb0 [[function_constant(FC_FAIRY2I_W2_DECODE + 2)]];
constant int FC_fairy2i_w1_decode_blocks  [[function_constant(FC_FAIRY2I_W1_DECODE + 0)]];
constant int FC_fairy2i_w1_decode_x_nb0   [[function_constant(FC_FAIRY2I_W1_DECODE + 1)]];
constant int FC_fairy2i_w1_decode_dst_nb0 [[function_constant(FC_FAIRY2I_W1_DECODE + 2)]];
constant int FC_fairy2i_bundle_w1_decode_blocks  [[function_constant(FC_FAIRY2I_BUNDLE_W1_DECODE + 0)]];
constant int FC_fairy2i_bundle_w1_decode_x_nb0   [[function_constant(FC_FAIRY2I_BUNDLE_W1_DECODE + 1)]];
constant int FC_fairy2i_bundle_w1_decode_dst_nb0 [[function_constant(FC_FAIRY2I_BUNDLE_W1_DECODE + 2)]];
constant int FC_fairy2i_bundle_w2_decode_blocks  [[function_constant(FC_FAIRY2I_BUNDLE_W2_DECODE + 0)]];
constant int FC_fairy2i_bundle_w2_decode_x_nb0   [[function_constant(FC_FAIRY2I_BUNDLE_W2_DECODE + 1)]];
constant int FC_fairy2i_bundle_w2_decode_dst_nb0 [[function_constant(FC_FAIRY2I_BUNDLE_W2_DECODE + 2)]];

#define FAIRY2I_DEFINE_BF16_DECODE_KERNEL(NAME, ROWS, BLOCK_SLOTS, NTHREADS, CHECK_TAIL, SUPPORT_BIAS)                \
    kernel void NAME(                                                                                                  \
            constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],                                    \
            device const block_fairy2i_tile64_v2 * u_s0 [[buffer(1)]],                                                \
            device const block_fairy2i_tile64_v2 * u_s1 [[buffer(2)]],                                                \
            device const block_fairy2i_tile64_v2 * w_s0 [[buffer(3)]],                                                \
            device const block_fairy2i_tile64_v2 * w_s1 [[buffer(4)]],                                                \
            device const char * x [[buffer(5)]],                                                                       \
            device const char * bias [[buffer(6)]],                                                                    \
            device char * dst [[buffer(7)]],                                                                           \
            threadgroup float * tmp_real [[threadgroup(0)]],                                                          \
            threadgroup float * tmp_imag [[threadgroup(1)]],                                                          \
            uint2 tgpig [[threadgroup_position_in_grid]],                                                             \
            uint2 tiitg [[thread_position_in_threadgroup]]) {                                                         \
        const uint lid = tiitg.x;                                                                                      \
        const int row_base = (int) tgpig.x * (ROWS);                                                                   \
        const int block_slot = (int) lid >> 4;                                                                         \
        const int lane = (int) lid & 15;                                                                               \
        const int simd_lane = (int) lid & 31;                                                                          \
        const int simd_group = (int) lid >> 5;                                                                         \
        const int blocks = args.k / QK_FAIRY2I_TILE64;                                                                 \
        const int simd_groups = (NTHREADS) / 32;                                                                       \
                                                                                                                        \
        float acc_real[ROWS];                                                                                          \
        float acc_imag[ROWS];                                                                                          \
        for (int tr = 0; tr < (ROWS); ++tr) {                                                                          \
            acc_real[tr] = 0.0f;                                                                                       \
            acc_imag[tr] = 0.0f;                                                                                       \
        }                                                                                                              \
                                                                                                                       \
        for (int wb_base = 0; wb_base < blocks; wb_base += (BLOCK_SLOTS)) {                                           \
            const int wb = wb_base + block_slot;                                                                       \
            float4 act_real = float4(0.0f);                                                                            \
            float4 act_imag = float4(0.0f);                                                                            \
                                                                                                                        \
            if (wb < blocks) {                                                                                         \
                FOR_UNROLL (int part = 0; part < 4; ++part) {                                                          \
                    const int j = lane + 16 * part;                                                                    \
                    const int k_idx = wb * QK_FAIRY2I_TILE64 + j;                                                     \
                    const uint pair = *((device const uint *) (x + (ulong) k_idx * args.x_nb0));                      \
                    act_real[part] = fairy2i_bf16_to_f32((ushort) (pair & 0xffffU));                                 \
                    act_imag[part] = fairy2i_bf16_to_f32((ushort) (pair >> 16));                                      \
                }                                                                                                      \
            }                                                                                                          \
                                                                                                                        \
            if (wb < blocks) {                                                                                         \
                for (int tr = 0; tr < (ROWS); ++tr) {                                                                  \
                    const int row = row_base + tr;                                                                     \
                    if ((CHECK_TAIL) && row >= args.m) {                                                               \
                        continue;                                                                                      \
                    }                                                                                                  \
                                                                                                                       \
                    const int w_index = row * blocks + wb;                                                            \
                    device const block_fairy2i_tile64_v2 & u0 = u_s0[w_index];                                       \
                    device const block_fairy2i_tile64_v2 & u1 = u_s1[w_index];                                       \
                    device const block_fairy2i_tile64_v2 & v0 = w_s0[w_index];                                       \
                    device const block_fairy2i_tile64_v2 & v1 = w_s1[w_index];                                       \
                    const uint4 packed = uint4((uint) u0.qs[lane], (uint) u1.qs[lane], (uint) v0.qs[lane], (uint) v1.qs[lane]); \
                    const float4 wr = float4((float) u0.d_real, (float) u1.d_real, (float) v0.d_real, (float) v1.d_real); \
                    const float4 wi = float4((float) u0.d_imag, (float) u1.d_imag, (float) v0.d_imag, (float) v1.d_imag); \
                    fairy2i_accumulate_tile_weight4_decode_f32_reg(                                                   \
                        packed, wr, wi, act_real, act_imag, tr, acc_real, acc_imag);                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        for (int out_idx = 0; out_idx < (ROWS); ++out_idx) {                                                          \
            const float real_sum = simd_sum(acc_real[out_idx]);                                                       \
            const float imag_sum = simd_sum(acc_imag[out_idx]);                                                       \
            if (simd_lane == 0) {                                                                                      \
                tmp_real[out_idx * simd_groups + simd_group] = real_sum;                                              \
                tmp_imag[out_idx * simd_groups + simd_group] = imag_sum;                                              \
            }                                                                                                          \
        }                                                                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                              \
                                                                                                                       \
        if (lid < (uint) (ROWS)) {                                                                                     \
            const int tr = (int) lid;                                                                                  \
            const int row = row_base + tr;                                                                             \
            if (!(CHECK_TAIL) || row < args.m) {                                                                       \
                float real = 0.0f;                                                                                     \
                float imag = 0.0f;                                                                                     \
                for (int sg = 0; sg < simd_groups; ++sg) {                                                            \
                    real += tmp_real[tr * simd_groups + sg];                                                          \
                    imag += tmp_imag[tr * simd_groups + sg];                                                          \
                }                                                                                                      \
                if ((SUPPORT_BIAS) && args.has_bias) {                                                                 \
                    real += *((device const float *) (bias + (ulong) (row % args.bias_ne0) * args.bias_nb0));         \
                    imag += *((device const float *) (bias + (ulong) ((row + args.m) % args.bias_ne0) * args.bias_nb0)); \
                }                                                                                                      \
                                                                                                                       \
                *((device uint *) (dst + (ulong) row * args.dst_nb0)) = fairy2i_pack_bf16_pair(real, imag);           \
            }                                                                                                          \
        }                                                                                                              \
    }

FAIRY2I_DEFINE_BF16_DECODE_KERNEL(kernel_fairy2i_wide_linear_w2_bf16_tile4x1_w8_simd, 4, 8, 128, true, true)
FAIRY2I_DEFINE_BF16_DECODE_KERNEL(kernel_fairy2i_wide_linear_w2_bf16_tile4x1_w8_full_simd, 4, 8, 128, false, true)

#undef FAIRY2I_DEFINE_BF16_DECODE_KERNEL

#define FAIRY2I_DEFINE_BF16_DECODE_FC_NOBIAS_KERNEL(NAME, ROWS, BLOCK_SLOTS, NTHREADS)                                \
    kernel void NAME(                                                                                                  \
            constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],                                    \
            device const block_fairy2i_tile64_v2 * u_s0 [[buffer(1)]],                                                \
            device const block_fairy2i_tile64_v2 * u_s1 [[buffer(2)]],                                                \
            device const block_fairy2i_tile64_v2 * w_s0 [[buffer(3)]],                                                \
            device const block_fairy2i_tile64_v2 * w_s1 [[buffer(4)]],                                                \
            device const char * x [[buffer(5)]],                                                                       \
            device char * dst [[buffer(6)]],                                                                           \
            threadgroup float * tmp_real [[threadgroup(0)]],                                                          \
            threadgroup float * tmp_imag [[threadgroup(1)]],                                                          \
            uint2 tgpig [[threadgroup_position_in_grid]],                                                             \
            uint2 tiitg [[thread_position_in_threadgroup]]) {                                                         \
        const uint lid = tiitg.x;                                                                                      \
        const int row_base = (int) tgpig.x * (ROWS);                                                                   \
        const int block_slot = (int) lid >> 4;                                                                         \
        const int lane = (int) lid & 15;                                                                               \
        const int simd_lane = (int) lid & 31;                                                                          \
        const int simd_group = (int) lid >> 5;                                                                         \
        const int blocks = FC_fairy2i_w2_decode_blocks;                                                               \
        const int simd_groups = (NTHREADS) / 32;                                                                       \
                                                                                                                       \
        float acc_real[ROWS];                                                                                          \
        float acc_imag[ROWS];                                                                                          \
        for (int tr = 0; tr < (ROWS); ++tr) {                                                                          \
            acc_real[tr] = 0.0f;                                                                                       \
            acc_imag[tr] = 0.0f;                                                                                       \
        }                                                                                                              \
                                                                                                                       \
        for (int wb_base = 0; wb_base < blocks; wb_base += (BLOCK_SLOTS)) {                                           \
            const int wb = wb_base + block_slot;                                                                       \
            float4 act_real = float4(0.0f);                                                                            \
            float4 act_imag = float4(0.0f);                                                                            \
                                                                                                                       \
            if (wb < blocks) {                                                                                         \
                FOR_UNROLL (int part = 0; part < 4; ++part) {                                                          \
                    const int j = lane + 16 * part;                                                                    \
                    const int k_idx = wb * QK_FAIRY2I_TILE64 + j;                                                     \
                    const uint pair = *((device const uint *) (x + (ulong) k_idx * FC_fairy2i_w2_decode_x_nb0));      \
                    act_real[part] = fairy2i_bf16_to_f32((ushort) (pair & 0xffffU));                                  \
                    act_imag[part] = fairy2i_bf16_to_f32((ushort) (pair >> 16));                                      \
                }                                                                                                      \
                                                                                                                       \
                for (int tr = 0; tr < (ROWS); ++tr) {                                                                  \
                    const int row = row_base + tr;                                                                     \
                    const int w_index = row * blocks + wb;                                                            \
                    device const block_fairy2i_tile64_v2 & u0 = u_s0[w_index];                                       \
                    device const block_fairy2i_tile64_v2 & u1 = u_s1[w_index];                                       \
                    device const block_fairy2i_tile64_v2 & v0 = w_s0[w_index];                                       \
                    device const block_fairy2i_tile64_v2 & v1 = w_s1[w_index];                                       \
                    const uint4 packed = uint4((uint) u0.qs[lane], (uint) u1.qs[lane], (uint) v0.qs[lane], (uint) v1.qs[lane]); \
                    const float4 wr = float4((float) u0.d_real, (float) u1.d_real, (float) v0.d_real, (float) v1.d_real); \
                    const float4 wi = float4((float) u0.d_imag, (float) u1.d_imag, (float) v0.d_imag, (float) v1.d_imag); \
                    fairy2i_accumulate_tile_weight4_decode_f32_reg(                                                   \
                        packed, wr, wi, act_real, act_imag, tr, acc_real, acc_imag);                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        for (int out_idx = 0; out_idx < (ROWS); ++out_idx) {                                                          \
            const float real_sum = simd_sum(acc_real[out_idx]);                                                       \
            const float imag_sum = simd_sum(acc_imag[out_idx]);                                                       \
            if (simd_lane == 0) {                                                                                      \
                tmp_real[out_idx * simd_groups + simd_group] = real_sum;                                              \
                tmp_imag[out_idx * simd_groups + simd_group] = imag_sum;                                              \
            }                                                                                                          \
        }                                                                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                              \
                                                                                                                       \
        if (lid < (uint) (ROWS)) {                                                                                     \
            const int row = row_base + (int) lid;                                                                      \
            float real = 0.0f;                                                                                         \
            float imag = 0.0f;                                                                                         \
            for (int sg = 0; sg < simd_groups; ++sg) {                                                                \
                real += tmp_real[(int) lid * simd_groups + sg];                                                       \
                imag += tmp_imag[(int) lid * simd_groups + sg];                                                       \
            }                                                                                                          \
            *((device uint *) (dst + (ulong) row * FC_fairy2i_w2_decode_dst_nb0)) = fairy2i_pack_bf16_pair(real, imag); \
        }                                                                                                              \
    }

FAIRY2I_DEFINE_BF16_DECODE_FC_NOBIAS_KERNEL(
    kernel_fairy2i_wide_linear_w2_bf16_tile4x1_w8_full_nobias_fc_simd, 4, 8, 128)

#undef FAIRY2I_DEFINE_BF16_DECODE_FC_NOBIAS_KERNEL

#define FAIRY2I_DEFINE_W1_BF16_DECODE_KERNEL(NAME, ROWS, BLOCK_SLOTS, NTHREADS, CHECK_TAIL, SUPPORT_BIAS)             \
    kernel void NAME(                                                                                                  \
            constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],                                    \
            device const block_fairy2i_tile64_v2 * u_s0 [[buffer(1)]],                                                \
            device const block_fairy2i_tile64_v2 * w_s0 [[buffer(2)]],                                                \
            device const char * x [[buffer(3)]],                                                                       \
            device const char * bias [[buffer(4)]],                                                                    \
            device char * dst [[buffer(5)]],                                                                           \
            threadgroup float * tmp_real [[threadgroup(0)]],                                                          \
            threadgroup float * tmp_imag [[threadgroup(1)]],                                                          \
            uint2 tgpig [[threadgroup_position_in_grid]],                                                             \
            uint2 tiitg [[thread_position_in_threadgroup]]) {                                                         \
        const uint lid = tiitg.x;                                                                                      \
        const int row_base = (int) tgpig.x * (ROWS);                                                                   \
        const int block_slot = (int) lid >> 4;                                                                         \
        const int lane = (int) lid & 15;                                                                               \
        const int simd_lane = (int) lid & 31;                                                                          \
        const int simd_group = (int) lid >> 5;                                                                         \
        const int blocks = args.k / QK_FAIRY2I_TILE64;                                                                 \
        const int simd_groups = (NTHREADS) / 32;                                                                       \
                                                                                                                       \
        float acc_real[ROWS];                                                                                          \
        float acc_imag[ROWS];                                                                                          \
        for (int tr = 0; tr < (ROWS); ++tr) {                                                                          \
            acc_real[tr] = 0.0f;                                                                                       \
            acc_imag[tr] = 0.0f;                                                                                       \
        }                                                                                                              \
                                                                                                                       \
        for (int wb_base = 0; wb_base < blocks; wb_base += (BLOCK_SLOTS)) {                                           \
            const int wb = wb_base + block_slot;                                                                       \
            float4 act_real = float4(0.0f);                                                                            \
            float4 act_imag = float4(0.0f);                                                                            \
                                                                                                                       \
            if (wb < blocks) {                                                                                         \
                FOR_UNROLL (int part = 0; part < 4; ++part) {                                                          \
                    const int j = lane + 16 * part;                                                                    \
                    const int k_idx = wb * QK_FAIRY2I_TILE64 + j;                                                     \
                    const uint pair = *((device const uint *) (x + (ulong) k_idx * args.x_nb0));                      \
                    act_real[part] = fairy2i_bf16_to_f32((ushort) (pair & 0xffffU));                                  \
                    act_imag[part] = fairy2i_bf16_to_f32((ushort) (pair >> 16));                                      \
                }                                                                                                      \
            }                                                                                                          \
                                                                                                                       \
            if (wb < blocks) {                                                                                         \
                for (int tr = 0; tr < (ROWS); ++tr) {                                                                  \
                    const int row = row_base + tr;                                                                     \
                    if ((CHECK_TAIL) && row >= args.m) {                                                               \
                        continue;                                                                                      \
                    }                                                                                                  \
                                                                                                                       \
                    const int w_index = row * blocks + wb;                                                            \
                    device const block_fairy2i_tile64_v2 & u0 = u_s0[w_index];                                        \
                    device const block_fairy2i_tile64_v2 & v0 = w_s0[w_index];                                        \
                    const uint2 packed = uint2((uint) u0.qs[lane], (uint) v0.qs[lane]);                               \
                    const float2 wr = float2((float) u0.d_real, (float) v0.d_real);                                   \
                    const float2 wi = float2((float) u0.d_imag, (float) v0.d_imag);                                   \
                    fairy2i_accumulate_tile_weight2_decode_f32_reg(                                                   \
                        packed, wr, wi, act_real, act_imag, tr, acc_real, acc_imag);                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        for (int out_idx = 0; out_idx < (ROWS); ++out_idx) {                                                          \
            const float real_sum = simd_sum(acc_real[out_idx]);                                                       \
            const float imag_sum = simd_sum(acc_imag[out_idx]);                                                       \
            if (simd_lane == 0) {                                                                                      \
                tmp_real[out_idx * simd_groups + simd_group] = real_sum;                                              \
                tmp_imag[out_idx * simd_groups + simd_group] = imag_sum;                                              \
            }                                                                                                          \
        }                                                                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                              \
                                                                                                                       \
        if (lid < (uint) (ROWS)) {                                                                                     \
            const int tr = (int) lid;                                                                                  \
            const int row = row_base + tr;                                                                             \
            if (!(CHECK_TAIL) || row < args.m) {                                                                       \
                float real = 0.0f;                                                                                     \
                float imag = 0.0f;                                                                                     \
                for (int sg = 0; sg < simd_groups; ++sg) {                                                            \
                    real += tmp_real[tr * simd_groups + sg];                                                          \
                    imag += tmp_imag[tr * simd_groups + sg];                                                          \
                }                                                                                                      \
                if ((SUPPORT_BIAS) && args.has_bias) {                                                                 \
                    real += *((device const float *) (bias + (ulong) (row % args.bias_ne0) * args.bias_nb0));         \
                    imag += *((device const float *) (bias + (ulong) ((row + args.m) % args.bias_ne0) * args.bias_nb0)); \
                }                                                                                                      \
                                                                                                                       \
                *((device uint *) (dst + (ulong) row * args.dst_nb0)) = fairy2i_pack_bf16_pair(real, imag);           \
            }                                                                                                          \
        }                                                                                                              \
    }

FAIRY2I_DEFINE_W1_BF16_DECODE_KERNEL(kernel_fairy2i_wide_linear_w1_bf16_tile8x1_w16_simd, 8, 16, 256, true, true)
FAIRY2I_DEFINE_W1_BF16_DECODE_KERNEL(kernel_fairy2i_wide_linear_w1_bf16_tile8x1_w16_full_simd, 8, 16, 256, false, true)
FAIRY2I_DEFINE_W1_BF16_DECODE_KERNEL(kernel_fairy2i_wide_linear_w1_bf16_tile8x1_w16_full_nobias_simd, 8, 16, 256, false, false)

#undef FAIRY2I_DEFINE_W1_BF16_DECODE_KERNEL

#define FAIRY2I_DEFINE_W1_BF16_DECODE_FC_NOBIAS_KERNEL(NAME, ROWS, BLOCK_SLOTS, NTHREADS)                            \
    kernel void NAME(                                                                                                  \
            constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],                                    \
            device const block_fairy2i_tile64_v2 * u_s0 [[buffer(1)]],                                                \
            device const block_fairy2i_tile64_v2 * w_s0 [[buffer(2)]],                                                \
            device const char * x [[buffer(3)]],                                                                       \
            device char * dst [[buffer(4)]],                                                                           \
            threadgroup float * tmp_real [[threadgroup(0)]],                                                          \
            threadgroup float * tmp_imag [[threadgroup(1)]],                                                          \
            uint2 tgpig [[threadgroup_position_in_grid]],                                                             \
            uint2 tiitg [[thread_position_in_threadgroup]]) {                                                         \
        const uint lid = tiitg.x;                                                                                      \
        const int row_base = (int) tgpig.x * (ROWS);                                                                   \
        const int block_slot = (int) lid >> 4;                                                                         \
        const int lane = (int) lid & 15;                                                                               \
        const int simd_lane = (int) lid & 31;                                                                          \
        const int simd_group = (int) lid >> 5;                                                                         \
        const int blocks = FC_fairy2i_w1_decode_blocks;                                                               \
        const int simd_groups = (NTHREADS) / 32;                                                                       \
                                                                                                                       \
        float acc_real[ROWS];                                                                                          \
        float acc_imag[ROWS];                                                                                          \
        for (int tr = 0; tr < (ROWS); ++tr) {                                                                          \
            acc_real[tr] = 0.0f;                                                                                       \
            acc_imag[tr] = 0.0f;                                                                                       \
        }                                                                                                              \
                                                                                                                       \
        for (int wb_base = 0; wb_base < blocks; wb_base += (BLOCK_SLOTS)) {                                           \
            const int wb = wb_base + block_slot;                                                                       \
            float4 act_real = float4(0.0f);                                                                            \
            float4 act_imag = float4(0.0f);                                                                            \
                                                                                                                       \
            if (wb < blocks) {                                                                                         \
                FOR_UNROLL (int part = 0; part < 4; ++part) {                                                          \
                    const int j = lane + 16 * part;                                                                    \
                    const int k_idx = wb * QK_FAIRY2I_TILE64 + j;                                                     \
                    const uint pair = *((device const uint *) (x + (ulong) k_idx * FC_fairy2i_w1_decode_x_nb0));      \
                    act_real[part] = fairy2i_bf16_to_f32((ushort) (pair & 0xffffU));                                  \
                    act_imag[part] = fairy2i_bf16_to_f32((ushort) (pair >> 16));                                      \
                }                                                                                                      \
                                                                                                                       \
                for (int tr = 0; tr < (ROWS); ++tr) {                                                                  \
                    const int row = row_base + tr;                                                                     \
                    const int w_index = row * blocks + wb;                                                            \
                    device const block_fairy2i_tile64_v2 & u0 = u_s0[w_index];                                       \
                    device const block_fairy2i_tile64_v2 & v0 = w_s0[w_index];                                       \
                    const uint2 packed = uint2((uint) u0.qs[lane], (uint) v0.qs[lane]);                              \
                    const float2 wr = float2((float) u0.d_real, (float) v0.d_real);                                  \
                    const float2 wi = float2((float) u0.d_imag, (float) v0.d_imag);                                  \
                    fairy2i_accumulate_tile_weight2_decode_f32_reg(                                                   \
                        packed, wr, wi, act_real, act_imag, tr, acc_real, acc_imag);                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        for (int out_idx = 0; out_idx < (ROWS); ++out_idx) {                                                          \
            const float real_sum = simd_sum(acc_real[out_idx]);                                                       \
            const float imag_sum = simd_sum(acc_imag[out_idx]);                                                       \
            if (simd_lane == 0) {                                                                                      \
                tmp_real[out_idx * simd_groups + simd_group] = real_sum;                                              \
                tmp_imag[out_idx * simd_groups + simd_group] = imag_sum;                                              \
            }                                                                                                          \
        }                                                                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                              \
                                                                                                                       \
        if (lid < (uint) (ROWS)) {                                                                                     \
            const int row = row_base + (int) lid;                                                                      \
            float real = 0.0f;                                                                                         \
            float imag = 0.0f;                                                                                         \
            for (int sg = 0; sg < simd_groups; ++sg) {                                                                \
                real += tmp_real[(int) lid * simd_groups + sg];                                                       \
                imag += tmp_imag[(int) lid * simd_groups + sg];                                                       \
            }                                                                                                          \
            *((device uint *) (dst + (ulong) row * FC_fairy2i_w1_decode_dst_nb0)) = fairy2i_pack_bf16_pair(real, imag); \
        }                                                                                                              \
    }

FAIRY2I_DEFINE_W1_BF16_DECODE_FC_NOBIAS_KERNEL(
    kernel_fairy2i_wide_linear_w1_bf16_tile8x1_w16_full_nobias_fc_simd, 8, 16, 256)

#undef FAIRY2I_DEFINE_W1_BF16_DECODE_FC_NOBIAS_KERNEL

kernel void kernel_fairy2i_bundle_w1_bf16_tile8x1_w16_full_simd(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const uchar * codes                              [[buffer(1)]],
        device const half * scales                              [[buffer(2)]],
        device const char * x                                   [[buffer(3)]],
        device const char * bias                                [[buffer(4)]],
        device char * dst                                       [[buffer(5)]],
        threadgroup float * tmp_real                            [[threadgroup(0)]],
        threadgroup float * tmp_imag                            [[threadgroup(1)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint2 tiitg                                             [[thread_position_in_threadgroup]]) {
    constexpr int rows = 8;
    constexpr int block_slots = 16;
    constexpr int n_threads = 256;

    const uint lid = tiitg.x;
    const int row_base = (int) tgpig.x * rows;
    const int block_slot = (int) lid >> 4;
    const int q4 = (int) lid & 15;
    const int simd_lane = (int) lid & 31;
    const int simd_group = (int) lid >> 5;
    const int blocks = args.k / QK_FAIRY2I_TILE64;
    const int simd_groups = n_threads / 32;
    const int m16 = (row_base & 63) >> 4;
    const int row_lane_base = row_base & 15;
    float acc_real[rows];
    float acc_imag[rows];
    for (int tr = 0; tr < rows; ++tr) {
        acc_real[tr] = 0.0f;
        acc_imag[tr] = 0.0f;
    }

    for (int wb_base = 0; wb_base < blocks; wb_base += block_slots) {
        const int wb = wb_base + block_slot;
        float4 act_real = float4(0.0f);
        float4 act_imag = float4(0.0f);

        if (wb < blocks) {
            FOR_UNROLL (int part = 0; part < 4; ++part) {
                // A bundle byte describes four consecutive K values, unlike the V2 residue-lane byte order.
                const int j = q4 * 4 + part;
                const int k_idx = wb * QK_FAIRY2I_TILE64 + j;
                const uint pair = *((device const uint *) (x + (ulong) k_idx * args.x_nb0));
                act_real[part] = fairy2i_bf16_to_f32((ushort) (pair & 0xffffU));
                act_imag[part] = fairy2i_bf16_to_f32((ushort) (pair >> 16));
            }
        }

        float4 branch_scales = float4(0.0f);
        if (wb < blocks) {
            const int physical_tile = (row_base / QK_FAIRY2I_TILE64) * blocks + wb;
            const int scale_base = physical_tile * 4;
            branch_scales = float4(
                (float) scales[scale_base + 0],
                (float) scales[scale_base + 1],
                (float) scales[scale_base + 2],
                (float) scales[scale_base + 3]);
        }

        if (wb < blocks) {
            const int physical_tile = (row_base / QK_FAIRY2I_TILE64) * blocks + wb;
            const int slot = m16 * 16 + q4;
            const ulong code_base =
                ((((ulong) physical_tile * 64 + (ulong) slot) * 2) * 16) + (ulong) row_lane_base;
            const ulong u_rows = *((device const ulong *) (codes + code_base));
            const ulong w_rows = *((device const ulong *) (codes + code_base + 16));
            const float2 wr = branch_scales.xz;
            const float2 wi = branch_scales.yw;

            for (int tr = 0; tr < rows; ++tr) {
                const uint2 packed = uint2(
                    (uint) ((u_rows >> (8 * tr)) & 0xffUL),
                    (uint) ((w_rows >> (8 * tr)) & 0xffUL));
                fairy2i_accumulate_tile_weight2_decode_f32_reg(
                    packed, wr, wi, act_real, act_imag, tr, acc_real, acc_imag);
            }
        }
    }

    for (int out_idx = 0; out_idx < rows; ++out_idx) {
        const float real_sum = simd_sum(acc_real[out_idx]);
        const float imag_sum = simd_sum(acc_imag[out_idx]);
        if (simd_lane == 0) {
            tmp_real[out_idx * simd_groups + simd_group] = real_sum;
            tmp_imag[out_idx * simd_groups + simd_group] = imag_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < rows) {
        const int row = row_base + (int) lid;
        float real = 0.0f;
        float imag = 0.0f;
        for (int sg = 0; sg < simd_groups; ++sg) {
            real += tmp_real[(int) lid * simd_groups + sg];
            imag += tmp_imag[(int) lid * simd_groups + sg];
        }
        if (args.has_bias) {
            real += *((device const float *) (bias + (ulong) (row % args.bias_ne0) * args.bias_nb0));
            imag += *((device const float *) (bias + (ulong) ((row + args.m) % args.bias_ne0) * args.bias_nb0));
        }

        *((device uint *) (dst + (ulong) row * args.dst_nb0)) = fairy2i_pack_bf16_pair(real, imag);
    }
}

kernel void kernel_fairy2i_bundle_w1_bf16_tile8x1_w8_full_nobias_fc_simd(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const uchar * codes                              [[buffer(1)]],
        device const half * scales                              [[buffer(2)]],
        device const char * x                                   [[buffer(3)]],
        device char * dst                                       [[buffer(4)]],
        threadgroup float * tmp_real                            [[threadgroup(0)]],
        threadgroup float * tmp_imag                            [[threadgroup(1)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint2 tiitg                                             [[thread_position_in_threadgroup]]) {
    constexpr int rows = 8;
    constexpr int block_slots = 8;
    constexpr int n_threads = 128;

    const uint lid = tiitg.x;
    const int row_base = (int) tgpig.x * rows;
    const int block_slot = (int) lid >> 4;
    const int q4 = (int) lid & 15;
    const int simd_lane = (int) lid & 31;
    const int simd_group = (int) lid >> 5;
    const int blocks = FC_fairy2i_bundle_w1_decode_blocks;
    const int simd_groups = n_threads / 32;
    const int m16 = (row_base & 63) >> 4;
    const int row_lane_base = row_base & 15;

    float acc_real[rows];
    float acc_imag[rows];
    for (int tr = 0; tr < rows; ++tr) {
        acc_real[tr] = 0.0f;
        acc_imag[tr] = 0.0f;
    }

    for (int wb_base = 0; wb_base < blocks; wb_base += block_slots) {
        const int wb = wb_base + block_slot;
        if (wb < blocks) {
            float4 act_real;
            float4 act_imag;
            FOR_UNROLL (int part = 0; part < 4; ++part) {
                const int j = q4 * 4 + part;
                const int k_idx = wb * QK_FAIRY2I_TILE64 + j;
                const uint pair = *((device const uint *) (x + (ulong) k_idx * FC_fairy2i_bundle_w1_decode_x_nb0));
                act_real[part] = fairy2i_bf16_to_f32((ushort) (pair & 0xffffU));
                act_imag[part] = fairy2i_bf16_to_f32((ushort) (pair >> 16));
            }

            const int physical_tile = (row_base / QK_FAIRY2I_TILE64) * blocks + wb;
            const int scale_base = physical_tile * 4;
            const float4 branch_scales = float4(*((device const half4 *) (scales + scale_base)));
            const float2 wr = branch_scales.xz;
            const float2 wi = branch_scales.yw;
            const int slot = m16 * 16 + q4;
            // codes[physical_tile][m16 * 16 + q4][U/W branch][row lane]
            const ulong code_base =
                ((((ulong) physical_tile * 64 + (ulong) slot) * 2) * 16) + (ulong) row_lane_base;
            const uint2 u_rows = *((device const uint2 *) (codes + code_base));
            const uint2 w_rows = *((device const uint2 *) (codes + code_base + 16));

            FOR_UNROLL (int tr = 0; tr < rows; ++tr) {
                const int shift = 8 * (tr & 3);
                const uint u_row4 = tr < 4 ? u_rows.x : u_rows.y;
                const uint w_row4 = tr < 4 ? w_rows.x : w_rows.y;
                const uint2 packed = uint2(
                    (u_row4 >> shift) & 0xffU,
                    (w_row4 >> shift) & 0xffU);
                fairy2i_accumulate_tile_weight2_decode_f32_reg(
                    packed, wr, wi, act_real, act_imag, tr, acc_real, acc_imag);
            }
        }
    }

    for (int out_idx = 0; out_idx < rows; ++out_idx) {
        const float real_sum = simd_sum(acc_real[out_idx]);
        const float imag_sum = simd_sum(acc_imag[out_idx]);
        if (simd_lane == 0) {
            tmp_real[out_idx * simd_groups + simd_group] = real_sum;
            tmp_imag[out_idx * simd_groups + simd_group] = imag_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < rows) {
        const int row = row_base + (int) lid;
        float real = 0.0f;
        float imag = 0.0f;
        for (int sg = 0; sg < simd_groups; ++sg) {
            real += tmp_real[(int) lid * simd_groups + sg];
            imag += tmp_imag[(int) lid * simd_groups + sg];
        }
        *((device uint *) (dst + (ulong) row * FC_fairy2i_bundle_w1_decode_dst_nb0)) =
            fairy2i_pack_bf16_pair(real, imag);
    }
}

#if defined(GGML_METAL_HAS_BF16)
template <bool check_exact_product>
static inline void fairy2i_bundle_w1_bf16_bf16scale_exact_tile8x1_w8(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args,
        int blocks,
        ulong x_nb0,
        ulong dst_nb0,
        bool has_bias,
        device const uchar * codes,
        device const ushort * scales,
        device const char * x,
        device const char * bias,
        device char * dst,
        device const uint * act_block_metrics,
        device const ushort4 * packed_coeff_lut,
        threadgroup float * tmp_real,
        threadgroup float * tmp_imag,
        threadgroup ushort4 * coeff_lut,
        threadgroup uchar * coeff_metric_lut,
        uint2 tgpig,
        uint2 tiitg) {
    constexpr int rows = 8;
    constexpr int block_slots = 8;
    constexpr int n_threads = 128;

    const uint lid = tiitg.x;
    const int row_base = (int) tgpig.x * rows;
    const int block_slot = (int) lid >> 4;
    const int pattern = (int) lid & 15;
    const int q4 = pattern;
    const int simd_lane = (int) lid & 31;
    const int simd_group = (int) lid >> 5;
    const int simd_groups = n_threads / 32;
    const int m16 = (row_base & 63) >> 4;
    const int row_lane_base = row_base & 15;
    const int act_row = (int) tgpig.y;
    const int i1 = act_row % args.x_ne1;
    const int i2 = (act_row / args.x_ne1) % args.x_ne2;
    const int i3 = act_row / (args.x_ne1 * args.x_ne2);
    const ulong x_row_offset =
        (ulong) i1 * args.x_nb1 + (ulong) i2 * args.x_nb2 + (ulong) i3 * args.x_nb3;
    const ulong dst_row_offset =
        (ulong) i1 * args.dst_nb1 + (ulong) i2 * args.dst_nb2 + (ulong) i3 * args.dst_nb3;
    device const uchar * packed_coeff_metrics =
        (device const uchar *) (packed_coeff_lut +
                                (ulong) (args.m / QK_FAIRY2I_TILE64) * (ulong) blocks * 16UL);

    float acc_real[rows];
    float acc_imag[rows];
    uint min_coeff_metric = 255U;
    uint min_act_metric = 255U;
    for (int tr = 0; tr < rows; ++tr) {
        acc_real[tr] = 0.0f;
        acc_imag[tr] = 0.0f;
    }

    for (int wb_base = 0; wb_base < blocks; wb_base += block_slots) {
        const int wb = wb_base + block_slot;
        if (wb < blocks) {
            if (check_exact_product && pattern == 0) {
                min_act_metric = min(min_act_metric, act_block_metrics[wb]);
            }
            const int physical_tile = (row_base / QK_FAIRY2I_TILE64) * blocks + wb;
            const ushort4 coeff = packed_coeff_lut[(ulong) physical_tile * 16UL + (ulong) pattern];
            coeff_lut[lid] = coeff;
            if (check_exact_product) {
                coeff_metric_lut[lid] = packed_coeff_metrics[(ulong) physical_tile * 16UL + (ulong) pattern];
            }
        } else {
            coeff_lut[lid] = ushort4(0);
            if (check_exact_product) {
                coeff_metric_lut[lid] = (uchar) 255U;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (wb < blocks) {
            ushort4 act_real_bits;
            ushort4 act_imag_bits;
            float4 act_real;
            float4 act_imag;
            FOR_UNROLL (int part = 0; part < 4; ++part) {
                const int j = q4 * 4 + part;
                const int k_idx = wb * QK_FAIRY2I_TILE64 + j;
                const uint pair = *((device const uint *) (x + x_row_offset + (ulong) k_idx * x_nb0));
                act_real_bits[part] = (ushort) (pair & 0xffffU);
                act_imag_bits[part] = (ushort) (pair >> 16);
                act_real[part] = fairy2i_bf16_to_f32(act_real_bits[part]);
                act_imag[part] = fairy2i_bf16_to_f32(act_imag_bits[part]);
            }

            const int physical_tile = (row_base / QK_FAIRY2I_TILE64) * blocks + wb;
            const int slot = m16 * 16 + q4;
            const ulong code_base =
                ((((ulong) physical_tile * 64 + (ulong) slot) * 2) * 16) + (ulong) row_lane_base;
            const ulong u_rows = *((device const ulong *) (codes + code_base));
            const ulong w_rows = *((device const ulong *) (codes + code_base + 16));
            threadgroup const ushort4 * block_lut = coeff_lut + block_slot * 16;

            FOR_UNROLL (int tr = 0; tr < rows; ++tr) {
                const uint u_packed = (uint) ((u_rows >> (8 * tr)) & 0xffUL);
                const uint w_packed = (uint) ((w_rows >> (8 * tr)) & 0xffUL);
                FOR_UNROLL (int part = 0; part < 4; ++part) {
                    const uint u_code = (u_packed >> (2 * part)) & 3U;
                    const uint w_code = (w_packed >> (2 * part)) & 3U;
                    const uint coeff_index = u_code | (w_code << 2);
                    const ushort4 coeff_bits = block_lut[coeff_index];
                    if (check_exact_product) {
                        min_coeff_metric = min(
                            min_coeff_metric,
                            (uint) coeff_metric_lut[block_slot * 16 + coeff_index]);
                    }
                    const float4 coeff = float4(
                        fairy2i_bf16_to_f32(coeff_bits.x),
                        fairy2i_bf16_to_f32(coeff_bits.y),
                        fairy2i_bf16_to_f32(coeff_bits.z),
                        fairy2i_bf16_to_f32(coeff_bits.w));
                    acc_real[tr] += coeff.x * act_real[part] + coeff.y * act_imag[part];
                    acc_imag[tr] += coeff.z * act_real[part] + coeff.w * act_imag[part];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    bool requires_software = false;
    if (check_exact_product) {
        const uint simd_coeff_metric = simd_min(min_coeff_metric);
        const uint simd_act_metric = simd_min(min_act_metric);
        if (simd_lane == 0) {
            ((threadgroup uint *) tmp_real)[simd_group] = simd_coeff_metric;
            ((threadgroup uint *) tmp_imag)[simd_group] = simd_act_metric;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            uint tile_coeff_metric = 255U;
            uint tile_act_metric = 255U;
            for (int sg = 0; sg < simd_groups; ++sg) {
                tile_coeff_metric = min(tile_coeff_metric, ((threadgroup uint *) tmp_real)[sg]);
                tile_act_metric = min(tile_act_metric, ((threadgroup uint *) tmp_imag)[sg]);
            }
            ((threadgroup uint *) tmp_real)[0] = tile_coeff_metric;
            ((threadgroup uint *) tmp_imag)[0] = tile_act_metric;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        requires_software = fairy2i_product_metrics_require_software(
            ((threadgroup uint *) tmp_real)[0], ((threadgroup uint *) tmp_imag)[0]);
    }
    if (requires_software) {
        if (lid < rows) {
            const int row = row_base + (int) lid;
            uint real_bits = 0;
            uint imag_bits = 0;
            for (int k = 0; k < blocks * QK_FAIRY2I_TILE64; ++k) {
                const uint pair = *((device const uint *) (x + x_row_offset + (ulong) k * x_nb0));
                const ushort xr_bits = (ushort) (pair & 0xffffU);
                const ushort xi_bits = (ushort) (pair >> 16);
                const ushort4 coeff_bits =
                    fairy2i_bundle_w1_coeff_at_bf16_exact_bits(codes, scales, blocks, row, k);
                real_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(real_bits, coeff_bits.x, xr_bits);
                real_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(real_bits, coeff_bits.y, xi_bits);
                imag_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(imag_bits, coeff_bits.z, xr_bits);
                imag_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(imag_bits, coeff_bits.w, xi_bits);
            }

            ushort out_real_bits;
            ushort out_imag_bits;
            if (has_bias) {
                const int b1 = i1 % args.bias_ne1;
                const int b2 = i2 % args.bias_ne2;
                const int b3 = i3 % args.bias_ne3;
                const float bias_real = *((device const float *)
                    (bias + (ulong) (row % args.bias_ne0) * args.bias_nb0 + (ulong) b1 * args.bias_nb1 +
                     (ulong) b2 * args.bias_nb2 + (ulong) b3 * args.bias_nb3));
                const float bias_imag = *((device const float *)
                    (bias + (ulong) ((row + args.m) % args.bias_ne0) * args.bias_nb0 +
                     (ulong) b1 * args.bias_nb1 + (ulong) b2 * args.bias_nb2 + (ulong) b3 * args.bias_nb3));
                out_real_bits = fairy2i_add_f32_bits_f32_bias_to_bf16_bits_rne(real_bits, bias_real);
                out_imag_bits = fairy2i_add_f32_bits_f32_bias_to_bf16_bits_rne(imag_bits, bias_imag);
            } else {
                out_real_bits = fairy2i_f32_to_bf16(as_type<float>(real_bits));
                out_imag_bits = fairy2i_f32_to_bf16(as_type<float>(imag_bits));
            }
            *((device uint *) (dst + dst_row_offset + (ulong) row * dst_nb0)) =
                (uint) out_real_bits | ((uint) out_imag_bits << 16);
        }
        return;
    }

    for (int out_idx = 0; out_idx < rows; ++out_idx) {
        const float real_sum = simd_sum(acc_real[out_idx]);
        const float imag_sum = simd_sum(acc_imag[out_idx]);
        if (simd_lane == 0) {
            tmp_real[out_idx * simd_groups + simd_group] = real_sum;
            tmp_imag[out_idx * simd_groups + simd_group] = imag_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < rows) {
        const int row = row_base + (int) lid;
        float real = 0.0f;
        float imag = 0.0f;
        for (int sg = 0; sg < simd_groups; ++sg) {
            real += tmp_real[(int) lid * simd_groups + sg];
            imag += tmp_imag[(int) lid * simd_groups + sg];
        }

        ushort real_bits;
        ushort imag_bits;
        if (has_bias) {
            const int b1 = i1 % args.bias_ne1;
            const int b2 = i2 % args.bias_ne2;
            const int b3 = i3 % args.bias_ne3;
            const float bias_real = *((device const float *)
                (bias + (ulong) (row % args.bias_ne0) * args.bias_nb0 + (ulong) b1 * args.bias_nb1 +
                 (ulong) b2 * args.bias_nb2 + (ulong) b3 * args.bias_nb3));
            const float bias_imag = *((device const float *)
                (bias + (ulong) ((row + args.m) % args.bias_ne0) * args.bias_nb0 +
                 (ulong) b1 * args.bias_nb1 + (ulong) b2 * args.bias_nb2 + (ulong) b3 * args.bias_nb3));
            real_bits = fairy2i_add_f32_bias_to_bf16_bits_rne(real, bias_real);
            imag_bits = fairy2i_add_f32_bias_to_bf16_bits_rne(imag, bias_imag);
        } else {
            real_bits = fairy2i_f32_to_bf16(real);
            imag_bits = fairy2i_f32_to_bf16(imag);
        }
        *((device uint *) (dst + dst_row_offset + (ulong) row * dst_nb0)) =
            (uint) real_bits | ((uint) imag_bits << 16);
    }
}

kernel void kernel_fairy2i_bundle_w1_bf16_bf16scale_exact_tile8x1_w8_simd(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const uchar * codes                              [[buffer(1)]],
        device const ushort * scales                            [[buffer(2)]],
        device const char * x                                   [[buffer(3)]],
        device const char * bias                                [[buffer(4)]],
        device char * dst                                       [[buffer(5)]],
        device const uint * act_block_metrics                   [[buffer(6)]],
        device const ushort4 * packed_coeff_lut                  [[buffer(7)]],
        threadgroup float * tmp_real                            [[threadgroup(0)]],
        threadgroup float * tmp_imag                            [[threadgroup(1)]],
        threadgroup ushort4 * coeff_lut                         [[threadgroup(2)]],
        threadgroup uchar * coeff_metric_lut                    [[threadgroup(3)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint2 tiitg                                             [[thread_position_in_threadgroup]]) {
    fairy2i_bundle_w1_bf16_bf16scale_exact_tile8x1_w8<true>(
        args,
        args.k / QK_FAIRY2I_TILE64,
        args.x_nb0,
        args.dst_nb0,
        args.has_bias,
        codes,
        scales,
        x,
        bias,
        dst,
        act_block_metrics,
        packed_coeff_lut,
        tmp_real,
        tmp_imag,
        coeff_lut,
        coeff_metric_lut,
        tgpig,
        tiitg);
}

kernel void kernel_fairy2i_bundle_w1_bf16_bf16scale_exact_tile8x1_w8_nobias_fc_simd(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const uchar * codes                              [[buffer(1)]],
        device const ushort * scales                            [[buffer(2)]],
        device const char * x                                   [[buffer(3)]],
        device char * dst                                       [[buffer(4)]],
        device const uint * act_block_metrics                   [[buffer(5)]],
        device const ushort4 * packed_coeff_lut                  [[buffer(6)]],
        threadgroup float * tmp_real                            [[threadgroup(0)]],
        threadgroup float * tmp_imag                            [[threadgroup(1)]],
        threadgroup ushort4 * coeff_lut                         [[threadgroup(2)]],
        threadgroup uchar * coeff_metric_lut                    [[threadgroup(3)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint2 tiitg                                             [[thread_position_in_threadgroup]]) {
    fairy2i_bundle_w1_bf16_bf16scale_exact_tile8x1_w8<true>(
        args,
        FC_fairy2i_bundle_w1_decode_blocks,
        (ulong) FC_fairy2i_bundle_w1_decode_x_nb0,
        (ulong) FC_fairy2i_bundle_w1_decode_dst_nb0,
        false,
        codes,
        scales,
        x,
        dst,
        dst,
        act_block_metrics,
        packed_coeff_lut,
        tmp_real,
        tmp_imag,
        coeff_lut,
        coeff_metric_lut,
        tgpig,
        tiitg);
}

kernel void kernel_fairy2i_bundle_w1_bf16_bf16scale_qat_tile8x1_w8_simd(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const uchar * codes                              [[buffer(1)]],
        device const ushort * scales                            [[buffer(2)]],
        device const char * x                                   [[buffer(3)]],
        device const char * bias                                [[buffer(4)]],
        device char * dst                                       [[buffer(5)]],
        device const uint * act_block_metrics                   [[buffer(6)]],
        device const ushort4 * packed_coeff_lut                  [[buffer(7)]],
        threadgroup float * tmp_real                            [[threadgroup(0)]],
        threadgroup float * tmp_imag                            [[threadgroup(1)]],
        threadgroup ushort4 * coeff_lut                         [[threadgroup(2)]],
        threadgroup uchar * coeff_metric_lut                    [[threadgroup(3)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint2 tiitg                                             [[thread_position_in_threadgroup]]) {
    fairy2i_bundle_w1_bf16_bf16scale_exact_tile8x1_w8<false>(
        args,
        args.k / QK_FAIRY2I_TILE64,
        args.x_nb0,
        args.dst_nb0,
        args.has_bias,
        codes,
        scales,
        x,
        bias,
        dst,
        act_block_metrics,
        packed_coeff_lut,
        tmp_real,
        tmp_imag,
        coeff_lut,
        coeff_metric_lut,
        tgpig,
        tiitg);
}

kernel void kernel_fairy2i_bundle_w1_bf16_bf16scale_qat_tile8x1_w8_nobias_fc_simd(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const uchar * codes                              [[buffer(1)]],
        device const ushort * scales                            [[buffer(2)]],
        device const char * x                                   [[buffer(3)]],
        device char * dst                                       [[buffer(4)]],
        device const uint * act_block_metrics                   [[buffer(5)]],
        device const ushort4 * packed_coeff_lut                  [[buffer(6)]],
        threadgroup float * tmp_real                            [[threadgroup(0)]],
        threadgroup float * tmp_imag                            [[threadgroup(1)]],
        threadgroup ushort4 * coeff_lut                         [[threadgroup(2)]],
        threadgroup uchar * coeff_metric_lut                    [[threadgroup(3)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint2 tiitg                                             [[thread_position_in_threadgroup]]) {
    fairy2i_bundle_w1_bf16_bf16scale_exact_tile8x1_w8<false>(
        args,
        FC_fairy2i_bundle_w1_decode_blocks,
        (ulong) FC_fairy2i_bundle_w1_decode_x_nb0,
        (ulong) FC_fairy2i_bundle_w1_decode_dst_nb0,
        false,
        codes,
        scales,
        x,
        dst,
        dst,
        act_block_metrics,
        packed_coeff_lut,
        tmp_real,
        tmp_imag,
        coeff_lut,
        coeff_metric_lut,
        tgpig,
        tiitg);
}
#endif

kernel void kernel_fairy2i_bundle_w2_bf16_tile4x1_w8_full_simd(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const uchar * codes                              [[buffer(1)]],
        device const half * scales                              [[buffer(2)]],
        device const char * x                                   [[buffer(3)]],
        device const char * bias                                [[buffer(4)]],
        device char * dst                                       [[buffer(5)]],
        threadgroup float * tmp_real                            [[threadgroup(0)]],
        threadgroup float * tmp_imag                            [[threadgroup(1)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint2 tiitg                                             [[thread_position_in_threadgroup]]) {
    constexpr int rows = 4;
    constexpr int block_slots = 8;
    constexpr int n_threads = 128;

    const uint lid = tiitg.x;
    const int row_base = (int) tgpig.x * rows;
    const int block_slot = (int) lid >> 4;
    const int q4 = (int) lid & 15;
    const int simd_lane = (int) lid & 31;
    const int simd_group = (int) lid >> 5;
    const int blocks = args.k / QK_FAIRY2I_TILE64;
    const int simd_groups = n_threads / 32;
    const int m16 = (row_base & 63) >> 4;
    const int row_lane_base = row_base & 15;

    float acc_real[rows];
    float acc_imag[rows];
    for (int tr = 0; tr < rows; ++tr) {
        acc_real[tr] = 0.0f;
        acc_imag[tr] = 0.0f;
    }

    for (int wb_base = 0; wb_base < blocks; wb_base += block_slots) {
        const int wb = wb_base + block_slot;
        if (wb < blocks) {
            float4 act_real;
            float4 act_imag;
            FOR_UNROLL (int part = 0; part < 4; ++part) {
                const int j = q4 * 4 + part;
                const int k_idx = wb * QK_FAIRY2I_TILE64 + j;
                const uint pair = *((device const uint *) (x + (ulong) k_idx * args.x_nb0));
                act_real[part] = fairy2i_bf16_to_f32((ushort) (pair & 0xffffU));
                act_imag[part] = fairy2i_bf16_to_f32((ushort) (pair >> 16));
            }

            const int physical_tile = (row_base / QK_FAIRY2I_TILE64) * blocks + wb;
            const int scale_base = physical_tile * 8;
            const half4 scale01 = *((device const half4 *) (scales + scale_base));
            const half4 scale23 = *((device const half4 *) (scales + scale_base + 4));
            const float4 wr = float4(scale01.x, scale01.z, scale23.x, scale23.z);
            const float4 wi = float4(scale01.y, scale01.w, scale23.y, scale23.w);
            const int slot = m16 * 16 + q4;
            // codes[physical_tile][m16 * 16 + q4][U0/U1/W0/W1 branch][row lane]
            const ulong code_base =
                ((((ulong) physical_tile * 64 + (ulong) slot) * 4) * 16) + (ulong) row_lane_base;
            const uint4 branch_rows = uint4(
                *((device const uint *) (codes + code_base)),
                *((device const uint *) (codes + code_base + 16)),
                *((device const uint *) (codes + code_base + 32)),
                *((device const uint *) (codes + code_base + 48)));

            FOR_UNROLL (int tr = 0; tr < rows; ++tr) {
                const uint4 packed = (branch_rows >> uint4(8 * tr)) & uint4(0xffU);
                fairy2i_accumulate_tile_weight4_decode_f32_reg(
                    packed, wr, wi, act_real, act_imag, tr, acc_real, acc_imag);
            }
        }
    }

    for (int out_idx = 0; out_idx < rows; ++out_idx) {
        const float real_sum = simd_sum(acc_real[out_idx]);
        const float imag_sum = simd_sum(acc_imag[out_idx]);
        if (simd_lane == 0) {
            tmp_real[out_idx * simd_groups + simd_group] = real_sum;
            tmp_imag[out_idx * simd_groups + simd_group] = imag_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < rows) {
        const int row = row_base + (int) lid;
        float real = 0.0f;
        float imag = 0.0f;
        for (int sg = 0; sg < simd_groups; ++sg) {
            real += tmp_real[(int) lid * simd_groups + sg];
            imag += tmp_imag[(int) lid * simd_groups + sg];
        }
        if (args.has_bias) {
            real += *((device const float *) (bias + (ulong) (row % args.bias_ne0) * args.bias_nb0));
            imag += *((device const float *) (bias + (ulong) ((row + args.m) % args.bias_ne0) * args.bias_nb0));
        }

        *((device uint *) (dst + (ulong) row * args.dst_nb0)) = fairy2i_pack_bf16_pair(real, imag);
    }
}

kernel void kernel_fairy2i_bundle_w2_bf16_tile4x1_w8_full_nobias_fc_simd(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const uchar * codes                              [[buffer(1)]],
        device const half * scales                              [[buffer(2)]],
        device const char * x                                   [[buffer(3)]],
        device char * dst                                       [[buffer(4)]],
        threadgroup float * tmp_real                            [[threadgroup(0)]],
        threadgroup float * tmp_imag                            [[threadgroup(1)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint2 tiitg                                             [[thread_position_in_threadgroup]]) {
    constexpr int rows = 4;
    constexpr int block_slots = 8;
    constexpr int n_threads = 128;

    const uint lid = tiitg.x;
    const int row_base = (int) tgpig.x * rows;
    const int block_slot = (int) lid >> 4;
    const int q4 = (int) lid & 15;
    const int simd_lane = (int) lid & 31;
    const int simd_group = (int) lid >> 5;
    const int blocks = FC_fairy2i_bundle_w2_decode_blocks;
    const int simd_groups = n_threads / 32;
    const int m16 = (row_base & 63) >> 4;
    const int row_lane_base = row_base & 15;

    float acc_real[rows];
    float acc_imag[rows];
    for (int tr = 0; tr < rows; ++tr) {
        acc_real[tr] = 0.0f;
        acc_imag[tr] = 0.0f;
    }

    for (int wb_base = 0; wb_base < blocks; wb_base += block_slots) {
        const int wb = wb_base + block_slot;
        if (wb < blocks) {
            float4 act_real;
            float4 act_imag;
            FOR_UNROLL (int part = 0; part < 4; ++part) {
                const int j = q4 * 4 + part;
                const int k_idx = wb * QK_FAIRY2I_TILE64 + j;
                const uint pair =
                    *((device const uint *) (x + (ulong) k_idx * FC_fairy2i_bundle_w2_decode_x_nb0));
                act_real[part] = fairy2i_bf16_to_f32((ushort) (pair & 0xffffU));
                act_imag[part] = fairy2i_bf16_to_f32((ushort) (pair >> 16));
            }

            const int physical_tile = (row_base / QK_FAIRY2I_TILE64) * blocks + wb;
            const int scale_base = physical_tile * 8;
            const half4 scale01 = *((device const half4 *) (scales + scale_base));
            const half4 scale23 = *((device const half4 *) (scales + scale_base + 4));
            const float4 wr = float4(scale01.x, scale01.z, scale23.x, scale23.z);
            const float4 wi = float4(scale01.y, scale01.w, scale23.y, scale23.w);
            const int slot = m16 * 16 + q4;
            const ulong code_base =
                ((((ulong) physical_tile * 64 + (ulong) slot) * 4) * 16) + (ulong) row_lane_base;
            const uint4 branch_rows = uint4(
                *((device const uint *) (codes + code_base)),
                *((device const uint *) (codes + code_base + 16)),
                *((device const uint *) (codes + code_base + 32)),
                *((device const uint *) (codes + code_base + 48)));

            FOR_UNROLL (int tr = 0; tr < rows; ++tr) {
                const uint4 packed = (branch_rows >> uint4(8 * tr)) & uint4(0xffU);
                fairy2i_accumulate_tile_weight4_decode_f32_reg(
                    packed, wr, wi, act_real, act_imag, tr, acc_real, acc_imag);
            }
        }
    }

    for (int out_idx = 0; out_idx < rows; ++out_idx) {
        const float real_sum = simd_sum(acc_real[out_idx]);
        const float imag_sum = simd_sum(acc_imag[out_idx]);
        if (simd_lane == 0) {
            tmp_real[out_idx * simd_groups + simd_group] = real_sum;
            tmp_imag[out_idx * simd_groups + simd_group] = imag_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < rows) {
        const int row = row_base + (int) lid;
        float real = 0.0f;
        float imag = 0.0f;
        for (int sg = 0; sg < simd_groups; ++sg) {
            real += tmp_real[(int) lid * simd_groups + sg];
            imag += tmp_imag[(int) lid * simd_groups + sg];
        }
        *((device uint *) (dst + (ulong) row * FC_fairy2i_bundle_w2_decode_dst_nb0)) =
            fairy2i_pack_bf16_pair(real, imag);
    }
}

#if defined(GGML_METAL_HAS_BF16)
kernel void kernel_fairy2i_bundle_w2_bf16_bf16scale_exact_tile4x1_w8_simd(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const uchar * codes                              [[buffer(1)]],
        device const ushort * scales                            [[buffer(2)]],
        device const char * x                                   [[buffer(3)]],
        device const char * bias                                [[buffer(4)]],
        device char * dst                                       [[buffer(5)]],
        threadgroup float * tmp_real                            [[threadgroup(0)]],
        threadgroup float * tmp_imag                            [[threadgroup(1)]],
        threadgroup ushort2 * qstage_table                      [[threadgroup(2)]],
        threadgroup ushort2 * uw_table                          [[threadgroup(3)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint2 tiitg                                             [[thread_position_in_threadgroup]]) {
    constexpr int rows = 4;
    constexpr int block_slots = 8;
    constexpr int n_threads = 128;

    const uint lid = tiitg.x;
    const int row_base = (int) tgpig.x * rows;
    const int block_slot = (int) lid >> 4;
    const int q4 = (int) lid & 15;
    const int simd_lane = (int) lid & 31;
    const int simd_group = (int) lid >> 5;
    const int blocks = args.k / QK_FAIRY2I_TILE64;
    const int simd_groups = n_threads / 32;
    const int m16 = (row_base & 63) >> 4;
    const int row_lane_base = row_base & 15;

    float acc_real[rows];
    float acc_imag[rows];
    uint min_coeff_metric = 255U;
    uint min_act_metric = 255U;
    for (int tr = 0; tr < rows; ++tr) {
        acc_real[tr] = 0.0f;
        acc_imag[tr] = 0.0f;
    }

    for (int wb_base = 0; wb_base < blocks; wb_base += block_slots) {
        const int wb = wb_base + block_slot;

        // Each thread owns one {block, branch, code} entry. The inactive component is always +0;
        // the active scale is already stored as the forward-visible BF16 magnitude payload.
        const uint branch_code = lid & 15U;
        const uint branch = branch_code >> 2;
        const uint code = branch_code & 3U;
        ushort2 qstage = ushort2(0);
        if (wb < blocks) {
            const int physical_tile = (row_base / QK_FAIRY2I_TILE64) * blocks + wb;
            const ushort magnitude =
                scales[(ulong) physical_tile * 8UL + (ulong) (2U * branch + (code >> 1))];
            const ushort stage =
                (ushort) ((uint) magnitude ^ ((code & 1U) != 0 ? 0x0000U : 0x8000U));
            if ((code & 2U) == 0) {
                qstage.x = stage;
            } else {
                qstage.y = stage;
            }
        }
        qstage_table[lid] = qstage;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Build both U and W tables from the shared stage payloads. Each thread produces one
        // 4-code pattern for U and one for W, preserving the BF16 merge boundary bit-for-bit.
        FOR_UNROLL (uint pair = 0; pair < 2; ++pair) {
            const uint pattern = lid & 15U;
            const uint pair_index = ((lid >> 4) * 2U + pair) * 16U + pattern;
            const uint branch0 = 2U * pair;
            const uint branch1 = branch0 + 1U;
            const uint code0 = pattern & 3U;
            const uint code1 = (pattern >> 2) & 3U;
            const ushort2 stage0 = qstage_table[((lid >> 4) * 4U + branch0) * 4U + code0];
            const ushort2 stage1 = qstage_table[((lid >> 4) * 4U + branch1) * 4U + code1];
            uw_table[pair_index] = ushort2(
                fairy2i_add_bf16_bits_rne(stage0.x, stage1.x),
                fairy2i_add_bf16_bits_rne(stage0.y, stage1.y));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (wb < blocks) {
            float4 act_real;
            float4 act_imag;
            FOR_UNROLL (int part = 0; part < 4; ++part) {
                const int j = q4 * 4 + part;
                const int k_idx = wb * QK_FAIRY2I_TILE64 + j;
                const uint pair = *((device const uint *) (x + (ulong) k_idx * args.x_nb0));
                const ushort act_real_bits = (ushort) (pair & 0xffffU);
                const ushort act_imag_bits = (ushort) (pair >> 16);
                min_act_metric = min(min_act_metric, fairy2i_bf16_product_metric(act_real_bits));
                min_act_metric = min(min_act_metric, fairy2i_bf16_product_metric(act_imag_bits));
                act_real[part] = fairy2i_bf16_to_f32(act_real_bits);
                act_imag[part] = fairy2i_bf16_to_f32(act_imag_bits);
            }

            const int physical_tile = (row_base / QK_FAIRY2I_TILE64) * blocks + wb;
            const int slot = m16 * 16 + q4;
            const ulong code_base =
                ((((ulong) physical_tile * 64 + (ulong) slot) * 4) * 16) + (ulong) row_lane_base;
            const uint4 branch_rows = uint4(
                *((device const uint *) (codes + code_base)),
                *((device const uint *) (codes + code_base + 16)),
                *((device const uint *) (codes + code_base + 32)),
                *((device const uint *) (codes + code_base + 48)));

            FOR_UNROLL (int tr = 0; tr < rows; ++tr) {
                const uint4 packed = (branch_rows >> uint4(8 * tr)) & uint4(0xffU);
                fairy2i_accumulate_tile_weight4_decode_bf16_exact_f32_reg(
                    packed, qstage_table + block_slot * 16, uw_table + block_slot * 32,
                    args.strict_staged_reconstruction != 0, act_real, act_imag, tr, acc_real, acc_imag,
                    min_coeff_metric);
            }
        }
        // All slots, including an invalid tail slot, must rendezvous before the shared tables are reused.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint simd_coeff_metric = simd_min(min_coeff_metric);
    const uint simd_act_metric = simd_min(min_act_metric);
    if (simd_lane == 0) {
        ((threadgroup uint *) tmp_real)[simd_group] = simd_coeff_metric;
        ((threadgroup uint *) tmp_imag)[simd_group] = simd_act_metric;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        uint tile_coeff_metric = 255U;
        uint tile_act_metric = 255U;
        for (int sg = 0; sg < simd_groups; ++sg) {
            tile_coeff_metric = min(tile_coeff_metric, ((threadgroup uint *) tmp_real)[sg]);
            tile_act_metric = min(tile_act_metric, ((threadgroup uint *) tmp_imag)[sg]);
        }
        ((threadgroup uint *) tmp_real)[0] = tile_coeff_metric;
        ((threadgroup uint *) tmp_imag)[0] = tile_act_metric;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Every non-zero product in a safe tile has a quantum of at least 2^-126. Native F32 reduction
    // therefore cannot flush a non-zero cancellation result. Rare low-exponent tiles are recomputed
    // in logical K order with a software fused BF16*BF16+F32 operation.
    const bool requires_software = fairy2i_product_metrics_require_software(
        ((threadgroup uint *) tmp_real)[0], ((threadgroup uint *) tmp_imag)[0]);
    if (requires_software) {
        if (lid < rows) {
            const int row = row_base + (int) lid;
            if (row < args.m) {
                uint real_bits = 0;
                uint imag_bits = 0;
                for (int k = 0; k < args.k; ++k) {
                    const uint pair = *((device const uint *) (x + (ulong) k * args.x_nb0));
                    const ushort xr_bits = (ushort) (pair & 0xffffU);
                    const ushort xi_bits = (ushort) (pair >> 16);
                    const ushort4 coeff_bits =
                        args.strict_staged_reconstruction != 0 ?
                            fairy2i_bundle_w2_coeff_at_bf16_exact_bits(codes, scales, blocks, row, k) :
                            fairy2i_bundle_w2_coeff_at_bf16_collapsed_f32_bits(codes, scales, blocks, row, k);
                    real_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(real_bits, coeff_bits.x, xr_bits);
                    real_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(real_bits, coeff_bits.y, xi_bits);
                    imag_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(imag_bits, coeff_bits.z, xr_bits);
                    imag_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(imag_bits, coeff_bits.w, xi_bits);
                }

                ushort out_real_bits;
                ushort out_imag_bits;
                if (args.has_bias) {
                    const float bias_real =
                        *((device const float *) (bias + (ulong) (row % args.bias_ne0) * args.bias_nb0));
                    const float bias_imag =
                        *((device const float *) (bias + (ulong) ((row + args.m) % args.bias_ne0) * args.bias_nb0));
                    out_real_bits = fairy2i_add_f32_bits_f32_bias_to_bf16_bits_rne(real_bits, bias_real);
                    out_imag_bits = fairy2i_add_f32_bits_f32_bias_to_bf16_bits_rne(imag_bits, bias_imag);
                } else {
                    out_real_bits = fairy2i_f32_to_bf16(as_type<float>(real_bits));
                    out_imag_bits = fairy2i_f32_to_bf16(as_type<float>(imag_bits));
                }
                *((device uint *) (dst + (ulong) row * args.dst_nb0)) =
                    (uint) out_real_bits | ((uint) out_imag_bits << 16);
            }
        }
        return;
    }

    for (int out_idx = 0; out_idx < rows; ++out_idx) {
        const float real_sum = simd_sum(acc_real[out_idx]);
        const float imag_sum = simd_sum(acc_imag[out_idx]);
        if (simd_lane == 0) {
            tmp_real[out_idx * simd_groups + simd_group] = real_sum;
            tmp_imag[out_idx * simd_groups + simd_group] = imag_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < rows) {
        const int row = row_base + (int) lid;
        if (row < args.m) {
            float real = 0.0f;
            float imag = 0.0f;
            for (int sg = 0; sg < simd_groups; ++sg) {
                real += tmp_real[(int) lid * simd_groups + sg];
                imag += tmp_imag[(int) lid * simd_groups + sg];
            }
            ushort real_bits;
            ushort imag_bits;
            if (args.has_bias) {
                const float bias_real =
                    *((device const float *) (bias + (ulong) (row % args.bias_ne0) * args.bias_nb0));
                const float bias_imag =
                    *((device const float *) (bias + (ulong) ((row + args.m) % args.bias_ne0) * args.bias_nb0));
                real_bits = fairy2i_add_f32_bias_to_bf16_bits_rne(real, bias_real);
                imag_bits = fairy2i_add_f32_bias_to_bf16_bits_rne(imag, bias_imag);
            } else {
                real_bits = fairy2i_f32_to_bf16(real);
                imag_bits = fairy2i_f32_to_bf16(imag);
            }

            *((device uint *) (dst + (ulong) row * args.dst_nb0)) =
                (uint) real_bits | ((uint) imag_bits << 16);
        }
    }
}

template <bool HAS_BIAS>
static inline void fairy2i_bundle_w2_bf16_bf16scale_exact_m64x1_impl(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args,
        device const uchar * codes,
        device const ushort * scales,
        device const char * x,
        device const char * bias,
        device char * dst,
        int blocks,
        ulong x_nb0,
        ulong dst_nb0,
        threadgroup uint * coeff01_lut,
        threadgroup uint * coeff23_lut,
        threadgroup uint * act_pairs,
        threadgroup uint * metric_scratch,
        threadgroup ushort2 * qstage,
        threadgroup ushort2 * uw_table,
        uint2 tgpig,
        uint2 tiitg) {
    constexpr uint rows = 64;
    constexpr uint block_group = 4;
    constexpr uint coeffs_per_block = 256;
    constexpr uint acts_per_block = QK_FAIRY2I_TILE64;
    constexpr uint qstage_per_block = 16;
    constexpr uint uw_per_block = 32;
    constexpr uint metrics_per_block = 4;

    const uint lid = tiitg.x;
    const uint simd_lane = lid & 31U;
    const uint simd_group = lid >> 5;
    const uint block_slot = lid >> 6;
    const uint block_lid = lid & 63U;
    const uint block_simd = simd_group & 1U;
    const uint kseg = simd_lane >> 3;
    const uint row8 = simd_lane & 7U;
    const int row_base = (int) tgpig.x * (int) rows;
    const int physical_m_tile = (int) tgpig.x;
    const uint m16 = simd_group >> 1;
    const uint row_lane = ((simd_group & 1U) << 3) | row8;

    float acc_real = 0.0f;
    float acc_imag = 0.0f;
    uint leader_needs_software = 0U;

    for (int wb_base = 0; wb_base < blocks; wb_base += (int) block_group) {
        const uint group_blocks = (uint) min((int) block_group, blocks - wb_base);
        const bool block_active = block_slot < group_blocks;
        const int wb = wb_base + (int) block_slot;
        const int physical_tile = physical_m_tile * blocks + wb;
        uint       coeff_metric_bound = 255U;
        uint       min_act_metric     = 255U;

        // Four 64-thread subgroups stage four physical K64 tiles concurrently.
        if (block_active) {
            const uint pair =
                *((device const uint *) (x + ((ulong) wb * QK_FAIRY2I_TILE64 + block_lid) * x_nb0));
            const uint real_metric = fairy2i_bf16_product_metric((ushort) (pair & 0xffffU));
            const uint imag_metric = fairy2i_bf16_product_metric((ushort) (pair >> 16));
            act_pairs[block_slot * acts_per_block + block_lid] = pair;
            min_act_metric = min(real_metric, imag_metric);
        }

        // The first SIMD-group in each 64-thread subgroup stages that K64 tile's
        // four branch/code payloads. Strict regression mode also materializes
        // the two intermediate BF16 U/W tables.
        if (block_active && block_simd == 0U) {
            threadgroup ushort2 * block_qstage       = qstage + block_slot * qstage_per_block;
            threadgroup ushort2 * block_uw_table     = uw_table + block_slot * uw_per_block;
            uint                  stage_metric       = 255U;
            uint                  stage_max_exponent = 0U;
            const uint q_index = simd_lane & 15U;
            const uint branch = q_index >> 2;
            const uint code = q_index & 3U;
            const uint scale_index = branch * 2U + (code >> 1);
            const uint scale_source =
                simd_lane < 8U ? (uint) scales[(ulong) physical_tile * 8UL + simd_lane] : 0U;
            const ushort magnitude = (ushort) simd_shuffle(scale_source, scale_index);

            if (simd_lane < 16U) {
                const ushort stage =
                    (ushort) ((uint) magnitude ^ ((code & 1U) != 0 ? 0x0000U : 0x8000U));
                block_qstage[q_index] = (code & 2U) == 0 ? ushort2(stage, 0) : ushort2(0, stage);
                if ((code & 1U) == 0U) {
                    // One sign of each of the eight unique qstage scale payloads is sufficient:
                    // the product metric ignores the sign bit.
                    stage_metric       = fairy2i_bf16_product_metric(stage);
                    stage_max_exponent = stage_metric == 255U ? 0U : stage_metric;
                }
            }
            const uint min_stage_metric   = simd_min(stage_metric);
            const uint max_stage_exponent = simd_max(stage_max_exponent);
            if (simd_lane == 0U) {
                // A non-zero BF16 add can lower its smallest input exponent field by at most
                // seven. U/W and then Aij cross two such boundaries, so min_stage - 14 is a
                // lower bound for every non-zero final coefficient. Low/exceptional stages and
                // inputs large enough for the two adds to overflow conservatively force replay.
                coeff_metric_bound =
                    fairy2i_bf16_two_add_coefficient_metric_bound(min_stage_metric, max_stage_exponent);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            if (args.strict_staged_reconstruction != 0) {
                const uint pair = simd_lane >> 4;
                const uint pattern = simd_lane & 15U;
                const uint branch0 = 2U * pair;
                const uint code0 = pattern & 3U;
                const uint code1 = (pattern >> 2) & 3U;
                const ushort2 stage0 = block_qstage[branch0 * 4U + code0];
                const ushort2 stage1 = block_qstage[(branch0 + 1U) * 4U + code1];
                block_uw_table[simd_lane] = ushort2(
                    fairy2i_add_bf16_bits_rne(stage0.x, stage1.x),
                    fairy2i_add_bf16_bits_rne(stage0.y, stage1.y));
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each 64-thread subgroup builds all 256 final Aij entries for one K64 tile.
        if (block_active) {
            threadgroup ushort2 * block_qstage = qstage + block_slot * qstage_per_block;
            threadgroup ushort2 * block_uw_table = uw_table + block_slot * uw_per_block;
            const uint coeff_base = block_slot * coeffs_per_block;
            FOR_UNROLL (uint pattern_quarter = 0; pattern_quarter < 4U; ++pattern_quarter) {
                const uint pattern = block_lid + pattern_quarter * 64U;
                ushort4 coeff;
                if (args.strict_staged_reconstruction != 0) {
                    coeff = fairy2i_reconstruct_w2_coeff_from_uw_bf16_exact_bits(
                        block_uw_table[pattern & 15U], block_uw_table[16U + (pattern >> 4)]);
                } else {
                    const uint4 code = uint4(
                        pattern & 3U,
                        (pattern >> 2) & 3U,
                        (pattern >> 4) & 3U,
                        (pattern >> 6) & 3U);
                    const ushort2 u0 = block_qstage[code.x];
                    const ushort2 u1 = block_qstage[4U + code.y];
                    const ushort2 w0 = block_qstage[8U + code.z];
                    const ushort2 w1 = block_qstage[12U + code.w];
                    coeff = fairy2i_reconstruct_w2_coeff_from_stage_collapsed_f32_native_bits(
                        ushort4(u0.x, u1.x, w0.x, w1.x),
                        ushort4(u0.y, u1.y, w0.y, w1.y));
                }
                coeff01_lut[coeff_base + pattern] = (uint) coeff.x | ((uint) coeff.y << 16);
                coeff23_lut[coeff_base + pattern] = (uint) coeff.z | ((uint) coeff.w << 16);
            }
        }

        const uint simd_act_metric = simd_min(min_act_metric);
        if (simd_lane == 0U) {
            const uint metric_base = block_slot * metrics_per_block;
            if (block_simd == 0U) {
                metric_scratch[metric_base] = coeff_metric_bound;
            }
            metric_scratch[metric_base + 1U + block_simd * 2U] = simd_act_metric;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0U) {
            for (uint metric_slot = 0; metric_slot < group_blocks; ++metric_slot) {
                const uint metric_base       = metric_slot * metrics_per_block;
                const uint tile_coeff_metric = metric_scratch[metric_base];
                const uint tile_act_metric   = min(metric_scratch[metric_base + 1U], metric_scratch[metric_base + 3U]);
                leader_needs_software |= fairy2i_product_metrics_require_software(tile_coeff_metric, tile_act_metric);
            }
        }

        // Consume the four K64 tiles in their original order so every F32 FMA boundary is
        // unchanged. The shared tables remain read-only until the complete group is consumed.
        for (uint compute_slot = 0; compute_slot < group_blocks; ++compute_slot) {
            const int compute_wb = wb_base + (int) compute_slot;
            const int compute_physical_tile = physical_m_tile * blocks + compute_wb;
            device const uchar * code_ptr =
                codes + (ulong) compute_physical_tile * 64UL * 4UL * 16UL +
                (ulong) (m16 * 16U + kseg * 4U) * 4UL * 16UL + row_lane;
            const uint coeff_base = compute_slot * coeffs_per_block;
            const uint act_base = compute_slot * acts_per_block;

            FOR_UNROLL (uint q4_in_seg = 0; q4_in_seg < 4; ++q4_in_seg) {
                // Transpose four branch-major code bytes into four byte-sized LUT indices.
                uint packed_lut_indices =
                    (uint) code_ptr[0] |
                    ((uint) code_ptr[16] << 8) |
                    ((uint) code_ptr[32] << 16) |
                    ((uint) code_ptr[48] << 24);
                uint swap = (packed_lut_indices ^ (packed_lut_indices >> 12)) & 0x0000f0f0U;
                packed_lut_indices ^= swap ^ (swap << 12);
                swap = (packed_lut_indices ^ (packed_lut_indices >> 6)) & 0x00cc00ccU;
                packed_lut_indices ^= swap ^ (swap << 6);

                FOR_UNROLL (uint part = 0; part < 4; ++part) {
                    const uint lut_index = (packed_lut_indices >> (8U * part)) & 0xffU;
                    const uint k_local = kseg * 16U + q4_in_seg * 4U + part;
                    const uint act_pair = act_pairs[act_base + k_local];
                    const ushort xr_bits = (ushort) (act_pair & 0xffffU);
                    const ushort xi_bits = (ushort) (act_pair >> 16);
                    const uint coeff01 = coeff01_lut[coeff_base + lut_index];
                    const uint coeff23 = coeff23_lut[coeff_base + lut_index];
                    const ushort rr_bits = (ushort) (coeff01 & 0xffffU);
                    const ushort ri_bits = (ushort) (coeff01 >> 16);
                    const ushort ir_bits = (ushort) (coeff23 & 0xffffU);
                    const ushort ii_bits = (ushort) (coeff23 >> 16);
                    const float xr = fairy2i_bf16_to_f32(xr_bits);
                    const float xi = fairy2i_bf16_to_f32(xi_bits);

                    acc_real = fma(fairy2i_bf16_to_f32(rr_bits), xr, acc_real);
                    acc_real = fma(fairy2i_bf16_to_f32(ri_bits), xi, acc_real);
                    acc_imag = fma(fairy2i_bf16_to_f32(ir_bits), xr, acc_imag);
                    acc_imag = fma(fairy2i_bf16_to_f32(ii_bits), xi, acc_imag);
                }
                code_ptr += 4 * 16;
            }
        }

        // Publish the leader's cumulative predicate before the existing group-end rendezvous.
        // The next group may reuse this slot for metrics; the leader retains the cumulative value.
        if (lid == 0U) {
            metric_scratch[0] = leader_needs_software;
        }

        // Protect all four shared tables before their next grouped reconstruction and make the
        // final cumulative predicate visible uniformly after the last group.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const bool needs_software = blocks > 0 && metric_scratch[0] != 0U;

    // Four lanes hold the K16 partials for each row. Reduce only within the row's 4-lane set.
    acc_real += simd_shuffle_down(acc_real, 16);
    acc_imag += simd_shuffle_down(acc_imag, 16);
    acc_real += simd_shuffle_down(acc_real, 8);
    acc_imag += simd_shuffle_down(acc_imag, 8);

    if (simd_lane < 8U) {
        const int row = row_base + (int) (simd_group * 8U + simd_lane);
        ushort out_real_bits;
        ushort out_imag_bits;

        if (needs_software) {
            uint real_bits = 0;
            uint imag_bits = 0;
            const int k = blocks * QK_FAIRY2I_TILE64;
            for (int ik = 0; ik < k; ++ik) {
                const uint pair = *((device const uint *) (x + (ulong) ik * x_nb0));
                const ushort xr_bits = (ushort) (pair & 0xffffU);
                const ushort xi_bits = (ushort) (pair >> 16);
                const ushort4 coeff_bits =
                    args.strict_staged_reconstruction != 0 ?
                        fairy2i_bundle_w2_coeff_at_bf16_exact_bits(codes, scales, blocks, row, ik) :
                        fairy2i_bundle_w2_coeff_at_bf16_collapsed_f32_bits(codes, scales, blocks, row, ik);
                real_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(real_bits, coeff_bits.x, xr_bits);
                real_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(real_bits, coeff_bits.y, xi_bits);
                imag_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(imag_bits, coeff_bits.z, xr_bits);
                imag_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(imag_bits, coeff_bits.w, xi_bits);
            }

            if (HAS_BIAS && args.has_bias) {
                const float bias_real =
                    *((device const float *) (bias + (ulong) (row % args.bias_ne0) * args.bias_nb0));
                const float bias_imag =
                    *((device const float *) (bias + (ulong) ((row + args.m) % args.bias_ne0) * args.bias_nb0));
                out_real_bits = fairy2i_add_f32_bits_f32_bias_to_bf16_bits_rne(real_bits, bias_real);
                out_imag_bits = fairy2i_add_f32_bits_f32_bias_to_bf16_bits_rne(imag_bits, bias_imag);
            } else {
                out_real_bits = fairy2i_f32_to_bf16(as_type<float>(real_bits));
                out_imag_bits = fairy2i_f32_to_bf16(as_type<float>(imag_bits));
            }
        } else if (HAS_BIAS && args.has_bias) {
            const float bias_real =
                *((device const float *) (bias + (ulong) (row % args.bias_ne0) * args.bias_nb0));
            const float bias_imag =
                *((device const float *) (bias + (ulong) ((row + args.m) % args.bias_ne0) * args.bias_nb0));
            out_real_bits = fairy2i_add_f32_bias_to_bf16_bits_rne(acc_real, bias_real);
            out_imag_bits = fairy2i_add_f32_bias_to_bf16_bits_rne(acc_imag, bias_imag);
        } else {
            out_real_bits = fairy2i_f32_to_bf16(acc_real);
            out_imag_bits = fairy2i_f32_to_bf16(acc_imag);
        }

        *((device uint *) (dst + (ulong) row * dst_nb0)) =
            (uint) out_real_bits | ((uint) out_imag_bits << 16);
    }
}

kernel void kernel_fairy2i_bundle_w2_bf16_bf16scale_exact_m64x1_simd(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const uchar * codes                              [[buffer(1)]],
        device const ushort * scales                            [[buffer(2)]],
        device const char * x                                   [[buffer(3)]],
        device const char * bias                                [[buffer(4)]],
        device char * dst                                       [[buffer(5)]],
        threadgroup uint * coeff01_lut                          [[threadgroup(0)]],
        threadgroup uint * coeff23_lut                          [[threadgroup(1)]],
        threadgroup uint * act_pairs                            [[threadgroup(2)]],
        threadgroup uint * metric_scratch                       [[threadgroup(3)]],
        threadgroup ushort2 * qstage                            [[threadgroup(4)]],
        threadgroup ushort2 * uw_table                          [[threadgroup(5)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint2 tiitg                                             [[thread_position_in_threadgroup]]) {
    fairy2i_bundle_w2_bf16_bf16scale_exact_m64x1_impl<true>(
        args, codes, scales, x, bias, dst, args.k / QK_FAIRY2I_TILE64, args.x_nb0, args.dst_nb0,
        coeff01_lut, coeff23_lut, act_pairs, metric_scratch, qstage, uw_table, tgpig, tiitg);
}

kernel void kernel_fairy2i_bundle_w2_bf16_bf16scale_exact_m64x1_nobias_fc_simd(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const uchar * codes                              [[buffer(1)]],
        device const ushort * scales                            [[buffer(2)]],
        device const char * x                                   [[buffer(3)]],
        device char * dst                                       [[buffer(4)]],
        threadgroup uint * coeff01_lut                          [[threadgroup(0)]],
        threadgroup uint * coeff23_lut                          [[threadgroup(1)]],
        threadgroup uint * act_pairs                            [[threadgroup(2)]],
        threadgroup uint * metric_scratch                       [[threadgroup(3)]],
        threadgroup ushort2 * qstage                            [[threadgroup(4)]],
        threadgroup ushort2 * uw_table                          [[threadgroup(5)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint2 tiitg                                             [[thread_position_in_threadgroup]]) {
    fairy2i_bundle_w2_bf16_bf16scale_exact_m64x1_impl<false>(
        args, codes, scales, x, x, dst, FC_fairy2i_bundle_w2_decode_blocks,
        (ulong) FC_fairy2i_bundle_w2_decode_x_nb0, (ulong) FC_fairy2i_bundle_w2_decode_dst_nb0,
        coeff01_lut, coeff23_lut, act_pairs, metric_scratch, qstage, uw_table, tgpig, tiitg);
}
#endif

kernel void kernel_fairy2i_wide_linear_w1_half_w64scale_mma32x16_k16(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const block_fairy2i_tile64_v2 * u_s0             [[buffer(1)]],
        device const block_fairy2i_tile64_v2 * w_s0             [[buffer(2)]],
        device const half * act_h                               [[buffer(3)]],
        device const char * bias                                [[buffer(4)]],
        device char * dst                                       [[buffer(5)]],
        threadgroup half * coeff_real_from_real                 [[threadgroup(0)]],
        threadgroup half * coeff_real_from_imag                 [[threadgroup(1)]],
        threadgroup half * coeff_imag_from_real                 [[threadgroup(2)]],
        threadgroup half * coeff_imag_from_imag                 [[threadgroup(3)]],
        threadgroup half * act_real_tile0                       [[threadgroup(4)]],
        threadgroup half * act_imag_tile0                       [[threadgroup(5)]],
        threadgroup half * act_real_tile1                       [[threadgroup(6)]],
        threadgroup half * act_imag_tile1                       [[threadgroup(7)]],
        threadgroup float * out_tile                            [[threadgroup(8)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint tiitg                                              [[thread_index_in_threadgroup]],
        uint sgitg                                              [[simdgroup_index_in_threadgroup]]) {
    constexpr int row_tile = 32;
    constexpr int n_threads = 128;

    const int row_base = (int) tgpig.x * row_tile;
    const int col_base = (int) tgpig.y * 16;
    const int blocks = args.k / QK_FAIRY2I_TILE64;
    const int scale_row = (row_base / QK_FAIRY2I_TILE64) * QK_FAIRY2I_TILE64;
    const int coeff_base = (int) sgitg * 8 * 16;

    simdgroup_half8x8 a_rr;
    simdgroup_half8x8 a_ri;
    simdgroup_half8x8 a_ir;
    simdgroup_half8x8 a_ii;
    simdgroup_half8x8 b_r0;
    simdgroup_half8x8 b_i0;
    simdgroup_half8x8 b_r1;
    simdgroup_half8x8 b_i1;
    simdgroup_float8x8 c_r0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_i0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_r1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_i1 = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (int wb = 0; wb < blocks; ++wb) {
        const int scale_index = scale_row * blocks + wb;
        const half2 wr = half2(u_s0[scale_index].d_real, w_s0[scale_index].d_real);
        const half2 wi = half2(u_s0[scale_index].d_imag, w_s0[scale_index].d_imag);

        const int coeff_k_lane = (int) tiitg & 7;
        const int coeff_row0 = (int) tiitg >> 3;
        const int coeff_row1 = coeff_row0 + 16;
        uint4 packed_codes = uint4(0);
        if (row_base + coeff_row0 < args.m) {
            const int w_index = (row_base + coeff_row0) * blocks + wb;
            packed_codes.x = (uint) u_s0[w_index].qs[coeff_k_lane] |
                             ((uint) u_s0[w_index].qs[coeff_k_lane + 8] << 8);
            packed_codes.y = (uint) w_s0[w_index].qs[coeff_k_lane] |
                             ((uint) w_s0[w_index].qs[coeff_k_lane + 8] << 8);
        }
        if (row_base + coeff_row1 < args.m) {
            const int w_index = (row_base + coeff_row1) * blocks + wb;
            packed_codes.z = (uint) u_s0[w_index].qs[coeff_k_lane] |
                             ((uint) u_s0[w_index].qs[coeff_k_lane + 8] << 8);
            packed_codes.w = (uint) w_s0[w_index].qs[coeff_k_lane] |
                             ((uint) w_s0[w_index].qs[coeff_k_lane + 8] << 8);
        }

        for (int k_chunk = 0; k_chunk < QK_FAIRY2I_TILE64; k_chunk += 16) {
            for (int ik = 0; ik < 2; ++ik) {
                const int k_base = k_chunk + ik * 8;
                const int code_shift = (k_base & 8) + 2 * (k_base >> 4);
                const uint4 code = (packed_codes >> uint4(code_shift)) & uint4(3);

                half4 coeff0 = half4(0.0h);
                if (row_base + coeff_row0 < args.m) {
                    coeff0 = fairy2i_mma_coeff_w1_codes_scaled_half(code.xy, wr, wi);
                }
                const uint coeff_index0 = coeff_row0 * 16 + ik * 8 + coeff_k_lane;
                coeff_real_from_real[coeff_index0] = coeff0.x;
                coeff_real_from_imag[coeff_index0] = coeff0.y;
                coeff_imag_from_real[coeff_index0] = coeff0.z;
                coeff_imag_from_imag[coeff_index0] = coeff0.w;

                half4 coeff1 = half4(0.0h);
                if (row_base + coeff_row1 < args.m) {
                    coeff1 = fairy2i_mma_coeff_w1_codes_scaled_half(code.zw, wr, wi);
                }
                const uint coeff_index1 = coeff_row1 * 16 + ik * 8 + coeff_k_lane;
                coeff_real_from_real[coeff_index1] = coeff1.x;
                coeff_real_from_imag[coeff_index1] = coeff1.y;
                coeff_imag_from_real[coeff_index1] = coeff1.z;
                coeff_imag_from_imag[coeff_index1] = coeff1.w;
            }

            for (uint idx = tiitg; idx < 16 * 16; idx += n_threads) {
                const int col_local = (int) idx >> 4;
                const int k_local = (int) idx & 15;
                const int tile = col_local >> 3;
                const int col_lane = col_local & 7;
                const int col = col_base + col_local;

                half2 xh = half2(0.0h);
                if (col < args.act_rows) {
                    xh = fairy2i_load_staged_half_activation_pair(act_h, col, wb, blocks, k_chunk + k_local);
                }

                const int act_tile_idx = k_local * 8 + col_lane;
                if (tile == 0) {
                    act_real_tile0[act_tile_idx] = xh.x;
                    act_imag_tile0[act_tile_idx] = xh.y;
                } else {
                    act_real_tile1[act_tile_idx] = xh.x;
                    act_imag_tile1[act_tile_idx] = xh.y;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int ik = 0; ik < 2; ++ik) {
                simdgroup_load(a_rr, coeff_real_from_real + coeff_base + ik * 8, 16);
                simdgroup_load(a_ri, coeff_real_from_imag + coeff_base + ik * 8, 16);
                simdgroup_load(a_ir, coeff_imag_from_real + coeff_base + ik * 8, 16);
                simdgroup_load(a_ii, coeff_imag_from_imag + coeff_base + ik * 8, 16);
                simdgroup_load(b_r0, act_real_tile0 + ik * 64);
                simdgroup_load(b_i0, act_imag_tile0 + ik * 64);
                simdgroup_load(b_r1, act_real_tile1 + ik * 64);
                simdgroup_load(b_i1, act_imag_tile1 + ik * 64);

                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_multiply_accumulate(c_r0, a_rr, b_r0, c_r0);
                simdgroup_multiply_accumulate(c_r0, a_ri, b_i0, c_r0);
                simdgroup_multiply_accumulate(c_i0, a_ir, b_r0, c_i0);
                simdgroup_multiply_accumulate(c_i0, a_ii, b_i0, c_i0);
                simdgroup_multiply_accumulate(c_r1, a_rr, b_r1, c_r1);
                simdgroup_multiply_accumulate(c_r1, a_ri, b_i1, c_r1);
                simdgroup_multiply_accumulate(c_i1, a_ir, b_r1, c_i1);
                simdgroup_multiply_accumulate(c_i1, a_ii, b_i1, c_i1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int simdgroup_out_base = (int) sgitg * 256;
    simdgroup_store(c_r0, out_tile + simdgroup_out_base, 8);
    simdgroup_store(c_i0, out_tile + simdgroup_out_base + 64, 8);
    simdgroup_store(c_r1, out_tile + simdgroup_out_base + 128, 8);
    simdgroup_store(c_i1, out_tile + simdgroup_out_base + 192, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tiitg; idx < row_tile * 16; idx += n_threads) {
        const int row_lane = (int) idx / 16;
        const int col_local = (int) idx & 15;
        const int tile = col_local >> 3;
        const int col_lane = col_local & 7;
        const int row = row_base + row_lane;
        const int col = col_base + col_local;
        if (row < args.m && col < args.act_rows) {
            const int i1 = col % args.x_ne1;
            const int i2 = (col / args.x_ne1) % args.x_ne2;
            const int i3 = col / (args.x_ne1 * args.x_ne2);
            const int row_group = row_lane >> 3;
            const int row_in_group = row_lane & 7;
            const int out_base = row_group * 256 + tile * 128;
            float out_real = out_tile[out_base + row_in_group * 8 + col_lane];
            float out_imag = out_tile[out_base + 64 + row_in_group * 8 + col_lane];

            if (args.has_bias) {
                const int b0r = row % args.bias_ne0;
                const int b0i = (row + args.m) % args.bias_ne0;
                const int b1 = i1 % args.bias_ne1;
                const int b2 = i2 % args.bias_ne2;
                const int b3 = i3 % args.bias_ne3;
                out_real += *((device const float *) (bias + (ulong) b0r * args.bias_nb0 + (ulong) b1 * args.bias_nb1 +
                                                      (ulong) b2 * args.bias_nb2 + (ulong) b3 * args.bias_nb3));
                out_imag += *((device const float *) (bias + (ulong) b0i * args.bias_nb0 + (ulong) b1 * args.bias_nb1 +
                                                      (ulong) b2 * args.bias_nb2 + (ulong) b3 * args.bias_nb3));
            }

            *((device uint *) (dst + (ulong) row * args.dst_nb0 + (ulong) i1 * args.dst_nb1 + (ulong) i2 * args.dst_nb2 +
                               (ulong) i3 * args.dst_nb3)) = fairy2i_pack_bf16_pair(out_real, out_imag);
        }
    }
}

kernel void kernel_fairy2i_bundle_w1_half_mma32x16_k16(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const uchar * codes                              [[buffer(1)]],
        device const half * scales                              [[buffer(2)]],
        device const half * act_h                               [[buffer(3)]],
        device const char * bias                                [[buffer(4)]],
        device char * dst                                       [[buffer(5)]],
        threadgroup half * coeff_real_from_real                 [[threadgroup(0)]],
        threadgroup half * coeff_real_from_imag                 [[threadgroup(1)]],
        threadgroup half * coeff_imag_from_real                 [[threadgroup(2)]],
        threadgroup half * coeff_imag_from_imag                 [[threadgroup(3)]],
        threadgroup half * act_real_tile0                       [[threadgroup(4)]],
        threadgroup half * act_imag_tile0                       [[threadgroup(5)]],
        threadgroup half * act_real_tile1                       [[threadgroup(6)]],
        threadgroup half * act_imag_tile1                       [[threadgroup(7)]],
        threadgroup float * out_tile                            [[threadgroup(8)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint tiitg                                              [[thread_index_in_threadgroup]],
        uint sgitg                                              [[simdgroup_index_in_threadgroup]]) {
    constexpr int row_tile = 32;
    constexpr int n_threads = 128;

    const int row_base = (int) tgpig.x * row_tile;
    const int col_base = (int) tgpig.y * 16;
    const int blocks = args.k / QK_FAIRY2I_TILE64;
    const int physical_m_tile = row_base / QK_FAIRY2I_TILE64;
    const int coeff_base = (int) sgitg * 8 * 16;

    simdgroup_half8x8 a_rr;
    simdgroup_half8x8 a_ri;
    simdgroup_half8x8 a_ir;
    simdgroup_half8x8 a_ii;
    simdgroup_half8x8 b_r0;
    simdgroup_half8x8 b_i0;
    simdgroup_half8x8 b_r1;
    simdgroup_half8x8 b_i1;
    simdgroup_float8x8 c_r0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_i0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_r1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_i1 = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (int wb = 0; wb < blocks; ++wb) {
        const int physical_tile = physical_m_tile * blocks + wb;
        const int scale_base = physical_tile * 4;
        const half2 wr = half2(scales[scale_base + 0], scales[scale_base + 2]);
        const half2 wi = half2(scales[scale_base + 1], scales[scale_base + 3]);

        for (int k_chunk = 0; k_chunk < QK_FAIRY2I_TILE64; k_chunk += 16) {
            // One thread owns one (row, q4) pair and expands its four consecutive K codes into the MMA tile.
            const int coeff_row = (int) tiitg & 31;
            const int q4_local = (int) tiitg >> 5;
            const int row_in_m64 = (row_base + coeff_row) & 63;
            const int m16 = row_in_m64 >> 4;
            const int row_lane = row_in_m64 & 15;
            const int q4 = (k_chunk >> 2) + q4_local;
            const int slot = m16 * 16 + q4;
            const ulong code_base =
                ((((ulong) physical_tile * 64 + (ulong) slot) * 2) * 16) + (ulong) row_lane;
            const uint u_codes = (uint) codes[code_base];
            const uint w_codes = (uint) codes[code_base + 16];

            FOR_UNROLL (int part = 0; part < 4; ++part) {
                const uint2 code = uint2((u_codes >> (2 * part)) & 3, (w_codes >> (2 * part)) & 3);
                const half4 coeff = fairy2i_mma_coeff_w1_codes_scaled_half(code, wr, wi);
                const int coeff_index = coeff_row * 16 + q4_local * 4 + part;
                coeff_real_from_real[coeff_index] = coeff.x;
                coeff_real_from_imag[coeff_index] = coeff.y;
                coeff_imag_from_real[coeff_index] = coeff.z;
                coeff_imag_from_imag[coeff_index] = coeff.w;
            }

            for (uint idx = tiitg; idx < 16 * 16; idx += n_threads) {
                const int col_local = (int) idx >> 4;
                const int k_local = (int) idx & 15;
                const int tile = col_local >> 3;
                const int col_lane = col_local & 7;
                const int col = col_base + col_local;

                half2 xh = half2(0.0h);
                if (col < args.act_rows) {
                    xh = fairy2i_load_staged_half_activation_pair(act_h, col, wb, blocks, k_chunk + k_local);
                }

                const int act_tile_idx = k_local * 8 + col_lane;
                if (tile == 0) {
                    act_real_tile0[act_tile_idx] = xh.x;
                    act_imag_tile0[act_tile_idx] = xh.y;
                } else {
                    act_real_tile1[act_tile_idx] = xh.x;
                    act_imag_tile1[act_tile_idx] = xh.y;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int ik = 0; ik < 2; ++ik) {
                simdgroup_load(a_rr, coeff_real_from_real + coeff_base + ik * 8, 16);
                simdgroup_load(a_ri, coeff_real_from_imag + coeff_base + ik * 8, 16);
                simdgroup_load(a_ir, coeff_imag_from_real + coeff_base + ik * 8, 16);
                simdgroup_load(a_ii, coeff_imag_from_imag + coeff_base + ik * 8, 16);
                simdgroup_load(b_r0, act_real_tile0 + ik * 64);
                simdgroup_load(b_i0, act_imag_tile0 + ik * 64);
                simdgroup_load(b_r1, act_real_tile1 + ik * 64);
                simdgroup_load(b_i1, act_imag_tile1 + ik * 64);

                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_multiply_accumulate(c_r0, a_rr, b_r0, c_r0);
                simdgroup_multiply_accumulate(c_r0, a_ri, b_i0, c_r0);
                simdgroup_multiply_accumulate(c_i0, a_ir, b_r0, c_i0);
                simdgroup_multiply_accumulate(c_i0, a_ii, b_i0, c_i0);
                simdgroup_multiply_accumulate(c_r1, a_rr, b_r1, c_r1);
                simdgroup_multiply_accumulate(c_r1, a_ri, b_i1, c_r1);
                simdgroup_multiply_accumulate(c_i1, a_ir, b_r1, c_i1);
                simdgroup_multiply_accumulate(c_i1, a_ii, b_i1, c_i1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int simdgroup_out_base = (int) sgitg * 256;
    simdgroup_store(c_r0, out_tile + simdgroup_out_base, 8);
    simdgroup_store(c_i0, out_tile + simdgroup_out_base + 64, 8);
    simdgroup_store(c_r1, out_tile + simdgroup_out_base + 128, 8);
    simdgroup_store(c_i1, out_tile + simdgroup_out_base + 192, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tiitg; idx < row_tile * 16; idx += n_threads) {
        const int row_lane = (int) idx / 16;
        const int col_local = (int) idx & 15;
        const int tile = col_local >> 3;
        const int col_lane = col_local & 7;
        const int row = row_base + row_lane;
        const int col = col_base + col_local;
        if (row < args.m && col < args.act_rows) {
            const int i1 = col % args.x_ne1;
            const int i2 = (col / args.x_ne1) % args.x_ne2;
            const int i3 = col / (args.x_ne1 * args.x_ne2);
            const int row_group = row_lane >> 3;
            const int row_in_group = row_lane & 7;
            const int out_base = row_group * 256 + tile * 128;
            float out_real = out_tile[out_base + row_in_group * 8 + col_lane];
            float out_imag = out_tile[out_base + 64 + row_in_group * 8 + col_lane];

            if (args.has_bias) {
                const int b0r = row % args.bias_ne0;
                const int b0i = (row + args.m) % args.bias_ne0;
                const int b1 = i1 % args.bias_ne1;
                const int b2 = i2 % args.bias_ne2;
                const int b3 = i3 % args.bias_ne3;
                out_real += *((device const float *) (bias + (ulong) b0r * args.bias_nb0 + (ulong) b1 * args.bias_nb1 +
                                                      (ulong) b2 * args.bias_nb2 + (ulong) b3 * args.bias_nb3));
                out_imag += *((device const float *) (bias + (ulong) b0i * args.bias_nb0 + (ulong) b1 * args.bias_nb1 +
                                                      (ulong) b2 * args.bias_nb2 + (ulong) b3 * args.bias_nb3));
            }

            *((device uint *) (dst + (ulong) row * args.dst_nb0 + (ulong) i1 * args.dst_nb1 + (ulong) i2 * args.dst_nb2 +
                               (ulong) i3 * args.dst_nb3)) = fairy2i_pack_bf16_pair(out_real, out_imag);
        }
    }
}

kernel void kernel_fairy2i_bundle_w1_half_mma32x16_k16_direct_act(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const uchar * codes                              [[buffer(1)]],
        device const half * scales                              [[buffer(2)]],
        device const half * act_h                               [[buffer(3)]],
        device const char * bias                                [[buffer(4)]],
        device char * dst                                       [[buffer(5)]],
        threadgroup half * coeff_real_from_real                 [[threadgroup(0)]],
        threadgroup half * coeff_real_from_imag                 [[threadgroup(1)]],
        threadgroup half * coeff_imag_from_real                 [[threadgroup(2)]],
        threadgroup half * coeff_imag_from_imag                 [[threadgroup(3)]],
        threadgroup float * out_tile                            [[threadgroup(4)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint tiitg                                              [[thread_index_in_threadgroup]],
        uint sgitg                                              [[simdgroup_index_in_threadgroup]]) {
    constexpr int row_tile = 32;
    constexpr int n_threads = 128;

    const int row_base = (int) tgpig.x * row_tile;
    const int col_base = (int) tgpig.y * 16;
    const int blocks = args.k / QK_FAIRY2I_TILE64;
    const int physical_m_tile = row_base / QK_FAIRY2I_TILE64;
    const int coeff_base = (int) sgitg * 8 * 16;
    const int simd_lane = (int) tiitg & 31;
    const int coeff_row = (int) sgitg * 8 + (simd_lane >> 2);
    const int q4_local = simd_lane & 3;
    const int row_in_m64 = (row_base + coeff_row) & 63;
    const int m16 = row_in_m64 >> 4;
    const int row_lane = row_in_m64 & 15;
    const ulong act_plane_size =
        (ulong) QK_FAIRY2I_ACT_Q16_64 * (ulong) FC_fairy2i_bundle_w1_prefill_act_rows;

    simdgroup_half8x8 a_rr;
    simdgroup_half8x8 a_ri;
    simdgroup_half8x8 a_ir;
    simdgroup_half8x8 a_ii;
    simdgroup_half8x8 b_r0;
    simdgroup_half8x8 b_i0;
    simdgroup_half8x8 b_r1;
    simdgroup_half8x8 b_i1;
    simdgroup_float8x8 c_r0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_i0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_r1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_i1 = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (int wb = 0; wb < blocks; ++wb) {
        const int physical_tile = physical_m_tile * blocks + wb;
        const int scale_base = physical_tile * 4;
        const half2 wr = half2(scales[scale_base + 0], scales[scale_base + 2]);
        const half2 wi = half2(scales[scale_base + 1], scales[scale_base + 3]);
        const ulong act_block_base = (ulong) wb * 2 * act_plane_size;
        device const uchar * code_ptr =
            codes + (ulong) physical_tile * 64 * 2 * 16 +
            (ulong) (m16 * 16 + q4_local) * 2 * 16 + (ulong) row_lane;

        FOR_UNROLL (int k_chunk = 0; k_chunk < QK_FAIRY2I_TILE64; k_chunk += 16) {
            const uint u_codes = (uint) code_ptr[0];
            const uint w_codes = (uint) code_ptr[16];

            FOR_UNROLL (int part = 0; part < 4; ++part) {
                const uint2 code = uint2((u_codes >> (2 * part)) & 3, (w_codes >> (2 * part)) & 3);
                const half4 coeff = fairy2i_mma_coeff_w1_codes_scaled_half(code, wr, wi);
                const int coeff_index = coeff_row * 16 + q4_local * 4 + part;
                coeff_real_from_real[coeff_index] = coeff.x;
                coeff_real_from_imag[coeff_index] = coeff.y;
                coeff_imag_from_real[coeff_index] = coeff.z;
                coeff_imag_from_imag[coeff_index] = coeff.w;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (int ik = 0; ik < 2; ++ik) {
                const ulong act_real_base = act_block_base +
                                            (ulong) (k_chunk + ik * 8) *
                                                (ulong) FC_fairy2i_bundle_w1_prefill_act_rows +
                                            (ulong) col_base;
                const ulong act_imag_base = act_real_base + act_plane_size;

                simdgroup_load(a_rr, coeff_real_from_real + coeff_base + ik * 8, 16);
                simdgroup_load(a_ri, coeff_real_from_imag + coeff_base + ik * 8, 16);
                simdgroup_load(a_ir, coeff_imag_from_real + coeff_base + ik * 8, 16);
                simdgroup_load(a_ii, coeff_imag_from_imag + coeff_base + ik * 8, 16);
                simdgroup_load(b_r0, act_h + act_real_base, FC_fairy2i_bundle_w1_prefill_act_rows);
                simdgroup_load(b_i0, act_h + act_imag_base, FC_fairy2i_bundle_w1_prefill_act_rows);
                simdgroup_load(b_r1, act_h + act_real_base + 8, FC_fairy2i_bundle_w1_prefill_act_rows);
                simdgroup_load(b_i1, act_h + act_imag_base + 8, FC_fairy2i_bundle_w1_prefill_act_rows);

                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_multiply_accumulate(c_r0, a_rr, b_r0, c_r0);
                simdgroup_multiply_accumulate(c_r0, a_ri, b_i0, c_r0);
                simdgroup_multiply_accumulate(c_i0, a_ir, b_r0, c_i0);
                simdgroup_multiply_accumulate(c_i0, a_ii, b_i0, c_i0);
                simdgroup_multiply_accumulate(c_r1, a_rr, b_r1, c_r1);
                simdgroup_multiply_accumulate(c_r1, a_ri, b_i1, c_r1);
                simdgroup_multiply_accumulate(c_i1, a_ir, b_r1, c_i1);
                simdgroup_multiply_accumulate(c_i1, a_ii, b_i1, c_i1);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            code_ptr += 4 * 2 * 16;
        }
    }

    const int simdgroup_out_base = (int) sgitg * 256;
    simdgroup_store(c_r0, out_tile + simdgroup_out_base, 8);
    simdgroup_store(c_i0, out_tile + simdgroup_out_base + 64, 8);
    simdgroup_store(c_r1, out_tile + simdgroup_out_base + 128, 8);
    simdgroup_store(c_i1, out_tile + simdgroup_out_base + 192, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tiitg; idx < row_tile * 16; idx += n_threads) {
        const int output_row_lane = (int) idx / 16;
        const int col_local = (int) idx & 15;
        const int tile = col_local >> 3;
        const int col_lane = col_local & 7;
        const int row = row_base + output_row_lane;
        const int col = col_base + col_local;
        if (row < args.m) {
            const int i1 = col % args.x_ne1;
            const int i2 = (col / args.x_ne1) % args.x_ne2;
            const int i3 = col / (args.x_ne1 * args.x_ne2);
            const int row_group = output_row_lane >> 3;
            const int row_in_group = output_row_lane & 7;
            const int out_base = row_group * 256 + tile * 128;
            float out_real = out_tile[out_base + row_in_group * 8 + col_lane];
            float out_imag = out_tile[out_base + 64 + row_in_group * 8 + col_lane];

            if (args.has_bias) {
                const int b0r = row % args.bias_ne0;
                const int b0i = (row + args.m) % args.bias_ne0;
                const int b1 = i1 % args.bias_ne1;
                const int b2 = i2 % args.bias_ne2;
                const int b3 = i3 % args.bias_ne3;
                out_real += *((device const float *) (bias + (ulong) b0r * args.bias_nb0 + (ulong) b1 * args.bias_nb1 +
                                                      (ulong) b2 * args.bias_nb2 + (ulong) b3 * args.bias_nb3));
                out_imag += *((device const float *) (bias + (ulong) b0i * args.bias_nb0 + (ulong) b1 * args.bias_nb1 +
                                                      (ulong) b2 * args.bias_nb2 + (ulong) b3 * args.bias_nb3));
            }

            *((device uint *) (dst + (ulong) row * args.dst_nb0 + (ulong) i1 * args.dst_nb1 + (ulong) i2 * args.dst_nb2 +
                               (ulong) i3 * args.dst_nb3)) = fairy2i_pack_bf16_pair(out_real, out_imag);
        }
    }
}

kernel void kernel_fairy2i_bundle_w2_half_mma32x16(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const uchar * codes                              [[buffer(1)]],
        device const half * scales                              [[buffer(2)]],
        device const half * act_h                               [[buffer(3)]],
        device const char * bias                                [[buffer(4)]],
        device char * dst                                       [[buffer(5)]],
        threadgroup half * coeff_real_from_real                 [[threadgroup(0)]],
        threadgroup half * coeff_real_from_imag                 [[threadgroup(1)]],
        threadgroup half * coeff_imag_from_real                 [[threadgroup(2)]],
        threadgroup half * coeff_imag_from_imag                 [[threadgroup(3)]],
        threadgroup half * act_real_tile0                       [[threadgroup(4)]],
        threadgroup half * act_imag_tile0                       [[threadgroup(5)]],
        threadgroup half * act_real_tile1                       [[threadgroup(6)]],
        threadgroup half * act_imag_tile1                       [[threadgroup(7)]],
        threadgroup float * out_tile                            [[threadgroup(8)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint tiitg                                              [[thread_index_in_threadgroup]],
        uint sgitg                                              [[simdgroup_index_in_threadgroup]]) {
    constexpr int row_tile = 32;
    constexpr int k_tile = 8;
    constexpr int n_threads = 128;

    const int row_base = (int) tgpig.x * row_tile;
    const int col_base = (int) tgpig.y * 16;
    const int blocks = args.k / QK_FAIRY2I_TILE64;
    const int physical_m_tile = row_base / QK_FAIRY2I_TILE64;
    const int coeff_base = (int) sgitg * 64;
    const int coeff_row = (int) tiitg & 31;
    const int q4_local = (int) tiitg >> 5;
    const int row_in_m64 = (row_base + coeff_row) & 63;
    const int m16 = row_in_m64 >> 4;
    const int row_lane = row_in_m64 & 15;

    simdgroup_half8x8 a_rr;
    simdgroup_half8x8 a_ri;
    simdgroup_half8x8 a_ir;
    simdgroup_half8x8 a_ii;
    simdgroup_half8x8 b_r0;
    simdgroup_half8x8 b_i0;
    simdgroup_half8x8 b_r1;
    simdgroup_half8x8 b_i1;
    simdgroup_float8x8 c_r0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_i0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_r1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_i1 = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (int wb = 0; wb < blocks; ++wb) {
        const int physical_tile = physical_m_tile * blocks + wb;
        device const uchar * code_ptr =
            codes + (ulong) physical_tile * 64 * 4 * 16 +
            (ulong) (m16 * 16 + q4_local) * 4 * 16 + (ulong) row_lane;

        for (int k_chunk = 0; k_chunk < QK_FAIRY2I_TILE64; k_chunk += k_tile) {
            // One bundle byte supplies four consecutive K codes. Only 64 threads are needed to expand
            // the complete 32x8 coefficient tile; all 128 threads remain active for staging and MMA.
            if (tiitg < 64) {
                const uint4 packed_codes = uint4(
                    (uint) code_ptr[0],
                    (uint) code_ptr[16],
                    (uint) code_ptr[32],
                    (uint) code_ptr[48]);

                const int scale_base = physical_tile * 8;
                const half4 scale01 = *((device const half4 *) (scales + scale_base));
                const half4 scale23 = *((device const half4 *) (scales + scale_base + 4));
                const float4 wr = float4(scale01.x, scale01.z, scale23.x, scale23.z);
                const float4 wi = float4(scale01.y, scale01.w, scale23.y, scale23.w);

                FOR_UNROLL (int part = 0; part < 4; ++part) {
                    const uint4 code = (packed_codes >> uint4(2 * part)) & uint4(3);
                    const float4 coeff = fairy2i_mma_coeff_w2_codes_scaled(code, wr, wi);
                    const int coeff_index = coeff_row * k_tile + q4_local * 4 + part;
                    coeff_real_from_real[coeff_index] = (half) coeff.x;
                    coeff_real_from_imag[coeff_index] = (half) coeff.y;
                    coeff_imag_from_real[coeff_index] = (half) coeff.z;
                    coeff_imag_from_imag[coeff_index] = (half) coeff.w;
                }
                code_ptr += 2 * 4 * 16;
            }

            for (uint idx = tiitg; idx < 16 * k_tile; idx += n_threads) {
                const int col_local = (int) idx / k_tile;
                const int k_local = (int) idx % k_tile;
                const int tile = col_local >> 3;
                const int col_lane = col_local & 7;
                const int col = col_base + col_local;

                half2 xh = half2(0.0h);
                if (col < args.act_rows) {
                    xh = fairy2i_load_staged_half_activation_pair(act_h, col, wb, blocks, k_chunk + k_local);
                }

                const int act_tile_idx = k_local * 8 + col_lane;
                if (tile == 0) {
                    act_real_tile0[act_tile_idx] = xh.x;
                    act_imag_tile0[act_tile_idx] = xh.y;
                } else {
                    act_real_tile1[act_tile_idx] = xh.x;
                    act_imag_tile1[act_tile_idx] = xh.y;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_load(a_rr, coeff_real_from_real + coeff_base);
            simdgroup_load(a_ri, coeff_real_from_imag + coeff_base);
            simdgroup_load(a_ir, coeff_imag_from_real + coeff_base);
            simdgroup_load(a_ii, coeff_imag_from_imag + coeff_base);
            simdgroup_load(b_r0, act_real_tile0);
            simdgroup_load(b_i0, act_imag_tile0);
            simdgroup_load(b_r1, act_real_tile1);
            simdgroup_load(b_i1, act_imag_tile1);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(c_r0, a_rr, b_r0, c_r0);
            simdgroup_multiply_accumulate(c_r0, a_ri, b_i0, c_r0);
            simdgroup_multiply_accumulate(c_i0, a_ir, b_r0, c_i0);
            simdgroup_multiply_accumulate(c_i0, a_ii, b_i0, c_i0);
            simdgroup_multiply_accumulate(c_r1, a_rr, b_r1, c_r1);
            simdgroup_multiply_accumulate(c_r1, a_ri, b_i1, c_r1);
            simdgroup_multiply_accumulate(c_i1, a_ir, b_r1, c_i1);
            simdgroup_multiply_accumulate(c_i1, a_ii, b_i1, c_i1);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int simdgroup_out_base = (int) sgitg * 256;
    simdgroup_store(c_r0, out_tile + simdgroup_out_base, 8);
    simdgroup_store(c_i0, out_tile + simdgroup_out_base + 64, 8);
    simdgroup_store(c_r1, out_tile + simdgroup_out_base + 128, 8);
    simdgroup_store(c_i1, out_tile + simdgroup_out_base + 192, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tiitg; idx < row_tile * 16; idx += n_threads) {
        const int row_lane = (int) idx / 16;
        const int col_local = (int) idx & 15;
        const int tile = col_local >> 3;
        const int col_lane = col_local & 7;
        const int row = row_base + row_lane;
        const int col = col_base + col_local;
        if (row < args.m && col < args.act_rows) {
            const int i1 = col % args.x_ne1;
            const int i2 = (col / args.x_ne1) % args.x_ne2;
            const int i3 = col / (args.x_ne1 * args.x_ne2);
            const int row_group = row_lane >> 3;
            const int row_in_group = row_lane & 7;
            const int out_base = row_group * 256 + tile * 128;
            float out_real = out_tile[out_base + row_in_group * 8 + col_lane];
            float out_imag = out_tile[out_base + 64 + row_in_group * 8 + col_lane];

            if (args.has_bias) {
                const int b0r = row % args.bias_ne0;
                const int b0i = (row + args.m) % args.bias_ne0;
                const int b1 = i1 % args.bias_ne1;
                const int b2 = i2 % args.bias_ne2;
                const int b3 = i3 % args.bias_ne3;
                out_real += *((device const float *) (bias + (ulong) b0r * args.bias_nb0 + (ulong) b1 * args.bias_nb1 +
                                                      (ulong) b2 * args.bias_nb2 + (ulong) b3 * args.bias_nb3));
                out_imag += *((device const float *) (bias + (ulong) b0i * args.bias_nb0 + (ulong) b1 * args.bias_nb1 +
                                                      (ulong) b2 * args.bias_nb2 + (ulong) b3 * args.bias_nb3));
            }

            *((device uint *) (dst + (ulong) row * args.dst_nb0 + (ulong) i1 * args.dst_nb1 + (ulong) i2 * args.dst_nb2 +
                               (ulong) i3 * args.dst_nb3)) = fairy2i_pack_bf16_pair(out_real, out_imag);
        }
    }
}

#if defined(GGML_METAL_HAS_BF16)
kernel void kernel_fairy2i_bundle_w1_bfloat_bf16scale_exact_mma32x16_k32(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const uchar * codes                              [[buffer(1)]],
        device const ushort * scales                            [[buffer(2)]],
        device const bfloat * act_b                             [[buffer(3)]],
        device const char * bias                                [[buffer(4)]],
        device char * dst                                       [[buffer(5)]],
        device const uint * act_block_metrics                   [[buffer(6)]],
        device const ushort4 * packed_coeff_lut                  [[buffer(7)]],
        threadgroup bfloat * coeff_real_from_real               [[threadgroup(0)]],
        threadgroup bfloat * coeff_real_from_imag               [[threadgroup(1)]],
        threadgroup bfloat * coeff_imag_from_real               [[threadgroup(2)]],
        threadgroup bfloat * coeff_imag_from_imag               [[threadgroup(3)]],
        threadgroup bfloat * act_real_tile0                     [[threadgroup(4)]],
        threadgroup bfloat * act_imag_tile0                     [[threadgroup(5)]],
        threadgroup bfloat * act_real_tile1                     [[threadgroup(6)]],
        threadgroup bfloat * act_imag_tile1                     [[threadgroup(7)]],
        threadgroup float * out_tile                            [[threadgroup(8)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint tiitg                                              [[thread_index_in_threadgroup]],
        uint sgitg                                              [[simdgroup_index_in_threadgroup]]) {
    constexpr int row_tile = 32;
    constexpr int k_tile = 32;
    constexpr int n_threads = 128;

    const int row_base = (int) tgpig.x * row_tile;
    const int col_base = (int) tgpig.y * 16;
    const int blocks = args.k / QK_FAIRY2I_TILE64;
    const int physical_m_tile = row_base / QK_FAIRY2I_TILE64;
    const int coeff_base = (int) sgitg * 8 * k_tile;
    const int coeff_row = (int) tiitg & 31;
    const int q4_local = (int) tiitg >> 5;
    const int row_in_m64 = (row_base + coeff_row) & 63;
    const int m16 = row_in_m64 >> 4;
    const int row_lane = row_in_m64 & 15;
    const int simd_lane = (int) tiitg & 31;
    const ulong packed_coeff_entries =
        (ulong) (args.m / QK_FAIRY2I_TILE64) * (ulong) blocks * 16UL;
    device const uchar * packed_tile_metrics =
        (device const uchar *) (packed_coeff_lut + packed_coeff_entries) + packed_coeff_entries;
    simdgroup_bfloat8x8 a_rr;
    simdgroup_bfloat8x8 a_ri;
    simdgroup_bfloat8x8 a_ir;
    simdgroup_bfloat8x8 a_ii;
    simdgroup_bfloat8x8 b_r0;
    simdgroup_bfloat8x8 b_i0;
    simdgroup_bfloat8x8 b_r1;
    simdgroup_bfloat8x8 b_i1;
    simdgroup_float8x8 c_r0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_i0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_r1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_i1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    uint min_coeff_metric = 255U;
    uint min_act_metric = 255U;

    for (int wb = 0; wb < blocks; ++wb) {
        if (tiitg < 16) {
            const int col = col_base + (int) tiitg;
            if (col < args.act_rows) {
                min_act_metric = min(min_act_metric, act_block_metrics[col * blocks + wb]);
            }
        }
        const int physical_tile = physical_m_tile * blocks + wb;
        if (sgitg == 0U && simd_lane == 0) {
            min_coeff_metric = min(min_coeff_metric, (uint) packed_tile_metrics[physical_tile]);
        }
        device const uchar * code_ptr =
            codes + (ulong) physical_tile * 64 * 2 * 16 +
            (ulong) (m16 * 16 + q4_local) * 2 * 16 + (ulong) row_lane;

        for (int k_chunk = 0; k_chunk < QK_FAIRY2I_TILE64; k_chunk += k_tile) {
            if (tiitg < 128) {
                FOR_UNROLL (int code_group = 0; code_group < 2; ++code_group) {
                    device const uchar * group_code_ptr = code_ptr + code_group * 4 * 2 * 16;
                    const uint2 packed_codes = uint2(
                        (uint) group_code_ptr[0],
                        (uint) group_code_ptr[16]);
                    FOR_UNROLL (int part = 0; part < 4; ++part) {
                        const uint2 code = (packed_codes >> uint2(2 * part)) & uint2(3);
                        const ushort4 coeff =
                            packed_coeff_lut[(ulong) physical_tile * 16UL + (ulong) (code.x | (code.y << 2))];
                        const int coeff_index = coeff_row * k_tile + code_group * 16 + q4_local * 4 + part;
                        ((threadgroup ushort *) coeff_real_from_real)[coeff_index] = coeff.x;
                        ((threadgroup ushort *) coeff_real_from_imag)[coeff_index] = coeff.y;
                        ((threadgroup ushort *) coeff_imag_from_real)[coeff_index] = coeff.z;
                        ((threadgroup ushort *) coeff_imag_from_imag)[coeff_index] = coeff.w;
                    }
                }
                code_ptr += 8 * 2 * 16;
            }

            for (uint idx = tiitg; idx < 16 * k_tile; idx += n_threads) {
                const int col_local = (int) idx / k_tile;
                const int k_local = (int) idx % k_tile;
                const int tile = col_local >> 3;
                const int col_lane = col_local & 7;
                const int col = col_base + col_local;

                ushort xb_real_bits = 0;
                ushort xb_imag_bits = 0;
                if (col < args.act_rows) {
                    const int act_index = col * blocks + wb;
                    const int act_base = act_index * (2 * QK_FAIRY2I_ACT_Q16_64);
                    xb_real_bits = ((device const ushort *) act_b)[act_base + k_chunk + k_local];
                    xb_imag_bits =
                        ((device const ushort *) act_b)[act_base + QK_FAIRY2I_ACT_Q16_64 + k_chunk + k_local];
                }
                const int act_tile_idx = k_local * 8 + col_lane;
                if (tile == 0) {
                    ((threadgroup ushort *) act_real_tile0)[act_tile_idx] = xb_real_bits;
                    ((threadgroup ushort *) act_imag_tile0)[act_tile_idx] = xb_imag_bits;
                } else {
                    ((threadgroup ushort *) act_real_tile1)[act_tile_idx] = xb_real_bits;
                    ((threadgroup ushort *) act_imag_tile1)[act_tile_idx] = xb_imag_bits;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_load(a_rr, coeff_real_from_real + coeff_base, k_tile);
            simdgroup_load(a_ri, coeff_real_from_imag + coeff_base, k_tile);
            simdgroup_load(a_ir, coeff_imag_from_real + coeff_base, k_tile);
            simdgroup_load(a_ii, coeff_imag_from_imag + coeff_base, k_tile);
            simdgroup_load(b_r0, act_real_tile0);
            simdgroup_load(b_i0, act_imag_tile0);
            simdgroup_load(b_r1, act_real_tile1);
            simdgroup_load(b_i1, act_imag_tile1);

            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_multiply_accumulate(c_r0, a_rr, b_r0, c_r0);
            simdgroup_multiply_accumulate(c_r0, a_ri, b_i0, c_r0);
            simdgroup_multiply_accumulate(c_i0, a_ir, b_r0, c_i0);
            simdgroup_multiply_accumulate(c_i0, a_ii, b_i0, c_i0);
            simdgroup_multiply_accumulate(c_r1, a_rr, b_r1, c_r1);
            simdgroup_multiply_accumulate(c_r1, a_ri, b_i1, c_r1);
            simdgroup_multiply_accumulate(c_i1, a_ir, b_r1, c_i1);
            simdgroup_multiply_accumulate(c_i1, a_ii, b_i1, c_i1);

            simdgroup_load(a_rr, coeff_real_from_real + coeff_base + 8, k_tile);
            simdgroup_load(a_ri, coeff_real_from_imag + coeff_base + 8, k_tile);
            simdgroup_load(a_ir, coeff_imag_from_real + coeff_base + 8, k_tile);
            simdgroup_load(a_ii, coeff_imag_from_imag + coeff_base + 8, k_tile);
            simdgroup_load(b_r0, act_real_tile0 + 64);
            simdgroup_load(b_i0, act_imag_tile0 + 64);
            simdgroup_load(b_r1, act_real_tile1 + 64);
            simdgroup_load(b_i1, act_imag_tile1 + 64);

            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_multiply_accumulate(c_r0, a_rr, b_r0, c_r0);
            simdgroup_multiply_accumulate(c_r0, a_ri, b_i0, c_r0);
            simdgroup_multiply_accumulate(c_i0, a_ir, b_r0, c_i0);
            simdgroup_multiply_accumulate(c_i0, a_ii, b_i0, c_i0);
            simdgroup_multiply_accumulate(c_r1, a_rr, b_r1, c_r1);
            simdgroup_multiply_accumulate(c_r1, a_ri, b_i1, c_r1);
            simdgroup_multiply_accumulate(c_i1, a_ir, b_r1, c_i1);
            simdgroup_multiply_accumulate(c_i1, a_ii, b_i1, c_i1);

            simdgroup_load(a_rr, coeff_real_from_real + coeff_base + 16, k_tile);
            simdgroup_load(a_ri, coeff_real_from_imag + coeff_base + 16, k_tile);
            simdgroup_load(a_ir, coeff_imag_from_real + coeff_base + 16, k_tile);
            simdgroup_load(a_ii, coeff_imag_from_imag + coeff_base + 16, k_tile);
            simdgroup_load(b_r0, act_real_tile0 + 128);
            simdgroup_load(b_i0, act_imag_tile0 + 128);
            simdgroup_load(b_r1, act_real_tile1 + 128);
            simdgroup_load(b_i1, act_imag_tile1 + 128);

            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_multiply_accumulate(c_r0, a_rr, b_r0, c_r0);
            simdgroup_multiply_accumulate(c_r0, a_ri, b_i0, c_r0);
            simdgroup_multiply_accumulate(c_i0, a_ir, b_r0, c_i0);
            simdgroup_multiply_accumulate(c_i0, a_ii, b_i0, c_i0);
            simdgroup_multiply_accumulate(c_r1, a_rr, b_r1, c_r1);
            simdgroup_multiply_accumulate(c_r1, a_ri, b_i1, c_r1);
            simdgroup_multiply_accumulate(c_i1, a_ir, b_r1, c_i1);
            simdgroup_multiply_accumulate(c_i1, a_ii, b_i1, c_i1);

            simdgroup_load(a_rr, coeff_real_from_real + coeff_base + 24, k_tile);
            simdgroup_load(a_ri, coeff_real_from_imag + coeff_base + 24, k_tile);
            simdgroup_load(a_ir, coeff_imag_from_real + coeff_base + 24, k_tile);
            simdgroup_load(a_ii, coeff_imag_from_imag + coeff_base + 24, k_tile);
            simdgroup_load(b_r0, act_real_tile0 + 192);
            simdgroup_load(b_i0, act_imag_tile0 + 192);
            simdgroup_load(b_r1, act_real_tile1 + 192);
            simdgroup_load(b_i1, act_imag_tile1 + 192);

            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_multiply_accumulate(c_r0, a_rr, b_r0, c_r0);
            simdgroup_multiply_accumulate(c_r0, a_ri, b_i0, c_r0);
            simdgroup_multiply_accumulate(c_i0, a_ir, b_r0, c_i0);
            simdgroup_multiply_accumulate(c_i0, a_ii, b_i0, c_i0);
            simdgroup_multiply_accumulate(c_r1, a_rr, b_r1, c_r1);
            simdgroup_multiply_accumulate(c_r1, a_ri, b_i1, c_r1);
            simdgroup_multiply_accumulate(c_i1, a_ir, b_r1, c_i1);
            simdgroup_multiply_accumulate(c_i1, a_ii, b_i1, c_i1);
        }
    }

    const uint simd_coeff_metric = simd_min(min_coeff_metric);
    const uint simd_act_metric = simd_min(min_act_metric);
    if (simd_lane == 0) {
        ((threadgroup uint *) out_tile)[sgitg] = simd_coeff_metric;
        ((threadgroup uint *) out_tile)[4 + sgitg] = simd_act_metric;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiitg == 0) {
        uint tile_coeff_metric = 255U;
        uint tile_act_metric = 255U;
        for (int sg = 0; sg < 4; ++sg) {
            tile_coeff_metric = min(tile_coeff_metric, ((threadgroup uint *) out_tile)[sg]);
            tile_act_metric = min(tile_act_metric, ((threadgroup uint *) out_tile)[4 + sg]);
        }
        ((threadgroup uint *) out_tile)[0] = tile_coeff_metric;
        ((threadgroup uint *) out_tile)[1] = tile_act_metric;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool requires_software = fairy2i_product_metrics_require_software(
        ((threadgroup uint *) out_tile)[0], ((threadgroup uint *) out_tile)[1]);
    if (requires_software) {
        for (uint idx = tiitg; idx < row_tile * 16; idx += n_threads) {
            const int output_row_lane = (int) idx / 16;
            const int col_local = (int) idx & 15;
            const int row = row_base + output_row_lane;
            const int col = col_base + col_local;
            if (row < args.m && col < args.act_rows) {
                uint real_bits = 0;
                uint imag_bits = 0;
                for (int k = 0; k < args.k; ++k) {
                    const int wb = k / QK_FAIRY2I_TILE64;
                    const int k_local = k & 63;
                    const int act_index = col * blocks + wb;
                    const int act_base = act_index * (2 * QK_FAIRY2I_TILE64);
                    const ushort xr_bits = ((device const ushort *) act_b)[act_base + k_local];
                    const ushort xi_bits =
                        ((device const ushort *) act_b)[act_base + QK_FAIRY2I_TILE64 + k_local];
                    const ushort4 coeff_bits =
                        fairy2i_bundle_w1_coeff_at_bf16_exact_bits(codes, scales, blocks, row, k);
                    real_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(real_bits, coeff_bits.x, xr_bits);
                    real_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(real_bits, coeff_bits.y, xi_bits);
                    imag_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(imag_bits, coeff_bits.z, xr_bits);
                    imag_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(imag_bits, coeff_bits.w, xi_bits);
                }

                const int i1 = col % args.x_ne1;
                const int i2 = (col / args.x_ne1) % args.x_ne2;
                const int i3 = col / (args.x_ne1 * args.x_ne2);
                ushort out_real_bits;
                ushort out_imag_bits;
                if (args.has_bias) {
                    const int b0r = row % args.bias_ne0;
                    const int b0i = (row + args.m) % args.bias_ne0;
                    const int b1 = i1 % args.bias_ne1;
                    const int b2 = i2 % args.bias_ne2;
                    const int b3 = i3 % args.bias_ne3;
                    const float bias_real = *((device const float *) (bias + (ulong) b0r * args.bias_nb0 +
                                                                      (ulong) b1 * args.bias_nb1 +
                                                                      (ulong) b2 * args.bias_nb2 +
                                                                      (ulong) b3 * args.bias_nb3));
                    const float bias_imag = *((device const float *) (bias + (ulong) b0i * args.bias_nb0 +
                                                                      (ulong) b1 * args.bias_nb1 +
                                                                      (ulong) b2 * args.bias_nb2 +
                                                                      (ulong) b3 * args.bias_nb3));
                    out_real_bits = fairy2i_add_f32_bits_f32_bias_to_bf16_bits_rne(real_bits, bias_real);
                    out_imag_bits = fairy2i_add_f32_bits_f32_bias_to_bf16_bits_rne(imag_bits, bias_imag);
                } else {
                    out_real_bits = fairy2i_f32_to_bf16(as_type<float>(real_bits));
                    out_imag_bits = fairy2i_f32_to_bf16(as_type<float>(imag_bits));
                }
                *((device uint *) (dst + (ulong) row * args.dst_nb0 + (ulong) i1 * args.dst_nb1 +
                                   (ulong) i2 * args.dst_nb2 + (ulong) i3 * args.dst_nb3)) =
                    (uint) out_real_bits | ((uint) out_imag_bits << 16);
            }
        }
        return;
    }

    const int simdgroup_out_base = (int) sgitg * 256;
    simdgroup_store(c_r0, out_tile + simdgroup_out_base, 8);
    simdgroup_store(c_i0, out_tile + simdgroup_out_base + 64, 8);
    simdgroup_store(c_r1, out_tile + simdgroup_out_base + 128, 8);
    simdgroup_store(c_i1, out_tile + simdgroup_out_base + 192, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tiitg; idx < row_tile * 16; idx += n_threads) {
        const int output_row_lane = (int) idx / 16;
        const int col_local = (int) idx & 15;
        const int tile = col_local >> 3;
        const int col_lane = col_local & 7;
        const int row = row_base + output_row_lane;
        const int col = col_base + col_local;
        if (row < args.m && col < args.act_rows) {
            const int i1 = col % args.x_ne1;
            const int i2 = (col / args.x_ne1) % args.x_ne2;
            const int i3 = col / (args.x_ne1 * args.x_ne2);
            const int row_group = output_row_lane >> 3;
            const int row_in_group = output_row_lane & 7;
            const int out_base = row_group * 256 + tile * 128;
            const float out_real = out_tile[out_base + row_in_group * 8 + col_lane];
            const float out_imag = out_tile[out_base + 64 + row_in_group * 8 + col_lane];

            ushort out_real_bits;
            ushort out_imag_bits;
            if (args.has_bias) {
                const int b0r = row % args.bias_ne0;
                const int b0i = (row + args.m) % args.bias_ne0;
                const int b1 = i1 % args.bias_ne1;
                const int b2 = i2 % args.bias_ne2;
                const int b3 = i3 % args.bias_ne3;
                const float bias_real = *((device const float *) (bias + (ulong) b0r * args.bias_nb0 +
                                                                  (ulong) b1 * args.bias_nb1 +
                                                                  (ulong) b2 * args.bias_nb2 +
                                                                  (ulong) b3 * args.bias_nb3));
                const float bias_imag = *((device const float *) (bias + (ulong) b0i * args.bias_nb0 +
                                                                  (ulong) b1 * args.bias_nb1 +
                                                                  (ulong) b2 * args.bias_nb2 +
                                                                  (ulong) b3 * args.bias_nb3));
                out_real_bits = fairy2i_add_f32_bias_to_bf16_bits_rne(out_real, bias_real);
                out_imag_bits = fairy2i_add_f32_bias_to_bf16_bits_rne(out_imag, bias_imag);
            } else {
                out_real_bits = fairy2i_f32_to_bf16(out_real);
                out_imag_bits = fairy2i_f32_to_bf16(out_imag);
            }
            *((device uint *) (dst + (ulong) row * args.dst_nb0 + (ulong) i1 * args.dst_nb1 +
                               (ulong) i2 * args.dst_nb2 + (ulong) i3 * args.dst_nb3)) =
                (uint) out_real_bits | ((uint) out_imag_bits << 16);
        }
    }
}

kernel void kernel_fairy2i_bundle_w1_bfloat_bf16scale_exact_mma32x16_k32_direct_act(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const uchar * codes                              [[buffer(1)]],
        device const ushort * scales                            [[buffer(2)]],
        device const bfloat * act_b                             [[buffer(3)]],
        device const char * bias                                [[buffer(4)]],
        device char * dst                                       [[buffer(5)]],
        device const uint * act_block_metrics                   [[buffer(6)]],
        device const ushort4 * packed_coeff_lut                  [[buffer(7)]],
        threadgroup bfloat * coeff_real_from_real               [[threadgroup(0)]],
        threadgroup bfloat * coeff_real_from_imag               [[threadgroup(1)]],
        threadgroup bfloat * coeff_imag_from_real               [[threadgroup(2)]],
        threadgroup bfloat * coeff_imag_from_imag               [[threadgroup(3)]],
        threadgroup float * out_tile                            [[threadgroup(4)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint tiitg                                              [[thread_index_in_threadgroup]],
        uint sgitg                                              [[simdgroup_index_in_threadgroup]]) {
    constexpr int row_tile = 32;
    constexpr int k_tile = 32;
    constexpr int n_threads = 128;

    const int row_base = (int) tgpig.x * row_tile;
    const int col_tile = FC_fairy2i_bundle_w1_prefill_act_rows <= 8 ? 8 : 16;
    const int col_base = (int) tgpig.y * col_tile;
    const int blocks = args.k / QK_FAIRY2I_TILE64;
    const int physical_m_tile = row_base / QK_FAIRY2I_TILE64;
    const int coeff_base = (int) sgitg * 8 * k_tile;
    const int simd_lane = (int) tiitg & 31;
    const ulong act_plane_size =
        (ulong) QK_FAIRY2I_ACT_Q16_64 * (ulong) FC_fairy2i_bundle_w1_prefill_act_rows;
    const ulong packed_coeff_entries =
        (ulong) (args.m / QK_FAIRY2I_TILE64) * (ulong) blocks * 16UL;
    device const uchar * packed_tile_metrics =
        (device const uchar *) (packed_coeff_lut + packed_coeff_entries) + packed_coeff_entries;
    simdgroup_bfloat8x8 a_rr;
    simdgroup_bfloat8x8 a_ri;
    simdgroup_bfloat8x8 a_ir;
    simdgroup_bfloat8x8 a_ii;
    simdgroup_bfloat8x8 b_r0;
    simdgroup_bfloat8x8 b_i0;
    simdgroup_bfloat8x8 b_r1;
    simdgroup_bfloat8x8 b_i1;
    simdgroup_float8x8 c_r0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_i0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_r1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_i1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    uint min_coeff_metric = 255U;
    uint min_act_metric = 255U;

    for (int wb = 0; wb < blocks; ++wb) {
        if (tiitg < (uint) col_tile) {
            const int col = col_base + (int) tiitg;
            if (col < args.act_rows) {
                min_act_metric = min(min_act_metric, act_block_metrics[col * blocks + wb]);
            }
        }
        const int physical_tile = physical_m_tile * blocks + wb;
        if (sgitg == 0U && simd_lane == 0) {
            min_coeff_metric = min(min_coeff_metric, (uint) packed_tile_metrics[physical_tile]);
        }
        const ulong act_block_base = (ulong) wb * 2 * act_plane_size;

        FOR_UNROLL (int k_chunk = 0; k_chunk < QK_FAIRY2I_TILE64; k_chunk += k_tile) {
            {
                const int coeff_row = (int) sgitg * 8 + (simd_lane >> 2);
                const int q4_local = simd_lane & 3;
                const int row_in_m64 = (row_base + coeff_row) & 63;
                const int m16 = row_in_m64 >> 4;
                const int row_lane = row_in_m64 & 15;
                FOR_UNROLL (int code_group = 0; code_group < 2; ++code_group) {
                    const int q4 = (k_chunk >> 2) + code_group * 4 + q4_local;
                    const int slot = m16 * 16 + q4;
                    const ulong code_base =
                        ((((ulong) physical_tile * 64 + (ulong) slot) * 2) * 16) + (ulong) row_lane;
                    const uint2 packed_codes = uint2(
                        (uint) codes[code_base],
                        (uint) codes[code_base + 16]);

                    FOR_UNROLL (int part = 0; part < 4; ++part) {
                        const uint2 code = (packed_codes >> uint2(2 * part)) & uint2(3);
                        const ushort4 coeff =
                            packed_coeff_lut[(ulong) physical_tile * 16UL + (ulong) (code.x | (code.y << 2))];
                        const int coeff_index = coeff_row * k_tile + code_group * 16 + q4_local * 4 + part;
                        ((threadgroup ushort *) coeff_real_from_real)[coeff_index] = coeff.x;
                        ((threadgroup ushort *) coeff_real_from_imag)[coeff_index] = coeff.y;
                        ((threadgroup ushort *) coeff_imag_from_real)[coeff_index] = coeff.z;
                        ((threadgroup ushort *) coeff_imag_from_imag)[coeff_index] = coeff.w;
                    }
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            const ulong act_real_base =
                act_block_base + (ulong) k_chunk * (ulong) FC_fairy2i_bundle_w1_prefill_act_rows +
                (ulong) col_base;
            const ulong act_imag_base = act_real_base + act_plane_size;
            simdgroup_load(a_rr, coeff_real_from_real + coeff_base, k_tile);
            simdgroup_load(a_ri, coeff_real_from_imag + coeff_base, k_tile);
            simdgroup_load(a_ir, coeff_imag_from_real + coeff_base, k_tile);
            simdgroup_load(a_ii, coeff_imag_from_imag + coeff_base, k_tile);
            simdgroup_load(
                b_r0, act_b + act_real_base, FC_fairy2i_bundle_w1_prefill_act_rows);
            simdgroup_load(
                b_i0, act_b + act_imag_base, FC_fairy2i_bundle_w1_prefill_act_rows);
            if (col_tile > 8) {
                simdgroup_load(
                    b_r1, act_b + act_real_base + 8, FC_fairy2i_bundle_w1_prefill_act_rows);
                simdgroup_load(
                    b_i1, act_b + act_imag_base + 8, FC_fairy2i_bundle_w1_prefill_act_rows);
            }

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(c_r0, a_rr, b_r0, c_r0);
            simdgroup_multiply_accumulate(c_r0, a_ri, b_i0, c_r0);
            simdgroup_multiply_accumulate(c_i0, a_ir, b_r0, c_i0);
            simdgroup_multiply_accumulate(c_i0, a_ii, b_i0, c_i0);
            if (col_tile > 8) {
                simdgroup_multiply_accumulate(c_r1, a_rr, b_r1, c_r1);
                simdgroup_multiply_accumulate(c_r1, a_ri, b_i1, c_r1);
                simdgroup_multiply_accumulate(c_i1, a_ir, b_r1, c_i1);
                simdgroup_multiply_accumulate(c_i1, a_ii, b_i1, c_i1);
            }

            simdgroup_load(a_rr, coeff_real_from_real + coeff_base + 8, k_tile);
            simdgroup_load(a_ri, coeff_real_from_imag + coeff_base + 8, k_tile);
            simdgroup_load(a_ir, coeff_imag_from_real + coeff_base + 8, k_tile);
            simdgroup_load(a_ii, coeff_imag_from_imag + coeff_base + 8, k_tile);
            simdgroup_load(
                b_r0, act_b + act_real_base + 8 * FC_fairy2i_bundle_w1_prefill_act_rows,
                FC_fairy2i_bundle_w1_prefill_act_rows);
            simdgroup_load(
                b_i0, act_b + act_imag_base + 8 * FC_fairy2i_bundle_w1_prefill_act_rows,
                FC_fairy2i_bundle_w1_prefill_act_rows);
            if (col_tile > 8) {
                simdgroup_load(
                    b_r1, act_b + act_real_base + 8 * FC_fairy2i_bundle_w1_prefill_act_rows + 8,
                    FC_fairy2i_bundle_w1_prefill_act_rows);
                simdgroup_load(
                    b_i1, act_b + act_imag_base + 8 * FC_fairy2i_bundle_w1_prefill_act_rows + 8,
                    FC_fairy2i_bundle_w1_prefill_act_rows);
            }

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(c_r0, a_rr, b_r0, c_r0);
            simdgroup_multiply_accumulate(c_r0, a_ri, b_i0, c_r0);
            simdgroup_multiply_accumulate(c_i0, a_ir, b_r0, c_i0);
            simdgroup_multiply_accumulate(c_i0, a_ii, b_i0, c_i0);
            if (col_tile > 8) {
                simdgroup_multiply_accumulate(c_r1, a_rr, b_r1, c_r1);
                simdgroup_multiply_accumulate(c_r1, a_ri, b_i1, c_r1);
                simdgroup_multiply_accumulate(c_i1, a_ir, b_r1, c_i1);
                simdgroup_multiply_accumulate(c_i1, a_ii, b_i1, c_i1);
            }

            simdgroup_load(a_rr, coeff_real_from_real + coeff_base + 16, k_tile);
            simdgroup_load(a_ri, coeff_real_from_imag + coeff_base + 16, k_tile);
            simdgroup_load(a_ir, coeff_imag_from_real + coeff_base + 16, k_tile);
            simdgroup_load(a_ii, coeff_imag_from_imag + coeff_base + 16, k_tile);
            simdgroup_load(
                b_r0, act_b + act_real_base + 16 * FC_fairy2i_bundle_w1_prefill_act_rows,
                FC_fairy2i_bundle_w1_prefill_act_rows);
            simdgroup_load(
                b_i0, act_b + act_imag_base + 16 * FC_fairy2i_bundle_w1_prefill_act_rows,
                FC_fairy2i_bundle_w1_prefill_act_rows);
            if (col_tile > 8) {
                simdgroup_load(
                    b_r1, act_b + act_real_base + 16 * FC_fairy2i_bundle_w1_prefill_act_rows + 8,
                    FC_fairy2i_bundle_w1_prefill_act_rows);
                simdgroup_load(
                    b_i1, act_b + act_imag_base + 16 * FC_fairy2i_bundle_w1_prefill_act_rows + 8,
                    FC_fairy2i_bundle_w1_prefill_act_rows);
            }

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(c_r0, a_rr, b_r0, c_r0);
            simdgroup_multiply_accumulate(c_r0, a_ri, b_i0, c_r0);
            simdgroup_multiply_accumulate(c_i0, a_ir, b_r0, c_i0);
            simdgroup_multiply_accumulate(c_i0, a_ii, b_i0, c_i0);
            if (col_tile > 8) {
                simdgroup_multiply_accumulate(c_r1, a_rr, b_r1, c_r1);
                simdgroup_multiply_accumulate(c_r1, a_ri, b_i1, c_r1);
                simdgroup_multiply_accumulate(c_i1, a_ir, b_r1, c_i1);
                simdgroup_multiply_accumulate(c_i1, a_ii, b_i1, c_i1);
            }

            simdgroup_load(a_rr, coeff_real_from_real + coeff_base + 24, k_tile);
            simdgroup_load(a_ri, coeff_real_from_imag + coeff_base + 24, k_tile);
            simdgroup_load(a_ir, coeff_imag_from_real + coeff_base + 24, k_tile);
            simdgroup_load(a_ii, coeff_imag_from_imag + coeff_base + 24, k_tile);
            simdgroup_load(
                b_r0, act_b + act_real_base + 24 * FC_fairy2i_bundle_w1_prefill_act_rows,
                FC_fairy2i_bundle_w1_prefill_act_rows);
            simdgroup_load(
                b_i0, act_b + act_imag_base + 24 * FC_fairy2i_bundle_w1_prefill_act_rows,
                FC_fairy2i_bundle_w1_prefill_act_rows);
            if (col_tile > 8) {
                simdgroup_load(
                    b_r1, act_b + act_real_base + 24 * FC_fairy2i_bundle_w1_prefill_act_rows + 8,
                    FC_fairy2i_bundle_w1_prefill_act_rows);
                simdgroup_load(
                    b_i1, act_b + act_imag_base + 24 * FC_fairy2i_bundle_w1_prefill_act_rows + 8,
                    FC_fairy2i_bundle_w1_prefill_act_rows);
            }

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(c_r0, a_rr, b_r0, c_r0);
            simdgroup_multiply_accumulate(c_r0, a_ri, b_i0, c_r0);
            simdgroup_multiply_accumulate(c_i0, a_ir, b_r0, c_i0);
            simdgroup_multiply_accumulate(c_i0, a_ii, b_i0, c_i0);
            if (col_tile > 8) {
                simdgroup_multiply_accumulate(c_r1, a_rr, b_r1, c_r1);
                simdgroup_multiply_accumulate(c_r1, a_ri, b_i1, c_r1);
                simdgroup_multiply_accumulate(c_i1, a_ir, b_r1, c_i1);
                simdgroup_multiply_accumulate(c_i1, a_ii, b_i1, c_i1);
            }
        }
    }

    const uint simd_coeff_metric = simd_min(min_coeff_metric);
    const uint simd_act_metric = simd_min(min_act_metric);
    if (simd_lane == 0) {
        ((threadgroup uint *) out_tile)[sgitg] = simd_coeff_metric;
        ((threadgroup uint *) out_tile)[4 + sgitg] = simd_act_metric;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiitg == 0) {
        uint tile_coeff_metric = 255U;
        uint tile_act_metric = 255U;
        for (int sg = 0; sg < 4; ++sg) {
            tile_coeff_metric = min(tile_coeff_metric, ((threadgroup uint *) out_tile)[sg]);
            tile_act_metric = min(tile_act_metric, ((threadgroup uint *) out_tile)[4 + sg]);
        }
        ((threadgroup uint *) out_tile)[0] = tile_coeff_metric;
        ((threadgroup uint *) out_tile)[1] = tile_act_metric;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool requires_software = fairy2i_product_metrics_require_software(
        ((threadgroup uint *) out_tile)[0], ((threadgroup uint *) out_tile)[1]);
    if (requires_software) {
        for (uint idx = tiitg; idx < row_tile * col_tile; idx += n_threads) {
            const int output_row_lane = (int) idx / col_tile;
            const int col_local = (int) idx % col_tile;
            const int row = row_base + output_row_lane;
            const int col = col_base + col_local;
            if (row < args.m && col < args.act_rows) {
                uint real_bits = 0;
                uint imag_bits = 0;
                for (int k = 0; k < args.k; ++k) {
                    const int wb = k / QK_FAIRY2I_TILE64;
                    const int k_local = k & 63;
                    const ulong block_base = (ulong) wb * 2 * act_plane_size;
                    const ulong real_index =
                        block_base + (ulong) k_local * (ulong) FC_fairy2i_bundle_w1_prefill_act_rows +
                        (ulong) col;
                    const ushort xr_bits = ((device const ushort *) act_b)[real_index];
                    const ushort xi_bits = ((device const ushort *) act_b)[real_index + act_plane_size];
                    const ushort4 coeff_bits =
                        fairy2i_bundle_w1_coeff_at_bf16_exact_bits(codes, scales, blocks, row, k);
                    real_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(real_bits, coeff_bits.x, xr_bits);
                    real_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(real_bits, coeff_bits.y, xi_bits);
                    imag_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(imag_bits, coeff_bits.z, xr_bits);
                    imag_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(imag_bits, coeff_bits.w, xi_bits);
                }

                const int i1 = col % args.x_ne1;
                const int i2 = (col / args.x_ne1) % args.x_ne2;
                const int i3 = col / (args.x_ne1 * args.x_ne2);
                ushort out_real_bits;
                ushort out_imag_bits;
                if (args.has_bias) {
                    const int b0r = row % args.bias_ne0;
                    const int b0i = (row + args.m) % args.bias_ne0;
                    const int b1 = i1 % args.bias_ne1;
                    const int b2 = i2 % args.bias_ne2;
                    const int b3 = i3 % args.bias_ne3;
                    const float bias_real = *((device const float *) (bias + (ulong) b0r * args.bias_nb0 +
                                                                      (ulong) b1 * args.bias_nb1 +
                                                                      (ulong) b2 * args.bias_nb2 +
                                                                      (ulong) b3 * args.bias_nb3));
                    const float bias_imag = *((device const float *) (bias + (ulong) b0i * args.bias_nb0 +
                                                                      (ulong) b1 * args.bias_nb1 +
                                                                      (ulong) b2 * args.bias_nb2 +
                                                                      (ulong) b3 * args.bias_nb3));
                    out_real_bits = fairy2i_add_f32_bits_f32_bias_to_bf16_bits_rne(real_bits, bias_real);
                    out_imag_bits = fairy2i_add_f32_bits_f32_bias_to_bf16_bits_rne(imag_bits, bias_imag);
                } else {
                    out_real_bits = fairy2i_f32_to_bf16(as_type<float>(real_bits));
                    out_imag_bits = fairy2i_f32_to_bf16(as_type<float>(imag_bits));
                }
                *((device uint *) (dst + (ulong) row * args.dst_nb0 + (ulong) i1 * args.dst_nb1 +
                                   (ulong) i2 * args.dst_nb2 + (ulong) i3 * args.dst_nb3)) =
                    (uint) out_real_bits | ((uint) out_imag_bits << 16);
            }
        }
        return;
    }

    const int simdgroup_out_stride = col_tile * 16;
    const int simdgroup_out_base = (int) sgitg * simdgroup_out_stride;
    simdgroup_store(c_r0, out_tile + simdgroup_out_base, 8);
    simdgroup_store(c_i0, out_tile + simdgroup_out_base + 64, 8);
    if (col_tile > 8) {
        simdgroup_store(c_r1, out_tile + simdgroup_out_base + 128, 8);
        simdgroup_store(c_i1, out_tile + simdgroup_out_base + 192, 8);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tiitg; idx < row_tile * col_tile; idx += n_threads) {
        const int output_row_lane = (int) idx / col_tile;
        const int col_local = (int) idx % col_tile;
        const int tile = col_local >> 3;
        const int col_lane = col_local & 7;
        const int row = row_base + output_row_lane;
        const int col = col_base + col_local;
        if (row < args.m && col < args.act_rows) {
            const int i1 = col % args.x_ne1;
            const int i2 = (col / args.x_ne1) % args.x_ne2;
            const int i3 = col / (args.x_ne1 * args.x_ne2);
            const int row_group = output_row_lane >> 3;
            const int row_in_group = output_row_lane & 7;
            const int out_base = row_group * simdgroup_out_stride + tile * 128;
            const float out_real = out_tile[out_base + row_in_group * 8 + col_lane];
            const float out_imag = out_tile[out_base + 64 + row_in_group * 8 + col_lane];

            ushort out_real_bits;
            ushort out_imag_bits;
            if (args.has_bias) {
                const int b0r = row % args.bias_ne0;
                const int b0i = (row + args.m) % args.bias_ne0;
                const int b1 = i1 % args.bias_ne1;
                const int b2 = i2 % args.bias_ne2;
                const int b3 = i3 % args.bias_ne3;
                const float bias_real = *((device const float *) (bias + (ulong) b0r * args.bias_nb0 +
                                                                  (ulong) b1 * args.bias_nb1 +
                                                                  (ulong) b2 * args.bias_nb2 +
                                                                  (ulong) b3 * args.bias_nb3));
                const float bias_imag = *((device const float *) (bias + (ulong) b0i * args.bias_nb0 +
                                                                  (ulong) b1 * args.bias_nb1 +
                                                                  (ulong) b2 * args.bias_nb2 +
                                                                  (ulong) b3 * args.bias_nb3));
                out_real_bits = fairy2i_add_f32_bias_to_bf16_bits_rne(out_real, bias_real);
                out_imag_bits = fairy2i_add_f32_bias_to_bf16_bits_rne(out_imag, bias_imag);
            } else {
                out_real_bits = fairy2i_f32_to_bf16(out_real);
                out_imag_bits = fairy2i_f32_to_bf16(out_imag);
            }
            *((device uint *) (dst + (ulong) row * args.dst_nb0 + (ulong) i1 * args.dst_nb1 +
                               (ulong) i2 * args.dst_nb2 + (ulong) i3 * args.dst_nb3)) =
                (uint) out_real_bits | ((uint) out_imag_bits << 16);
        }
    }
}

kernel void kernel_fairy2i_bundle_w2_bfloat_bf16scale_exact_mma32x16(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const uchar * codes                              [[buffer(1)]],
        device const ushort * scales                            [[buffer(2)]],
        device const bfloat * act_b                             [[buffer(3)]],
        device const char * bias                                [[buffer(4)]],
        device char * dst                                       [[buffer(5)]],
        threadgroup bfloat * coeff_real_from_real               [[threadgroup(0)]],
        threadgroup bfloat * coeff_real_from_imag               [[threadgroup(1)]],
        threadgroup bfloat * coeff_imag_from_real               [[threadgroup(2)]],
        threadgroup bfloat * coeff_imag_from_imag               [[threadgroup(3)]],
        threadgroup bfloat * act_real_tile0                     [[threadgroup(4)]],
        threadgroup bfloat * act_imag_tile0                     [[threadgroup(5)]],
        threadgroup bfloat * act_real_tile1                     [[threadgroup(6)]],
        threadgroup bfloat * act_imag_tile1                     [[threadgroup(7)]],
        threadgroup float * out_tile                            [[threadgroup(8)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint tiitg                                              [[thread_index_in_threadgroup]],
        uint sgitg                                              [[simdgroup_index_in_threadgroup]]) {
    constexpr int row_tile = 32;
    constexpr int k_tile = 8;
    constexpr int n_threads = 128;

    const int row_base = (int) tgpig.x * row_tile;
    const int col_base = (int) tgpig.y * 16;
    const int blocks = args.k / QK_FAIRY2I_TILE64;
    const int physical_m_tile = row_base / QK_FAIRY2I_TILE64;
    const int coeff_base = (int) sgitg * 64;
    const int coeff_row = (int) tiitg & 31;
    const int q4_local = (int) tiitg >> 5;
    const int row_in_m64 = (row_base + coeff_row) & 63;
    const int m16 = row_in_m64 >> 4;
    const int row_lane = row_in_m64 & 15;
    const int simd_lane       = (int) tiitg & 31;

    simdgroup_bfloat8x8 a_rr;
    simdgroup_bfloat8x8 a_ri;
    simdgroup_bfloat8x8 a_ir;
    simdgroup_bfloat8x8 a_ii;
    simdgroup_bfloat8x8 b_r0;
    simdgroup_bfloat8x8 b_i0;
    simdgroup_bfloat8x8 b_r1;
    simdgroup_bfloat8x8 b_i1;
    simdgroup_float8x8 c_r0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_i0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_r1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_i1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    uint min_coeff_metric = 255U;
    uint min_act_metric = 255U;
    threadgroup ushort4 * coeff_lut = (threadgroup ushort4 *) out_tile;
    threadgroup ushort * scale_staging = ((threadgroup ushort *) out_tile) + 1024;

    for (int wb = 0; wb < blocks; ++wb) {
        const int physical_tile = physical_m_tile * blocks + wb;
        if (tiitg == 0) {
            const int scale_base = physical_tile * 8;
            *((threadgroup ushort4 *) scale_staging) =
                *((device const ushort4 *) (scales + scale_base));
            *((threadgroup ushort4 *) (scale_staging + 4)) =
                *((device const ushort4 *) (scales + scale_base + 4));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort4 scale01 = *((threadgroup ushort4 *) scale_staging);
        const ushort4 scale23 = *((threadgroup ushort4 *) (scale_staging + 4));
        const ushort4 scale_real = ushort4(scale01.x, scale01.z, scale23.x, scale23.z);
        const ushort4 scale_imag = ushort4(scale01.y, scale01.w, scale23.y, scale23.w);
        if (sgitg == 0U) {
            uint stage_metric       = 255U;
            uint stage_max_exponent = 0U;
            if (simd_lane < 8) {
                const uint branch = (uint) simd_lane >> 1;
                const ushort scale = (simd_lane & 1) == 0 ? scale_real[branch] : scale_imag[branch];
                stage_metric       = fairy2i_bf16_product_metric(scale);
                stage_max_exponent = stage_metric == 255U ? 0U : stage_metric;
            }
            const uint min_stage_metric   = simd_min(stage_metric);
            const uint max_stage_exponent = simd_max(stage_max_exponent);
            if (simd_lane == 0) {
                const uint block_coeff_metric =
                    fairy2i_bf16_two_add_coefficient_metric_bound(min_stage_metric, max_stage_exponent);
                min_coeff_metric = min(min_coeff_metric, block_coeff_metric);
            }
        }
        fairy2i_build_bundle_w2_coeff_lut_bf16_exact(
            coeff_lut, scale_real, scale_imag, tiitg, n_threads,
            args.strict_staged_reconstruction != 0);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const uchar * code_ptr =
            codes + (ulong) physical_tile * 64 * 4 * 16 +
            (ulong) (m16 * 16 + q4_local) * 4 * 16 + (ulong) row_lane;

        for (int k_chunk = 0; k_chunk < QK_FAIRY2I_TILE64; k_chunk += k_tile) {
            if (tiitg < 64) {
                const uint4 packed_codes = uint4(
                    (uint) code_ptr[0],
                    (uint) code_ptr[16],
                    (uint) code_ptr[32],
                    (uint) code_ptr[48]);

                FOR_UNROLL (int part = 0; part < 4; ++part) {
                    const uint4 code = (packed_codes >> uint4(2 * part)) & uint4(3);
                    const ushort4 coeff = coeff_lut[fairy2i_bundle_w2_coeff_lut_index(code)];
                    const int coeff_index = coeff_row * k_tile + q4_local * 4 + part;
                    ((threadgroup ushort *) coeff_real_from_real)[coeff_index] = coeff.x;
                    ((threadgroup ushort *) coeff_real_from_imag)[coeff_index] = coeff.y;
                    ((threadgroup ushort *) coeff_imag_from_real)[coeff_index] = coeff.z;
                    ((threadgroup ushort *) coeff_imag_from_imag)[coeff_index] = coeff.w;
                }
                code_ptr += 2 * 4 * 16;
            }

            for (uint idx = tiitg; idx < 16 * k_tile; idx += n_threads) {
                const int col_local = (int) idx / k_tile;
                const int k_local = (int) idx % k_tile;
                const int tile = col_local >> 3;
                const int col_lane = col_local & 7;
                const int col = col_base + col_local;

                ushort xb_real_bits = 0;
                ushort xb_imag_bits = 0;
                if (col < args.act_rows) {
                    const int act_index = col * blocks + wb;
                    const int act_base = act_index * (2 * QK_FAIRY2I_ACT_Q16_64);
                    xb_real_bits = ((device const ushort *) act_b)[act_base + k_chunk + k_local];
                    xb_imag_bits =
                        ((device const ushort *) act_b)[act_base + QK_FAIRY2I_ACT_Q16_64 + k_chunk + k_local];
                }

                const int act_tile_idx = k_local * 8 + col_lane;
                if (tile == 0) {
                    ((threadgroup ushort *) act_real_tile0)[act_tile_idx] = xb_real_bits;
                    ((threadgroup ushort *) act_imag_tile0)[act_tile_idx] = xb_imag_bits;
                } else {
                    ((threadgroup ushort *) act_real_tile1)[act_tile_idx] = xb_real_bits;
                    ((threadgroup ushort *) act_imag_tile1)[act_tile_idx] = xb_imag_bits;
                }
                min_act_metric = min(min_act_metric, fairy2i_bf16_product_metric(xb_real_bits));
                min_act_metric = min(min_act_metric, fairy2i_bf16_product_metric(xb_imag_bits));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_load(a_rr, coeff_real_from_real + coeff_base);
            simdgroup_load(a_ri, coeff_real_from_imag + coeff_base);
            simdgroup_load(a_ir, coeff_imag_from_real + coeff_base);
            simdgroup_load(a_ii, coeff_imag_from_imag + coeff_base);
            simdgroup_load(b_r0, act_real_tile0);
            simdgroup_load(b_i0, act_imag_tile0);
            simdgroup_load(b_r1, act_real_tile1);
            simdgroup_load(b_i1, act_imag_tile1);

            // Once every SIMD-group has loaded the shared A/B tiles into its matrix registers, the
            // threadgroup staging can be reused by the next K8 chunk while slower groups finish MMA.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_multiply_accumulate(c_r0, a_rr, b_r0, c_r0);
            simdgroup_multiply_accumulate(c_r0, a_ri, b_i0, c_r0);
            simdgroup_multiply_accumulate(c_i0, a_ir, b_r0, c_i0);
            simdgroup_multiply_accumulate(c_i0, a_ii, b_i0, c_i0);
            simdgroup_multiply_accumulate(c_r1, a_rr, b_r1, c_r1);
            simdgroup_multiply_accumulate(c_r1, a_ri, b_i1, c_r1);
            simdgroup_multiply_accumulate(c_i1, a_ir, b_r1, c_i1);
            simdgroup_multiply_accumulate(c_i1, a_ii, b_i1, c_i1);
        }
    }

    const uint simd_coeff_metric = simd_min(min_coeff_metric);
    const uint simd_act_metric = simd_min(min_act_metric);
    if (simd_lane == 0) {
        ((threadgroup uint *) out_tile)[sgitg] = simd_coeff_metric;
        ((threadgroup uint *) out_tile)[4 + sgitg] = simd_act_metric;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiitg == 0) {
        uint tile_coeff_metric = 255U;
        uint tile_act_metric = 255U;
        for (int sg = 0; sg < 4; ++sg) {
            tile_coeff_metric = min(tile_coeff_metric, ((threadgroup uint *) out_tile)[sg]);
            tile_act_metric = min(tile_act_metric, ((threadgroup uint *) out_tile)[4 + sg]);
        }
        ((threadgroup uint *) out_tile)[0] = tile_coeff_metric;
        ((threadgroup uint *) out_tile)[1] = tile_act_metric;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool requires_software = fairy2i_product_metrics_require_software(
        ((threadgroup uint *) out_tile)[0], ((threadgroup uint *) out_tile)[1]);
    if (requires_software) {
        for (uint idx = tiitg; idx < row_tile * 16; idx += n_threads) {
            const int output_row_lane = (int) idx / 16;
            const int col_local = (int) idx & 15;
            const int row = row_base + output_row_lane;
            const int col = col_base + col_local;
            if (row < args.m && col < args.act_rows) {
                uint real_bits = 0;
                uint imag_bits = 0;
                for (int k = 0; k < args.k; ++k) {
                    const int wb = k / QK_FAIRY2I_TILE64;
                    const int k_local = k & 63;
                    const int act_index = col * blocks + wb;
                    const int act_base = act_index * (2 * QK_FAIRY2I_TILE64);
                    const ushort xr_bits = ((device const ushort *) act_b)[act_base + k_local];
                    const ushort xi_bits =
                        ((device const ushort *) act_b)[act_base + QK_FAIRY2I_TILE64 + k_local];
                    const ushort4 coeff_bits =
                        args.strict_staged_reconstruction != 0 ?
                            fairy2i_bundle_w2_coeff_at_bf16_exact_bits(
                                codes, scales, blocks, row, k) :
                            fairy2i_bundle_w2_coeff_at_bf16_collapsed_f32_bits(
                                codes, scales, blocks, row, k);
                    real_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(real_bits, coeff_bits.x, xr_bits);
                    real_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(real_bits, coeff_bits.y, xi_bits);
                    imag_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(imag_bits, coeff_bits.z, xr_bits);
                    imag_bits = fairy2i_accumulate_bf16_product_f32_bits_rne(imag_bits, coeff_bits.w, xi_bits);
                }

                const int i1 = col % args.x_ne1;
                const int i2 = (col / args.x_ne1) % args.x_ne2;
                const int i3 = col / (args.x_ne1 * args.x_ne2);
                ushort out_real_bits;
                ushort out_imag_bits;
                if (args.has_bias) {
                    const int b0r = row % args.bias_ne0;
                    const int b0i = (row + args.m) % args.bias_ne0;
                    const int b1 = i1 % args.bias_ne1;
                    const int b2 = i2 % args.bias_ne2;
                    const int b3 = i3 % args.bias_ne3;
                    const float bias_real = *((device const float *) (bias + (ulong) b0r * args.bias_nb0 +
                                                                      (ulong) b1 * args.bias_nb1 +
                                                                      (ulong) b2 * args.bias_nb2 +
                                                                      (ulong) b3 * args.bias_nb3));
                    const float bias_imag = *((device const float *) (bias + (ulong) b0i * args.bias_nb0 +
                                                                      (ulong) b1 * args.bias_nb1 +
                                                                      (ulong) b2 * args.bias_nb2 +
                                                                      (ulong) b3 * args.bias_nb3));
                    out_real_bits = fairy2i_add_f32_bits_f32_bias_to_bf16_bits_rne(real_bits, bias_real);
                    out_imag_bits = fairy2i_add_f32_bits_f32_bias_to_bf16_bits_rne(imag_bits, bias_imag);
                } else {
                    out_real_bits = fairy2i_f32_to_bf16(as_type<float>(real_bits));
                    out_imag_bits = fairy2i_f32_to_bf16(as_type<float>(imag_bits));
                }
                *((device uint *) (dst + (ulong) row * args.dst_nb0 + (ulong) i1 * args.dst_nb1 +
                                   (ulong) i2 * args.dst_nb2 + (ulong) i3 * args.dst_nb3)) =
                    (uint) out_real_bits | ((uint) out_imag_bits << 16);
            }
        }
        return;
    }

    const int simdgroup_out_base = (int) sgitg * 256;
    simdgroup_store(c_r0, out_tile + simdgroup_out_base, 8);
    simdgroup_store(c_i0, out_tile + simdgroup_out_base + 64, 8);
    simdgroup_store(c_r1, out_tile + simdgroup_out_base + 128, 8);
    simdgroup_store(c_i1, out_tile + simdgroup_out_base + 192, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tiitg; idx < row_tile * 16; idx += n_threads) {
        const int output_row_lane = (int) idx / 16;
        const int col_local = (int) idx & 15;
        const int tile = col_local >> 3;
        const int col_lane = col_local & 7;
        const int row = row_base + output_row_lane;
        const int col = col_base + col_local;
        if (row < args.m && col < args.act_rows) {
            const int i1 = col % args.x_ne1;
            const int i2 = (col / args.x_ne1) % args.x_ne2;
            const int i3 = col / (args.x_ne1 * args.x_ne2);
            const int row_group = output_row_lane >> 3;
            const int row_in_group = output_row_lane & 7;
            const int out_base = row_group * 256 + tile * 128;
            float out_real = out_tile[out_base + row_in_group * 8 + col_lane];
            float out_imag = out_tile[out_base + 64 + row_in_group * 8 + col_lane];

            ushort out_real_bits;
            ushort out_imag_bits;
            if (args.has_bias) {
                const int b0r = row % args.bias_ne0;
                const int b0i = (row + args.m) % args.bias_ne0;
                const int b1 = i1 % args.bias_ne1;
                const int b2 = i2 % args.bias_ne2;
                const int b3 = i3 % args.bias_ne3;
                const float bias_real = *((device const float *) (bias + (ulong) b0r * args.bias_nb0 +
                                                                  (ulong) b1 * args.bias_nb1 +
                                                                  (ulong) b2 * args.bias_nb2 +
                                                                  (ulong) b3 * args.bias_nb3));
                const float bias_imag = *((device const float *) (bias + (ulong) b0i * args.bias_nb0 +
                                                                  (ulong) b1 * args.bias_nb1 +
                                                                  (ulong) b2 * args.bias_nb2 +
                                                                  (ulong) b3 * args.bias_nb3));
                out_real_bits = fairy2i_add_f32_bias_to_bf16_bits_rne(out_real, bias_real);
                out_imag_bits = fairy2i_add_f32_bias_to_bf16_bits_rne(out_imag, bias_imag);
            } else {
                out_real_bits = fairy2i_f32_to_bf16(out_real);
                out_imag_bits = fairy2i_f32_to_bf16(out_imag);
            }

            *((device uint *) (dst + (ulong) row * args.dst_nb0 + (ulong) i1 * args.dst_nb1 +
                               (ulong) i2 * args.dst_nb2 + (ulong) i3 * args.dst_nb3)) =
                (uint) out_real_bits | ((uint) out_imag_bits << 16);
        }
    }
}

#endif

template<int row_tile, int k_tile>
kernel void kernel_fairy2i_wide_linear_w2_half_w64scale_mma_rows(
        constant ggml_metal_kargs_fairy2i_wide_linear_w2 & args [[buffer(0)]],
        device const block_fairy2i_tile64_v2 * u_s0             [[buffer(1)]],
        device const block_fairy2i_tile64_v2 * u_s1             [[buffer(2)]],
        device const block_fairy2i_tile64_v2 * w_s0             [[buffer(3)]],
        device const block_fairy2i_tile64_v2 * w_s1             [[buffer(4)]],
        device const half * act_h                               [[buffer(5)]],
        device const char * bias                                [[buffer(6)]],
        device char * dst                                       [[buffer(7)]],
        threadgroup half * coeff_real_from_real                 [[threadgroup(0)]],
        threadgroup half * coeff_real_from_imag                 [[threadgroup(1)]],
        threadgroup half * coeff_imag_from_real                 [[threadgroup(2)]],
        threadgroup half * coeff_imag_from_imag                 [[threadgroup(3)]],
        threadgroup half * act_real_tile0                       [[threadgroup(4)]],
        threadgroup half * act_imag_tile0                       [[threadgroup(5)]],
        threadgroup half * act_real_tile1                       [[threadgroup(6)]],
        threadgroup half * act_imag_tile1                       [[threadgroup(7)]],
        threadgroup float * out_tile                            [[threadgroup(8)]],
        uint2 tgpig                                             [[threadgroup_position_in_grid]],
        uint tiitg                                              [[thread_index_in_threadgroup]],
        uint sgitg                                              [[simdgroup_index_in_threadgroup]]) {
    constexpr int n_threads = row_tile * 4;

    const int row_base = (int) tgpig.x * row_tile;
    const int col_base = (int) tgpig.y * 16;
    const int blocks = args.k / QK_FAIRY2I_TILE64;
    const int scale_row = (row_base / QK_FAIRY2I_TILE64) * QK_FAIRY2I_TILE64;
    const int coeff_base = (int) sgitg * (k_tile / 8) * 64;

    simdgroup_half8x8 a_rr;
    simdgroup_half8x8 a_ri;
    simdgroup_half8x8 a_ir;
    simdgroup_half8x8 a_ii;
    simdgroup_half8x8 b_r0;
    simdgroup_half8x8 b_i0;
    simdgroup_half8x8 b_r1;
    simdgroup_half8x8 b_i1;
    simdgroup_float8x8 c_r0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_i0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_r1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c_i1 = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (int wb = 0; wb < blocks; ++wb) {
        const int scale_index = scale_row * blocks + wb;
        const float4 wr = float4(
            (float) u_s0[scale_index].d_real,
            (float) u_s1[scale_index].d_real,
            (float) w_s0[scale_index].d_real,
            (float) w_s1[scale_index].d_real);
        const float4 wi = float4(
            (float) u_s0[scale_index].d_imag,
            (float) u_s1[scale_index].d_imag,
            (float) w_s0[scale_index].d_imag,
            (float) w_s1[scale_index].d_imag);

        const int coeff_k_lane = (int) tiitg & 7;
        const int coeff_row0 = (int) tiitg >> 3;
        const int coeff_row1 = coeff_row0 + row_tile / 2;
        uint4 packed_codes0 = uint4(0);
        uint4 packed_codes1 = uint4(0);
        if (row_base + coeff_row0 < args.m) {
            const int w_index = (row_base + coeff_row0) * blocks + wb;
            packed_codes0.x = (uint) u_s0[w_index].qs[coeff_k_lane] |
                              ((uint) u_s0[w_index].qs[coeff_k_lane + 8] << 8);
            packed_codes0.y = (uint) u_s1[w_index].qs[coeff_k_lane] |
                              ((uint) u_s1[w_index].qs[coeff_k_lane + 8] << 8);
            packed_codes0.z = (uint) w_s0[w_index].qs[coeff_k_lane] |
                              ((uint) w_s0[w_index].qs[coeff_k_lane + 8] << 8);
            packed_codes0.w = (uint) w_s1[w_index].qs[coeff_k_lane] |
                              ((uint) w_s1[w_index].qs[coeff_k_lane + 8] << 8);
        }
        if (row_base + coeff_row1 < args.m) {
            const int w_index = (row_base + coeff_row1) * blocks + wb;
            packed_codes1.x = (uint) u_s0[w_index].qs[coeff_k_lane] |
                              ((uint) u_s0[w_index].qs[coeff_k_lane + 8] << 8);
            packed_codes1.y = (uint) u_s1[w_index].qs[coeff_k_lane] |
                              ((uint) u_s1[w_index].qs[coeff_k_lane + 8] << 8);
            packed_codes1.z = (uint) w_s0[w_index].qs[coeff_k_lane] |
                              ((uint) w_s0[w_index].qs[coeff_k_lane + 8] << 8);
            packed_codes1.w = (uint) w_s1[w_index].qs[coeff_k_lane] |
                              ((uint) w_s1[w_index].qs[coeff_k_lane + 8] << 8);
        }

        for (int k_chunk = 0; k_chunk < QK_FAIRY2I_TILE64; k_chunk += k_tile) {
#pragma unroll
            for (int ik = 0; ik < k_tile / 8; ++ik) {
                const int k_base = k_chunk + ik * 8;
                const int code_shift = (k_base & 8) + 2 * (k_base >> 4);
                const uint4 code0 = (packed_codes0 >> uint4(code_shift)) & uint4(3);
                const uint4 code1 = (packed_codes1 >> uint4(code_shift)) & uint4(3);

                float4 coeff0 = float4(0.0f);
                if (row_base + coeff_row0 < args.m) {
                    coeff0 = fairy2i_mma_coeff_w2_codes_scaled(code0, wr, wi);
                }
                const uint coeff_index0 = (coeff_row0 >> 3) * (k_tile / 8) * 64 + ik * 64 +
                                          (coeff_row0 & 7) * 8 + coeff_k_lane;
                coeff_real_from_real[coeff_index0] = (half) coeff0.x;
                coeff_real_from_imag[coeff_index0] = (half) coeff0.y;
                coeff_imag_from_real[coeff_index0] = (half) coeff0.z;
                coeff_imag_from_imag[coeff_index0] = (half) coeff0.w;

                float4 coeff1 = float4(0.0f);
                if (row_base + coeff_row1 < args.m) {
                    coeff1 = fairy2i_mma_coeff_w2_codes_scaled(code1, wr, wi);
                }
                const uint coeff_index1 = (coeff_row1 >> 3) * (k_tile / 8) * 64 + ik * 64 +
                                          (coeff_row1 & 7) * 8 + coeff_k_lane;
                coeff_real_from_real[coeff_index1] = (half) coeff1.x;
                coeff_real_from_imag[coeff_index1] = (half) coeff1.y;
                coeff_imag_from_real[coeff_index1] = (half) coeff1.z;
                coeff_imag_from_imag[coeff_index1] = (half) coeff1.w;
            }

            for (uint idx = tiitg; idx < 16 * k_tile; idx += n_threads) {
                const int col_local = (int) idx / k_tile;
                const int k_local = (int) idx % k_tile;
                const int tile = col_local >> 3;
                const int col_lane = col_local & 7;
                const int col = col_base + col_local;

                half2 xh = half2(0.0h);
                if (col < args.act_rows) {
                    xh = fairy2i_load_staged_half_activation_pair(act_h, col, wb, blocks, k_chunk + k_local);
                }

                const int act_tile_idx = k_local * 8 + col_lane;
                if (tile == 0) {
                    act_real_tile0[act_tile_idx] = xh.x;
                    act_imag_tile0[act_tile_idx] = xh.y;
                } else {
                    act_real_tile1[act_tile_idx] = xh.x;
                    act_imag_tile1[act_tile_idx] = xh.y;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
            for (int ik = 0; ik < k_tile / 8; ++ik) {
                simdgroup_load(a_rr, coeff_real_from_real + coeff_base + ik * 64);
                simdgroup_load(a_ri, coeff_real_from_imag + coeff_base + ik * 64);
                simdgroup_load(a_ir, coeff_imag_from_real + coeff_base + ik * 64);
                simdgroup_load(a_ii, coeff_imag_from_imag + coeff_base + ik * 64);
                simdgroup_load(b_r0, act_real_tile0 + ik * 64);
                simdgroup_load(b_i0, act_imag_tile0 + ik * 64);
                simdgroup_load(b_r1, act_real_tile1 + ik * 64);
                simdgroup_load(b_i1, act_imag_tile1 + ik * 64);

                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_multiply_accumulate(c_r0, a_rr, b_r0, c_r0);
                simdgroup_multiply_accumulate(c_r0, a_ri, b_i0, c_r0);
                simdgroup_multiply_accumulate(c_i0, a_ir, b_r0, c_i0);
                simdgroup_multiply_accumulate(c_i0, a_ii, b_i0, c_i0);
                simdgroup_multiply_accumulate(c_r1, a_rr, b_r1, c_r1);
                simdgroup_multiply_accumulate(c_r1, a_ri, b_i1, c_r1);
                simdgroup_multiply_accumulate(c_i1, a_ir, b_r1, c_i1);
                simdgroup_multiply_accumulate(c_i1, a_ii, b_i1, c_i1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int simdgroup_out_base = (int) sgitg * 256;
    simdgroup_store(c_r0, out_tile + simdgroup_out_base, 8);
    simdgroup_store(c_i0, out_tile + simdgroup_out_base + 64, 8);
    simdgroup_store(c_r1, out_tile + simdgroup_out_base + 128, 8);
    simdgroup_store(c_i1, out_tile + simdgroup_out_base + 192, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tiitg; idx < row_tile * 16; idx += n_threads) {
        const int row_lane = (int) idx / 16;
        const int col_local = (int) idx & 15;
        const int tile = col_local >> 3;
        const int col_lane = col_local & 7;
        const int row = row_base + row_lane;
        const int col = col_base + col_local;
        if (row < args.m && col < args.act_rows) {
            const int i1 = col % args.x_ne1;
            const int i2 = (col / args.x_ne1) % args.x_ne2;
            const int i3 = col / (args.x_ne1 * args.x_ne2);
            const int row_group = row_lane >> 3;
            const int row_in_group = row_lane & 7;
            const int out_base = row_group * 256 + tile * 128;
            float out_real = out_tile[out_base + row_in_group * 8 + col_lane];
            float out_imag = out_tile[out_base + 64 + row_in_group * 8 + col_lane];

            if (args.has_bias) {
                const int b0r = row % args.bias_ne0;
                const int b0i = (row + args.m) % args.bias_ne0;
                const int b1 = i1 % args.bias_ne1;
                const int b2 = i2 % args.bias_ne2;
                const int b3 = i3 % args.bias_ne3;
                out_real += *((device const float *) (bias + (ulong) b0r * args.bias_nb0 + (ulong) b1 * args.bias_nb1 +
                                                      (ulong) b2 * args.bias_nb2 + (ulong) b3 * args.bias_nb3));
                out_imag += *((device const float *) (bias + (ulong) b0i * args.bias_nb0 + (ulong) b1 * args.bias_nb1 +
                                                      (ulong) b2 * args.bias_nb2 + (ulong) b3 * args.bias_nb3));
            }

            *((device uint *) (dst + (ulong) row * args.dst_nb0 + (ulong) i1 * args.dst_nb1 + (ulong) i2 * args.dst_nb2 +
                               (ulong) i3 * args.dst_nb3)) = fairy2i_pack_bf16_pair(out_real, out_imag);
        }
    }
}

typedef decltype(kernel_fairy2i_wide_linear_w2_half_w64scale_mma_rows<32, 8>) fairy2i_w2_w64scale_mma_rows_t;

template [[host_name("kernel_fairy2i_wide_linear_w2_half_w64scale_mma32x16")]]
kernel fairy2i_w2_w64scale_mma_rows_t kernel_fairy2i_wide_linear_w2_half_w64scale_mma_rows<32, 8>;

template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4>
void dequantize_f32_t4(device const float4 * src, short il, thread type4 & reg) {
    reg = (type4)(*src);
}

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4>
void dequantize_f16_t4(device const half4 * src, short il, thread type4 & reg) {
    reg = (type4)(*(src));
}

#if defined(GGML_METAL_HAS_BF16)
template <typename type4x4>
void dequantize_bf16(device const bfloat4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4>
void dequantize_bf16_t4(device const bfloat4 * src, short il, thread type4 & reg) {
    reg = (type4)(*(src));
}
#endif

template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q4_0_t4(device const block_q4_0 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = (il/4) ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + md;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + md;
    }
}

void quantize_q4_0(device const float * src, device block_q4_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK4_0; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / -8;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = src[0       + j]*id;
        const float x1 = src[QK4_0/2 + j]*id;

        const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

void quantize_q4_1(device const float * src, device block_q4_1 & dst) {
#pragma METAL fp math_mode(safe)
    float min = FLT_MAX;
    float max = -FLT_MAX;

    for (int j = 0; j < QK4_1; j++) {
        const float v = src[j];
        if (min > v) min = v;
        if (max < v) max = v;
    }

    const float d = (max - min) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (src[0       + j] - min)*id;
        const float x1 = (src[QK4_1/2 + j] - min)*id;

        const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

void quantize_q5_0(device const float * src, device block_q5_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK5_0; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / -16;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = src[0       + j]*id;
        const float x1 = src[QK5_0/2 + j]*id;

        const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread const uint8_t * qh8 = (thread const uint8_t *)&qh;

    for (int j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

void quantize_q5_1(device const float * src, device block_q5_1 & dst) {
#pragma METAL fp math_mode(safe)
    float max = src[0];
    float min = src[0];

    for (int j = 1; j < QK5_1; j++) {
        const float v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (src[0       + j] - min)*id;
        const float x1 = (src[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread const uint8_t * qh8 = (thread const uint8_t *)&qh;

    for (int j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

void quantize_q8_0(device const float * src, device block_q8_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = src[j];
        amax = MAX(amax, fabs(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

void quantize_iq4_nl(device const float * src, device block_iq4_nl & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK4_NL; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / kvalues_iq4nl_f[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = src[0        + j]*id;
        const float x1 = src[QK4_NL/2 + j]*id;

        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        const float v0 = kvalues_iq4nl_f[xi0];
        const float v1 = kvalues_iq4nl_f[xi1];
        const float w0 = src[0        + j]*src[0        + j];
        const float w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

template <typename type4x4>
void dequantize_q4_1(device const block_q4_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float  m = xb->m;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = ((qs[i] & mask0) * d1) + m;
        reg_f[i/2][2*(i%2) + 1] = ((qs[i] & mask1) * d2) + m;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q4_1_t4(device const block_q4_1 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = (il/4) ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float  m = xb->m;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + m;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + m;
    }
}

template <typename type4x4>
void dequantize_q5_0(device const block_q5_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.h * xb->d;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + md;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + md;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q5_0_t4(device const block_q5_0 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.h * xb->d;
    const ushort mask = (il/4) ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = (il/4) ? 4 : 0;

    const int gh_mv = (il/4) ? 12 : 0;
    const int gh_bk = (il/4) ?  0 : 4;

    for (int ii = 0; ii < 2; ii++) {
        int i = 2*(il%4) + ii;

        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + md;
        reg[2*ii + 1] = d * x1 + md;
    }
}

template <typename type4x4>
void dequantize_q5_1(device const block_q5_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + m;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + m;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q5_1_t4(device const block_q5_1 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = (il/4) ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = (il/4) ? 4 : 0;

    const int gh_mv = (il/4) ? 12 : 0;
    const int gh_bk = (il/4) ?  0 : 4;

    for (int ii = 0; ii < 2; ii++) {
        int i = 2*(il%4) + ii;

        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + m;
        reg[2*ii + 1] = d * x1 + m;
    }
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q8_0_t4(device const block_q8_0 *xb, short il, thread type4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    for (int i = 0; i < 4; i++) {
        reg[i] = (qs[4*(il%4) + i + 16*(il/4)] * d);
    }
}

template <typename type4x4>
void dequantize_mxfp4(device const block_mxfp4 * xb, short il, thread type4x4 & reg) {
    device const uint8_t * q2 = (device const uint8_t *)xb->qs;

    const float d = e8m0_to_fp32(xb->e);
    const uint8_t shr = il >= 1 ? 4 : 0;

    for (int i = 0; i < 4; ++i) {
        reg[i][0] = d * kvalues_mxfp4_f[(q2[4*i + 0] >> shr) & 0x0F];
        reg[i][1] = d * kvalues_mxfp4_f[(q2[4*i + 1] >> shr) & 0x0F];
        reg[i][2] = d * kvalues_mxfp4_f[(q2[4*i + 2] >> shr) & 0x0F];
        reg[i][3] = d * kvalues_mxfp4_f[(q2[4*i + 3] >> shr) & 0x0F];
    }
}

template <typename type4>
void dequantize_mxfp4_t4(device const block_mxfp4 * xb, short il, thread type4 & reg) {
    device const uint8_t * q2 = (device const uint8_t *)xb->qs;

    const float d = e8m0_to_fp32(xb->e);
    const short il4 = il%4;

    const uint8_t shr = il >= 4 ? 4 : 0;

    reg[0] = d * kvalues_mxfp4_f[(q2[4*il4 + 0] >> shr) & 0x0F];
    reg[1] = d * kvalues_mxfp4_f[(q2[4*il4 + 1] >> shr) & 0x0F];
    reg[2] = d * kvalues_mxfp4_f[(q2[4*il4 + 2] >> shr) & 0x0F];
    reg[3] = d * kvalues_mxfp4_f[(q2[4*il4 + 3] >> shr) & 0x0F];
}

template <typename type4x4>
void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];

    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;

    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q3_K(device const block_q3_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    device const uint8_t * h = (device const uint8_t *)xb->hmask;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    q = q + 32 * (il/8) + 16 * (il&1);
    h = h + 16 * (il&1);
    uint8_t m = 1 << (il/2);
    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : \
                                 ((il/4)>0 ? 12  : 3);
    uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
    uint16_t scale_2 = scales[il%8], scale_1 = scales[8 + il%4];
    int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
                               : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
    float dl = il<8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
    const float ml = 4.f * dl;

    il = (il/2) & 3;
    const half    coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    const uint8_t mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl *= coef;

    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - (h[i] & m ? 0 : ml);
    }
}

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;

    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask  = il<2 ? 0x0F : 0xF0;
    const float qh_val = il<2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
}

template <typename type4x4>
void dequantize_iq2_xxs(device const block_iq2_xxs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    // each block of 32 needs 2 uint32_t's for the quants & scale, so 4 uint16_t's.
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const uint32_t aux32_g = q2[0] | (q2[1] << 16);
    const uint32_t aux32_s = q2[2] | (q2[3] << 16);
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq2_xs(device const block_iq2_xs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+0] & 511));
    uint8_t signs = ksigns_iq2xs[q2[2*il+0] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+1] & 511));
    signs = ksigns_iq2xs[q2[2*il+1] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq3_xxs(device const block_iq3_xxs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * q3 = xb->qs + 8*ib32;
    device const uint16_t * gas = (device const uint16_t *)(xb->qs + QK_K/4) + 2*ib32;
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float dl = d * (0.5f + (aux32 >> 28)) * 0.5f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+0]);
    constant uint8_t * grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+1]);
    uint8_t signs = ksigns_iq2xs[(aux32 >> 14*il) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[1][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
    grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+2]);
    grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+3]);
    signs = ksigns_iq2xs[(aux32 >> (14*il+7)) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[3][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq3_s(device const block_iq3_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * qs = xb->qs + 8*ib32;
    device const uint8_t * signs = xb->signs + 4*ib32 + 2*il;
    const uint8_t qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (1 + 2*((xb->scales[ib32/2] >> 4*(ib32%2)) & 0xf));
    constant uint8_t * grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+0] | ((qh << 8) & 256)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+1] | ((qh << 7) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i+0]);
        reg[1][i] = dl * grid2[i] * select(1, -1, signs[0] & kmask_iq2xs[i+4]);
    }
    grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+2] | ((qh << 6) & 256)));
    grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+3] | ((qh << 5) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * select(1, -1, signs[1] & kmask_iq2xs[i+0]);
        reg[3][i] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i+4]);
    }
}

template <typename type4x4>
void dequantize_iq2_s(device const block_iq2_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device const uint8_t * signs = qs + QK_K/8;
    const uint8_t qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[0] | ((qh << 8) & 0x300)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[1] | ((qh << 6) & 0x300)));
    for (int i = 0; i < 8; ++i) {
        reg[i/4+0][i%4] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i]);
        reg[i/4+2][i%4] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i]);
    }
}

template <typename type4x4>
void dequantize_iq1_s(device const block_iq1_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    const float d = xb->d;
    device const uint8_t  * qs = xb->qs + 4*ib32 + 2*il;
    device const uint16_t * qh = xb->qh;
    const float dl = d * (2*((qh[ib32] >> 12) & 7) + 1);
    const float ml = dl * (qh[ib32] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA);
    const uint16_t h = qh[ib32] >> 6*il;
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((h << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((h << 5) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml;
        reg[1][i] = dl * (grid1[i] >>  4) + ml;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml;
        reg[3][i] = dl * (grid2[i] >>  4) + ml;
    }
}

template <typename type4x4>
void dequantize_iq1_m(device const block_iq1_m * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    device const uint16_t * sc = (device const uint16_t *)xb->scales;

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = scale.f16;

    device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device const uint8_t * qh = xb->qh + 2*ib32 + il;

    const float dl  = d * (2*((sc[ib32/2] >> (6*(ib32%2)+3*il)) & 7) + 1);
    const float ml1 = dl * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    const float ml2 = dl * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml1;
        reg[1][i] = dl * (grid1[i] >>  4) + ml1;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml2;
        reg[3][i] = dl * (grid2[i] >>  4) + ml2;
    }
}

template <typename type4x4>
void dequantize_iq4_nl(device const block_iq4_nl * xb, short il, thread type4x4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = ((q4[2*i] | (q4[2*i+1] << 16)) >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

template <typename type4>
void dequantize_iq4_nl_t4(device const block_iq4_nl * xb, short il, thread type4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    aux32 = ((q4[2*(il%4)] | (q4[2*(il%4)+1] << 16)) >> 4*(il/4)) & 0x0f0f0f0f;
    reg[0] = d * kvalues_iq4nl_f[q8[0]];
    reg[1] = d * kvalues_iq4nl_f[q8[1]];
    reg[2] = d * kvalues_iq4nl_f[q8[2]];
    reg[3] = d * kvalues_iq4nl_f[q8[3]];
}

template <typename type4x4>
void dequantize_iq4_xs(device const block_iq4_xs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint32_t * q4 = (device const uint32_t *)xb->qs + 4*ib32;
    const int ls = ((xb->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((xb->scales_h >> 2*ib32) & 3) << 4);
    const float d = (float)xb->d * (ls - 32);
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = (q4[i] >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

enum ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

// general-purpose kernel for addition, subtraction, multiplication and division of two tensors
// pros: works for non-contiguous tensors, supports broadcast across all dims
// cons: not very efficient
template <int F>
kernel void kernel_add_fuse_impl(
        constant ggml_metal_kargs_bin & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig.z;
    const int i02 = tgpig.y;
    const int i01 = tgpig.x;

    const int i13 = i03%args.ne13;
    const int i12 = i02%args.ne12;
    const int i11 = i01%args.ne11;

    device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + args.offs);
    device       float * dst_ptr  = (device       float *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1  + args.offs);

    device const float * src1_ptr[F];
    for (short j = 0; j < F; ++j) {
        src1_ptr[j] = (device const float *) (src1 + args.o1[j] + i13*args.nb13 + i12*args.nb12 + i11*args.nb11);
    }

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int i10 = i0%args.ne10;

        float res = src0_ptr[i0];

#pragma unroll
        for (short j = 0; j < F; ++j) {
            res += src1_ptr[j][i10];
        }

        dst_ptr[i0] = res;
    }
}

typedef decltype(kernel_add_fuse_impl<2>) kernel_add_fuse_t;

template [[host_name("kernel_add_fuse_1")]] kernel kernel_add_fuse_t kernel_add_fuse_impl<1>;
template [[host_name("kernel_add_fuse_2")]] kernel kernel_add_fuse_t kernel_add_fuse_impl<2>;
template [[host_name("kernel_add_fuse_3")]] kernel kernel_add_fuse_t kernel_add_fuse_impl<3>;
template [[host_name("kernel_add_fuse_4")]] kernel kernel_add_fuse_t kernel_add_fuse_impl<4>;
template [[host_name("kernel_add_fuse_5")]] kernel kernel_add_fuse_t kernel_add_fuse_impl<5>;
template [[host_name("kernel_add_fuse_6")]] kernel kernel_add_fuse_t kernel_add_fuse_impl<6>;
template [[host_name("kernel_add_fuse_7")]] kernel kernel_add_fuse_t kernel_add_fuse_impl<7>;
template [[host_name("kernel_add_fuse_8")]] kernel kernel_add_fuse_t kernel_add_fuse_impl<8>;

kernel void kernel_sub_fuse_1(
        constant ggml_metal_kargs_bin & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig.z;
    const int i02 = tgpig.y;
    const int i01 = tgpig.x;

    const int i13 = i03%args.ne13;
    const int i12 = i02%args.ne12;
    const int i11 = i01%args.ne11;

    device const char * src0_ptr = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + args.offs;
    device const char * src1_ptr = src1 + i13*args.nb13 + i12*args.nb12 + i11*args.nb11 + args.o1[0];
    device       char * dst_ptr  = dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1  + args.offs;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int i10 = i0%args.ne10;
        *((device float *)(dst_ptr + i0*args.nb0)) = *((device float *)(src0_ptr + i0*args.nb00)) - *((device float *)(src1_ptr + i10*args.nb10));
    }
}

kernel void kernel_mul_fuse_1(
        constant ggml_metal_kargs_bin & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig.z;
    const int i02 = tgpig.y;
    const int i01 = tgpig.x;

    const int i13 = i03%args.ne13;
    const int i12 = i02%args.ne12;
    const int i11 = i01%args.ne11;

    device const char * src0_ptr = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + args.offs;
    device const char * src1_ptr = src1 + i13*args.nb13 + i12*args.nb12 + i11*args.nb11 + args.o1[0];
    device       char * dst_ptr  = dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1  + args.offs;

    if (args.ne10 == 1) {
        const float x = *((device float *)(src1_ptr));
        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            *((device float *)(dst_ptr + i0*args.nb0)) = *((device float *)(src0_ptr + i0*args.nb00)) * x;
        }
    } else {
        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            const int i10 = i0%args.ne10;
            *((device float *)(dst_ptr + i0*args.nb0)) = *((device float *)(src0_ptr + i0*args.nb00)) * *((device float *)(src1_ptr + i10*args.nb10));
        }
    }
}

kernel void kernel_div_fuse_1(
        constant ggml_metal_kargs_bin & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig.z;
    const int i02 = tgpig.y;
    const int i01 = tgpig.x;

    const int i13 = i03%args.ne13;
    const int i12 = i02%args.ne12;
    const int i11 = i01%args.ne11;

    device const char * src0_ptr = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + args.offs;
    device const char * src1_ptr = src1 + i13*args.nb13 + i12*args.nb12 + i11*args.nb11 + args.o1[0];
    device       char * dst_ptr  = dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1  + args.offs;

    if (args.ne10 == 1) {
        const float x = 1.0f / *((device float *)(src1_ptr));
        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            *((device float *)(dst_ptr + i0*args.nb0)) = *((device float *)(src0_ptr + i0*args.nb00)) * x;
        }
    } else {
        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            const int i10 = i0%args.ne10;
            *((device float *)(dst_ptr + i0*args.nb0)) = *((device float *)(src0_ptr + i0*args.nb00)) / *((device float *)(src1_ptr + i10*args.nb10));
        }
    }
}

kernel void kernel_add_id(
        constant ggml_metal_kargs_add_id & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i1 = tgpig.x;
    const int i2 = tgpig.y;

    const int i11 = *((device const int32_t *) (src2 + i1*sizeof(int32_t) + i2*args.nb21));

    const size_t nb1 = args.ne0 * sizeof(float);
    const size_t nb2 = args.ne1 * nb1;

    device       float * dst_row  = (device       float *)((device char *)dst + i1*nb1 + i2*nb2);
    device const float * src0_row = (device const float *)((device char *)src0 +  i1*args.nb01 + i2*args.nb02);
    device const float * src1_row = (device const float *)((device char *)src1 + i11*args.nb11);

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_row[i0] = src0_row[i0] + src1_row[i0];
    }
}

template<typename T>
kernel void kernel_repeat(
        constant ggml_metal_kargs_repeat & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    const int i03 = i3%args.ne03;
    const int i02 = i2%args.ne02;
    const int i01 = i1%args.ne01;

    device const char * src0_ptr = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01;
    device       char * dst_ptr  = dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int i00 = i0%args.ne00;
        *((device T *)(dst_ptr + i0*args.nb0)) = *((device T *)(src0_ptr + i00*args.nb00));
    }
}

typedef decltype(kernel_repeat<float>) kernel_repeat_t;

template [[host_name("kernel_repeat_f32")]] kernel kernel_repeat_t kernel_repeat<float>;
template [[host_name("kernel_repeat_f16")]] kernel kernel_repeat_t kernel_repeat<half>;
template [[host_name("kernel_repeat_i32")]] kernel kernel_repeat_t kernel_repeat<int>;
template [[host_name("kernel_repeat_i16")]] kernel kernel_repeat_t kernel_repeat<short>;

// assumption: src1 is a row
// broadcast src1 into src0
template <short F>
kernel void kernel_add_row_c4_fuse_impl(
        constant ggml_metal_kargs_bin & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tpig[[thread_position_in_grid]]) {
    const uint nb = args.ne00/4;
    const uint i  = tpig % nb;

    device const float4 * src0_row = (device const float4 *) (src0);
    device       float4 *  dst_row = (device       float4 *) (dst);

    float4 res = src0_row[tpig];

#pragma unroll(F)
    for (short j = 0; j < F; ++j) {
        res += ((device const float4 *) (src1 + args.o1[j]))[i];
    }

    dst_row[tpig] = res;
}

typedef decltype(kernel_add_row_c4_fuse_impl<1>) kernel_add_row_c4_fuse_t;

template [[host_name("kernel_add_row_c4_fuse_1")]] kernel kernel_add_row_c4_fuse_t kernel_add_row_c4_fuse_impl<1>;
template [[host_name("kernel_add_row_c4_fuse_2")]] kernel kernel_add_row_c4_fuse_t kernel_add_row_c4_fuse_impl<2>;
template [[host_name("kernel_add_row_c4_fuse_3")]] kernel kernel_add_row_c4_fuse_t kernel_add_row_c4_fuse_impl<3>;
template [[host_name("kernel_add_row_c4_fuse_4")]] kernel kernel_add_row_c4_fuse_t kernel_add_row_c4_fuse_impl<4>;
template [[host_name("kernel_add_row_c4_fuse_5")]] kernel kernel_add_row_c4_fuse_t kernel_add_row_c4_fuse_impl<5>;
template [[host_name("kernel_add_row_c4_fuse_6")]] kernel kernel_add_row_c4_fuse_t kernel_add_row_c4_fuse_impl<6>;
template [[host_name("kernel_add_row_c4_fuse_7")]] kernel kernel_add_row_c4_fuse_t kernel_add_row_c4_fuse_impl<7>;
template [[host_name("kernel_add_row_c4_fuse_8")]] kernel kernel_add_row_c4_fuse_t kernel_add_row_c4_fuse_impl<8>;

template <short F>
kernel void kernel_sub_row_c4_fuse_impl(
        constant ggml_metal_kargs_bin & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tpig[[thread_position_in_grid]]) {

    const uint nb = args.ne00/4;
    const uint i  = tpig % nb;

    device const float4 * src0_row = (device const float4 *) (src0);
    device       float4 *  dst_row = (device       float4 *) (dst);

    device const float4 * src1_row[F];
    for (short j = 0; j < F; ++j) {
        src1_row[j] = (device const float4 *) (src1 + args.o1[j]);
    }

    float4 res = src0_row[tpig];

#pragma unroll(F)
    for (short j = 0; j < F; ++j) {
        res -= src1_row[j][i];
    }

    dst_row[tpig] = res;
}

typedef decltype(kernel_sub_row_c4_fuse_impl<1>) kernel_sub_row_c4_fuse_t;

template [[host_name("kernel_sub_row_c4_fuse_1")]] kernel kernel_sub_row_c4_fuse_t kernel_sub_row_c4_fuse_impl<1>;

template <short F>
kernel void kernel_mul_row_c4_fuse_impl(
        constant ggml_metal_kargs_bin & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tpig[[thread_position_in_grid]]) {

    const uint nb = args.ne00/4;
    const uint i  = tpig % nb;

    device const float4 * src0_row = (device const float4 *) (src0);
    device       float4 *  dst_row = (device       float4 *) (dst);

    device const float4 * src1_row[F];
    for (short j = 0; j < F; ++j) {
        src1_row[j] = (device const float4 *) (src1 + args.o1[j]);
    }

    float4 res = src0_row[tpig];

#pragma unroll(F)
    for (short j = 0; j < F; ++j) {
        res *= src1_row[j][i];
    }

    dst_row[tpig] = res;
}

typedef decltype(kernel_mul_row_c4_fuse_impl<1>) kernel_mul_row_c4_fuse_t;

template [[host_name("kernel_mul_row_c4_fuse_1")]] kernel kernel_mul_row_c4_fuse_t kernel_mul_row_c4_fuse_impl<1>;

template <short F>
kernel void kernel_div_row_c4_fuse_impl(
        constant ggml_metal_kargs_bin & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tpig[[thread_position_in_grid]]) {

    const uint nb = args.ne00/4;
    const uint i  = tpig % nb;

    device const float4 * src0_row = (device const float4 *) (src0);
    device       float4 *  dst_row = (device       float4 *) (dst);

    device const float4 * src1_row[F];
    for (short j = 0; j < F; ++j) {
        src1_row[j] = (device const float4 *) (src1 + args.o1[j]);
    }

    float4 res = src0_row[tpig];

#pragma unroll(F)
    for (short j = 0; j < F; ++j) {
        res /= src1_row[j][i];
    }

    dst_row[tpig] = res;
}

typedef decltype(kernel_div_row_c4_fuse_impl<1>) kernel_div_row_c4_fuse_t;

template [[host_name("kernel_div_row_c4_fuse_1")]] kernel kernel_div_row_c4_fuse_t kernel_div_row_c4_fuse_impl<1>;

kernel void kernel_scale_f32(
        constant ggml_metal_kargs_scale & args,
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * args.scale + args.bias;
}

kernel void kernel_scale_f32_4(
        constant ggml_metal_kargs_scale & args,
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * args.scale + args.bias;
}

kernel void kernel_clamp_f32(
        constant ggml_metal_kargs_clamp & args,
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = clamp(src0[tpig], args.min, args.max);
}

kernel void kernel_clamp_f32_4(
        constant ggml_metal_kargs_clamp & args,
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = clamp(src0[tpig], args.min, args.max);
}

kernel void kernel_relu_f32(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = max(0.0f, src0[tpig]);
}

kernel void kernel_relu_f32_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = max(0.0f, src0[tpig]);
}

kernel void kernel_sigmoid_f32(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = 1.0f / (1.0f + exp(-src0[tpig]));
}

kernel void kernel_sigmoid_f32_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = 1.0f / (1.0f + exp(-src0[tpig]));
}

kernel void kernel_tanh_f32(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = precise::tanh(src0[tpig]);
}

kernel void kernel_tanh_f32_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = precise::tanh(src0[tpig]);
}

constant float GELU_COEF_A     = 0.044715f;
constant float GELU_QUICK_COEF = -1.702f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant float SQRT_2_INV      = 0.70710678118654752440084436210484f;

kernel void kernel_gelu_f32(
    device const float * src0,
    device       float * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];

    dst[tpig] = 0.5f*x*(1.0f + precise::tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_gelu_f32_4(
    device const float4 * src0,
    device       float4 * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];

    // BEWARE !!!
    // Simply using "tanh" instead of "precise::tanh" will sometimes results in NaNs!
    // This was observed with Falcon 7B and 40B models
    //
    dst[tpig] = 0.5f*x*(1.0f + precise::tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_gelu_quick_f32(
    device const float * src0,
    device       float * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];

    dst[tpig] = x*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x)));
}

kernel void kernel_gelu_quick_f32_4(
    device const float4 * src0,
    device       float4 * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];

    dst[tpig] = x*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x)));
}

// based on Abramowitz and Stegun formula 7.1.26 or similar Hastings' approximation
// ref: https://www.johndcook.com/blog/python_erf/
constant float p_erf  = 0.3275911f;
constant float a1_erf = 0.254829592f;
constant float a2_erf = -0.284496736f;
constant float a3_erf = 1.421413741f;
constant float a4_erf = -1.453152027f;
constant float a5_erf = 1.061405429f;

template<typename T>
T erf_approx(T x) {
    T sign_x = sign(x);
    x = fabs(x);
    T t = 1.0f / (1.0f + p_erf * x);
    T y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * exp(-x * x);
    return sign_x * y;
}

kernel void kernel_gelu_erf_f32(
    device const float * src0,
    device       float * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];

    dst[tpig] = 0.5f*x*(1.0f+erf_approx<float>(x*SQRT_2_INV));
}

kernel void kernel_gelu_erf_f32_4(
    device const float4 * src0,
    device       float4 * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];

    dst[tpig] = 0.5f*x*(1.0f+erf_approx<float4>(x*SQRT_2_INV));
}

kernel void kernel_silu_f32(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];
    dst[tpig] = x / (1.0f + exp(-x));
}

kernel void kernel_silu_f32_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];
    dst[tpig] = x / (1.0f + exp(-x));
}

kernel void kernel_elu_f32(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    const float x = src0[tpig];
    dst[tpig] = (x > 0.0f) ? x : (exp(x) - 1.0f);
}

kernel void kernel_elu_f32_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    const float4 x = src0[tpig];
    dst[tpig][0] = (x[0] > 0.0f) ? x[0] : (exp(x[0]) - 1.0f);
    dst[tpig][1] = (x[1] > 0.0f) ? x[1] : (exp(x[1]) - 1.0f);
    dst[tpig][2] = (x[2] > 0.0f) ? x[2] : (exp(x[2]) - 1.0f);
    dst[tpig][3] = (x[3] > 0.0f) ? x[3] : (exp(x[3]) - 1.0f);
}

kernel void kernel_sqr_f32(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src0[tpig];
}

kernel void kernel_sqr_f32_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src0[tpig];
}

kernel void kernel_sqrt_f32(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = sqrt(src0[tpig]);
}

kernel void kernel_sqrt_f32_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = sqrt(src0[tpig]);
}

kernel void kernel_sin_f32(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = sin(src0[tpig]);
}

kernel void kernel_sin_f32_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = sin(src0[tpig]);
}

kernel void kernel_cos_f32(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = cos(src0[tpig]);
}

kernel void kernel_cos_f32_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = cos(src0[tpig]);
}

kernel void kernel_log_f32(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = log(src0[tpig]);
}

kernel void kernel_log_f32_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = log(src0[tpig]);
}

kernel void kernel_neg_f32(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = -src0[tpig];
}

kernel void kernel_neg_f32_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = -src0[tpig];
}

kernel void kernel_abs_f32(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = fabs(src0[tpig]);
}

kernel void kernel_abs_f32_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = fabs(src0[tpig]);
}

kernel void kernel_sgn_f32(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = sign(src0[tpig]);
}

kernel void kernel_sgn_f32_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = sign(src0[tpig]);
}

kernel void kernel_step_f32(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = step(0.0f, src0[tpig]);
}

kernel void kernel_step_f32_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = step(0.0f, src0[tpig]);
}

kernel void kernel_hardswish_f32(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    const float x = src0[tpig];
    dst[tpig] = x * fmin(1.0f, fmax(0.0f, (x + 3.0f) / 6.0f));
}

kernel void kernel_hardswish_f32_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    const float4 x = src0[tpig];
    dst[tpig] = x * fmin(1.0f, fmax(0.0f, (x + 3.0f) / 6.0f));
}

kernel void kernel_hardsigmoid_f32(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    const float x = src0[tpig];
    dst[tpig] = fmin(1.0f, fmax(0.0f, (x + 3.0f) / 6.0f));
}

kernel void kernel_hardsigmoid_f32_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    const float4 x = src0[tpig];
    dst[tpig] = fmin(1.0f, fmax(0.0f, (x + 3.0f) / 6.0f));
}

kernel void kernel_exp_f32(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = exp(src0[tpig]);
}

kernel void kernel_exp_f32_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = exp(src0[tpig]);
}

kernel void kernel_reglu_f32(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;
    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;
    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig*args.nb1);

    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        dst_row[i0] = x0*x1*(x0 > 0.0f);
    }
}

kernel void kernel_geglu_f32(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;
    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;
    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig*args.nb1);

    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu = 0.5f*x0*(1.0f + precise::tanh(SQRT_2_OVER_PI*x0*(1.0f + GELU_COEF_A*x0*x0)));

        dst_row[i0] = gelu*x1;
    }
}

kernel void kernel_swiglu_f32(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;
    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;
    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig*args.nb1);

    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float silu = x0 / (1.0f + exp(-x0));

        dst_row[i0] = silu*x1;
    }
}

kernel void kernel_swiglu_oai_f32(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;
    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;
    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig*args.nb1);

    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {
        float x0 = src0_row[i0];
        float x1 = src1_row[i0];

        x0 = min(x0, args.limit);
        x1 = max(min(x1, args.limit), -args.limit);

        float out_glu = x0 / (1.0f + exp(-x0 * args.alpha));
        out_glu = out_glu * (1.0f + x1);

        dst_row[i0] = out_glu;
    }
}

kernel void kernel_geglu_erf_f32(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;
    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;
    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig*args.nb1);

    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu_erf = 0.5f*x0*(1.0f+erf_approx<float>(x0*SQRT_2_INV));

        dst_row[i0] = gelu_erf*x1;
    }
}

kernel void kernel_geglu_quick_f32(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;
    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;
    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig*args.nb1);

    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu_quick = x0*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x0)));

        dst_row[i0] = gelu_quick*x1;
    }
}

template <bool norm>
kernel void kernel_sum_rows(
        constant ggml_metal_kargs_sum_rows & args,
        device const float * src0,
        device       float * dst,
        threadgroup  float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    int64_t i3 = tgpig.z;
    int64_t i2 = tgpig.y;
    int64_t i1 = tgpig.x;

    if (i3 >= args.ne03 || i2 >= args.ne02 || i1 >= args.ne01) {
        return;
    }

    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    device const float * src_row = (device const float *) ((device const char *) src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03);
    device       float * dst_row = (device       float *) ((device       char *) dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3);

    float sumf = 0;

    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        sumf += src_row[i0];
    }

    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    if (tpitg.x == 0) {
        dst_row[0] = norm ? sumf / args.ne00 : sumf;
    }
}

typedef decltype(kernel_sum_rows<false>) kernel_sum_rows_t;

template [[host_name("kernel_sum_rows_f32")]] kernel kernel_sum_rows_t kernel_sum_rows<false>;
template [[host_name("kernel_mean_f32")]]     kernel kernel_sum_rows_t kernel_sum_rows<true>;

template<typename T>
kernel void kernel_soft_max(
        constant ggml_metal_kargs_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float * psrc0 =                (device const float *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const     T * pmask = src1 != src0 ? (device const T *    ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float * psrc2 = src2 != src0 ? (device const float *) (src2)                                                 : nullptr;
    device       float * pdst  =                (device       float *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    // ALiBi
    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float lmax = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        lmax = MAX(lmax, psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f));
    }

    // find the max value in the block
    float max_val = simd_max(lmax);
    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        const float exp_psrc0 = exp((psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    // This barrier fixes a failing test
    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        pdst[i00] *= inv_sum;
    }
}

template<typename T>
kernel void kernel_soft_max_4(
        constant ggml_metal_kargs_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float4 * psrc4 =                (device const float4 *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const      T * pmask = src1 != src0 ? (device const T *     ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float *  psrc2 = src2 != src0 ? (device const float * ) (src2)                                                 : nullptr;
    device       float4 * pdst4 =                (device       float4 *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        lmax4 = fmax(lmax4, psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f)));
    }

    const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        const float4 exp_psrc4 = exp((psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }

    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    // This barrier fixes a failing test
    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        pdst4[i00] *= inv_sum;
    }
}

typedef decltype(kernel_soft_max<float>)    kernel_soft_max_t;
typedef decltype(kernel_soft_max_4<float4>) kernel_soft_max_4_t;

template [[host_name("kernel_soft_max_f16")]]   kernel kernel_soft_max_t   kernel_soft_max<half>;
template [[host_name("kernel_soft_max_f32")]]   kernel kernel_soft_max_t   kernel_soft_max<float>;
template [[host_name("kernel_soft_max_f16_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<half4>;
template [[host_name("kernel_soft_max_f32_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<float4>;

// ref: ggml.c:ggml_compute_forward_ssm_conv_f32
kernel void kernel_ssm_conv_f32_f32(
        constant ggml_metal_kargs_ssm_conv & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t ir = tgpig.x;
    const int64_t i2 = tgpig.y;
    const int64_t i3 = tgpig.z;

    const int64_t nc  = args.ne10;
  //const int64_t ncs = args.ne00;
  //const int64_t nr  = args.ne01;
  //const int64_t n_t = args.ne1;
  //const int64_t n_s = args.ne2;

    device const float * s = (device const float *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);
    device const float * c = (device const float *) ((device const char *) src1 + ir*args.nb11);
    device       float * x = (device       float *) ((device       char *) dst  + ir*args.nb0  + i2*args.nb1  + i3*args.nb2);

    float sumf = 0.0f;

    for (int64_t i0 = 0; i0 < nc; ++i0) {
        sumf += s[i0] * c[i0];
    }

    x[0] = sumf;
}

// ref: ggml.c:ggml_compute_forward_ssm_scan_f32, Mamba-1 part
kernel void kernel_ssm_scan_f32(
        constant ggml_metal_kargs_ssm_scan & args,
        device const void * src0,
        device const void * src1,
        device const void * src2,
        device const void * src3,
        device const void * src4,
        device const void * src5,
        device const void * src6,
        device      float * dst,
        threadgroup float * shared [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        uint3  tpitg[[thread_position_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgptg[[simdgroups_per_threadgroup]],
        uint3   tgpg[[threadgroups_per_grid]]) {

    const int64_t i0 = tpitg.x;
    const int64_t i1 = 0;
    const int64_t ir = tgpig.x; // current head
    const int64_t i3 = tgpig.y; // current seq

    const uint64_t nb00 = sizeof(float);
    const uint64_t nb10 = sizeof(float);
    const uint64_t nb20 = sizeof(float);

    const int64_t nc  = args.d_state;
    const int64_t nr  = args.d_inner;
    const int64_t nh  = args.n_head;
    const int64_t ng  = args.n_group;
    const int64_t n_t = args.n_seq_tokens;

    const int64_t s_off = args.s_off;

    device const int32_t * ids = (device const int32_t *) src6;

    device const float * s0_buff = (device const float *) ((device const char *) src0 + ir*args.nb02 + ids[i3]*args.nb03);
    device       float * s_buff  = (device       float *) ((device       char *) dst  + ir*args.nb02 +      i3*args.nb03 + s_off);
    const int64_t i = i0 + i1*nc;
    const int64_t g = ir / (nh / ng); // repeat_interleave
    float s0 = s0_buff[i];
    float s  = s_buff[i];

        device const float * A        = (device const float *) ((device const char *) src3 + ir*args.nb31);
        device const float * x_block  = (device const float *) ((device const char *) src1 + i1*nb10 + ir*args.nb11 + i3*args.nb13);
        device const float * dt_block = (device const float *) ((device const char *) src2 + ir*nb20 + i3*args.nb22);
        device const float * B_block  = (device const float *) ((device const char *) src4 + g*args.nb41 + i3*args.nb43);
        device const float * C_block  = (device const float *) ((device const char *) src5 + g*args.nb51 + i3*args.nb53);
        device       float * y_block  = (device       float *) ((device       char *) dst  + (i1 + ir*(nr) + i3*(n_t*nh*nr))*nb00);

    for (int64_t i2 = 0; i2 < n_t; ++i2) {
        device const float * x  = (device const float *) ((device const char *) x_block + i2*args.nb12);    // {dim, nh, nt, ns}
        device const float * dt = (device const float *) ((device const char *) dt_block + i2*args.nb21);   // {nh, nt, ns}
        device const float * B  = (device const float *) ((device const char *) B_block + i2*args.nb42);    // {d_state, ng, nt, ns}
        device const float * C  = (device const float *) ((device const char *) C_block + i2*args.nb52);    // {d_state, ng, nt, ns}
        device       float * y  = (device       float *) ((device       char *) y_block + i2*(nh*nr*nb00)); // {dim, nh, nt, ns}

        const float dt_soft_plus = dt[0] <= 20.0f ? log(1.0f + exp(dt[0])) : dt[0];
        const float x_dt = x[0] * dt_soft_plus;

        const float state = (s0 * exp(dt_soft_plus * A[i0])) + (B[i0] * x_dt);
        s = state;

        // Parallel sum: This relies on the fact that this kernel will be
        // dispatched with each threadgroup having (d_state, 1, 1) threads which
        // are subdivided into SIMD groups of size `sgptg`. The goal is to
        // compute y = sum({state * C[i] for i in range(d_state)}).
        // To parallelize this effectively, we first use simd_sum over each SIMD
        // group to compute the sum of each SIMD group, then place the result in
        // the SIMD group's indexed bucket in the shared memory. We then sum
        // over the individual group sums to compute the final sum.

        // Computed for each thread
        float sumf = state * C[i0];

        // Sum the threads in the simd group => simd sum
        sumf = simd_sum(sumf);

        if (sgptg > 1) {

            // Once per simd group, place the group sum into the shared buffer
            if (tiisg == 0) {
                shared[sgitg] = sumf;
            }

            // Wait for all threads in the threadgroup to reach this point. This
            // ensures that all elements of the shared buffer are populated with the
            // sum of the individual simd groups.
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // For simd group 0 at indices < num simd groups, extract the shared
            // simd sum
            sumf = 0.0f;
            if (sgitg == 0) {
                if (tiisg < sgptg) {
                    sumf = shared[tiisg];
                }
                sumf = simd_sum(sumf);
                if (tiisg == 0) {
                    y[0] = sumf;
                }
            }
        } else if (tiisg == 0) {
            y[0] = sumf;
        }

        // recurse
        s0 = s;
    }

    // Assign the final state to the output buffer
    s_buff[i] = s;
}

// ref: ggml.c:ggml_compute_forward_ssm_scan_f32, Mamba-2 part
kernel void kernel_ssm_scan_group_f32(
        constant ggml_metal_kargs_ssm_scan & args,
        device const void * src0,
        device const void * src1,
        device const void * src2,
        device const void * src3,
        device const void * src4,
        device const void * src5,
        device const void * src6,
        device      float * dst,
        threadgroup float * shared [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        uint3  tpitg[[thread_position_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgptg[[simdgroups_per_threadgroup]],
        uint3   tgpg[[threadgroups_per_grid]]) {

    const int64_t i0 = tpitg.x;
    const int64_t i1 = tgpig.x;
    const int64_t ir = tgpig.y; // current head
    const int64_t i3 = tgpig.z; // current seq

    const uint64_t nb00 = sizeof(float);
    const uint64_t nb10 = sizeof(float);
    const uint64_t nb20 = sizeof(float);

    const int64_t nc  = args.d_state;
    const int64_t nr  = args.d_inner;
    const int64_t nh  = args.n_head;
    const int64_t ng  = args.n_group;
    const int64_t n_t = args.n_seq_tokens;

    const int64_t s_off = args.s_off;

    device const int32_t * ids = (device const int32_t *) src6;

    device const float * s0_buff = (device const float *) ((device const char *) src0 + ir*args.nb02 + ids[i3]*args.nb03);
    device       float * s_buff  = (device       float *) ((device       char *) dst  + ir*args.nb02 +      i3*args.nb03 + s_off);
    const int64_t i = i0 + i1*nc;
    const int64_t g = ir / (nh / ng); // repeat_interleave
    float s0 = s0_buff[i];
    float s  = s_buff[i];

    device const float * A        = (device const float *) ((device const char *) src3 + ir*args.nb31); // {1, nh}
    device const float * x_block  = (device const float *) ((device const char *) src1 + i1*nb10 + ir*args.nb11 + i3*args.nb13);
    device const float * dt_block = (device const float *) ((device const char *) src2 + ir*nb20 + i3*args.nb22);
    device const float * B_block  = (device const float *) ((device const char *) src4 + g*args.nb41 + i3*args.nb43);
    device const float * C_block  = (device const float *) ((device const char *) src5 + g*args.nb51 + i3*args.nb53);
    device       float * y_block  = (device       float *) ((device       char *) dst  + (i1 + ir*(nr) + i3*(n_t*nh*nr))*nb00);

    for (int64_t i2 = 0; i2 < n_t; ++i2) {
        device const float * x  = (device const float *) ((device const char *) x_block  + i2*args.nb12);    // {dim, nh, nt, ns}
        device const float * dt = (device const float *) ((device const char *) dt_block + i2*args.nb21);    // {nh, nt, ns}
        device const float * B  = (device const float *) ((device const char *) B_block  + i2*args.nb42);    // {d_state, ng, nt, ns}
        device const float * C  = (device const float *) ((device const char *) C_block  + i2*args.nb52);    // {d_state, ng, nt, ns}
        device       float * y  = (device       float *) ((device       char *) y_block  + i2*(nh*nr*nb00)); // {dim, nh, nt, ns}

        const float dt_soft_plus = dt[0] <= 20.0f ? log(1.0f + exp(dt[0])) : dt[0];
        const float x_dt = x[0] * dt_soft_plus;
        const float dA = exp(dt_soft_plus * A[0]);

        const float state = (s0 * dA) + (B[i0] * x_dt);
        s = state;

        // Parallel sum: This relies on the fact that this kernel will be
        // dispatched with each threadgroup having (d_state, 1, 1) threads which
        // are subdivided into SIMD groups of size `sgptg`. The goal is to
        // compute y = sum({state * C[i] for i in range(d_state)}).
        // To parallelize this effectively, we first use simd_sum over each SIMD
        // group to compute the sum of each SIMD group, then place the result in
        // the SIMD group's indexed bucket in the shared memory. We then sum
        // over the individual group sums to compute the final sum.

        // Computed for each thread
        float sumf = state * C[i0];

        // Sum the threads in the simd group => simd sum
        sumf = simd_sum(sumf);

        // Once per simd group, place the group sum into the shared buffer
        if (tiisg == 0) {
            shared[sgitg] = sumf;
        }

        // Wait for all threads in the threadgroup to reach this point. This
        // ensures that all elements of the shared buffer are populated with the
        // sum of the individual simd groups.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // For simd group 0 at indices < num simd groups, extract the shared
        // simd sum
        sumf = 0.0f;
        if (sgitg == 0) {
            if (tiisg < sgptg) {
                sumf = shared[tiisg];
            }
            sumf = simd_sum(sumf);
            if (tiisg == 0) {
                y[0] = sumf;
            }
        }

        // recurse
        s0 = s;
    }

    // Assign the final state to the output buffer
    s_buff[i] = s;
}

kernel void kernel_rwkv_wkv6_f32(
    device const float * k,
    device const float * v,
    device const float * r,
    device const float * tf,
    device const float * td,
    device const float * state_in,
    device       float * dst,
    constant    uint & B,
    constant    uint & T,
    constant    uint & C,
    constant    uint & H,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]])  {

    const uint head_size = 64; // TODO: support head_size = 128
    const uint batch_id = tgpig.x / H;
    const uint head_id = tgpig.x % H;
    const uint tid = tpitg.x;

    if (batch_id >= B || head_id >= H) {
        return;
    }

    const uint state_size = C * head_size;
    const uint n_seq_tokens = T / B;

    threadgroup float _k[head_size];
    threadgroup float _r[head_size];
    threadgroup float _tf[head_size];
    threadgroup float _td[head_size];

    float state[head_size];

    for (uint i = 0; i < head_size; i++) {
        state[i] = state_in[batch_id * state_size + head_id * head_size * head_size
                          + i * head_size + tid];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    _tf[tid] = tf[head_id * head_size + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint start_t = batch_id * n_seq_tokens * C + head_id * head_size + tid;
    const uint end_t = (batch_id + 1) * n_seq_tokens * C + head_id * head_size + tid;

    for (uint t = start_t; t < end_t; t += C) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float v_val = v[t];
        float y = 0.0;

        for (uint j = 0; j < head_size; j += 4) {
            float4 k_vec = float4(_k[j], _k[j+1], _k[j+2], _k[j+3]);
            float4 r_vec = float4(_r[j], _r[j+1], _r[j+2], _r[j+3]);
            float4 tf_vec = float4(_tf[j], _tf[j+1], _tf[j+2], _tf[j+3]);
            float4 td_vec = float4(_td[j], _td[j+1], _td[j+2], _td[j+3]);
            float4 s_vec = float4(state[j], state[j+1], state[j+2], state[j+3]);

            float4 kv = k_vec * v_val;

            float4 temp = tf_vec * kv + s_vec;
            y += dot(r_vec, temp);

            s_vec = s_vec * td_vec + kv;
            state[j]   = s_vec[0];
            state[j+1] = s_vec[1];
            state[j+2] = s_vec[2];
            state[j+3] = s_vec[3];
        }

        dst[t] = y;
    }

    for (uint i = 0; i < head_size; i++) {
        dst[T * C + batch_id * state_size + head_id * head_size * head_size
            + i * head_size + tid] = state[i];
    }
}

kernel void kernel_rwkv_wkv7_f32(
    device const float * r,
    device const float * w,
    device const float * k,
    device const float * v,
    device const float * a,
    device const float * b,
    device const float * state_in,
    device       float * dst,
    constant    uint & B,
    constant    uint & T,
    constant    uint & C,
    constant    uint & H,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]])  {

    const uint head_size = 64; // TODO: support head_size = 128
    const uint batch_id = tgpig.x / H;
    const uint head_id = tgpig.x % H;
    const uint tid = tpitg.x;

    if (batch_id >= B || head_id >= H) {
        return;
    }

    const uint state_size = C * head_size;
    const uint n_seq_tokens = T / B;

    threadgroup float _r[head_size];
    threadgroup float _w[head_size];
    threadgroup float _k[head_size];
    threadgroup float _a[head_size];
    threadgroup float _b[head_size];

    float state[head_size];

    for (uint i = 0; i < head_size; i++) {
        state[i] = state_in[batch_id * state_size + head_id * head_size * head_size
                          + tid * head_size + i];
    }

    const uint start_t = batch_id * n_seq_tokens * C + head_id * head_size + tid;
    const uint end_t = (batch_id + 1) * n_seq_tokens * C + head_id * head_size + tid;

    for (uint t = start_t; t < end_t; t += C) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float v_val = v[t];
        float y = 0.0, sa = 0.0;

        float4 sa_vec(0.0);

        for (uint j = 0; j < head_size; j += 4) {
            float4 a_vec = float4(_a[j], _a[j+1], _a[j+2], _a[j+3]);
            float4 s_vec = float4(state[j], state[j+1], state[j+2], state[j+3]);
            sa_vec += a_vec * s_vec;
        }
        sa = sa_vec[0] + sa_vec[1] + sa_vec[2] + sa_vec[3];

        for (uint j = 0; j < head_size; j += 4) {
            float4 r_vec = float4(_r[j], _r[j+1], _r[j+2], _r[j+3]);
            float4 w_vec = float4(_w[j], _w[j+1], _w[j+2], _w[j+3]);
            float4 k_vec = float4(_k[j], _k[j+1], _k[j+2], _k[j+3]);
            float4 b_vec = float4(_b[j], _b[j+1], _b[j+2], _b[j+3]);
            float4 s_vec = float4(state[j], state[j+1], state[j+2], state[j+3]);

            float4 kv = k_vec * v_val;

            s_vec = s_vec * w_vec + kv + sa * b_vec;
            y += dot(s_vec, r_vec);

            state[j]   = s_vec[0];
            state[j+1] = s_vec[1];
            state[j+2] = s_vec[2];
            state[j+3] = s_vec[3];
        }

        dst[t] = y;
    }

    for (uint i = 0; i < head_size; i++) {
        dst[T * C + batch_id * state_size + head_id * head_size * head_size
            + tid * head_size + i] = state[i];
    }
}

kernel void kernel_argmax_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    device const float * x_row = (device const float *) ((device const char *) src0 + tgpig * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = tpitg; i00 < args.ne00; i00 += ntg) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    device int32_t * dst_i32 = (device int32_t *) dst;

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        dst_i32[tgpig] = arg_val_reduced;

        return;
    }

    dst_i32[tgpig] = arg_val;
}

kernel void kernel_norm_f32(
        constant ggml_metal_kargs_norm & args,
        device const char * src0,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    device const float4 * x = (device const float4 *) (src0 + tgpig*args.nb01);

    float4 sumf4(0.0f);

    float sumf = 0.0f;

    for (int i00 = tpitg; i00 < args.ne00_4; i00 += ntg) {
        sumf4 += x[i00];
    }
    sumf = sumf4[0] + sumf4[1] + sumf4[2] + sumf4[3];
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean = sumf/args.ne00;

    device float4 * y = (device float4 *) dst + tgpig*args.ne00_4;

    sumf = 0.0f;
    for (int i00 = tpitg; i00 < args.ne00_4; i00 += ntg) {
        y[i00] = x[i00] - mean;
        sumf += dot(y[i00], y[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float variance = sumf/args.ne00;

    const float scale = 1.0f/sqrt(variance + args.eps);
    for (int i00 = tpitg; i00 < args.ne00_4; i00 += ntg) {
        y[i00] = y[i00] * scale;
    }
}

// F == 1 : rms_norm (no fuse)
// F == 2 : rms_norm + mul
// F == 3 : rms_norm + mul + add
template <short F>
kernel void kernel_rms_norm_fuse_impl(
        constant ggml_metal_kargs_rms_norm & args,
        device const char * src0,
        device const char * src1_0,
        device const char * src1_1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const float4 * x = (device const float4 *) (src0 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);

    device const float4 * f0 = (device const float4 *) (src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);
    device const float4 * f1 = (device const float4 *) (src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);

    float sumf = 0.0f;

    // parallel sum
    for (int i00 = tpitg.x; i00 < args.ne00_4; i00 += ntg.x) {
        sumf += dot(x[i00], x[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf/args.ne00;
    const float scale = 1.0f/sqrt(mean + args.eps);

    device float4 * y = (device float4 *) (dst + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);
    for (int i00 = tpitg.x; i00 < args.ne00_4; i00 += ntg.x) {
        if (F == 1) {
            y[i00] = (x[i00]*scale);
        }
        if (F == 2) {
            y[i00] = (x[i00]*scale)*f0[i00];
        }
        if (F == 3) {
            y[i00] = (x[i00]*scale)*f0[i00] + f1[i00];
        }
    }
}

typedef decltype(kernel_rms_norm_fuse_impl<1>) kernel_rms_norm_fuse_t;

template [[host_name("kernel_rms_norm_f32")]]         kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<1>;
template [[host_name("kernel_rms_norm_mul_f32")]]     kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<2>;
template [[host_name("kernel_rms_norm_mul_add_f32")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<3>;

static inline uint fairy2i_mul_bf16_to_f32_bits_rne_ftz_safe(ushort a, ushort b) {
    const uint abs_a = (uint) a & 0x7fffU;
    const uint abs_b = (uint) b & 0x7fffU;
    const uint exp_a = abs_a >> 7;
    const uint exp_b = abs_b >> 7;
    if (exp_a == 0xffU || exp_b == 0xffU) {
        return fairy2i_mul_bf16_to_f32_bits_rne(a, b);
    }
    if (abs_a == 0U || abs_b == 0U) {
        return ((((uint) a ^ (uint) b) >> 15) & 1U) << 31;
    }

    if (exp_a > 0U && exp_b > 0U && exp_a + exp_b >= 128U) {
        return as_type<uint>(fairy2i_bf16_to_f32(a) * fairy2i_bf16_to_f32(b));
    }
    return fairy2i_mul_bf16_to_f32_bits_rne(a, b);
}

static inline ushort fairy2i_silu_qat_bf16(ushort value_bits) {
    const float value  = fairy2i_bf16_to_f32(value_bits);
    const float result = value >= 0.0f ? value / (1.0f + exp(-value)) : value * exp(value) / (1.0f + exp(value));
    return fairy2i_f32_to_bf16(result);
}

static inline ushort fairy2i_mul_qat_bf16(ushort lhs_bits, ushort rhs_bits) {
    const float lhs = fairy2i_bf16_to_f32(lhs_bits);
    const float rhs = fairy2i_bf16_to_f32(rhs_bits);
    return fairy2i_f32_to_bf16(lhs * rhs);
}

kernel void kernel_fairy2i_silu_exact_f32(constant ggml_metal_kargs_fairy2i_elementwise_exact & args,
                                          const device uint *                                   src0,
                                          device uint *                                         dst,
                                          uint gid [[thread_position_in_grid]]) {
    if (gid >= args.ne) {
        return;
    }

    const ulong         row        = gid / args.ne0;
    const ulong         col        = gid - row * args.ne0;
    const device uint * src0_row   = (const device uint *) ((const device uchar *) src0 + row * args.src0_nb1);
    const ushort        value_bits = (ushort) (src0_row[col] >> 16);
    const uint          abs_bits   = (uint) value_bits & 0x7fffU;
    const uint          exponent   = abs_bits >> 7;

    ushort result_bits;
    if (exponent <= 1U) {
        // For BF16 subnormals and the smallest normal bin, SiLU rounds as x*0.5.
        // Construct that product from payload bits because Apple GPUs flush an F32
        // subnormal produced by native arithmetic.
        const uint product_bits = fairy2i_mul_bf16_to_f32_bits_rne(value_bits, (ushort) 0x3f00U);
        result_bits             = fairy2i_f32_bits_to_bf16_rne(product_bits);
    } else {
        const float value = fairy2i_bf16_to_f32(value_bits);
        if ((value_bits & 0x8000U) == 0) {
            result_bits = fairy2i_f32_to_bf16(value / (1.0f + exp(-value)));
        } else if (value < -80.0f) {
            // exp(value) is F32-subnormal in the part of the negative tail that
            // can still round to a non-zero BF16 SiLU result. Build it from two
            // normal factors and keep the product as payload bits. At this scale,
            // 1 + exp(value) rounds exactly to 1 in F32.
            const uint exp_bits =
                fairy2i_mul_f32_bits_rne(as_type<uint>(exp(value + 64.0f)), as_type<uint>(exp(-64.0f)));
            const uint numerator_bits = fairy2i_mul_f32_bits_rne(((uint) value_bits) << 16, exp_bits);
            result_bits               = fairy2i_f32_bits_to_bf16_rne(numerator_bits);
        } else {
            const float exp_value = exp(value);
            result_bits           = fairy2i_f32_to_bf16((value * exp_value) / (1.0f + exp_value));
        }
    }

    dst[gid] = (uint) result_bits << 16;
}

kernel void kernel_fairy2i_silu_qat_f32(constant ggml_metal_kargs_fairy2i_elementwise_exact & args,
                                        const device uint *                                   src0,
                                        device uint *                                         dst,
                                        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.ne) {
        return;
    }

    const ulong         row        = gid / args.ne0;
    const ulong         col        = gid - row * args.ne0;
    const device uint * src0_row   = (const device uint *) ((const device uchar *) src0 + row * args.src0_nb1);
    const ushort        value_bits = (ushort) (src0_row[col] >> 16);
    dst[gid]                       = (uint) fairy2i_silu_qat_bf16(value_bits) << 16;
}

kernel void kernel_fairy2i_mul_exact_f32(constant ggml_metal_kargs_fairy2i_elementwise_exact & args,
                                         const device uint *                                   src0,
                                         const device uint *                                   src1,
                                         device uint *                                         dst,
                                         uint gid [[thread_position_in_grid]]) {
    if (gid >= args.ne) {
        return;
    }

    const ulong         row          = gid / args.ne0;
    const ulong         col          = gid - row * args.ne0;
    const device uint * src0_row     = (const device uint *) ((const device uchar *) src0 + row * args.src0_nb1);
    const device uint * src1_row     = (const device uint *) ((const device uchar *) src1 + row * args.src1_nb1);
    const ushort        lhs_bits     = (ushort) (src0_row[col] >> 16);
    const ushort        rhs_bits     = (ushort) (src1_row[col] >> 16);
    const uint          product_bits = fairy2i_mul_bf16_to_f32_bits_rne_ftz_safe(lhs_bits, rhs_bits);
    dst[gid]                         = (uint) fairy2i_f32_bits_to_bf16_rne(product_bits) << 16;
}

kernel void kernel_fairy2i_mul_qat_f32(constant ggml_metal_kargs_fairy2i_elementwise_exact & args,
                                       const device uint *                                   src0,
                                       const device uint *                                   src1,
                                       device uint *                                         dst,
                                       uint gid [[thread_position_in_grid]]) {
    if (gid >= args.ne) {
        return;
    }

    const ulong         row      = gid / args.ne0;
    const ulong         col      = gid - row * args.ne0;
    const device uint * src0_row = (const device uint *) ((const device uchar *) src0 + row * args.src0_nb1);
    const device uint * src1_row = (const device uint *) ((const device uchar *) src1 + row * args.src1_nb1);
    const ushort        lhs_bits = (ushort) (src0_row[col] >> 16);
    const ushort        rhs_bits = (ushort) (src1_row[col] >> 16);
    dst[gid]                     = (uint) fairy2i_mul_qat_bf16(lhs_bits, rhs_bits) << 16;
}

kernel void kernel_fairy2i_swiglu_qat_f32(constant ggml_metal_kargs_fairy2i_elementwise_exact & args,
                                          const device uint *                                   gate,
                                          const device uint *                                   up,
                                          device uint *                                         dst,
                                          uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint row = gid.y;
    if (col >= args.ne0 || (ulong) row * args.ne0 >= args.ne) {
        return;
    }

    const device uint * gate_row  = (const device uint *) ((const device uchar *) gate + (ulong) row * args.src0_nb1);
    const device uint * up_row    = (const device uint *) ((const device uchar *) up + (ulong) row * args.src1_nb1);
    const ushort        gate_bits = (ushort) (gate_row[col] >> 16);
    const ushort        up_bits   = (ushort) (up_row[col] >> 16);

    // Materialize the SiLU result as BF16 bits before multiplication. This is
    // the checkpoint QAT boundary, even though the intermediate tensor is no
    // longer written to device memory.
    const ushort silu_bits            = fairy2i_silu_qat_bf16(gate_bits);
    const ushort out_bits             = fairy2i_mul_qat_bf16(silu_bits, up_bits);
    dst[(ulong) row * args.ne0 + col] = (uint) out_bits << 16;
}

kernel void kernel_fairy2i_pack_bf16_exact(constant ulong &    ne,
                                           const device uint * src0,
                                           device ushort *     dst,
                                           uint                gid [[thread_position_in_grid]]) {
    if (gid < ne) {
        dst[gid] = (ushort) (src0[gid] >> 16);
    }
}

kernel void kernel_fairy2i_round_bf16_f32(
        constant ulong & ne,
        device const float * src0,
        device       uint * dst,
        uint gid [[thread_position_in_grid]]) {
    if (gid < ne) {
        dst[gid] = (uint) fairy2i_f32_to_bf16(src0[gid]) << 16;
    }
}

// RMSNorm squares are non-negative, so an F32 accumulator can only be subnormal
// when the first non-zero square is subnormal. BF16 exponent 52 is the
// round-to-zero/subnormal boundary and exponent 64 starts the normal F32 range.
// Non-finite values also use the bit-domain helper to preserve canonical NaNs.
static inline bool fairy2i_rms_square_requires_software_fma(ushort value) {
    const uint abs_value = (uint) value & 0x7fffU;
    const uint exponent  = abs_value >> 7;
    return exponent == 0xffU ||
           (exponent >= 52U && exponent < 64U);
}

static inline bool fairy2i_rms_accumulator_requires_software_fma(uint value) {
    const uint abs_value = value & 0x7fffffffU;
    const uint exponent  = abs_value >> 23;
    return (abs_value != 0U && exponent == 0U) || exponent == 0xffU;
}

kernel void kernel_fairy2i_rms_norm_exact_f32(
        constant ggml_metal_kargs_fairy2i_rms_norm_exact & args,
        device const char * src0,
        device const char * weight,
        device       char * dst,
        threadgroup float * inv_rms_shared [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort  tiitg [[thread_index_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const int i1 = (int) tgpig.x;
    const int i2 = (int) tgpig.y;
    const int i3 = (int) tgpig.z;

    device const float * x = (device const float *)
        (src0 + (ulong) i1 * args.nb01 + (ulong) i2 * args.nb02 + (ulong) i3 * args.nb03);
    device float * y = (device float *)
        (dst + (ulong) i1 * args.nb1 + (ulong) i2 * args.nb2 + (ulong) i3 * args.nb3);

    const int w1 = i1 % args.ne11;
    const int w2 = i2 % args.ne12;
    const int w3 = i3 % args.ne13;
    device const char * w_row =
        weight + (ulong) w1 * args.nb11 + (ulong) w2 * args.nb12 + (ulong) w3 * args.nb13;

    float fast_sum = 0.0f;
    uint lane_requires_fallback = 0U;
    for (int i0 = tiitg; i0 < args.ne00; i0 += ntg.x) {
        const ushort x_bits = fairy2i_f32_to_bf16(x[i0]);
        if (fairy2i_rms_square_requires_software_fma(x_bits)) {
            lane_requires_fallback = 1U;
            continue;
        }

        const float value = fairy2i_bf16_to_f32(x_bits);
        fast_sum = precise::fma(value, value, fast_sum);
        if (fairy2i_rms_accumulator_requires_software_fma(as_type<uint>(fast_sum))) {
            lane_requires_fallback = 1U;
        }
    }

    const float row_fast_sum = simd_sum(fast_sum);
    const bool row_requires_fallback =
        simd_max(lane_requires_fallback) != 0U ||
        fairy2i_rms_accumulator_requires_software_fma(as_type<uint>(row_fast_sum));

    if (tiitg == 0) {
        uint sum_bits = as_type<uint>(row_fast_sum);
        if (row_requires_fallback) {
            sum_bits = 0;
            for (int i0 = 0; i0 < args.ne00; ++i0) {
                const ushort x_bits = fairy2i_f32_to_bf16(x[i0]);
                if (fairy2i_rms_square_requires_software_fma(x_bits) ||
                    fairy2i_rms_accumulator_requires_software_fma(sum_bits)) {
                    sum_bits = fairy2i_fma_bf16_bf16_f32_bits_rne(x_bits, x_bits, sum_bits);
                } else {
                    const float value = fairy2i_bf16_to_f32(x_bits);
                    sum_bits = as_type<uint>(
                        precise::fma(value, value, as_type<float>(sum_bits)));
                }
            }
        }
        const uint mean_bits =
            fairy2i_div_f32_by_positive_int_bits_rne(sum_bits, (uint) args.ne00);
        const uint denominator_bits = fairy2i_add_f32_bits_rne(mean_bits, as_type<uint>(args.eps));
        const uint denominator_exp = (denominator_bits & 0x7fffffffU) >> 23;
        if ((denominator_bits & 0x7fffffffU) != 0 && denominator_exp == 0) {
            const uint scaled_bits = fairy2i_mul_f32_bits_rne(denominator_bits, as_type<uint>(0x1p126f));
            const float root = precise::sqrt(as_type<float>(scaled_bits));
            const uint reciprocal_bits =
                fairy2i_reciprocal_positive_normal_f32_bits_rne(as_type<uint>(root));
            inv_rms_shared[0] = as_type<float>(
                fairy2i_mul_f32_bits_rne(reciprocal_bits, as_type<uint>(0x1p63f)));
        } else {
            const float root = precise::sqrt(as_type<float>(denominator_bits));
            inv_rms_shared[0] = as_type<float>(
                fairy2i_reciprocal_positive_normal_f32_bits_rne(as_type<uint>(root)));
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = inv_rms_shared[0];
    for (int i0 = tiitg; i0 < args.ne00; i0 += ntg.x) {
        const ushort x_bits = fairy2i_f32_to_bf16(x[i0]);
        const uint normalized_f32_bits =
            fairy2i_mul_f32_bits_rne_ftz_safe(((uint) x_bits) << 16, as_type<uint>(inv_rms));
        const ushort normalized_bits = fairy2i_f32_to_bf16(as_type<float>(normalized_f32_bits));
        const float w = *((device const float *) (w_row + (ulong) (i0 % args.ne10) * args.nb10));
        const ushort w_bits = fairy2i_f32_to_bf16(w);
        const uint result_f32_bits =
            fairy2i_mul_bf16_to_f32_bits_rne_ftz_safe(normalized_bits, w_bits);
        const ushort result_bits = fairy2i_f32_to_bf16(as_type<float>(result_f32_bits));
        ((device uint *) y)[i0] = ((uint) result_bits) << 16;
    }
}

kernel void kernel_fairy2i_rms_norm_qat_f32(
        constant ggml_metal_kargs_fairy2i_rms_norm_exact & args,
        device const char * src0,
        device const char * weight,
        device       char * dst,
        threadgroup float * inv_rms_shared [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort  tiitg [[thread_index_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const int i1 = (int) tgpig.x;
    const int i2 = (int) tgpig.y;
    const int i3 = (int) tgpig.z;

    device const float * x = (device const float *)
        (src0 + (ulong) i1 * args.nb01 + (ulong) i2 * args.nb02 + (ulong) i3 * args.nb03);
    device float * y = (device float *)
        (dst + (ulong) i1 * args.nb1 + (ulong) i2 * args.nb2 + (ulong) i3 * args.nb3);

    const int w1 = i1 % args.ne11;
    const int w2 = i2 % args.ne12;
    const int w3 = i3 % args.ne13;
    device const char * w_row =
        weight + (ulong) w1 * args.nb11 + (ulong) w2 * args.nb12 + (ulong) w3 * args.nb13;

    float sum = 0.0f;
    for (int i0 = tiitg; i0 < args.ne00; i0 += ntg.x) {
        const float value = fairy2i_bf16_to_f32(fairy2i_f32_to_bf16(x[i0]));
        sum = precise::fma(value, value, sum);
    }
    sum = simd_sum(sum);

    const ushort tiisg = tiitg % N_SIMDWIDTH;
    const ushort sgitg = tiitg / N_SIMDWIDTH;
    if (tiisg == 0) {
        inv_rms_shared[sgitg] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiitg == 0) {
        float row_sum = 0.0f;
        for (ushort isg = 0; isg < ntg.x / N_SIMDWIDTH; ++isg) {
            row_sum = precise::fma(1.0f, inv_rms_shared[isg], row_sum);
        }
        inv_rms_shared[0] = 1.0f / precise::sqrt(row_sum / (float) args.ne00 + args.eps);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = inv_rms_shared[0];
    for (int i0 = tiitg; i0 < args.ne00; i0 += ntg.x) {
        const float value = fairy2i_bf16_to_f32(fairy2i_f32_to_bf16(x[i0]));
        const float normalized = fairy2i_round_to_bf16_f32(value * inv_rms);
        const float w = *((device const float *) (w_row + (ulong) (i0 % args.ne10) * args.nb10));
        const float w_bf16 = fairy2i_bf16_to_f32(fairy2i_f32_to_bf16(w));
        ((device uint *) y)[i0] = (uint) fairy2i_f32_to_bf16(normalized * w_bf16) << 16;
    }
}

kernel void kernel_l2_norm_f32(
        constant ggml_metal_kargs_l2_norm & args,
        device const char * src0,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    device const float4 * x = (device const float4 *) (src0 + tgpig*args.nb01);

    float sumf = 0.0f;

    // parallel sum
    for (int i00 = tpitg; i00 < args.ne00_4; i00 += ntg) {
        sumf += dot(x[i00], x[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float scale = 1.0f/sqrt(max(sumf, args.eps));

    device float4 * y = (device float4 *) dst + tgpig*args.ne00_4;
    for (int i00 = tpitg; i00 < args.ne00_4; i00 += ntg) {
        y[i00] = x[i00] * scale;
    }
}

kernel void kernel_group_norm_f32(
        constant ggml_metal_kargs_group_norm & args,
        device const float * src0,
        device       float * dst,
        threadgroup float  * buf [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    const int64_t ne = args.ne00*args.ne01*args.ne02;
    const int64_t gs = args.ne00*args.ne01*((args.ne02 + args.ngrp - 1) / args.ngrp);

    int start = tgpig * gs;
    int end   = start + gs;

    start += tpitg;

    if (end >= ne) {
        end = ne;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += ntg) {
        tmp += src0[j];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    tmp = simd_sum(tmp);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float mean = tmp / gs;
    tmp = 0.0f;

    for (int j = start; j < end; j += ntg) {
        float xi = src0[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = simd_sum(tmp);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float variance = tmp / gs;
    const float scale = 1.0f/sqrt(variance + args.eps);
    for (int j = start; j < end; j += ntg) {
        dst[j] *= scale;
    }
}

// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 1 + il/2);

    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// function for calculate inner product between half a q4_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 2 + il/2);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

// function for calculate inner product between half a q5_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 3 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }

    return d * (sumy * -16.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// function for calculate inner product between half a q5_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_1/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 4 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }

    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = tot;
        }
    }
}

constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_nxpsg [[function_constant(FC_MUL_MV + 1)]];

template<typename block_q_type, short NR0, typename args_t>
void mul_vec_q_n_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 16;

    const int nb = args.ne00/QK4_0;

    const int r0 = (tgpig.x*NSG + sgitg)*NR0;
  //const int r0 =  tgpig.x*NR0;
    const int r1 =  tgpig.y;
    const int im =  tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

  //device const block_q_type * x = (device const block_q_type *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    // pointers to src0 rows
    device const block_q_type * ax[NR0];
    FOR_UNROLL (int row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const block_q_type *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = {0.f};

    const short ix = (tiisg/(NW/NQ));
    const short il = (tiisg%(NW/NQ))*8;

    //const int ib0 = sgitg*NQ + ix;
    const int ib0 = ix;

    float yl[16]; // src1 vector cache

    //device const float * yb = y + ix*QK4_0 + il;
    device const float * yb = y + ib0*QK4_0 + il;

    // each thread in a SIMD group deals with half a block.
    //for (int ib = ib0; ib < nb; ib += NSG*NQ) {
    for (int ib = ib0; ib < nb; ib += NQ) {
        float sumy[2] = { 0.f, 0.f };

        FOR_UNROLL (short i = 0; i < 8; i += 2) {
            sumy[0]  += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1]/256.f;

            sumy[1]  += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16]/16.f;
            yl[i + 9] = yb[i + 17]/4096.f;
        }

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }

        yb += QK4_0 * 16;
        //yb += NSG*NQ*QK4_0;
    }

    device float * dst_f32 = (device float *) dst + im*args.ne0*args.ne1 + r1*args.ne0;

    //helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);

    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && r0 + row < args.ne01) {
            dst_f32[r0 + row] = tot;
        }
    }
}

kernel void kernel_mul_mv_q4_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q4_1_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
     mul_vec_q_n_f32_impl<block_q4_1, N_R0_Q4_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q5_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_0, N_R0_Q5_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q5_1_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_1, N_R0_Q5_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<short NR0, typename args_t>
void kernel_mul_mv_q8_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    const int nb = args.ne00/QK8_0;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

  //device const block_q8_0 * x = (device const block_q8_0 *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    // pointers to src0 rows
    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);

    const int ib0 = sgitg*NQ + ix;

    float yl[NQ];

    device const float * yb = y + ib0*QK8_0 + il*NQ;

    // each thread in a SIMD group deals with NQ quants at a time
    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += NSG*NQ*QK8_0;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// mat-vec kernel processing in chunks of float4
// chpb - chunks per quantization block
template<short r1ptg, typename q_t, short chpb, void (*deq_t4)(device const q_t *, short, thread float4 &) >
void kernel_mul_mv_ext_q4_f32_impl(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short chpt = 4; // chunks per thread

  //const short nxpsg = (32);
    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%args.ne12;
    const int i13 = i1m/args.ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4 * y4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4 *) src1;
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb; // current chunk index

    for (int ich = tx; 4*ich < args.ne00; ich += chpt*nxpsg) {
        float4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch*nxpsg]);
            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4[ir1] += chpt*nxpsg;
        }
    }

    // reduce only the threads in each row
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
        }

        //sumf[ir1] = simd_sum(sumf[ir1]);
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// mat-vec kernel processing in chunks of float4x4
template<short r1ptg, typename q_t, short chpb, void (*deq_t4x4)(device const q_t *, short, thread float4x4 &) >
void kernel_mul_mv_ext_q4x4_f32_impl(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short chpt = 1;

  //const short nxpsg = (32);
    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%args.ne12;
    const int i13 = i1m/args.ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4x4 * y4x4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4x4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4x4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4x4 *) src1;
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb;

    for (int ich = tx; 16*ich < args.ne00; ich += chpt*nxpsg) {
        float4x4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4x4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] +=
                    dot(lx[ch][0], y4x4[ir1][ch*nxpsg][0]) +
                    dot(lx[ch][1], y4x4[ir1][ch*nxpsg][1]) +
                    dot(lx[ch][2], y4x4[ir1][ch*nxpsg][2]) +
                    dot(lx[ch][3], y4x4[ir1][ch*nxpsg][3]);

            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4x4[ir1] += chpt*nxpsg;
        }
    }

    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
        }

        //sumf[ir1] = simd_sum(sumf[ir1]);
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// dispatchers needed for compile-time nxpsg
// epb - elements per quantization block
template<short r1ptg, typename q_t, short epb, void (*deq_t4)(device const q_t *, short, thread float4 &)>
kernel void kernel_mul_mv_ext_q4_f32_disp(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ext_q4_f32_impl<r1ptg, q_t, epb/4, deq_t4>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}

template<short r1ptg, typename q_t, short epb, void (*deq_t4x4)(device const q_t *, short, thread float4x4 &)>
kernel void kernel_mul_mv_ext_q4x4_f32_disp(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ext_q4x4_f32_impl<r1ptg, q_t, epb/16, deq_t4x4>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_ext_q4_f32_disp  <2, block_q8_0, 32,  dequantize_q8_0_t4>) mul_mv_ext_q4_f32_t;
typedef decltype(kernel_mul_mv_ext_q4x4_f32_disp<2, block_q4_K, 256, dequantize_q4_K>)    mul_mv_ext_q4x4_f32_t;

template [[host_name("kernel_mul_mv_ext_f32_f32_r1_2")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, float4,       4,  dequantize_f32_t4>;
template [[host_name("kernel_mul_mv_ext_f32_f32_r1_3")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, float4,       4,  dequantize_f32_t4>;
template [[host_name("kernel_mul_mv_ext_f32_f32_r1_4")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, float4,       4,  dequantize_f32_t4>;
template [[host_name("kernel_mul_mv_ext_f32_f32_r1_5")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, float4,       4,  dequantize_f32_t4>;

template [[host_name("kernel_mul_mv_ext_f16_f32_r1_2")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, half4,        4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_3")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, half4,        4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_4")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, half4,        4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_5")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, half4,        4,  dequantize_f16_t4>;

template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q4_0,   32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q4_0,   32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q4_0,   32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q4_0,   32, dequantize_q4_0_t4>;

template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q4_1,   32, dequantize_q4_1_t4>;
template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q4_1,   32, dequantize_q4_1_t4>;
template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q4_1,   32, dequantize_q4_1_t4>;
template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q4_1,   32, dequantize_q4_1_t4>;

template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q5_0,   32, dequantize_q5_0_t4>;
template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q5_0,   32, dequantize_q5_0_t4>;
template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q5_0,   32, dequantize_q5_0_t4>;
template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q5_0,   32, dequantize_q5_0_t4>;

template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q5_1,   32, dequantize_q5_1_t4>;
template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q5_1,   32, dequantize_q5_1_t4>;
template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q5_1,   32, dequantize_q5_1_t4>;
template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q5_1,   32, dequantize_q5_1_t4>;

template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q8_0,   32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q8_0,   32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q8_0,   32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q8_0,   32, dequantize_q8_0_t4>;

template [[host_name("kernel_mul_mv_ext_mxfp4_f32_r1_2")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_mxfp4,  32, dequantize_mxfp4_t4>;
template [[host_name("kernel_mul_mv_ext_mxfp4_f32_r1_3")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_mxfp4,  32, dequantize_mxfp4_t4>;
template [[host_name("kernel_mul_mv_ext_mxfp4_f32_r1_4")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_mxfp4,  32, dequantize_mxfp4_t4>;
template [[host_name("kernel_mul_mv_ext_mxfp4_f32_r1_5")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_mxfp4,  32, dequantize_mxfp4_t4>;

template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_2")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_3")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_4")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_5")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_iq4_nl, 32, dequantize_iq4_nl_t4>;

template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q4_K, 256, dequantize_q4_K>;

template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q5_K, 256, dequantize_q5_K>;

template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q6_K, 256, dequantize_q6_K>;

template<typename T0, typename T1, short NR0, typename args_t>
void kernel_mul_mv_t_t_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB = 32;
    constexpr short NF = 8;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

  //device const T0 * x = (device const T0 *) (src0 + offset0);
    device const T1 * y = (device const T1 *) (src1 + offset1);

    // pointers to src0 rows
    device const T0 * ax [NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const T0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    T1 yl[NF];

    device const T1 * yb = y + (ib0*NB + il*NF);

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const T0 * xb = ax[row] + (ib*NB + il*NF);

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF; ++i) {
                sumq += xb[i] * yl[i];
            }

            sumf[row] += sumq;
        }

        yb += NSG*NF*NW;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename T0, typename T1, short NR0>
kernel void kernel_mul_mv_t_t(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_impl<T0, T1, NR0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t<half, half, N_R0_F>) mul_mv_t_t;

template [[host_name("kernel_mul_mv_f32_f32")]]   kernel mul_mv_t_t kernel_mul_mv_t_t<float, float, N_R0_F>;
template [[host_name("kernel_mul_mv_f16_f32")]]   kernel mul_mv_t_t kernel_mul_mv_t_t<half,  float, N_R0_F>;
template [[host_name("kernel_mul_mv_f16_f16")]]   kernel mul_mv_t_t kernel_mul_mv_t_t<half,  half,  N_R0_F>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_bf16_f32")]]  kernel mul_mv_t_t kernel_mul_mv_t_t<bfloat, float,  N_R0_F>;
template [[host_name("kernel_mul_mv_bf16_bf16")]] kernel mul_mv_t_t kernel_mul_mv_t_t<bfloat, bfloat, N_R0_F>;
#endif

template<typename T0, typename T04, typename T1, typename T14, short NR0, typename args_t>
void kernel_mul_mv_t_t_4_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const T1  * y  = (device const T1  *) (src1 + offset1);
    device const T14 * y4 = (device const T14 *) (src1 + offset1);

    // pointers to src0 rows
    device const T0  * ax [NR0];
    device const T04 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax [row] = (device const T0  *) ((device char *) src0 + offset0);
        ax4[row] = (device const T04 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    T14 yl4[NF4];

    device const T14 * yb4 = y4 + (ib0*NB + il*NF)/4;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const T04 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), float4(yl4[i]));
            }

            sumf[row] += sumq;
        }

        yb4 += NSG*NF*NW/4;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename T0, typename T04, typename T1, typename T14, short NR0>
kernel void kernel_mul_mv_t_t_4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, NR0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t_4<half, half4, half, half4, N_R0_F>) mul_mv_t_t_4;

template [[host_name("kernel_mul_mv_f32_f32_4")]]   kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<float, float4, float, float4, N_R0_F>;
template [[host_name("kernel_mul_mv_f16_f32_4")]]   kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<half,  half4,  float, float4, N_R0_F>;
template [[host_name("kernel_mul_mv_f16_f16_4")]]   kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<half,  half4,  half,  half4,  N_R0_F>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_bf16_f32_4")]]  kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<bfloat, bfloat4, float,  float4,  N_R0_F>;
template [[host_name("kernel_mul_mv_bf16_bf16_4")]] kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<bfloat, bfloat4, bfloat, bfloat4, N_R0_F>;
#endif

#define N_MV_T_T 4

template<typename T04, typename T14, typename args_t>
void kernel_mul_mv_c4_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg) {
    const int r0 = tgpig.x*32 + tiisg;
    const int rb = tgpig.y*N_MV_T_T;
    const int im = tgpig.z;

    if (r0 >= args.ne01) {
        return;
    }

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const T04 * x = (device const T04 *) (src0 + offset0);

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1;

    for (int row = 0; row < N_MV_T_T; ++row) {
        int r1 = rb + row;
        if (r1 >= args.ne11) {
            break;
        }

        const uint64_t offset1 = r1*args.nb11 + (i12   )*args.nb12 + (i13   )*args.nb13;

        device const T14 * y = (device const T14 *) (src1 + offset1);

        dst_f32[(uint64_t)r1*args.ne0 + r0] = dot((float4) x[0], (float4) y[0]);
    }
}

template<typename T04, typename T14>
kernel void kernel_mul_mv_c4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_c4_impl<T04, T14, constant ggml_metal_kargs_mul_mv &>(
        args,
        src0,
        src1,
        dst,
        tgpig,
        tiisg);
}

typedef decltype(kernel_mul_mv_c4<half4, half4>) mul_mv_c4_t;

template [[host_name("kernel_mul_mv_f32_f32_c4")]]  kernel mul_mv_c4_t kernel_mul_mv_c4<float4,  float4>;
template [[host_name("kernel_mul_mv_f16_f32_c4")]]  kernel mul_mv_c4_t kernel_mul_mv_c4<half4,   float4>;
template [[host_name("kernel_mul_mv_f16_f16_c4")]]  kernel mul_mv_c4_t kernel_mul_mv_c4<half4,   half4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_bf16_f32_c4")]]  kernel mul_mv_c4_t kernel_mul_mv_c4<bfloat4, float4>;
template [[host_name("kernel_mul_mv_bf16_bf16_c4")]] kernel mul_mv_c4_t kernel_mul_mv_c4<bfloat4, bfloat4>;
#endif

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int i0, float ext_factor, float mscale,
    thread float * cos_theta, thread float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_fac(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    dims[0] = max(0.0f,         floor(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f, ceil(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

template<typename T>
kernel void kernel_rope_norm(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        if (i0 < args.n_dims) {
            const int ic = i0/2;

            const float theta = theta_base * pow(args.freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? ((device const float *) src2)[ic] : 1.0f;

            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            const float x0 = src[0];
            const float x1 = src[1];

            dst_data[0] = x0*cos_theta - x1*sin_theta;
            dst_data[1] = x0*sin_theta + x1*cos_theta;
        } else {
            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

template<typename T>
kernel void kernel_rope_neox(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        if (i0 < args.n_dims) {
            const int ic = i0/2;

            const float theta = theta_base * pow(args.freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? ((device const float *) src2)[ic] : 1.0f;

            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + ic*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + ic*args.nb0);

            const float x0 = src[0];
            const float x1 = src[args.n_dims/2];

            dst_data[0]             = x0*cos_theta - x1*sin_theta;
            dst_data[args.n_dims/2] = x0*sin_theta + x1*cos_theta;
        } else {
            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

template<typename T>
kernel void kernel_rope_multi(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;

    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        if (i0 < args.n_dims) {
            const int ic = i0/2;

            // mrope theta calculations
            // note: the rest is the same as kernel_rope_neox
            const int sect_dims = args.sect_0 + args.sect_1 + args.sect_2 + args.sect_3;
            const int sec_w01   = args.sect_0 + args.sect_1;               // end of section 1
            const int sec_w012  = args.sect_0 + args.sect_1 + args.sect_2; // end of section 2
            const int sector    = ic % sect_dims;

            float theta_base;
            if (sector < args.sect_0) {
                theta_base = (float) pos[i2];
            } else if (sector < sec_w01) {
                theta_base = (float) pos[i2 + args.ne02];
            } else if (sector < sec_w012) {
                theta_base = (float) pos[i2 + args.ne02 * 2];
            } else {
                theta_base = (float) pos[i2 + args.ne02 * 3];
            }
            // end of mrope

            const float theta = theta_base * pow(args.freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? ((device const float *) src2)[ic] : 1.0f;

            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + ic*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + ic*args.nb0);

            const float x0 = src[0];
            const float x1 = src[args.n_dims/2];

            dst_data[0]             = x0*cos_theta - x1*sin_theta;
            dst_data[args.n_dims/2] = x0*sin_theta + x1*cos_theta;
        } else {
            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

template<typename T>
kernel void kernel_rope_vision(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;

    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        if (i0 < 2*args.n_dims) { // different from kernel_rope_multi
            const int ic = i0/2;

            // mrope theta calculations (only support 2 dimensions)
            const int sect_dims = args.sect_0 + args.sect_1;
            const int sector    = ic % sect_dims;

            float p;
            float theta_base;
            if (sector < args.sect_1) {
                p = (float) sector;
                theta_base = (float) pos[i2];
            } else {
                p = (float) sector - args.sect_0;
                theta_base = (float) pos[i2 + args.ne02];
            }

            const float theta = theta_base * pow(args.freq_base, 2.0f * inv_ndims * p);
            // end of mrope

            const float freq_factor = src2 != src0 ? ((device const float *) src2)[ic] : 1.0f;

            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + ic*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + ic*args.nb0);

            const float x0 = src[0];
            const float x1 = src[args.n_dims]; // different from kernel_rope_multi

            dst_data[0]           = x0*cos_theta - x1*sin_theta;
            dst_data[args.n_dims] = x0*sin_theta + x1*cos_theta; // different from kernel_rope_multi
        } else {
            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

typedef decltype(kernel_rope_norm<float>) kernel_rope_norm_t;
typedef decltype(kernel_rope_neox<float>) kernel_rope_neox_t;
typedef decltype(kernel_rope_multi<float>) kernel_rope_multi_t;
typedef decltype(kernel_rope_vision<float>) kernel_rope_vision_t;

template [[host_name("kernel_rope_norm_f32")]] kernel kernel_rope_norm_t kernel_rope_norm<float>;
template [[host_name("kernel_rope_norm_f16")]] kernel kernel_rope_norm_t kernel_rope_norm<half>;

template [[host_name("kernel_rope_neox_f32")]] kernel kernel_rope_neox_t kernel_rope_neox<float>;
template [[host_name("kernel_rope_neox_f16")]] kernel kernel_rope_neox_t kernel_rope_neox<half>;

template [[host_name("kernel_rope_multi_f32")]] kernel kernel_rope_multi_t kernel_rope_multi<float>;
template [[host_name("kernel_rope_multi_f16")]] kernel kernel_rope_multi_t kernel_rope_multi<half>;

template [[host_name("kernel_rope_vision_f32")]] kernel kernel_rope_vision_t kernel_rope_vision<float>;
template [[host_name("kernel_rope_vision_f16")]] kernel kernel_rope_vision_t kernel_rope_vision<half>;

kernel void kernel_fairy2i_rope_neox_exact_f32(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        threadgroup float * trig [[threadgroup(0)]],
        ushort  tiitg [[thread_index_in_threadgroup]],
        ushort3 tptg  [[threads_per_threadgroup]],
        uint3   tgpig [[threadgroup_position_in_grid]]) {
    const int i3 = (int) tgpig.z;
    const int i2 = (int) tgpig.y;

    device const int32_t * pos = (device const int32_t *) src1;

    // Match the scalar oracle's recurrence instead of independently evaluating
    // pow() for each lane. Trig values remain F32 until the explicit BF16 barrier.
    if (tiitg == 0) {
        float corr_dims[2];
        rope_yarn_corr_dims(
            args.n_dims,
            args.n_ctx_orig,
            args.freq_base,
            args.beta_fast,
            args.beta_slow,
            corr_dims);

        float theta = (float) pos[i2];
        const float theta_scale = pow(args.freq_base, -2.0f / (float) args.n_dims);
        for (int i0 = 0; i0 < args.n_dims; i0 += 2) {
            const int ic = i0 / 2;
            const float freq_factor = src2 != src0 ? ((device const float *) src2)[ic] : 1.0f;
            float cos_theta;
            float sin_theta;
            rope_yarn(
                theta / freq_factor,
                args.freq_scale,
                corr_dims,
                i0,
                args.ext_factor,
                args.attn_factor,
                &cos_theta,
                &sin_theta);
            trig[i0 + 0] = fairy2i_round_to_bf16_f32(cos_theta);
            trig[i0 + 1] = fairy2i_round_to_bf16_f32(sin_theta);
            theta *= theta_scale;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The trig sequence depends only on token/batch, not on the attention head.
    // Flatten all heads into this threadgroup so it is generated exactly once.
    const int n_pairs = args.n_dims / 2;
    const int n_rotated = args.ne01 * n_pairs;
    for (int linear = tiitg; linear < n_rotated; linear += tptg.x) {
        const int i1 = linear / n_pairs;
        const int ic = linear - i1 * n_pairs;
        device const char * src_row =
            src0 + (ulong) i3 * args.nb03 + (ulong) i2 * args.nb02 + (ulong) i1 * args.nb01;
        device char * dst_row =
            dst + (ulong) i3 * args.nb3 + (ulong) i2 * args.nb2 + (ulong) i1 * args.nb1;
        const ushort x0_bits = fairy2i_f32_to_bf16(
            *((device const float *) (src_row + (ulong) ic * args.nb00)));
        const ushort x1_bits = fairy2i_f32_to_bf16(
            *((device const float *) (src_row + (ulong) (ic + args.n_dims / 2) * args.nb00)));
        const ushort cos_bits = fairy2i_f32_to_bf16(trig[2 * ic + 0]);
        const ushort sin_bits = fairy2i_f32_to_bf16(trig[2 * ic + 1]);

        const ushort x0_cos_bits =
            fairy2i_f32_to_bf16(as_type<float>(fairy2i_mul_bf16_to_f32_bits_rne_ftz_safe(x0_bits, cos_bits)));
        const ushort x1_sin_bits =
            fairy2i_f32_to_bf16(as_type<float>(fairy2i_mul_bf16_to_f32_bits_rne_ftz_safe(x1_bits, sin_bits)));
        const ushort x0_sin_bits =
            fairy2i_f32_to_bf16(as_type<float>(fairy2i_mul_bf16_to_f32_bits_rne_ftz_safe(x0_bits, sin_bits)));
        const ushort x1_cos_bits =
            fairy2i_f32_to_bf16(as_type<float>(fairy2i_mul_bf16_to_f32_bits_rne_ftz_safe(x1_bits, cos_bits)));

        const ushort out0_bits =
            fairy2i_add_bf16_bits_rne(x0_cos_bits, (ushort) ((uint) x1_sin_bits ^ 0x8000U));
        const ushort out1_bits = fairy2i_add_bf16_bits_rne(x0_sin_bits, x1_cos_bits);
        *((device uint *) (dst_row + (ulong) ic * args.nb0)) = ((uint) out0_bits) << 16;
        *((device uint *) (dst_row + (ulong) (ic + args.n_dims / 2) * args.nb0)) =
            ((uint) out1_bits) << 16;
    }

    const int n_tail = args.ne00 - args.n_dims;
    if (n_tail > 0) {
        const int n_passthrough = args.ne01 * n_tail;
        for (int linear = tiitg; linear < n_passthrough; linear += tptg.x) {
            const int i1 = linear / n_tail;
            const int i0 = args.n_dims + linear - i1 * n_tail;
            device const char * src_row =
                src0 + (ulong) i3 * args.nb03 + (ulong) i2 * args.nb02 + (ulong) i1 * args.nb01;
            device char * dst_row =
                dst + (ulong) i3 * args.nb3 + (ulong) i2 * args.nb2 + (ulong) i1 * args.nb1;
            const float x = *((device const float *) (src_row + (ulong) i0 * args.nb00));
            *((device uint *) (dst_row + (ulong) i0 * args.nb0)) =
                ((uint) fairy2i_f32_to_bf16(x)) << 16;
        }
    }
}

kernel void kernel_fairy2i_rope_neox_qat_f32(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg [[thread_index_in_threadgroup]],
        ushort3 tptg  [[threads_per_threadgroup]],
        uint3   tgpig [[threadgroup_position_in_grid]]) {
    const int i3 = (int) tgpig.z;
    const int i2 = (int) tgpig.y;
    const int i1 = (int) tgpig.x;

    float corr_dims[2];
    rope_yarn_corr_dims(
        args.n_dims,
        args.n_ctx_orig,
        args.freq_base,
        args.beta_fast,
        args.beta_slow,
        corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;
    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.0f / (float) args.n_dims;

    for (int i0 = 2 * tiitg; i0 < args.ne0; i0 += 2 * tptg.x) {
        device const char * src_row =
            src0 + (ulong) i3 * args.nb03 + (ulong) i2 * args.nb02 + (ulong) i1 * args.nb01;
        device char * dst_row =
            dst + (ulong) i3 * args.nb3 + (ulong) i2 * args.nb2 + (ulong) i1 * args.nb1;

        if (i0 < args.n_dims) {
            const int ic = i0 / 2;
            const float theta = theta_base * pow(args.freq_base, inv_ndims * (float) i0);
            const float freq_factor = src2 != src0 ? ((device const float *) src2)[ic] : 1.0f;

            float cos_theta;
            float sin_theta;
            rope_yarn(
                theta / freq_factor,
                args.freq_scale,
                corr_dims,
                i0,
                args.ext_factor,
                args.attn_factor,
                &cos_theta,
                &sin_theta);

            const float x0 = fairy2i_bf16_to_f32(fairy2i_f32_to_bf16(
                *((device const float *) (src_row + (ulong) ic * args.nb00))));
            const float x1 = fairy2i_bf16_to_f32(fairy2i_f32_to_bf16(
                *((device const float *) (src_row + (ulong) (ic + args.n_dims / 2) * args.nb00))));
            const float cos_bf16 = fairy2i_round_to_bf16_f32(cos_theta);
            const float sin_bf16 = fairy2i_round_to_bf16_f32(sin_theta);

            const float x0_cos = fairy2i_round_to_bf16_f32(x0 * cos_bf16);
            const float x1_sin = fairy2i_round_to_bf16_f32(x1 * sin_bf16);
            const float x0_sin = fairy2i_round_to_bf16_f32(x0 * sin_bf16);
            const float x1_cos = fairy2i_round_to_bf16_f32(x1 * cos_bf16);

            *((device uint *) (dst_row + (ulong) ic * args.nb0)) =
                (uint) fairy2i_f32_to_bf16(x0_cos - x1_sin) << 16;
            *((device uint *) (dst_row + (ulong) (ic + args.n_dims / 2) * args.nb0)) =
                (uint) fairy2i_f32_to_bf16(x0_sin + x1_cos) << 16;
        } else {
            const float x0 = *((device const float *) (src_row + (ulong) i0 * args.nb00));
            const float x1 = *((device const float *) (src_row + (ulong) (i0 + 1) * args.nb00));
            *((device uint *) (dst_row + (ulong) i0 * args.nb0)) =
                (uint) fairy2i_f32_to_bf16(x0) << 16;
            *((device uint *) (dst_row + (ulong) (i0 + 1) * args.nb0)) =
                (uint) fairy2i_f32_to_bf16(x1) << 16;
        }
    }
}

// TODO: obolete -- remove
//typedef void (im2col_t)(
//        constant ggml_metal_kargs_im2col & args,
//        device const float * x,
//        device        char * dst,
//        uint3 tgpig[[threadgroup_position_in_grid]],
//        uint3  tgpg[[threadgroups_per_grid]],
//        uint3 tpitg[[thread_position_in_threadgroup]],
//        uint3   ntg[[threads_per_threadgroup]]);
//
//template <typename T>
//kernel void kernel_im2col(
//        constant ggml_metal_kargs_im2col & args,
//        device const float * x,
//        device        char * dst,
//        uint3 tgpig[[threadgroup_position_in_grid]],
//        uint3  tgpg[[threadgroups_per_grid]],
//        uint3 tpitg[[thread_position_in_threadgroup]],
//        uint3   ntg[[threads_per_threadgroup]]) {
////    const int64_t IC = tgpg[0];
//    const int64_t OH = tgpg[1];
//    const int64_t OW = tgpg[2];
//
////    const int64_t N  = ntg[0];
//    const int64_t KH = ntg[1];
//    const int64_t KW = ntg[2];
//
//    const int64_t in  = tpitg[0];
//    const int64_t ikh = tpitg[1];
//    const int64_t ikw = tpitg[2];
//
//    const int64_t iic = tgpig[0];
//    const int64_t ioh = tgpig[1];
//    const int64_t iow = tgpig[2];
//
//    const int64_t iiw = iow*args.s0 + ikw*args.d0 - args.p0;
//    const int64_t iih = ioh*args.s1 + ikh*args.d1 - args.p1;
//
//    const int64_t offset_dst = (in*OH*OW + ioh*OW + iow)*args.CHW + (iic*(KH*KW) + ikh*KW + ikw);
//
//    device T * pdst = (device T *) (dst);
//
//    if (iih < 0 || iih >= args.IH || iiw < 0 || iiw >= args.IW) {
//        pdst[offset_dst] = 0.0f;
//    } else {
//        const int64_t offset_src = in*args.ofs0 + iic*args.ofs1 + iih*args.IW + iiw;
//        pdst[offset_dst] = x[offset_src];
//    }
//}
//
//template [[host_name("kernel_im2col_f32")]] kernel im2col_t kernel_im2col<float>;
//template [[host_name("kernel_im2col_f16")]] kernel im2col_t kernel_im2col<half>;

typedef void (im2col_ext_t)(
        constant ggml_metal_kargs_im2col & args,
        device const float * x,
        device        char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]);

template <typename T>
kernel void kernel_im2col_ext(
        constant ggml_metal_kargs_im2col & args,
        device const float * x,
        device        char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],      // tgpg[0] = D x IC x KH x KW, CHW = IC x KH x KW
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {  // [M, 1, 1]
    const int64_t KHW = (int64_t)args.KHW;

    const int64_t d   = tgpig[0] / args.CHW;
    const int64_t chw = tgpig[0] % args.CHW;
    const int64_t tgpig_0 = chw / KHW;  // 0 ~ (IC - 1)
    const int64_t HW = tgpig[0] % KHW;

    const int64_t tpitg_0 = (d * ntg[0]) + tpitg[0];
    if (tpitg_0 >= args.N) {
        return;
    }

    const int64_t tpitg_1 = HW / args.KW;
    const int64_t tpitg_2 = HW % args.KW;

    const int64_t iiw = tgpig[2] * args.s0 + tpitg_2 * args.d0 - args.p0;
    const int64_t iih = tgpig[1] * args.s1 + tpitg_1 * args.d1 - args.p1;

    const int64_t offset_dst =
        (tpitg_0 * tgpg[1] * tgpg[2] + tgpig[1] * tgpg[2] + tgpig[2]) * args.CHW +
        (tgpig_0 * KHW + tpitg_1 * args.KW + tpitg_2);

    device T * pdst = (device T *) (dst);

    if (iih < 0 || iih >= args.IH || iiw < 0 || iiw >= args.IW) {
        pdst[offset_dst] = 0.0f;
    } else {
        const int64_t offset_src = tpitg_0 * args.ofs0 + tgpig_0 * args.ofs1;
        pdst[offset_dst] = x[offset_src + iih * args.IW + iiw];
    }
}

template [[host_name("kernel_im2col_ext_f32")]] kernel im2col_ext_t kernel_im2col_ext<float>;
template [[host_name("kernel_im2col_ext_f16")]] kernel im2col_ext_t kernel_im2col_ext<half>;

typedef void (conv_transpose_1d_t)(
        constant ggml_metal_kargs_conv_transpose_1d & args,
        device const float * src0,
        device const float * src1,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3    tgpg[[threadgroups_per_grid]]);

template <typename T>
kernel void kernel_conv_transpose_1d(
        constant ggml_metal_kargs_conv_transpose_1d & args,
        device const     T * src0,
        device const float * src1,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tgpg[[threadgroups_per_grid]]) {

    float v = 0.0f;

    for (int64_t c = 0; c < args.IC; c++) {
        const int32_t kernel_offset = c * tgpg[1] * args.K + args.K * tgpig[1];
        const int32_t input_offset = c * args.IL;

        for (int64_t i = 0; i < args.IL; i++) {
            if (tgpig[0] >= i * args.s0 && tgpig[0] < i * args.s0 + args.K) {
                v += src0[kernel_offset + tgpig[0] - i * args.s0] * src1[input_offset + i];
            }
        }
    }

    device float * dst_ptr = (device float *) (dst + tgpig[0] * args.nb0 + tgpig[1] * args.nb1);

    dst_ptr[0] = v;
}

template [[host_name("kernel_conv_transpose_1d_f32_f32")]]
kernel void kernel_conv_transpose_1d<float>(
    constant ggml_metal_kargs_conv_transpose_1d & args,
    device const float * src0,
    device const float * src1,
    device        char * dst,
    uint3   tgpig[[threadgroup_position_in_grid]],
    uint3    tgpg[[threadgroups_per_grid]]);

template [[host_name("kernel_conv_transpose_1d_f16_f32")]]
kernel void kernel_conv_transpose_1d<half>(
    constant ggml_metal_kargs_conv_transpose_1d & args,
    device const half  * src0,
    device const float * src1,
    device        char * dst,
    uint3   tgpig[[threadgroup_position_in_grid]],
    uint3    tgpg[[threadgroups_per_grid]]);

kernel void kernel_upscale_f32(
    constant ggml_metal_kargs_upscale & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3/args.sf3;
    const int64_t i02 = i2/args.sf2;
    const int64_t i01 = i1/args.sf1;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int64_t i00 = i0/args.sf0;

        device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
        device       float * dst_ptr  = (device       float *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1  +  i0*args.nb0);

        dst_ptr[0] = src0_ptr[0];
    }
}

kernel void kernel_pad_f32(
    constant ggml_metal_kargs_pad & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);

    if (i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            if (i0 < args.ne00) {
                dst_ptr[i0] = src0_ptr[i0];
            } else {
                dst_ptr[i0] = 0.0f;
            }
        }

        return;
    }

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_ptr[i0] = 0.0f;
    }
}

kernel void kernel_pad_reflect_1d_f32(
    constant   ggml_metal_kargs_pad_reflect_1d & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3  tgpg[[threadgroups_per_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);

    if (i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            if (i0 < args.p0) {
                dst_ptr[i0] = src0_ptr[args.p0 - i0];
            } else if (i0 < args.ne0 - args.p1) {
                dst_ptr[i0] = src0_ptr[i0 - args.p0];
            } else {
                dst_ptr[i0] = src0_ptr[(args.ne0 - args.p1 - args.p0) - (args.p1 + 1 - (args.ne0 - i0)) - 1];
            }
        }
    }
}

kernel void kernel_arange_f32(
    constant   ggml_metal_kargs_arange & args,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    device float * dst_ptr = (device float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_ptr[i0] = args.start + args.step * i0;
    }
}

kernel void kernel_timestep_embedding_f32(
    constant  ggml_metal_kargs_timestep_embedding & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    int i = tgpig.x;
    device float * embed_data = (device float *)(dst + i*args.nb1);

    int half_ = args.dim / 2;
    for (int j = tpitg.x; j < half_; j += ntg.x) {
        float timestep = ((device float *)src0)[i];
        float freq = (float)exp(-log((float)args.max_period) * j / half_);
        float arg = timestep * freq;
        embed_data[j        ] = cos(arg);
        embed_data[j + half_] = sin(arg);
    }

    if (args.dim % 2 != 0 && tpitg.x == 0) {
        embed_data[2 * half_] = 0.f;
    }
}

// bitonic sort implementation following the CUDA kernels as reference
typedef void (argsort_t)(
        constant   ggml_metal_kargs_argsort & args,
        device  const float * x,
        device      int32_t * dst,
        threadgroup int32_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]);

template<ggml_sort_order order>
kernel void kernel_argsort_f32_i32(
        constant   ggml_metal_kargs_argsort & args,
        device const float  * x,
        device      int32_t * dst,
        threadgroup int32_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]) {
    // bitonic sort
    int col = tpitg[0];
    int row = tgpig[1];

    if (col >= args.ncols_pad) return;

    device const float   * x_row   = x + row * args.ncols;
    threadgroup int32_t  * dst_row = shared_values;

    // initialize indices
    dst_row[col] = col;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= args.ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= args.ncols ||
                        (dst_row[ixj] < args.ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= args.ncols ||
                        (dst_row[col] < args.ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // copy the result to dst without the padding
    if (col < args.ncols) {
        dst[row * args.ncols + col] = dst_row[col];
    }
}

template [[host_name("kernel_argsort_f32_i32_asc")]]  kernel argsort_t kernel_argsort_f32_i32<GGML_SORT_ORDER_ASC>;
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<GGML_SORT_ORDER_DESC>;

kernel void kernel_leaky_relu_f32(
        constant     ggml_metal_kargs_leaky_relu & args,
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    const float x = src0[tpig];
    dst[tpig] = x > 0.0f ? x : x * args.slope;
}

kernel void kernel_leaky_relu_f32_4(
        constant     ggml_metal_kargs_leaky_relu & args,
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    const float4 x = src0[tpig];
    dst[tpig] = float4(x > 0.0f)*x + float4(x <= 0.0f)*(x * args.slope);
}

constant bool FC_flash_attn_ext_has_mask  [[function_constant(FC_FLASH_ATTN_EXT + 0)]];
constant bool FC_flash_attn_ext_has_sinks [[function_constant(FC_FLASH_ATTN_EXT + 1)]];
constant bool FC_flash_attn_ext_has_bias  [[function_constant(FC_FLASH_ATTN_EXT + 2)]];
constant bool FC_flash_attn_ext_has_scap  [[function_constant(FC_FLASH_ATTN_EXT + 3)]];

//constant float FC_flash_attn_ext_scale         [[function_constant(FC_FLASH_ATTN_EXT + 10)]];
//constant float FC_flash_attn_ext_max_bias      [[function_constant(FC_FLASH_ATTN_EXT + 11)]];
//constant float FC_flash_attn_ext_logit_softcap [[function_constant(FC_FLASH_ATTN_EXT + 12)]];

constant int32_t FC_flash_attn_ext_ns10 [[function_constant(FC_FLASH_ATTN_EXT + 20)]];
constant int32_t FC_flash_attn_ext_ns20 [[function_constant(FC_FLASH_ATTN_EXT + 21)]];
constant int32_t FC_flash_attn_ext_nsg  [[function_constant(FC_FLASH_ATTN_EXT + 22)]];

#if defined(GGML_METAL_HAS_BF16)
template<short DK>
static inline void fairy2i_flash_attn_exact_qk_tile(
        threadgroup const bfloat * q,
        device const bfloat * k,
        int k_stride,
        int key_base,
        threadgroup float * scores,
        ushort lane) {
    uint q_metric = 255U;
    uint k_metric = 255U;
    for (int i = lane; i < 8 * DK; i += N_SIMDWIDTH) {
        q_metric = min(q_metric, fairy2i_bf16_product_metric(((threadgroup const ushort *) q)[i]));
        k_metric = min(
            k_metric,
            fairy2i_bf16_product_metric(
                ((device const ushort *) k)[key_base * k_stride + (i / DK) * k_stride + i % DK]));
    }
    q_metric = simd_min(q_metric);
    k_metric = simd_min(k_metric);

    if (fairy2i_product_metrics_require_software(q_metric, k_metric)) {
        for (int i = lane; i < 8 * 8; i += N_SIMDWIDTH) {
            const int query = i / 8;
            const int key = i % 8;
            uint acc_bits = 0;
            for (short d = 0; d < DK; ++d) {
                const ushort q_bits = ((threadgroup const ushort *) q)[query * DK + d];
                const ushort k_bits =
                    ((device const ushort *) k)[(key_base + key) * k_stride + d];
                acc_bits = fairy2i_fma_bf16_bf16_f32_bits_rne(q_bits, k_bits, acc_bits);
            }
            ((threadgroup uint *) scores)[i] = acc_bits;
        }
        return;
    }

    simdgroup_float8x8 qk = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (short d = 0; d < DK; d += 8) {
        simdgroup_bfloat8x8 mq;
        simdgroup_bfloat8x8 mk;

        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_load(mq, q + d, DK);
        simdgroup_load(mk, k + key_base * k_stride + d, k_stride, 0, true);
        simdgroup_barrier(mem_flags::mem_none);

        simdgroup_multiply_accumulate(qk, mq, mk, qk);
    }

    simdgroup_store(qk, scores, 8);
}

static inline float fairy2i_flash_attn_exact_logit(
        float qk,
        float scale,
        float mask) {
    const uint scaled_f32_bits =
        fairy2i_mul_f32_bits_rne_ftz_safe(as_type<uint>(qk), as_type<uint>(scale));
    const ushort scaled_bits = fairy2i_f32_to_bf16(as_type<float>(scaled_f32_bits));
    const uint logit_f32_bits =
        fairy2i_add_f32_bits_rne(((uint) scaled_bits) << 16, as_type<uint>(mask));
    return fairy2i_bf16_to_f32(fairy2i_f32_to_bf16(as_type<float>(logit_f32_bits)));
}

static inline ushort fairy2i_flash_attn_exact_probability_bits(
        float logit,
        float row_max,
        float row_sum,
        float row_log_sum) {
    if (row_sum == 0.0f) {
        return (ushort) 0;
    }

    const uint delta_bits =
        fairy2i_add_f32_bits_rne(as_type<uint>(logit), as_type<uint>(row_max) ^ 0x80000000U);
    const float delta = as_type<float>(delta_bits);
    uint probability_bits;
    if (delta - row_log_sum < -80.0f) {
        const float shifted = exp(delta + 64.0f) / row_sum;
        probability_bits =
            fairy2i_mul_f32_bits_rne_ftz_safe(as_type<uint>(shifted), as_type<uint>(exp(-64.0f)));
    } else {
        probability_bits = as_type<uint>(exp(delta) / row_sum);
    }
    return fairy2i_f32_to_bf16(as_type<float>(probability_bits));
}

template<short Q, short KV>
static inline bool fairy2i_flash_attn_exact_block_all_masked(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * mask,
        int iq1,
        int iq2,
        int iq3,
        int key_base,
        bool decode_gqa,
        int gqa_ratio,
        ushort lane) {
    bool query_block_masked = true;
    const bool query_valid =
        lane < Q && (decode_gqa ? lane < gqa_ratio : iq1 + lane < args.ne01);
    if (query_valid) {
        const int query_index = decode_gqa ? 0 : iq1 + lane;
        const int head_index = decode_gqa ? iq2 + lane : iq2;
        device const float * mask_row = (device const float *)
            (mask + (ulong) query_index * args.nb31 +
             (ulong) (head_index % args.ne32) * args.nb32 +
             (ulong) (iq3 % args.ne33) * args.nb33);
        for (short c = 0; c < KV; ++c) {
            query_block_masked =
                query_block_masked && mask_row[key_base + c] == -INFINITY;
        }
    }
    return simd_all(query_block_masked);
}

template<short DK, short DV, short Q = 8>
kernel void kernel_fairy2i_flash_attn_ext_exact_bf16(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device       char * dst,
        threadgroup uchar * smem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr short KV = 8;

    const int iq3 = (int) tgpig.z;
    const int gqa_ratio = args.ne02 / args.ne_12_2;
    const bool decode_gqa =
        args.ne01 == 1 && args.ne02 > args.ne_12_2 &&
        args.ne02 % args.ne_12_2 == 0 && gqa_ratio <= Q;
    const int iq2 = decode_gqa ? (int) tgpig.y * gqa_ratio : (int) tgpig.y;
    const int iq1 = decode_gqa ? 0 : (int) tgpig.x * Q;

    threadgroup bfloat * q_shared = (threadgroup bfloat *) smem;
    threadgroup float * scores =
        (threadgroup float *) (q_shared + Q * DK);
    threadgroup bfloat * probabilities =
        (threadgroup bfloat *) (scores + Q * KV);
    threadgroup float * stats =
        (threadgroup float *) (probabilities + Q * KV);
    threadgroup float * output =
        stats + 3 * Q;

    device const char * q_base =
        q + (ulong) iq1 * args.nb01 + (ulong) iq2 * args.nb02 + (ulong) iq3 * args.nb03;

    const int ikv2 = iq2 / (args.ne02 / args.ne_12_2);
    const int ikv3 = iq3 / (args.ne03 / args.ne_12_3);
    device const bfloat * k_base = (device const bfloat *)
        (k + (ulong) ikv2 * args.nb12 + (ulong) ikv3 * args.nb13);
    device const bfloat * v_base = (device const bfloat *)
        (v + (ulong) ikv2 * args.nb22 + (ulong) ikv3 * args.nb23);

    for (int i = tiisg; i < Q * DK; i += N_SIMDWIDTH) {
        const int query = i / DK;
        const int d = i % DK;
        float value = 0.0f;
        const bool query_valid =
            decode_gqa ? query < gqa_ratio : iq1 + query < args.ne01;
        if (query_valid) {
            const ulong query_offset =
                decode_gqa ? (ulong) query * args.nb02 : (ulong) query * args.nb01;
            value = *((device const float *) (q_base + query_offset) + d);
        }
        ((threadgroup ushort *) q_shared)[i] = fairy2i_f32_to_bf16(value);
    }

    for (int i = tiisg; i < Q * DV; i += N_SIMDWIDTH) {
        output[i] = 0.0f;
    }
    if (tiisg < Q) {
        stats[tiisg] = -INFINITY;
        stats[Q + tiisg] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 1: obtain the global F32 softmax maximum and sum. Scores are
    // reconstructed in row-major key order so probabilities can be normalized
    // before their mandatory BF16 conversion in pass 2.
    for (int key_base = 0; key_base < args.ne11; key_base += KV) {
        if (fairy2i_flash_attn_exact_block_all_masked<Q, KV>(
                args, mask, iq1, iq2, iq3, key_base, decode_gqa, gqa_ratio, tiisg)) {
            continue;
        }

        fairy2i_flash_attn_exact_qk_tile<DK>(
            q_shared, k_base, args.ns10, key_base, scores, tiisg);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const bool query_valid =
            tiisg < Q && (decode_gqa ? tiisg < gqa_ratio : iq1 + tiisg < args.ne01);
        if (query_valid) {
            const int query_index = decode_gqa ? 0 : iq1 + tiisg;
            const int head_index = decode_gqa ? iq2 + tiisg : iq2;
            device const float * mask_row = (device const float *)
                (mask + (ulong) query_index * args.nb31 +
                 (ulong) (head_index % args.ne32) * args.nb32 +
                 (ulong) (iq3 % args.ne33) * args.nb33);

            float row_max = stats[tiisg];
            float row_sum = stats[Q + tiisg];
            for (short c = 0; c < KV; ++c) {
                const float mask_value = mask_row[key_base + c];
                if (mask_value == -INFINITY) {
                    continue;
                }
                const float logit = fairy2i_flash_attn_exact_logit(
                    scores[tiisg * KV + c],
                    args.scale,
                    mask_value);
                if (logit > -INFINITY) {
                    if (row_sum == 0.0f) {
                        row_max = logit;
                        row_sum = 1.0f;
                    } else {
                        const float next_max = max(row_max, logit);
                        row_sum = row_sum * exp(row_max - next_max) + exp(logit - next_max);
                        row_max = next_max;
                    }
                }
            }
            stats[tiisg] = row_max;
            stats[Q + tiisg] = row_sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tiisg < Q) {
        stats[2 * Q + tiisg] = log(stats[Q + tiisg]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: reconstruct the same BF16 logits, normalize in F32, cast each
    // probability to BF16, then use native BF16 MMA with an F32 accumulator.
    // Once a block requires the bit-domain path, keep all following blocks there
    // so a preserved subnormal accumulator is never fed back through native MMA.
    bool pv_software_mode = false;
    for (int key_base = 0; key_base < args.ne11; key_base += KV) {
        if (fairy2i_flash_attn_exact_block_all_masked<Q, KV>(
                args, mask, iq1, iq2, iq3, key_base, decode_gqa, gqa_ratio, tiisg)) {
            continue;
        }

        fairy2i_flash_attn_exact_qk_tile<DK>(
            q_shared, k_base, args.ns10, key_base, scores, tiisg);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg < Q) {
            const bool query_valid =
                decode_gqa ? tiisg < gqa_ratio : iq1 + tiisg < args.ne01;
            if (query_valid) {
                const int query_index = decode_gqa ? 0 : iq1 + tiisg;
                const int head_index = decode_gqa ? iq2 + tiisg : iq2;
                device const float * mask_row = (device const float *)
                    (mask + (ulong) query_index * args.nb31 +
                     (ulong) (head_index % args.ne32) * args.nb32 +
                     (ulong) (iq3 % args.ne33) * args.nb33);
                const float row_max = stats[tiisg];
                const float row_sum = stats[Q + tiisg];
                const float row_log_sum = stats[2 * Q + tiisg];
                for (short c = 0; c < KV; ++c) {
                    const float mask_value = mask_row[key_base + c];
                    if (mask_value == -INFINITY) {
                        ((threadgroup ushort *) probabilities)[tiisg * KV + c] = (ushort) 0;
                        continue;
                    }
                    const float logit = fairy2i_flash_attn_exact_logit(
                        scores[tiisg * KV + c],
                        args.scale,
                        mask_value);
                    ((threadgroup ushort *) probabilities)[tiisg * KV + c] =
                        fairy2i_flash_attn_exact_probability_bits(
                            logit, row_max, row_sum, row_log_sum);
                }
            } else {
                for (short c = 0; c < KV; ++c) {
                    ((threadgroup ushort *) probabilities)[tiisg * KV + c] = (ushort) 0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint probability_metric = 255U;
        uint value_metric = 255U;
        for (int i = tiisg; i < Q * KV; i += N_SIMDWIDTH) {
            probability_metric = min(
                probability_metric,
                fairy2i_bf16_product_metric(((threadgroup const ushort *) probabilities)[i]));
        }
        for (int i = tiisg; i < KV * DV; i += N_SIMDWIDTH) {
            value_metric = min(
                value_metric,
                fairy2i_bf16_product_metric(
                    ((device const ushort *) v_base)[(key_base + i / DV) * args.ns20 + i % DV]));
        }
        probability_metric = simd_min(probability_metric);
        value_metric = simd_min(value_metric);
        pv_software_mode =
            pv_software_mode ||
            fairy2i_product_metrics_require_software(probability_metric, value_metric);

        if (pv_software_mode) {
            for (int i = tiisg; i < Q * DV; i += N_SIMDWIDTH) {
                const int query = i / DV;
                const int d = i % DV;
                uint acc_bits = as_type<uint>(output[i]);
                for (short c = 0; c < KV; ++c) {
                    const ushort probability_bits =
                        ((threadgroup const ushort *) probabilities)[query * KV + c];
                    const ushort value_bits =
                        ((device const ushort *) v_base)[(key_base + c) * args.ns20 + d];
                    acc_bits = fairy2i_fma_bf16_bf16_f32_bits_rne(
                        probability_bits, value_bits, acc_bits);
                }
                ((threadgroup uint *) output)[i] = acc_bits;
            }
        } else {
            simdgroup_bfloat8x8 mp;
            simdgroup_load(mp, probabilities, KV);

            for (short d = 0; d < DV; d += 8) {
                simdgroup_bfloat8x8 mv;
                simdgroup_float8x8 mo;

                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_load(mv, v_base + key_base * args.ns20 + d, args.ns20);
                simdgroup_load(mo, output + d, DV);
                simdgroup_barrier(mem_flags::mem_none);

                simdgroup_multiply_accumulate(mo, mp, mv, mo);
                simdgroup_store(mo, output + d, DV);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = tiisg; i < Q * DV; i += N_SIMDWIDTH) {
        const int query = i / DV;
        const int d = i % DV;
        const bool query_valid =
            decode_gqa ? query < gqa_ratio : iq1 + query < args.ne01;
        if (query_valid) {
            const int query_index = decode_gqa ? 0 : iq1 + query;
            const int head_index = decode_gqa ? iq2 + query : iq2;
            device float * dst_row = (device float *) dst +
                ((ulong) iq3 * args.ne2 * args.ne1 +
                 (ulong) query_index * args.ne1 + (ulong) head_index) * DV;
            ((device uint *) dst_row)[d] = ((uint) fairy2i_f32_to_bf16(output[i])) << 16;
        }
    }

    (void) sinks;
}

kernel void kernel_fairy2i_flash_attn_ext_exact_bf16_decode_vec_logits_dk128_dv128(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * mask,
        device       char * tmp,
        threadgroup uchar * smem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr short DK  = 128;
    constexpr short Q   = 8;
    constexpr short KV  = 8;
    constexpr short NWG = 32;

    const int iwg        = (int) tgpig.x;
    const int kv_head    = (int) tgpig.y;
    const int batch      = (int) tgpig.z;
    const int gqa_ratio  = args.ne02 / args.ne_12_2;
    const int q_head_base = kv_head * gqa_ratio;

    const int n_blocks        = args.ne11 / KV;
    const int blocks_per_wg   = (n_blocks + NWG - 1) / NWG;
    const int block_begin     = iwg * blocks_per_wg;
    const int block_end       = min(block_begin + blocks_per_wg, n_blocks);

    threadgroup bfloat * q_shared = (threadgroup bfloat *) smem;
    threadgroup float * scores = (threadgroup float *) (q_shared + Q * DK);

    device const char * q_base =
        q + (ulong) q_head_base * args.nb02 + (ulong) batch * args.nb03;
    const int ikv3 = batch / (args.ne03 / args.ne_12_3);
    device const bfloat * k_base = (device const bfloat *)
        (k + (ulong) kv_head * args.nb12 + (ulong) ikv3 * args.nb13);

    device ushort * logits = (device ushort *) tmp;

    for (int i = tiisg; i < Q * DK; i += N_SIMDWIDTH) {
        const int query = i / DK;
        const int d = i % DK;
        float value = 0.0f;
        if (query < gqa_ratio && q_head_base + query < args.ne02) {
            value = *((device const float *)
                (q_base + (ulong) query * args.nb02) + d);
        }
        ((threadgroup ushort *) q_shared)[i] = fairy2i_f32_to_bf16(value);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int block = block_begin; block < block_end; ++block) {
        const int key_base = block * KV;
        fairy2i_flash_attn_exact_qk_tile<DK>(
            q_shared, k_base, args.ns10, key_base, scores, tiisg);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = tiisg; i < Q * KV; i += N_SIMDWIDTH) {
            const int query = i / KV;
            const int key_offset = i % KV;
            if (query >= gqa_ratio || q_head_base + query >= args.ne02) {
                continue;
            }

            const int q_head = q_head_base + query;
            const int row = batch * args.ne02 + q_head;
            device const float * mask_row = (device const float *)
                (mask + (ulong) (q_head % args.ne32) * args.nb32 +
                 (ulong) (batch % args.ne33) * args.nb33);
            const float mask_value = mask_row[key_base + key_offset];
            const ushort logit_bits =
                mask_value == -INFINITY ?
                    (ushort) 0xff80U :
                    fairy2i_f32_to_bf16(fairy2i_flash_attn_exact_logit(
                        scores[query * KV + key_offset], args.scale, mask_value));
            logits[(ulong) row * args.ne11 + key_base + key_offset] = logit_bits;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kernel_fairy2i_flash_attn_ext_exact_bf16_decode_vec_softmax_dk128_dv128(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device       char * tmp,
        threadgroup float * stats [[threadgroup(0)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = 8;

    const int row = (int) tgpig;
    const int nrows = args.ne01 * args.ne02 * args.ne03;
    if (row >= nrows) {
        return;
    }

    device ushort * logits = (device ushort *) tmp;
    device ushort * probabilities = logits + (ulong) nrows * args.ne11;
    device const ushort * row_logits = logits + (ulong) row * args.ne11;
    device ushort * row_probabilities = probabilities + (ulong) row * args.ne11;

    float lane_max = -INFINITY;
    for (int key = tid; key < args.ne11; key += NSG * N_SIMDWIDTH) {
        lane_max = max(lane_max, fairy2i_bf16_to_f32(row_logits[key]));
    }
    lane_max = simd_max(lane_max);
    if (tiisg == 0) {
        stats[sgitg] = lane_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        const float group_max = tiisg < NSG ? stats[tiisg] : -INFINITY;
        const float row_max = simd_max(group_max);
        if (tiisg == 0) {
            stats[0] = row_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float lane_sum = 0.0f;
    for (int key = tid; key < args.ne11; key += NSG * N_SIMDWIDTH) {
        const float logit = fairy2i_bf16_to_f32(row_logits[key]);
        if (logit != -INFINITY) {
            lane_sum += exp(logit - stats[0]);
        }
    }
    lane_sum = simd_sum(lane_sum);
    if (tiisg == 0) {
        stats[NSG + sgitg] = lane_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        const float group_sum = tiisg < NSG ? stats[NSG + tiisg] : 0.0f;
        const float row_sum = simd_sum(group_sum);
        if (tiisg == 0) {
            stats[1] = row_sum;
            stats[2] = log(row_sum);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int key = tid; key < args.ne11; key += NSG * N_SIMDWIDTH) {
        const float logit = fairy2i_bf16_to_f32(row_logits[key]);
        row_probabilities[key] =
            logit == -INFINITY ?
                (ushort) 0 :
                fairy2i_flash_attn_exact_probability_bits(
                    logit, stats[0], stats[1], stats[2]);
    }
}

kernel void kernel_fairy2i_flash_attn_ext_exact_bf16_decode_vec_output_partial_dk128_dv128(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * v,
        device       char * tmp,
        threadgroup uchar * smem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr short DV = 128;
    constexpr short Q = 8;
    constexpr short KV = 8;
    constexpr short NWG = 32;

    const int iwg = (int) tgpig.x;
    const int kv_head = (int) tgpig.y;
    const int batch = (int) tgpig.z;
    const int gqa_ratio = args.ne02 / args.ne_12_2;
    const int q_head_base = kv_head * gqa_ratio;
    const int nrows = args.ne01 * args.ne02 * args.ne03;
    if (iwg >= NWG) {
        return;
    }

    const int ikv3 = batch / (args.ne03 / args.ne_12_3);
    device const bfloat * v_base = (device const bfloat *)
        (v + (ulong) kv_head * args.nb22 + (ulong) ikv3 * args.nb23);
    device const ushort * logits = (device const ushort *) tmp;
    device const ushort * probabilities =
        logits + (ulong) nrows * args.ne11;
    device uint * partials = (device uint *)
        (logits + (ulong) 2 * nrows * args.ne11);

    threadgroup bfloat * probabilities_shared = (threadgroup bfloat *) smem;
    threadgroup float * output = (threadgroup float *) (probabilities_shared + Q * KV);

    for (int i = tiisg; i < Q * DV; i += N_SIMDWIDTH) {
        output[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int n_blocks = args.ne11 / KV;
    const int blocks_per_wg = (n_blocks + NWG - 1) / NWG;
    const int block_begin = iwg * blocks_per_wg;
    const int block_end = min(block_begin + blocks_per_wg, n_blocks);
    bool pv_software_mode = false;

    for (int block = block_begin; block < block_end; ++block) {
        const int key_base = block * KV;
        for (int i = tiisg; i < Q * KV; i += N_SIMDWIDTH) {
            const int query = i / KV;
            const int key_offset = i % KV;
            ushort probability_bits = 0;
            if (query < gqa_ratio && q_head_base + query < args.ne02) {
                const int row = batch * args.ne02 + q_head_base + query;
                probability_bits =
                    probabilities[(ulong) row * args.ne11 + key_base + key_offset];
            }
            ((threadgroup ushort *) probabilities_shared)[i] = probability_bits;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint probability_metric = 255U;
        uint value_metric = 255U;
        for (int i = tiisg; i < Q * KV; i += N_SIMDWIDTH) {
            probability_metric = min(
                probability_metric,
                fairy2i_bf16_product_metric(
                    ((threadgroup const ushort *) probabilities_shared)[i]));
        }
        for (int i = tiisg; i < KV * DV; i += N_SIMDWIDTH) {
            value_metric = min(
                value_metric,
                fairy2i_bf16_product_metric(
                    ((device const ushort *) v_base)[
                        (ulong) (key_base + i / DV) * args.ns20 + i % DV]));
        }
        probability_metric = simd_min(probability_metric);
        value_metric = simd_min(value_metric);
        pv_software_mode =
            pv_software_mode ||
            fairy2i_product_metrics_require_software(probability_metric, value_metric);

        if (pv_software_mode) {
            for (int i = tiisg; i < Q * DV; i += N_SIMDWIDTH) {
                const int query = i / DV;
                const int d = i % DV;
                uint acc_bits = as_type<uint>(output[i]);
                for (short c = 0; c < KV; ++c) {
                    const ushort probability_bits =
                        ((threadgroup const ushort *) probabilities_shared)[query * KV + c];
                    const ushort value_bits =
                        ((device const ushort *) v_base)[
                            (ulong) (key_base + c) * args.ns20 + d];
                    acc_bits = fairy2i_fma_bf16_bf16_f32_bits_rne(
                        probability_bits, value_bits, acc_bits);
                }
                ((threadgroup uint *) output)[i] = acc_bits;
            }
        } else {
            simdgroup_bfloat8x8 mp;
            simdgroup_load(mp, probabilities_shared, KV);

            for (short d = 0; d < DV; d += 8) {
                simdgroup_bfloat8x8 mv;
                simdgroup_float8x8 mo;

                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_load(mv, v_base + key_base * args.ns20 + d, args.ns20);
                simdgroup_load(mo, output + d, DV);
                simdgroup_barrier(mem_flags::mem_none);

                simdgroup_multiply_accumulate(mo, mp, mv, mo);
                simdgroup_store(mo, output + d, DV);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = tiisg; i < gqa_ratio * DV; i += N_SIMDWIDTH) {
        const int query = i / DV;
        const int d = i % DV;
        const int row = batch * args.ne02 + q_head_base + query;
        partials[((ulong) row * NWG + iwg) * DV + d] =
            ((threadgroup const uint *) output)[i];
    }
}

kernel void kernel_fairy2i_flash_attn_ext_exact_bf16_decode_vec_output_reduce_dk128_dv128(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * tmp,
        device       char * dst,
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr short DV = 128;
    constexpr short NWG = 32;

    const int row = (int) tgpig;
    const int nrows = args.ne01 * args.ne02 * args.ne03;
    if (row >= nrows || tid >= DV) {
        return;
    }

    device const ushort * logits = (device const ushort *) tmp;
    device const uint * partials = (device const uint *)
        (logits + (ulong) 2 * nrows * args.ne11);
    uint acc_bits = 0;
    for (short iwg = 0; iwg < NWG; ++iwg) {
        acc_bits = fairy2i_add_f32_bits_rne(
            acc_bits,
            partials[((ulong) row * NWG + iwg) * DV + tid]);
    }

    ((device uint *) dst)[(ulong) row * DV + tid] =
        ((uint) fairy2i_f32_to_bf16(as_type<float>(acc_bits))) << 16;
}

typedef decltype(kernel_fairy2i_flash_attn_ext_exact_bf16<64, 64>) fairy2i_flash_attn_ext_exact_t;

template [[host_name("kernel_fairy2i_flash_attn_ext_exact_bf16_dk40_dv40")]]
kernel fairy2i_flash_attn_ext_exact_t kernel_fairy2i_flash_attn_ext_exact_bf16<40, 40>;
template [[host_name("kernel_fairy2i_flash_attn_ext_exact_bf16_dk64_dv64")]]
kernel fairy2i_flash_attn_ext_exact_t kernel_fairy2i_flash_attn_ext_exact_bf16<64, 64>;
template [[host_name("kernel_fairy2i_flash_attn_ext_exact_bf16_dk80_dv80")]]
kernel fairy2i_flash_attn_ext_exact_t kernel_fairy2i_flash_attn_ext_exact_bf16<80, 80>;
template [[host_name("kernel_fairy2i_flash_attn_ext_exact_bf16_dk96_dv96")]]
kernel fairy2i_flash_attn_ext_exact_t kernel_fairy2i_flash_attn_ext_exact_bf16<96, 96>;
template [[host_name("kernel_fairy2i_flash_attn_ext_exact_bf16_dk112_dv112")]]
kernel fairy2i_flash_attn_ext_exact_t kernel_fairy2i_flash_attn_ext_exact_bf16<112, 112>;
template [[host_name("kernel_fairy2i_flash_attn_ext_exact_bf16_dk128_dv128")]]
kernel fairy2i_flash_attn_ext_exact_t kernel_fairy2i_flash_attn_ext_exact_bf16<128, 128>;
template [[host_name("kernel_fairy2i_flash_attn_ext_exact_bf16_dk192_dv192")]]
kernel fairy2i_flash_attn_ext_exact_t kernel_fairy2i_flash_attn_ext_exact_bf16<192, 192>;
template [[host_name("kernel_fairy2i_flash_attn_ext_exact_bf16_dk192_dv128")]]
kernel fairy2i_flash_attn_ext_exact_t kernel_fairy2i_flash_attn_ext_exact_bf16<192, 128>;
template [[host_name("kernel_fairy2i_flash_attn_ext_exact_bf16_dk256_dv256")]]
kernel fairy2i_flash_attn_ext_exact_t kernel_fairy2i_flash_attn_ext_exact_bf16<256, 256>;
#endif

// ref: https://arxiv.org/pdf/2307.08691.pdf
template<
    typename q_t,     // query types in shared memory
    typename q4_t,
    typename q8x8_t,
    typename k_t,     // key types in shared memory
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,     // value types in shared memory
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,    // Q*K types
    typename qk8x8_t,
    typename s_t,     // soft-max types
    typename s2_t,
    typename s8x8_t,
    typename o_t,     // attention accumulation types
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t, // key type in device memory
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t, // value type in device memory
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,         // K head size
    short DV,         // V head size
    short Q,          // queries per threadgroup
    short C,          // cache items per threadgroup
    short NSG>        // number of simd groups
void kernel_flash_attn_ext_impl(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device       char * dst,
        threadgroup  half * shmem_f16,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    const ushort iq3 = tgpig[2];
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0]*Q;

#define NS10 (FC_flash_attn_ext_ns10)
#define NS20 (FC_flash_attn_ext_ns20)

    // note: I had some concerns that using this instead of the ugly macros above was affecting performance
    //       need to re-check carefully and if no regressions are observerd - remove the macros
    //       the concerns is that maybe using const variables requires extra registers? but not sure if the compiler
    //         is clever enough to avoid this. unfortunately, using constexpr is not possible with FC
    //const short NS10 = FC_flash_attn_ext_ns10;
    //const short NS20 = FC_flash_attn_ext_ns20;

    constexpr short KV   = 8;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;
    constexpr short DV4  = DV/4;
  //constexpr short DV8  = DV/8;
    constexpr short DV16 = DV/16;

    constexpr short PV   = PAD2(DV, 64);
    constexpr short PV4  = PV/4;
    constexpr short PV8  = PV/8;
  //constexpr short PV16 = PV/16;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NQ  = Q/NSG;
    constexpr short SH  = 2*C; // shared memory per simdgroup (s_t == float)

    constexpr short TS = 2*SH;
    constexpr short T  = DK + 2*PV; // shared memory size per query in (half)

    threadgroup q_t  * sq  = (threadgroup q_t  *) (shmem_f16 + 0*T); // holds the query data
    threadgroup q4_t * sq4 = (threadgroup q4_t *) (shmem_f16 + 0*T); // same as above but in q4_t
    threadgroup o_t  * so  = (threadgroup o_t  *) (shmem_f16 + 0*T + Q*DK); // the result for all queries in 8x8 matrices (the O matrix from the paper)
    threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 + 0*T + Q*DK);
    threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + Q*T); // scratch buffer for attention, mask and diagonal matrix
    threadgroup s2_t * ss2 = (threadgroup s2_t *) (shmem_f16 + Q*T); // same as above but in s2_t

    threadgroup k_t    * sk    = (threadgroup k_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // scratch buffer to load K in shared memory
    threadgroup k4x4_t * sk4x4 = (threadgroup k4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // same as above but in k4x4_t

    threadgroup v_t    * sv    = (threadgroup v_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // scratch buffer to load V in shared memory
    threadgroup v4x4_t * sv4x4 = (threadgroup v4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // same as above but in v4x4_t

    // mask storage in shared mem
    threadgroup half2 * sm2 = (threadgroup half2 *) (shmem_f16 + Q*T + 2*C);

    // per-query mask pointers
    device const half2 * pm2[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        pm2[jj] = (device const half2 *) ((device const char *) mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    // load heads from Q to shared memory
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        device const float4 * q4 = (device const float4 *) ((device const char *) q + j*args.nb01);

        for (short i = tiisg; i < DK4; i += NW) {
            if (iq1 + j < args.ne01) {
                sq4[j*DK4 + i] = (q4_t) q4[i];
            } else {
                sq4[j*DK4 + i] = 0;
            }
        }
    }

    // zero out
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        for (short i = tiisg; i < DV4; i += NW) {
            so4[j*PV4 + i] = 0;
        }

        for (short i = tiisg; i < SH; i += NW) {
            ss[j*SH + i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ] = { [0 ... NQ-1] = 0.0f };

    {
        float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };

        float slope = 1.0f;

        // ALiBi
        if (FC_flash_attn_ext_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic = 0; ic < args.ne11; ic += C) {
            // read the mask into shared mem
            if (FC_flash_attn_ext_has_mask) {
                FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                    const short j = jj*NSG + sgitg;

                    sm2[j*SH + tiisg] = pm2[jj][tiisg];
                    pm2[jj] += NW;
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                // used to detect blocks full of -INF
                // skip only when the entire threadgroup is masked
                half2 smax2(-MAXHALF/2, -MAXHALF/2);

                FOR_UNROLL (short j = 0; j < Q; ++j) {
                    smax2 = max(smax2, sm2[j*SH + tiisg]);
                }

                smax2 = simd_max(smax2);

                if (max(smax2[0], smax2[1]) <= -MAXHALF/2) {
                    // this barrier is important
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    continue;
                }
            }

            // Q*K^T
            // this is compile-time check, so it does not have runtime overhead
            if (is_same<kd4x4_t, k4x4_t>::value) {
                // we can read directly from global memory
                device      const k_t * pk = (device const k_t *) ((device const char *) k + ic*args.nb11);
                threadgroup const q_t * pq = sq;
                threadgroup       s_t * ps = ss;

                pk += sgitg*(8*NS10);
                ps += sgitg*(8*1);

                static_assert((C/8) % NSG == 0, "");

                constexpr short NC = (C/8)/NSG;

                // TODO: not good to unroll for large contexts - not sure why?
                for (short cc = 0; cc < NC; ++cc) {
                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    if (DK8 % 16 != 0) {
                        k8x8_t mk;
                        q8x8_t mq;

                        FOR_UNROLL (short i = 0; i < DK8; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mk, pk, NS10, 0, true);
                            simdgroup_load(mq, pq, DK);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                            pk += 8;
                            pq += 8;
                        }
                    } else {
                        k8x8_t mk[2];
                        q8x8_t mq[2];

                        FOR_UNROLL (short i = 0; i < DK8/2; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mk[0], pk + 0*8, NS10, 0, true);
                            simdgroup_load(mk[1], pk + 1*8, NS10, 0, true);

                            simdgroup_load(mq[0], pq + 0*8, DK);
                            simdgroup_load(mq[1], pq + 1*8, DK);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                            simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);

                            pk += 16;
                            pq += 16;
                        }
                    }

                    simdgroup_store(mqk, ps, SH, 0, false);

                    pk += 8*(NSG*NS10 - DK8);
                    pq += 8*(NSG*0    - DK8);
                    ps += 8*(NSG);
                }
            } else {
                // TODO: this is the quantized K cache branch - not optimized yet
                for (short ccc = 0; ccc < (C/8)/NSG; ++ccc) {
                    const short cc = ccc*NSG + sgitg;

                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    for (short ii = 0; ii < DK16; ii += 4) {
                        device const kd4x4_t * pk4x4 = (device const kd4x4_t *) ((device const char *) k + ((ic + 8*cc + ty)*args.nb11));

                        if (DK16%4 == 0) {
                            // the head is evenly divisible by 4*16 = 64, so no need for bound checks
                            {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            FOR_UNROLL (short k = 0; k < 4; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true); // transpose
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true); // transpose
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        } else {
                            if (ii + tx < DK16) {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            for (short k = 0; k < 4 && ii + k < DK16; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true); // transpose
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true); // transpose
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        }
                    }

                    simdgroup_store(mqk, ss + 8*cc, SH, 0, false);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];

                // scale and apply the logitcap / mask
                float2 s2 = ss2[j*SH/2 + tiisg]*args.scale;

                if (FC_flash_attn_ext_has_scap) {
                    s2 = args.logit_softcap*precise::tanh(s2);
                }

                // mqk = mqk + slope*mask
                if (FC_flash_attn_ext_has_bias) {
                    s2 += s2_t(sm2[j*SH + tiisg])*slope;
                } else {
                    s2 += s2_t(sm2[j*SH + tiisg]);
                }

                M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));

                const float  ms  = exp(m  - M[jj]);
                const float2 vs2 = exp(s2 - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);

                // the P matrix from the paper (Q rows, C columns)
                ss2[j*SH/2 + tiisg] = vs2;

                if (DV4 % NW == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                        const short i = ii*NW + tiisg;

                        so4[j*PV4 + i] *= ms;
                    }
                } else {
                    for (short i = tiisg; i < DV4; i += NW) {
                        so4[j*PV4 + i] *= ms;
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // O = O + (Q*K^T)*V
            {
                // we can read directly from global memory
                if (is_same<vd4x4_t, v4x4_t>::value) {
                    static_assert(PV8 % NSG == 0, "");

                    constexpr short NO = PV8/NSG;

                    o8x8_t lo[NO];

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_load(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }

                    {
                        auto sst = ss;

                        device const v_t * pv = (device const v_t *) ((device const char *) v + ic*args.nb21);

                        pv += 8*sgitg;

                        FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {
                            s8x8_t vs;
                            simdgroup_load(vs, sst, SH, 0, false);

                            FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                                v8x8_t mv;

                                simdgroup_load(mv, pv, NS20, 0, false);
                                simdgroup_multiply_accumulate(lo[ii], vs, mv, lo[ii]);

                                pv += 8*NSG;
                            }

                            pv  += 8*(NS20 - NO*NSG);
                            sst += 8;
                        }
                    }

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_store(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }
                } else {
                    // TODO: this is the quantized V cache branch - not optimized yet

                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    for (short cc = 0; cc < C/8; ++cc) {
                        s8x8_t vs;
                        simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                        for (short ii = 4*sgitg; ii < DV16; ii += 4*NSG) {
                            device const vd4x4_t * pv4x4 = (device const vd4x4_t *) ((device const char *) v + ((ic + 8*cc + ty)*args.nb21));

                            if (DV16%4 == 0) {
                                // no need for bound checks
                                {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                FOR_UNROLL (short k = 0; k < 4; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            } else {
                                if (ii + tx < DV16) {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                for (short k = 0; k < 4 && ii + k < DV16; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            }
                        }
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (FC_flash_attn_ext_has_sinks) {
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];
                const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

                M[jj] = simd_max(max(M[jj], s));

                const float ms = exp(m - M[jj]);
                const float vs = exp(s - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs);

                for (short i = tiisg; i < DV4; i += NW) {
                    so4[j*PV4 + i] *= ms;
                }
            }
        }
    }

    // store to global memory
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        if (iq1 + j >= args.ne01) {
            break;
        }

        device float4 * dst4 = (device float4 *) dst + ((uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)(iq1 + j)*args.ne1)*DV4;

        const float scale = 1.0f/S[jj];

        if (DV4 % NW == 0) {
            FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                const short i = ii*NW + tiisg;

                float4 result = (float4) so4[j*PV4 + i]*scale;
#if defined(GGML_METAL_HAS_BF16)
                if (is_same<q_t, bfloat>::value && is_same<o_t, float>::value) {
                    result = fairy2i_round_to_bf16_f32(result);
                }
#endif
                dst4[i] = result;
            }
        } else {
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        }
    }

#undef NS10
#undef NS20
}

template<
    typename q_t,     // query types in shared memory
    typename q4_t,
    typename q8x8_t,
    typename k_t,     // key types in shared memory
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,     // value types in shared memory
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,    // Q*K types
    typename qk8x8_t,
    typename s_t,     // soft-max types
    typename s2_t,
    typename s8x8_t,
    typename o_t,     // attention accumulation types
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t, // key type in device memory
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t, // value type in device memory
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,         // K head size
    short DV,         // V head size
    short Q  = 8,     // queries per threadgroup
    short C  = 64>    // cache items per threadgroup
kernel void kernel_flash_attn_ext(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
#define FWD_TMPL q_t, q4_t, q8x8_t, k_t, k4x4_t, k8x8_t, v_t, v4x4_t, v8x8_t, qk_t, qk8x8_t, s_t, s2_t, s8x8_t, o_t, o4_t, o8x8_t, kd4x4_t, nl_k, deq_k, vd4x4_t, nl_v, deq_v, DK, DV, Q, C
#define FWD_ARGS args, q, k, v, mask, sinks, dst, shmem_f16, tgpig, tiisg, sgitg
    switch (FC_flash_attn_ext_nsg) {
      // note: disabled cases to reduce library load time
      //case 1: kernel_flash_attn_ext_impl<FWD_TMPL, 1>(FWD_ARGS); break;
      //case 2: kernel_flash_attn_ext_impl<FWD_TMPL, 2>(FWD_ARGS); break;
        case 4: kernel_flash_attn_ext_impl<FWD_TMPL, 4>(FWD_ARGS); break;
    }
#undef FWD_TMPL
#undef FWD_ARGS
}

// TODO: this is quite ugly. in the future these types will be hardcoded in the kernel, but for now keep them as
//       template to be able to explore different combinations
//
#define FA_TYPES \
    half,   half4,     simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8
    //half,   half4,     simdgroup_half8x8

#define FA_TYPES_BF \
    bfloat, bfloat4,   simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    float,             simdgroup_float8x8,  \
    float,  float2,    simdgroup_float8x8,  \
    half,   half4,     simdgroup_half8x8
    //float,  float4,    simdgroup_float8x8

#define FA_TYPES_FAIRY_BF \
    bfloat, bfloat4,   simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    float,             simdgroup_float8x8,  \
    float,  float2,    simdgroup_float8x8,  \
    float,  float4,    simdgroup_float8x8

typedef decltype(kernel_flash_attn_ext<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 64, 64>) flash_attn_ext_t;

template [[host_name("kernel_flash_attn_ext_f16_dk40_dv40"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  40,  40>;
template [[host_name("kernel_flash_attn_ext_f16_dk64_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  64,  64>;
template [[host_name("kernel_flash_attn_ext_f16_dk80_dv80"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  80,  80>;
template [[host_name("kernel_flash_attn_ext_f16_dk96_dv96"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  96,  96>;
template [[host_name("kernel_flash_attn_ext_f16_dk112_dv112")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  112, 112>;
template [[host_name("kernel_flash_attn_ext_f16_dk128_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  128, 128>;
template [[host_name("kernel_flash_attn_ext_f16_dk192_dv192")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  192, 192>;
template [[host_name("kernel_flash_attn_ext_f16_dk192_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  192, 128>;
template [[host_name("kernel_flash_attn_ext_f16_dk256_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  256, 256>;
template [[host_name("kernel_flash_attn_ext_f16_dk576_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  576, 512>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_fairy_bf16_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_FAIRY_BF, bfloat4x4, 1, dequantize_bf16, bfloat4x4, 1, dequantize_bf16, 128, 128>;
template [[host_name("kernel_flash_attn_ext_bf16_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 40,  40>;
template [[host_name("kernel_flash_attn_ext_bf16_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 64,  64>;
template [[host_name("kernel_flash_attn_ext_bf16_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 80,  80>;
template [[host_name("kernel_flash_attn_ext_bf16_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 96,  96>;
template [[host_name("kernel_flash_attn_ext_bf16_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 112, 112>;
template [[host_name("kernel_flash_attn_ext_bf16_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 128, 128>;
template [[host_name("kernel_flash_attn_ext_bf16_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 192, 192>;
template [[host_name("kernel_flash_attn_ext_bf16_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 192, 128>;
template [[host_name("kernel_flash_attn_ext_bf16_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 256, 256>;
template [[host_name("kernel_flash_attn_ext_bf16_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 576, 512>;
#endif

template [[host_name("kernel_flash_attn_ext_q4_0_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 576, 512>;

template [[host_name("kernel_flash_attn_ext_q4_1_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 576, 512>;

template [[host_name("kernel_flash_attn_ext_q5_0_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 576, 512>;

template [[host_name("kernel_flash_attn_ext_q5_1_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 576, 512>;

template [[host_name("kernel_flash_attn_ext_q8_0_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 576, 512>;

#undef FA_TYPES
#undef FA_TYPES_BF
#undef FA_TYPES_FAIRY_BF

constant bool FC_flash_attn_ext_vec_has_mask  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 0)]];
constant bool FC_flash_attn_ext_vec_has_sinks [[function_constant(FC_FLASH_ATTN_EXT_VEC + 1)]];
constant bool FC_flash_attn_ext_vec_has_bias  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 2)]];
constant bool FC_flash_attn_ext_vec_has_scap  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 3)]];

//constant float FC_flash_attn_ext_vec_scale         [[function_constant(FC_FLASH_ATTN_EXT_VEC + 10)]];
//constant float FC_flash_attn_ext_vec_max_bias      [[function_constant(FC_FLASH_ATTN_EXT_VEC + 11)]];
//constant float FC_flash_attn_ext_vec_logit_softcap [[function_constant(FC_FLASH_ATTN_EXT_VEC + 12)]];

constant int32_t FC_flash_attn_ext_vec_ns10 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 20)]];
constant int32_t FC_flash_attn_ext_vec_ns20 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 21)]];
constant int32_t FC_flash_attn_ext_vec_nsg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 22)]];
constant int32_t FC_flash_attn_ext_vec_nwg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 23)]];

template<
    typename q4_t,  // query types in shared memory
    typename k4_t,  // key types in shared memory
    typename v4_t,  // value types in shared memory
    typename qk_t,  // Q*K types
    typename s_t,   // soft-max types
    typename s4_t,
    typename o4_t,  // attention accumulation types
    typename kd4_t, // key type in device memory
    short nl_k,
    void (*deq_k_t4)(device const kd4_t *, short, thread k4_t &),
    typename vd4_t, // value type in device memory
    short nl_v,
    void (*deq_v_t4)(device const vd4_t *, short, thread v4_t &),
    short DK,       // K head size
    short DV,       // V head size
    short NE = 4,   // head elements per thread
    short Q  = 1,   // queries per threadgroup
    short C  = 32,  // cache items per threadgroup
    short NSG>      // number of simd groups
void kernel_flash_attn_ext_vec_impl(
        constant ggml_metal_kargs_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");

#define NWG  (FC_flash_attn_ext_vec_nwg)

#define NS10 (FC_flash_attn_ext_vec_ns10)
#define NS20 (FC_flash_attn_ext_vec_ns20)

    const short iwg = tgpig[2]%NWG;

    const ushort iq3 = tgpig[2]/NWG;
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0];

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;

    constexpr short PK  = PAD2(DK, 128);
    constexpr short PK4 = PK/4;

    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV/4;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW/NE; // note: this can be adjusted to support different head sizes and simdgroup work loads
    constexpr short SH  = 4*C;   // shared memory per simdgroup

    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

    const short T = PK + NSG*SH; // shared memory size per query in (half)

  //threadgroup q_t   * sq  = (threadgroup q_t   *) (shmem_f16 +                    0*PK); // holds the query data
    threadgroup q4_t  * sq4 = (threadgroup q4_t  *) (shmem_f16 +                    0*PK); // same as above but in q4_t
    threadgroup s_t   * ss  = (threadgroup s_t   *) (shmem_f16 +   sgitg*SH       + Q*PK); // scratch buffer for attention
    threadgroup s4_t  * ss4 = (threadgroup s4_t  *) (shmem_f16 +   sgitg*SH       + Q*PK); // same as above but in s4_t
    threadgroup half  * sm  = (threadgroup half  *) (shmem_f16 +   sgitg*SH + 2*C + Q*PK); // scratch buffer for mask
    threadgroup o4_t  * so4 = (threadgroup o4_t  *) (shmem_f16 + 2*sgitg*PV       + Q*T);  // scratch buffer for the results

    // store the result for all queries in shared memory (the O matrix from the paper)
    so4 += tiisg;

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    // load heads from Q to shared memory
    device const float4 * q4 = (device const float4 *) ((device const char *) q);

    for (short i = tiisg; i < PK4; i += NW) {
        if (iq1 < args.ne01 && i < DK4) {
            sq4[i] = (q4_t) q4[i];
        } else {
            sq4[i] = (q4_t) float4(0.0f);
        }
    }

    // zero out so
    for (short i = 0; i < DV4/NL; ++i) {
        so4[i*NL] = (o4_t) 0.0f;
    }

    // zero out shared memory SH
    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = (s4_t) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S = 0.0f;
        float M = -FLT_MAX/2;

        // thread indices inside the simdgroup
        const short tx = tiisg%NL;
        const short ty = tiisg/NL;

        // pointer to the mask
        device const half * pm = (device const half *) (mask + iq1*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);

        float slope = 1.0f;

        // ALiBi
        if (FC_flash_attn_ext_vec_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = (int) iwg*C*NSG; ic0 < args.ne11; ic0 += (int) NWG*C*NSG) {
            const int ic = ic0 + C*sgitg;
            if (ic >= args.ne11) {
                break;
            }

            if (FC_flash_attn_ext_vec_has_mask) {
                sm[tiisg] = pm[ic + tiisg];
            }

            // skip -INF blocks
            if (simd_max(sm[tiisg]) == -INFINITY) {
                continue;
            }

            // Q*K^T
            {
                device      const k4_t * pk4 = (device const k4_t *) ((device const char *) k + ic*args.nb11);
                threadgroup const q4_t * pq4 = sq4;

                pk4 += ty*NS10/4 + tx;
                pq4 += tx;

                qk_t mqk[C/NE] = { [ 0 ... C/NE - 1] = 0.0f };

                // each simdgroup processes 1 query and NE (NW/NL) cache elements
                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    if (is_same<kd4_t, k4_t>::value) {
                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            mqk[cc] += dot((float4) pk4[cc*NE*NS10/4 +  ii*NL], (float4) pq4[ii*NL]);
                        }
                    } else {
                        device const kd4_t * pk = (device const kd4_t *) ((device const char *) k + ((ic + NE*cc + ty)*args.nb11));

                        k4_t mk;

                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            deq_k_t4(pk + i/nl_k, i%nl_k, mk);

                            mqk[cc] += dot((float4) mk, (float4) sq4[i]);
                        }
                    }

                    if (NE == 1) {
                        mqk[cc] = simd_sum(mqk[cc]);
                    } else {
                        // simdgroup reduce (NE = 4)
                        // [ 0 ..  7] -> [ 0]
                        // [ 8 .. 15] -> [ 8]
                        // [16 .. 23] -> [16]
                        // [24 .. 31] -> [24]
                        if (NE <= 1) {
                            mqk[cc] += simd_shuffle_down(mqk[cc], 16);
                        }
                        if (NE <= 2) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  8);
                        }
                        if (NE <= 4) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  4);
                        }
                        if (NE <= 8) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  2);
                        }
                        if (NE <= 16) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  1);
                        }

                        // broadcast
                        mqk[cc] = simd_shuffle(mqk[cc], NL*ty);
                    }
                }

                if (FC_flash_attn_ext_vec_has_mask &&
                   !FC_flash_attn_ext_vec_has_scap &&
                   !FC_flash_attn_ext_vec_has_bias) {
                    ss[NE*tx + ty] = fma(mqk[tx], args.scale, (qk_t) sm[NE*tx + ty]);
                } else {
                    mqk[tx] *= args.scale;

                    if (FC_flash_attn_ext_vec_has_scap) {
                        mqk[tx] = args.logit_softcap*precise::tanh(mqk[tx]);
                    }

                    if (FC_flash_attn_ext_vec_has_bias) {
                        mqk[tx] += (qk_t) sm[NE*tx + ty]*slope;
                    } else {
                        mqk[tx] += (qk_t) sm[NE*tx + ty];
                    }

                    ss[NE*tx + ty] = mqk[tx];
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax
            {
                const float m = M;
                const float s = ss[tiisg];

                M = simd_max(max(M, s));

                const float ms = exp(m - M);
                const float vs = exp(s - M);

                S = S*ms + simd_sum(vs);

                // the P matrix from the paper (Q rows, C columns)
                ss[tiisg] = vs;

                // O = diag(ms)*O
                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[ii*NL] *= ms;
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // O = O + (Q*K^T)*V
            {
                o4_t lo[DV4/NL];
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    lo[ii] = 0.0f;
                }

                if (is_same<vd4_t, v4_t>::value) {
                    device const v4_t * pv4 = (device const v4_t *) ((device const char *) v + ic*args.nb21);

                    pv4 += ty*NS20/4 + tx;

                    const auto sst = ss + ty;

                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            lo[ii] += o4_t(float4(pv4[cc*NE*NS20/4 + ii*NL])*float4(sst[cc*NE]));
                        }
                    }
                } else {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        device const vd4_t * pv4 = (device const vd4_t *) ((device const char *) v + ((ic + NE*cc + ty)*args.nb21));

                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            v4_t mv;
                            deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);

                            lo[ii] += o4_t(float4(mv)*float4(ss[NE*cc + ty]));
                        }
                    }
                }

                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    if (NE > 1) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0], 16);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1], 16);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2], 16);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3], 16);
                    }

                    if (NE > 2) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  8);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  8);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  8);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  8);
                    }

                    if (NE > 4) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  4);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  4);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  4);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  4);
                    }

                    if (NE > 8) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  2);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  2);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  2);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  2);
                    }

                    if (NE > 16) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  1);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  1);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  1);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  1);
                    }
                }

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[ii*NL] += lo[ii];
                    }
                }
            }
        }

        if (FC_flash_attn_ext_vec_has_sinks && sgitg == 0 && iwg == 0) {
            const float m = M;
            const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

            M = simd_max(max(M, s));

            const float ms = exp(m - M);
            const float vs = exp(s - M);

            S = S*ms + simd_sum(vs);

            if ((DV4/NL % NW == 0) || ty == 0) {
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    so4[ii*NL] *= ms;
                }
            }
        }

        // these are needed for reducing the results from the simdgroups (reuse the ss buffer)
        if (tiisg == 0) {
            ss[0] = (s_t) S;
            ss[1] = (s_t) M;
        }
    }

    so4 -= tiisg;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduce
    for (short r = NSG/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            const float S0 = ss[           0];
            const float S1 = ss[r*(SH/2) + 0];

            const float M0 = ss[           1];
            const float M1 = ss[r*(SH/2) + 1];

            const float M = max(M0, M1);

            const float ms0 = exp(M0 - M);
            const float ms1 = exp(M1 - M);

            const float S = S0*ms0 + S1*ms1;

            if (tiisg == 0) {
                ss[0] = S;
                ss[1] = M;
            }

            // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
            for (short i = tiisg; i < DV4; i += NW) {
                so4[i] = so4[i]*ms0 + so4[i + r*PV4]*ms1;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // final rescale with 1/S and store to global memory
    if (sgitg == 0) {
        const int64_t nrows = args.ne3*args.ne2*args.ne1;
        const int64_t rid   = iq3*args.ne2*args.ne1 + iq2 + iq1*args.ne1;

        device float4 * dst4 = (device float4 *) dst;
        device float  * dst1 = (device float  *) dst + nrows*DV*NWG; // the S and M are stored after the results

        const float S = NWG == 1 ? 1.0f/ss[0] : 1.0f;

        // interleave the workgroup data
        for (short i = tiisg; i < DV4; i += NW) {
            float4 result = (float4) so4[i]*S;
#if defined(GGML_METAL_HAS_BF16)
            if (NWG == 1 && is_same<q4_t, bfloat4>::value && is_same<o4_t, float4>::value) {
                result = fairy2i_round_to_bf16_f32(result);
            }
#endif
            dst4[rid*DV4*NWG + NWG*i + iwg] = result;
        }

        // store S and M
        if (NWG > 1) {
            if (tiisg == 0) {
                dst1[rid*(2*NWG) + 2*iwg + 0] = ss[0];
                dst1[rid*(2*NWG) + 2*iwg + 1] = ss[1];
            }
        }
    }

#undef NWG
#undef NS10
#undef NS20
}

template<
    typename q4_t,  // query types in shared memory
    typename k4_t,  // key types in shared memory
    typename v4_t,  // value types in shared memory
    typename qk_t,  // Q*K types
    typename s_t,   // soft-max types
    typename s4_t,
    typename o4_t,  // attention accumulation types
    typename kd4_t, // key type in device memory
    short nl_k,
    void (*deq_k_t4)(device const kd4_t *, short, thread k4_t &),
    typename vd4_t, // value type in device memory
    short nl_v,
    void (*deq_v_t4)(device const vd4_t *, short, thread v4_t &),
    short DK,       // K head size
    short DV,       // V head size
    short NE = 4,   // head elements per thread
    short Q  = 1,   // queries per threadgroup
    short C  = 32>  // cache items per threadgroup
kernel void kernel_flash_attn_ext_vec(
        constant ggml_metal_kargs_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
#define FWD_TMPL q4_t, k4_t, v4_t, qk_t, s_t, s4_t, o4_t, kd4_t, nl_k, deq_k_t4, vd4_t, nl_v, deq_v_t4, DK, DV, NE, Q, C
#define FWD_ARGS args, q, k, v, mask, sinks, dst, shmem_f16, tgpig, tiisg, sgitg
    switch (FC_flash_attn_ext_vec_nsg) {
      // note: disabled cases to reduce library load time
        case 1:  kernel_flash_attn_ext_vec_impl<FWD_TMPL,  1>(FWD_ARGS); break;
        case 2:  kernel_flash_attn_ext_vec_impl<FWD_TMPL,  2>(FWD_ARGS); break;
        case 4:  kernel_flash_attn_ext_vec_impl<FWD_TMPL,  4>(FWD_ARGS); break;
      //case 8:  kernel_flash_attn_ext_vec_impl<FWD_TMPL,  8>(FWD_ARGS); break;
      //case 16: kernel_flash_attn_ext_vec_impl<FWD_TMPL, 16>(FWD_ARGS); break;
      //case 32: kernel_flash_attn_ext_vec_impl<FWD_TMPL, 32>(FWD_ARGS); break;
    }
#undef FWD_TMPL
#undef FWD_ARGS
}

// note: I think the s_t can be half instead of float, because the Q*K scaling is done before storing to shared mem
//       in the other (non-vec) kernel, we need s_t to also be float because we scale during the soft_max
//
#define FA_TYPES \
           half4,  \
           half4,  \
           half4,  \
    float,         \
    float, float4, \
           float4

#define FA_TYPES_FAIRY_BF \
         bfloat4,  \
         bfloat4,  \
         bfloat4,  \
    float,         \
    float, float4, \
           float4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;

template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 2>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 64, 64, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2>;

template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv96")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 96, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 96, 96, 4>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 1>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_fairy_bf16_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_FAIRY_BF, bfloat4, 1, dequantize_bf16_t4, bfloat4, 1, dequantize_bf16_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 128, 128, 1>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1>;

template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 2>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 192, 192, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2>;

template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 2>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 192, 128, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2>;

template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 1>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 256, 256, 1>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1>;

template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 2>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 576, 512, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2>;

#undef FA_TYPES
#undef FA_TYPES_FAIRY_BF

constant int32_t FC_flash_attn_ext_vec_reduce_DV  [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 0)]];
constant int32_t FC_flash_attn_ext_vec_reduce_NWG [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 1)]];
constant bool FC_flash_attn_ext_vec_reduce_round_bf16 [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 2)]];

kernel void kernel_flash_attn_ext_vec_reduce(
        constant ggml_metal_kargs_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
#define NWG (FC_flash_attn_ext_vec_reduce_NWG)
#define DV  (FC_flash_attn_ext_vec_reduce_DV)

    const uint64_t rid = tgpig;

    const short iwg = tiisg;

    device const float  * ss    = (device const float  *) htmp + (uint64_t)args.nrows*DV*NWG;

    float S = ss[rid*(2*NWG) + 2*iwg + 0];
    float M = ss[rid*(2*NWG) + 2*iwg + 1];

    const float m  = simd_max(M);
    const float ms = exp(M - m);

    S = 1.0f/simd_sum(S*ms);

    const short DV4 = DV/4;

    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG;
    device       float4 * dst4  = (device       float4 *) dst  + rid*DV4;

    for (short i = sgitg; i < DV4; i += NWG) {
        const float4 v = simd_sum(htmp4[i*NWG + iwg]*ms);

        if (iwg == 0) {
            const float4 result = v*S;
            dst4[i] = FC_flash_attn_ext_vec_reduce_round_bf16 ? fairy2i_round_to_bf16_f32(result) : result;
        }
    }

#undef NWG
#undef DV
}

template<typename T0, typename T1>
kernel void kernel_cpy(
        constant ggml_metal_kargs_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint    tiitg[[thread_index_in_threadgroup]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3  tptg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = tgpig[0]*tptg.y + tiitg/tptg.x;

    if (i01 >= args.ne01) {
        return;
    }

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device T1 * dst_data = (device T1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = tiitg%tptg.x; i00 < args.ne00; i00 += tptg.x) {
        device const T0 * src = (device T0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
        dst_data[i00] = (T1) src[0];
    }
}

typedef decltype(kernel_cpy<float, float>) kernel_cpy_t;

template [[host_name("kernel_cpy_f32_f32")]]   kernel kernel_cpy_t kernel_cpy<float,  float>;
template [[host_name("kernel_cpy_f32_f16")]]   kernel kernel_cpy_t kernel_cpy<float,  half>;
template [[host_name("kernel_cpy_f32_i32")]]   kernel kernel_cpy_t kernel_cpy<float,  int32_t>;
template [[host_name("kernel_cpy_i32_f32")]]   kernel kernel_cpy_t kernel_cpy<int32_t, float>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_cpy_f32_bf16")]]  kernel kernel_cpy_t kernel_cpy<float,  bfloat>;
#endif
template [[host_name("kernel_cpy_f16_f32")]]   kernel kernel_cpy_t kernel_cpy<half,   float>;
template [[host_name("kernel_cpy_f16_f16")]]   kernel kernel_cpy_t kernel_cpy<half,   half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_cpy_bf16_f32")]]  kernel kernel_cpy_t kernel_cpy<bfloat, float>;
template [[host_name("kernel_cpy_bf16_bf16")]] kernel kernel_cpy_t kernel_cpy<bfloat, bfloat>;
#endif

// TODO: templetify these kernels
kernel void kernel_cpy_f32_q8_0(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = tgpig[0];

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n / (args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0) / (args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0) / args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK8_0;

    device block_q8_0 * dst_data = (device block_q8_0 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = tpitg.x*QK8_0; i00 < args.ne00; i00 += ntg.x*QK8_0) {
        device const float * src = (device float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);

        quantize_q8_0(src, dst_data[i00/QK8_0]);
    }
}

kernel void kernel_cpy_f32_q4_0(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = tgpig[0];

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n / (args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0) / (args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0) / args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK4_0;

    device block_q4_0 * dst_data = (device block_q4_0 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = tpitg.x*QK4_0; i00 < args.ne00; i00 += ntg.x*QK4_0) {
        device const float * src = (device float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);

        quantize_q4_0(src, dst_data[i00/QK4_0]);
    }
}

kernel void kernel_cpy_f32_q4_1(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = tgpig[0];

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n / (args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0) / (args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0) / args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK4_1;

    device block_q4_1 * dst_data = (device block_q4_1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = tpitg.x*QK4_1; i00 < args.ne00; i00 += ntg.x*QK4_1) {
        device const float * src = (device float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);

        quantize_q4_1(src, dst_data[i00/QK4_1]);
    }
}

kernel void kernel_cpy_f32_q5_0(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = tgpig[0];

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n / (args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0) / (args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0) / args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK5_0;

    device block_q5_0 * dst_data = (device block_q5_0 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = tpitg.x*QK5_0; i00 < args.ne00; i00 += ntg.x*QK5_0) {
        device const float * src = (device float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);

        quantize_q5_0(src, dst_data[i00/QK5_0]);
    }
}

kernel void kernel_cpy_f32_q5_1(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = tgpig[0];

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n / (args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0) / (args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0) / args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK5_1;

    device block_q5_1 * dst_data = (device block_q5_1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = tpitg.x*QK5_1; i00 < args.ne00; i00 += ntg.x*QK5_1) {
        device const float * src = (device float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);

        quantize_q5_1(src, dst_data[i00/QK5_1]);
    }
}

kernel void kernel_cpy_f32_iq4_nl(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = tgpig[0];

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n / (args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0) / (args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0) / args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK4_NL;

    device block_iq4_nl * dst_data = (device block_iq4_nl *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = tpitg.x*QK4_NL; i00 < args.ne00; i00 += ntg.x*QK4_NL) {
        device const float * src = (device float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);

        quantize_iq4_nl(src, dst_data[i00/QK4_NL]);
    }
}

template<typename T4x4, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread T4x4 &)>
kernel void kernel_cpy_q_f32(
        constant ggml_metal_kargs_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = tgpig[0];

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device const block_q * src_data = (device const block_q *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       T4x4    * dst_data = (device       T4x4    *)(dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = tpitg.x; i00 < args.ne00/16; i00 += ntg.x) {
        T4x4 temp;
        dequantize_func(src_data + i00/nl, i00%nl, temp);
        dst_data[i00] = temp;
    }
}

typedef decltype(kernel_cpy_q_f32<float4x4, block_q4_0, 2, dequantize_q4_0>) cpy_q_f_t;

template [[host_name("kernel_cpy_q4_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q4_0, 2, dequantize_q4_0>;
template [[host_name("kernel_cpy_q4_1_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q4_1, 2, dequantize_q4_1>;
template [[host_name("kernel_cpy_q5_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q5_0, 2, dequantize_q5_0>;
template [[host_name("kernel_cpy_q5_1_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q5_1, 2, dequantize_q5_1>;
template [[host_name("kernel_cpy_q8_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q8_0, 2, dequantize_q8_0>;

template [[host_name("kernel_cpy_q4_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q4_0, 2, dequantize_q4_0>;
template [[host_name("kernel_cpy_q4_1_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q4_1, 2, dequantize_q4_1>;
template [[host_name("kernel_cpy_q5_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q5_0, 2, dequantize_q5_0>;
template [[host_name("kernel_cpy_q5_1_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q5_1, 2, dequantize_q5_1>;
template [[host_name("kernel_cpy_q8_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q8_0, 2, dequantize_q8_0>;

kernel void kernel_concat(
    constant ggml_metal_kargs_concat & args,
    device  const char * src0,
    device  const char * src1,
    device        char * dst,
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]) {

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    int o[4] = {0, 0, 0, 0};
    o[args.dim] = args.dim == 0 ? args.ne00 : (args.dim == 1 ? args.ne01 : (args.dim == 2 ? args.ne02 : args.ne03));

    device const float * x;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        if (i0 < args.ne00 && i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
            x = (device const float *)(src0 + (i3       )*args.nb03 + (i2       )*args.nb02 + (i1       )*args.nb01 + (i0       )*args.nb00);
        } else {
            x = (device const float *)(src1 + (i3 - o[3])*args.nb13 + (i2 - o[2])*args.nb12 + (i1 - o[1])*args.nb11 + (i0 - o[0])*args.nb10);
        }

        device float * y = (device float *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

        *y = *x;
    }
}

kernel void kernel_complex_split(
    constant ggml_metal_kargs_concat & args,
    device  const char * src0,
    device        char * dst,
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]) {
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;
    const int n_dims = args.ne0 / 2;

    device const char * src_row = src0 + i3 * args.nb03 + i2 * args.nb02 + i1 * args.nb01;
    device       char * dst_row = dst  + i3 * args.nb3  + i2 * args.nb2  + i1 * args.nb1;

    for (int i0 = tpitg.x; i0 < n_dims; i0 += ntg.x) {
        const uint v = *((device const uint *) (src_row + i0 * args.nb00));
        *((device float *) (dst_row + i0 * args.nb0)) = fairy2i_bf16_to_f32((ushort) (v & 0xffffU));
        *((device float *) (dst_row + (i0 + n_dims) * args.nb0)) = fairy2i_bf16_to_f32((ushort) (v >> 16));
    }
}

kernel void kernel_complex_merge(
    constant ggml_metal_kargs_concat & args,
    device  const char * src0,
    device        char * dst,
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]) {
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;
    const int half_dims = args.ne0;

    device const char * src_row = src0 + i3 * args.nb03 + i2 * args.nb02 + i1 * args.nb01;
    device       char * dst_row = dst  + i3 * args.nb3  + i2 * args.nb2  + i1 * args.nb1;

    for (int i0 = tpitg.x; i0 < half_dims; i0 += ntg.x) {
        const float real = *((device const float *) (src_row + i0 * args.nb00));
        const float imag = *((device const float *) (src_row + (i0 + half_dims) * args.nb00));
        *((device uint *) (dst_row + i0 * args.nb0)) = fairy2i_pack_bf16_pair(real, imag);
    }
}

kernel void kernel_complex_add(
    constant ggml_metal_kargs_bin & args,
    device  const char * src0,
    device  const char * src1,
    device        char * dst,
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig.z;
    const int i02 = tgpig.y;
    const int i01 = tgpig.x;

    const int i13 = i03 % args.ne13;
    const int i12 = i02 % args.ne12;
    const int i11 = i01 % args.ne11;

    device const char * src0_row = src0 + i03 * args.nb03 + i02 * args.nb02 + i01 * args.nb01;
    device const char * src1_row = src1 + i13 * args.nb13 + i12 * args.nb12 + i11 * args.nb11;
    device       char * dst_row  = dst  + i03 * args.nb3  + i02 * args.nb2  + i01 * args.nb1;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int i10 = i0 % args.ne10;
        const uint a = *((device const uint *) (src0_row + i0 * args.nb00));
        const uint b = *((device const uint *) (src1_row + i10 * args.nb10));

        const ushort real = fairy2i_add_bf16_bits_rne((ushort) (a & 0xffffU), (ushort) (b & 0xffffU));
        const ushort imag = fairy2i_add_bf16_bits_rne((ushort) (a >> 16), (ushort) (b >> 16));
        *((device uint *) (dst_row + i0 * args.nb0)) = (uint) real | ((uint) imag << 16);
    }
}

kernel void kernel_complex_add_qat(
    constant ggml_metal_kargs_bin & args,
    device  const char * src0,
    device  const char * src1,
    device        char * dst,
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig.z;
    const int i02 = tgpig.y;
    const int i01 = tgpig.x;

    const int i13 = i03 % args.ne13;
    const int i12 = i02 % args.ne12;
    const int i11 = i01 % args.ne11;

    device const char * src0_row = src0 + i03 * args.nb03 + i02 * args.nb02 + i01 * args.nb01;
    device const char * src1_row = src1 + i13 * args.nb13 + i12 * args.nb12 + i11 * args.nb11;
    device       char * dst_row  = dst  + i03 * args.nb3  + i02 * args.nb2  + i01 * args.nb1;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int i10 = i0 % args.ne10;
        const uint a = *((device const uint *) (src0_row + i0 * args.nb00));
        const uint b = *((device const uint *) (src1_row + i10 * args.nb10));

        const float real = fairy2i_bf16_to_f32((ushort) (a & 0xffffU)) +
                           fairy2i_bf16_to_f32((ushort) (b & 0xffffU));
        const float imag = fairy2i_bf16_to_f32((ushort) (a >> 16)) +
                           fairy2i_bf16_to_f32((ushort) (b >> 16));
        *((device uint *) (dst_row + i0 * args.nb0)) = fairy2i_pack_bf16_pair(real, imag);
    }
}

template<int nr0, typename args_t>
void kernel_mul_mv_q2_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q2_K * x = (device const block_q2_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const short ix = tiisg/8;  // 0...3
    const short it = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3
    const short is = (8*ir)/16;// 0 or 1

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * 1.f/16.f;
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += args.nb01/2;
            sc += args.nb01;
            dh += args.nb01/2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_q2_K_f32")]]
kernel void kernel_mul_mv_q2_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q3_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q3_K * x = (device const block_q3_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float yl[32];

    //const uint16_t kmask1 = 0x3030;
    //const uint16_t kmask2 = 0x0f0f;

    const short tid = tiisg/4;
    const short ix  = tiisg%4;
    const short ip  = tid/4;          // 0 or 1
    const short il  = 2*((tid%4)/2);  // 0 or 2
    const short ir  = tid%2;
    const short l0  = 8*ir;

    // One would think that the Metal compiler would figure out that ip and il can only have
    // 4 possible states, and optimize accordingly. Well, no. It needs help, and we do it
    // with these two tales.
    //
    // Possible masks for the high bit
    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},  // ip = 0, il = 0
                           {0x0004, 0x0400, 0x0008, 0x0800},  // ip = 0, il = 2
                           {0x0010, 0x1000, 0x0020, 0x2000},  // ip = 1, il = 0
                           {0x0040, 0x4000, 0x0080, 0x8000}}; // ip = 1, il = 2

    // Possible masks for the low 2 bits
    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00}, {0x0030, 0x3000, 0x00c0, 0xc000}};

    const ushort4 hm = mm[2*ip + il/2];

    const short shift = 2*il;

    const float v1 = il == 0 ? 4.f : 64.f;
    const float v2 = 4.f * v1;

    const uint16_t s_shift1 = 4*ip;
    const uint16_t s_shift2 = s_shift1 + il;

    const short q_offset = 32*ip + l0;
    const short y_offset = 128*ip + 32*il + l0;

    device const float * y1 = yy + ix*QK_K + y_offset;

    uint32_t scales32, aux32;
    thread uint16_t * scales16 = (thread uint16_t *)&scales32;
    thread const int8_t * scales = (thread const int8_t *)&scales32;

    float sumf1[nr0] = {0.f};
    float sumf2[nr0] = {0.f};

    for (int i = ix; i < nb; i += 4) {
        for (short l = 0; l < 8; ++l) {
            yl[l+ 0] = y1[l+ 0];
            yl[l+ 8] = y1[l+16];
            yl[l+16] = y1[l+32];
            yl[l+24] = y1[l+48];
        }

        device const uint16_t * q = (device const uint16_t *)(x[i].qs + q_offset);
        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + l0);
        device const uint16_t * a = (device const uint16_t *)(x[i].scales);
        device const half * dh = &x[i].d;

        for (short row = 0; row < nr0; ++row) {
            const float d_all = (float)dh[0];

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il+0];
            scales16[1] = a[il+1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l/2];
                s1 += yl[l+0] * (qs & qm[il/2][0]);
                s2 += yl[l+1] * (qs & qm[il/2][1]);
                s3 += ((h[l/2] & hm[0]) ? 0.f : yl[l+0]) + ((h[l/2] & hm[1]) ? 0.f : yl[l+1]);
                s4 += yl[l+16] * (qs & qm[il/2][2]);
                s5 += yl[l+17] * (qs & qm[il/2][3]);
                s6 += ((h[l/2] & hm[2]) ? 0.f : yl[l+16]) + ((h[l/2] & hm[3]) ? 0.f : yl[l+17]);
            }
            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l/2+8];
                s1 += yl[l+8] * (qs & qm[il/2][0]);
                s2 += yl[l+9] * (qs & qm[il/2][1]);
                s3 += ((h[l/2+8] & hm[0]) ? 0.f : yl[l+8]) + ((h[l/2+8] & hm[1]) ? 0.f : yl[l+9]);
                s4 += yl[l+24] * (qs & qm[il/2][2]);
                s5 += yl[l+25] * (qs & qm[il/2][3]);
                s6 += ((h[l/2+8] & hm[2]) ? 0.f : yl[l+24]) + ((h[l/2+8] & hm[3]) ? 0.f : yl[l+25]);
            }
            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[1] - 32);
            sumf2[row] += d2 * (scales[3] - 32);

            q  += args.nb01/2;
            h  += args.nb01/2;
            a  += args.nb01/2;
            dh += args.nb01/2;
        }

        y1 += 4 * QK_K;
    }

    for (int row = 0; row < nr0; ++row) {
        const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift);
        sumf1[row] = simd_sum(sumf);
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (tiisg == 0) {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            dst_f32[first_row + row] = sumf1[row];
        }
    }
}

[[host_name("kernel_mul_mv_q3_K_f32")]]
kernel void kernel_mul_mv_q3_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q3_K_f32_impl<N_R0_Q3_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q4_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg/8;  // 0...3
    const short it = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q4_K * x = (device const block_q4_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[16];
    float yh[16];

    float sumf[nr0]={0.f};

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01/2;
            sc += args.nb01/2;
            dh += args.nb01/2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_q4_K_f32")]]
kernel void kernel_mul_mv_q4_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q5_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q5_K * x = (device const block_q5_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0]={0.f};

    float yl[16], yh[16];

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short tid = tiisg/4;
    const short ix  = tiisg%4;
    const short iq  = tid/4;
    const short ir  = tid%4;

    const short l0 = 8*ir;
    const short q_offset = 32*iq + l0;
    const short y_offset = 64*iq + l0;

    const uint8_t hm1 = 1u << (2*iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix*QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;

        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        for (short row = 0; row < nr0; ++row) {
            device const uint8_t * q2 = q1 + 64;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1 = {0.f};
            float4 acc2 = {0.f};
            FOR_UNROLL (short l = 0; l < 8; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l+0] : 0.f;
                acc2[1] += h & hm2 ? yl[l+8] : 0.f;
                acc2[2] += h & hm3 ? yh[l+0] : 0.f;
                acc2[3] += h & hm4 ? yh[l+8] : 0.f;
            }

            sumf[row] += dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                                  sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                  sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                                  sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01;
            qh += args.nb01;
            dh += args.nb01/2;
            a  += args.nb01/2;
        }

        y1 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q5_K_f32")]]
kernel void kernel_mul_mv_q5_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q5_K_f32_impl<N_R0_Q5_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q6_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0] = { 0.f };

    float yl[16];

    const short tid = tiisg/2;
    const short ix  = tiisg%2;
    const short ip  = tid/8;         // 0 or 1
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;
        device const half    * dh = &x[i].d;

        device const float * y = yy + i * QK_K + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        for (short row = 0; row < nr0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

            q1 += args.nb01;
            q2 += args.nb01;
            qh += args.nb01;
            sc += args.nb01;
            dh += args.nb01/2;
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_q6_K_f32")]]
kernel void kernel_mul_mv_q6_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q6_K_f32_impl<N_R0_Q6_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

// ======================= "True" 2-bit

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xxs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xxs * x = (device const block_iq2_xxs *) (src0 + offset0);
    device const float         * y = (device const float         *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float sum = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + aux8[l]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 8; ++j) {
                    sum += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * sum;

            dh += args.nb01/2;
            q2 += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xxs_f32")]]
kernel void kernel_mul_mv_iq2_xxs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xs * x = (device const block_iq2_xs *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 512);
    {
        int nval = 8;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const uint8_t  * sc = xr->scales + ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const uint8_t ls1 = sc[0] & 0xf;
            const uint8_t ls2 = sc[0] >>  4;
            const float d1 = db * (0.5f + ls1);
            const float d2 = db * (0.5f + ls2);

            float sum1 = 0, sum2 = 0;
            for (short l = 0; l < 2; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + (q2[l] & 511));
                const uint8_t signs = ssigns[(q2[l] >> 9)];
                for (short j = 0; j < 8; ++j) {
                    sum1 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            for (short l = 2; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + (q2[l] & 511));
                const uint8_t signs = ssigns[(q2[l] >> 9)];
                for (short j = 0; j < 8; ++j) {
                    sum2 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d1 * sum1 + d2 * sum2;

            dh += args.nb01/2;
            q2 += args.nb01/2;
            sc += args.nb01;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xs_f32")]]
kernel void kernel_mul_mv_iq2_xs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xs_f32_impl<N_R0_IQ2_XS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq3_xxs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq3_xxs * x = (device const block_iq3_xxs *) (src0 + offset0);
    device const float         * y = (device const float         *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * svalues = (threadgroup uint32_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_xxs * xr = x + ibl;
        device const uint8_t  * q3 = xr->qs + 8 * ib;
        device const uint16_t * gas = (device const uint16_t *)(xr->qs + QK_K/4) + 2 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const uint32_t aux32 = gas[0] | (gas[1] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + q3[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + q3[2*l+1]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh  += args.nb01/2;
            q3  += args.nb01;
            gas += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all * 0.5f;
        }
    }
}

[[host_name("kernel_mul_mv_iq3_xxs_f32")]]
kernel void kernel_mul_mv_iq3_xxs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_xxs_f32_impl<N_R0_IQ3_XXS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq3_s_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq3_s * x = (device const block_iq3_s *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
    {
        int nval = 8;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3s_grid[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 8 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + (ib/2);
        device const uint8_t * signs = xr->signs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const float d = db * (1 + 2*((sc[0] >> 4*(ib%2)) & 0xf));

            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint32_t * table1 = qh[0] & kmask_iq2xs[2*l+0] ? svalues + 256 : svalues;
                const threadgroup uint32_t * table2 = qh[0] & kmask_iq2xs[2*l+1] ? svalues + 256 : svalues;
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(table1 + qs[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(table2 + qs[2*l+1]);
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * select(1, -1, signs[l] & kmask_iq2xs[j+0]);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * select(1, -1, signs[l] & kmask_iq2xs[j+4]);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh    += args.nb01/2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq3_s_f32")]]
kernel void kernel_mul_mv_iq3_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_s_f32_impl<N_R0_IQ3_S, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_s_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_s * x = (device const block_iq2_s *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    //threadgroup uint64_t * svalues = (threadgroup uint64_t *) shmem;
    //{
    //    int nval = 32;
    //    int pos  = (32*sgitg + tiisg)*nval;
    //    for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2s_grid[pos + i];
    //    threadgroup_barrier(mem_flags::mem_threadgroup);
    //}

    const short ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 4 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + ib;
        device const uint8_t * signs = qs + QK_K/8;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const float d1 = db * (0.5f + (sc[0] & 0xf));
            const float d2 = db * (0.5f + (sc[0] >>  4));

            float2 sum = {0};
            for (short l = 0; l < 2; ++l) {
                //const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                //const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                for (short j = 0; j < 8; ++j) {
                    sum[0] += yl[8*l + j +  0] * grid1[j] * select(1, -1, signs[l+0] & kmask_iq2xs[j]);
                    sum[1] += yl[8*l + j + 16] * grid2[j] * select(1, -1, signs[l+2] & kmask_iq2xs[j]);
                }
            }
            sumf[row] += d1 * sum[0] + d2 * sum[1];

            dh    += args.nb01/2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_s_f32")]]
kernel void kernel_mul_mv_iq2_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_s_f32_impl<N_R0_IQ2_S, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq1_s_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq1_s * x = (device const block_iq1_s *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    const short ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        float sumy = 0;
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
            sumy += yl[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_s * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint16_t * qh = xr->qh + ib;
        device const half     * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 5) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[0] << 2) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[0] >> 1) & 0x700)));

            float sum = 0;
            for (short j = 0; j < 4; ++j) {
                sum += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                     + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4)
                     + yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                     + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            sumf[row] += (float)dh[0] * (sum + sumy * (qh[0] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA)) * (2*((qh[0] >> 12) & 7) + 1);

            dh += args.nb01/2;
            qs += args.nb01;
            qh += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq1_s_f32")]]
kernel void kernel_mul_mv_iq1_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_s_f32_impl<N_R0_IQ1_S, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq1_m_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq1_m * x = (device const block_iq1_m *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    const short ix = tiisg;

    device const float * y4 = y + 32 * ix;

    iq1m_scale_t scale;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        float4 sumy = {0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+ 8]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+16]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+24]; sumy[3] += yl[i+24];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_m * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint8_t  * qh = xr->qh + 2 * ib;
        device const uint16_t * sc = (device const uint16_t *)xr->scales;

        for (short row = 0; row < nr0; row++) {
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[1] << 8) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[1] << 4) & 0x700)));

            float2 sum = {0.f};
            for (short j = 0; j < 4; ++j) {
                sum[0] += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                        + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4);
                sum[1] += yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                        + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            const float delta1 = sumy[0] * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[1] * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
            const float delta2 = sumy[2] * (qh[1] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[3] * (qh[1] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);

            sumf[row] += (float)scale.f16 * ((sum[0] + delta1) * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1) +
                                             (sum[1] + delta2) * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 7) + 1));

            sc += args.nb01/2;
            qs += args.nb01;
            qh += args.nb01;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq1_m_f32")]]
kernel void kernel_mul_mv_iq1_m_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_m_f32_impl<N_R0_IQ1_M, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq4_nl_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;
    const int nb = args.ne00/QK4_NL;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq4_nl * x = (device const block_iq4_nl *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    const short ix = tiisg/2;  // 0...15
    const short it = tiisg%2;  // 0 or 1

    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[nr0]={0.f};

    device const float * yb = y + ix * QK4_NL + it * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ib = ix; ib < nb; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < nr0; row++) {
            device const block_iq4_nl & xb = x[row*nb + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] | (q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[2] | (q4[3] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            sumf[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 16 * QK4_NL;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq4_nl_f32")]]
kernel void kernel_mul_mv_iq4_nl_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_nl_f32_impl<N_R0_IQ4_NL, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq4_xs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;
    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq4_xs * x = (device const block_iq4_xs *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    const short ix = tiisg/16;  // 0 or 1
    const short it = tiisg%16;  // 0...15
    const short ib = it/2;
    const short il = it%2;

    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[nr0]={0.f};

    device const float * yb = y + ix * QK_K + ib * 32 + il * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ibl = ix; ibl < nb; ibl += 2) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < nr0; ++row) {
            device const block_iq4_xs & xb = x[row*nb + ibl];
            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 16*ib + 8*il);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
            sumf[row] += (float)xb.d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 2 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq4_xs_f32")]]
kernel void kernel_mul_mv_iq4_xs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_xs_f32_impl<N_R0_IQ4_XS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_mxfp4_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;
    const int nb = args.ne00/QK_MXFP4;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_mxfp4 * x = (device const block_mxfp4 *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    const short ix = tiisg/2;  // 0...15
    const short it = tiisg%2;  // 0 or 1

    shmem_f32[tiisg] = kvalues_mxfp4_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[nr0]={0.f};

    device const float * yb = y + ix * QK_MXFP4 + it * 8;

    for (int ib = ix; ib < nb; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

#pragma unroll(nr0)
        for (short row = 0; row < nr0; row++) {
            device const block_mxfp4 & xb = x[row*nb + ib];
            device const uint8_t     * q2 = (device const uint8_t *)(xb.qs + 8*it);

            float4 acc1 = yl[0]*float4(shmem_f32[q2[0] &  0x0F], shmem_f32[q2[1] &  0x0F], shmem_f32[q2[2] &  0x0F], shmem_f32[q2[3] &  0x0F]);
            float4 acc2 = yl[1]*float4(shmem_f32[q2[0] >> 4   ], shmem_f32[q2[1] >> 4   ], shmem_f32[q2[2] >> 4   ], shmem_f32[q2[3] >> 4   ]);
            float4 acc3 = yl[2]*float4(shmem_f32[q2[4] &  0x0F], shmem_f32[q2[5] &  0x0F], shmem_f32[q2[6] &  0x0F], shmem_f32[q2[7] &  0x0F]);
            float4 acc4 = yl[3]*float4(shmem_f32[q2[4] >> 4   ], shmem_f32[q2[5] >> 4   ], shmem_f32[q2[6] >> 4   ], shmem_f32[q2[7] >> 4   ]);

            acc1 = (acc1 + acc3) + (acc2 + acc4);

            sumf[row] += e8m0_to_fp32(xb.e) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
        }

        yb += 16 * QK_MXFP4;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_mxfp4_f32")]]
kernel void kernel_mul_mv_mxfp4_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_get_rows_q(
        constant ggml_metal_kargs_get_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((const device int32_t *) ((const device char *) src1 + i11*args.nb11 + i10*args.nb10))[0];

    const int64_t i02 = i11;

    for (int64_t ind = tiitg; ind < args.ne00/16; ind += tptg.x) {
        float4x4 temp;
        dequantize_func(((device const block_q *) ((const device char *) src0 + r*args.nb01 + i02*args.nb02)) + ind/nl, ind%nl, temp);
        *(((device float4x4 *) ((device char *) dst + i11*args.nb2 + i10*args.nb1)) + ind) = temp;
    }
}

template<typename T>
kernel void kernel_get_rows_f(
        constant ggml_metal_kargs_get_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((const device int32_t *) ((const device char *) src1 + i11*args.nb11 + i10*args.nb10))[0];

    const int64_t i02 = i11;

    for (int ind = tiitg; ind < args.ne00; ind += tptg.x) {
        ((      device float *) ((      device char *)  dst + i11*args.nb2  + i10*args.nb1))[ind] =
        ((const device T     *) ((const device char *) src0 + i02*args.nb02 +  r*args.nb01))[ind];
    }
}

kernel void kernel_get_rows_i32(
        constant ggml_metal_kargs_get_rows & args,
        device const  void * src0,
        device const  void * src1,
        device     int32_t * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((const device int32_t *) ((const device char *) src1 + i11*args.nb11 + i10*args.nb10))[0];

    const int64_t i02 = i11;

    for (int ind = tiitg; ind < args.ne00; ind += tptg.x) {
        ((      device int32_t *) ((      device char *) dst  + i11*args.nb2 + i10*args.nb1))[ind] =
        ((const device int32_t *) ((const device char *) src0 + i02*args.nb02 + r*args.nb01))[ind];
    }
}

template<typename block_q, void (*quantize_func)(device const float *, device block_q &)>
kernel void kernel_set_rows_q32(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;

    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const int64_t i1 = ((const device int64_t *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_q * dst_row = (      device block_q *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device float   * src_row = (const device float   *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    for (int ind = tiitg%tptg.x; ind < args.nk0; ind += tptg.x) {
        quantize_func(src_row + 32*ind, dst_row[ind]);
    }
}

template<typename T>
kernel void kernel_set_rows_f(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;

    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const int64_t i1 = ((const device int64_t *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device T     * dst_row = (      device T     *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device float * src_row = (const device float *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    for (int ind = tiitg%tptg.x; ind < args.nk0; ind += tptg.x) {
        dst_row[ind] = (T) src_row[ind];
    }
}

kernel void kernel_set_rows_bf16_raw(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device        void * dst,
        uint3                tgpig [[threadgroup_position_in_grid]],
        uint                 tiitg [[thread_index_in_threadgroup]],
        uint3                tptg  [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03 % args.ne12;
    const int32_t i11 = i02 % args.ne11;

    const int32_t i01 = tgpig.x * tptg.y + tiitg / tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int64_t i1 = ((const device int64_t *) ((const device char *) src1 +
                                                  i01 * args.nb10 + i11 * args.nb11 + i12 * args.nb12))[0];

    device ushort * dst_row =
        (device ushort *) ((device char *) dst + i1 * args.nb1 + i02 * args.nb2 + i03 * args.nb3);
    const device ushort * src_row =
        (const device ushort *) ((const device char *) src0 +
                                 i01 * args.nb01 + i02 * args.nb02 + i03 * args.nb03);

    for (int ind = tiitg % tptg.x; ind < args.nk0; ind += tptg.x) {
        dst_row[ind] = src_row[ind];
    }
}

kernel void kernel_set_rows_bf16_carrier_rows(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device        void * dst,
        uint3                tgpig [[threadgroup_position_in_grid]],
        uint                 tiitg [[thread_index_in_threadgroup]],
        uint3                tptg  [[threads_per_threadgroup]]) {
    const int32_t i02 = tgpig.y;
    const int32_t i03 = tgpig.z;
    const int32_t i11 = i02 % args.ne11;
    const int32_t i12 = i03 % args.ne12;
    const int32_t i01 = tgpig.x * tptg.y + tiitg / tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int64_t i1 = ((const device int64_t *) ((const device char *) src1 +
                                                  i01 * args.nb10 + i11 * args.nb11 + i12 * args.nb12))[0];
    device ushort * dst_row =
        (device ushort *) ((device char *) dst + i1 * args.nb1 + i02 * args.nb2 + i03 * args.nb3);
    const device uint * src_row =
        (const device uint *) ((const device char *) src0 +
                               i01 * args.nb01 + i02 * args.nb02 + i03 * args.nb03);

    for (int ind = tiitg % tptg.x; ind < args.nk0; ind += tptg.x) {
        dst_row[ind] = (ushort) (src_row[ind] >> 16);
    }
}

kernel void kernel_set_rows_bf16_carrier_elements(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device        void * dst,
        uint3                tgpig [[threadgroup_position_in_grid]],
        uint                 tiitg [[thread_index_in_threadgroup]],
        uint3                tptg  [[threads_per_threadgroup]]) {
    const uint64_t element = (uint64_t) tgpig.x * tptg.x + tiitg;
    const uint64_t n_elements = (uint64_t) args.ne00 * args.ne01 * args.ne02;
    if (element >= n_elements) {
        return;
    }

    const uint64_t i0 = element % args.ne00;
    const uint64_t i1 = (element / args.ne00) % args.ne01;
    const uint64_t i2 = element / ((uint64_t) args.ne00 * args.ne01);
    const device uint * src = (const device uint *) ((const device char *) src0 +
                                                     i0 * sizeof(uint) + i1 * args.nb01 + i2 * args.nb02);
    const int64_t dst_element = ((const device int64_t *) ((const device char *) src1 + element * args.nb10))[0];
    *(device ushort *) ((device char *) dst + dst_element * args.nb1) = (ushort) (*src >> 16);
}

#define BLOCK_SIZE_M 64 // 8 simdgroup matrices from matrix A
#define BLOCK_SIZE_N 32 // 4 simdgroup matrices from matrix B
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 4 // each thread take 4 simdgroup matrices from matrix A
#define THREAD_MAT_N 2 // each thread take 2 simdgroup matrices from matrix B
#define THREAD_PER_BLOCK 128
#define THREAD_PER_ROW 2 // 2 thread for each row in matrix A to load numbers
#define THREAD_PER_COL 4 // 4 thread for each row in matrix B to load numbers
#define SG_MAT_SIZE 64 // simdgroup matrix is of shape 8x8
#define SG_MAT_ROW 8

// each block_q contains 16*nl weights
template<typename T, typename T4x4, typename simdgroup_T8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread T4x4 &)>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup T     * sa = (threadgroup T     *)(shmem);
    threadgroup float * sb = (threadgroup float *)(shmem + 4096);

    const int r0 = tgpig.y;
    const int r1 = tgpig.x;
    const int im = tgpig.z;

    // if this block is of 64x32 shape or smaller
    const short n_rows = (args.ne0 - r0*BLOCK_SIZE_M < BLOCK_SIZE_M) ? (args.ne0 - r0*BLOCK_SIZE_M) : BLOCK_SIZE_M;
    const short n_cols = (args.ne1 - r1*BLOCK_SIZE_N < BLOCK_SIZE_N) ? (args.ne1 - r1*BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    const short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    const short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_T8x8     ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    short il = (tiitg % THREAD_PER_ROW);

    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const short    offset1 = il/nl;

    device const block_q * x = (device const block_q *)(src0
        + args.nb01*(r0*BLOCK_SIZE_M + thread_row) + offset0) + offset1;

    device const float   * y = (device const float   *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1*BLOCK_SIZE_N + thread_col)
        + args.nb10*(BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < args.ne00; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        T4x4 temp_a;
        dequantize_func(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (short i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg/THREAD_PER_ROW/8) \
            +                     (tiitg%THREAD_PER_ROW)*16 + (i/8)*8) \
            +                     (tiitg/THREAD_PER_ROW)%8  + (i&7)*8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + 32*8*(tiitg%THREAD_PER_COL) + 8*(tiitg/THREAD_PER_COL)) = *((device float2x4 *) y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const T     * lsma = (sa + THREAD_MAT_M*SG_MAT_SIZE*(sgitg%2));
        threadgroup const float * lsmb = (sb + THREAD_MAT_N*SG_MAT_SIZE*(sgitg/2));

        #pragma unroll(4)
        for (short ik = 0; ik < BLOCK_SIZE_K/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            #pragma unroll(4)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + SG_MAT_SIZE * i);
            }

            #pragma unroll(2)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + SG_MAT_SIZE * i);
            }

            simdgroup_barrier(mem_flags::mem_none);

            #pragma unroll(8)
            for (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += (BLOCK_SIZE_M/SG_MAT_ROW)*SG_MAT_SIZE;
            lsmb += (BLOCK_SIZE_N/SG_MAT_ROW)*SG_MAT_SIZE;
        }
    }

    if ((r0 + 1) * BLOCK_SIZE_M <= args.ne0 && (r1 + 1) * BLOCK_SIZE_N <= args.ne1) {
        device float * C = (device float *) dst +
            (BLOCK_SIZE_M * r0 + 32*(sgitg &  1)) + \
            (BLOCK_SIZE_N * r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i%4) + 8 * args.ne0 * (i/4), args.ne0);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *) shmem) \
                                     + 32*(sgitg&1) + (16*(sgitg >> 1))*BLOCK_SIZE_M;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*BLOCK_SIZE_M*(i/4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                device float  * D  = (device float  *) dst + (r0*BLOCK_SIZE_M) + (r1*BLOCK_SIZE_N + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*BLOCK_SIZE_M);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < n_rows/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < n_rows; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

template<short ne20> // n_expert_used
kernel void kernel_mul_mm_id_map0(
        constant ggml_metal_kargs_mul_mm_id_map0 & args,
        device  const char * src2,
        device        char * htpe,
        device        char * hids,
        threadgroup   char * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const short ide = tpitg; // expert id

    uint32_t n_all = 0;

    device int32_t * ids_i32 = (device int32_t *) hids + ide*args.ne21;

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) { // n_tokens
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (i21 + tpitg)*args.nb21);

            threadgroup uint16_t * sids = (threadgroup uint16_t *) shmem + tpitg*ne20;

            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sids[i20] = src2_i32[i20];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) {
                break;
            }

            threadgroup const uint16_t * sids = (threadgroup const uint16_t *) shmem + t*ne20;

            short sel = 0;
            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sel += (sids[i20] == ide)*(i20 + 1);
            }

            ids_i32[n_all] = (i21 + t)*ne20 + sel - 1;

            n_all += sel > 0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device uint32_t * tpe_u32 = (device uint32_t *) (htpe);
    tpe_u32[ide] = n_all;
}

typedef decltype(kernel_mul_mm_id_map0<1>) kernel_mul_mm_id_map0_t;

template [[host_name("kernel_mul_mm_id_map0_ne20_1" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<1>;
template [[host_name("kernel_mul_mm_id_map0_ne20_2" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<2>;
template [[host_name("kernel_mul_mm_id_map0_ne20_4" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<4>;
template [[host_name("kernel_mul_mm_id_map0_ne20_6" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<6>;
template [[host_name("kernel_mul_mm_id_map0_ne20_8" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<8>;
template [[host_name("kernel_mul_mm_id_map0_ne20_10")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<10>;
template [[host_name("kernel_mul_mm_id_map0_ne20_16")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<16>;

template<typename T, typename T4x4, typename simdgroup_T8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread T4x4 &)>
kernel void kernel_mul_mm_id(
        constant ggml_metal_kargs_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup T    * sa = (threadgroup T    *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int r0 = tgpig.y;
    const int r1 = tgpig.x;
    const int im = tgpig.z; // expert

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1*BLOCK_SIZE_N >= neh1) {
        return;
    }

    // if this block is of 64x32 shape or smaller
    const short n_rows = (args.ne0 - r0*BLOCK_SIZE_M < BLOCK_SIZE_M) ? (args.ne0 - r0*BLOCK_SIZE_M) : BLOCK_SIZE_M;
    const short n_cols = (    neh1 - r1*BLOCK_SIZE_N < BLOCK_SIZE_N) ? (    neh1 - r1*BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    const short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    const short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_T8x8     ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    short il = (tiitg % THREAD_PER_ROW);

    const int id = ids_i32[im*args.ne21 + r1*BLOCK_SIZE_N + thread_col];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il/nl;

    device const block_q * x = (device const block_q *)(src0
        + args.nb01*(r0*BLOCK_SIZE_M + thread_row) + offset0) + offset1;

    device const float   * y = (device const float   *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*(BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < args.ne00; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        T4x4 temp_a;
        dequantize_func(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (short i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg/THREAD_PER_ROW/8) \
            +                     (tiitg%THREAD_PER_ROW)*16 + (i/8)*8) \
            +                     (tiitg/THREAD_PER_ROW)%8  + (i&7)*8) = temp_a[i/4][i%4];
        }

        *(threadgroup half2x4 *)(sb + 32*8*(tiitg%THREAD_PER_COL) + 8*(tiitg/THREAD_PER_COL)) = (half2x4)(*((device float2x4 *) y));

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const T    * lsma = (sa + THREAD_MAT_M*SG_MAT_SIZE*(sgitg%2));
        threadgroup const half * lsmb = (sb + THREAD_MAT_N*SG_MAT_SIZE*(sgitg/2));

        #pragma unroll(4)
        for (short ik = 0; ik < BLOCK_SIZE_K/8; ik++) {
            #pragma unroll(4)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + SG_MAT_SIZE * i);
            }

            simdgroup_barrier(mem_flags::mem_none);

            #pragma unroll(2)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + SG_MAT_SIZE * i);
            }

            #pragma unroll(8)
            for (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += (BLOCK_SIZE_M/SG_MAT_ROW)*SG_MAT_SIZE;
            lsmb += (BLOCK_SIZE_N/SG_MAT_ROW)*SG_MAT_SIZE;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *) shmem) \
                                 + 32*(sgitg&1) + (16*(sgitg >> 1))*BLOCK_SIZE_M;

    #pragma unroll(8)
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*BLOCK_SIZE_M*(i/4), BLOCK_SIZE_M);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < n_cols; j += 4) {
        const int id = ids_i32[im*args.ne21 + r1*BLOCK_SIZE_N + j];

        const short ide = id % args.ne20;
        const short idt = id / args.ne20;

        device float  * D  = (device float  *) dst + (r0*BLOCK_SIZE_M) + ide*args.ne0 + idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float  *) shmem + (j*BLOCK_SIZE_M);
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < n_rows/4; i += 32) {
            *(D4 + i) = *(C4 + i);
        }

        i = (4*(n_rows/4)) + tiisg;
        for (; i < n_rows; i += 32) {
            *(D + i) = *(C + i);
        }
    }
}

#define QK_NL 16

//
// get rows
//

typedef decltype(kernel_get_rows_f<float>) get_rows_f_t;

template [[host_name("kernel_get_rows_f32")]]  kernel get_rows_f_t kernel_get_rows_f<float>;
template [[host_name("kernel_get_rows_f16")]]  kernel get_rows_f_t kernel_get_rows_f<half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_get_rows_bf16")]] kernel get_rows_f_t kernel_get_rows_f<bfloat>;
#endif

typedef decltype(kernel_get_rows_q<block_q4_0, 2, dequantize_q4_0>) get_rows_q_t;

template [[host_name("kernel_get_rows_q4_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_0,    2, dequantize_q4_0>;
template [[host_name("kernel_get_rows_q4_1")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_1,    2, dequantize_q4_1>;
template [[host_name("kernel_get_rows_q5_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_0,    2, dequantize_q5_0>;
template [[host_name("kernel_get_rows_q5_1")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_1,    2, dequantize_q5_1>;
template [[host_name("kernel_get_rows_q8_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q8_0,    2, dequantize_q8_0>;
template [[host_name("kernel_get_rows_mxfp4")]]   kernel get_rows_q_t kernel_get_rows_q<block_mxfp4,   2, dequantize_mxfp4>;
template [[host_name("kernel_get_rows_q2_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q2_K,    QK_NL, dequantize_q2_K>;
template [[host_name("kernel_get_rows_q3_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q3_K,    QK_NL, dequantize_q3_K>;
template [[host_name("kernel_get_rows_q4_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_K,    QK_NL, dequantize_q4_K>;
template [[host_name("kernel_get_rows_q5_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_K,    QK_NL, dequantize_q5_K>;
template [[host_name("kernel_get_rows_q6_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q6_K,    QK_NL, dequantize_q6_K>;
template [[host_name("kernel_get_rows_iq2_xxs")]] kernel get_rows_q_t kernel_get_rows_q<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_get_rows_iq2_xs")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_get_rows_iq3_xxs")]] kernel get_rows_q_t kernel_get_rows_q<block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_get_rows_iq3_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq3_s,   QK_NL, dequantize_iq3_s>;
template [[host_name("kernel_get_rows_iq2_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq2_s,   QK_NL, dequantize_iq2_s>;
template [[host_name("kernel_get_rows_iq1_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_get_rows_iq1_m")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq1_m,   QK_NL, dequantize_iq1_m>;
template [[host_name("kernel_get_rows_iq4_nl")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq4_nl,  2,     dequantize_iq4_nl>;
template [[host_name("kernel_get_rows_iq4_xs")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq4_xs,  QK_NL, dequantize_iq4_xs>;

//
// set rows
//

typedef decltype(kernel_set_rows_f<float>) set_rows_f_t;

template [[host_name("kernel_set_rows_f32")]]  kernel set_rows_f_t kernel_set_rows_f<float>;
template [[host_name("kernel_set_rows_f16")]]  kernel set_rows_f_t kernel_set_rows_f<half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_set_rows_bf16")]] kernel set_rows_f_t kernel_set_rows_f<bfloat>;
#endif

typedef decltype(kernel_set_rows_q32<block_q8_0, quantize_q8_0>) set_rows_q32_t;

template [[host_name("kernel_set_rows_q8_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<block_q8_0,   quantize_q8_0>;
template [[host_name("kernel_set_rows_q4_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<block_q4_0,   quantize_q4_0>;
template [[host_name("kernel_set_rows_q4_1")]]   kernel set_rows_q32_t kernel_set_rows_q32<block_q4_1,   quantize_q4_1>;
template [[host_name("kernel_set_rows_q5_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<block_q5_0,   quantize_q5_0>;
template [[host_name("kernel_set_rows_q5_1")]]   kernel set_rows_q32_t kernel_set_rows_q32<block_q5_1,   quantize_q5_1>;
template [[host_name("kernel_set_rows_iq4_nl")]] kernel set_rows_q32_t kernel_set_rows_q32<block_iq4_nl, quantize_iq4_nl>;

//
// matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, float4x4, 1, dequantize_f32>) mul_mm_t;

template [[host_name("kernel_mul_mm_f32_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32>;
template [[host_name("kernel_mul_mm_f16_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_bf16_f32")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16>;
#endif
template [[host_name("kernel_mul_mm_q4_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0>;
template [[host_name("kernel_mul_mm_q4_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1>;
template [[host_name("kernel_mul_mm_q5_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0>;
template [[host_name("kernel_mul_mm_q5_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1>;
template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0>;
template [[host_name("kernel_mul_mm_mxfp4_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4>;
template [[host_name("kernel_mul_mm_q2_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K>;
template [[host_name("kernel_mul_mm_q3_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K>;
template [[host_name("kernel_mul_mm_q4_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K>;
template [[host_name("kernel_mul_mm_q5_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K>;
template [[host_name("kernel_mul_mm_q6_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mm_iq2_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_mul_mm_iq3_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s>;
template [[host_name("kernel_mul_mm_iq2_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s>;
template [[host_name("kernel_mul_mm_iq1_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_mul_mm_iq1_m_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m>;
template [[host_name("kernel_mul_mm_iq4_nl_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl>;
template [[host_name("kernel_mul_mm_iq4_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs>;

//
// indirect matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, float4x4, 1, dequantize_f32>) mul_mm_id;

template [[host_name("kernel_mul_mm_id_f32_f16")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32>;
template [[host_name("kernel_mul_mm_id_f16_f16")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_bf16_f16")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16>;
#endif
template [[host_name("kernel_mul_mm_id_q4_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0>;
template [[host_name("kernel_mul_mm_id_q4_1_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1>;
template [[host_name("kernel_mul_mm_id_q5_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0>;
template [[host_name("kernel_mul_mm_id_q5_1_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1>;
template [[host_name("kernel_mul_mm_id_q8_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0>;
template [[host_name("kernel_mul_mm_id_mxfp4_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4>;
template [[host_name("kernel_mul_mm_id_q2_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K>;
template [[host_name("kernel_mul_mm_id_q3_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K>;
template [[host_name("kernel_mul_mm_id_q4_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K>;
template [[host_name("kernel_mul_mm_id_q5_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K>;
template [[host_name("kernel_mul_mm_id_q6_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_mul_mm_id_iq3_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s>;
template [[host_name("kernel_mul_mm_id_iq2_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s>;
template [[host_name("kernel_mul_mm_id_iq1_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_mul_mm_id_iq1_m_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs>;


//
// matrix-vector multiplication
//

typedef void (kernel_mul_mv_impl_t)(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg);

typedef void (kernel_mul_mv2_impl_t)(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg);

template<kernel_mul_mv_impl_t impl_fn>
void mmv_fn(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    impl_fn(args, src0, src1, dst, tgpig, tiisg);
}

template<kernel_mul_mv2_impl_t impl_fn>
void mmv_fn(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    impl_fn(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(mmv_fn<kernel_mul_mv_t_t_impl<half, half, N_R0_F, ggml_metal_kargs_mul_mv>>) mul_mv_impl_fn_t;

template<mul_mv_impl_fn_t impl_fn>
kernel void kernel_mul_mv_id(
        constant ggml_metal_kargs_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    const int64_t i1 = idx;
    const int64_t i2 = i12;

    device const char * src0_cur = src0s + i02*args.nb02;
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;

    device char * dst_cur = dst + (i1*args.ne0 + i2*args.ne1*args.ne0)*sizeof(float);

    ggml_metal_kargs_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1, // args.ne02,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02, // args.ne02 == 1
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1, // args.ne11,
        /*.ne12 =*/ 1, // args.ne12,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12, // ne12 == 1
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1, // args.ne1,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    impl_fn(
        args0,
        /* src0 */ src0_cur,
        /* src1 */ src1_cur,
        /* dst  */ dst_cur,
        shmem,
        tgpig,
        tiitg,
        tiisg,
        sgitg);
}

typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_impl<float, float, N_R0_F>>>) kernel_mul_mv_id_t;

typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_4_impl<float, float4, float, float4, N_R0_F>>>) kernel_mul_mv_id_4_t;

template [[host_name("kernel_mul_mv_id_f32_f32")]]     kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_impl<float, float, N_R0_F>>>;
template [[host_name("kernel_mul_mv_id_f16_f32")]]     kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_impl<half,  float, N_R0_F>>>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_id_bf16_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_impl<bfloat, float, N_R0_F>>>;
#endif
template [[host_name("kernel_mul_mv_id_f32_f32_4")]]   kernel kernel_mul_mv_id_4_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_4_impl<float, float4, float, float4, N_R0_F>>>;
template [[host_name("kernel_mul_mv_id_f16_f32_4")]]   kernel kernel_mul_mv_id_4_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_4_impl<half,  half4,  float, float4, N_R0_F>>>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_id_bf16_f32_4")]]  kernel kernel_mul_mv_id_4_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_4_impl<bfloat, bfloat4, float, float4, N_R0_F>>>;
#endif

template [[host_name("kernel_mul_mv_id_q8_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0>>>;

template [[host_name("kernel_mul_mv_id_q4_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0>>>;
template [[host_name("kernel_mul_mv_id_q4_1_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q4_1, N_R0_Q4_1>>>;
template [[host_name("kernel_mul_mv_id_q5_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_0, N_R0_Q5_0>>>;
template [[host_name("kernel_mul_mv_id_q5_1_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_1, N_R0_Q5_1>>>;

template [[host_name("kernel_mul_mv_id_mxfp4_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>>>;

template [[host_name("kernel_mul_mv_id_q2_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl   <N_R0_Q2_K>>>;
template [[host_name("kernel_mul_mv_id_q3_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q3_K_f32_impl   <N_R0_Q3_K>>>;
template [[host_name("kernel_mul_mv_id_q4_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl   <N_R0_Q4_K>>>;
template [[host_name("kernel_mul_mv_id_q5_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q5_K_f32_impl   <N_R0_Q5_K>>>;
template [[host_name("kernel_mul_mv_id_q6_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q6_K_f32_impl   <N_R0_Q6_K>>>;
template [[host_name("kernel_mul_mv_id_iq1_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_s_f32_impl  <N_R0_IQ1_S>>>;
template [[host_name("kernel_mul_mv_id_iq1_m_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_m_f32_impl  <N_R0_IQ1_M>>>;
template [[host_name("kernel_mul_mv_id_iq2_xxs_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS>>>;
template [[host_name("kernel_mul_mv_id_iq2_xs_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xs_f32_impl <N_R0_IQ2_XS>>>;
template [[host_name("kernel_mul_mv_id_iq3_xxs_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_xxs_f32_impl<N_R0_IQ3_XXS>>>;
template [[host_name("kernel_mul_mv_id_iq3_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_s_f32_impl  <N_R0_IQ3_S>>>;
template [[host_name("kernel_mul_mv_id_iq2_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_s_f32_impl  <N_R0_IQ2_S>>>;
template [[host_name("kernel_mul_mv_id_iq4_nl_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_nl_f32_impl <N_R0_IQ4_NL>>>;
template [[host_name("kernel_mul_mv_id_iq4_xs_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_xs_f32_impl <N_R0_IQ4_XS>>>;

kernel void kernel_pool_2d_max_f32(
        constant    ggml_metal_kargs_pool_2d & args,
        device  const float * src0,
        device        float * dst,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const int idx = gid;
    const int I_HW = args.IH * args.IW;
    const int O_HW = args.OH * args.OW;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / args.OW;
    const int cur_ow = idx % O_HW % args.OW;

    device const float * i_ptr = src0 + nc * I_HW;
    device       float * o_ptr = dst  + nc * O_HW;

    const int start_h = cur_oh * args.s1 - args.p1;
    const int bh = MAX(0,  start_h);
    const int eh = MIN(args.IH, start_h + args.k1);
    const int start_w = cur_ow * args.s0 - args.p0;
    const int bw = MAX(0,  start_w);
    const int ew = MIN(args.IW, start_w + args.k0);

    float res = -INFINITY;

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
            res = MAX(res, i_ptr[i * args.IW + j]);
        }
    }

    o_ptr[cur_oh * args.OW + cur_ow] = res;
}

kernel void kernel_pool_2d_avg_f32(
        constant    ggml_metal_kargs_pool_2d & args,
        device  const float * src0,
        device        float * dst,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const int idx = gid;
    const int I_HW = args.IH * args.IW;
    const int O_HW = args.OH * args.OW;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / args.OW;
    const int cur_ow = idx % O_HW % args.OW;

    device const float * i_ptr = src0 + nc * I_HW;
    device       float * o_ptr = dst  + nc * O_HW;

    const int start_h = cur_oh * args.s1 - args.p1;
    const int bh = MAX(0,  start_h);
    const int eh = MIN(args.IH, start_h + args.k1);
    const int start_w = cur_ow * args.s0 - args.p0;
    const int bw = MAX(0,  start_w);
    const int ew = MIN(args.IW, start_w + args.k0);
    // const float scale = 1. / ((eh - bh) * (ew - bw));
    const float scale = 1. / (args.k0 * args.k1);

    float res = 0;

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
            float cur = i_ptr[i * args.IW + j];
            res += cur * scale;
        }
    }

    o_ptr[cur_oh * args.OW + cur_ow] = res;
}
