#define GGML_COMMON_DECL_CPP
#include "ggml-backend.h"
#include "ggml-common.h"
#include "ggml-ifairy-lut-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#ifndef GGML_FP16_TO_FP32
#    define GGML_FP16_TO_FP32 ggml_fp16_to_fp32
#endif
#ifndef GGML_FP32_TO_BF16
#    define GGML_FP32_TO_BF16 ggml_fp32_to_bf16
#endif
#ifndef GGML_BF16_TO_FP32
#    define GGML_BF16_TO_FP32 ggml_bf16_to_fp32
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <type_traits>

#if defined(_MSC_VER)
#    define GGML_IFAIRY_LUT_NOINLINE __declspec(noinline)
#else
#    define GGML_IFAIRY_LUT_NOINLINE __attribute__((noinline))
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)
#    include <arm_neon.h>
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
#    include <immintrin.h>
#endif

// iFairy LUT V2: 2-weight direct 4-bit index, 16-entry LUT + packed 16-row weight tiles.
static_assert(QK_IFAIRY_GROUPS_PER_BLOCK % 2 == 0, "groups_per_block must be even for unroll-by-2");

static inline size_t ggml_ifairy_checked_mul_size(size_t a, size_t b) {
    GGML_ASSERT(a == 0 || b <= SIZE_MAX / a);
    return a * b;
}

static inline size_t ggml_ifairy_checked_add_size(size_t a, size_t b) {
    GGML_ASSERT(a <= SIZE_MAX - b);
    return a + b;
}

static inline float ggml_ifairy_lut_scale_to_f32(float v) {
    return v;
}

static inline float ggml_ifairy_lut_scale_to_f32(ggml_half v) {
    return GGML_FP16_TO_FP32(v);
}

namespace {
struct ggml_ifairy_tl_buf {
    uint8_t * ptr = nullptr;
    size_t    cap = 0;

    ~ggml_ifairy_tl_buf() {
        if (ptr) {
            ggml_aligned_free(ptr, cap);
        }
    }
};

uint8_t * ggml_ifairy_tl_reserve(ggml_ifairy_tl_buf & tl, size_t bytes) {
    if (tl.cap < bytes) {
        if (tl.ptr) {
            ggml_aligned_free(tl.ptr, tl.cap);
            tl.ptr = nullptr;
            tl.cap = 0;
        }
        tl.ptr = (uint8_t *) ggml_aligned_malloc(bytes);
        if (!tl.ptr) {
            return nullptr;
        }
        tl.cap = bytes;
    }
    return tl.ptr;
}
}  // namespace

#if defined(__ARM_NEON) && defined(__aarch64__)
static inline float32x4_t ggml_ifairy_s16x4_to_f32(int16x4_t v) {
    return vcvtq_f32_s32(vmovl_s16(v));
}

static inline float32x4_t ggml_ifairy_lut_load_scale4_arm(const float * src) {
    return vld1q_f32(src);
}

static inline float32x4_t ggml_ifairy_lut_load_scale4_arm(const ggml_half * src) {
    const uint16x4_t v = vld1_u16((const uint16_t *) src);
    return vcvt_f32_f16(vreinterpret_f16_u16(v));
}

template <typename wtile_type>
static inline void ggml_ifairy_lut_apply_tile_sums_arm(const wtile_type * wt,
                                                       const float32x4_t  v_lr,
                                                       const float32x4_t  v_li,
                                                       const int16x8_t    sum_ac_0,
                                                       const int16x8_t    sum_ac_1,
                                                       const int16x8_t    sum_bc_0,
                                                       const int16x8_t    sum_bc_1,
                                                       const int16x8_t    sum_ad_0,
                                                       const int16x8_t    sum_ad_1,
                                                       const int16x8_t    sum_bd_0,
                                                       const int16x8_t    sum_bd_1,
                                                       float32x4_t &      acc_r0,
                                                       float32x4_t &      acc_r1,
                                                       float32x4_t &      acc_r2,
                                                       float32x4_t &      acc_r3,
                                                       float32x4_t &      acc_i0,
                                                       float32x4_t &      acc_i1,
                                                       float32x4_t &      acc_i2,
                                                       float32x4_t &      acc_i3) {
    {
        const float32x4_t wr    = ggml_ifairy_lut_load_scale4_arm(wt->d_real + 0);
        const float32x4_t wi    = ggml_ifairy_lut_load_scale4_arm(wt->d_imag + 0);
        const float32x4_t lr_wr = vmulq_f32(v_lr, wr);
        const float32x4_t li_wi = vmulq_f32(v_li, wi);
        const float32x4_t lr_wi = vmulq_f32(v_lr, wi);
        const float32x4_t li_wr = vmulq_f32(v_li, wr);

        acc_r0 = vmlaq_f32(acc_r0, ggml_ifairy_s16x4_to_f32(vget_low_s16(sum_ac_0)), lr_wr);
        acc_r0 = vmlaq_f32(acc_r0, ggml_ifairy_s16x4_to_f32(vget_low_s16(sum_bd_0)), li_wi);
        acc_i0 = vmlaq_f32(acc_i0, ggml_ifairy_s16x4_to_f32(vget_low_s16(sum_bc_0)), lr_wi);
        acc_i0 = vmlaq_f32(acc_i0, ggml_ifairy_s16x4_to_f32(vget_low_s16(sum_ad_0)), li_wr);
    }
    {
        const float32x4_t wr    = ggml_ifairy_lut_load_scale4_arm(wt->d_real + 4);
        const float32x4_t wi    = ggml_ifairy_lut_load_scale4_arm(wt->d_imag + 4);
        const float32x4_t lr_wr = vmulq_f32(v_lr, wr);
        const float32x4_t li_wi = vmulq_f32(v_li, wi);
        const float32x4_t lr_wi = vmulq_f32(v_lr, wi);
        const float32x4_t li_wr = vmulq_f32(v_li, wr);

        acc_r1 = vmlaq_f32(acc_r1, ggml_ifairy_s16x4_to_f32(vget_high_s16(sum_ac_0)), lr_wr);
        acc_r1 = vmlaq_f32(acc_r1, ggml_ifairy_s16x4_to_f32(vget_high_s16(sum_bd_0)), li_wi);
        acc_i1 = vmlaq_f32(acc_i1, ggml_ifairy_s16x4_to_f32(vget_high_s16(sum_bc_0)), lr_wi);
        acc_i1 = vmlaq_f32(acc_i1, ggml_ifairy_s16x4_to_f32(vget_high_s16(sum_ad_0)), li_wr);
    }
    {
        const float32x4_t wr    = ggml_ifairy_lut_load_scale4_arm(wt->d_real + 8);
        const float32x4_t wi    = ggml_ifairy_lut_load_scale4_arm(wt->d_imag + 8);
        const float32x4_t lr_wr = vmulq_f32(v_lr, wr);
        const float32x4_t li_wi = vmulq_f32(v_li, wi);
        const float32x4_t lr_wi = vmulq_f32(v_lr, wi);
        const float32x4_t li_wr = vmulq_f32(v_li, wr);

        acc_r2 = vmlaq_f32(acc_r2, ggml_ifairy_s16x4_to_f32(vget_low_s16(sum_ac_1)), lr_wr);
        acc_r2 = vmlaq_f32(acc_r2, ggml_ifairy_s16x4_to_f32(vget_low_s16(sum_bd_1)), li_wi);
        acc_i2 = vmlaq_f32(acc_i2, ggml_ifairy_s16x4_to_f32(vget_low_s16(sum_bc_1)), lr_wi);
        acc_i2 = vmlaq_f32(acc_i2, ggml_ifairy_s16x4_to_f32(vget_low_s16(sum_ad_1)), li_wr);
    }
    {
        const float32x4_t wr    = ggml_ifairy_lut_load_scale4_arm(wt->d_real + 12);
        const float32x4_t wi    = ggml_ifairy_lut_load_scale4_arm(wt->d_imag + 12);
        const float32x4_t lr_wr = vmulq_f32(v_lr, wr);
        const float32x4_t li_wi = vmulq_f32(v_li, wi);
        const float32x4_t lr_wi = vmulq_f32(v_lr, wi);
        const float32x4_t li_wr = vmulq_f32(v_li, wr);

        acc_r3 = vmlaq_f32(acc_r3, ggml_ifairy_s16x4_to_f32(vget_high_s16(sum_ac_1)), lr_wr);
        acc_r3 = vmlaq_f32(acc_r3, ggml_ifairy_s16x4_to_f32(vget_high_s16(sum_bd_1)), li_wi);
        acc_i3 = vmlaq_f32(acc_i3, ggml_ifairy_s16x4_to_f32(vget_high_s16(sum_bc_1)), lr_wi);
        acc_i3 = vmlaq_f32(acc_i3, ggml_ifairy_s16x4_to_f32(vget_high_s16(sum_ad_1)), li_wr);
    }
}

static inline void ggml_ifairy_lut_store_tile_arm(int           tile,
                                                  int           m,
                                                  uint8_t *     dst_col,
                                                  size_t        dst_row_stride,
                                                  bool          pack_bf16,
                                                  float32x4_t   acc_r0,
                                                  float32x4_t   acc_r1,
                                                  float32x4_t   acc_r2,
                                                  float32x4_t   acc_r3,
                                                  float32x4_t   acc_i0,
                                                  float32x4_t   acc_i1,
                                                  float32x4_t   acc_i2,
                                                  float32x4_t   acc_i3) {
    const int rows_left = m - (tile << 4);
    if (rows_left <= 0) {
        return;
    }
    const int rows_in_tile = rows_left >= 16 ? 16 : rows_left;

    alignas(16) float out_r[16];
    alignas(16) float out_i[16];
    vst1q_f32(out_r + 0, acc_r0);
    vst1q_f32(out_r + 4, acc_r1);
    vst1q_f32(out_r + 8, acc_r2);
    vst1q_f32(out_r + 12, acc_r3);
    vst1q_f32(out_i + 0, acc_i0);
    vst1q_f32(out_i + 4, acc_i1);
    vst1q_f32(out_i + 8, acc_i2);
    vst1q_f32(out_i + 12, acc_i3);

    for (int lane = 0; lane < rows_in_tile; ++lane) {
        uint8_t * out_base = dst_col + ((tile << 4) + lane) * dst_row_stride;
        if (pack_bf16) {
            ((ggml_bf16_t *) out_base)[0] = GGML_FP32_TO_BF16(out_r[lane]);
            ((ggml_bf16_t *) out_base)[1] = GGML_FP32_TO_BF16(out_i[lane]);
        } else {
            ((float *) out_base)[0] = out_r[lane];
            ((float *) out_base)[1] = out_i[lane];
        }
    }
}
#endif

#if defined(__AVX2__)
static inline __m256 ggml_ifairy_lut_load_scale8_avx2(const float * src) {
    return _mm256_loadu_ps(src);
}

static inline __m256 ggml_ifairy_lut_load_scale8_avx2(const ggml_half * src) {
#    if defined(__F16C__)
    const __m128i v = _mm_loadu_si128((const __m128i *) src);
    return _mm256_cvtph_ps(v);
#    else
    alignas(32) float tmp[8];
    for (int i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP16_TO_FP32(src[i]);
    }
    return _mm256_load_ps(tmp);
#    endif
}
#endif

static inline int8_t ggml_ifairy_lut_sat_s8(int v) {
    if (v > INT8_MAX) {
        return INT8_MAX;
    }
    if (v < INT8_MIN) {
        return INT8_MIN;
    }
    return (int8_t) v;
}

static inline int ggml_ifairy_u8_to_s8_int(uint8_t v) {
    return v < 128 ? (int) v : (int) v - 256;
}

static inline void ggml_ifairy_lut_fill_group_lut16(int xr0, int xi0, int xr1, int xi1, int8_t * tbl) {
#if defined(__AVX2__)
    const __m256i c0_ac_bc = _mm256_setr_epi8(-1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0,  // lower = c0_r
                                              0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1   // upper = c0_i
    );
    const __m256i c1_ac_bc = _mm256_setr_epi8(-1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,  // lower = c1_r
                                              0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1   // upper = c1_i
    );

    const __m256i c0_bd_ad = _mm256_setr_epi8(0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1,  // lower = -c0_i
                                              -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0   // upper = c0_r
    );
    const __m256i c1_bd_ad = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1,  // lower = -c1_i
                                              -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0   // upper = c1_r
    );
#endif

    const int r0 = xr0;
    const int r1 = xr1;
    const int i0 = -xi0;
    const int i1 = -xi1;

#if defined(__AVX2__)
    const __m256i v_r0 = _mm256_set1_epi8((char) r0);
    const __m256i v_r1 = _mm256_set1_epi8((char) r1);
    const __m256i v_i0 = _mm256_set1_epi8((char) i0);
    const __m256i v_i1 = _mm256_set1_epi8((char) i1);

    __m256i tbl_ac_bc = _mm256_adds_epi8(_mm256_sign_epi8(v_r0, c0_ac_bc), _mm256_sign_epi8(v_r1, c1_ac_bc));
    __m256i tbl_bd_ad = _mm256_adds_epi8(_mm256_sign_epi8(v_i0, c0_bd_ad), _mm256_sign_epi8(v_i1, c1_bd_ad));

    _mm_storeu_si128((__m128i *) (tbl + 0), _mm256_castsi256_si128(tbl_ac_bc));
    _mm_storeu_si128((__m128i *) (tbl + 16), _mm256_castsi256_si128(tbl_bd_ad));
    _mm_storeu_si128((__m128i *) (tbl + 32), _mm256_extracti128_si256(tbl_ac_bc, 1));
    _mm_storeu_si128((__m128i *) (tbl + 48), _mm256_extracti128_si256(tbl_bd_ad, 1));
#elif defined(__ARM_NEON) && defined(__aarch64__)
    const int8x16_t c0_ac = {
        -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0,
    };
    const int8x16_t c1_ac = {
        -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    const int8x16_t c0_bd = {
        0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1,
    };
    const int8x16_t c1_bd = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1,
    };
    const int8x16_t c0_bc = {
        0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1,
    };
    const int8x16_t c1_bc = {
        0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1,
    };
    const int8x16_t c0_ad = {
        -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0,
    };
    const int8x16_t c1_ad = {
        -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    const int8x16_t v_r0 = vdupq_n_s8((int8_t) r0);
    const int8x16_t v_r1 = vdupq_n_s8((int8_t) r1);
    const int8x16_t v_i0 = vdupq_n_s8((int8_t) i0);
    const int8x16_t v_i1 = vdupq_n_s8((int8_t) i1);

    const int8x16_t ac_tbl = vqaddq_s8(vmulq_s8(v_r0, c0_ac), vmulq_s8(v_r1, c1_ac));
    const int8x16_t bd_tbl = vqaddq_s8(vmulq_s8(v_i0, c0_bd), vmulq_s8(v_i1, c1_bd));
    const int8x16_t bc_tbl = vqaddq_s8(vmulq_s8(v_r0, c0_bc), vmulq_s8(v_r1, c1_bc));
    const int8x16_t ad_tbl = vqaddq_s8(vmulq_s8(v_i0, c0_ad), vmulq_s8(v_i1, c1_ad));

    vst1q_s8(tbl + 0, ac_tbl);
    vst1q_s8(tbl + 16, bd_tbl);
    vst1q_s8(tbl + 32, bc_tbl);
    vst1q_s8(tbl + 48, ad_tbl);
#else
    alignas(16) int8_t ac_tbl[16] = {
        ggml_ifairy_lut_sat_s8(-r0 - r1),
        ggml_ifairy_lut_sat_s8(+r0 - r1),
        ggml_ifairy_lut_sat_s8(-r1),
        ggml_ifairy_lut_sat_s8(-r1),
        ggml_ifairy_lut_sat_s8(-r0 + r1),
        ggml_ifairy_lut_sat_s8(+r0 + r1),
        ggml_ifairy_lut_sat_s8(+r1),
        ggml_ifairy_lut_sat_s8(+r1),
        ggml_ifairy_lut_sat_s8(-r0),
        ggml_ifairy_lut_sat_s8(+r0),
        0,
        0,
        ggml_ifairy_lut_sat_s8(-r0),
        ggml_ifairy_lut_sat_s8(+r0),
        0,
        0,
    };
    alignas(16) int8_t bd_tbl[16] = {
        0,
        0,
        ggml_ifairy_lut_sat_s8(+i0),
        ggml_ifairy_lut_sat_s8(-i0),
        0,
        0,
        ggml_ifairy_lut_sat_s8(+i0),
        ggml_ifairy_lut_sat_s8(-i0),
        ggml_ifairy_lut_sat_s8(+i1),
        ggml_ifairy_lut_sat_s8(+i1),
        ggml_ifairy_lut_sat_s8(+i0 + i1),
        ggml_ifairy_lut_sat_s8(-i0 + i1),
        ggml_ifairy_lut_sat_s8(-i1),
        ggml_ifairy_lut_sat_s8(-i1),
        ggml_ifairy_lut_sat_s8(+i0 - i1),
        ggml_ifairy_lut_sat_s8(-i0 - i1),
    };
    alignas(16) int8_t bc_tbl[16] = {
        0,
        0,
        ggml_ifairy_lut_sat_s8(-r0),
        ggml_ifairy_lut_sat_s8(+r0),
        0,
        0,
        ggml_ifairy_lut_sat_s8(-r0),
        ggml_ifairy_lut_sat_s8(+r0),
        ggml_ifairy_lut_sat_s8(-r1),
        ggml_ifairy_lut_sat_s8(-r1),
        ggml_ifairy_lut_sat_s8(-r0 - r1),
        ggml_ifairy_lut_sat_s8(+r0 - r1),
        ggml_ifairy_lut_sat_s8(+r1),
        ggml_ifairy_lut_sat_s8(+r1),
        ggml_ifairy_lut_sat_s8(-r0 + r1),
        ggml_ifairy_lut_sat_s8(+r0 + r1),
    };
    alignas(16) int8_t ad_tbl[16] = {
        ggml_ifairy_lut_sat_s8(-i0 - i1),
        ggml_ifairy_lut_sat_s8(+i0 - i1),
        ggml_ifairy_lut_sat_s8(-i1),
        ggml_ifairy_lut_sat_s8(-i1),
        ggml_ifairy_lut_sat_s8(-i0 + i1),
        ggml_ifairy_lut_sat_s8(+i0 + i1),
        ggml_ifairy_lut_sat_s8(+i1),
        ggml_ifairy_lut_sat_s8(+i1),
        ggml_ifairy_lut_sat_s8(-i0),
        ggml_ifairy_lut_sat_s8(+i0),
        0,
        0,
        ggml_ifairy_lut_sat_s8(-i0),
        ggml_ifairy_lut_sat_s8(+i0),
        0,
        0,
    };
    memcpy(tbl + 0, ac_tbl, 16);
    memcpy(tbl + 16, bd_tbl, 16);
    memcpy(tbl + 32, bc_tbl, 16);
    memcpy(tbl + 48, ad_tbl, 16);
#endif
}

// ===========================================================================
// 终极优化 1：向量化 Preprocess（拯救 TG256 首Token延迟）
// 修复了复数点乘的 LUT 构建逻辑，彻底解耦实部与虚部！
// ===========================================================================
static void ggml_ifairy_lut_preprocess_lut16_one(const block_ifairy_q16 * act_blocks,
                                                 int64_t                  blocks,
                                                 int64_t                  groups_per_block,
                                                 float *                  scales_out,
                                                 int8_t *                 lut_out,
                                                 int64_t                  g0,
                                                 int64_t                  gstep) {
    if (g0 == 0) {
        for (int64_t blk = 0; blk < blocks; ++blk) {
            scales_out[blk * 2 + 0] = GGML_FP16_TO_FP32(act_blocks[blk].d_real);
            scales_out[blk * 2 + 1] = GGML_FP16_TO_FP32(act_blocks[blk].d_imag);
        }
    }

    const int64_t groups = blocks * groups_per_block;
    for (int64_t g = g0; g < groups; g += gstep) {
        const int64_t blk      = g / groups_per_block;
        const int64_t base_off = (g % groups_per_block) * 2;

        int xr0 = ggml_ifairy_u8_to_s8_int(act_blocks[blk].x_real[base_off + 0]);
        int xi0 = ggml_ifairy_u8_to_s8_int(act_blocks[blk].x_imag[base_off + 0]);
        int xr1 = ggml_ifairy_u8_to_s8_int(act_blocks[blk].x_real[base_off + 1]);
        int xi1 = ggml_ifairy_u8_to_s8_int(act_blocks[blk].x_imag[base_off + 1]);

        int8_t * tbl = lut_out + (size_t) g * k_ifairy_lut_group_bytes;

        ggml_ifairy_lut_fill_group_lut16(xr0, xi0, xr1, xi1, tbl);
    }
}

static void ggml_ifairy64_lut_preprocess_lut16_one(const block_ifairy_q16 * act_blocks,
                                                   int64_t                  weight_blocks,
                                                   float *                  scales_out,
                                                   int8_t *                 lut_out,
                                                   int64_t                  g0,
                                                   int64_t                  gstep) {
    if (g0 == 0) {
        for (int64_t blk = 0; blk < weight_blocks; ++blk) {
            const block_ifairy_q16 & act_blk = act_blocks[blk / 4];
            scales_out[blk * 2 + 0]          = GGML_FP16_TO_FP32(act_blk.d_real);
            scales_out[blk * 2 + 1]          = GGML_FP16_TO_FP32(act_blk.d_imag);
        }
    }

    const int64_t groups = weight_blocks * QK_IFAIRY64_GROUPS_PER_BLOCK;
    for (int64_t g = g0; g < groups; g += gstep) {
        const int64_t blk64      = g / QK_IFAIRY64_GROUPS_PER_BLOCK;
        const int64_t local_group = g - blk64 * QK_IFAIRY64_GROUPS_PER_BLOCK;
        const int64_t act_block  = blk64 / 4;
        const int64_t subblock   = blk64 & 0x3;
        const int64_t base_off   = subblock * QK_IFAIRY64 + local_group * 2;

        int xr0 = ggml_ifairy_u8_to_s8_int(act_blocks[act_block].x_real[base_off + 0]);
        int xi0 = ggml_ifairy_u8_to_s8_int(act_blocks[act_block].x_imag[base_off + 0]);
        int xr1 = ggml_ifairy_u8_to_s8_int(act_blocks[act_block].x_real[base_off + 1]);
        int xi1 = ggml_ifairy_u8_to_s8_int(act_blocks[act_block].x_imag[base_off + 1]);

        int8_t * tbl = lut_out + (size_t) g * k_ifairy_lut_group_bytes;
        ggml_ifairy_lut_fill_group_lut16(xr0, xi0, xr1, xi1, tbl);
    }
}

static void ggml_ifairy_lut_preprocess_lut16(int          m,
                                             int          k,
                                             int          n,
                                             const void * act,
                                             size_t       act_stride,
                                             void *       lut_scales,
                                             void *       lut_buf,
                                             int          ith,
                                             int          nth) {
    (void) m;
    if (!act || !lut_scales || !lut_buf) {
        return;
    }

    nth = std::max(nth, 1);
    if (ith < 0 || ith >= nth) {
        return;
    }

    const int64_t K                = k;
    const int64_t blocks           = K / QK_IFAIRY;
    const int64_t groups_per_block = QK_IFAIRY_GROUPS_PER_BLOCK;
    const int64_t groups           = blocks * groups_per_block;
    const bool    shard_by_col     = n >= nth;

    const int     col_start = shard_by_col ? ith : 0;
    const int     col_step  = shard_by_col ? nth : 1;
    const int64_t g0        = shard_by_col ? 0 : ith;
    const int64_t gstep     = shard_by_col ? 1 : (int64_t) nth;

    for (int col = col_start; col < n; col += col_step) {
        const uint8_t *          act_col_bytes = (const uint8_t *) act + (size_t) col * act_stride;
        const block_ifairy_q16 * act_blocks    = (const block_ifairy_q16 *) act_col_bytes;
        float *                  scales_out    = (float *) lut_scales + (size_t) col * (size_t) blocks * 2;
        int8_t * lut_out = (int8_t *) ((uint8_t *) lut_buf + (size_t) col * (size_t) groups * k_ifairy_lut_group_bytes);

        ggml_ifairy_lut_preprocess_lut16_one(act_blocks, blocks, groups_per_block, scales_out, lut_out, g0, gstep);
    }
}

static void ggml_ifairy64_lut_preprocess_lut16(int          m,
                                               int          k,
                                               int          n,
                                               const void * act,
                                               size_t       act_stride,
                                               void *       lut_scales,
                                               void *       lut_buf,
                                               int          ith,
                                               int          nth) {
    (void) m;
    if (!act || !lut_scales || !lut_buf) {
        return;
    }

    nth = std::max(nth, 1);
    if (ith < 0 || ith >= nth) {
        return;
    }

    const int64_t K             = k;
    const int64_t weight_blocks = K / QK_IFAIRY64;
    const int64_t groups        = weight_blocks * QK_IFAIRY64_GROUPS_PER_BLOCK;
    const bool    shard_by_col  = n >= nth;

    const int     col_start = shard_by_col ? ith : 0;
    const int     col_step  = shard_by_col ? nth : 1;
    const int64_t g0        = shard_by_col ? 0 : ith;
    const int64_t gstep     = shard_by_col ? 1 : (int64_t) nth;

    for (int col = col_start; col < n; col += col_step) {
        const uint8_t *          act_col_bytes = (const uint8_t *) act + (size_t) col * act_stride;
        const block_ifairy_q16 * act_blocks    = (const block_ifairy_q16 *) act_col_bytes;
        float *                  scales_out    = (float *) lut_scales + (size_t) col * (size_t) weight_blocks * 2u;
        int8_t * lut_out = (int8_t *) ((uint8_t *) lut_buf + (size_t) col * (size_t) groups * k_ifairy_lut_group_bytes);

        ggml_ifairy64_lut_preprocess_lut16_one(act_blocks, weight_blocks, scales_out, lut_out, g0, gstep);
    }
}

void ggml_ifairy_lut_preprocess_ex_lut16(int          m,
                                         int          k,
                                         int          n,
                                         const void * act,
                                         size_t       act_stride,
                                         void *       lut_scales,
                                         void *       lut_buf,
                                         int          ith,
                                         int          nth) {
    ggml_ifairy_lut_preprocess_lut16(m, k, n, act, act_stride, lut_scales, lut_buf, ith, nth);
}

void ggml_ifairy_lut_preprocess_ex_lut_c(int          m,
                                         int          k,
                                         int          n,
                                         const void * act,
                                         size_t       act_stride,
                                         void *       lut_scales,
                                         void *       lut_buf,
                                         int          ith,
                                         int          nth) {
    ggml_ifairy_lut_preprocess_lut16(m, k, n, act, act_stride, lut_scales, lut_buf, ith, nth);
}

void ggml_ifairy64_lut_preprocess_ex_lut16(int          m,
                                           int          k,
                                           int          n,
                                           const void * act,
                                           size_t       act_stride,
                                           void *       lut_scales,
                                           void *       lut_buf,
                                           int          ith,
                                           int          nth) {
    ggml_ifairy64_lut_preprocess_lut16(m, k, n, act, act_stride, lut_scales, lut_buf, ith, nth);
}

void ggml_ifairy64_lut_preprocess_ex_lut_c(int          m,
                                           int          k,
                                           int          n,
                                           const void * act,
                                           size_t       act_stride,
                                           void *       lut_scales,
                                           void *       lut_buf,
                                           int          ith,
                                           int          nth) {
    ggml_ifairy64_lut_preprocess_lut16(m, k, n, act, act_stride, lut_scales, lut_buf, ith, nth);
}

// 纯 4-bit 标量解码：无掩码，无标志位，直接查表命中真理
static inline void ggml_ifairy_lut_decode_lane_scalar(const uint8_t  code,
                                                      const int8_t * tbl,
                                                      int8_t &       out0,
                                                      int8_t &       out1,
                                                      int8_t &       out2,
                                                      int8_t &       out3) {
    const uint8_t idx = code & 0x0fu;
    out0              = tbl[0 * 16 + idx];
    out1              = tbl[2 * 16 + idx];
    out2              = tbl[3 * 16 + idx];
    out3              = tbl[1 * 16 + idx];
}

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX2__)
static inline __m512 ggml_ifairy_lut_load_scale16_avx512(const float * src) {
    return _mm512_load_ps(src);
}

static inline __m512 ggml_ifairy_lut_load_scale16_avx512(const ggml_half * src) {
    return _mm512_cvtph_ps(_mm256_load_si256((const __m256i *) src));
}

static inline __m512i ggml_ifairy_lut_fp32_to_bf16_u32_avx512(const __m512 v) {
    const __m512i x       = _mm512_castps_si512(v);
    const __m512i lsb     = _mm512_and_si512(_mm512_srli_epi32(x, 16), _mm512_set1_epi32(1));
    const __m512i rounded = _mm512_add_epi32(x, _mm512_add_epi32(_mm512_set1_epi32(0x7fff), lsb));
    const __m512i bf16    = _mm512_srli_epi32(rounded, 16);

    const __m512i   abs_x  = _mm512_and_si512(x, _mm512_set1_epi32(0x7fffffff));
    const __mmask16 is_nan = _mm512_cmpgt_epi32_mask(abs_x, _mm512_set1_epi32(0x7f800000));
    const __m512i   qnan   = _mm512_or_si512(_mm512_srli_epi32(x, 16), _mm512_set1_epi32(64));
    return _mm512_mask_blend_epi32(is_nan, bf16, qnan);
}

template <int lane>
static inline __m128i ggml_ifairy_lut_extract_i32x4_avx512(const __m512i v) {
    if constexpr (lane == 0) {
        return _mm512_castsi512_si128(v);
    } else {
        return _mm512_extracti32x4_epi32(v, lane);
    }
}

template <int lane>
static inline __m512 ggml_ifairy_lut_channel_sums_to_f32_avx512(const __m512i sum_lo, const __m512i sum_hi) {
    const __m128i lo  = ggml_ifairy_lut_extract_i32x4_avx512<lane>(sum_lo);
    const __m128i hi  = ggml_ifairy_lut_extract_i32x4_avx512<lane>(sum_hi);
    const __m256i s16 = _mm256_set_m128i(hi, lo);
    return _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(s16));
}

template <typename wtile_type>
static inline void ggml_ifairy_lut_apply_tile_sums_avx512(const wtile_type * wt,
                                                          const __m512i      sum_lo,
                                                          const __m512i      sum_hi,
                                                          const __m512       v_lr,
                                                          const __m512       v_li,
                                                          __m512 &           acc_r,
                                                          __m512 &           acc_i) {
    const __m512 v_ac = ggml_ifairy_lut_channel_sums_to_f32_avx512<0>(sum_lo, sum_hi);
    const __m512 v_bd = ggml_ifairy_lut_channel_sums_to_f32_avx512<1>(sum_lo, sum_hi);
    const __m512 v_bc = ggml_ifairy_lut_channel_sums_to_f32_avx512<2>(sum_lo, sum_hi);
    const __m512 v_ad = ggml_ifairy_lut_channel_sums_to_f32_avx512<3>(sum_lo, sum_hi);

    const __m512 wr = ggml_ifairy_lut_load_scale16_avx512(wt->d_real);
    const __m512 wi = ggml_ifairy_lut_load_scale16_avx512(wt->d_imag);

#    ifdef __FMA__
    acc_r = _mm512_fmadd_ps(v_ac, _mm512_mul_ps(v_lr, wr), acc_r);
    acc_r = _mm512_fmadd_ps(v_bd, _mm512_mul_ps(v_li, wi), acc_r);
    acc_i = _mm512_fmadd_ps(v_bc, _mm512_mul_ps(v_lr, wi), acc_i);
    acc_i = _mm512_fmadd_ps(v_ad, _mm512_mul_ps(v_li, wr), acc_i);
#    else
    const __m512 lr_wr = _mm512_mul_ps(v_lr, wr);
    const __m512 li_wi = _mm512_mul_ps(v_li, wi);
    const __m512 lr_wi = _mm512_mul_ps(v_lr, wi);
    const __m512 li_wr = _mm512_mul_ps(v_li, wr);

    acc_r = _mm512_add_ps(acc_r, _mm512_mul_ps(v_ac, lr_wr));
    acc_r = _mm512_add_ps(acc_r, _mm512_mul_ps(v_bd, li_wi));
    acc_i = _mm512_add_ps(acc_i, _mm512_mul_ps(v_bc, lr_wi));
    acc_i = _mm512_add_ps(acc_i, _mm512_mul_ps(v_ad, li_wr));
#    endif
}

static inline void ggml_ifairy_lut_store_tile_avx512(int            tile,
                                                     int            m,
                                                     uint8_t *      dst_col,
                                                     size_t         dst_row_stride,
                                                     bool           pack_bf16,
                                                     const __m512 & acc_r,
                                                     const __m512 & acc_i) {
    const int base_row = tile << 4;
    if (pack_bf16 && dst_row_stride == sizeof(float) && base_row + 15 < m) {
        const __m512i out_r = ggml_ifairy_lut_fp32_to_bf16_u32_avx512(acc_r);
        const __m512i out_i = ggml_ifairy_lut_fp32_to_bf16_u32_avx512(acc_i);
        const __m512i out   = _mm512_or_si512(out_r, _mm512_slli_epi32(out_i, 16));

        uint8_t * out_base = dst_col + (size_t) base_row * sizeof(float);
        _mm512_storeu_si512((void *) out_base, out);
        return;
    }

    alignas(64) float out_r[16];
    alignas(64) float out_i[16];
    _mm512_store_ps(out_r, acc_r);
    _mm512_store_ps(out_i, acc_i);

    for (int lane = 0; lane < 16; ++lane) {
        const int row = base_row + lane;
        if (row >= m) {
            break;
        }

        uint8_t * out_base = dst_col + (size_t) row * dst_row_stride;
        if (pack_bf16) {
            ((ggml_bf16_t *) out_base)[0] = GGML_FP32_TO_BF16(out_r[lane]);
            ((ggml_bf16_t *) out_base)[1] = GGML_FP32_TO_BF16(out_i[lane]);
        } else {
            ((float *) out_base)[0] = out_r[lane];
            ((float *) out_base)[1] = out_i[lane];
        }
    }
}

static inline void ggml_ifairy_lut_accumulate_indexes_avx512(const __m512i   lut_0,
                                                             const __m512i   lut_1,
                                                             const __m128i   packed_128,
                                                             const __m512i & one,
                                                             const __m512i & mask_idx,
                                                             __m512i &       sum_lo,
                                                             __m512i &       sum_hi) {
    const __m512i packed = _mm512_broadcast_i32x4(packed_128);
    const __m512i idx_lo = _mm512_and_si512(packed, mask_idx);
    const __m512i idx_hi = _mm512_and_si512(_mm512_srli_epi16(packed, 4), mask_idx);

    const __m512i out_0 = _mm512_shuffle_epi8(lut_0, idx_lo);
    const __m512i out_1 = _mm512_shuffle_epi8(lut_1, idx_hi);

    const __m512i lo = _mm512_unpacklo_epi8(out_0, out_1);
    const __m512i hi = _mm512_unpackhi_epi8(out_0, out_1);
    sum_lo           = _mm512_add_epi16(sum_lo, _mm512_maddubs_epi16(one, lo));
    sum_hi           = _mm512_add_epi16(sum_hi, _mm512_maddubs_epi16(one, hi));
}

template <typename wtile_type, int num_bytes>
static inline void ggml_ifairy_lut_accumulate_single_tile_avx512(const wtile_type * wt,
                                                                 const int8_t *     lut_blk,
                                                                 const __m512i &    one,
                                                                 const __m512i &    mask_idx,
                                                                 const __m512 &     v_lr,
                                                                 const __m512 &     v_li,
                                                                 __m512 &           acc_r,
                                                                 __m512 &           acc_i) {
    __m512i sum_lo = _mm512_setzero_si512();
    __m512i sum_hi = _mm512_setzero_si512();

#    if defined(__GNUC__) || defined(__clang__)
#        pragma GCC unroll 2
#    endif
    for (int byte_idx = 0; byte_idx < num_bytes; ++byte_idx) {
        const int8_t * lut_base = lut_blk + (size_t) byte_idx * 2u * (size_t) k_ifairy_lut_group_bytes;
        const __m512i lut_0     = _mm512_loadu_si512((const void *) (lut_base + 0));
        const __m512i lut_1     = _mm512_loadu_si512((const void *) (lut_base + 64));
        const __m128i packed    = _mm_load_si128((const __m128i *) &wt->qs[byte_idx]);

        ggml_ifairy_lut_accumulate_indexes_avx512(lut_0, lut_1, packed, one, mask_idx, sum_lo, sum_hi);
    }

    ggml_ifairy_lut_apply_tile_sums_avx512(wt, sum_lo, sum_hi, v_lr, v_li, acc_r, acc_i);
}

template <typename wtile_type, int num_bytes>
static inline void ggml_ifairy_lut_accumulate_tile_pair_avx512(const wtile_type * wt0,
                                                               const wtile_type * wt1,
                                                               const int8_t *     lut_blk,
                                                               const __m512i &    one,
                                                               const __m512i &    mask_idx,
                                                               const __m512 &     v_lr,
                                                               const __m512 &     v_li,
                                                               __m512 &           acc0_r,
                                                               __m512 &           acc0_i,
                                                               __m512 &           acc1_r,
                                                               __m512 &           acc1_i) {
    __m512i sum0_lo = _mm512_setzero_si512();
    __m512i sum0_hi = _mm512_setzero_si512();
    __m512i sum1_lo = _mm512_setzero_si512();
    __m512i sum1_hi = _mm512_setzero_si512();

#    if defined(__GNUC__) || defined(__clang__)
#        pragma GCC unroll 2
#    endif
    for (int byte_idx = 0; byte_idx < num_bytes; ++byte_idx) {
        const int8_t * lut_base = lut_blk + (size_t) byte_idx * 2u * (size_t) k_ifairy_lut_group_bytes;
        const __m512i lut_0     = _mm512_loadu_si512((const void *) (lut_base + 0));
        const __m512i lut_1     = _mm512_loadu_si512((const void *) (lut_base + 64));
        const __m128i packed0   = _mm_load_si128((const __m128i *) &wt0->qs[byte_idx]);
        const __m128i packed1   = _mm_load_si128((const __m128i *) &wt1->qs[byte_idx]);

        ggml_ifairy_lut_accumulate_indexes_avx512(lut_0, lut_1, packed0, one, mask_idx, sum0_lo, sum0_hi);
        ggml_ifairy_lut_accumulate_indexes_avx512(lut_0, lut_1, packed1, one, mask_idx, sum1_lo, sum1_hi);
    }

    ggml_ifairy_lut_apply_tile_sums_avx512(wt0, sum0_lo, sum0_hi, v_lr, v_li, acc0_r, acc0_i);
    ggml_ifairy_lut_apply_tile_sums_avx512(wt1, sum1_lo, sum1_hi, v_lr, v_li, acc1_r, acc1_i);
}

#    if defined(__AVX512VNNI__)
template <int lane>
static inline __m512 ggml_ifairy_lut_channel_sums_i32_avx512(const __m512i sum_03,
                                                             const __m512i sum_47,
                                                             const __m512i sum_8b,
                                                             const __m512i sum_cf) {
    __m512i s = _mm512_castsi128_si512(ggml_ifairy_lut_extract_i32x4_avx512<lane>(sum_03));
    s         = _mm512_inserti32x4(s, ggml_ifairy_lut_extract_i32x4_avx512<lane>(sum_47), 1);
    s         = _mm512_inserti32x4(s, ggml_ifairy_lut_extract_i32x4_avx512<lane>(sum_8b), 2);
    s         = _mm512_inserti32x4(s, ggml_ifairy_lut_extract_i32x4_avx512<lane>(sum_cf), 3);
    return _mm512_cvtepi32_ps(s);
}

template <typename wtile_type>
static inline void ggml_ifairy_lut_apply_tile_sums_i32_avx512(const wtile_type * wt,
                                                              const __m512i      sum_03,
                                                              const __m512i      sum_47,
                                                              const __m512i      sum_8b,
                                                              const __m512i      sum_cf,
                                                              const __m512       v_lr,
                                                              const __m512       v_li,
                                                              __m512 &           acc_r,
                                                              __m512 &           acc_i) {
    const __m512 v_ac = ggml_ifairy_lut_channel_sums_i32_avx512<0>(sum_03, sum_47, sum_8b, sum_cf);
    const __m512 v_bd = ggml_ifairy_lut_channel_sums_i32_avx512<1>(sum_03, sum_47, sum_8b, sum_cf);
    const __m512 v_bc = ggml_ifairy_lut_channel_sums_i32_avx512<2>(sum_03, sum_47, sum_8b, sum_cf);
    const __m512 v_ad = ggml_ifairy_lut_channel_sums_i32_avx512<3>(sum_03, sum_47, sum_8b, sum_cf);

    const __m512 wr = ggml_ifairy_lut_load_scale16_avx512(wt->d_real);
    const __m512 wi = ggml_ifairy_lut_load_scale16_avx512(wt->d_imag);

#        ifdef __FMA__
    acc_r = _mm512_fmadd_ps(v_ac, _mm512_mul_ps(v_lr, wr), acc_r);
    acc_r = _mm512_fmadd_ps(v_bd, _mm512_mul_ps(v_li, wi), acc_r);
    acc_i = _mm512_fmadd_ps(v_bc, _mm512_mul_ps(v_lr, wi), acc_i);
    acc_i = _mm512_fmadd_ps(v_ad, _mm512_mul_ps(v_li, wr), acc_i);
#        else
    const __m512 lr_wr = _mm512_mul_ps(v_lr, wr);
    const __m512 li_wi = _mm512_mul_ps(v_li, wi);
    const __m512 lr_wi = _mm512_mul_ps(v_lr, wi);
    const __m512 li_wr = _mm512_mul_ps(v_li, wr);

    acc_r = _mm512_add_ps(acc_r, _mm512_mul_ps(v_ac, lr_wr));
    acc_r = _mm512_add_ps(acc_r, _mm512_mul_ps(v_bd, li_wi));
    acc_i = _mm512_add_ps(acc_i, _mm512_mul_ps(v_bc, lr_wi));
    acc_i = _mm512_add_ps(acc_i, _mm512_mul_ps(v_ad, li_wr));
#        endif
}

static inline void ggml_ifairy_lut_accumulate_indexes_pair_vnni_avx512(const __m512i   lut_0a,
                                                                       const __m512i   lut_1a,
                                                                       const __m128i   packed_a_128,
                                                                       const __m512i   lut_0b,
                                                                       const __m512i   lut_1b,
                                                                       const __m128i   packed_b_128,
                                                                       const __m512i & one,
                                                                       const __m512i & mask_idx,
                                                                       __m512i &       sum_03,
                                                                       __m512i &       sum_47,
                                                                       __m512i &       sum_8b,
                                                                       __m512i &       sum_cf) {
    const __m512i packed_a = _mm512_broadcast_i32x4(packed_a_128);
    const __m512i idx_a_lo = _mm512_and_si512(packed_a, mask_idx);
    const __m512i idx_a_hi = _mm512_and_si512(_mm512_srli_epi16(packed_a, 4), mask_idx);

    const __m512i packed_b = _mm512_broadcast_i32x4(packed_b_128);
    const __m512i idx_b_lo = _mm512_and_si512(packed_b, mask_idx);
    const __m512i idx_b_hi = _mm512_and_si512(_mm512_srli_epi16(packed_b, 4), mask_idx);

    const __m512i out_0a = _mm512_shuffle_epi8(lut_0a, idx_a_lo);
    const __m512i out_1a = _mm512_shuffle_epi8(lut_1a, idx_a_hi);
    const __m512i out_0b = _mm512_shuffle_epi8(lut_0b, idx_b_lo);
    const __m512i out_1b = _mm512_shuffle_epi8(lut_1b, idx_b_hi);

    const __m512i lo_a = _mm512_unpacklo_epi8(out_0a, out_1a);
    const __m512i hi_a = _mm512_unpackhi_epi8(out_0a, out_1a);
    const __m512i lo_b = _mm512_unpacklo_epi8(out_0b, out_1b);
    const __m512i hi_b = _mm512_unpackhi_epi8(out_0b, out_1b);

    sum_03 = _mm512_dpbusd_epi32(sum_03, one, _mm512_unpacklo_epi16(lo_a, lo_b));
    sum_47 = _mm512_dpbusd_epi32(sum_47, one, _mm512_unpackhi_epi16(lo_a, lo_b));
    sum_8b = _mm512_dpbusd_epi32(sum_8b, one, _mm512_unpacklo_epi16(hi_a, hi_b));
    sum_cf = _mm512_dpbusd_epi32(sum_cf, one, _mm512_unpackhi_epi16(hi_a, hi_b));
}

template <typename wtile_type, int num_bytes>
static inline void ggml_ifairy_lut_accumulate_single_tile_vnni_avx512(const wtile_type * wt,
                                                                      const int8_t *     lut_blk,
                                                                      const __m512i &    one,
                                                                      const __m512i &    mask_idx,
                                                                      const __m512 &     v_lr,
                                                                      const __m512 &     v_li,
                                                                      __m512 &           acc_r,
                                                                      __m512 &           acc_i) {
    static_assert(num_bytes % 2 == 0, "VNNI path consumes two byte_idx entries per iteration");

    __m512i sum_03 = _mm512_setzero_si512();
    __m512i sum_47 = _mm512_setzero_si512();
    __m512i sum_8b = _mm512_setzero_si512();
    __m512i sum_cf = _mm512_setzero_si512();

#        if defined(__GNUC__) || defined(__clang__)
#            pragma GCC unroll 2
#        endif
    for (int byte_idx = 0; byte_idx < num_bytes; byte_idx += 2) {
        const int8_t * lut_base_a = lut_blk + (size_t) byte_idx * 2u * (size_t) k_ifairy_lut_group_bytes;
        const int8_t * lut_base_b = lut_base_a + 2u * (size_t) k_ifairy_lut_group_bytes;

        ggml_ifairy_lut_accumulate_indexes_pair_vnni_avx512(
            _mm512_loadu_si512((const void *) (lut_base_a + 0)), _mm512_loadu_si512((const void *) (lut_base_a + 64)),
            _mm_load_si128((const __m128i *) &wt->qs[byte_idx]),
            _mm512_loadu_si512((const void *) (lut_base_b + 0)), _mm512_loadu_si512((const void *) (lut_base_b + 64)),
            _mm_load_si128((const __m128i *) &wt->qs[byte_idx + 1]), one, mask_idx, sum_03, sum_47, sum_8b, sum_cf);
    }

    ggml_ifairy_lut_apply_tile_sums_i32_avx512(wt, sum_03, sum_47, sum_8b, sum_cf, v_lr, v_li, acc_r, acc_i);
}

template <typename wtile_type, int num_bytes>
static inline void ggml_ifairy_lut_accumulate_tile_pair_vnni_avx512(const wtile_type * wt0,
                                                                    const wtile_type * wt1,
                                                                    const int8_t *     lut_blk,
                                                                    const __m512i &    one,
                                                                    const __m512i &    mask_idx,
                                                                    const __m512 &     v_lr,
                                                                    const __m512 &     v_li,
                                                                    __m512 &           acc0_r,
                                                                    __m512 &           acc0_i,
                                                                    __m512 &           acc1_r,
                                                                    __m512 &           acc1_i) {
    static_assert(num_bytes % 2 == 0, "VNNI path consumes two byte_idx entries per iteration");

    __m512i sum0_03 = _mm512_setzero_si512();
    __m512i sum0_47 = _mm512_setzero_si512();
    __m512i sum0_8b = _mm512_setzero_si512();
    __m512i sum0_cf = _mm512_setzero_si512();
    __m512i sum1_03 = _mm512_setzero_si512();
    __m512i sum1_47 = _mm512_setzero_si512();
    __m512i sum1_8b = _mm512_setzero_si512();
    __m512i sum1_cf = _mm512_setzero_si512();

#        if defined(__GNUC__) || defined(__clang__)
#            pragma GCC unroll 2
#        endif
    for (int byte_idx = 0; byte_idx < num_bytes; byte_idx += 2) {
        const int8_t * lut_base_a = lut_blk + (size_t) byte_idx * 2u * (size_t) k_ifairy_lut_group_bytes;
        const int8_t * lut_base_b = lut_base_a + 2u * (size_t) k_ifairy_lut_group_bytes;

        const __m512i lut_0a = _mm512_loadu_si512((const void *) (lut_base_a + 0));
        const __m512i lut_1a = _mm512_loadu_si512((const void *) (lut_base_a + 64));
        const __m512i lut_0b = _mm512_loadu_si512((const void *) (lut_base_b + 0));
        const __m512i lut_1b = _mm512_loadu_si512((const void *) (lut_base_b + 64));

        ggml_ifairy_lut_accumulate_indexes_pair_vnni_avx512(
            lut_0a, lut_1a, _mm_load_si128((const __m128i *) &wt0->qs[byte_idx]), lut_0b, lut_1b,
            _mm_load_si128((const __m128i *) &wt0->qs[byte_idx + 1]), one, mask_idx, sum0_03, sum0_47, sum0_8b,
            sum0_cf);
        ggml_ifairy_lut_accumulate_indexes_pair_vnni_avx512(
            lut_0a, lut_1a, _mm_load_si128((const __m128i *) &wt1->qs[byte_idx]), lut_0b, lut_1b,
            _mm_load_si128((const __m128i *) &wt1->qs[byte_idx + 1]), one, mask_idx, sum1_03, sum1_47, sum1_8b,
            sum1_cf);
    }

    ggml_ifairy_lut_apply_tile_sums_i32_avx512(wt0, sum0_03, sum0_47, sum0_8b, sum0_cf, v_lr, v_li, acc0_r, acc0_i);
    ggml_ifairy_lut_apply_tile_sums_i32_avx512(wt1, sum1_03, sum1_47, sum1_8b, sum1_cf, v_lr, v_li, acc1_r, acc1_i);
}

static GGML_IFAIRY_LUT_NOINLINE void ggml_ifairy64_lut_qgemm_vnni_avx512(int64_t                        blocks,
                                                                         int                            m,
                                                                         const ifairy64_lut_wtile_16 * wtiles,
                                                                         const int8_t *                lut_col,
                                                                         const float *                 scales,
                                                                         uint8_t *                     dst_col,
                                                                         size_t                        dst_row_stride,
                                                                         bool                          pack_bf16) {
    const __m512i one      = _mm512_set1_epi8(1);
    const __m512i mask_idx = _mm512_set1_epi8(0x0f);

    constexpr int groups_per_block_const = QK_IFAIRY64_GROUPS_PER_BLOCK;
    constexpr int num_bytes              = groups_per_block_const / 2;

    const size_t lut_block_stride = (size_t) groups_per_block_const * (size_t) k_ifairy_lut_group_bytes;
    const int    tiles            = (m + 15) / 16;
    const int    tile_pairs       = tiles & ~1;

    for (int t0 = 0; t0 < tile_pairs; t0 += 2) {
        const int t1 = t0 + 1;

        __m512 acc0_r = _mm512_setzero_ps();
        __m512 acc0_i = _mm512_setzero_ps();
        __m512 acc1_r = _mm512_setzero_ps();
        __m512 acc1_i = _mm512_setzero_ps();

        for (int64_t blk = 0; blk < blocks; ++blk) {
            const ifairy64_lut_wtile_16 * wt0     = wtiles + (size_t) t0 * (size_t) blocks + (size_t) blk;
            const ifairy64_lut_wtile_16 * wt1     = wtiles + (size_t) t1 * (size_t) blocks + (size_t) blk;
            const int8_t *                lut_blk = lut_col + (size_t) blk * lut_block_stride;
            const __m512                  v_lr    = _mm512_set1_ps(scales[(size_t) blk * 2 + 0]);
            const __m512                  v_li    = _mm512_set1_ps(scales[(size_t) blk * 2 + 1]);

            ggml_ifairy_lut_accumulate_tile_pair_vnni_avx512<ifairy64_lut_wtile_16, num_bytes>(
                wt0, wt1, lut_blk, one, mask_idx, v_lr, v_li, acc0_r, acc0_i, acc1_r, acc1_i);
        }

        ggml_ifairy_lut_store_tile_avx512(t0, m, dst_col, dst_row_stride, pack_bf16, acc0_r, acc0_i);
        ggml_ifairy_lut_store_tile_avx512(t1, m, dst_col, dst_row_stride, pack_bf16, acc1_r, acc1_i);
    }

    if (tile_pairs < tiles) {
        const int t = tile_pairs;

        __m512 acc_r = _mm512_setzero_ps();
        __m512 acc_i = _mm512_setzero_ps();

        for (int64_t blk = 0; blk < blocks; ++blk) {
            const ifairy64_lut_wtile_16 * wt      = wtiles + (size_t) t * (size_t) blocks + (size_t) blk;
            const int8_t *                lut_blk = lut_col + (size_t) blk * lut_block_stride;
            const __m512                  v_lr    = _mm512_set1_ps(scales[(size_t) blk * 2 + 0]);
            const __m512                  v_li    = _mm512_set1_ps(scales[(size_t) blk * 2 + 1]);

            ggml_ifairy_lut_accumulate_single_tile_vnni_avx512<ifairy64_lut_wtile_16, num_bytes>(
                wt, lut_blk, one, mask_idx, v_lr, v_li, acc_r, acc_i);
        }

        ggml_ifairy_lut_store_tile_avx512(t, m, dst_col, dst_row_stride, pack_bf16, acc_r, acc_i);
    }
}
#    endif

#endif

#if defined(__AVX2__)
static inline __m256i ggml_ifairy_lut_fp32_to_bf16_u32_avx2(const __m256 v) {
    const __m256i x       = _mm256_castps_si256(v);
    const __m256i lsb     = _mm256_and_si256(_mm256_srli_epi32(x, 16), _mm256_set1_epi32(1));
    const __m256i rounded = _mm256_add_epi32(x, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), lsb));
    const __m256i bf16    = _mm256_srli_epi32(rounded, 16);

    const __m256i abs_x  = _mm256_and_si256(x, _mm256_set1_epi32(0x7fffffff));
    const __m256i is_nan = _mm256_cmpgt_epi32(abs_x, _mm256_set1_epi32(0x7f800000));
    const __m256i qnan   = _mm256_or_si256(_mm256_srli_epi32(x, 16), _mm256_set1_epi32(64));
    return _mm256_blendv_epi8(bf16, qnan, is_nan);
}

template <typename wtile_type>
static inline void ggml_ifairy_lut_apply_tile_sums_avx2(const wtile_type * wt,
                                                        const __m256i &    sum_01_lo,
                                                        const __m256i &    sum_01_hi,
                                                        const __m256i &    sum_23_lo,
                                                        const __m256i &    sum_23_hi,
                                                        const __m256 &     v_lr,
                                                        const __m256 &     v_li,
                                                        __m256 &           acc_r_lo,
                                                        __m256 &           acc_r_hi,
                                                        __m256 &           acc_i_lo,
                                                        __m256 &           acc_i_hi) {
    const __m128i sum_ac_lo_s16 = _mm256_castsi256_si128(sum_01_lo);
    const __m128i sum_bd_lo_s16 = _mm256_extracti128_si256(sum_01_lo, 1);
    const __m128i sum_ac_hi_s16 = _mm256_castsi256_si128(sum_01_hi);
    const __m128i sum_bd_hi_s16 = _mm256_extracti128_si256(sum_01_hi, 1);

    const __m128i sum_bc_lo_s16 = _mm256_castsi256_si128(sum_23_lo);
    const __m128i sum_ad_lo_s16 = _mm256_extracti128_si256(sum_23_lo, 1);
    const __m128i sum_bc_hi_s16 = _mm256_castsi256_si128(sum_23_hi);
    const __m128i sum_ad_hi_s16 = _mm256_extracti128_si256(sum_23_hi, 1);

    const __m256 v_ac_lo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sum_ac_lo_s16));
    const __m256 v_ac_hi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sum_ac_hi_s16));
    const __m256 v_bc_lo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sum_bc_lo_s16));
    const __m256 v_bc_hi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sum_bc_hi_s16));
    const __m256 v_ad_lo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sum_ad_lo_s16));
    const __m256 v_ad_hi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sum_ad_hi_s16));
    const __m256 v_bd_lo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sum_bd_lo_s16));
    const __m256 v_bd_hi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sum_bd_hi_s16));

    const __m256 wr_lo = ggml_ifairy_lut_load_scale8_avx2(wt->d_real + 0);
    const __m256 wr_hi = ggml_ifairy_lut_load_scale8_avx2(wt->d_real + 8);
    const __m256 wi_lo = ggml_ifairy_lut_load_scale8_avx2(wt->d_imag + 0);
    const __m256 wi_hi = ggml_ifairy_lut_load_scale8_avx2(wt->d_imag + 8);

#    ifdef __FMA__
    acc_r_lo = _mm256_fmadd_ps(v_ac_lo, _mm256_mul_ps(v_lr, wr_lo), acc_r_lo);
    acc_r_lo = _mm256_fmadd_ps(v_bd_lo, _mm256_mul_ps(v_li, wi_lo), acc_r_lo);
    acc_r_hi = _mm256_fmadd_ps(v_ac_hi, _mm256_mul_ps(v_lr, wr_hi), acc_r_hi);
    acc_r_hi = _mm256_fmadd_ps(v_bd_hi, _mm256_mul_ps(v_li, wi_hi), acc_r_hi);

    acc_i_lo = _mm256_fmadd_ps(v_bc_lo, _mm256_mul_ps(v_lr, wi_lo), acc_i_lo);
    acc_i_lo = _mm256_fmadd_ps(v_ad_lo, _mm256_mul_ps(v_li, wr_lo), acc_i_lo);
    acc_i_hi = _mm256_fmadd_ps(v_bc_hi, _mm256_mul_ps(v_lr, wi_hi), acc_i_hi);
    acc_i_hi = _mm256_fmadd_ps(v_ad_hi, _mm256_mul_ps(v_li, wr_hi), acc_i_hi);
#    else
    const __m256 lr_wr_lo = _mm256_mul_ps(v_lr, wr_lo);
    const __m256 lr_wr_hi = _mm256_mul_ps(v_lr, wr_hi);
    const __m256 li_wi_lo = _mm256_mul_ps(v_li, wi_lo);
    const __m256 li_wi_hi = _mm256_mul_ps(v_li, wi_hi);
    const __m256 lr_wi_lo = _mm256_mul_ps(v_lr, wi_lo);
    const __m256 lr_wi_hi = _mm256_mul_ps(v_lr, wi_hi);
    const __m256 li_wr_lo = _mm256_mul_ps(v_li, wr_lo);
    const __m256 li_wr_hi = _mm256_mul_ps(v_li, wr_hi);

    acc_r_lo = _mm256_add_ps(acc_r_lo, _mm256_mul_ps(v_ac_lo, lr_wr_lo));
    acc_r_lo = _mm256_add_ps(acc_r_lo, _mm256_mul_ps(v_bd_lo, li_wi_lo));
    acc_r_hi = _mm256_add_ps(acc_r_hi, _mm256_mul_ps(v_ac_hi, lr_wr_hi));
    acc_r_hi = _mm256_add_ps(acc_r_hi, _mm256_mul_ps(v_bd_hi, li_wi_hi));

    acc_i_lo = _mm256_add_ps(acc_i_lo, _mm256_mul_ps(v_bc_lo, lr_wi_lo));
    acc_i_lo = _mm256_add_ps(acc_i_lo, _mm256_mul_ps(v_ad_lo, li_wr_lo));
    acc_i_hi = _mm256_add_ps(acc_i_hi, _mm256_mul_ps(v_bc_hi, lr_wi_hi));
    acc_i_hi = _mm256_add_ps(acc_i_hi, _mm256_mul_ps(v_ad_hi, li_wr_hi));
#    endif
}

static inline void ggml_ifairy_lut_store_tile_avx2(int             tile,
                                                   int             m,
                                                   uint8_t *       dst_col,
                                                   size_t          dst_row_stride,
                                                   bool            pack_bf16,
                                                   const __m256 &  acc_r_lo,
                                                   const __m256 &  acc_r_hi,
                                                   const __m256 &  acc_i_lo,
                                                   const __m256 &  acc_i_hi) {
    const int base_row = tile << 4;
    if (pack_bf16 && dst_row_stride == sizeof(float) && base_row + 15 < m) {
        __m256i out_lo = ggml_ifairy_lut_fp32_to_bf16_u32_avx2(acc_r_lo);
        __m256i out_hi = ggml_ifairy_lut_fp32_to_bf16_u32_avx2(acc_r_hi);
        out_lo         = _mm256_or_si256(
            out_lo, _mm256_slli_epi32(ggml_ifairy_lut_fp32_to_bf16_u32_avx2(acc_i_lo), 16));
        out_hi = _mm256_or_si256(
            out_hi, _mm256_slli_epi32(ggml_ifairy_lut_fp32_to_bf16_u32_avx2(acc_i_hi), 16));

        uint8_t * out_base = dst_col + (size_t) base_row * sizeof(float);
        _mm256_storeu_si256((__m256i *) (out_base + 0), out_lo);
        _mm256_storeu_si256((__m256i *) (out_base + 8 * sizeof(float)), out_hi);
        return;
    }

    alignas(32) float out_r[16];
    alignas(32) float out_i[16];
    _mm256_store_ps(out_r + 0, acc_r_lo);
    _mm256_store_ps(out_r + 8, acc_r_hi);
    _mm256_store_ps(out_i + 0, acc_i_lo);
    _mm256_store_ps(out_i + 8, acc_i_hi);

    for (int lane = 0; lane < 16; ++lane) {
        const int row = base_row + lane;
        if (row >= m) {
            break;
        }

        uint8_t * out_base = dst_col + (size_t) row * dst_row_stride;
        if (pack_bf16) {
            ((ggml_bf16_t *) out_base)[0] = GGML_FP32_TO_BF16(out_r[lane]);
            ((ggml_bf16_t *) out_base)[1] = GGML_FP32_TO_BF16(out_i[lane]);
        } else {
            ((float *) out_base)[0] = out_r[lane];
            ((float *) out_base)[1] = out_i[lane];
        }
    }
}

template <typename wtile_type, int num_bytes>
static inline void ggml_ifairy_lut_accumulate_single_tile_avx2(const wtile_type * wt,
                                                               const int8_t *     lut_blk,
                                                               const __m256i &    one,
                                                               const __m256i &    mask_idx,
                                                               const __m256 &     v_lr,
                                                               const __m256 &     v_li,
                                                               __m256 &           acc_r_lo,
                                                               __m256 &           acc_r_hi,
                                                               __m256 &           acc_i_lo,
                                                               __m256 &           acc_i_hi) {
    __m256i sum_01_lo = _mm256_setzero_si256();
    __m256i sum_01_hi = _mm256_setzero_si256();
    __m256i sum_23_lo = _mm256_setzero_si256();
    __m256i sum_23_hi = _mm256_setzero_si256();

#    if defined(__GNUC__) || defined(__clang__)
#        pragma GCC unroll 2
#    endif
    for (int byte_idx = 0; byte_idx < num_bytes; ++byte_idx) {
        const int8_t * lut_base = lut_blk + (size_t) byte_idx * 2u * (size_t) k_ifairy_lut_group_bytes;

        const __m256i lut01_0 = _mm256_loadu_si256((const __m256i *) (lut_base + 0));
        const __m256i lut23_0 = _mm256_loadu_si256((const __m256i *) (lut_base + 32));
        const __m256i lut01_1 = _mm256_loadu_si256((const __m256i *) (lut_base + 64));
        const __m256i lut23_1 = _mm256_loadu_si256((const __m256i *) (lut_base + 96));

        const __m128i packed_128 = _mm_load_si128((const __m128i *) &wt->qs[byte_idx]);
        const __m256i packed     = _mm256_broadcastsi128_si256(packed_128);

        const __m256i idx_lo = _mm256_and_si256(packed, mask_idx);
        const __m256i idx_hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_idx);

        const __m256i out01_0 = _mm256_shuffle_epi8(lut01_0, idx_lo);
        const __m256i out23_0 = _mm256_shuffle_epi8(lut23_0, idx_lo);
        const __m256i out01_1 = _mm256_shuffle_epi8(lut01_1, idx_hi);
        const __m256i out23_1 = _mm256_shuffle_epi8(lut23_1, idx_hi);

        const __m256i lo01 = _mm256_unpacklo_epi8(out01_0, out01_1);
        const __m256i hi01 = _mm256_unpackhi_epi8(out01_0, out01_1);
        sum_01_lo          = _mm256_add_epi16(sum_01_lo, _mm256_maddubs_epi16(one, lo01));
        sum_01_hi          = _mm256_add_epi16(sum_01_hi, _mm256_maddubs_epi16(one, hi01));

        const __m256i lo23 = _mm256_unpacklo_epi8(out23_0, out23_1);
        const __m256i hi23 = _mm256_unpackhi_epi8(out23_0, out23_1);
        sum_23_lo          = _mm256_add_epi16(sum_23_lo, _mm256_maddubs_epi16(one, lo23));
        sum_23_hi          = _mm256_add_epi16(sum_23_hi, _mm256_maddubs_epi16(one, hi23));
    }

    ggml_ifairy_lut_apply_tile_sums_avx2(wt, sum_01_lo, sum_01_hi, sum_23_lo, sum_23_hi, v_lr, v_li, acc_r_lo,
                                         acc_r_hi, acc_i_lo, acc_i_hi);
}

template <typename wtile_type, int num_bytes>
static inline void ggml_ifairy_lut_accumulate_tile_pair_avx2(const wtile_type * wt0,
                                                             const wtile_type * wt1,
                                                             const int8_t *     lut_blk,
                                                             const __m256i &    one,
                                                             const __m256i &    mask_idx,
                                                             const __m256 &     v_lr,
                                                             const __m256 &     v_li,
                                                             __m256 &           acc0_r_lo,
                                                             __m256 &           acc0_r_hi,
                                                             __m256 &           acc0_i_lo,
                                                             __m256 &           acc0_i_hi,
                                                             __m256 &           acc1_r_lo,
                                                             __m256 &           acc1_r_hi,
                                                             __m256 &           acc1_i_lo,
                                                             __m256 &           acc1_i_hi) {
    __m256i sum0_01_lo = _mm256_setzero_si256();
    __m256i sum0_01_hi = _mm256_setzero_si256();
    __m256i sum0_23_lo = _mm256_setzero_si256();
    __m256i sum0_23_hi = _mm256_setzero_si256();

    __m256i sum1_01_lo = _mm256_setzero_si256();
    __m256i sum1_01_hi = _mm256_setzero_si256();
    __m256i sum1_23_lo = _mm256_setzero_si256();
    __m256i sum1_23_hi = _mm256_setzero_si256();

#    if defined(__GNUC__) || defined(__clang__)
#        pragma GCC unroll 2
#    endif
    for (int byte_idx = 0; byte_idx < num_bytes; ++byte_idx) {
        const int8_t * lut_base = lut_blk + (size_t) byte_idx * 2u * (size_t) k_ifairy_lut_group_bytes;

        const __m256i lut01_0 = _mm256_loadu_si256((const __m256i *) (lut_base + 0));
        const __m256i lut23_0 = _mm256_loadu_si256((const __m256i *) (lut_base + 32));
        const __m256i lut01_1 = _mm256_loadu_si256((const __m256i *) (lut_base + 64));
        const __m256i lut23_1 = _mm256_loadu_si256((const __m256i *) (lut_base + 96));

        const __m128i packed0_128 = _mm_load_si128((const __m128i *) &wt0->qs[byte_idx]);
        const __m256i packed0     = _mm256_broadcastsi128_si256(packed0_128);

        const __m256i idx0_lo = _mm256_and_si256(packed0, mask_idx);
        const __m256i idx0_hi = _mm256_and_si256(_mm256_srli_epi16(packed0, 4), mask_idx);

        const __m256i out01_0a = _mm256_shuffle_epi8(lut01_0, idx0_lo);
        const __m256i out23_0a = _mm256_shuffle_epi8(lut23_0, idx0_lo);
        const __m256i out01_1a = _mm256_shuffle_epi8(lut01_1, idx0_hi);
        const __m256i out23_1a = _mm256_shuffle_epi8(lut23_1, idx0_hi);

        const __m256i lo01_a = _mm256_unpacklo_epi8(out01_0a, out01_1a);
        const __m256i hi01_a = _mm256_unpackhi_epi8(out01_0a, out01_1a);
        sum0_01_lo           = _mm256_add_epi16(sum0_01_lo, _mm256_maddubs_epi16(one, lo01_a));
        sum0_01_hi           = _mm256_add_epi16(sum0_01_hi, _mm256_maddubs_epi16(one, hi01_a));

        const __m256i lo23_a = _mm256_unpacklo_epi8(out23_0a, out23_1a);
        const __m256i hi23_a = _mm256_unpackhi_epi8(out23_0a, out23_1a);
        sum0_23_lo           = _mm256_add_epi16(sum0_23_lo, _mm256_maddubs_epi16(one, lo23_a));
        sum0_23_hi           = _mm256_add_epi16(sum0_23_hi, _mm256_maddubs_epi16(one, hi23_a));

        const __m128i packed1_128 = _mm_load_si128((const __m128i *) &wt1->qs[byte_idx]);
        const __m256i packed1     = _mm256_broadcastsi128_si256(packed1_128);

        const __m256i idx1_lo = _mm256_and_si256(packed1, mask_idx);
        const __m256i idx1_hi = _mm256_and_si256(_mm256_srli_epi16(packed1, 4), mask_idx);

        const __m256i out01_0b = _mm256_shuffle_epi8(lut01_0, idx1_lo);
        const __m256i out23_0b = _mm256_shuffle_epi8(lut23_0, idx1_lo);
        const __m256i out01_1b = _mm256_shuffle_epi8(lut01_1, idx1_hi);
        const __m256i out23_1b = _mm256_shuffle_epi8(lut23_1, idx1_hi);

        const __m256i lo01_b = _mm256_unpacklo_epi8(out01_0b, out01_1b);
        const __m256i hi01_b = _mm256_unpackhi_epi8(out01_0b, out01_1b);
        sum1_01_lo           = _mm256_add_epi16(sum1_01_lo, _mm256_maddubs_epi16(one, lo01_b));
        sum1_01_hi           = _mm256_add_epi16(sum1_01_hi, _mm256_maddubs_epi16(one, hi01_b));

        const __m256i lo23_b = _mm256_unpacklo_epi8(out23_0b, out23_1b);
        const __m256i hi23_b = _mm256_unpackhi_epi8(out23_0b, out23_1b);
        sum1_23_lo           = _mm256_add_epi16(sum1_23_lo, _mm256_maddubs_epi16(one, lo23_b));
        sum1_23_hi           = _mm256_add_epi16(sum1_23_hi, _mm256_maddubs_epi16(one, hi23_b));
    }

    ggml_ifairy_lut_apply_tile_sums_avx2(wt0, sum0_01_lo, sum0_01_hi, sum0_23_lo, sum0_23_hi, v_lr, v_li, acc0_r_lo,
                                         acc0_r_hi, acc0_i_lo, acc0_i_hi);
    ggml_ifairy_lut_apply_tile_sums_avx2(wt1, sum1_01_lo, sum1_01_hi, sum1_23_lo, sum1_23_hi, v_lr, v_li, acc1_r_lo,
                                         acc1_r_hi, acc1_i_lo, acc1_i_hi);
}
#endif

template <typename wtile_type>
static void ggml_ifairy_lut_qgemm_lut16_one(int64_t             blocks,
                                            int64_t             groups_per_block,
                                            int64_t             groups,
                                            int                 m,
                                            const wtile_type *  wtiles,
                                            const int8_t *      lut_col,
                                            const float *       scales,
                                            uint8_t *           dst_col,
                                            size_t              dst_row_stride,
                                            bool                pack_bf16,
                                            bool                add) {
    (void) groups;
    const int tiles = (m + 15) / 16;

    if (add) {
        for (int row = 0; row < m; ++row) {
            const int tile = row >> 4;
            const int lane = row & 15;

            uint8_t * out_base = dst_col + (size_t) row * dst_row_stride;
            float     out_r    = 0.0f;
            float     out_i    = 0.0f;
            if (pack_bf16) {
                out_r = GGML_BF16_TO_FP32(((const ggml_bf16_t *) out_base)[0]);
                out_i = GGML_BF16_TO_FP32(((const ggml_bf16_t *) out_base)[1]);
            } else {
                out_r = ((const float *) out_base)[0];
                out_i = ((const float *) out_base)[1];
            }

            for (int64_t blk = 0; blk < blocks; ++blk) {
                const wtile_type * wt = wtiles + (size_t) tile * (size_t) blocks + (size_t) blk;
                const float        lr = scales[blk * 2 + 0];
                const float        li = scales[blk * 2 + 1];
                const float        wr = ggml_ifairy_lut_scale_to_f32(wt->d_real[lane]);
                const float        wi = ggml_ifairy_lut_scale_to_f32(wt->d_imag[lane]);

                int sum_ac = 0;
                int sum_bc = 0;
                int sum_ad = 0;
                int sum_bd = 0;

                const int8_t * lut_blk =
                    lut_col + (size_t) blk * (size_t) groups_per_block * (size_t) k_ifairy_lut_group_bytes;
                for (int byte_idx = 0; byte_idx < groups_per_block / 2; ++byte_idx) {
                    const uint8_t  packed = wt->qs[byte_idx][lane];
                    const int8_t * tbl_0  = lut_blk + (size_t) (byte_idx * 2) * (size_t) k_ifairy_lut_group_bytes;
                    const int8_t * tbl_1  = tbl_0 + (size_t) k_ifairy_lut_group_bytes;

                    int8_t v0 = 0;
                    int8_t v1 = 0;
                    int8_t v2 = 0;
                    int8_t v3 = 0;
                    ggml_ifairy_lut_decode_lane_scalar(packed & 0x0fu, tbl_0, v0, v1, v2, v3);
                    sum_ac += (int) v0;
                    sum_bc += (int) v1;
                    sum_ad += (int) v2;
                    sum_bd += (int) v3;

                    ggml_ifairy_lut_decode_lane_scalar((packed >> 4) & 0x0fu, tbl_1, v0, v1, v2, v3);
                    sum_ac += (int) v0;
                    sum_bc += (int) v1;
                    sum_ad += (int) v2;
                    sum_bd += (int) v3;
                }

                out_r += (float) sum_ac * (lr * wr) + (float) sum_bd * (li * wi);
                out_i += (float) sum_bc * (lr * wi) + (float) sum_ad * (li * wr);
            }

            if (pack_bf16) {
                ((ggml_bf16_t *) out_base)[0] = GGML_FP32_TO_BF16(out_r);
                ((ggml_bf16_t *) out_base)[1] = GGML_FP32_TO_BF16(out_i);
            } else {
                ((float *) out_base)[0] = out_r;
                ((float *) out_base)[1] = out_i;
            }
        }
        return;
    }

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX2__)
    if constexpr (std::is_same_v<wtile_type, ifairy64_lut_wtile_16>) {
        constexpr int groups_per_block_const = QK_IFAIRY64_GROUPS_PER_BLOCK;
        constexpr int num_bytes              = groups_per_block_const / 2;
        GGML_ASSERT(groups_per_block == groups_per_block_const);

#    if defined(__AVX512VNNI__)
        if (blocks >= 32 && blocks <= 128) {
            ggml_ifairy64_lut_qgemm_vnni_avx512(
                blocks, m, wtiles, lut_col, scales, dst_col, dst_row_stride, pack_bf16);
            return;
        }
#    endif

        const __m512i one      = _mm512_set1_epi8(1);
        const __m512i mask_idx = _mm512_set1_epi8(0x0f);

        const size_t lut_block_stride = (size_t) groups_per_block_const * (size_t) k_ifairy_lut_group_bytes;
        const int    tile_pairs       = tiles & ~1;

        for (int t0 = 0; t0 < tile_pairs; t0 += 2) {
            const int t1 = t0 + 1;

            __m512 acc0_r = _mm512_setzero_ps();
            __m512 acc0_i = _mm512_setzero_ps();
            __m512 acc1_r = _mm512_setzero_ps();
            __m512 acc1_i = _mm512_setzero_ps();

            for (int64_t blk = 0; blk < blocks; ++blk) {
                const wtile_type * wt0     = wtiles + (size_t) t0 * (size_t) blocks + (size_t) blk;
                const wtile_type * wt1     = wtiles + (size_t) t1 * (size_t) blocks + (size_t) blk;
                const int8_t *     lut_blk = lut_col + (size_t) blk * lut_block_stride;
                const __m512       v_lr    = _mm512_set1_ps(scales[(size_t) blk * 2 + 0]);
                const __m512       v_li    = _mm512_set1_ps(scales[(size_t) blk * 2 + 1]);

                ggml_ifairy_lut_accumulate_tile_pair_avx512<wtile_type, num_bytes>(
                    wt0, wt1, lut_blk, one, mask_idx, v_lr, v_li, acc0_r, acc0_i, acc1_r, acc1_i);
            }

            ggml_ifairy_lut_store_tile_avx512(t0, m, dst_col, dst_row_stride, pack_bf16, acc0_r, acc0_i);
            ggml_ifairy_lut_store_tile_avx512(t1, m, dst_col, dst_row_stride, pack_bf16, acc1_r, acc1_i);
        }

        if (tile_pairs < tiles) {
            const int t = tile_pairs;

            __m512 acc_r = _mm512_setzero_ps();
            __m512 acc_i = _mm512_setzero_ps();

            for (int64_t blk = 0; blk < blocks; ++blk) {
                const wtile_type * wt      = wtiles + (size_t) t * (size_t) blocks + (size_t) blk;
                const int8_t *     lut_blk = lut_col + (size_t) blk * lut_block_stride;
                const __m512       v_lr    = _mm512_set1_ps(scales[(size_t) blk * 2 + 0]);
                const __m512       v_li    = _mm512_set1_ps(scales[(size_t) blk * 2 + 1]);

                ggml_ifairy_lut_accumulate_single_tile_avx512<wtile_type, num_bytes>(
                    wt, lut_blk, one, mask_idx, v_lr, v_li, acc_r, acc_i);
            }

            ggml_ifairy_lut_store_tile_avx512(t, m, dst_col, dst_row_stride, pack_bf16, acc_r, acc_i);
        }
        return;
    }
#endif

#if defined(__AVX2__)
    const __m256i one      = _mm256_set1_epi8(1);
    const __m256i mask_idx = _mm256_set1_epi8(0x0f);

    constexpr int groups_per_block_const =
        std::is_same_v<wtile_type, ifairy64_lut_wtile_16> ? QK_IFAIRY64_GROUPS_PER_BLOCK : QK_IFAIRY_GROUPS_PER_BLOCK;
    constexpr int num_bytes = groups_per_block_const / 2;
    GGML_ASSERT(groups_per_block == groups_per_block_const);

    const size_t lut_block_stride = (size_t) groups_per_block_const * (size_t) k_ifairy_lut_group_bytes;
    const int    tile_pairs       = tiles & ~1;

    for (int t0 = 0; t0 < tile_pairs; t0 += 2) {
        const int t1 = t0 + 1;

        __m256 acc0_r_lo = _mm256_setzero_ps();
        __m256 acc0_r_hi = _mm256_setzero_ps();
        __m256 acc0_i_lo = _mm256_setzero_ps();
        __m256 acc0_i_hi = _mm256_setzero_ps();

        __m256 acc1_r_lo = _mm256_setzero_ps();
        __m256 acc1_r_hi = _mm256_setzero_ps();
        __m256 acc1_i_lo = _mm256_setzero_ps();
        __m256 acc1_i_hi = _mm256_setzero_ps();

        for (int64_t blk = 0; blk < blocks; ++blk) {
            const wtile_type * wt0     = wtiles + (size_t) t0 * (size_t) blocks + (size_t) blk;
            const wtile_type * wt1     = wtiles + (size_t) t1 * (size_t) blocks + (size_t) blk;
            const int8_t *     lut_blk = lut_col + (size_t) blk * lut_block_stride;
            const __m256       v_lr    = _mm256_set1_ps(scales[(size_t) blk * 2 + 0]);
            const __m256       v_li    = _mm256_set1_ps(scales[(size_t) blk * 2 + 1]);

            ggml_ifairy_lut_accumulate_tile_pair_avx2<wtile_type, num_bytes>(
                wt0, wt1, lut_blk, one, mask_idx, v_lr, v_li, acc0_r_lo, acc0_r_hi, acc0_i_lo, acc0_i_hi, acc1_r_lo,
                acc1_r_hi, acc1_i_lo, acc1_i_hi);
        }

        ggml_ifairy_lut_store_tile_avx2(t0, m, dst_col, dst_row_stride, pack_bf16, acc0_r_lo, acc0_r_hi, acc0_i_lo,
                                        acc0_i_hi);
        ggml_ifairy_lut_store_tile_avx2(t1, m, dst_col, dst_row_stride, pack_bf16, acc1_r_lo, acc1_r_hi, acc1_i_lo,
                                        acc1_i_hi);
    }

    if (tile_pairs < tiles) {
        const int t = tile_pairs;

        __m256 acc_r_lo = _mm256_setzero_ps();
        __m256 acc_r_hi = _mm256_setzero_ps();
        __m256 acc_i_lo = _mm256_setzero_ps();
        __m256 acc_i_hi = _mm256_setzero_ps();

        for (int64_t blk = 0; blk < blocks; ++blk) {
            const wtile_type * wt      = wtiles + (size_t) t * (size_t) blocks + (size_t) blk;
            const int8_t *     lut_blk = lut_col + (size_t) blk * lut_block_stride;
            const __m256       v_lr    = _mm256_set1_ps(scales[(size_t) blk * 2 + 0]);
            const __m256       v_li    = _mm256_set1_ps(scales[(size_t) blk * 2 + 1]);

            ggml_ifairy_lut_accumulate_single_tile_avx2<wtile_type, num_bytes>(
                wt, lut_blk, one, mask_idx, v_lr, v_li, acc_r_lo, acc_r_hi, acc_i_lo, acc_i_hi);
        }

        ggml_ifairy_lut_store_tile_avx2(t, m, dst_col, dst_row_stride, pack_bf16, acc_r_lo, acc_r_hi, acc_i_lo,
                                        acc_i_hi);
    }
    return;
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)
    const uint8x16_t mask_4bit = vdupq_n_u8(0x0f);

    if constexpr (std::is_same_v<wtile_type, ifairy64_lut_wtile_16>) {
        const int tiles_per_pass = 4;

        for (int t0 = 0; t0 < tiles; t0 += tiles_per_pass) {
            const int t1    = t0 + 1;
            const int t2    = t0 + 2;
            const int t3    = t0 + 3;
            const bool has1 = t1 < tiles;
            const bool has2 = t2 < tiles;
            const bool has3 = t3 < tiles;

            float32x4_t acc0_r0 = vdupq_n_f32(0.0f);
            float32x4_t acc0_r1 = vdupq_n_f32(0.0f);
            float32x4_t acc0_r2 = vdupq_n_f32(0.0f);
            float32x4_t acc0_r3 = vdupq_n_f32(0.0f);
            float32x4_t acc0_i0 = vdupq_n_f32(0.0f);
            float32x4_t acc0_i1 = vdupq_n_f32(0.0f);
            float32x4_t acc0_i2 = vdupq_n_f32(0.0f);
            float32x4_t acc0_i3 = vdupq_n_f32(0.0f);

            float32x4_t acc1_r0 = vdupq_n_f32(0.0f);
            float32x4_t acc1_r1 = vdupq_n_f32(0.0f);
            float32x4_t acc1_r2 = vdupq_n_f32(0.0f);
            float32x4_t acc1_r3 = vdupq_n_f32(0.0f);
            float32x4_t acc1_i0 = vdupq_n_f32(0.0f);
            float32x4_t acc1_i1 = vdupq_n_f32(0.0f);
            float32x4_t acc1_i2 = vdupq_n_f32(0.0f);
            float32x4_t acc1_i3 = vdupq_n_f32(0.0f);

            float32x4_t acc2_r0 = vdupq_n_f32(0.0f);
            float32x4_t acc2_r1 = vdupq_n_f32(0.0f);
            float32x4_t acc2_r2 = vdupq_n_f32(0.0f);
            float32x4_t acc2_r3 = vdupq_n_f32(0.0f);
            float32x4_t acc2_i0 = vdupq_n_f32(0.0f);
            float32x4_t acc2_i1 = vdupq_n_f32(0.0f);
            float32x4_t acc2_i2 = vdupq_n_f32(0.0f);
            float32x4_t acc2_i3 = vdupq_n_f32(0.0f);

            float32x4_t acc3_r0 = vdupq_n_f32(0.0f);
            float32x4_t acc3_r1 = vdupq_n_f32(0.0f);
            float32x4_t acc3_r2 = vdupq_n_f32(0.0f);
            float32x4_t acc3_r3 = vdupq_n_f32(0.0f);
            float32x4_t acc3_i0 = vdupq_n_f32(0.0f);
            float32x4_t acc3_i1 = vdupq_n_f32(0.0f);
            float32x4_t acc3_i2 = vdupq_n_f32(0.0f);
            float32x4_t acc3_i3 = vdupq_n_f32(0.0f);

            for (int64_t blk = 0; blk < blocks; ++blk) {
                const wtile_type * wt0 = wtiles + (size_t) t0 * (size_t) blocks + (size_t) blk;
                const wtile_type * wt1 = has1 ? wtiles + (size_t) t1 * (size_t) blocks + (size_t) blk : nullptr;
                const wtile_type * wt2 = has2 ? wtiles + (size_t) t2 * (size_t) blocks + (size_t) blk : nullptr;
                const wtile_type * wt3 = has3 ? wtiles + (size_t) t3 * (size_t) blocks + (size_t) blk : nullptr;

                int16x8_t sum0_ac_0 = vdupq_n_s16(0);
                int16x8_t sum0_ac_1 = vdupq_n_s16(0);
                int16x8_t sum0_bc_0 = vdupq_n_s16(0);
                int16x8_t sum0_bc_1 = vdupq_n_s16(0);
                int16x8_t sum0_ad_0 = vdupq_n_s16(0);
                int16x8_t sum0_ad_1 = vdupq_n_s16(0);
                int16x8_t sum0_bd_0 = vdupq_n_s16(0);
                int16x8_t sum0_bd_1 = vdupq_n_s16(0);

                int16x8_t sum1_ac_0 = vdupq_n_s16(0);
                int16x8_t sum1_ac_1 = vdupq_n_s16(0);
                int16x8_t sum1_bc_0 = vdupq_n_s16(0);
                int16x8_t sum1_bc_1 = vdupq_n_s16(0);
                int16x8_t sum1_ad_0 = vdupq_n_s16(0);
                int16x8_t sum1_ad_1 = vdupq_n_s16(0);
                int16x8_t sum1_bd_0 = vdupq_n_s16(0);
                int16x8_t sum1_bd_1 = vdupq_n_s16(0);

                int16x8_t sum2_ac_0 = vdupq_n_s16(0);
                int16x8_t sum2_ac_1 = vdupq_n_s16(0);
                int16x8_t sum2_bc_0 = vdupq_n_s16(0);
                int16x8_t sum2_bc_1 = vdupq_n_s16(0);
                int16x8_t sum2_ad_0 = vdupq_n_s16(0);
                int16x8_t sum2_ad_1 = vdupq_n_s16(0);
                int16x8_t sum2_bd_0 = vdupq_n_s16(0);
                int16x8_t sum2_bd_1 = vdupq_n_s16(0);

                int16x8_t sum3_ac_0 = vdupq_n_s16(0);
                int16x8_t sum3_ac_1 = vdupq_n_s16(0);
                int16x8_t sum3_bc_0 = vdupq_n_s16(0);
                int16x8_t sum3_bc_1 = vdupq_n_s16(0);
                int16x8_t sum3_ad_0 = vdupq_n_s16(0);
                int16x8_t sum3_ad_1 = vdupq_n_s16(0);
                int16x8_t sum3_bd_0 = vdupq_n_s16(0);
                int16x8_t sum3_bd_1 = vdupq_n_s16(0);

                const int8_t * lut_ptr = lut_col + blk * groups_per_block * k_ifairy_lut_group_bytes;

#    if defined(__GNUC__) || defined(__clang__)
#        pragma GCC unroll 2
#    endif
                for (int byte_idx = 0; byte_idx < groups_per_block / 2; ++byte_idx) {
                    const int8x16x4_t ilut_0 = vld1q_s8_x4(lut_ptr + 0);
                    const int8x16x4_t ilut_1 = vld1q_s8_x4(lut_ptr + 64);
                    lut_ptr += 128;

                    auto accumulate_tile = [&](const wtile_type * wt,
                                               int16x8_t &       sum_ac_0,
                                               int16x8_t &       sum_ac_1,
                                               int16x8_t &       sum_bc_0,
                                               int16x8_t &       sum_bc_1,
                                               int16x8_t &       sum_ad_0,
                                               int16x8_t &       sum_ad_1,
                                               int16x8_t &       sum_bd_0,
                                               int16x8_t &       sum_bd_1) {
                        const uint8x16_t packed = vld1q_u8(wt->qs[byte_idx]);
                        const uint8x16_t idx_lo = vandq_u8(packed, mask_4bit);
                        const uint8x16_t idx_hi = vandq_u8(vshrq_n_u8(packed, 4), mask_4bit);

                        const int8x16_t v_ac_0 = vqtbl1q_s8(ilut_0.val[0], idx_lo);
                        const int8x16_t v_bd_0 = vqtbl1q_s8(ilut_0.val[1], idx_lo);
                        const int8x16_t v_bc_0 = vqtbl1q_s8(ilut_0.val[2], idx_lo);
                        const int8x16_t v_ad_0 = vqtbl1q_s8(ilut_0.val[3], idx_lo);

                        const int8x16_t v_ac_1 = vqtbl1q_s8(ilut_1.val[0], idx_hi);
                        const int8x16_t v_bd_1 = vqtbl1q_s8(ilut_1.val[1], idx_hi);
                        const int8x16_t v_bc_1 = vqtbl1q_s8(ilut_1.val[2], idx_hi);
                        const int8x16_t v_ad_1 = vqtbl1q_s8(ilut_1.val[3], idx_hi);

                        sum_ac_0 = vaddw_s8(sum_ac_0, vget_low_s8(v_ac_0));
                        sum_ac_1 = vaddw_s8(sum_ac_1, vget_high_s8(v_ac_0));
                        sum_ac_0 = vaddw_s8(sum_ac_0, vget_low_s8(v_ac_1));
                        sum_ac_1 = vaddw_s8(sum_ac_1, vget_high_s8(v_ac_1));

                        sum_bc_0 = vaddw_s8(sum_bc_0, vget_low_s8(v_bc_0));
                        sum_bc_1 = vaddw_s8(sum_bc_1, vget_high_s8(v_bc_0));
                        sum_bc_0 = vaddw_s8(sum_bc_0, vget_low_s8(v_bc_1));
                        sum_bc_1 = vaddw_s8(sum_bc_1, vget_high_s8(v_bc_1));

                        sum_ad_0 = vaddw_s8(sum_ad_0, vget_low_s8(v_ad_0));
                        sum_ad_1 = vaddw_s8(sum_ad_1, vget_high_s8(v_ad_0));
                        sum_ad_0 = vaddw_s8(sum_ad_0, vget_low_s8(v_ad_1));
                        sum_ad_1 = vaddw_s8(sum_ad_1, vget_high_s8(v_ad_1));

                        sum_bd_0 = vaddw_s8(sum_bd_0, vget_low_s8(v_bd_0));
                        sum_bd_1 = vaddw_s8(sum_bd_1, vget_high_s8(v_bd_0));
                        sum_bd_0 = vaddw_s8(sum_bd_0, vget_low_s8(v_bd_1));
                        sum_bd_1 = vaddw_s8(sum_bd_1, vget_high_s8(v_bd_1));
                    };

                    accumulate_tile(wt0, sum0_ac_0, sum0_ac_1, sum0_bc_0, sum0_bc_1, sum0_ad_0, sum0_ad_1, sum0_bd_0,
                                    sum0_bd_1);
                    if (has1) {
                        accumulate_tile(wt1, sum1_ac_0, sum1_ac_1, sum1_bc_0, sum1_bc_1, sum1_ad_0, sum1_ad_1,
                                        sum1_bd_0, sum1_bd_1);
                    }
                    if (has2) {
                        accumulate_tile(wt2, sum2_ac_0, sum2_ac_1, sum2_bc_0, sum2_bc_1, sum2_ad_0, sum2_ad_1,
                                        sum2_bd_0, sum2_bd_1);
                    }
                    if (has3) {
                        accumulate_tile(wt3, sum3_ac_0, sum3_ac_1, sum3_bc_0, sum3_bc_1, sum3_ad_0, sum3_ad_1,
                                        sum3_bd_0, sum3_bd_1);
                    }
                }

                const float lr = scales[blk * 2 + 0];
                const float li = scales[blk * 2 + 1];
                const float32x4_t v_lr = vdupq_n_f32(lr);
                const float32x4_t v_li = vdupq_n_f32(li);

                ggml_ifairy_lut_apply_tile_sums_arm(wt0, v_lr, v_li, sum0_ac_0, sum0_ac_1, sum0_bc_0, sum0_bc_1,
                                                    sum0_ad_0, sum0_ad_1, sum0_bd_0, sum0_bd_1, acc0_r0, acc0_r1,
                                                    acc0_r2, acc0_r3, acc0_i0, acc0_i1, acc0_i2, acc0_i3);
                if (has1) {
                    ggml_ifairy_lut_apply_tile_sums_arm(wt1, v_lr, v_li, sum1_ac_0, sum1_ac_1, sum1_bc_0, sum1_bc_1,
                                                        sum1_ad_0, sum1_ad_1, sum1_bd_0, sum1_bd_1, acc1_r0, acc1_r1,
                                                        acc1_r2, acc1_r3, acc1_i0, acc1_i1, acc1_i2, acc1_i3);
                }
                if (has2) {
                    ggml_ifairy_lut_apply_tile_sums_arm(wt2, v_lr, v_li, sum2_ac_0, sum2_ac_1, sum2_bc_0, sum2_bc_1,
                                                        sum2_ad_0, sum2_ad_1, sum2_bd_0, sum2_bd_1, acc2_r0, acc2_r1,
                                                        acc2_r2, acc2_r3, acc2_i0, acc2_i1, acc2_i2, acc2_i3);
                }
                if (has3) {
                    ggml_ifairy_lut_apply_tile_sums_arm(wt3, v_lr, v_li, sum3_ac_0, sum3_ac_1, sum3_bc_0, sum3_bc_1,
                                                        sum3_ad_0, sum3_ad_1, sum3_bd_0, sum3_bd_1, acc3_r0, acc3_r1,
                                                        acc3_r2, acc3_r3, acc3_i0, acc3_i1, acc3_i2, acc3_i3);
                }
            }

            ggml_ifairy_lut_store_tile_arm(t0, m, dst_col, dst_row_stride, pack_bf16, acc0_r0, acc0_r1, acc0_r2,
                                           acc0_r3, acc0_i0, acc0_i1, acc0_i2, acc0_i3);
            if (has1) {
                ggml_ifairy_lut_store_tile_arm(t1, m, dst_col, dst_row_stride, pack_bf16, acc1_r0, acc1_r1, acc1_r2,
                                               acc1_r3, acc1_i0, acc1_i1, acc1_i2, acc1_i3);
            }
            if (has2) {
                ggml_ifairy_lut_store_tile_arm(t2, m, dst_col, dst_row_stride, pack_bf16, acc2_r0, acc2_r1, acc2_r2,
                                               acc2_r3, acc2_i0, acc2_i1, acc2_i2, acc2_i3);
            }
            if (has3) {
                ggml_ifairy_lut_store_tile_arm(t3, m, dst_col, dst_row_stride, pack_bf16, acc3_r0, acc3_r1, acc3_r2,
                                               acc3_r3, acc3_i0, acc3_i1, acc3_i2, acc3_i3);
            }
        }
        return;
    }

    for (int t = 0; t < tiles; ++t) {
        const int rows_left = m - (t << 4);
        if (rows_left <= 0) {
            break;
        }

        float32x4_t acc_r0 = vdupq_n_f32(0.0f);
        float32x4_t acc_r1 = vdupq_n_f32(0.0f);
        float32x4_t acc_r2 = vdupq_n_f32(0.0f);
        float32x4_t acc_r3 = vdupq_n_f32(0.0f);
        float32x4_t acc_i0 = vdupq_n_f32(0.0f);
        float32x4_t acc_i1 = vdupq_n_f32(0.0f);
        float32x4_t acc_i2 = vdupq_n_f32(0.0f);
        float32x4_t acc_i3 = vdupq_n_f32(0.0f);

        for (int64_t blk = 0; blk < blocks; ++blk) {
            const wtile_type * wt      = wtiles + t * blocks + blk;
            const int8_t *     lut_ptr = lut_col + blk * groups_per_block * k_ifairy_lut_group_bytes;

            int16x8_t sum_ac_0 = vdupq_n_s16(0);
            int16x8_t sum_ac_1 = vdupq_n_s16(0);
            int16x8_t sum_bc_0 = vdupq_n_s16(0);
            int16x8_t sum_bc_1 = vdupq_n_s16(0);
            int16x8_t sum_ad_0 = vdupq_n_s16(0);
            int16x8_t sum_ad_1 = vdupq_n_s16(0);
            int16x8_t sum_bd_0 = vdupq_n_s16(0);
            int16x8_t sum_bd_1 = vdupq_n_s16(0);

#    if defined(__GNUC__) || defined(__clang__)
#        pragma GCC unroll 2
#    endif
            for (int byte_idx = 0; byte_idx < groups_per_block / 2; ++byte_idx) {
                const uint8x16_t packed = vld1q_u8(wt->qs[byte_idx]);
                const uint8x16_t idx_lo = vandq_u8(packed, mask_4bit);
                const uint8x16_t idx_hi = vandq_u8(vshrq_n_u8(packed, 4), mask_4bit);

                const int8x16x4_t ilut_0 = vld1q_s8_x4(lut_ptr + 0);
                const int8x16x4_t ilut_1 = vld1q_s8_x4(lut_ptr + 64);
                lut_ptr += 128;

                const int8x16_t v_ac_0 = vqtbl1q_s8(ilut_0.val[0], idx_lo);
                const int8x16_t v_bd_0 = vqtbl1q_s8(ilut_0.val[1], idx_lo);
                const int8x16_t v_bc_0 = vqtbl1q_s8(ilut_0.val[2], idx_lo);
                const int8x16_t v_ad_0 = vqtbl1q_s8(ilut_0.val[3], idx_lo);

                const int8x16_t v_ac_1 = vqtbl1q_s8(ilut_1.val[0], idx_hi);
                const int8x16_t v_bd_1 = vqtbl1q_s8(ilut_1.val[1], idx_hi);
                const int8x16_t v_bc_1 = vqtbl1q_s8(ilut_1.val[2], idx_hi);
                const int8x16_t v_ad_1 = vqtbl1q_s8(ilut_1.val[3], idx_hi);

                sum_ac_0 = vaddw_s8(sum_ac_0, vget_low_s8(v_ac_0));
                sum_ac_1 = vaddw_s8(sum_ac_1, vget_high_s8(v_ac_0));
                sum_ac_0 = vaddw_s8(sum_ac_0, vget_low_s8(v_ac_1));
                sum_ac_1 = vaddw_s8(sum_ac_1, vget_high_s8(v_ac_1));

                sum_bc_0 = vaddw_s8(sum_bc_0, vget_low_s8(v_bc_0));
                sum_bc_1 = vaddw_s8(sum_bc_1, vget_high_s8(v_bc_0));
                sum_bc_0 = vaddw_s8(sum_bc_0, vget_low_s8(v_bc_1));
                sum_bc_1 = vaddw_s8(sum_bc_1, vget_high_s8(v_bc_1));

                sum_ad_0 = vaddw_s8(sum_ad_0, vget_low_s8(v_ad_0));
                sum_ad_1 = vaddw_s8(sum_ad_1, vget_high_s8(v_ad_0));
                sum_ad_0 = vaddw_s8(sum_ad_0, vget_low_s8(v_ad_1));
                sum_ad_1 = vaddw_s8(sum_ad_1, vget_high_s8(v_ad_1));

                sum_bd_0 = vaddw_s8(sum_bd_0, vget_low_s8(v_bd_0));
                sum_bd_1 = vaddw_s8(sum_bd_1, vget_high_s8(v_bd_0));
                sum_bd_0 = vaddw_s8(sum_bd_0, vget_low_s8(v_bd_1));
                sum_bd_1 = vaddw_s8(sum_bd_1, vget_high_s8(v_bd_1));
            }

            const float lr = scales[blk * 2 + 0];
            const float li = scales[blk * 2 + 1];

            const float32x4_t v_lr = vdupq_n_f32(lr);
            const float32x4_t v_li = vdupq_n_f32(li);
            ggml_ifairy_lut_apply_tile_sums_arm(wt, v_lr, v_li, sum_ac_0, sum_ac_1, sum_bc_0, sum_bc_1, sum_ad_0,
                                                sum_ad_1, sum_bd_0, sum_bd_1, acc_r0, acc_r1, acc_r2, acc_r3, acc_i0,
                                                acc_i1, acc_i2, acc_i3);
        }

        ggml_ifairy_lut_store_tile_arm(t, m, dst_col, dst_row_stride, pack_bf16, acc_r0, acc_r1, acc_r2, acc_r3,
                                       acc_i0, acc_i1, acc_i2, acc_i3);
    }
    return;
#endif

    for (int row = 0; row < m; ++row) {
        const int tile = row >> 4;
        const int lane = row & 15;

        float out_r = 0.0f;
        float out_i = 0.0f;

        for (int64_t blk = 0; blk < blocks; ++blk) {
            const wtile_type * wt = wtiles + (size_t) tile * (size_t) blocks + (size_t) blk;

            const float lr = scales[blk * 2 + 0];
            const float li = scales[blk * 2 + 1];
            const float wr = ggml_ifairy_lut_scale_to_f32(wt->d_real[lane]);
            const float wi = ggml_ifairy_lut_scale_to_f32(wt->d_imag[lane]);

            int sum_ac = 0;
            int sum_bc = 0;
            int sum_ad = 0;
            int sum_bd = 0;

            const int8_t * lut_blk =
                lut_col + (size_t) blk * (size_t) groups_per_block * (size_t) k_ifairy_lut_group_bytes;
            for (int byte_idx = 0; byte_idx < groups_per_block / 2; ++byte_idx) {
                const uint8_t  packed = wt->qs[byte_idx][lane];
                const int8_t * tbl_0  = lut_blk + (size_t) (byte_idx * 2) * (size_t) k_ifairy_lut_group_bytes;
                const int8_t * tbl_1  = tbl_0 + (size_t) k_ifairy_lut_group_bytes;

                int8_t v0 = 0;
                int8_t v1 = 0;
                int8_t v2 = 0;
                int8_t v3 = 0;
                ggml_ifairy_lut_decode_lane_scalar(packed & 0x0fu, tbl_0, v0, v1, v2, v3);
                sum_ac += (int) v0;
                sum_bc += (int) v1;
                sum_ad += (int) v2;
                sum_bd += (int) v3;

                ggml_ifairy_lut_decode_lane_scalar((packed >> 4) & 0x0fu, tbl_1, v0, v1, v2, v3);
                sum_ac += (int) v0;
                sum_bc += (int) v1;
                sum_ad += (int) v2;
                sum_bd += (int) v3;
            }

            out_r += (float) sum_ac * (lr * wr) + (float) sum_bd * (li * wi);
            out_i += (float) sum_bc * (lr * wi) + (float) sum_ad * (li * wr);
        }

        uint8_t * out_base = dst_col + (size_t) row * dst_row_stride;
        if (pack_bf16) {
            ((ggml_bf16_t *) out_base)[0] = GGML_FP32_TO_BF16(out_r);
            ((ggml_bf16_t *) out_base)[1] = GGML_FP32_TO_BF16(out_i);
        } else {
            ((float *) out_base)[0] = out_r;
            ((float *) out_base)[1] = out_i;
        }
    }
}

template <typename wtile_type>
static void ggml_ifairy_lut_qgemm_fused_lut16_impl(int          m,
                                                   int          k,
                                                   int          n,
                                                   const void * packed_wtiles,
                                                   const void * act,
                                                   size_t       act_stride,
                                                   void *       lut_tmp,
                                                   void *       lut_scales_tmp,
                                                   float *      dst,
                                                   size_t       dst_col_stride,
                                                   size_t       dst_row_stride,
                                                   bool         pack_bf16,
                                                   bool         add,
                                                   int64_t      block_k,
                                                   int64_t      groups_per_block,
                                                   void (*preprocess_ex)(int,
                                                                         int,
                                                                         int,
                                                                         const void *,
                                                                         size_t,
                                                                         void *,
                                                                         void *,
                                                                         int,
                                                                         int)) {
    if (!packed_wtiles || !act || !lut_tmp || !lut_scales_tmp || !dst || m <= 0 || k <= 0 || n <= 0) {
        if (!packed_wtiles || !act || !dst || m <= 0 || k <= 0 || n <= 0) {
            return;
        }
    }

    const uint8_t *           act_bytes = (const uint8_t *) act;
    uint8_t *                 dst_bytes = (uint8_t *) dst;
    const int64_t             blocks    = k / block_k;
    const int64_t             groups    = blocks * groups_per_block;
    const wtile_type *        wtiles    = (const wtile_type *) packed_wtiles;

    uint8_t * scratch_base = NULL;
    if (!lut_tmp || !lut_scales_tmp) {
        const size_t lut_bytes         = (size_t) n * (size_t) groups * (size_t) k_ifairy_lut_group_bytes;
        const size_t lut_bytes_aligned = GGML_PAD(lut_bytes, 64);
        const size_t scale_bytes       = (size_t) n * (size_t) blocks * 2u * sizeof(float);
        const size_t scratch_bytes     = ggml_ifairy_checked_add_size(lut_bytes_aligned, scale_bytes);

        static thread_local ggml_ifairy_tl_buf tl_scratch;
        scratch_base = ggml_ifairy_tl_reserve(tl_scratch, scratch_bytes);
        if (!scratch_base) {
            return;
        }
        lut_tmp        = scratch_base;
        lut_scales_tmp = scratch_base + lut_bytes_aligned;
    }

    for (int col = 0; col < n; ++col) {
        const void * act_col = act_bytes + (size_t) col * act_stride;
        uint8_t *    dst_col = dst_bytes + (size_t) col * dst_col_stride;

        preprocess_ex(m, k, 1, act_col, act_stride, lut_scales_tmp, lut_tmp, 0, 1);
        ggml_ifairy_lut_qgemm_lut16_one(blocks, groups_per_block, groups, m, wtiles, (const int8_t *) lut_tmp,
                                        (const float *) lut_scales_tmp, dst_col, dst_row_stride, pack_bf16, add);
    }
}

void ggml_ifairy_lut_qgemm_lut16(int          m,
                                 int          k,
                                 int          n,
                                 const void * packed_wtiles,
                                 const void * lut,
                                 const void * lut_scales,
                                 float *      dst,
                                 size_t       dst_col_stride,
                                 size_t       dst_row_stride,
                                 bool         pack_bf16,
                                 bool         add) {
    if (!packed_wtiles || !dst || !lut || !lut_scales || m <= 0 || k <= 0 || n <= 0) {
        return;
    }

    const int64_t                      blocks           = k / QK_IFAIRY;
    const int64_t                      groups_per_block = QK_IFAIRY_GROUPS_PER_BLOCK;
    const int64_t                      groups           = blocks * groups_per_block;
    const struct ifairy_lut_wtile_16 * wtiles           = (const struct ifairy_lut_wtile_16 *) packed_wtiles;
    for (int col = 0; col < n; ++col) {
        const int8_t * lut_col = (const int8_t *) lut + col * groups * k_ifairy_lut_group_bytes;
        const float *  scales  = (const float *) lut_scales + col * blocks * 2u;
        uint8_t *      dst_col = (uint8_t *) dst + col * dst_col_stride;
        ggml_ifairy_lut_qgemm_lut16_one(blocks, groups_per_block, groups, m, wtiles, lut_col, scales, dst_col,
                                        dst_row_stride, pack_bf16, add);
    }
}

void ggml_ifairy_lut_qgemm_fused_lut16(int          m,
                                       int          k,
                                       int          n,
                                       const void * packed_wtiles,
                                       const void * act,
                                       size_t       act_stride,
                                       void *       lut_tmp,
                                       void *       lut_scales_tmp,
                                       float *      dst,
                                       size_t       dst_col_stride,
                                       size_t       dst_row_stride,
                                       bool         pack_bf16,
                                       bool         add) {
    ggml_ifairy_lut_qgemm_fused_lut16_impl<ifairy_lut_wtile_16>(m, k, n, packed_wtiles, act, act_stride, lut_tmp,
                                                                 lut_scales_tmp, dst, dst_col_stride, dst_row_stride,
                                                                 pack_bf16, add, QK_IFAIRY, QK_IFAIRY_GROUPS_PER_BLOCK,
                                                                 ggml_ifairy_lut_preprocess_ex_lut16);
}

void ggml_ifairy64_lut_qgemm_lut16(int          m,
                                   int          k,
                                   int          n,
                                   const void * packed_wtiles,
                                   const void * lut,
                                   const void * lut_scales,
                                   float *      dst,
                                   size_t       dst_col_stride,
                                   size_t       dst_row_stride,
                                   bool         pack_bf16,
                                   bool         add) {
    if (!packed_wtiles || !dst || !lut || !lut_scales || m <= 0 || k <= 0 || n <= 0) {
        return;
    }

    const int64_t                        blocks           = k / QK_IFAIRY64;
    const int64_t                        groups_per_block = QK_IFAIRY64_GROUPS_PER_BLOCK;
    const int64_t                        groups           = blocks * groups_per_block;
    const struct ifairy64_lut_wtile_16 * wtiles           = (const struct ifairy64_lut_wtile_16 *) packed_wtiles;
    for (int col = 0; col < n; ++col) {
        const int8_t * lut_col = (const int8_t *) lut + col * groups * k_ifairy_lut_group_bytes;
        const float *  scales  = (const float *) lut_scales + col * blocks * 2u;
        uint8_t *      dst_col = (uint8_t *) dst + col * dst_col_stride;
        ggml_ifairy_lut_qgemm_lut16_one(blocks, groups_per_block, groups, m, wtiles, lut_col, scales, dst_col,
                                        dst_row_stride, pack_bf16, add);
    }
}

void ggml_ifairy64_lut_qgemm_fused_lut16(int          m,
                                         int          k,
                                         int          n,
                                         const void * packed_wtiles,
                                         const void * act,
                                         size_t       act_stride,
                                         void *       lut_tmp,
                                         void *       lut_scales_tmp,
                                         float *      dst,
                                         size_t       dst_col_stride,
                                         size_t       dst_row_stride,
                                         bool         pack_bf16,
                                         bool         add) {
    ggml_ifairy_lut_qgemm_fused_lut16_impl<ifairy64_lut_wtile_16>(m, k, n, packed_wtiles, act, act_stride, lut_tmp,
                                                                   lut_scales_tmp, dst, dst_col_stride, dst_row_stride,
                                                                   pack_bf16, add, QK_IFAIRY64,
                                                                   QK_IFAIRY64_GROUPS_PER_BLOCK,
                                                                   ggml_ifairy64_lut_preprocess_ex_lut16);
}

void ggml_ifairy_lut_qgemm_lut_c(int          m,
                                 int          k,
                                 int          n,
                                 const void * packed_wtiles,
                                 const void * lut,
                                 const void * lut_scales,
                                 float *      dst,
                                 size_t       dst_col_stride,
                                 size_t       dst_row_stride,
                                 bool         pack_bf16,
                                 bool         add) {
    ggml_ifairy_lut_qgemm_lut16(m, k, n, packed_wtiles, lut, lut_scales, dst, dst_col_stride, dst_row_stride, pack_bf16,
                                add);
}

void ggml_ifairy64_lut_qgemm_lut_c(int          m,
                                   int          k,
                                   int          n,
                                   const void * packed_wtiles,
                                   const void * lut,
                                   const void * lut_scales,
                                   float *      dst,
                                   size_t       dst_col_stride,
                                   size_t       dst_row_stride,
                                   bool         pack_bf16,
                                   bool         add) {
    ggml_ifairy64_lut_qgemm_lut16(m, k, n, packed_wtiles, lut, lut_scales, dst, dst_col_stride, dst_row_stride,
                                  pack_bf16, add);
}

void ggml_ifairy_lut_qgemm_fused_lut_c(int          m,
                                       int          k,
                                       int          n,
                                       const void * packed_wtiles,
                                       const void * act,
                                       size_t       act_stride,
                                       void *       lut_tmp,
                                       void *       lut_scales_tmp,
                                       float *      dst,
                                       size_t       dst_col_stride,
                                       size_t       dst_row_stride,
                                       bool         pack_bf16,
                                       bool         add) {
    ggml_ifairy_lut_qgemm_fused_lut16(m, k, n, packed_wtiles, act, act_stride, lut_tmp, lut_scales_tmp, dst,
                                      dst_col_stride, dst_row_stride, pack_bf16, add);
}

void ggml_ifairy64_lut_qgemm_fused_lut_c(int          m,
                                         int          k,
                                         int          n,
                                         const void * packed_wtiles,
                                         const void * act,
                                         size_t       act_stride,
                                         void *       lut_tmp,
                                         void *       lut_scales_tmp,
                                         float *      dst,
                                         size_t       dst_col_stride,
                                         size_t       dst_row_stride,
                                         bool         pack_bf16,
                                         bool         add) {
    ggml_ifairy64_lut_qgemm_fused_lut16(m, k, n, packed_wtiles, act, act_stride, lut_tmp, lut_scales_tmp, dst,
                                        dst_col_stride, dst_row_stride, pack_bf16, add);
}

void ggml_ifairy_lut_mul_mat_scalar(int          m,
                                    int          k,
                                    int          n,
                                    const void * qweights,
                                    const void * act,
                                    size_t       act_stride,
                                    float *      dst) {
    if (!qweights || !act || !dst) {
        return;
    }

    const int64_t K                = k;
    const int64_t blocks           = K / QK_IFAIRY;
    const int64_t groups_per_block = QK_IFAIRY_GROUPS_PER_BLOCK;
    const int64_t groups           = blocks * groups_per_block;

    const size_t index_bytes_raw = (size_t) m * (size_t) groups;
    const size_t index_bytes     = GGML_PAD(index_bytes_raw, 64);

    const int64_t tiles        = (m + 15) / 16;
    // 确保分配在 64 字节完美对齐边界，消除 AVX2 跨线惩罚
    const size_t  packed_bytes = GGML_PAD((size_t) tiles * (size_t) blocks * sizeof(struct ifairy_lut_wtile_16), 64);

    const size_t lut_bytes   = (size_t) n * (size_t) groups * (size_t) k_ifairy_lut_group_bytes;
    const size_t scale_bytes = (size_t) n * (size_t) blocks * 2u * sizeof(float);

    const size_t tmp0        = ggml_ifairy_checked_add_size(index_bytes, packed_bytes);
    const size_t tmp1        = ggml_ifairy_checked_add_size(tmp0, lut_bytes);
    const size_t total_bytes = ggml_ifairy_checked_add_size(tmp1, scale_bytes);

    static thread_local ggml_ifairy_tl_buf tl;
    uint8_t *                              buf = ggml_ifairy_tl_reserve(tl, total_bytes);
    if (!buf) {
        return;
    }
    uint8_t * indexes  = buf;
    uint8_t * packed_p = buf + index_bytes;
    uint8_t * lut_p    = packed_p + packed_bytes;
    float *   scales   = (float *) (lut_p + lut_bytes);

    ggml_ifairy_2w_encode((const block_ifairy *) qweights, K, m, indexes, index_bytes_raw);

    struct ifairy_lut_wtile_16 * packed_w = (struct ifairy_lut_wtile_16 *) packed_p;
    const block_ifairy *         w_blocks = (const block_ifairy *) qweights;

    for (int64_t tile = 0; tile < tiles; ++tile) {
        for (int64_t blk = 0; blk < blocks; ++blk) {
            struct ifairy_lut_wtile_16 * t = packed_w + (size_t) tile * (size_t) blocks + (size_t) blk;

            for (int lane = 0; lane < 16; ++lane) {
                const int64_t row = tile * 16 + lane;
                if (row >= m) {
                    t->d_real[lane] = 0.0f;
                    t->d_imag[lane] = 0.0f;
                } else {
                    const block_ifairy * wb = w_blocks + (size_t) row * (size_t) blocks + (size_t) blk;
                    t->d_real[lane]         = GGML_FP16_TO_FP32(wb->d_real);
                    t->d_imag[lane]         = GGML_FP16_TO_FP32(wb->d_imag);
                }
            }

            for (int byte_idx = 0; byte_idx < groups_per_block / 2; ++byte_idx) {
                for (int lane = 0; lane < 16; ++lane) {
                    const int64_t row = tile * 16 + lane;
                    if (row >= m) {
                        t->qs[byte_idx][lane] = 0;
                    } else {
                        const uint8_t * row_indexes = indexes + (size_t) row * (size_t) groups;
                        const uint8_t * blk_idx     = row_indexes + (size_t) blk * (size_t) groups_per_block;
                        t->qs[byte_idx][lane] =
                            (blk_idx[byte_idx * 2 + 0] & 0x0fu) | (uint8_t) ((blk_idx[byte_idx * 2 + 1] & 0x0fu) << 4);
                    }
                }
            }
        }
    }

    ggml_ifairy_lut_preprocess_ex_lut16(m, k, n, act, act_stride, scales, lut_p, 0, 1);

    const size_t dst_col_stride = (size_t) m * 2u * sizeof(float);
    const size_t dst_row_stride = 2u * sizeof(float);
    ggml_ifairy_lut_qgemm_lut16(m, k, n, packed_w, lut_p, scales, dst, dst_col_stride, dst_row_stride,
                                /*pack_bf16*/ false, /*add*/ false);
}
