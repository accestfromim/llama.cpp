// Copyright (c) 2024 The ggml authors
//
// SPDX-License-Identifier: MIT

#include "ggml-ifairy.h"
#define GGML_COMMON_DECL_C
#include "../ggml/src/ggml-common.h"
#undef GGML_COMMON_DECL_C
#include "../ggml/src/ggml-impl.h"

#ifdef __cplusplus
extern "C" {
#endif
void ggml_vec_dot_ifairy_q16_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_ifairy_qgemm_lut_ref(const void * w, const int8_t * qlut_r, const int8_t * qlut_i, const float * lut_scales, int64_t k, int64_t m, float * dst);
void ggml_ifairy_qgemm_lut3_ref(const void * w, const int16_t * qlut3, const int8_t * qr, const int8_t * qi, const float * lut_scales, int64_t k, int64_t m, float * dst);
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
void ggml_ifairy_qgemm_lut_neon(const void * w, const int8_t * qlut_r, const int8_t * qlut_i, const float * lut_scales, int64_t k, int64_t m, float * dst);
#endif
#ifdef __cplusplus
}
#endif

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

static void fill_random_weights(std::mt19937 & rng, block_ifairy * w_row, int blocks_per_row) {
    std::uniform_int_distribution<int> code_dist(0, 3);
    std::uniform_real_distribution<float> scale_dist(0.1f, 1.0f);

    for (int b = 0; b < blocks_per_row; ++b) {
        block_ifairy & blk = w_row[b];
        blk.d_real = GGML_FP32_TO_FP16(scale_dist(rng));
        blk.d_imag = GGML_FP32_TO_FP16(scale_dist(rng));
        std::fill(std::begin(blk.qs), std::end(blk.qs), 0);
        for (int j = 0; j < QK_K; ++j) {
            const int code = code_dist(rng) & 0x3;
            const int chunk    = j >> 6;
            const int lane     = j & 0xF;
            const int part     = (j >> 4) & 0x3;
            const int byte_idx = (chunk << 4) + lane;
            const int bit_off  = part * 2;
            blk.qs[byte_idx] |= (uint8_t) (code << bit_off);
        }
    }
}

static void fill_random_activation(std::mt19937 & rng, block_ifairy_q16 & act) {
    std::uniform_int_distribution<int> int8_dist(-8, 8);
    std::uniform_real_distribution<float> scale_dist(0.1f, 1.0f);

    act.d_real = GGML_FP32_TO_FP16(scale_dist(rng));
    act.d_imag = GGML_FP32_TO_FP16(scale_dist(rng));

    for (int j = 0; j < QK_K; ++j) {
        act.x_real[j] = (int8_t) int8_dist(rng);
        act.x_imag[j] = (int8_t) int8_dist(rng);
    }
}

int main(void) {
    const int k = QK_K;     // 256
    const int m = 4;        // rows
    const int blocks_per_row = k / QK_K;

    std::mt19937 rng(42);

    std::vector<block_ifairy> weights((size_t) m * blocks_per_row);
    for (int row = 0; row < m; ++row) {
        fill_random_weights(rng, weights.data() + (size_t) row * blocks_per_row, blocks_per_row);
    }

    block_ifairy_q16 act = {};
    fill_random_activation(rng, act);

    const size_t pairs = (size_t) k / 2;
    const size_t qlut_bytes = pairs * 32;
    const size_t packed_bytes = pairs * 4; // remains 2*k bytes
    const size_t lut_scales_bytes = 2 * sizeof(float);
    const size_t workspace_bytes = GGML_PAD(2 * qlut_bytes + packed_bytes + lut_scales_bytes, 64);

    int8_t * workspace = (int8_t *) ggml_aligned_malloc(workspace_bytes);
    assert(workspace != nullptr);

    int8_t * qlut_r = workspace;
    int8_t * qlut_i = qlut_r + qlut_bytes;
    int8_t * ar_pack = qlut_i + qlut_bytes; // k bytes
    int8_t * ai_pack = ar_pack + k;         // k bytes
    float  * lut_scales = (float *) (ai_pack + k);
    lut_scales[0] = lut_scales[1] = 0.0f;

    ggml_ifairy_preprocessor(m, k, &act, lut_scales, qlut_r, qlut_i, /*use_three_weight=*/false, ar_pack);

    // Scalar reference for LUT construction to validate NEON path
    auto clamp_s8 = [](float v) -> int8_t {
        const float clipped = v > 127.f ? 127.f : (v < -127.f ? -127.f : v);
        return (int8_t) std::lrint(clipped);
    };

    float lut_scales_ref[2] = {0.f, 0.f};
    {
        double max_r = 0.0, max_i = 0.0;
        const double d_r = GGML_FP16_TO_FP32(act.d_real);
        const double d_i = GGML_FP16_TO_FP32(act.d_imag);
        for (int j = 0; j < k; ++j) {
            max_r = std::max(max_r, std::abs((double) (int8_t) act.x_real[j] * d_r));
            max_i = std::max(max_i, std::abs((double) (int8_t) act.x_imag[j] * d_i));
        }
        lut_scales_ref[0] = max_r > 0.0 ? (float) (127.0 / max_r) : 0.0f;
        lut_scales_ref[1] = max_i > 0.0 ? (float) (127.0 / max_i) : 0.0f;
    }

    std::vector<int8_t> qlut_r_ref(qlut_bytes);
    std::vector<int8_t> qlut_i_ref(qlut_bytes);
    for (size_t pair = 0; pair < pairs; ++pair) {
        const int j0 = pair * 2;
        const int j1 = j0 + 1;
        const float d_r = GGML_FP16_TO_FP32(act.d_real);
        const float d_i = GGML_FP16_TO_FP32(act.d_imag);
        const float inv_r = lut_scales_ref[0];
        const float inv_i = lut_scales_ref[1];
        const int8_t qr0 = clamp_s8((float) (int8_t) act.x_real[j0] * d_r * inv_r);
        const int8_t qr1 = clamp_s8((float) (int8_t) act.x_real[j1] * d_r * inv_r);
        const int8_t qi0 = clamp_s8((float) (int8_t) act.x_imag[j0] * d_i * inv_i);
        const int8_t qi1 = clamp_s8((float) (int8_t) act.x_imag[j1] * d_i * inv_i);
        const size_t base = (size_t) pair * 32;
        std::fill_n(qlut_r_ref.data() + base + 0,  16, qr0);
        std::fill_n(qlut_r_ref.data() + base + 16, 16, qr1);
        std::fill_n(qlut_i_ref.data() + base + 0,  16, qi0);
        std::fill_n(qlut_i_ref.data() + base + 16, 16, qi1);

    }

    std::vector<int8_t> ar_pack_ref((size_t) k);
    std::vector<int8_t> ai_pack_ref((size_t) k);
    for (size_t pair = 0; pair < pairs; pair += 8) {
        const size_t pack_base = pair * 2;
        for (int idx = 0; idx < 8; ++idx) {
            const size_t p = pair + (size_t) idx;
            const size_t base_pack = pack_base + (size_t) idx;
            const size_t qbase = p * 32;
            ar_pack_ref[base_pack]       = qlut_r_ref[qbase + 0];
            ar_pack_ref[base_pack + 8]   = qlut_r_ref[qbase + 16];
            ai_pack_ref[base_pack]       = qlut_i_ref[qbase + 0];
            ai_pack_ref[base_pack + 8]   = qlut_i_ref[qbase + 16];
        }
    }

    const float tol_scale = 1e-5f;
    if (std::abs(lut_scales[0] - lut_scales_ref[0]) > tol_scale || std::abs(lut_scales[1] - lut_scales_ref[1]) > tol_scale) {
        std::cerr << "lut_scales mismatch: got (" << lut_scales[0] << ", " << lut_scales[1] << ")"
                  << " ref (" << lut_scales_ref[0] << ", " << lut_scales_ref[1] << ")\n";
        ggml_aligned_free(workspace, workspace_bytes);
        return 1;
    }
    if (!std::equal(qlut_r, qlut_r + qlut_bytes, qlut_r_ref.begin()) ||
        !std::equal(qlut_i, qlut_i + qlut_bytes, qlut_i_ref.begin())) {
        std::cerr << "qlut mismatch\n";
        ggml_aligned_free(workspace, workspace_bytes);
        return 1;
    }
    if (!std::equal(ar_pack, ar_pack + k, ar_pack_ref.begin()) ||
        !std::equal(ai_pack, ai_pack + k, ai_pack_ref.begin())) {
        for (int idx = 0; idx < k; ++idx) {
            if (ar_pack[idx] != ar_pack_ref[(size_t) idx] || ai_pack[idx] != ai_pack_ref[(size_t) idx]) {
                std::cerr << "packed qlut mismatch at idx " << idx
                          << " ar=" << (int) ar_pack[idx] << " ref_ar=" << (int) ar_pack_ref[(size_t) idx]
                          << " ai=" << (int) ai_pack[idx] << " ref_ai=" << (int) ai_pack_ref[(size_t) idx]
                          << std::endl;
                break;
            }
        }
        ggml_aligned_free(workspace, workspace_bytes);
        return 1;
    }

#if defined(GGML_IFAIRY_ARM_LUT_3W)
    const size_t qlut3_bytes = (size_t) (k / QK_K) * (size_t) (QK_K / 3) * 64u * 4u * sizeof(int16_t);
    const size_t packed3_bytes = (size_t) k * 2u;
    const size_t workspace3_bytes = GGML_PAD(qlut3_bytes + packed3_bytes + lut_scales_bytes, 64);
    int8_t * workspace3 = (int8_t *) ggml_aligned_malloc(workspace3_bytes);
    assert(workspace3 != nullptr);

    int16_t * qlut3 = (int16_t *) workspace3;
    int8_t  * qr_seq = (int8_t *) ((char *) workspace3 + qlut3_bytes);
    int8_t  * qi_seq = qr_seq + k;
    float   * lut_scales3 = (float *) (qi_seq + k);
    lut_scales3[0] = lut_scales3[1] = 0.0f;

    ggml_ifairy_preprocessor(m, k, &act, lut_scales3, qlut3, nullptr, /*use_three_weight=*/true, qr_seq);

    std::vector<float> dst_lut3((size_t) m, 0.f);
    ggml_ifairy_qgemm_lut3_ref(weights.data(), qlut3, qr_seq, qi_seq, lut_scales3, k, m, dst_lut3.data());
#endif

    std::vector<float> dst_lut((size_t) m, 0.f);
    ggml_ifairy_qgemm_lut_ref(weights.data(), qlut_r, qlut_i, lut_scales, k, m, dst_lut.data());

    // High-precision reference by dequantizing to float and applying conj(activation)
    std::vector<float> dst_ref((size_t) m * 2, 0.f);
    for (int row = 0; row < m; ++row) {
        const block_ifairy * w_row = weights.data() + (size_t) row * blocks_per_row;
        double acc_r = 0.0;
        double acc_i = 0.0;
        const double d_ar = GGML_FP16_TO_FP32(act.d_real);
        const double d_ai = GGML_FP16_TO_FP32(act.d_imag);

        for (int j = 0; j < k; ++j) {
            const int chunk    = j >> 6;
            const int lane     = j & 0xF;
            const int part     = (j >> 4) & 0x3;
            const int byte_idx = (chunk << 4) + lane;
            const int bit_off  = part * 2;
            const uint8_t code = (w_row[0].qs[byte_idx] >> bit_off) & 0x3;
            int wr = 0, wi = 0;
            switch (code) {
                case 0: wr = -1; wi = 0; break;
                case 1: wr =  1; wi = 0; break;
                case 2: wr =  0; wi = -1; break;
                case 3: wr =  0; wi =  1; break;
            }
            const double w_real = wr * GGML_FP16_TO_FP32(w_row[0].d_real);
            const double w_imag = wi * GGML_FP16_TO_FP32(w_row[0].d_imag);

            const double ar = (double) (int8_t) act.x_real[j] * d_ar;
            const double ai = (double) (int8_t) act.x_imag[j] * d_ai;

            acc_r += w_real * ar + w_imag * ai;
            acc_i += w_imag * ar - w_real * ai;
        }

        dst_ref[(size_t) row * 2 + 0] = (float) acc_r;
        dst_ref[(size_t) row * 2 + 1] = (float) acc_i;
    }

    const float rel = 2e-2f;
    for (int row = 0; row < m; ++row) {
        const ggml_bf16_t * packed = (const ggml_bf16_t *) &dst_lut[row];
        const float lut_r = GGML_BF16_TO_FP32(packed[0]);
        const float lut_i = GGML_BF16_TO_FP32(packed[1]);
        const float dr = std::abs(lut_r - dst_ref[(size_t) row * 2 + 0]);
        const float di = std::abs(lut_i - dst_ref[(size_t) row * 2 + 1]);
        const float thr_r = rel * std::max(std::abs(dst_ref[(size_t) row * 2 + 0]), 1.0f);
        const float thr_i = rel * std::max(std::abs(dst_ref[(size_t) row * 2 + 1]), 1.0f);
        if (dr > thr_r || di > thr_i) {
            std::cerr << "Mismatch at row " << row << " dr=" << dr << " di=" << di
                      << " thr_r=" << thr_r << " thr_i=" << thr_i << std::endl;
            ggml_aligned_free(workspace, workspace_bytes);
            return 1;
        }
    }

#if defined(GGML_IFAIRY_ARM_LUT_3W)
    const float rel3w = 4e-2f;
    for (int row = 0; row < m; ++row) {
        const ggml_bf16_t * packed = (const ggml_bf16_t *) &dst_lut3[row];
        const float lut_r = GGML_BF16_TO_FP32(packed[0]);
        const float lut_i = GGML_BF16_TO_FP32(packed[1]);
        const float dr = std::abs(lut_r - dst_ref[(size_t) row * 2 + 0]);
        const float di = std::abs(lut_i - dst_ref[(size_t) row * 2 + 1]);
        const float thr_r = rel3w * std::max(std::abs(dst_ref[(size_t) row * 2 + 0]), 1.0f);
        const float thr_i = rel3w * std::max(std::abs(dst_ref[(size_t) row * 2 + 1]), 1.0f);
        if (dr > thr_r || di > thr_i) {
            std::cerr << "3W mismatch at row " << row << " dr=" << dr << " di=" << di
                      << " thr_r=" << thr_r << " thr_i=" << thr_i << std::endl;
            ggml_aligned_free(workspace3, workspace3_bytes);
            ggml_aligned_free(workspace, workspace_bytes);
            return 1;
        }
    }
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    std::vector<float> dst_neon((size_t) m, 0.f);
    ggml_ifairy_qgemm_lut_neon(weights.data(), qlut_r, qlut_i, lut_scales, k, m, dst_neon.data());

    for (int row = 0; row < m; ++row) {
        const ggml_bf16_t * packed = (const ggml_bf16_t *) &dst_neon[row];
        const float lut_r = GGML_BF16_TO_FP32(packed[0]);
        const float lut_i = GGML_BF16_TO_FP32(packed[1]);
        const float dr = std::abs(lut_r - dst_ref[(size_t) row * 2 + 0]);
        const float di = std::abs(lut_i - dst_ref[(size_t) row * 2 + 1]);
        const float thr_r = rel * std::max(std::abs(dst_ref[(size_t) row * 2 + 0]), 1.0f);
        const float thr_i = rel * std::max(std::abs(dst_ref[(size_t) row * 2 + 1]), 1.0f);
        if (dr > thr_r || di > thr_i) {
            std::cerr << "NEON mismatch at row " << row << " dr=" << dr << " di=" << di
                      << " thr_r=" << thr_r << " thr_i=" << thr_i << std::endl;
            ggml_aligned_free(workspace, workspace_bytes);
            return 1;
        }
    }
#endif

    const char * bench_env = std::getenv("IFAIRY_LUT_BENCH");
    if (bench_env && bench_env[0] != '\0') {
        const int iters = 5000;
        float * dst_tmp = (float *) ggml_aligned_malloc(sizeof(float) * (size_t) m);
        assert(dst_tmp != nullptr);

        auto bench = [&](auto fn) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                fn();
            }
            const auto end = std::chrono::steady_clock::now();
            const double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            return us / iters;
        };

        const double ref_us = bench([&]() {
            ggml_ifairy_qgemm_lut_ref(weights.data(), qlut_r, qlut_i, lut_scales, k, m, dst_tmp);
        });

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        const double neon_us = bench([&]() {
            ggml_ifairy_qgemm_lut_neon(weights.data(), qlut_r, qlut_i, lut_scales, k, m, dst_tmp);
        });
        std::cout << "bench(us/call): ref=" << ref_us << " neon=" << neon_us << std::endl;
#else
        std::cout << "bench(us/call): ref=" << ref_us << " neon=N/A" << std::endl;
#endif

        ggml_aligned_free(dst_tmp, sizeof(float) * (size_t) m);
    }

#if defined(GGML_IFAIRY_ARM_LUT_3W)
    ggml_aligned_free(workspace3, workspace3_bytes);
#endif
    ggml_aligned_free(workspace, workspace_bytes);
    std::cout << "ok\n";
    return 0;
}
