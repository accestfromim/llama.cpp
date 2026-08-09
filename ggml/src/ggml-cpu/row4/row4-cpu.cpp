#include "row4-cpu.h"

#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <set>
#include <tuple>

namespace {

constexpr int64_t row4_layout_version = 1;
constexpr int64_t row4_tile_k         = 128;
constexpr int64_t row4_tile_o         = 16;
constexpr int64_t row4_panel_rows     = 8;
constexpr size_t  row4_cache_line     = 64;

enum class row4_path {
    scalar,
    dotprod,
    i8mm,
};

constexpr int8_t row4_codebook[4][16] = {
    { 2, 0,  1,  1,  0, -2, -1, -1, 1, -1, 0, 0, 1,  -1, 0,  0  },
    { 0, 0,  -1, 1,  0, 0,  -1, 1,  1, 1,  0, 2, -1, -1, -2, 0  },
    { 0, 0,  1,  -1, 0, 0,  1,  -1, 1, 1,  2, 0, -1, -1, 0,  -2 },
    { 0, -2, -1, -1, 2, 0,  1,  1,  1, -1, 0, 0, 1,  -1, 0,  0  },
};

static size_t align_cache_line(size_t n) {
    return (n + row4_cache_line - 1) & ~(row4_cache_line - 1);
}

static int64_t align_panel_tokens(int64_t tokens) {
    return (tokens + 7) & ~INT64_C(7);
}

static bool env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static const char * path_name(row4_path path) {
    switch (path) {
        case row4_path::scalar:
            return "scalar";
        case row4_path::dotprod:
            return "dotprod";
        case row4_path::i8mm:
            return "i8mm";
    }
    return "unknown";
}

static void log_path_marker_once(const struct ggml_compute_params * params,
                                 const struct ggml_tensor *         dst,
                                 row4_path                          path,
                                 int64_t                            tokens,
                                 int64_t                            k) {
    if (params->ith != 0 || !env_enabled("GGML_ROW4_CPU_DEBUG")) {
        return;
    }

    const bool panel = path == row4_path::i8mm && tokens >= 8;
    using marker_key = std::tuple<int, int, int64_t, int64_t, int64_t, bool>;
    const marker_key key{ (int) dst->op, (int) path, tokens, dst->ne[0], k, panel };

    static std::mutex                 marker_mutex;
    static std::set<marker_key>       seen_markers;
    const std::lock_guard<std::mutex> lock(marker_mutex);
    if (!seen_markers.insert(key).second) {
        return;
    }

    const char * layout = dst->op == GGML_OP_ROW4_LINEAR ? "m16k128_split8_v1" : "s8_m16k128_rowmajor_v1";
    const char * aqpack = panel ? "bf16_rne_a8_away_pairk8_v1" : "bf16_rne_a8_away_v1";
    GGML_LOG_INFO(
        "row4_cpu: op=%s path=%s layout=%s B=%lld O=%lld K=%lld nth=%d "
        "aqpack=%s panel=%d prepack=0\n",
        dst->op == GGML_OP_ROW4_LINEAR ? "row4" : "w8a8", path_name(path), layout, (long long) tokens,
        (long long) dst->ne[0], (long long) k, params->nth, aqpack, panel ? 1 : 0);
}

static bool has_dotprod() {
#if defined(__ARM_FEATURE_DOTPROD) && defined(__aarch64__)
    return true;
#else
    return false;
#endif
}

static bool has_i8mm() {
#if defined(__ARM_FEATURE_MATMUL_INT8) && defined(__aarch64__)
    return true;
#else
    return false;
#endif
}

static bool select_path(const struct ggml_tensor * dst, row4_path * path) {
    const char * forced = std::getenv("GGML_ROW4_TEST_FORCE_PATH");
    if (forced && forced[0] != '\0') {
        if (std::strcmp(forced, "scalar") == 0) {
            *path = row4_path::scalar;
            return true;
        }
        if (std::strcmp(forced, "dotprod") == 0) {
            if (!has_dotprod()) {
                GGML_LOG_ERROR("row4_cpu: forced dotprod path is unavailable in this CPU build\n");
                return false;
            }
            *path = row4_path::dotprod;
            return true;
        }
        if (std::strcmp(forced, "i8mm") == 0) {
            if (!has_i8mm()) {
                GGML_LOG_ERROR("row4_cpu: forced i8mm path is unavailable in this CPU build\n");
                return false;
            }
            *path = row4_path::i8mm;
            return true;
        }
        GGML_LOG_ERROR("row4_cpu: invalid GGML_ROW4_TEST_FORCE_PATH=%s\n", forced);
        return false;
    }

    const int64_t tokens = ggml_nrows(dst->src[0]);
    if (tokens == 1 && has_dotprod()) {
        *path = row4_path::dotprod;
    } else if (has_i8mm()) {
        *path = row4_path::i8mm;
    } else if (has_dotprod()) {
        *path = row4_path::dotprod;
    } else {
        GGML_LOG_ERROR("row4_cpu: no ARM dotprod/i8mm path is available; scalar must be forced for tests\n");
        return false;
    }
    return true;
}

static float bf16_round_trip(float value) {
    return GGML_BF16_TO_FP32(GGML_FP32_TO_BF16(value));
}

static int8_t quantize_activation(float value, float scale) {
    const float magnitude = std::floor(std::fabs(value / scale) + 0.5f);
    const float rounded   = std::copysign(magnitude, value);
    return (int8_t) std::max(-127.0f, std::min(127.0f, rounded));
}

static void quantize_activation_row(const float * src, int8_t * dst, int64_t k, float * scale) {
    float amax = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        amax = std::max(amax, std::fabs(bf16_round_trip(src[i])));
    }

    *scale = std::max(amax / 127.0f, 1.0e-8f);
    for (int64_t i = 0; i < k; ++i) {
        dst[i] = quantize_activation(bf16_round_trip(src[i]), *scale);
    }
}

static void quantize_activation_panel_row(const float * src, int8_t * dst, int64_t token, int64_t k, float * scale) {
    float amax = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        amax = std::max(amax, std::fabs(bf16_round_trip(src[i])));
    }

    *scale                  = std::max(amax / 127.0f, 1.0e-8f);
    int8_t *      pair      = dst + (token / 2) * (2 * k);
    const int64_t token_off = (token % 2) * 8;
    for (int64_t ik = 0; ik < k; ik += 8) {
        int8_t * packed = pair + (ik / 8) * 16 + token_off;
        for (int lane = 0; lane < 8; ++lane) {
            packed[lane] = quantize_activation(bf16_round_trip(src[ik + lane]), *scale);
        }
    }
}

static void zero_activation_panel_row(int8_t * dst, int64_t token, int64_t k) {
    int8_t *      pair      = dst + (token / 2) * (2 * k);
    const int64_t token_off = (token % 2) * 8;
    for (int64_t ik = 0; ik < k; ik += 8) {
        std::memset(pair + (ik / 8) * 16 + token_off, 0, 8);
    }
}

static int8_t row4_weight_at(const struct ggml_tensor * codes, int64_t o, int64_t k) {
    const int64_t ot      = o / row4_tile_o;
    const int64_t row     = o % row4_tile_o;
    const int64_t group   = row / 4;
    const int64_t channel = row % 4;
    const int64_t kt      = k / row4_tile_k;
    const int64_t inner   = k % row4_tile_k;
    const int64_t split   = inner / 16;
    const int64_t lane    = inner % 16;
    const int64_t byte    = split * 8 + lane % 8;

    const uint8_t packed = *(const uint8_t *) ((const char *) codes->data + ot * codes->nb[3] + kt * codes->nb[2] +
                                               group * codes->nb[1] + byte);
    const uint8_t code   = lane < 8 ? packed & 0x0f : packed >> 4;
    return row4_codebook[channel][code];
}

static int8_t w8_weight_at(const struct ggml_tensor * codes, int64_t o, int64_t k) {
    const int64_t ot    = o / row4_tile_o;
    const int64_t row   = o % row4_tile_o;
    const int64_t kt    = k / row4_tile_k;
    const int64_t inner = k % row4_tile_k;
    return *(const int8_t *) ((const char *) codes->data + ot * codes->nb[3] + kt * codes->nb[2] + row * codes->nb[1] +
                              inner);
}

static float weight_scale_at(const struct ggml_tensor * scales, int64_t o) {
    if (scales->type == GGML_TYPE_BF16) {
        const ggml_bf16_t value = *(const ggml_bf16_t *) ((const char *) scales->data + o * scales->nb[0]);
        return GGML_BF16_TO_FP32(value);
    }
    return *(const float *) ((const char *) scales->data + o * scales->nb[0]);
}

static void store_result(struct ggml_tensor *       dst,
                         const struct ggml_tensor * scales,
                         const float *              activation_scales,
                         int64_t                    token,
                         int64_t                    o,
                         int32_t                    accumulator) {
    volatile float activation_scaled              = (float) accumulator * activation_scales[token];
    const float    scaled                         = activation_scaled * weight_scale_at(scales, o);
    ((float *) dst->data)[token * dst->ne[0] + o] = bf16_round_trip(scaled);
}

static void compute_scalar(const struct ggml_compute_params * params,
                           struct ggml_tensor *               dst,
                           const int8_t *                     activations,
                           const float *                      activation_scales) {
    const struct ggml_tensor * codes  = dst->src[1];
    const struct ggml_tensor * scales = dst->src[2];
    const int64_t              k      = dst->src[0]->ne[0];
    const int64_t              o      = dst->ne[0];
    const int64_t              tokens = ggml_nrows(dst->src[0]);
    const bool                 row4   = dst->op == GGML_OP_ROW4_LINEAR;
    const int64_t              total  = tokens * o;
    const int64_t              begin  = total * params->ith / params->nth;
    const int64_t              end    = total * (params->ith + 1) / params->nth;

    for (int64_t index = begin; index < end; ++index) {
        const int64_t  token       = index / o;
        const int64_t  row         = index % o;
        const int8_t * qx          = activations + token * k;
        int32_t        accumulator = 0;
        for (int64_t ik = 0; ik < k; ++ik) {
            const int8_t weight = row4 ? row4_weight_at(codes, row, ik) : w8_weight_at(codes, row, ik);
            accumulator += (int32_t) weight * qx[ik];
        }
        store_result(dst, scales, activation_scales, token, row, accumulator);
    }
}

#if defined(__ARM_FEATURE_DOTPROD) && defined(__aarch64__)

static void compute_row4_dotprod(const struct ggml_compute_params * params,
                                 struct ggml_tensor *               dst,
                                 const int8_t *                     activations,
                                 const float *                      activation_scales,
                                 std::atomic<int64_t> *             next_job) {
    const struct ggml_tensor * codes     = dst->src[1];
    const struct ggml_tensor * scales    = dst->src[2];
    const int64_t              k         = dst->src[0]->ne[0];
    const int64_t              o         = dst->ne[0];
    const int64_t              tokens    = ggml_nrows(dst->src[0]);
    const int64_t              k_tiles   = k / row4_tile_k;
    const int64_t              o_tiles   = o / row4_tile_o;
    const int64_t              jobs      = tokens * o_tiles;
    const int8x16_t            tables[4] = {
        vld1q_s8(row4_codebook[0]),
        vld1q_s8(row4_codebook[1]),
        vld1q_s8(row4_codebook[2]),
        vld1q_s8(row4_codebook[3]),
    };
    const uint8x8_t mask = vdup_n_u8(0x0f);

    const bool dynamic_jobs = tokens == 1 && next_job;
    for (int64_t job = dynamic_jobs ? next_job->fetch_add(1, std::memory_order_relaxed) : params->ith; job < jobs;
         job         = dynamic_jobs ? next_job->fetch_add(1, std::memory_order_relaxed) : job + params->nth) {
        const int64_t  token = job / o_tiles;
        const int64_t  ot    = job % o_tiles;
        const int8_t * qx    = activations + token * k;
        int32x4_t      sums[16];
        for (int row = 0; row < 16; ++row) {
            sums[row] = vdupq_n_s32(0);
        }

        for (int64_t kt = 0; kt < k_tiles; ++kt) {
            for (int split = 0; split < 8; ++split) {
                const int8x16_t q = vld1q_s8(qx + kt * row4_tile_k + split * 16);
                for (int group = 0; group < 4; ++group) {
                    const uint8_t *  packed  = (const uint8_t *) codes->data + ot * codes->nb[3] + kt * codes->nb[2] +
                                               group * codes->nb[1] + split * 8;
                    const uint8x8_t  bytes   = vld1_u8(packed);
                    const uint8x16_t indexes = vcombine_u8(vand_u8(bytes, mask), vshr_n_u8(bytes, 4));
                    for (int channel = 0; channel < 4; ++channel) {
                        const int8x16_t weights   = vqtbl1q_s8(tables[channel], indexes);
                        sums[group * 4 + channel] = vdotq_s32(sums[group * 4 + channel], weights, q);
                    }
                }
            }
        }

        for (int row = 0; row < 16; ++row) {
            store_result(dst, scales, activation_scales, token, ot * 16 + row, vaddvq_s32(sums[row]));
        }
    }
}

static void compute_w8_dotprod(const struct ggml_compute_params * params,
                               struct ggml_tensor *               dst,
                               const int8_t *                     activations,
                               const float *                      activation_scales,
                               std::atomic<int64_t> *             next_job) {
    const struct ggml_tensor * codes   = dst->src[1];
    const struct ggml_tensor * scales  = dst->src[2];
    const int64_t              k       = dst->src[0]->ne[0];
    const int64_t              o       = dst->ne[0];
    const int64_t              tokens  = ggml_nrows(dst->src[0]);
    const int64_t              k_tiles = k / row4_tile_k;
    const int64_t              o_tiles = o / row4_tile_o;
    const int64_t              jobs    = tokens * o_tiles;

    const bool dynamic_jobs = tokens == 1 && next_job;
    for (int64_t job = dynamic_jobs ? next_job->fetch_add(1, std::memory_order_relaxed) : params->ith; job < jobs;
         job         = dynamic_jobs ? next_job->fetch_add(1, std::memory_order_relaxed) : job + params->nth) {
        const int64_t  token = job / o_tiles;
        const int64_t  ot    = job % o_tiles;
        const int8_t * qx    = activations + token * k;
        int32x4_t      sums[16];
        for (int row = 0; row < 16; ++row) {
            sums[row] = vdupq_n_s32(0);
        }

        for (int64_t kt = 0; kt < k_tiles; ++kt) {
            for (int ik = 0; ik < 128; ik += 16) {
                const int8x16_t q = vld1q_s8(qx + kt * row4_tile_k + ik);
                for (int row = 0; row < 16; ++row) {
                    const int8_t * weight = (const int8_t *) ((const char *) codes->data + ot * codes->nb[3] +
                                                              kt * codes->nb[2] + row * codes->nb[1] + ik);
                    sums[row]             = vdotq_s32(sums[row], vld1q_s8(weight), q);
                }
            }
        }

        for (int row = 0; row < 16; ++row) {
            store_result(dst, scales, activation_scales, token, ot * 16 + row, vaddvq_s32(sums[row]));
        }
    }
}

#endif

static bool compute_dotprod(const struct ggml_compute_params * params,
                            struct ggml_tensor *               dst,
                            const int8_t *                     activations,
                            const float *                      activation_scales,
                            std::atomic<int64_t> *             next_job) {
#if defined(__ARM_FEATURE_DOTPROD) && defined(__aarch64__)
    if (dst->op == GGML_OP_ROW4_LINEAR) {
        compute_row4_dotprod(params, dst, activations, activation_scales, next_job);
    } else {
        compute_w8_dotprod(params, dst, activations, activation_scales, next_job);
    }
    return true;
#else
    GGML_UNUSED(params);
    GGML_UNUSED(dst);
    GGML_UNUSED(activations);
    GGML_UNUSED(activation_scales);
    GGML_UNUSED(next_job);
    return false;
#endif
}

#if defined(__ARM_FEATURE_MATMUL_INT8) && defined(__aarch64__)

static void decode_row4_group_16(const struct ggml_tensor * codes, int64_t row0, int64_t k0, int8x16_t weights[4]) {
    const int64_t   ot    = row0 / row4_tile_o;
    const int64_t   group = (row0 % row4_tile_o) / 4;
    const int64_t   kt    = k0 / row4_tile_k;
    const int64_t   split = (k0 % row4_tile_k) / 16;
    const uint8_t * packed =
        (const uint8_t *) codes->data + ot * codes->nb[3] + kt * codes->nb[2] + group * codes->nb[1] + split * 8;
    const uint8x8_t  bytes   = vld1_u8(packed);
    const uint8x8_t  mask    = vdup_n_u8(0x0f);
    const uint8x16_t indexes = vcombine_u8(vand_u8(bytes, mask), vshr_n_u8(bytes, 4));
    for (int channel = 0; channel < 4; ++channel) {
        weights[channel] = vqtbl1q_s8(vld1q_s8(row4_codebook[channel]), indexes);
    }
}

static int8x16_t load_w8_row_16(const struct ggml_tensor * codes, int64_t row, int64_t k0) {
    const int64_t  ot        = row / row4_tile_o;
    const int64_t  inner_row = row % row4_tile_o;
    const int64_t  kt        = k0 / row4_tile_k;
    const int64_t  inner     = k0 % row4_tile_k;
    const int8_t * ptr       = (const int8_t *) ((const char *) codes->data + ot * codes->nb[3] + kt * codes->nb[2] +
                                                 inner_row * codes->nb[1] + inner);
    return vld1q_s8(ptr);
}

static void compute_i8mm_direct(const struct ggml_compute_params * params,
                                struct ggml_tensor *               dst,
                                const int8_t *                     activations,
                                const float *                      activation_scales) {
    const struct ggml_tensor * codes       = dst->src[1];
    const struct ggml_tensor * scales      = dst->src[2];
    const int64_t              k           = dst->src[0]->ne[0];
    const int64_t              o           = dst->ne[0];
    const int64_t              tokens      = ggml_nrows(dst->src[0]);
    const bool                 row4        = dst->op == GGML_OP_ROW4_LINEAR;
    const int64_t              token_pairs = (tokens + 1) / 2;
    const int64_t              row_groups  = o / 4;
    const int64_t              jobs        = token_pairs * row_groups;

    for (int64_t job = params->ith; job < jobs; job += params->nth) {
        const int64_t token0     = 2 * (job / row_groups);
        const int64_t row0       = 4 * (job % row_groups);
        const bool    has_token1 = token0 + 1 < tokens;
        int32x4_t     sums[2]    = { vdupq_n_s32(0), vdupq_n_s32(0) };

        for (int64_t ik = 0; ik < k; ik += 16) {
            const int8x16_t input0 = vld1q_s8(activations + token0 * k + ik);
            const int8x16_t input1 = has_token1 ? vld1q_s8(activations + (token0 + 1) * k + ik) : vdupq_n_s8(0);
            int8x16_t       weights[4];
            if (row4) {
                decode_row4_group_16(codes, row0, ik, weights);
            } else {
                for (int row = 0; row < 4; ++row) {
                    weights[row] = load_w8_row_16(codes, row0 + row, ik);
                }
            }
            const int8x16_t input_lo = vcombine_s8(vget_low_s8(input0), vget_low_s8(input1));
            const int8x16_t input_hi = vcombine_s8(vget_high_s8(input0), vget_high_s8(input1));
            for (int row_pair = 0; row_pair < 2; ++row_pair) {
                const int row  = 2 * row_pair;
                sums[row_pair] = vmmlaq_s32(sums[row_pair], input_lo,
                                            vcombine_s8(vget_low_s8(weights[row]), vget_low_s8(weights[row + 1])));
                sums[row_pair] = vmmlaq_s32(sums[row_pair], input_hi,
                                            vcombine_s8(vget_high_s8(weights[row]), vget_high_s8(weights[row + 1])));
            }
        }

        for (int row_pair = 0; row_pair < 2; ++row_pair) {
            const int row = 2 * row_pair;
            store_result(dst, scales, activation_scales, token0, row0 + row, vgetq_lane_s32(sums[row_pair], 0));
            store_result(dst, scales, activation_scales, token0, row0 + row + 1, vgetq_lane_s32(sums[row_pair], 1));
            if (has_token1) {
                store_result(dst, scales, activation_scales, token0 + 1, row0 + row, vgetq_lane_s32(sums[row_pair], 2));
                store_result(dst, scales, activation_scales, token0 + 1, row0 + row + 1,
                             vgetq_lane_s32(sums[row_pair], 3));
            }
        }
    }
}

static void expand_four_rows(const struct ggml_tensor * codes, bool row4, int64_t row0, int64_t k, int8_t * panel) {
    for (int64_t ik = 0; ik < k; ik += 16) {
        int8x16_t weights[4];
        if (row4) {
            decode_row4_group_16(codes, row0, ik, weights);
        } else {
            for (int row = 0; row < 4; ++row) {
                weights[row] = load_w8_row_16(codes, row0 + row, ik);
            }
        }

        for (int half = 0; half < 2; ++half) {
            const int64_t  panel_offset = ((ik / 8) + half) * 32;
            const int8x8_t weight0      = half == 0 ? vget_low_s8(weights[0]) : vget_high_s8(weights[0]);
            const int8x8_t weight1      = half == 0 ? vget_low_s8(weights[1]) : vget_high_s8(weights[1]);
            const int8x8_t weight2      = half == 0 ? vget_low_s8(weights[2]) : vget_high_s8(weights[2]);
            const int8x8_t weight3      = half == 0 ? vget_low_s8(weights[3]) : vget_high_s8(weights[3]);
            vst1q_s8(panel + panel_offset, vcombine_s8(weight0, weight1));
            vst1q_s8(panel + panel_offset + 16, vcombine_s8(weight2, weight3));
        }
    }
}

static void compute_i8mm_panel(struct ggml_tensor *               dst,
                               const int8_t *                     activations,
                               const float *                      activation_scales,
                               int8_t *                           panel,
                               std::atomic<int64_t> *             next_job) {
    const struct ggml_tensor * codes  = dst->src[1];
    const struct ggml_tensor * scales = dst->src[2];
    const int64_t              k      = dst->src[0]->ne[0];
    const int64_t              o      = dst->ne[0];
    const int64_t              tokens = ggml_nrows(dst->src[0]);
    const bool                 row4   = dst->op == GGML_OP_ROW4_LINEAR;
    const int64_t              o_tiles = o / row4_tile_o;

    GGML_ASSERT(next_job != nullptr);
    for (int64_t ot = next_job->fetch_add(1, std::memory_order_relaxed); ot < o_tiles;
         ot         = next_job->fetch_add(1, std::memory_order_relaxed)) {
        for (int group = 0; group < 4; group += 2) {
            const int64_t row0 = ot * row4_tile_o + group * 4;
            int8_t *      panel_hi = panel + 4 * k;
            expand_four_rows(codes, row4, row0, k, panel);
            expand_four_rows(codes, row4, row0 + 4, k, panel_hi);

            // Keep every accumulator's K traversal unchanged, but interleave four
            // token pairs and four row pairs so each activation panel load feeds
            // eight output rows.
            for (int64_t token0 = 0; token0 < tokens; token0 += 8) {
                int32x4_t sums[4][4];
                for (int pair = 0; pair < 4; ++pair) {
                    for (int row_pair = 0; row_pair < 4; ++row_pair) {
                        sums[pair][row_pair] = vdupq_n_s32(0);
                    }
                }

                for (int64_t ik = 0; ik < k; ik += 8) {
                    const int64_t   k8_offset = (ik / 8) * 16;
                    const int8x16_t weights01 = vld1q_s8(panel + (ik / 8) * 32);
                    const int8x16_t weights23 = vld1q_s8(panel + (ik / 8) * 32 + 16);
                    const int8x16_t weights45 = vld1q_s8(panel_hi + (ik / 8) * 32);
                    const int8x16_t weights67 = vld1q_s8(panel_hi + (ik / 8) * 32 + 16);
                    const int8_t *  inputs    = activations + (token0 / 2) * (2 * k) + k8_offset;
                    const int8x16_t input01   = vld1q_s8(inputs + 0 * (2 * k));
                    const int8x16_t input23   = vld1q_s8(inputs + 1 * (2 * k));
                    const int8x16_t input45   = vld1q_s8(inputs + 2 * (2 * k));
                    const int8x16_t input67   = vld1q_s8(inputs + 3 * (2 * k));

                    sums[0][0] = vmmlaq_s32(sums[0][0], input01, weights01);
                    sums[0][1] = vmmlaq_s32(sums[0][1], input01, weights23);
                    sums[0][2] = vmmlaq_s32(sums[0][2], input01, weights45);
                    sums[0][3] = vmmlaq_s32(sums[0][3], input01, weights67);
                    sums[1][0] = vmmlaq_s32(sums[1][0], input23, weights01);
                    sums[1][1] = vmmlaq_s32(sums[1][1], input23, weights23);
                    sums[1][2] = vmmlaq_s32(sums[1][2], input23, weights45);
                    sums[1][3] = vmmlaq_s32(sums[1][3], input23, weights67);
                    sums[2][0] = vmmlaq_s32(sums[2][0], input45, weights01);
                    sums[2][1] = vmmlaq_s32(sums[2][1], input45, weights23);
                    sums[2][2] = vmmlaq_s32(sums[2][2], input45, weights45);
                    sums[2][3] = vmmlaq_s32(sums[2][3], input45, weights67);
                    sums[3][0] = vmmlaq_s32(sums[3][0], input67, weights01);
                    sums[3][1] = vmmlaq_s32(sums[3][1], input67, weights23);
                    sums[3][2] = vmmlaq_s32(sums[3][2], input67, weights45);
                    sums[3][3] = vmmlaq_s32(sums[3][3], input67, weights67);
                }

                for (int pair = 0; pair < 4; ++pair) {
                    const int64_t token = token0 + 2 * pair;
                    if (token >= tokens) {
                        break;
                    }
                    for (int row_pair = 0; row_pair < 4; ++row_pair) {
                        const int row = 2 * row_pair;
                        store_result(dst, scales, activation_scales, token, row0 + row,
                                     vgetq_lane_s32(sums[pair][row_pair], 0));
                        store_result(dst, scales, activation_scales, token, row0 + row + 1,
                                     vgetq_lane_s32(sums[pair][row_pair], 1));
                        if (token + 1 < tokens) {
                            store_result(dst, scales, activation_scales, token + 1, row0 + row,
                                         vgetq_lane_s32(sums[pair][row_pair], 2));
                            store_result(dst, scales, activation_scales, token + 1, row0 + row + 1,
                                         vgetq_lane_s32(sums[pair][row_pair], 3));
                        }
                    }
                }
            }
        }
    }
}

#endif

static bool compute_i8mm(const struct ggml_compute_params * params,
                         struct ggml_tensor *               dst,
                         const int8_t *                     activations,
                         const float *                      activation_scales,
                         int8_t *                           panel,
                         std::atomic<int64_t> *             next_job) {
#if defined(__ARM_FEATURE_MATMUL_INT8) && defined(__aarch64__)
    if (ggml_nrows(dst->src[0]) >= 8) {
        compute_i8mm_panel(dst, activations, activation_scales, panel, next_job);
    } else {
        compute_i8mm_direct(params, dst, activations, activation_scales);
    }
    return true;
#else
    GGML_UNUSED(params);
    GGML_UNUSED(dst);
    GGML_UNUSED(activations);
    GGML_UNUSED(activation_scales);
    GGML_UNUSED(panel);
    GGML_UNUSED(next_job);
    return false;
#endif
}

static bool validate_op(const struct ggml_tensor * dst) {
    if (!dst || (dst->op != GGML_OP_ROW4_LINEAR && dst->op != GGML_OP_W8A8_LINEAR)) {
        return false;
    }

    const struct ggml_tensor * x      = dst->src[0];
    const struct ggml_tensor * codes  = dst->src[1];
    const struct ggml_tensor * scales = dst->src[2];
    if (!x || !codes || !scales || x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    for (int i = 3; i < GGML_MAX_SRC; ++i) {
        if (dst->src[i]) {
            return false;
        }
    }
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(codes) || !ggml_is_contiguous(scales) ||
        !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t version = ggml_get_op_params_i32(dst, 0);
    const int64_t o       = ggml_get_op_params_i32(dst, 1);
    const int64_t k       = ggml_get_op_params_i32(dst, 2);
    if (version != row4_layout_version || o <= 0 || k <= 0 || o % 128 != 0 || k % 128 != 0 || x->ne[0] != k ||
        dst->ne[0] != o || dst->ne[1] != x->ne[1] || dst->ne[2] != x->ne[2] || dst->ne[3] != x->ne[3]) {
        return false;
    }
    if (scales->ne[0] != o || scales->ne[1] != 1 || scales->ne[2] != 1 || scales->ne[3] != 1) {
        return false;
    }

    if (dst->op == GGML_OP_ROW4_LINEAR) {
        return codes->type == GGML_TYPE_ROW4_CODES && scales->type == GGML_TYPE_BF16 && codes->ne[0] == 64 &&
               codes->ne[1] == 4 && codes->ne[2] == k / 128 && codes->ne[3] == o / 16;
    }
    return codes->type == GGML_TYPE_I8 && scales->type == GGML_TYPE_F32 && codes->ne[0] == 128 && codes->ne[1] == 16 &&
           codes->ne[2] == k / 128 && codes->ne[3] == o / 16;
}

}  // namespace

extern "C" bool ggml_row4_cpu_supports_op(const struct ggml_tensor * dst) {
    if (!validate_op(dst)) {
        return false;
    }

    row4_path path;
    return select_path(dst, &path);
}

extern "C" int ggml_row4_cpu_n_tasks(const struct ggml_tensor * dst, int n_threads) {
    return validate_op(dst) ? n_threads : 0;
}

extern "C" size_t ggml_row4_cpu_work_size(const struct ggml_tensor * dst, int n_tasks) {
    if (!validate_op(dst)) {
        return 0;
    }

    row4_path path;
    if (!select_path(dst, &path)) {
        return 0;
    }

    const int64_t tokens      = ggml_nrows(dst->src[0]);
    const int64_t k           = dst->src[0]->ne[0];
    const bool    uses_panel    = path == row4_path::i8mm && tokens >= 8;
    const int64_t q_tokens      = uses_panel ? align_panel_tokens(tokens) : tokens;
    const size_t  q_bytes       = align_cache_line((size_t) q_tokens * k);
    const size_t  scale_bytes = align_cache_line((size_t) tokens * sizeof(float));
    const size_t  panel_bytes   = uses_panel ? (size_t) n_tasks * align_cache_line((size_t) row4_panel_rows * k) : 0;
    const size_t counter_bytes = align_cache_line(sizeof(std::atomic<int64_t>));
    return q_bytes + scale_bytes + panel_bytes + counter_bytes;
}

extern "C" bool ggml_row4_cpu_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (!params || !validate_op(dst)) {
        return false;
    }

    row4_path path;
    if (!select_path(dst, &path)) {
        return false;
    }

    const struct ggml_tensor * x             = dst->src[0];
    const int64_t              tokens        = ggml_nrows(x);
    const int64_t              k             = x->ne[0];
    const bool                 uses_panel    = path == row4_path::i8mm && tokens >= 8;
    const int64_t              q_tokens      = uses_panel ? align_panel_tokens(tokens) : tokens;
    const size_t               q_bytes       = align_cache_line((size_t) q_tokens * k);
    const size_t               scale_bytes   = align_cache_line((size_t) tokens * sizeof(float));
    const size_t               panel_stride  = align_cache_line((size_t) row4_panel_rows * k);
    const size_t               panel_bytes   = uses_panel ? (size_t) params->nth * panel_stride : 0;
    const size_t               counter_bytes = align_cache_line(sizeof(std::atomic<int64_t>));
    const size_t               required      = q_bytes + scale_bytes + panel_bytes + counter_bytes;
    if (!params->wdata || params->wsize < required) {
        GGML_LOG_ERROR("row4_cpu: insufficient workspace: have=%zu need=%zu\n", params->wsize, required);
        return false;
    }

    int8_t * activations       = (int8_t *) params->wdata;
    float *  activation_scales = (float *) ((char *) params->wdata + q_bytes);
    const int64_t token_begin = tokens * params->ith / params->nth;
    const int64_t token_end   = tokens * (params->ith + 1) / params->nth;

    const bool uses_dynamic_jobs = (path == row4_path::dotprod && tokens == 1) || uses_panel;
    std::atomic<int64_t> * next_job = nullptr;
    if (uses_dynamic_jobs) {
        next_job =
            reinterpret_cast<std::atomic<int64_t> *>((char *) params->wdata + q_bytes + scale_bytes + panel_bytes);
        GGML_ASSERT(((uintptr_t) next_job % alignof(std::atomic<int64_t>)) == 0);
        if (params->ith == 0) {
            new (next_job) std::atomic<int64_t>();
            next_job->store(0, std::memory_order_relaxed);
        }
    }

    for (int64_t token = token_begin; token < token_end; ++token) {
        const float * src = (const float *) x->data + token * k;
        if (uses_panel) {
            quantize_activation_panel_row(src, activations, token, k, activation_scales + token);
        } else {
            quantize_activation_row(src, activations + token * k, k, activation_scales + token);
        }
    }
    if (uses_panel && params->ith == 0) {
        for (int64_t token = tokens; token < q_tokens; ++token) {
            zero_activation_panel_row(activations, token, k);
        }
    }
    ggml_barrier(params->threadpool);

    int8_t * panel =
        uses_panel ? (int8_t *) ((char *) params->wdata + q_bytes + scale_bytes + (size_t) params->ith * panel_stride) :
                     nullptr;
    switch (path) {
        case row4_path::scalar:
            compute_scalar(params, dst, activations, activation_scales);
            break;
        case row4_path::dotprod:
            if (!compute_dotprod(params, dst, activations, activation_scales, next_job)) {
                return false;
            }
            break;
        case row4_path::i8mm:
            if (!compute_i8mm(params, dst, activations, activation_scales, panel, next_job)) {
                return false;
            }
            break;
    }

    log_path_marker_once(params, dst, path, tokens, k);
    return true;
}
