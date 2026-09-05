/*
 * TurboQuant: KV cache compression via signed WHT + PolarQuant
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements GGML_TYPE_TURBO2_0 (2-bit), GGML_TYPE_TURBO3_0 (3-bit) and
 * GGML_TYPE_TURBO4_0 (4-bit) for use as --cache-type-k turboN in llama-server.
 */

#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include <assert.h>
#include <math.h>
#include <string.h>

/* Forward declarations for GGML_API symbols defined in this file (satisfies
 * -Wmissing-prototypes under upstream CI's -Werror policy). */
GGML_API void turbo_cpu_fwht_inverse(float * x, int group_size);
GGML_API void turbo_cpu_fwht_forward(float * x, int group_size);

/* Global: WHT group size for CPU quantize path (set by CPU SET_ROWS handler) */
/* Declared with GGML_API so the symbol carries dllexport/visibility, then
 * defined plainly: `GGML_API` now expands with `extern` on every path, and
 * `extern int x = 0;` is rejected under -Werror (-Wextern-initializer). */
GGML_API int turbo3_cpu_wht_group_size;
int          turbo3_cpu_wht_group_size = 0;

/* ---------- constants ---------- */

#define TURBO_D 128 /* rotation group size = head_dim (independent of block size) */

/* Optimal centroids from paper (scaled by 1/sqrt(d)) */
/* 2-bit: {±0.453, ±1.51} / sqrt(d) */
static const float CENTROIDS_2BIT[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };

/* 3-bit: Lloyd-Max for N(0, 1/128), pre-computed */
static const float CENTROIDS_3BIT[8] = { -0.190207f, -0.118786f, -0.066822f, -0.021663f,
                                         0.021663f,  0.066822f,  0.118786f,  0.190207f };

static const float CENTROIDS_4BIT[16] = { -0.241529f, -0.182877f, -0.143016f, -0.111036f, -0.083292f, -0.058050f,
                                          -0.034299f, -0.011349f, 0.011349f,  0.034299f,  0.058050f,  0.083292f,
                                          0.111036f,  0.143016f,  0.182877f,  0.241529f };

/* ---------- nearest centroid ---------- */

static int nearest_centroid_2bit(float val) {
    /* Binary search on midpoints: {-0.133, -0.040, 0.040, 0.133} */
    if (val < -0.086728f) {
        return 0; /* midpoint(-0.133, -0.040) */
    }
    if (val < 0.000000f) {
        return 1; /* midpoint(-0.040, 0.040) */
    }
    if (val < 0.086728f) {
        return 2; /* midpoint(0.040, 0.133) */
    }
    return 3;
}

static int nearest_centroid_3bit(float val) {
    /* 8 centroids, find nearest via midpoints */
    if (val < -0.154496f) {
        return 0;
    }
    if (val < -0.092804f) {
        return 1;
    }
    if (val < -0.044243f) {
        return 2;
    }
    if (val < 0.000000f) {
        return 3;
    }
    if (val < 0.044243f) {
        return 4;
    }
    if (val < 0.092804f) {
        return 5;
    }
    if (val < 0.154496f) {
        return 6;
    }
    return 7;
}

static int nearest_centroid_4bit(float val) {
    /* 16 centroids, optimal for N(0, 1/sqrt(128)), find nearest via midpoints */
    if (val < -0.212203f) {
        return 0;
    }
    if (val < -0.162947f) {
        return 1;
    }
    if (val < -0.127026f) {
        return 2;
    }
    if (val < -0.097164f) {
        return 3;
    }
    if (val < -0.070671f) {
        return 4;
    }
    if (val < -0.046174f) {
        return 5;
    }
    if (val < -0.022824f) {
        return 6;
    }
    if (val < 0.000000f) {
        return 7;
    }
    if (val < 0.022824f) {
        return 8;
    }
    if (val < 0.046174f) {
        return 9;
    }
    if (val < 0.070671f) {
        return 10;
    }
    if (val < 0.097164f) {
        return 11;
    }
    if (val < 0.127026f) {
        return 12;
    }
    if (val < 0.162947f) {
        return 13;
    }
    if (val < 0.212203f) {
        return 14;
    }
    return 15;
}

/* ---------- WHT sign arrays (must match CUDA/Metal, seed=42) ---------- */

static const float turbo_cpu_s1[128] = { -1, 1,  1,  -1, -1, 1,  -1, 1,  -1, -1, 1,  1,  1,  1,  1,  1,  1,  -1, 1,
                                         -1, 1,  -1, -1, 1,  1,  1,  -1, 1,  1,  -1, -1, -1, -1, 1,  1,  -1, 1,  1,
                                         -1, 1,  -1, 1,  1,  -1, -1, 1,  -1, 1,  1,  1,  1,  -1, -1, -1, -1, -1, 1,
                                         -1, 1,  1,  1,  1,  -1, 1,  -1, -1, 1,  -1, -1, -1, 1,  -1, -1, -1, 1,  -1,
                                         -1, -1, 1,  1,  1,  -1, -1, 1,  1,  1,  -1, -1, 1,  1,  -1, 1,  1,  -1, 1,
                                         -1, -1, 1,  1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  1,  -1, 1,  -1, 1,  1,  -1,
                                         1,  1,  -1, -1, -1, -1, -1, 1,  1,  -1, 1,  1,  -1, 1 };

static const float turbo_cpu_s2[128] = { 1,  1,  1,  1,  -1, 1,  1,  -1, 1,  -1, -1, -1, 1,  -1, -1, -1, 1,  1,  -1,
                                         -1, 1,  -1, 1,  -1, 1,  -1, -1, 1,  -1, 1,  1,  1,  1,  1,  -1, -1, -1, 1,
                                         -1, -1, -1, -1, -1, -1, 1,  1,  1,  -1, 1,  -1, 1,  1,  1,  -1, -1, 1,  -1,
                                         -1, -1, -1, -1, -1, 1,  1,  1,  -1, 1,  -1, -1, -1, -1, 1,  -1, 1,  -1, 1,
                                         -1, -1, 1,  1,  -1, 1,  -1, 1,  1,  -1, 1,  -1, -1, -1, -1, 1,  -1, -1, 1,
                                         -1, 1,  -1, 1,  1,  1,  -1, -1, 1,  -1, 1,  -1, 1,  1,  -1, -1, 1,  -1, 1,
                                         -1, 1,  1,  -1, 1,  -1, 1,  -1, -1, -1, -1, -1, 1,  -1 };

/* ---------- CPU forward WHT (in-place, group_size elements) ---------- */

GGML_API void turbo_cpu_fwht_forward(float * x, int group_size) {
    const float * s1       = turbo_cpu_s1;
    const float * s2       = turbo_cpu_s2;
    const float   inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;

    // signs1
    for (int i = 0; i < group_size; i++) {
        x[i] *= s1[i];
    }

    // butterfly stages
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // normalize + signs2
    for (int i = 0; i < group_size; i++) {
        x[i] *= inv_sqrt * s2[i];
    }
}

/* ---------- CPU inverse WHT (in-place, group_size elements) ----------
 *
 * Forward is  y = D(s2) * N * H * D(s1) * x   (N = 1/sqrt(group_size))
 * H is the unnormalized Hadamard butterfly with H*H = group_size * I, so
 * (N*H) is self-inverse.  s1 and s2 are ±1 diagonals, also self-inverse.
 * The inverse therefore has the same structure with s1 and s2 swapped:
 *     x = D(s1) * N * H * D(s2) * y
 */
GGML_API void turbo_cpu_fwht_inverse(float * x, int group_size) {
    const float * s1       = turbo_cpu_s1;
    const float * s2       = turbo_cpu_s2;
    const float   inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;

    // signs2 (undoes the s2 that was applied last in the forward pass)
    for (int i = 0; i < group_size; i++) {
        x[i] *= s2[i];
    }

    // butterfly stages (same as forward — self-inverse up to the inv_sqrt scaling below)
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // normalize + signs1
    for (int i = 0; i < group_size; i++) {
        x[i] *= inv_sqrt * s1[i];
    }
}

/* ---------- TURBO3_0: 3-bit PolarQuant with WHT rotation ---------- */

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);

    // Read WHT group size from global (set by CPU SET_ROWS handler before each call).
    // Fallback: 128 if row is 128-aligned, else 64.
    extern int turbo3_cpu_wht_group_size;
    int        group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) {
        group_size = (group_size == 128) ? 64 : 128;
    }
    assert(k % group_size == 0);

    const int n_groups         = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO3;

    for (int g = 0; g < n_groups; g++) {
        const float *    grp_src = x + g * group_size;
        block_turbo3_0 * grp_dst = y + g * blocks_per_group;

        // 1. L2 norm over the group
        float norm_sq = 0.0f;
        float buf[128];  // max group_size
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        // 2. Normalize
        for (int j = 0; j < group_size; j++) {
            buf[j] *= inv_norm;
        }

        // 3. Forward WHT rotation
        turbo_cpu_fwht_forward(buf, group_size);

        // 4. Quantize + pack into sub-blocks
        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo3_0 * blk = &grp_dst[b];
            const int        off = b * QK_TURBO3;

            memset(blk->qs, 0, QK_TURBO3 / 4);
            memset(blk->signs, 0, QK_TURBO3 / 8);

            for (int j = 0; j < QK_TURBO3; j++) {
                int idx = nearest_centroid_3bit(buf[off + j]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                if (idx & 0x4) {
                    blk->signs[j / 8] |= (1 << (j % 8));
                }
                recon_sq += CENTROIDS_3BIT[idx] * CENTROIDS_3BIT[idx];
            }
        }

        // 5. Corrected norm: grp_norm / recon_norm (matching CUDA kernel)
        float recon_norm = sqrtf(recon_sq);
        float corrected  = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    // Stub — Metal shader handles dequant on GPU.
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t low2             = (x[block].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            uint8_t hi1              = (x[block].signs[j / 8] >> (j % 8)) & 0x1;
            uint8_t idx              = low2 | (hi1 << 2);
            y[block * QK_TURBO3 + j] = CENTROIDS_3BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src,
                         void * GGML_RESTRICT        dst,
                         int64_t                     nrows,
                         int64_t                     n_per_row,
                         const float *               imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_ref(src + row * n_per_row, (block_turbo3_0 *) ((char *) dst + row * row_size), n_per_row);
    }
    return nrows * row_size;
}

/* ---------- TURBO2_0: 2-bit PolarQuant (no QJL) ---------- */

void quantize_row_turbo2_0_ref(const float * GGML_RESTRICT x, block_turbo2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);

    extern int turbo3_cpu_wht_group_size;
    int        group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) {
        group_size = (group_size == 128) ? 64 : 128;
    }
    assert(k % group_size == 0);

    const int n_groups         = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO2;

    for (int g = 0; g < n_groups; g++) {
        const float *    grp_src = x + g * group_size;
        block_turbo2_0 * grp_dst = y + g * blocks_per_group;

        /* 1. L2 norm over the group */
        float norm_sq = 0.0f;
        float buf[128];
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        /* 2. Normalize */
        for (int j = 0; j < group_size; j++) {
            buf[j] *= inv_norm;
        }

        /* 3. Forward WHT rotation */
        turbo_cpu_fwht_forward(buf, group_size);

        /* 4. Quantize + pack into sub-blocks */
        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo2_0 * blk = &grp_dst[b];
            const int        off = b * QK_TURBO2;

            memset(blk->qs, 0, QK_TURBO2 / 4);

            for (int j = 0; j < QK_TURBO2; j++) {
                int idx = nearest_centroid_2bit(buf[off + j]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                recon_sq += CENTROIDS_2BIT[idx] * CENTROIDS_2BIT[idx];
            }
        }

        /* 5. Corrected norm */
        float recon_norm = sqrtf(recon_sq);
        float corrected  = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void dequantize_row_turbo2_0(const block_turbo2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);
    const int nb = k / QK_TURBO2;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO2; j++) {
            uint8_t idx              = (x[block].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            y[block * QK_TURBO2 + j] = CENTROIDS_2BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo2_0(const float * GGML_RESTRICT src,
                         void * GGML_RESTRICT        dst,
                         int64_t                     nrows,
                         int64_t                     n_per_row,
                         const float *               imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO2 == 0);

    size_t row_size = (n_per_row / QK_TURBO2) * sizeof(block_turbo2_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo2_0_ref(src + row * n_per_row, (block_turbo2_0 *) ((char *) dst + row * row_size), n_per_row);
    }
    return nrows * row_size;
}

/* ---------- TURBO4_0: 4-bit PolarQuant ---------- */

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        /* Step 1: Extract norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            norm_sq += src[i] * src[i];
        }
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[TURBO_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) {
                normalized[i] = src[i] * inv;
            }
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        /* Step 2: Forward WHT rotation (matches CUDA set_rows) */
        float rotated[TURBO_D];
        memcpy(rotated, normalized, d * sizeof(float));
        turbo_cpu_fwht_forward(rotated, d);

        /* Step 3: 4-bit quantization (16 centroids) */
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t) nearest_centroid_4bit(rotated[i]);
        }

        /* Norm correction */
        float recon_norm_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            recon_norm_sq += CENTROIDS_4BIT[indices[i]] * CENTROIDS_4BIT[indices[i]];
        }
        float recon_norm     = sqrtf(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
        y[block].norm        = GGML_FP32_TO_FP16(corrected_norm);

        /* Pack */

        /* 4-bit PolarQuant: nibble pack into qs[64] */
        memset(y[block].qs, 0, d / 2);
        for (int i = 0; i < d; i++) {
            y[block].qs[i / 2] |= (uint8_t) ((indices[i] & 0xF) << ((i % 2) * 4));
        }
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    /* 4-bit PolarQuant: nibble unpack, centroid lookup and scale. */
    for (int block = 0; block < nb; block++) {
        float   norm = GGML_FP16_TO_FP32(x[block].norm);
        float * dst  = y + block * d;
        for (int i = 0; i < d; i++) {
            uint8_t idx = (x[block].qs[i / 2] >> ((i % 2) * 4)) & 0xF;
            dst[i]      = CENTROIDS_4BIT[idx] * norm;
        }
        /* Dequant stays in the rotated domain. */
    }
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src,
                         void * GGML_RESTRICT        dst,
                         int64_t                     nrows,
                         int64_t                     n_per_row,
                         const float *               imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(src + row * n_per_row, (block_turbo4_0 *) ((char *) dst + row * row_size), n_per_row);
    }
    return nrows * row_size;
}

static void turbo2m4_sort_positions(uint8_t * positions, int n) {
    for (int i = 1; i < n; ++i) {
        const uint8_t value = positions[i];
        int           j     = i;
        while (j > 0 && positions[j - 1] > value) {
            positions[j] = positions[j - 1];
            --j;
        }
        positions[j] = value;
    }
}

static void turbo2m4_select_scalar(const float * x, uint8_t * positions, int n) {
    uint8_t selected[QK_TURBO2M4] = { 0 };
    for (int rank = 0; rank < n; ++rank) {
        int   best       = -1;
        float best_score = -1.0f;
        for (int i = 0; i < QK_TURBO2M4; ++i) {
            const float score = fabsf(x[i]);
            if (!selected[i] && score > best_score) {
                best       = i;
                best_score = score;
            }
        }
        assert(best >= 0);
        selected[best]  = 1;
        positions[rank] = (uint8_t) best;
    }
    turbo2m4_sort_positions(positions, n);
}

static void turbo2m4_select_group(const float * x, uint8_t * positions, int n_groups) {
    uint8_t selected[QK_TURBO2M4 / 4] = { 0 };
    for (int rank = 0; rank < n_groups; ++rank) {
        int   best       = -1;
        float best_score = -1.0f;
        for (int group = 0; group < QK_TURBO2M4 / 4; ++group) {
            float score = 0.0f;
            for (int lane = 0; lane < 4; ++lane) {
                score = fmaxf(score, fabsf(x[4 * group + lane]));
            }
            if (!selected[group] && score > best_score) {
                best       = group;
                best_score = score;
            }
        }
        assert(best >= 0);
        selected[best]  = 1;
        positions[rank] = (uint8_t) best;
    }
    turbo2m4_sort_positions(positions, n_groups);
}

static void quantize_row_turbo2m4_ref_impl(const float * GGML_RESTRICT x,
                                           void * GGML_RESTRICT        vy,
                                           int64_t                     k,
                                           int                         n,
                                           int                         group,
                                           size_t                      block_size) {
    assert(k % QK_TURBO2M4 == 0);
    const int nblocks = (int) (k / QK_TURBO2M4);

    for (int block = 0; block < nblocks; ++block) {
        const float * src         = x + block * QK_TURBO2M4;
        uint8_t *     dst         = (uint8_t *) vy + block * block_size;
        uint8_t *     qs          = dst + sizeof(ggml_half);
        uint8_t *     positions   = qs + QK_TURBO2M4 / 4;
        const int     n_positions = group ? n / 4 : n;
        uint8_t *     qh          = positions + n_positions;
        float         rotated[QK_TURBO2M4];
        float         norm_sq = 0.0f;

        for (int i = 0; i < QK_TURBO2M4; ++i) {
            rotated[i] = src[i];
            norm_sq += src[i] * src[i];
        }
        const float norm     = sqrtf(norm_sq);
        const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
        for (int i = 0; i < QK_TURBO2M4; ++i) {
            rotated[i] *= inv_norm;
        }
        turbo_cpu_fwht_forward(rotated, QK_TURBO2M4);

        memset(qs, 0, QK_TURBO2M4 / 4);
        memset(qh, 0, (size_t) n / 2);
        float recon_sq = 0.0f;
        for (int i = 0; i < QK_TURBO2M4; ++i) {
            const int q = nearest_centroid_2bit(rotated[i]);
            qs[i / 4] |= (uint8_t) (q << (2 * (i % 4)));
            recon_sq += CENTROIDS_2BIT[q] * CENTROIDS_2BIT[q];
        }

        if (group) {
            turbo2m4_select_group(rotated, positions, n_positions);
            for (int slot = 0; slot < n_positions; ++slot) {
                const int base = 4 * positions[slot];
                for (int lane = 0; lane < 4; ++lane) {
                    const int i  = base + lane;
                    const int q2 = (qs[i / 4] >> (2 * (i % 4))) & 0x3;
                    const int q4 = nearest_centroid_4bit(rotated[i]);
                    recon_sq += CENTROIDS_4BIT[q4] * CENTROIDS_4BIT[q4] - CENTROIDS_2BIT[q2] * CENTROIDS_2BIT[q2];
                    qh[2 * slot + lane / 2] |= (uint8_t) (q4 << (4 * (lane % 2)));
                }
            }
        } else {
            turbo2m4_select_scalar(rotated, positions, n);
            for (int slot = 0; slot < n; ++slot) {
                const int i  = positions[slot];
                const int q2 = (qs[i / 4] >> (2 * (i % 4))) & 0x3;
                const int q4 = nearest_centroid_4bit(rotated[i]);
                recon_sq += CENTROIDS_4BIT[q4] * CENTROIDS_4BIT[q4] - CENTROIDS_2BIT[q2] * CENTROIDS_2BIT[q2];
                qh[slot / 2] |= (uint8_t) (q4 << (4 * (slot % 2)));
            }
        }

        const float     recon_norm = sqrtf(recon_sq);
        const ggml_half d          = GGML_FP32_TO_FP16(recon_norm > 1e-10f ? norm / recon_norm : norm);
        memcpy(dst, &d, sizeof(d));
    }
}

static void dequantize_row_turbo2m4_impl(const void * GGML_RESTRICT vx,
                                         float * GGML_RESTRICT      y,
                                         int64_t                    k,
                                         int                        n,
                                         int                        group,
                                         size_t                     block_size) {
    assert(k % QK_TURBO2M4 == 0);
    const int nblocks = (int) (k / QK_TURBO2M4);

    for (int block = 0; block < nblocks; ++block) {
        const uint8_t * src = (const uint8_t *) vx + block * block_size;
        ggml_half       d;
        memcpy(&d, src, sizeof(d));
        const float     norm        = GGML_FP16_TO_FP32(d);
        const uint8_t * qs          = src + sizeof(ggml_half);
        const uint8_t * positions   = qs + QK_TURBO2M4 / 4;
        const int       n_positions = group ? n / 4 : n;
        const uint8_t * qh          = positions + n_positions;
        float *         dst         = y + block * QK_TURBO2M4;

        for (int i = 0; i < QK_TURBO2M4; ++i) {
            const int q = (qs[i / 4] >> (2 * (i % 4))) & 0x3;
            dst[i]      = norm * CENTROIDS_2BIT[q];
        }
        if (group) {
            for (int slot = 0; slot < n_positions; ++slot) {
                const int base = 4 * positions[slot];
                for (int lane = 0; lane < 4; ++lane) {
                    const int q      = (qh[2 * slot + lane / 2] >> (4 * (lane % 2))) & 0xf;
                    dst[base + lane] = norm * CENTROIDS_4BIT[q];
                }
            }
        } else {
            for (int slot = 0; slot < n; ++slot) {
                const int q          = (qh[slot / 2] >> (4 * (slot % 2))) & 0xf;
                dst[positions[slot]] = norm * CENTROIDS_4BIT[q];
            }
        }
    }
}

#define DEFINE_TURBO2M4_CODEC(suffix, block_type, n, group)                                                         \
    void quantize_row_turbo2m4_##suffix##_ref(const float * GGML_RESTRICT x, block_type * GGML_RESTRICT y,          \
                                              int64_t k) {                                                          \
        quantize_row_turbo2m4_ref_impl(x, y, k, n, group, sizeof(block_type));                                      \
    }                                                                                                               \
    void dequantize_row_turbo2m4_##suffix(const block_type * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) { \
        dequantize_row_turbo2m4_impl(x, y, k, n, group, sizeof(block_type));                                        \
    }                                                                                                               \
    size_t quantize_turbo2m4_##suffix(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows,     \
                                      int64_t n_per_row, const float * imatrix) {                                   \
        GGML_UNUSED(imatrix);                                                                                       \
        assert(n_per_row % QK_TURBO2M4 == 0);                                                                       \
        const size_t row_size = (size_t) (n_per_row / QK_TURBO2M4) * sizeof(block_type);                            \
        for (int64_t row = 0; row < nrows; ++row) {                                                                 \
            quantize_row_turbo2m4_##suffix##_ref(src + row * n_per_row,                                             \
                                                 (block_type *) ((char *) dst + row * row_size), n_per_row);        \
        }                                                                                                           \
        return (size_t) nrows * row_size;                                                                           \
    }

DEFINE_TURBO2M4_CODEC(s4, block_turbo2m4_s4, 4, 0)
DEFINE_TURBO2M4_CODEC(s8, block_turbo2m4_s8, 8, 0)
DEFINE_TURBO2M4_CODEC(s16, block_turbo2m4_s16, 16, 0)
DEFINE_TURBO2M4_CODEC(g4, block_turbo2m4_g4, 4, 1)
DEFINE_TURBO2M4_CODEC(g8, block_turbo2m4_g8, 8, 1)
DEFINE_TURBO2M4_CODEC(g16, block_turbo2m4_g16, 16, 1)

#undef DEFINE_TURBO2M4_CODEC
