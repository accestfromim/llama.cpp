#include "../lut-qgemm.h"
#include "../lut/ggml-fairy2i-lut-impl.h"
#include "../lut/ggml-fairy2i-lut.h"

#include "ggml-impl.h"
#include "ggml-quants.h"

#include <algorithm>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE2)
#    include <arm_sve.h>
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE2)
namespace {

constexpr int k_fairy2i_lut_rows_per_tile       = 16;
constexpr int k_fairy2i_lut_max_tiles_per_pass  = 4;
constexpr int k_fairy2i_lut_max_rows_per_pass   =
    k_fairy2i_lut_rows_per_tile * k_fairy2i_lut_max_tiles_per_pass;

static bool ggml_fairy2i_lut_sve2_debug_enabled(void) {
    const char * env = getenv("GGML_FAIRY2I_CPU_DEBUG");
    return env && strcmp(env, "0") != 0;
}

static void ggml_fairy2i_lut_debug_log_arm_sve2_four_once(int m, int64_t blocks, bool pack_bf16) {
    if (!ggml_fairy2i_lut_sve2_debug_enabled()) {
        return;
    }

    static bool logged = false;
    if (logged) {
        return;
    }
    logged = true;

    GGML_LOG_INFO("fairy2i_lut: path=arm_sve2_four M=%d blocks=%lld pack_bf16=%d\n", m, (long long) blocks,
                  pack_bf16 ? 1 : 0);
}

// Load one 16-entry table into the first 16 lanes. All packed weight indexes
// remain in [0, 15], so ordinary SVE TBL lets every output lane share this
// single table; no per-128-bit-segment table replication is required.
static inline svint8_t ggml_fairy2i_lut_load_table16_sve2(const int8_t * table) {
    const svbool_t pg16 = svwhilelt_b8_u64(0, 16);
    return svld1_s8(pg16, table);
}

static inline svuint8_t ggml_fairy2i_lut_pack_tiles_sve2(
    const fairy2i_tile64_lut_wtile_16 * wtiles,
    int64_t                              blocks,
    int                                  tile_base,
    int64_t                              blk,
    int                                  byte_idx,
    int                                  tiles_this_pass,
    svbool_t                             pg_rows,
    svuint8_t                            segment_id) {
    const svbool_t  all    = svptrue_b8();
    svuint8_t       packed = svdup_n_u8(0);

    for (int tile = 0; tile < tiles_this_pass; ++tile) {
        const fairy2i_tile64_lut_wtile_16 * wt =
            wtiles + (size_t) (tile_base + tile) * (size_t) blocks + (size_t) blk;

        // ld1rq repeats this tile's 16 indexes only to make them available at
        // the same lane offsets in every 128-bit segment. The LUT itself is
        // not repeated: all selected indexes still address one 16-entry table.
        const svuint8_t tile_indexes = svld1rq_u8(all, wt->qs[byte_idx]);
        const svbool_t  tile_pg      = svcmpeq_n_u8(pg_rows, segment_id, (uint8_t) tile);
        packed                       = svsel_u8(tile_pg, tile_indexes, packed);
    }

    return packed;
}

static inline void ggml_fairy2i_lut_accumulate_channel_sve2(
    svint8_t    table_lo,
    svint8_t    table_hi,
    svuint8_t   index_lo,
    svuint8_t   index_hi,
    svint16_t & sum_lo,
    svint16_t & sum_hi) {
    const svint8_t out_lo = svtbl_s8(table_lo, index_lo);
    const svint8_t out_hi = svtbl_s8(table_hi, index_hi);

    const svint16_t pair_lo = svadd_s16_x(svptrue_b16(), svunpklo_s16(out_lo), svunpklo_s16(out_hi));
    const svint16_t pair_hi = svadd_s16_x(svptrue_b16(), svunpkhi_s16(out_lo), svunpkhi_s16(out_hi));

    sum_lo = svadd_s16_x(svptrue_b16(), sum_lo, pair_lo);
    sum_hi = svadd_s16_x(svptrue_b16(), sum_hi, pair_hi);
}

static inline svint32_t ggml_fairy2i_lut_sum_quarter_sve2(svint16_t sum_lo, svint16_t sum_hi, int quarter) {
    switch (quarter) {
        case 0:
            return svunpklo_s32(sum_lo);
        case 1:
            return svunpkhi_s32(sum_lo);
        case 2:
            return svunpklo_s32(sum_hi);
        case 3:
            return svunpkhi_s32(sum_hi);
        default:
            return svdup_n_s32(0);
    }
}

static inline svfloat32_t ggml_fairy2i_lut_load_weight_scale_sve2(
    const fairy2i_tile64_lut_wtile_16 * wtiles,
    int64_t                              blocks,
    int                                  tile_base,
    int64_t                              blk,
    size_t                               virtual_row,
    size_t                               count,
    bool                                 imag) {
    GGML_ASSERT(count <= svcntw());
    GGML_ASSERT(virtual_row + count <= k_fairy2i_lut_max_rows_per_pass);

    const size_t first_tile = virtual_row / k_fairy2i_lut_rows_per_tile;
    const size_t first_lane = virtual_row % k_fairy2i_lut_rows_per_tile;

    const ggml_half * src = nullptr;
    alignas(64) ggml_half gathered[k_fairy2i_lut_max_rows_per_pass];

    if (first_lane + count <= k_fairy2i_lut_rows_per_tile) {
        const fairy2i_tile64_lut_wtile_16 * wt =
            wtiles + (size_t) (tile_base + (int) first_tile) * (size_t) blocks + (size_t) blk;
        src = (imag ? wt->d_imag : wt->d_real) + first_lane;
    } else {
        size_t copied = 0;
        while (copied < count) {
            const size_t row       = virtual_row + copied;
            const size_t tile      = row / k_fairy2i_lut_rows_per_tile;
            const size_t lane      = row % k_fairy2i_lut_rows_per_tile;
            const size_t available = std::min(count - copied, (size_t) k_fairy2i_lut_rows_per_tile - lane);
            const fairy2i_tile64_lut_wtile_16 * wt =
                wtiles + (size_t) (tile_base + (int) tile) * (size_t) blocks + (size_t) blk;
            const ggml_half * tile_scale = imag ? wt->d_imag : wt->d_real;
            memcpy(gathered + copied, tile_scale + lane, available * sizeof(ggml_half));
            copied += available;
        }
        src = gathered;
    }

    const svbool_t   pg_w = svwhilelt_b32_u64(0, (uint64_t) count);
    const svuint32_t raw  = svld1uh_u32(pg_w, (const uint16_t *) src);
    const svfloat16_t fp16 = svreinterpret_f16_u32(raw);
    return svcvt_f32_f16_x(pg_w, fp16);
}

static inline void ggml_fairy2i_lut_apply_quarter_sve2(
    const fairy2i_tile64_lut_wtile_16 * wtiles,
    int64_t                              blocks,
    int                                  tile_base,
    int64_t                              blk,
    size_t                               virtual_row,
    size_t                               count,
    float                                lr,
    float                                li,
    svint32_t                            sum_ac,
    svint32_t                            sum_bd,
    svint32_t                            sum_bc,
    svint32_t                            sum_ad,
    svfloat32_t &                        acc_r,
    svfloat32_t &                        acc_i) {
    const svbool_t pg = svwhilelt_b32_u64(0, (uint64_t) count);

    const svfloat32_t wr = ggml_fairy2i_lut_load_weight_scale_sve2(
        wtiles, blocks, tile_base, blk, virtual_row, count, false);
    const svfloat32_t wi = ggml_fairy2i_lut_load_weight_scale_sve2(
        wtiles, blocks, tile_base, blk, virtual_row, count, true);

    const svfloat32_t fac = svcvt_f32_s32_x(pg, sum_ac);
    const svfloat32_t fbd = svcvt_f32_s32_x(pg, sum_bd);
    const svfloat32_t fbc = svcvt_f32_s32_x(pg, sum_bc);
    const svfloat32_t fad = svcvt_f32_s32_x(pg, sum_ad);

    const svfloat32_t lr_wr = svmul_n_f32_x(pg, wr, lr);
    const svfloat32_t li_wi = svmul_n_f32_x(pg, wi, li);
    const svfloat32_t lr_wi = svmul_n_f32_x(pg, wi, lr);
    const svfloat32_t li_wr = svmul_n_f32_x(pg, wr, li);

    acc_r = svmla_f32_x(pg, acc_r, fac, lr_wr);
    acc_r = svmla_f32_x(pg, acc_r, fbd, li_wi);
    acc_i = svmla_f32_x(pg, acc_i, fbc, lr_wi);
    acc_i = svmla_f32_x(pg, acc_i, fad, li_wr);
}

static inline void ggml_fairy2i_lut_store_quarter_sve2(
    uint8_t *   dst_col,
    size_t      dst_row_stride,
    size_t      output_row,
    size_t      count,
    bool        pack_bf16,
    bool        add,
    svfloat32_t acc_r,
    svfloat32_t acc_i) {
    const svbool_t pg = svwhilelt_b32_u64(0, (uint64_t) count);

    if (!pack_bf16 && dst_row_stride == 2u * sizeof(float)) {
        float * out = (float *) (dst_col + output_row * dst_row_stride);
        if (add) {
            const svfloat32x2_t previous = svld2_f32(pg, out);
            acc_r = svadd_f32_x(pg, svget2_f32(previous, 0), acc_r);
            acc_i = svadd_f32_x(pg, svget2_f32(previous, 1), acc_i);
        }
        svst2_f32(pg, out, svcreate2_f32(acc_r, acc_i));
        return;
    }

    alignas(64) float tmp_r[k_fairy2i_lut_max_rows_per_pass];
    alignas(64) float tmp_i[k_fairy2i_lut_max_rows_per_pass];
    svst1_f32(pg, tmp_r, acc_r);
    svst1_f32(pg, tmp_i, acc_i);

    for (size_t lane = 0; lane < count; ++lane) {
        uint8_t * out = dst_col + (output_row + lane) * dst_row_stride;
        if (pack_bf16) {
            const float previous_r = add ? ggml_bf16_to_fp32(((const ggml_bf16_t *) out)[0]) : 0.0f;
            const float previous_i = add ? ggml_bf16_to_fp32(((const ggml_bf16_t *) out)[1]) : 0.0f;
            ((ggml_bf16_t *) out)[0] = ggml_fp32_to_bf16(previous_r + tmp_r[lane]);
            ((ggml_bf16_t *) out)[1] = ggml_fp32_to_bf16(previous_i + tmp_i[lane]);
        } else {
            ((float *) out)[0] = (add ? ((const float *) out)[0] : 0.0f) + tmp_r[lane];
            ((float *) out)[1] = (add ? ((const float *) out)[1] : 0.0f) + tmp_i[lane];
        }
    }
}

static bool ggml_fairy2i_tile64_lut_qgemm_pair_sve2(
    int          m,
    int          k,
    int          n,
    const void * packed_wtiles0,
    const void * packed_wtiles1,
    const void * lut,
    const void * lut_scales,
    float *      dst,
    size_t       dst_col_stride,
    size_t       dst_row_stride,
    bool         pack_bf16,
    bool         negate_imag_scale,
    bool         add) {
    if (m == 0) {
        return true;
    }
    if (!packed_wtiles0 || !packed_wtiles1 || !lut || !lut_scales || !dst || m < 0 || k <= 0 || n <= 0 ||
        k % QK_FAIRY2I_TILE64 != 0) {
        return false;
    }

    const int64_t blocks           = k / QK_FAIRY2I_TILE64;
    const int64_t groups_per_block = QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK;
    const int64_t groups           = blocks * groups_per_block;
    const int     tiles            = (m + k_fairy2i_lut_rows_per_tile - 1) / k_fairy2i_lut_rows_per_tile;
    const int     tiles_per_pass   = std::max(1, std::min((int) (svcntb() / k_fairy2i_lut_rows_per_tile),
                                                               k_fairy2i_lut_max_tiles_per_pass));

    const auto * wtiles0 = (const fairy2i_tile64_lut_wtile_16 *) packed_wtiles0;
    const auto * wtiles1 = (const fairy2i_tile64_lut_wtile_16 *) packed_wtiles1;

    const svbool_t  all_b8     = svptrue_b8();
    const svuint8_t lane_index = svindex_u8(0, 1);
    const svuint8_t segment_id = svlsr_n_u8_x(all_b8, lane_index, 4);

    for (int col = 0; col < n; ++col) {
        const int8_t * lut_col =
            (const int8_t *) lut + (size_t) col * (size_t) groups * k_fairy2i_lut_group_bytes;
        const float * scales = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2u;
        uint8_t *     dst_col = (uint8_t *) dst + (size_t) col * dst_col_stride;

        for (int tile_base = 0; tile_base < tiles; tile_base += tiles_per_pass) {
            const int tiles_this_pass = std::min(tiles_per_pass, tiles - tile_base);
            const int first_row       = tile_base * k_fairy2i_lut_rows_per_tile;
            const int active_rows     = std::min(m - first_row, tiles_this_pass * k_fairy2i_lut_rows_per_tile);
            const svbool_t pg_rows    = svwhilelt_b8_u64(0, (uint64_t) active_rows);

            svfloat32_t acc_r0 = svdup_n_f32(0.0f);
            svfloat32_t acc_r1 = svdup_n_f32(0.0f);
            svfloat32_t acc_r2 = svdup_n_f32(0.0f);
            svfloat32_t acc_r3 = svdup_n_f32(0.0f);
            svfloat32_t acc_i0 = svdup_n_f32(0.0f);
            svfloat32_t acc_i1 = svdup_n_f32(0.0f);
            svfloat32_t acc_i2 = svdup_n_f32(0.0f);
            svfloat32_t acc_i3 = svdup_n_f32(0.0f);

            for (int64_t blk = 0; blk < blocks; ++blk) {
                const svint16_t zero_h = svdup_n_s16(0);

                svint16_t sum0_ac_lo = zero_h;
                svint16_t sum0_ac_hi = zero_h;
                svint16_t sum0_bd_lo = zero_h;
                svint16_t sum0_bd_hi = zero_h;
                svint16_t sum0_bc_lo = zero_h;
                svint16_t sum0_bc_hi = zero_h;
                svint16_t sum0_ad_lo = zero_h;
                svint16_t sum0_ad_hi = zero_h;

                svint16_t sum1_ac_lo = zero_h;
                svint16_t sum1_ac_hi = zero_h;
                svint16_t sum1_bd_lo = zero_h;
                svint16_t sum1_bd_hi = zero_h;
                svint16_t sum1_bc_lo = zero_h;
                svint16_t sum1_bc_hi = zero_h;
                svint16_t sum1_ad_lo = zero_h;
                svint16_t sum1_ad_hi = zero_h;

                const int8_t * lut_ptr =
                    lut_col + (size_t) blk * (size_t) groups_per_block * k_fairy2i_lut_group_bytes;

                for (int byte_idx = 0; byte_idx < groups_per_block / 2; ++byte_idx) {
                    const svuint8_t packed0 = ggml_fairy2i_lut_pack_tiles_sve2(
                        wtiles0, blocks, tile_base, blk, byte_idx, tiles_this_pass, pg_rows, segment_id);
                    const svuint8_t packed1 = ggml_fairy2i_lut_pack_tiles_sve2(
                        wtiles1, blocks, tile_base, blk, byte_idx, tiles_this_pass, pg_rows, segment_id);

                    const svuint8_t index0_lo = svand_n_u8_x(pg_rows, packed0, 0x0f);
                    const svuint8_t index0_hi = svand_n_u8_x(pg_rows, svlsr_n_u8_x(pg_rows, packed0, 4), 0x0f);
                    const svuint8_t index1_lo = svand_n_u8_x(pg_rows, packed1, 0x0f);
                    const svuint8_t index1_hi = svand_n_u8_x(pg_rows, svlsr_n_u8_x(pg_rows, packed1, 4), 0x0f);

                    const int8_t * lut_pair = lut_ptr + (size_t) byte_idx * 2u * k_fairy2i_lut_group_bytes;

#define GGML_FAIRY2I_LUT_ACCUMULATE_SVE2(channel, offset)                                                   \
    do {                                                                                                    \
        const svint8_t table_lo = ggml_fairy2i_lut_load_table16_sve2(lut_pair + (offset));                 \
        const svint8_t table_hi =                                                                             \
            ggml_fairy2i_lut_load_table16_sve2(lut_pair + k_fairy2i_lut_group_bytes + (offset));           \
        ggml_fairy2i_lut_accumulate_channel_sve2(                                                            \
            table_lo, table_hi, index0_lo, index0_hi, sum0_##channel##_lo, sum0_##channel##_hi);            \
        ggml_fairy2i_lut_accumulate_channel_sve2(                                                            \
            table_lo, table_hi, index1_lo, index1_hi, sum1_##channel##_lo, sum1_##channel##_hi);            \
    } while (false)

                    GGML_FAIRY2I_LUT_ACCUMULATE_SVE2(ac, 0);
                    GGML_FAIRY2I_LUT_ACCUMULATE_SVE2(bd, 16);
                    GGML_FAIRY2I_LUT_ACCUMULATE_SVE2(bc, 32);
                    GGML_FAIRY2I_LUT_ACCUMULATE_SVE2(ad, 48);

#undef GGML_FAIRY2I_LUT_ACCUMULATE_SVE2
                }

                const float lr = scales[blk * 2 + 0];
                const float li = negate_imag_scale ? -scales[blk * 2 + 1] : scales[blk * 2 + 1];

#define GGML_FAIRY2I_LUT_APPLY_QUARTER_SVE2(q, acc_r, acc_i)                                                \
    do {                                                                                                    \
        const size_t virtual_row = (size_t) (q) * svcntw();                                                \
        if (virtual_row < (size_t) active_rows) {                                                           \
            const size_t count = std::min((size_t) active_rows - virtual_row, (size_t) svcntw());          \
            ggml_fairy2i_lut_apply_quarter_sve2(                                                            \
                wtiles0, blocks, tile_base, blk, virtual_row, count, lr, li,                               \
                ggml_fairy2i_lut_sum_quarter_sve2(sum0_ac_lo, sum0_ac_hi, q),                              \
                ggml_fairy2i_lut_sum_quarter_sve2(sum0_bd_lo, sum0_bd_hi, q),                              \
                ggml_fairy2i_lut_sum_quarter_sve2(sum0_bc_lo, sum0_bc_hi, q),                              \
                ggml_fairy2i_lut_sum_quarter_sve2(sum0_ad_lo, sum0_ad_hi, q), acc_r, acc_i);                \
            ggml_fairy2i_lut_apply_quarter_sve2(                                                            \
                wtiles1, blocks, tile_base, blk, virtual_row, count, lr, li,                               \
                ggml_fairy2i_lut_sum_quarter_sve2(sum1_ac_lo, sum1_ac_hi, q),                              \
                ggml_fairy2i_lut_sum_quarter_sve2(sum1_bd_lo, sum1_bd_hi, q),                              \
                ggml_fairy2i_lut_sum_quarter_sve2(sum1_bc_lo, sum1_bc_hi, q),                              \
                ggml_fairy2i_lut_sum_quarter_sve2(sum1_ad_lo, sum1_ad_hi, q), acc_r, acc_i);                \
        }                                                                                                   \
    } while (false)

                GGML_FAIRY2I_LUT_APPLY_QUARTER_SVE2(0, acc_r0, acc_i0);
                GGML_FAIRY2I_LUT_APPLY_QUARTER_SVE2(1, acc_r1, acc_i1);
                GGML_FAIRY2I_LUT_APPLY_QUARTER_SVE2(2, acc_r2, acc_i2);
                GGML_FAIRY2I_LUT_APPLY_QUARTER_SVE2(3, acc_r3, acc_i3);

#undef GGML_FAIRY2I_LUT_APPLY_QUARTER_SVE2
            }

#define GGML_FAIRY2I_LUT_STORE_QUARTER_SVE2(q, acc_r, acc_i)                                                \
    do {                                                                                                    \
        const size_t virtual_row = (size_t) (q) * svcntw();                                                \
        if (virtual_row < (size_t) active_rows) {                                                           \
            const size_t count = std::min((size_t) active_rows - virtual_row, (size_t) svcntw());          \
            ggml_fairy2i_lut_store_quarter_sve2(dst_col, dst_row_stride, (size_t) first_row + virtual_row, \
                                                 count, pack_bf16, add, acc_r, acc_i);                       \
        }                                                                                                   \
    } while (false)

            GGML_FAIRY2I_LUT_STORE_QUARTER_SVE2(0, acc_r0, acc_i0);
            GGML_FAIRY2I_LUT_STORE_QUARTER_SVE2(1, acc_r1, acc_i1);
            GGML_FAIRY2I_LUT_STORE_QUARTER_SVE2(2, acc_r2, acc_i2);
            GGML_FAIRY2I_LUT_STORE_QUARTER_SVE2(3, acc_r3, acc_i3);

#undef GGML_FAIRY2I_LUT_STORE_QUARTER_SVE2
        }
    }

    return true;
}

}  // namespace
#endif

bool ggml_fairy2i_tile64_lut_qgemm_four_sve2(
    int          m,
    int          k,
    int          n,
    const void * packed_u0,
    const void * packed_u1,
    const void * packed_w0,
    const void * packed_w1,
    const void * lut,
    const void * lut_scales,
    float *      dst,
    size_t       dst_col_stride,
    size_t       dst_row_stride,
    bool         pack_bf16) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE2)
    ggml_fairy2i_lut_debug_log_arm_sve2_four_once(m, (int64_t) k / QK_FAIRY2I_TILE64, pack_bf16);

    if (!ggml_fairy2i_tile64_lut_qgemm_pair_sve2(m, k, n, packed_u0, packed_u1, lut, lut_scales, dst,
                                                  dst_col_stride, dst_row_stride, pack_bf16,
                                                  /* negate_imag_scale = */ true,
                                                  /* add = */ false)) {
        return false;
    }

    return ggml_fairy2i_tile64_lut_qgemm_pair_sve2(m, k, n, packed_w0, packed_w1, lut, lut_scales, dst,
                                                    dst_col_stride, dst_row_stride, pack_bf16,
                                                    /* negate_imag_scale = */ false,
                                                    /* add = */ true);
#else
    (void) m;
    (void) k;
    (void) n;
    (void) packed_u0;
    (void) packed_u1;
    (void) packed_w0;
    (void) packed_w1;
    (void) lut;
    (void) lut_scales;
    (void) dst;
    (void) dst_col_stride;
    (void) dst_row_stride;
    (void) pack_bf16;
    return false;
#endif
}
