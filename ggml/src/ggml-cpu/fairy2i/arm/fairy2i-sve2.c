#include "fairy2i-quants.h"

#include <stddef.h>

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE2)
#    include <arm_sve.h>
#endif

bool ggml_fairy2i_tile64_w2_arm_sve2_available(void) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE2)
    return true;
#else
    return false;
#endif
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE2)
static inline svuint8_t ggml_fairy2i_tile64_codes_part_sve2(svbool_t pg, svuint8_t packed, int part) {
    switch (part) {
        case 0:
            return svand_n_u8_x(pg, packed, 0x03);
        case 1:
            return svand_n_u8_x(pg, svlsr_n_u8_x(pg, packed, 2), 0x03);
        case 2:
            return svand_n_u8_x(pg, svlsr_n_u8_x(pg, packed, 4), 0x03);
        case 3:
            return svlsr_n_u8_x(pg, packed, 6);
        default:
            return svdup_n_u8(0);
    }
}

static inline void ggml_fairy2i_tile64_fuse_accumulate_one_sve2(const block_fairy2i_tile64_v2 *  w,
                                                                const block_fairy2i_act_q16_64 * x,
                                                                int32_t                          sums[4]) {
    const svint8_t zero = svdup_n_s8(0);

    for (int part = 0; part < 4; ++part) {
        const size_t base = (size_t) part * 16u;

        for (size_t off = 0; off < 16u; off += svcntb()) {
            const svbool_t pg = svwhilelt_b8_u64((uint64_t) off, 16u);

            const svuint8_t packed = svld1_u8(pg, w->qs + off);
            const svuint8_t codes  = ggml_fairy2i_tile64_codes_part_sve2(pg, packed, part);

            const svint8_t xr = svld1_s8(pg, (const int8_t *) x->x_real + base + off);
            const svint8_t xi = svld1_s8(pg, (const int8_t *) x->x_imag + base + off);

            const svuint8_t sign_bits = svand_n_u8_x(pg, codes, 0x01);
            const svuint8_t imag_bits = svand_n_u8_x(pg, codes, 0x02);
            const svbool_t  positive  = svcmpne_n_u8(pg, sign_bits, 0);
            const svbool_t  real      = svcmpeq_n_u8(pg, imag_bits, 0);
            const svbool_t  imag      = svcmpne_n_u8(pg, imag_bits, 0);

            const svint8_t signed_xr = svsel_s8(positive, xr, svneg_s8_x(pg, xr));
            const svint8_t signed_xi = svsel_s8(positive, xi, svneg_s8_x(pg, xi));

            const svint8_t real_xr = svsel_s8(real, signed_xr, zero);
            const svint8_t real_xi = svsel_s8(real, signed_xi, zero);
            const svint8_t imag_xr = svsel_s8(imag, signed_xr, zero);
            const svint8_t imag_xi = svsel_s8(imag, signed_xi, zero);

            sums[0] += (int32_t) svaddv_s8(pg, real_xr);
            sums[1] += (int32_t) svaddv_s8(pg, real_xi);
            sums[2] += (int32_t) svaddv_s8(pg, imag_xr);
            sums[3] += (int32_t) svaddv_s8(pg, imag_xi);
        }
    }
}

static inline void ggml_fairy2i_tile64_fuse_accumulate_block_four_unfused_sve2(
    const block_fairy2i_tile64_v2 * u0,
    const block_fairy2i_tile64_v2 * u1,
    const block_fairy2i_tile64_v2 * w0,
    const block_fairy2i_tile64_v2 * w1,
    const block_fairy2i_act_q16_64 * x,
    int32_t sums[4][4]) {
    ggml_fairy2i_tile64_fuse_accumulate_one_sve2(u0, x, sums[0]);
    ggml_fairy2i_tile64_fuse_accumulate_one_sve2(u1, x, sums[1]);
    ggml_fairy2i_tile64_fuse_accumulate_one_sve2(w0, x, sums[2]);
    ggml_fairy2i_tile64_fuse_accumulate_one_sve2(w1, x, sums[3]);
}

static inline void ggml_fairy2i_tile64_fuse_reduce_branch_sve2(svbool_t pg,
                                                               svint8_t real_xr,
                                                               svint8_t real_xi,
                                                               svint8_t imag_xr,
                                                               svint8_t imag_xi,
                                                               int32_t  sums[4]) {
    sums[0] += (int32_t) svaddv_s8(pg, real_xr);
    sums[1] += (int32_t) svaddv_s8(pg, real_xi);
    sums[2] += (int32_t) svaddv_s8(pg, imag_xr);
    sums[3] += (int32_t) svaddv_s8(pg, imag_xi);
}

static inline void ggml_fairy2i_tile64_fuse_accumulate_block_four_fused_sve2(
    const block_fairy2i_tile64_v2 * u0,
    const block_fairy2i_tile64_v2 * u1,
    const block_fairy2i_tile64_v2 * w0,
    const block_fairy2i_tile64_v2 * w1,
    const block_fairy2i_act_q16_64 * x,
    int32_t sums[4][4]) {
    const svbool_t all = svptrue_b8();
    const svint8_t zero = svdup_n_s8(0);

    // Replicate each 16-byte packed block across the SVE register.  The
    // per-lane branch predicates below select one copy into the logical
    // [u0 | u1 | w0 | w1] 64-byte space without a temporary memory buffer.
    const svuint8_t packed_u0 = svld1rq_u8(all, u0->qs);
    const svuint8_t packed_u1 = svld1rq_u8(all, u1->qs);
    const svuint8_t packed_w0 = svld1rq_u8(all, w0->qs);
    const svuint8_t packed_w1 = svld1rq_u8(all, w1->qs);

    for (int part = 0; part < 4; ++part) {
        const size_t base = (size_t) part * 16u;

        // All four weight branches consume the same 16 activation values.
        // ld1rq repeats those values once per 128-bit SVE segment.
        const svint8_t xr = svld1rq_s8(all, (const int8_t *) x->x_real + base);
        const svint8_t xi = svld1rq_s8(all, (const int8_t *) x->x_imag + base);

        for (size_t off = 0; off < 64u; off += svcntb()) {
            const svbool_t pg = svwhilelt_b8_u64((uint64_t) off, 64u);

            const svuint8_t virtual_index = svindex_u8((uint8_t) off, 1);
            const svuint8_t branch_id     = svlsr_n_u8_x(pg, virtual_index, 4);
            const svbool_t  branch_u0     = svcmpeq_n_u8(pg, branch_id, 0);
            const svbool_t  branch_u1     = svcmpeq_n_u8(pg, branch_id, 1);
            const svbool_t  branch_w0     = svcmpeq_n_u8(pg, branch_id, 2);

            svuint8_t packed = svsel_u8(branch_w0, packed_w0, packed_w1);
            packed           = svsel_u8(branch_u1, packed_u1, packed);
            packed           = svsel_u8(branch_u0, packed_u0, packed);

            const svuint8_t codes = ggml_fairy2i_tile64_codes_part_sve2(pg, packed, part);

            const svuint8_t sign_bits = svand_n_u8_x(pg, codes, 0x01);
            const svuint8_t imag_bits = svand_n_u8_x(pg, codes, 0x02);
            const svbool_t  positive  = svcmpne_n_u8(pg, sign_bits, 0);
            const svbool_t  real      = svcmpeq_n_u8(pg, imag_bits, 0);
            const svbool_t  imag      = svcmpne_n_u8(pg, imag_bits, 0);

            const svint8_t signed_xr = svsel_s8(positive, xr, svneg_s8_x(pg, xr));
            const svint8_t signed_xi = svsel_s8(positive, xi, svneg_s8_x(pg, xi));
            const svint8_t real_xr   = svsel_s8(real, signed_xr, zero);
            const svint8_t real_xi   = svsel_s8(real, signed_xi, zero);
            const svint8_t imag_xr   = svsel_s8(imag, signed_xr, zero);
            const svint8_t imag_xi   = svsel_s8(imag, signed_xi, zero);

            const size_t chunk_end = off + svcntb();
            for (int branch = 0; branch < 4; ++branch) {
                const size_t branch_base = (size_t) branch * 16u;
                if (branch_base < off || branch_base >= chunk_end) {
                    continue;
                }

                const svbool_t branch_pg = svcmpeq_n_u8(pg, branch_id, (uint8_t) branch);
                ggml_fairy2i_tile64_fuse_reduce_branch_sve2(
                    branch_pg, real_xr, real_xi, imag_xr, imag_xi, sums[branch]);
            }
        }
    }
}
#endif

void ggml_fairy2i_tile64_fuse_accumulate_block_four_sve2(const block_fairy2i_tile64_v2 *  u0,
                                                         const block_fairy2i_tile64_v2 *  u1,
                                                         const block_fairy2i_tile64_v2 *  w0,
                                                         const block_fairy2i_tile64_v2 *  w1,
                                                         const block_fairy2i_act_q16_64 * x,
                                                         int32_t                          sums[4][4]) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE2)
// Internal A/B switch: the default is the fused virtual-64-lane kernel, while
// defining this macro keeps the original four independent calls available.
#if defined(GGML_FAIRY2I_CPU_ARM_SVE2_UNFUSED)
    ggml_fairy2i_tile64_fuse_accumulate_block_four_unfused_sve2(u0, u1, w0, w1, x, sums);
#else
    ggml_fairy2i_tile64_fuse_accumulate_block_four_fused_sve2(u0, u1, w0, w1, x, sums);
#endif
#else
    ggml_fairy2i_tile64_fuse_accumulate_block_four_neon(u0, u1, w0, w1, x, sums);
#endif
}
