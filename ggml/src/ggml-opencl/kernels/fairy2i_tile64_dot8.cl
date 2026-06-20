#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_dot_product8 : enable

#define QK_FAIRY2I_TILE64     64
#define QK_FAIRY2I_ACT_Q16_64 64
#define FAIRY2I_TILE64_WIDE_TILE_M 4

static inline ushort fairy2i_f32_to_bf16(float x) {
    uint bits = as_uint(x);
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return (ushort) ((bits >> 16) | 64U);
    }
    return (ushort) ((bits + (0x7fffU + ((bits >> 16) & 1U))) >> 16);
}

static inline uint fairy2i_tile64_pack_bf16_pair(float real, float imag) {
    const uint r = (uint) fairy2i_f32_to_bf16(real);
    const uint i = (uint) fairy2i_f32_to_bf16(imag);
    return r | (i << 16);
}

static inline uint fairy2i_tile64_pack4_s8(int x0, int x1, int x2, int x3) {
    return (uint) (uchar) (char) x0 |
          ((uint) (uchar) (char) x1 << 8) |
          ((uint) (uchar) (char) x2 << 16) |
          ((uint) (uchar) (char) x3 << 24);
}

static inline int fairy2i_tile64_sum4_s8(uint packed) {
    return (int) (char) (packed & 0xffU) +
           (int) (char) ((packed >> 8) & 0xffU) +
           (int) (char) ((packed >> 16) & 0xffU) +
           (int) (char) ((packed >> 24) & 0xffU);
}

static inline int fairy2i_tile64_code_real(uint code) {
    const int sign = (int) ((code & 1U) << 1) - 1;
    const int imag = (int) (code >> 1);
    return sign * (1 - imag);
}

static inline int fairy2i_tile64_code_imag(uint code) {
    const int sign = (int) ((code & 1U) << 1) - 1;
    const int imag = (int) (code >> 1);
    return sign * imag;
}

static inline uint fairy2i_tile64_pack4_coeff_real_u8(uint packed) {
    return (uint) (uchar) (128 + fairy2i_tile64_code_real(packed & 3U)) |
          ((uint) (uchar) (128 + fairy2i_tile64_code_real((packed >> 2) & 3U)) << 8) |
          ((uint) (uchar) (128 + fairy2i_tile64_code_real((packed >> 4) & 3U)) << 16) |
          ((uint) (uchar) (128 + fairy2i_tile64_code_real((packed >> 6) & 3U)) << 24);
}

static inline uint fairy2i_tile64_pack4_coeff_imag_u8(uint packed) {
    return (uint) (uchar) (128 + fairy2i_tile64_code_imag(packed & 3U)) |
          ((uint) (uchar) (128 + fairy2i_tile64_code_imag((packed >> 2) & 3U)) << 8) |
          ((uint) (uchar) (128 + fairy2i_tile64_code_imag((packed >> 4) & 3U)) << 16) |
          ((uint) (uchar) (128 + fairy2i_tile64_code_imag((packed >> 6) & 3U)) << 24);
}

static inline int fairy2i_tile64_dot4_s8u8_qcom(uint x_s8, uint coeff_u8, int correction) {
    return qcom_dot8_acc(x_s8, coeff_u8, correction);
}

static inline int4 fairy2i_tile64_dot8_sums_packed(
        uint packed,
        uint xr_pack,
        uint xi_pack,
        int  xr_correction,
        int  xi_correction
) {
    const uint cr_pack = fairy2i_tile64_pack4_coeff_real_u8(packed);
    const uint ci_pack = fairy2i_tile64_pack4_coeff_imag_u8(packed);
    return (int4)(
        fairy2i_tile64_dot4_s8u8_qcom(xr_pack, cr_pack, xr_correction),
        fairy2i_tile64_dot4_s8u8_qcom(xi_pack, cr_pack, xi_correction),
        fairy2i_tile64_dot4_s8u8_qcom(xr_pack, ci_pack, xr_correction),
        fairy2i_tile64_dot4_s8u8_qcom(xi_pack, ci_pack, xi_correction));
}

/**
 * Fused Fairy2i W2 using Qualcomm dot8 for the four 2-bit lanes
 * packed in each weight byte. The activation q16 staging and final complex
 * accumulation semantics match kernel_fairy2i_tile64_wide_linear_w2_f32_act_q16_64.
 */
__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void kernel_fairy2i_tile64_wide_linear_w2_f32_act_q16_64_dot8(
        global uchar * u0_q,
        global half  * u0_d,
        global uchar * u1_q,
        global half  * u1_d,
        global uchar * w0_q,
        global half  * w0_d,
        global uchar * w1_q,
        global half  * w1_d,
        global char  * act_q,
        global half  * act_d,
        global char  * bias,
        ulong         offsetb,
        int           has_bias,
        global char  * dst,
        ulong         offsetd,
        int           k,
        int           m,
        int           n,
        int           x_ne1,
        int           x_ne2,
        int           bias_ne0,
        int           bias_ne1,
        int           bias_ne2,
        int           bias_ne3,
        ulong         bias_nb0,
        ulong         bias_nb1,
        ulong         bias_nb2,
        ulong         bias_nb3,
        ulong         nb0,
        ulong         nb1,
        local float * tmp_real,
        local float * tmp_imag
) {
    dst = dst + offsetd;
    bias = bias + offsetb;

    const int row_base = get_group_id(0) * FAIRY2I_TILE64_WIDE_TILE_M;
    const int col = get_group_id(1);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int block_slot = lid >> 4;
    const int lane = lid & 15;
    const int nb64 = k / QK_FAIRY2I_TILE64;
    const int nbq = k / QK_FAIRY2I_ACT_Q16_64;

    if (col >= n) {
        return;
    }

    float acc_real[FAIRY2I_TILE64_WIDE_TILE_M];
    float acc_imag[FAIRY2I_TILE64_WIDE_TILE_M];
#pragma unroll
    for (int i = 0; i < FAIRY2I_TILE64_WIDE_TILE_M; ++i) {
        acc_real[i] = 0.0f;
        acc_imag[i] = 0.0f;
    }

    for (int wb_base = 0; wb_base < nb64; wb_base += 4) {
        const int wb = wb_base + block_slot;
        const int act_block = wb;

        if (wb < nb64) {
            const int act_index = col * nbq + act_block;
            const int q_base = act_index * (2 * QK_FAIRY2I_ACT_Q16_64);
            const float x_real = vload_half(act_index * 2 + 0, act_d);
            const float x_imag = vload_half(act_index * 2 + 1, act_d);

            const uint xr_pack = fairy2i_tile64_pack4_s8(
                (int) act_q[q_base + lane + 0],
                (int) act_q[q_base + lane + 16],
                (int) act_q[q_base + lane + 32],
                (int) act_q[q_base + lane + 48]);
            const uint xi_pack = fairy2i_tile64_pack4_s8(
                (int) act_q[q_base + QK_FAIRY2I_ACT_Q16_64 + lane + 0],
                (int) act_q[q_base + QK_FAIRY2I_ACT_Q16_64 + lane + 16],
                (int) act_q[q_base + QK_FAIRY2I_ACT_Q16_64 + lane + 32],
                (int) act_q[q_base + QK_FAIRY2I_ACT_Q16_64 + lane + 48]);
            const int xr_sum = fairy2i_tile64_sum4_s8(xr_pack);
            const int xi_sum = fairy2i_tile64_sum4_s8(xi_pack);
            const int xr_correction = -(xr_sum << 7);
            const int xi_correction = -(xi_sum << 7);

#pragma unroll
            for (int tr = 0; tr < FAIRY2I_TILE64_WIDE_TILE_M; ++tr) {
                const int row = row_base + tr;
                if (row >= m) {
                    continue;
                }

                const int w_block = row * nb64 + wb;
                const uint packed_u0 = (uint) u0_q[w_block * 16 + lane];
                const uint packed_u1 = (uint) u1_q[w_block * 16 + lane];
                const uint packed_w0 = (uint) w0_q[w_block * 16 + lane];
                const uint packed_w1 = (uint) w1_q[w_block * 16 + lane];

                const float2 u0_s = vload_half2(w_block, u0_d);
                const float2 u1_s = vload_half2(w_block, u1_d);
                const float2 w0_s = vload_half2(w_block, w0_d);
                const float2 w1_s = vload_half2(w_block, w1_d);

                const int4 u0 = fairy2i_tile64_dot8_sums_packed(packed_u0, xr_pack, xi_pack, xr_correction, xi_correction);
                const int4 u1 = fairy2i_tile64_dot8_sums_packed(packed_u1, xr_pack, xi_pack, xr_correction, xi_correction);
                const int4 w0 = fairy2i_tile64_dot8_sums_packed(packed_w0, xr_pack, xi_pack, xr_correction, xi_correction);
                const int4 w1 = fairy2i_tile64_dot8_sums_packed(packed_w1, xr_pack, xi_pack, xr_correction, xi_correction);

                acc_real[tr] +=
                    u0_s.x * x_real * (float) u0.x - u0_s.y * x_imag * (float) u0.w +
                    u1_s.x * x_real * (float) u1.x - u1_s.y * x_imag * (float) u1.w +
                    w0_s.x * x_real * (float) w0.x + w0_s.y * x_imag * (float) w0.w +
                    w1_s.x * x_real * (float) w1.x + w1_s.y * x_imag * (float) w1.w;

                acc_imag[tr] +=
                    u0_s.y * x_real * (float) u0.z + u0_s.x * x_imag * (float) u0.y +
                    u1_s.y * x_real * (float) u1.z + u1_s.x * x_imag * (float) u1.y +
                    w0_s.y * x_real * (float) w0.z - w0_s.x * x_imag * (float) w0.y +
                    w1_s.y * x_real * (float) w1.z - w1_s.x * x_imag * (float) w1.y;
            }
        }
    }

#pragma unroll
    for (int tr = 0; tr < FAIRY2I_TILE64_WIDE_TILE_M; ++tr) {
        tmp_real[tr * lsize + lid] = acc_real[tr];
        tmp_imag[tr * lsize + lid] = acc_imag[tr];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
#pragma unroll
            for (int tr = 0; tr < FAIRY2I_TILE64_WIDE_TILE_M; ++tr) {
                tmp_real[tr * lsize + lid] += tmp_real[tr * lsize + lid + stride];
                tmp_imag[tr * lsize + lid] += tmp_imag[tr * lsize + lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        const int i1 = col % x_ne1;
        const int i2 = (col / x_ne1) % x_ne2;
        const int i3 = col / (x_ne1 * x_ne2);
#pragma unroll
        for (int tr = 0; tr < FAIRY2I_TILE64_WIDE_TILE_M; ++tr) {
            const int row = row_base + tr;
            if (row < m) {
                float out_real = tmp_real[tr * lsize];
                float out_imag = tmp_imag[tr * lsize];

                if (has_bias) {
                    const ulong bias_real_off =
                        (ulong) (row % bias_ne0) * bias_nb0 +
                        (ulong) (i1 % bias_ne1) * bias_nb1 +
                        (ulong) (i2 % bias_ne2) * bias_nb2 +
                        (ulong) (i3 % bias_ne3) * bias_nb3;
                    const ulong bias_imag_off =
                        (ulong) ((row + m) % bias_ne0) * bias_nb0 +
                        (ulong) (i1 % bias_ne1) * bias_nb1 +
                        (ulong) (i2 % bias_ne2) * bias_nb2 +
                        (ulong) (i3 % bias_ne3) * bias_nb3;
                    out_real += *((global float *) (bias + bias_real_off));
                    out_imag += *((global float *) (bias + bias_imag_off));
                }

                *((global uint *) (dst + (ulong) col * nb1 + (ulong) row * nb0)) =
                    fairy2i_tile64_pack_bf16_pair(out_real, out_imag);
            }
        }
    }
}
