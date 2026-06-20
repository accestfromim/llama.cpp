#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK_FAIRY2I_ACT_Q16_64 64
#define QK_FAIRY2I_TILE64   64

#define FAIRY2I_TILE64_TILE_M 2
#define FAIRY2I_TILE64_TILE_N 2
#define FAIRY2I_TILE64_TILE_OUT (FAIRY2I_TILE64_TILE_M * FAIRY2I_TILE64_TILE_N)

#define FAIRY2I_TILE64_GEMV_TILE_M 2
#define FAIRY2I_TILE64_GEMV4_TILE_M 4
#define FAIRY2I_TILE64_WIDE_TILE_M 4

static inline float fairy2i_bf16_to_f32(ushort h) {
    return as_float(((uint) h) << 16);
}

static inline ushort fairy2i_f32_to_bf16(float x) {
    uint bits = as_uint(x);
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return (ushort) ((bits >> 16) | 64U);
    }
    return (ushort) ((bits + (0x7fffU + ((bits >> 16) & 1U))) >> 16);
}

static inline int fairy2i_clamp_i32(int v, int lo, int hi) {
    return min(hi, max(lo, v));
}

static inline uint fairy2i_tile64_pack_bf16_pair(float real, float imag) {
    const uint r = (uint) fairy2i_f32_to_bf16(real);
    const uint i = (uint) fairy2i_f32_to_bf16(imag);
    return r | (i << 16);
}

static inline int2 fairy2i_tile64_decode_code_branchless(uint code) {
    const int sign = (int) ((code & 1U) << 1) - 1;
    const int imag = (int) (code >> 1);
    return (int2)(sign * (1 - imag), sign * imag);
}

/**
 * Converts packed-BF16 Fairy2i activations from F32 carrier format to a SoA
 * q16 staging layout. The q buffer stores real and imaginary int8 planes for
 * each 64-value block; d stores the corresponding fp16 scale pair.
 */
kernel void kernel_fairy2i_tile64_act_q16_64_quantize(
        global char * src,
        ulong         offset,
        global char * act_q,
        global half * act_d,
        int           k,
        int           n,
        ulong         nb10,
        ulong         nb11,
        local float * tmp_real,
        local float * tmp_imag
) {
    src = src + offset;

    const int block = get_group_id(0);
    const int col   = get_group_id(1);
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);
    const int blocks_per_col = k / QK_FAIRY2I_ACT_Q16_64;

    if (col >= n || block >= blocks_per_col) {
        return;
    }

    float max_real = 1.0e-5f;
    float max_imag = 1.0e-5f;
    for (int j = lid; j < QK_FAIRY2I_ACT_Q16_64; j += lsize) {
        const int k_idx = block * QK_FAIRY2I_ACT_Q16_64 + j;
        const uint pair = *((global uint *) (src + (ulong) col * nb11 + (ulong) k_idx * nb10));
        const float xr = fairy2i_bf16_to_f32((ushort) (pair & 0xffffU));
        const float xi = fairy2i_bf16_to_f32((ushort) (pair >> 16));
        max_real = fmax(max_real, fabs(xr));
        max_imag = fmax(max_imag, fabs(xi));
    }

    tmp_real[lid] = max_real;
    tmp_imag[lid] = max_imag;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            tmp_real[lid] = fmax(tmp_real[lid], tmp_real[lid + stride]);
            tmp_imag[lid] = fmax(tmp_imag[lid], tmp_imag[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float scale_real = tmp_real[0] / 127.0f;
    const float scale_imag = tmp_imag[0] / 127.0f;
    const float iscale_real = 1.0f / scale_real;
    const float iscale_imag = 1.0f / scale_imag;
    const int block_index = col * blocks_per_col + block;

    if (lid == 0) {
        vstore_half(scale_real, 0, act_d + block_index * 2 + 0);
        vstore_half(scale_imag, 0, act_d + block_index * 2 + 1);
    }

    const int q_base = block_index * (2 * QK_FAIRY2I_ACT_Q16_64);
    for (int j = lid; j < QK_FAIRY2I_ACT_Q16_64; j += lsize) {
        const int k_idx = block * QK_FAIRY2I_ACT_Q16_64 + j;
        const uint pair = *((global uint *) (src + (ulong) col * nb11 + (ulong) k_idx * nb10));
        const float xr = fairy2i_bf16_to_f32((ushort) (pair & 0xffffU));
        const float xi = fairy2i_bf16_to_f32((ushort) (pair >> 16));

        const int qr = fairy2i_clamp_i32((int) rint(xr * iscale_real), -127, 127);
        const int qi = fairy2i_clamp_i32((int) rint(xi * iscale_imag), -127, 127);

        act_q[q_base + j] = (char) qr;
        act_q[q_base + QK_FAIRY2I_ACT_Q16_64 + j] = (char) qi;
    }
}

/**
 * Fused Fairy2i W2 wide-linear. Activations are first staged to q16_64 SoA,
 * then this kernel consumes U.s0/U.s1/W.s0/W.s1 tile64 weights and writes the
 * final packed-BF16 complex output. U branches use w*x; W branches use
 * w*conj(x), matching the CPU fused wide-linear implementation.
 */
kernel void kernel_fairy2i_tile64_wide_linear_w2_f32_act_q16_64(
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
        local float * tmp_imag,
        local char  * act_real_tile,
        local char  * act_imag_tile
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
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int j = lane + 16 * part;
                act_real_tile[block_slot * QK_FAIRY2I_TILE64 + j] = act_q[q_base + j];
                act_imag_tile[block_slot * QK_FAIRY2I_TILE64 + j] = act_q[q_base + QK_FAIRY2I_ACT_Q16_64 + j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (wb < nb64) {
            const int act_index = col * nbq + act_block;
            const float x_real = vload_half(act_index * 2 + 0, act_d);
            const float x_imag = vload_half(act_index * 2 + 1, act_d);

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

                const float u0_real = vload_half(w_block * 2 + 0, u0_d);
                const float u0_imag = vload_half(w_block * 2 + 1, u0_d);
                const float u1_real = vload_half(w_block * 2 + 0, u1_d);
                const float u1_imag = vload_half(w_block * 2 + 1, u1_d);
                const float w0_real = vload_half(w_block * 2 + 0, w0_d);
                const float w0_imag = vload_half(w_block * 2 + 1, w0_d);
                const float w1_real = vload_half(w_block * 2 + 0, w1_d);
                const float w1_imag = vload_half(w_block * 2 + 1, w1_d);

                int u0_ac = 0, u0_ad = 0, u0_bc = 0, u0_bd = 0;
                int u1_ac = 0, u1_ad = 0, u1_bc = 0, u1_bd = 0;
                int w0_ac = 0, w0_ad = 0, w0_bc = 0, w0_bd = 0;
                int w1_ac = 0, w1_ad = 0, w1_bc = 0, w1_bd = 0;

#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    const int j = lane + 16 * part;
                    const int xr = (int) act_real_tile[block_slot * QK_FAIRY2I_TILE64 + j];
                    const int xi = (int) act_imag_tile[block_slot * QK_FAIRY2I_TILE64 + j];

                    const int2 cu0 = fairy2i_tile64_decode_code_branchless((packed_u0 >> (2 * part)) & 3U);
                    const int2 cu1 = fairy2i_tile64_decode_code_branchless((packed_u1 >> (2 * part)) & 3U);
                    const int2 cw0 = fairy2i_tile64_decode_code_branchless((packed_w0 >> (2 * part)) & 3U);
                    const int2 cw1 = fairy2i_tile64_decode_code_branchless((packed_w1 >> (2 * part)) & 3U);

                    u0_ac += xr * cu0.x; u0_ad += xi * cu0.x; u0_bc += xr * cu0.y; u0_bd += xi * cu0.y;
                    u1_ac += xr * cu1.x; u1_ad += xi * cu1.x; u1_bc += xr * cu1.y; u1_bd += xi * cu1.y;
                    w0_ac += xr * cw0.x; w0_ad += xi * cw0.x; w0_bc += xr * cw0.y; w0_bd += xi * cw0.y;
                    w1_ac += xr * cw1.x; w1_ad += xi * cw1.x; w1_bc += xr * cw1.y; w1_bd += xi * cw1.y;
                }

                acc_real[tr] +=
                    u0_real * x_real * (float) u0_ac - u0_imag * x_imag * (float) u0_bd +
                    u1_real * x_real * (float) u1_ac - u1_imag * x_imag * (float) u1_bd +
                    w0_real * x_real * (float) w0_ac + w0_imag * x_imag * (float) w0_bd +
                    w1_real * x_real * (float) w1_ac + w1_imag * x_imag * (float) w1_bd;

                acc_imag[tr] +=
                    u0_imag * x_real * (float) u0_bc + u0_real * x_imag * (float) u0_ad +
                    u1_imag * x_real * (float) u1_bc + u1_real * x_imag * (float) u1_ad +
                    w0_imag * x_real * (float) w0_bc - w0_real * x_imag * (float) w0_ad +
                    w1_imag * x_real * (float) w1_bc - w1_real * x_imag * (float) w1_ad;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
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

/**
 * Tiled Fairy2i tile64 matmul. One work-group computes an output tile from SoA
 * FAIRY2I_TILE64 weights and SoA q16 activation staging buffers.
 */
kernel void kernel_fairy2i_tile64_mul_mat_f32_act_q16_64(
        global uchar * w_q,
        global half  * w_d,
        global char  * act_q,
        global half  * act_d,
        global char  * dst,
        ulong         offsetd,
        int           k,
        int           m,
        int           n,
        ulong         nb0,
        ulong         nb1,
        local float * tmp_real,
        local float * tmp_imag,
        local char  * act_real_tile,
        local char  * act_imag_tile
) {
    dst = dst + offsetd;

    const int row_base = get_group_id(0) * FAIRY2I_TILE64_TILE_M;
    const int col_base = get_group_id(1) * FAIRY2I_TILE64_TILE_N;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int block_slot = lid >> 4;
    const int lane = lid & 15;
    const int nb64 = k / QK_FAIRY2I_TILE64;
    const int nbq = k / QK_FAIRY2I_ACT_Q16_64;

    float acc_real[FAIRY2I_TILE64_TILE_OUT];
    float acc_imag[FAIRY2I_TILE64_TILE_OUT];
#pragma unroll
    for (int i = 0; i < FAIRY2I_TILE64_TILE_OUT; ++i) {
        acc_real[i] = 0.0f;
        acc_imag[i] = 0.0f;
    }

    for (int wb_base = 0; wb_base < nb64; wb_base += 4) {
        const int wb = wb_base + block_slot;
        const int act_block = wb;

        if (wb < nb64) {
#pragma unroll
            for (int tc = 0; tc < FAIRY2I_TILE64_TILE_N; ++tc) {
                const int col = col_base + tc;
                const int local_base = (block_slot * FAIRY2I_TILE64_TILE_N + tc) * QK_FAIRY2I_TILE64;
                if (col < n) {
                    const int act_index = col * nbq + act_block;
                    const int q_base = act_index * (2 * QK_FAIRY2I_ACT_Q16_64);
#pragma unroll
                    for (int part = 0; part < 4; ++part) {
                        const int j = lane + 16 * part;
                        act_real_tile[local_base + j] = act_q[q_base + j];
                        act_imag_tile[local_base + j] = act_q[q_base + QK_FAIRY2I_ACT_Q16_64 + j];
                    }
                } else {
#pragma unroll
                    for (int part = 0; part < 4; ++part) {
                        const int j = lane + 16 * part;
                        act_real_tile[local_base + j] = 0;
                        act_imag_tile[local_base + j] = 0;
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (wb < nb64) {
            float x_real[FAIRY2I_TILE64_TILE_N];
            float x_imag[FAIRY2I_TILE64_TILE_N];
#pragma unroll
            for (int tc = 0; tc < FAIRY2I_TILE64_TILE_N; ++tc) {
                const int col = col_base + tc;
                if (col < n) {
                    const int act_index = col * nbq + act_block;
                    x_real[tc] = vload_half(act_index * 2 + 0, act_d);
                    x_imag[tc] = vload_half(act_index * 2 + 1, act_d);
                } else {
                    x_real[tc] = 0.0f;
                    x_imag[tc] = 0.0f;
                }
            }

#pragma unroll
            for (int tr = 0; tr < FAIRY2I_TILE64_TILE_M; ++tr) {
                const int row = row_base + tr;
                if (row >= m) {
                    continue;
                }

                const int w_block = row * nb64 + wb;
                const uint packed = (uint) w_q[w_block * 16 + lane];
                const float w_real = vload_half(w_block * 2 + 0, w_d);
                const float w_imag = vload_half(w_block * 2 + 1, w_d);

                int sum_ac[FAIRY2I_TILE64_TILE_N];
                int sum_ad[FAIRY2I_TILE64_TILE_N];
                int sum_bc[FAIRY2I_TILE64_TILE_N];
                int sum_bd[FAIRY2I_TILE64_TILE_N];
#pragma unroll
                for (int tc = 0; tc < FAIRY2I_TILE64_TILE_N; ++tc) {
                    sum_ac[tc] = 0;
                    sum_ad[tc] = 0;
                    sum_bc[tc] = 0;
                    sum_bd[tc] = 0;
                }

#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    const int j = lane + 16 * part;
                    const uint code = (packed >> (2 * part)) & 3U;
                    int wr = 0;
                    int wi = 0;
                    if (code == 0U) {
                        wr = -1;
                    } else if (code == 1U) {
                        wr = 1;
                    } else if (code == 2U) {
                        wi = -1;
                    } else {
                        wi = 1;
                    }

#pragma unroll
                    for (int tc = 0; tc < FAIRY2I_TILE64_TILE_N; ++tc) {
                        const int local_base = (block_slot * FAIRY2I_TILE64_TILE_N + tc) * QK_FAIRY2I_TILE64;
                        const int xr = (int) act_real_tile[local_base + j];
                        const int xi = (int) act_imag_tile[local_base + j];

                        sum_ac[tc] += xr * wr;
                        sum_ad[tc] += xi * wr;
                        sum_bc[tc] += xr * wi;
                        sum_bd[tc] += xi * wi;
                    }
                }

#pragma unroll
                for (int tc = 0; tc < FAIRY2I_TILE64_TILE_N; ++tc) {
                    const int out_idx = tr * FAIRY2I_TILE64_TILE_N + tc;
                    acc_real[out_idx] += w_real * x_real[tc] * (float) sum_ac[tc] +
                                         w_imag * x_imag[tc] * (float) sum_bd[tc];
                    acc_imag[out_idx] += w_imag * x_real[tc] * (float) sum_bc[tc] -
                                         w_real * x_imag[tc] * (float) sum_ad[tc];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

#pragma unroll
    for (int out_idx = 0; out_idx < FAIRY2I_TILE64_TILE_OUT; ++out_idx) {
        tmp_real[out_idx * lsize + lid] = acc_real[out_idx];
        tmp_imag[out_idx * lsize + lid] = acc_imag[out_idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
#pragma unroll
            for (int out_idx = 0; out_idx < FAIRY2I_TILE64_TILE_OUT; ++out_idx) {
                tmp_real[out_idx * lsize + lid] += tmp_real[out_idx * lsize + lid + stride];
                tmp_imag[out_idx * lsize + lid] += tmp_imag[out_idx * lsize + lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
#pragma unroll
        for (int tr = 0; tr < FAIRY2I_TILE64_TILE_M; ++tr) {
            const int row = row_base + tr;
            if (row >= m) {
                continue;
            }
#pragma unroll
            for (int tc = 0; tc < FAIRY2I_TILE64_TILE_N; ++tc) {
                const int col = col_base + tc;
                if (col >= n) {
                    continue;
                }
                const int out_idx = tr * FAIRY2I_TILE64_TILE_N + tc;
                *((global uint *) (dst + (ulong) col * nb1 + (ulong) row * nb0)) =
                    fairy2i_tile64_pack_bf16_pair(tmp_real[out_idx * lsize], tmp_imag[out_idx * lsize]);
            }
        }
    }
}

/**
 * Tiled Fairy2i tile64 matmul without activation quantization. We keep the SoA
 * Fairy2i tile64 weight layout and the 2x2 output tile, but read packed-BF16
 * activation values directly from the F32 carrier tensor.
 */
kernel void kernel_fairy2i_tile64_mul_mat_f32_direct(
        global uchar * w_q,
        global half  * w_d,
        global char  * src,
        ulong         offset1,
        global char  * dst,
        ulong         offsetd,
        int           k,
        int           m,
        int           n,
        ulong         nb10,
        ulong         nb11,
        ulong         nb0,
        ulong         nb1,
        local float * tmp_real,
        local float * tmp_imag,
        local float * act_real_tile,
        local float * act_imag_tile
) {
    src = src + offset1;
    dst = dst + offsetd;

    const int row_base = get_group_id(0) * FAIRY2I_TILE64_TILE_M;
    const int col_base = get_group_id(1) * FAIRY2I_TILE64_TILE_N;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int block_slot = lid >> 4;
    const int lane = lid & 15;
    const int nb64 = k / QK_FAIRY2I_TILE64;

    float acc_real[FAIRY2I_TILE64_TILE_OUT];
    float acc_imag[FAIRY2I_TILE64_TILE_OUT];
#pragma unroll
    for (int i = 0; i < FAIRY2I_TILE64_TILE_OUT; ++i) {
        acc_real[i] = 0.0f;
        acc_imag[i] = 0.0f;
    }

    for (int wb_base = 0; wb_base < nb64; wb_base += 4) {
        const int wb = wb_base + block_slot;

        if (wb < nb64) {
#pragma unroll
            for (int tc = 0; tc < FAIRY2I_TILE64_TILE_N; ++tc) {
                const int col = col_base + tc;
                const int local_base = (block_slot * FAIRY2I_TILE64_TILE_N + tc) * QK_FAIRY2I_TILE64;
                if (col < n) {
#pragma unroll
                    for (int part = 0; part < 4; ++part) {
                        const int j = lane + 16 * part;
                        const int k_idx = wb * QK_FAIRY2I_TILE64 + j;
                        const uint pair = *((global uint *) (src + (ulong) col * nb11 + (ulong) k_idx * nb10));
                        act_real_tile[local_base + j] = fairy2i_bf16_to_f32((ushort) (pair & 0xffffU));
                        act_imag_tile[local_base + j] = fairy2i_bf16_to_f32((ushort) (pair >> 16));
                    }
                } else {
#pragma unroll
                    for (int part = 0; part < 4; ++part) {
                        const int j = lane + 16 * part;
                        act_real_tile[local_base + j] = 0.0f;
                        act_imag_tile[local_base + j] = 0.0f;
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (wb < nb64) {
#pragma unroll
            for (int tr = 0; tr < FAIRY2I_TILE64_TILE_M; ++tr) {
                const int row = row_base + tr;
                if (row >= m) {
                    continue;
                }

                const int w_block = row * nb64 + wb;
                const uint packed = (uint) w_q[w_block * 16 + lane];
                const float w_real = vload_half(w_block * 2 + 0, w_d);
                const float w_imag = vload_half(w_block * 2 + 1, w_d);

                float sum_ac[FAIRY2I_TILE64_TILE_N];
                float sum_ad[FAIRY2I_TILE64_TILE_N];
                float sum_bc[FAIRY2I_TILE64_TILE_N];
                float sum_bd[FAIRY2I_TILE64_TILE_N];
#pragma unroll
                for (int tc = 0; tc < FAIRY2I_TILE64_TILE_N; ++tc) {
                    sum_ac[tc] = 0.0f;
                    sum_ad[tc] = 0.0f;
                    sum_bc[tc] = 0.0f;
                    sum_bd[tc] = 0.0f;
                }

#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    const int j = lane + 16 * part;
                    const uint code = (packed >> (2 * part)) & 3U;
                    float wr = 0.0f;
                    float wi = 0.0f;
                    if (code == 0U) {
                        wr = -1.0f;
                    } else if (code == 1U) {
                        wr = 1.0f;
                    } else if (code == 2U) {
                        wi = -1.0f;
                    } else {
                        wi = 1.0f;
                    }

#pragma unroll
                    for (int tc = 0; tc < FAIRY2I_TILE64_TILE_N; ++tc) {
                        const int local_base = (block_slot * FAIRY2I_TILE64_TILE_N + tc) * QK_FAIRY2I_TILE64;
                        const float xr = act_real_tile[local_base + j];
                        const float xi = act_imag_tile[local_base + j];

                        sum_ac[tc] += xr * wr;
                        sum_ad[tc] += xi * wr;
                        sum_bc[tc] += xr * wi;
                        sum_bd[tc] += xi * wi;
                    }
                }

#pragma unroll
                for (int tc = 0; tc < FAIRY2I_TILE64_TILE_N; ++tc) {
                    const int out_idx = tr * FAIRY2I_TILE64_TILE_N + tc;
                    acc_real[out_idx] += w_real * sum_ac[tc] + w_imag * sum_bd[tc];
                    acc_imag[out_idx] += w_imag * sum_bc[tc] - w_real * sum_ad[tc];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

#pragma unroll
    for (int out_idx = 0; out_idx < FAIRY2I_TILE64_TILE_OUT; ++out_idx) {
        tmp_real[out_idx * lsize + lid] = acc_real[out_idx];
        tmp_imag[out_idx * lsize + lid] = acc_imag[out_idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
#pragma unroll
            for (int out_idx = 0; out_idx < FAIRY2I_TILE64_TILE_OUT; ++out_idx) {
                tmp_real[out_idx * lsize + lid] += tmp_real[out_idx * lsize + lid + stride];
                tmp_imag[out_idx * lsize + lid] += tmp_imag[out_idx * lsize + lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
#pragma unroll
        for (int tr = 0; tr < FAIRY2I_TILE64_TILE_M; ++tr) {
            const int row = row_base + tr;
            if (row >= m) {
                continue;
            }
#pragma unroll
            for (int tc = 0; tc < FAIRY2I_TILE64_TILE_N; ++tc) {
                const int col = col_base + tc;
                if (col >= n) {
                    continue;
                }
                const int out_idx = tr * FAIRY2I_TILE64_TILE_N + tc;
                *((global uint *) (dst + (ulong) col * nb1 + (ulong) row * nb0)) =
                    fairy2i_tile64_pack_bf16_pair(tmp_real[out_idx * lsize], tmp_imag[out_idx * lsize]);
            }
        }
    }
}

/**
 * Fairy2i tile64 GEMV kernel. Each work-group computes one activation column and a
 * small row tile, so it can be used either for N == 1 routing or direct GEMV
 * benchmarking across multiple columns.
 */
kernel void kernel_fairy2i_tile64_mul_vec_f32_act_q16_64(
        global uchar * w_q,
        global half  * w_d,
        global char  * act_q,
        global half  * act_d,
        global char  * dst,
        ulong         offsetd,
        int           k,
        int           m,
        int           n,
        ulong         nb0,
        ulong         nb1,
        local float * tmp_real,
        local float * tmp_imag,
        local char  * act_real_tile,
        local char  * act_imag_tile
) {
    dst = dst + offsetd;

    const int row_base = get_group_id(0) * FAIRY2I_TILE64_GEMV_TILE_M;
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

    float acc_real[FAIRY2I_TILE64_GEMV_TILE_M];
    float acc_imag[FAIRY2I_TILE64_GEMV_TILE_M];
#pragma unroll
    for (int i = 0; i < FAIRY2I_TILE64_GEMV_TILE_M; ++i) {
        acc_real[i] = 0.0f;
        acc_imag[i] = 0.0f;
    }

    for (int wb_base = 0; wb_base < nb64; wb_base += 4) {
        const int wb = wb_base + block_slot;
        const int act_block = wb;

        if (wb < nb64) {
            const int act_index = col * nbq + act_block;
            const int q_base = act_index * (2 * QK_FAIRY2I_ACT_Q16_64);
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int j = lane + 16 * part;
                act_real_tile[block_slot * QK_FAIRY2I_TILE64 + j] = act_q[q_base + j];
                act_imag_tile[block_slot * QK_FAIRY2I_TILE64 + j] = act_q[q_base + QK_FAIRY2I_ACT_Q16_64 + j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (wb < nb64) {
            const int act_index = col * nbq + act_block;
            const float x_real = vload_half(act_index * 2 + 0, act_d);
            const float x_imag = vload_half(act_index * 2 + 1, act_d);

#pragma unroll
            for (int tr = 0; tr < FAIRY2I_TILE64_GEMV_TILE_M; ++tr) {
                const int row = row_base + tr;
                if (row >= m) {
                    continue;
                }

                const int w_block = row * nb64 + wb;
                const uint packed = (uint) w_q[w_block * 16 + lane];
                const float w_real = vload_half(w_block * 2 + 0, w_d);
                const float w_imag = vload_half(w_block * 2 + 1, w_d);

                int sum_ac = 0;
                int sum_ad = 0;
                int sum_bc = 0;
                int sum_bd = 0;

#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    const int j = lane + 16 * part;
                    const uint code = (packed >> (2 * part)) & 3U;
                    int wr = 0;
                    int wi = 0;
                    if (code == 0U) {
                        wr = -1;
                    } else if (code == 1U) {
                        wr = 1;
                    } else if (code == 2U) {
                        wi = -1;
                    } else {
                        wi = 1;
                    }

                    const int xr = (int) act_real_tile[block_slot * QK_FAIRY2I_TILE64 + j];
                    const int xi = (int) act_imag_tile[block_slot * QK_FAIRY2I_TILE64 + j];

                    sum_ac += xr * wr;
                    sum_ad += xi * wr;
                    sum_bc += xr * wi;
                    sum_bd += xi * wi;
                }

                acc_real[tr] += w_real * x_real * (float) sum_ac +
                                w_imag * x_imag * (float) sum_bd;
                acc_imag[tr] += w_imag * x_real * (float) sum_bc -
                                w_real * x_imag * (float) sum_ad;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

#pragma unroll
    for (int tr = 0; tr < FAIRY2I_TILE64_GEMV_TILE_M; ++tr) {
        tmp_real[tr * lsize + lid] = acc_real[tr];
        tmp_imag[tr * lsize + lid] = acc_imag[tr];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
#pragma unroll
            for (int tr = 0; tr < FAIRY2I_TILE64_GEMV_TILE_M; ++tr) {
                tmp_real[tr * lsize + lid] += tmp_real[tr * lsize + lid + stride];
                tmp_imag[tr * lsize + lid] += tmp_imag[tr * lsize + lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
#pragma unroll
        for (int tr = 0; tr < FAIRY2I_TILE64_GEMV_TILE_M; ++tr) {
            const int row = row_base + tr;
            if (row < m) {
                *((global uint *) (dst + (ulong) col * nb1 + (ulong) row * nb0)) =
                    fairy2i_tile64_pack_bf16_pair(tmp_real[tr * lsize], tmp_imag[tr * lsize]);
            }
        }
    }
}

/**
 * Fairy2i tile64 GEMV kernel with a 4-row tile. This is intentionally separate from
 * the 2-row kernel so the compiler can unroll fixed-size accumulators.
 */
kernel void kernel_fairy2i_tile64_mul_vec4_f32_act_q16_64(
        global uchar * w_q,
        global half  * w_d,
        global char  * act_q,
        global half  * act_d,
        global char  * dst,
        ulong         offsetd,
        int           k,
        int           m,
        int           n,
        ulong         nb0,
        ulong         nb1,
        local float * tmp_real,
        local float * tmp_imag,
        local char  * act_real_tile,
        local char  * act_imag_tile
) {
    dst = dst + offsetd;

    const int row_base = get_group_id(0) * FAIRY2I_TILE64_GEMV4_TILE_M;
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

    float acc_real[FAIRY2I_TILE64_GEMV4_TILE_M];
    float acc_imag[FAIRY2I_TILE64_GEMV4_TILE_M];
#pragma unroll
    for (int i = 0; i < FAIRY2I_TILE64_GEMV4_TILE_M; ++i) {
        acc_real[i] = 0.0f;
        acc_imag[i] = 0.0f;
    }

    for (int wb_base = 0; wb_base < nb64; wb_base += 4) {
        const int wb = wb_base + block_slot;
        const int act_block = wb;

        if (wb < nb64) {
            const int act_index = col * nbq + act_block;
            const int q_base = act_index * (2 * QK_FAIRY2I_ACT_Q16_64);
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int j = lane + 16 * part;
                act_real_tile[block_slot * QK_FAIRY2I_TILE64 + j] = act_q[q_base + j];
                act_imag_tile[block_slot * QK_FAIRY2I_TILE64 + j] = act_q[q_base + QK_FAIRY2I_ACT_Q16_64 + j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (wb < nb64) {
            const int act_index = col * nbq + act_block;
            const float x_real = vload_half(act_index * 2 + 0, act_d);
            const float x_imag = vload_half(act_index * 2 + 1, act_d);

#pragma unroll
            for (int tr = 0; tr < FAIRY2I_TILE64_GEMV4_TILE_M; ++tr) {
                const int row = row_base + tr;
                if (row >= m) {
                    continue;
                }

                const int w_block = row * nb64 + wb;
                const uint packed = (uint) w_q[w_block * 16 + lane];
                const float w_real = vload_half(w_block * 2 + 0, w_d);
                const float w_imag = vload_half(w_block * 2 + 1, w_d);

                int sum_ac = 0;
                int sum_ad = 0;
                int sum_bc = 0;
                int sum_bd = 0;

#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    const int j = lane + 16 * part;
                    const uint code = (packed >> (2 * part)) & 3U;
                    int wr = 0;
                    int wi = 0;
                    if (code == 0U) {
                        wr = -1;
                    } else if (code == 1U) {
                        wr = 1;
                    } else if (code == 2U) {
                        wi = -1;
                    } else {
                        wi = 1;
                    }

                    const int xr = (int) act_real_tile[block_slot * QK_FAIRY2I_TILE64 + j];
                    const int xi = (int) act_imag_tile[block_slot * QK_FAIRY2I_TILE64 + j];

                    sum_ac += xr * wr;
                    sum_ad += xi * wr;
                    sum_bc += xr * wi;
                    sum_bd += xi * wi;
                }

                acc_real[tr] += w_real * x_real * (float) sum_ac +
                                w_imag * x_imag * (float) sum_bd;
                acc_imag[tr] += w_imag * x_real * (float) sum_bc -
                                w_real * x_imag * (float) sum_ad;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

#pragma unroll
    for (int tr = 0; tr < FAIRY2I_TILE64_GEMV4_TILE_M; ++tr) {
        tmp_real[tr * lsize + lid] = acc_real[tr];
        tmp_imag[tr * lsize + lid] = acc_imag[tr];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
#pragma unroll
            for (int tr = 0; tr < FAIRY2I_TILE64_GEMV4_TILE_M; ++tr) {
                tmp_real[tr * lsize + lid] += tmp_real[tr * lsize + lid + stride];
                tmp_imag[tr * lsize + lid] += tmp_imag[tr * lsize + lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
#pragma unroll
        for (int tr = 0; tr < FAIRY2I_TILE64_GEMV4_TILE_M; ++tr) {
            const int row = row_base + tr;
            if (row < m) {
                *((global uint *) (dst + (ulong) col * nb1 + (ulong) row * nb0)) =
                    fairy2i_tile64_pack_bf16_pair(tmp_real[tr * lsize], tmp_imag[tr * lsize]);
            }
        }
    }
}
