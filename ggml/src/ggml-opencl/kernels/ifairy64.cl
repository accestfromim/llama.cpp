#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK_IFAIRY_Q16 256
#define QK_IFAIRY64   64

#define IFAIRY64_TILE_M 2
#define IFAIRY64_TILE_N 2
#define IFAIRY64_TILE_OUT (IFAIRY64_TILE_M * IFAIRY64_TILE_N)

#define IFAIRY64_GEMV_TILE_M 2
#define IFAIRY64_GEMV4_TILE_M 4

static inline float ifairy_bf16_to_f32(ushort h) {
    return as_float(((uint) h) << 16);
}

static inline ushort ifairy_f32_to_bf16(float x) {
    uint bits = as_uint(x);
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return (ushort) ((bits >> 16) | 64U);
    }
    return (ushort) ((bits + (0x7fffU + ((bits >> 16) & 1U))) >> 16);
}

static inline int ifairy_clamp_i32(int v, int lo, int hi) {
    return min(hi, max(lo, v));
}

static inline uint ifairy64_pack_bf16_pair(float real, float imag) {
    const uint r = (uint) ifairy_f32_to_bf16(real);
    const uint i = (uint) ifairy_f32_to_bf16(imag);
    return r | (i << 16);
}

/**
 * Converts packed-BF16 iFairy activations from F32 carrier format to a SoA
 * q16 staging layout. The q buffer stores real and imaginary int8 planes for
 * each 256-value block; d stores the corresponding fp16 scale pair.
 */
kernel void kernel_ifairy_q16_quantize_block127(
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
    const int blocks_per_col = k / QK_IFAIRY_Q16;

    if (col >= n || block >= blocks_per_col) {
        return;
    }

    float max_real = 1.0e-5f;
    float max_imag = 1.0e-5f;
    for (int j = lid; j < QK_IFAIRY_Q16; j += lsize) {
        const int k_idx = block * QK_IFAIRY_Q16 + j;
        const uint pair = *((global uint *) (src + (ulong) col * nb11 + (ulong) k_idx * nb10));
        const float xr = ifairy_bf16_to_f32((ushort) (pair & 0xffffU));
        const float xi = ifairy_bf16_to_f32((ushort) (pair >> 16));
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

    const int q_base = block_index * (2 * QK_IFAIRY_Q16);
    for (int j = lid; j < QK_IFAIRY_Q16; j += lsize) {
        const int k_idx = block * QK_IFAIRY_Q16 + j;
        const uint pair = *((global uint *) (src + (ulong) col * nb11 + (ulong) k_idx * nb10));
        const float xr = ifairy_bf16_to_f32((ushort) (pair & 0xffffU));
        const float xi = ifairy_bf16_to_f32((ushort) (pair >> 16));

        const int qr = ifairy_clamp_i32((int) rint(xr * iscale_real), -127, 127);
        const int qi = ifairy_clamp_i32((int) rint(xi * iscale_imag), -127, 127);

        act_q[q_base + j] = (char) qr;
        act_q[q_base + QK_IFAIRY_Q16 + j] = (char) qi;
    }
}

/**
 * Tiled iFairy64 matmul. One work-group computes an output tile from SoA
 * IFAIRY64 weights and SoA q16 activation staging buffers.
 */
kernel void kernel_ifairy64_mul_mat_f32_q16(
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

    const int row_base = get_group_id(0) * IFAIRY64_TILE_M;
    const int col_base = get_group_id(1) * IFAIRY64_TILE_N;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int block_slot = lid >> 4;
    const int lane = lid & 15;
    const int nb64 = k / QK_IFAIRY64;
    const int nbq = k / QK_IFAIRY_Q16;

    float acc_real[IFAIRY64_TILE_OUT];
    float acc_imag[IFAIRY64_TILE_OUT];
#pragma unroll
    for (int i = 0; i < IFAIRY64_TILE_OUT; ++i) {
        acc_real[i] = 0.0f;
        acc_imag[i] = 0.0f;
    }

    for (int wb_base = 0; wb_base < nb64; wb_base += 4) {
        const int wb = wb_base + block_slot;
        const int act_block = wb >> 2;
        const int act_base = (wb & 3) * QK_IFAIRY64;

        if (wb < nb64) {
#pragma unroll
            for (int tc = 0; tc < IFAIRY64_TILE_N; ++tc) {
                const int col = col_base + tc;
                const int local_base = (block_slot * IFAIRY64_TILE_N + tc) * QK_IFAIRY64;
                if (col < n) {
                    const int act_index = col * nbq + act_block;
                    const int q_base = act_index * (2 * QK_IFAIRY_Q16) + act_base;
#pragma unroll
                    for (int part = 0; part < 4; ++part) {
                        const int j = lane + 16 * part;
                        act_real_tile[local_base + j] = act_q[q_base + j];
                        act_imag_tile[local_base + j] = act_q[q_base + QK_IFAIRY_Q16 + j];
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
            float x_real[IFAIRY64_TILE_N];
            float x_imag[IFAIRY64_TILE_N];
#pragma unroll
            for (int tc = 0; tc < IFAIRY64_TILE_N; ++tc) {
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
            for (int tr = 0; tr < IFAIRY64_TILE_M; ++tr) {
                const int row = row_base + tr;
                if (row >= m) {
                    continue;
                }

                const int w_block = row * nb64 + wb;
                const uint packed = (uint) w_q[w_block * 16 + lane];
                const float w_real = vload_half(w_block * 2 + 0, w_d);
                const float w_imag = vload_half(w_block * 2 + 1, w_d);

                int sum_ac[IFAIRY64_TILE_N];
                int sum_ad[IFAIRY64_TILE_N];
                int sum_bc[IFAIRY64_TILE_N];
                int sum_bd[IFAIRY64_TILE_N];
#pragma unroll
                for (int tc = 0; tc < IFAIRY64_TILE_N; ++tc) {
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
                    for (int tc = 0; tc < IFAIRY64_TILE_N; ++tc) {
                        const int local_base = (block_slot * IFAIRY64_TILE_N + tc) * QK_IFAIRY64;
                        const int xr = (int) act_real_tile[local_base + j];
                        const int xi = (int) act_imag_tile[local_base + j];

                        sum_ac[tc] += xr * wr;
                        sum_ad[tc] += xi * wr;
                        sum_bc[tc] += xr * wi;
                        sum_bd[tc] += xi * wi;
                    }
                }

#pragma unroll
                for (int tc = 0; tc < IFAIRY64_TILE_N; ++tc) {
                    const int out_idx = tr * IFAIRY64_TILE_N + tc;
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
    for (int out_idx = 0; out_idx < IFAIRY64_TILE_OUT; ++out_idx) {
        tmp_real[out_idx * lsize + lid] = acc_real[out_idx];
        tmp_imag[out_idx * lsize + lid] = acc_imag[out_idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
#pragma unroll
            for (int out_idx = 0; out_idx < IFAIRY64_TILE_OUT; ++out_idx) {
                tmp_real[out_idx * lsize + lid] += tmp_real[out_idx * lsize + lid + stride];
                tmp_imag[out_idx * lsize + lid] += tmp_imag[out_idx * lsize + lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
#pragma unroll
        for (int tr = 0; tr < IFAIRY64_TILE_M; ++tr) {
            const int row = row_base + tr;
            if (row >= m) {
                continue;
            }
#pragma unroll
            for (int tc = 0; tc < IFAIRY64_TILE_N; ++tc) {
                const int col = col_base + tc;
                if (col >= n) {
                    continue;
                }
                const int out_idx = tr * IFAIRY64_TILE_N + tc;
                *((global uint *) (dst + (ulong) col * nb1 + (ulong) row * nb0)) =
                    ifairy64_pack_bf16_pair(tmp_real[out_idx * lsize], tmp_imag[out_idx * lsize]);
            }
        }
    }
}

/**
 * Tiled iFairy64 matmul without activation quantization. We keep the SoA
 * iFairy64 weight layout and the 2x2 output tile, but read packed-BF16
 * activation values directly from the F32 carrier tensor.
 */
kernel void kernel_ifairy64_mul_mat_f32_direct(
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

    const int row_base = get_group_id(0) * IFAIRY64_TILE_M;
    const int col_base = get_group_id(1) * IFAIRY64_TILE_N;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int block_slot = lid >> 4;
    const int lane = lid & 15;
    const int nb64 = k / QK_IFAIRY64;

    float acc_real[IFAIRY64_TILE_OUT];
    float acc_imag[IFAIRY64_TILE_OUT];
#pragma unroll
    for (int i = 0; i < IFAIRY64_TILE_OUT; ++i) {
        acc_real[i] = 0.0f;
        acc_imag[i] = 0.0f;
    }

    for (int wb_base = 0; wb_base < nb64; wb_base += 4) {
        const int wb = wb_base + block_slot;

        if (wb < nb64) {
#pragma unroll
            for (int tc = 0; tc < IFAIRY64_TILE_N; ++tc) {
                const int col = col_base + tc;
                const int local_base = (block_slot * IFAIRY64_TILE_N + tc) * QK_IFAIRY64;
                if (col < n) {
#pragma unroll
                    for (int part = 0; part < 4; ++part) {
                        const int j = lane + 16 * part;
                        const int k_idx = wb * QK_IFAIRY64 + j;
                        const uint pair = *((global uint *) (src + (ulong) col * nb11 + (ulong) k_idx * nb10));
                        act_real_tile[local_base + j] = ifairy_bf16_to_f32((ushort) (pair & 0xffffU));
                        act_imag_tile[local_base + j] = ifairy_bf16_to_f32((ushort) (pair >> 16));
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
            for (int tr = 0; tr < IFAIRY64_TILE_M; ++tr) {
                const int row = row_base + tr;
                if (row >= m) {
                    continue;
                }

                const int w_block = row * nb64 + wb;
                const uint packed = (uint) w_q[w_block * 16 + lane];
                const float w_real = vload_half(w_block * 2 + 0, w_d);
                const float w_imag = vload_half(w_block * 2 + 1, w_d);

                float sum_ac[IFAIRY64_TILE_N];
                float sum_ad[IFAIRY64_TILE_N];
                float sum_bc[IFAIRY64_TILE_N];
                float sum_bd[IFAIRY64_TILE_N];
#pragma unroll
                for (int tc = 0; tc < IFAIRY64_TILE_N; ++tc) {
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
                    for (int tc = 0; tc < IFAIRY64_TILE_N; ++tc) {
                        const int local_base = (block_slot * IFAIRY64_TILE_N + tc) * QK_IFAIRY64;
                        const float xr = act_real_tile[local_base + j];
                        const float xi = act_imag_tile[local_base + j];

                        sum_ac[tc] += xr * wr;
                        sum_ad[tc] += xi * wr;
                        sum_bc[tc] += xr * wi;
                        sum_bd[tc] += xi * wi;
                    }
                }

#pragma unroll
                for (int tc = 0; tc < IFAIRY64_TILE_N; ++tc) {
                    const int out_idx = tr * IFAIRY64_TILE_N + tc;
                    acc_real[out_idx] += w_real * sum_ac[tc] + w_imag * sum_bd[tc];
                    acc_imag[out_idx] += w_imag * sum_bc[tc] - w_real * sum_ad[tc];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

#pragma unroll
    for (int out_idx = 0; out_idx < IFAIRY64_TILE_OUT; ++out_idx) {
        tmp_real[out_idx * lsize + lid] = acc_real[out_idx];
        tmp_imag[out_idx * lsize + lid] = acc_imag[out_idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
#pragma unroll
            for (int out_idx = 0; out_idx < IFAIRY64_TILE_OUT; ++out_idx) {
                tmp_real[out_idx * lsize + lid] += tmp_real[out_idx * lsize + lid + stride];
                tmp_imag[out_idx * lsize + lid] += tmp_imag[out_idx * lsize + lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
#pragma unroll
        for (int tr = 0; tr < IFAIRY64_TILE_M; ++tr) {
            const int row = row_base + tr;
            if (row >= m) {
                continue;
            }
#pragma unroll
            for (int tc = 0; tc < IFAIRY64_TILE_N; ++tc) {
                const int col = col_base + tc;
                if (col >= n) {
                    continue;
                }
                const int out_idx = tr * IFAIRY64_TILE_N + tc;
                *((global uint *) (dst + (ulong) col * nb1 + (ulong) row * nb0)) =
                    ifairy64_pack_bf16_pair(tmp_real[out_idx * lsize], tmp_imag[out_idx * lsize]);
            }
        }
    }
}

/**
 * iFairy64 GEMV kernel. Each work-group computes one activation column and a
 * small row tile, so it can be used either for N == 1 routing or direct GEMV
 * benchmarking across multiple columns.
 */
kernel void kernel_ifairy64_mul_vec_f32_q16(
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

    const int row_base = get_group_id(0) * IFAIRY64_GEMV_TILE_M;
    const int col = get_group_id(1);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int block_slot = lid >> 4;
    const int lane = lid & 15;
    const int nb64 = k / QK_IFAIRY64;
    const int nbq = k / QK_IFAIRY_Q16;

    if (col >= n) {
        return;
    }

    float acc_real[IFAIRY64_GEMV_TILE_M];
    float acc_imag[IFAIRY64_GEMV_TILE_M];
#pragma unroll
    for (int i = 0; i < IFAIRY64_GEMV_TILE_M; ++i) {
        acc_real[i] = 0.0f;
        acc_imag[i] = 0.0f;
    }

    for (int wb_base = 0; wb_base < nb64; wb_base += 4) {
        const int wb = wb_base + block_slot;
        const int act_block = wb >> 2;
        const int act_base = (wb & 3) * QK_IFAIRY64;

        if (wb < nb64) {
            const int act_index = col * nbq + act_block;
            const int q_base = act_index * (2 * QK_IFAIRY_Q16) + act_base;
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int j = lane + 16 * part;
                act_real_tile[block_slot * QK_IFAIRY64 + j] = act_q[q_base + j];
                act_imag_tile[block_slot * QK_IFAIRY64 + j] = act_q[q_base + QK_IFAIRY_Q16 + j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (wb < nb64) {
            const int act_index = col * nbq + act_block;
            const float x_real = vload_half(act_index * 2 + 0, act_d);
            const float x_imag = vload_half(act_index * 2 + 1, act_d);

#pragma unroll
            for (int tr = 0; tr < IFAIRY64_GEMV_TILE_M; ++tr) {
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

                    const int xr = (int) act_real_tile[block_slot * QK_IFAIRY64 + j];
                    const int xi = (int) act_imag_tile[block_slot * QK_IFAIRY64 + j];

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
    for (int tr = 0; tr < IFAIRY64_GEMV_TILE_M; ++tr) {
        tmp_real[tr * lsize + lid] = acc_real[tr];
        tmp_imag[tr * lsize + lid] = acc_imag[tr];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
#pragma unroll
            for (int tr = 0; tr < IFAIRY64_GEMV_TILE_M; ++tr) {
                tmp_real[tr * lsize + lid] += tmp_real[tr * lsize + lid + stride];
                tmp_imag[tr * lsize + lid] += tmp_imag[tr * lsize + lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
#pragma unroll
        for (int tr = 0; tr < IFAIRY64_GEMV_TILE_M; ++tr) {
            const int row = row_base + tr;
            if (row < m) {
                *((global uint *) (dst + (ulong) col * nb1 + (ulong) row * nb0)) =
                    ifairy64_pack_bf16_pair(tmp_real[tr * lsize], tmp_imag[tr * lsize]);
            }
        }
    }
}

/**
 * iFairy64 GEMV kernel with a 4-row tile. This is intentionally separate from
 * the 2-row kernel so the compiler can unroll fixed-size accumulators.
 */
kernel void kernel_ifairy64_mul_vec4_f32_q16(
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

    const int row_base = get_group_id(0) * IFAIRY64_GEMV4_TILE_M;
    const int col = get_group_id(1);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int block_slot = lid >> 4;
    const int lane = lid & 15;
    const int nb64 = k / QK_IFAIRY64;
    const int nbq = k / QK_IFAIRY_Q16;

    if (col >= n) {
        return;
    }

    float acc_real[IFAIRY64_GEMV4_TILE_M];
    float acc_imag[IFAIRY64_GEMV4_TILE_M];
#pragma unroll
    for (int i = 0; i < IFAIRY64_GEMV4_TILE_M; ++i) {
        acc_real[i] = 0.0f;
        acc_imag[i] = 0.0f;
    }

    for (int wb_base = 0; wb_base < nb64; wb_base += 4) {
        const int wb = wb_base + block_slot;
        const int act_block = wb >> 2;
        const int act_base = (wb & 3) * QK_IFAIRY64;

        if (wb < nb64) {
            const int act_index = col * nbq + act_block;
            const int q_base = act_index * (2 * QK_IFAIRY_Q16) + act_base;
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int j = lane + 16 * part;
                act_real_tile[block_slot * QK_IFAIRY64 + j] = act_q[q_base + j];
                act_imag_tile[block_slot * QK_IFAIRY64 + j] = act_q[q_base + QK_IFAIRY_Q16 + j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (wb < nb64) {
            const int act_index = col * nbq + act_block;
            const float x_real = vload_half(act_index * 2 + 0, act_d);
            const float x_imag = vload_half(act_index * 2 + 1, act_d);

#pragma unroll
            for (int tr = 0; tr < IFAIRY64_GEMV4_TILE_M; ++tr) {
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

                    const int xr = (int) act_real_tile[block_slot * QK_IFAIRY64 + j];
                    const int xi = (int) act_imag_tile[block_slot * QK_IFAIRY64 + j];

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
    for (int tr = 0; tr < IFAIRY64_GEMV4_TILE_M; ++tr) {
        tmp_real[tr * lsize + lid] = acc_real[tr];
        tmp_imag[tr * lsize + lid] = acc_imag[tr];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
#pragma unroll
            for (int tr = 0; tr < IFAIRY64_GEMV4_TILE_M; ++tr) {
                tmp_real[tr * lsize + lid] += tmp_real[tr * lsize + lid + stride];
                tmp_imag[tr * lsize + lid] += tmp_imag[tr * lsize + lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
#pragma unroll
        for (int tr = 0; tr < IFAIRY64_GEMV4_TILE_M; ++tr) {
            const int row = row_base + tr;
            if (row < m) {
                *((global uint *) (dst + (ulong) col * nb1 + (ulong) row * nb0)) =
                    ifairy64_pack_bf16_pair(tmp_real[tr * lsize], tmp_imag[tr * lsize]);
            }
        }
    }
}
