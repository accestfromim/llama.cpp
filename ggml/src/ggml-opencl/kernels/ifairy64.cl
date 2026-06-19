#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK_IFAIRY64     64
#define QK_IFAIRY64_Q16 64

#define IFAIRY64_TILE_M 2
#define IFAIRY64_TILE_N 2
#define IFAIRY64_TILE_OUT (IFAIRY64_TILE_M * IFAIRY64_TILE_N)

#define IFAIRY64_GEMV_TILE_M 2
#define IFAIRY64_GEMV4_TILE_M 4
#define IFAIRY64_GEMV8_TILE_M 8
#define IFAIRY64_GEMV16_TILE_M 16
#define IFAIRY64_WIDE_TILE_M 4
#define IFAIRY64_LUT_TILE_ROWS 16
#define IFAIRY64_LUT_GROUP_PAIRS 16
#define IFAIRY64_LUT_QS_BYTES 256
#define IFAIRY64_LUT_GROUP_BYTES 64
#define IFAIRY64_LUT_WTILE_BYTES 320
#define IFAIRY64_LUT_MAX_REPS 4

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

static inline int2 ifairy64_decode_code(uint code) {
    if (code == 0U) {
        return (int2)(-1, 0);
    }
    if (code == 1U) {
        return (int2)(1, 0);
    }
    if (code == 2U) {
        return (int2)(0, -1);
    }
    return (int2)(0, 1);
}

static inline int2 ifairy64_decode_code_branchless(uint code) {
    const int sign = (int) ((code & 1U) << 1) - 1;
    const int imag = (int) (code >> 1);
    return (int2)(sign * (1 - imag), sign * imag);
}

static inline void ifairy64_accumulate_pair_code(
        uint          pair_code,
        int           xr0,
        int           xi0,
        int           xr1,
        int           xi1,
        private int * sum_ac,
        private int * sum_ad,
        private int * sum_bc,
        private int * sum_bd
) {
    const int2 c0 = ifairy64_decode_code(pair_code & 3U);
    const int2 c1 = ifairy64_decode_code((pair_code >> 2) & 3U);

    *sum_ac += xr0 * c0.x + xr1 * c1.x;
    *sum_ad += xi0 * c0.x + xi1 * c1.x;
    *sum_bc += xr0 * c0.y + xr1 * c1.y;
    *sum_bd += xi0 * c0.y + xi1 * c1.y;
}

static inline char ifairy64_sat_s8(int v) {
    return (char) ifairy_clamp_i32(v, -128, 127);
}

static inline int ifairy64_unpack_s8(uint packed, uint byte_idx) {
    return (int) ((char) ((packed >> (8U * byte_idx)) & 0xffU));
}

static inline void ifairy64_lut_fill_group(
        int           xr0,
        int           xi0,
        int           xr1,
        int           xi1,
        global char * tbl
) {
    for (uint code = 0; code < 16; ++code) {
        const int2 c0 = ifairy64_decode_code(code & 3U);
        const int2 c1 = ifairy64_decode_code((code >> 2) & 3U);

        const int ac = xr0 * c0.x + xr1 * c1.x;
        const int bd = xi0 * c0.y + xi1 * c1.y;
        const int bc = xr0 * c0.y + xr1 * c1.y;
        const int ad = -(xi0 * c0.x + xi1 * c1.x);

        tbl[0  + code] = ifairy64_sat_s8(ac);
        tbl[16 + code] = ifairy64_sat_s8(bd);
        tbl[32 + code] = ifairy64_sat_s8(bc);
        tbl[48 + code] = ifairy64_sat_s8(ad);
    }
}

static inline void ifairy64_lut_fill_group_local(
        int          xr0,
        int          xi0,
        int          xr1,
        int          xi1,
        local char * tbl
) {
    for (uint code = 0; code < 16; ++code) {
        const int2 c0 = ifairy64_decode_code(code & 3U);
        const int2 c1 = ifairy64_decode_code((code >> 2) & 3U);

        const int ac = xr0 * c0.x + xr1 * c1.x;
        const int bd = xi0 * c0.y + xi1 * c1.y;
        const int bc = xr0 * c0.y + xr1 * c1.y;
        const int ad = -(xi0 * c0.x + xi1 * c1.x);

        tbl[0  + code] = ifairy64_sat_s8(ac);
        tbl[16 + code] = ifairy64_sat_s8(bd);
        tbl[32 + code] = ifairy64_sat_s8(bc);
        tbl[48 + code] = ifairy64_sat_s8(ad);
    }
}

static inline void ifairy64_lut_fill_group4(
        int           xr0,
        int           xi0,
        int           xr1,
        int           xi1,
        global char * tbl
) {
    for (uint code = 0; code < 16; ++code) {
        const int2 c0 = ifairy64_decode_code(code & 3U);
        const int2 c1 = ifairy64_decode_code((code >> 2) & 3U);

        const int ac = xr0 * c0.x + xr1 * c1.x;
        const int bd = xi0 * c0.y + xi1 * c1.y;
        const int bc = xr0 * c0.y + xr1 * c1.y;
        const int ad = -(xi0 * c0.x + xi1 * c1.x);

        tbl[4 * code + 0] = ifairy64_sat_s8(ac);
        tbl[4 * code + 1] = ifairy64_sat_s8(bd);
        tbl[4 * code + 2] = ifairy64_sat_s8(bc);
        tbl[4 * code + 3] = ifairy64_sat_s8(ad);
    }
}

/**
 * Converts packed-BF16 iFairy activations from F32 carrier format to the
 * IFAIRY64 q16 staging layout. The q buffer stores real and imaginary int8
 * planes for each 64-value block; d stores the corresponding fp16 scale pair.
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
    const int blocks_per_col = k / QK_IFAIRY64_Q16;

    if (col >= n || block >= blocks_per_col) {
        return;
    }

    float max_real = 1.0e-5f;
    float max_imag = 1.0e-5f;
    for (int j = lid; j < QK_IFAIRY64_Q16; j += lsize) {
        const int k_idx = block * QK_IFAIRY64_Q16 + j;
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

    const int q_base = block_index * (2 * QK_IFAIRY64_Q16);
    for (int j = lid; j < QK_IFAIRY64_Q16; j += lsize) {
        const int k_idx = block * QK_IFAIRY64_Q16 + j;
        const uint pair = *((global uint *) (src + (ulong) col * nb11 + (ulong) k_idx * nb10));
        const float xr = ifairy_bf16_to_f32((ushort) (pair & 0xffffU));
        const float xi = ifairy_bf16_to_f32((ushort) (pair >> 16));

        const int qr = ifairy_clamp_i32((int) rint(xr * iscale_real), -127, 127);
        const int qi = ifairy_clamp_i32((int) rint(xi * iscale_imag), -127, 127);

        act_q[q_base + j] = (char) qr;
        act_q[q_base + QK_IFAIRY64_Q16 + j] = (char) qi;
    }
}

/**
 * Builds the true iFairy64 LUT input for decode-oriented OpenCL kernels.
 * Each work-group handles one K64 activation block and writes 32 groups of
 * 16-entry, 4-channel int8 LUT values plus the real/imag activation scales.
 */
kernel void kernel_ifairy64_lut_preprocess_f32(
        global char  * src,
        ulong          offset,
        global char  * lut,
        global float * lut_scales,
        int            k,
        int            n,
        ulong          nb10,
        ulong          nb11,
        local float  * tmp_real,
        local float  * tmp_imag
) {
    src = src + offset;

    const int block = get_group_id(0);
    const int col   = get_group_id(1);
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);
    const int blocks_per_col = k / QK_IFAIRY64;

    if (col >= n || block >= blocks_per_col) {
        return;
    }

    float max_real = 1.0e-5f;
    float max_imag = 1.0e-5f;
    for (int j = lid; j < QK_IFAIRY64; j += lsize) {
        const int k_idx = block * QK_IFAIRY64 + j;
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

    const float scale_real = tmp_real[0] / 63.0f;
    const float scale_imag = tmp_imag[0] / 63.0f;
    const float iscale_real = 1.0f / scale_real;
    const float iscale_imag = 1.0f / scale_imag;
    const int block_index = col * blocks_per_col + block;

    if (lid == 0) {
        lut_scales[block_index * 2 + 0] = scale_real;
        lut_scales[block_index * 2 + 1] = scale_imag;
    }

    if (lid < IFAIRY64_LUT_GROUP_PAIRS * 2) {
        const int group = lid;
        const int j0 = group * 2 + 0;
        const int j1 = group * 2 + 1;
        const int k0 = block * QK_IFAIRY64 + j0;
        const int k1 = block * QK_IFAIRY64 + j1;
        const uint pair0 = *((global uint *) (src + (ulong) col * nb11 + (ulong) k0 * nb10));
        const uint pair1 = *((global uint *) (src + (ulong) col * nb11 + (ulong) k1 * nb10));

        const int xr0 = ifairy_clamp_i32((int) rint(ifairy_bf16_to_f32((ushort) (pair0 & 0xffffU)) * iscale_real), -63, 63);
        const int xi0 = ifairy_clamp_i32((int) rint(ifairy_bf16_to_f32((ushort) (pair0 >> 16)) * iscale_imag), -63, 63);
        const int xr1 = ifairy_clamp_i32((int) rint(ifairy_bf16_to_f32((ushort) (pair1 & 0xffffU)) * iscale_real), -63, 63);
        const int xi1 = ifairy_clamp_i32((int) rint(ifairy_bf16_to_f32((ushort) (pair1 >> 16)) * iscale_imag), -63, 63);

        global char * tbl = lut + ((block_index * IFAIRY64_LUT_GROUP_PAIRS * 2 + group) * IFAIRY64_LUT_GROUP_BYTES);
        ifairy64_lut_fill_group(xr0, xi0, xr1, xi1, tbl);
    }
}

/**
 * Builds the same activation-side LUT as kernel_ifairy64_lut_preprocess_f32,
 * but stores each 16-entry group as contiguous char4 records. This avoids four
 * separate byte-plane reads in Adreno-oriented LUT consumers.
 */
kernel void kernel_ifairy64_lut_preprocess4_f32(
        global char  * src,
        ulong          offset,
        global char  * lut,
        global float * lut_scales,
        int            k,
        int            n,
        ulong          nb10,
        ulong          nb11,
        local float  * tmp_real,
        local float  * tmp_imag
) {
    src = src + offset;

    const int block = get_group_id(0);
    const int col   = get_group_id(1);
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);
    const int blocks_per_col = k / QK_IFAIRY64;

    if (col >= n || block >= blocks_per_col) {
        return;
    }

    float max_real = 1.0e-5f;
    float max_imag = 1.0e-5f;
    for (int j = lid; j < QK_IFAIRY64; j += lsize) {
        const int k_idx = block * QK_IFAIRY64 + j;
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

    const float scale_real = tmp_real[0] / 63.0f;
    const float scale_imag = tmp_imag[0] / 63.0f;
    const float iscale_real = 1.0f / scale_real;
    const float iscale_imag = 1.0f / scale_imag;
    const int block_index = col * blocks_per_col + block;

    if (lid == 0) {
        lut_scales[block_index * 2 + 0] = scale_real;
        lut_scales[block_index * 2 + 1] = scale_imag;
    }

    if (lid < IFAIRY64_LUT_GROUP_PAIRS * 2) {
        const int group = lid;
        const int j0 = group * 2 + 0;
        const int j1 = group * 2 + 1;
        const int k0 = block * QK_IFAIRY64 + j0;
        const int k1 = block * QK_IFAIRY64 + j1;
        const uint pair0 = *((global uint *) (src + (ulong) col * nb11 + (ulong) k0 * nb10));
        const uint pair1 = *((global uint *) (src + (ulong) col * nb11 + (ulong) k1 * nb10));

        const int xr0 = ifairy_clamp_i32((int) rint(ifairy_bf16_to_f32((ushort) (pair0 & 0xffffU)) * iscale_real), -63, 63);
        const int xi0 = ifairy_clamp_i32((int) rint(ifairy_bf16_to_f32((ushort) (pair0 >> 16)) * iscale_imag), -63, 63);
        const int xr1 = ifairy_clamp_i32((int) rint(ifairy_bf16_to_f32((ushort) (pair1 & 0xffffU)) * iscale_real), -63, 63);
        const int xi1 = ifairy_clamp_i32((int) rint(ifairy_bf16_to_f32((ushort) (pair1 >> 16)) * iscale_imag), -63, 63);

        global char * tbl = lut + ((block_index * IFAIRY64_LUT_GROUP_PAIRS * 2 + group) * IFAIRY64_LUT_GROUP_BYTES);
        ifairy64_lut_fill_group4(xr0, xi0, xr1, xi1, tbl);
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
    const int nbq = k / QK_IFAIRY64_Q16;

    float acc_real[IFAIRY64_TILE_OUT];
    float acc_imag[IFAIRY64_TILE_OUT];
#pragma unroll
    for (int i = 0; i < IFAIRY64_TILE_OUT; ++i) {
        acc_real[i] = 0.0f;
        acc_imag[i] = 0.0f;
    }

    for (int wb_base = 0; wb_base < nb64; wb_base += 4) {
        const int wb = wb_base + block_slot;
        const int act_block = wb;

        if (wb < nb64) {
#pragma unroll
            for (int tc = 0; tc < IFAIRY64_TILE_N; ++tc) {
                const int col = col_base + tc;
                const int local_base = (block_slot * IFAIRY64_TILE_N + tc) * QK_IFAIRY64;
                if (col < n) {
                    const int act_index = col * nbq + act_block;
                    const int q_base = act_index * (2 * QK_IFAIRY64_Q16);
#pragma unroll
                    for (int part = 0; part < 4; ++part) {
                        const int j = lane + 16 * part;
                        act_real_tile[local_base + j] = act_q[q_base + j];
                        act_imag_tile[local_base + j] = act_q[q_base + QK_IFAIRY64_Q16 + j];
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
    const int nbq = k / QK_IFAIRY64_Q16;

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
        const int act_block = wb;

        if (wb < nb64) {
            const int act_index = col * nbq + act_block;
            const int q_base = act_index * (2 * QK_IFAIRY64_Q16);
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int j = lane + 16 * part;
                act_real_tile[block_slot * QK_IFAIRY64 + j] = act_q[q_base + j];
                act_imag_tile[block_slot * QK_IFAIRY64 + j] = act_q[q_base + QK_IFAIRY64_Q16 + j];
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
    const int nbq = k / QK_IFAIRY64_Q16;

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
        const int act_block = wb;

        if (wb < nb64) {
            const int act_index = col * nbq + act_block;
            const int q_base = act_index * (2 * QK_IFAIRY64_Q16);
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int j = lane + 16 * part;
                act_real_tile[block_slot * QK_IFAIRY64 + j] = act_q[q_base + j];
                act_imag_tile[block_slot * QK_IFAIRY64 + j] = act_q[q_base + QK_IFAIRY64_Q16 + j];
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

#define DEFINE_IFAIRY64_MUL_VEC_Q16_KERNEL(KERNEL_NAME, TILE_M) \
kernel void KERNEL_NAME( \
        global uchar * w_q, \
        global half  * w_d, \
        global char  * act_q, \
        global half  * act_d, \
        global char  * dst, \
        ulong         offsetd, \
        int           k, \
        int           m, \
        int           n, \
        ulong         nb0, \
        ulong         nb1, \
        local float * tmp_real, \
        local float * tmp_imag, \
        local char  * act_real_tile, \
        local char  * act_imag_tile \
) { \
    dst = dst + offsetd; \
    const int row_base = get_group_id(0) * (TILE_M); \
    const int col = get_group_id(1); \
    const int lid = get_local_id(0); \
    const int lsize = get_local_size(0); \
    const int block_slot = lid >> 4; \
    const int lane = lid & 15; \
    const int nb64 = k / QK_IFAIRY64; \
    const int nbq = k / QK_IFAIRY64_Q16; \
    if (col >= n) { \
        return; \
    } \
    float acc_real[TILE_M]; \
    float acc_imag[TILE_M]; \
    for (int i = 0; i < (TILE_M); ++i) { \
        acc_real[i] = 0.0f; \
        acc_imag[i] = 0.0f; \
    } \
    for (int wb_base = 0; wb_base < nb64; wb_base += 4) { \
        const int wb = wb_base + block_slot; \
        const int act_block = wb; \
        if (wb < nb64) { \
            const int act_index = col * nbq + act_block; \
            const int q_base = act_index * (2 * QK_IFAIRY64_Q16); \
            for (int part = 0; part < 4; ++part) { \
                const int j = lane + 16 * part; \
                act_real_tile[block_slot * QK_IFAIRY64 + j] = act_q[q_base + j]; \
                act_imag_tile[block_slot * QK_IFAIRY64 + j] = act_q[q_base + QK_IFAIRY64_Q16 + j]; \
            } \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        if (wb < nb64) { \
            const int act_index = col * nbq + act_block; \
            const float x_real = vload_half(act_index * 2 + 0, act_d); \
            const float x_imag = vload_half(act_index * 2 + 1, act_d); \
            for (int tr = 0; tr < (TILE_M); ++tr) { \
                const int row = row_base + tr; \
                if (row >= m) { \
                    continue; \
                } \
                const int w_block = row * nb64 + wb; \
                const uint packed = (uint) w_q[w_block * 16 + lane]; \
                const float w_real = vload_half(w_block * 2 + 0, w_d); \
                const float w_imag = vload_half(w_block * 2 + 1, w_d); \
                int sum_ac = 0; \
                int sum_ad = 0; \
                int sum_bc = 0; \
                int sum_bd = 0; \
                for (int part = 0; part < 4; ++part) { \
                    const int j = lane + 16 * part; \
                    const uint code = (packed >> (2 * part)) & 3U; \
                    int wr = 0; \
                    int wi = 0; \
                    if (code == 0U) { \
                        wr = -1; \
                    } else if (code == 1U) { \
                        wr = 1; \
                    } else if (code == 2U) { \
                        wi = -1; \
                    } else { \
                        wi = 1; \
                    } \
                    const int xr = (int) act_real_tile[block_slot * QK_IFAIRY64 + j]; \
                    const int xi = (int) act_imag_tile[block_slot * QK_IFAIRY64 + j]; \
                    sum_ac += xr * wr; \
                    sum_ad += xi * wr; \
                    sum_bc += xr * wi; \
                    sum_bd += xi * wi; \
                } \
                acc_real[tr] += w_real * x_real * (float) sum_ac + \
                                w_imag * x_imag * (float) sum_bd; \
                acc_imag[tr] += w_imag * x_real * (float) sum_bc - \
                                w_real * x_imag * (float) sum_ad; \
            } \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    for (int tr = 0; tr < (TILE_M); ++tr) { \
        tmp_real[tr * lsize + lid] = acc_real[tr]; \
        tmp_imag[tr * lsize + lid] = acc_imag[tr]; \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) { \
        if (lid < stride) { \
            for (int tr = 0; tr < (TILE_M); ++tr) { \
                tmp_real[tr * lsize + lid] += tmp_real[tr * lsize + lid + stride]; \
                tmp_imag[tr * lsize + lid] += tmp_imag[tr * lsize + lid + stride]; \
            } \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    if (lid == 0) { \
        for (int tr = 0; tr < (TILE_M); ++tr) { \
            const int row = row_base + tr; \
            if (row < m) { \
                *((global uint *) (dst + (ulong) col * nb1 + (ulong) row * nb0)) = \
                    ifairy64_pack_bf16_pair(tmp_real[tr * lsize], tmp_imag[tr * lsize]); \
            } \
        } \
    } \
}

DEFINE_IFAIRY64_MUL_VEC_Q16_KERNEL(kernel_ifairy64_mul_vec8_f32_q16, IFAIRY64_GEMV8_TILE_M)
DEFINE_IFAIRY64_MUL_VEC_Q16_KERNEL(kernel_ifairy64_mul_vec16_f32_q16, IFAIRY64_GEMV16_TILE_M)

#undef DEFINE_IFAIRY64_MUL_VEC_Q16_KERNEL

/**
 * Experimental iFairy64 GEMV over CPU-LUT-style 16-row packed weight tiles.
 * Each 256-thread work-group covers 16, 32, or 64 output rows by reusing the
 * 16-row tile layout and reducing over the packed 2-weight group-pair lane.
 */
kernel void kernel_ifairy64_mul_vec_lutpack_f32_q16(
        global uchar * w_lut,
        global char  * act_q,
        global half  * act_d,
        global char  * dst,
        ulong         offsetd,
        int           k,
        int           m,
        int           n,
        ulong         nb0,
        ulong         nb1,
        int           tile_reps,
        local float * tmp_real,
        local float * tmp_imag
) {
    dst = dst + offsetd;

    const int row_base = get_group_id(0) * IFAIRY64_LUT_TILE_ROWS * tile_reps;
    const int col = get_group_id(1);
    const int lid = get_local_id(0);
    const int pair_lane = lid & (IFAIRY64_LUT_GROUP_PAIRS - 1);
    const int row_lane = (lid >> 4) & (IFAIRY64_LUT_TILE_ROWS - 1);
    const int nb64 = k / QK_IFAIRY64;
    const int nbq = k / QK_IFAIRY64_Q16;

    if (col >= n) {
        return;
    }

    float acc_real[IFAIRY64_LUT_MAX_REPS];
    float acc_imag[IFAIRY64_LUT_MAX_REPS];
#pragma unroll
    for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
        acc_real[rep] = 0.0f;
        acc_imag[rep] = 0.0f;
    }

    for (int wb = 0; wb < nb64; ++wb) {
        const int act_index = col * nbq + wb;
        const int q_base = act_index * (2 * QK_IFAIRY64_Q16);
        const int base0 = pair_lane * 4;

        const int xr0 = (int) act_q[q_base + base0 + 0];
        const int xi0 = (int) act_q[q_base + QK_IFAIRY64_Q16 + base0 + 0];
        const int xr1 = (int) act_q[q_base + base0 + 1];
        const int xi1 = (int) act_q[q_base + QK_IFAIRY64_Q16 + base0 + 1];
        const int xr2 = (int) act_q[q_base + base0 + 2];
        const int xi2 = (int) act_q[q_base + QK_IFAIRY64_Q16 + base0 + 2];
        const int xr3 = (int) act_q[q_base + base0 + 3];
        const int xi3 = (int) act_q[q_base + QK_IFAIRY64_Q16 + base0 + 3];

        const float x_real = vload_half(act_index * 2 + 0, act_d);
        const float x_imag = vload_half(act_index * 2 + 1, act_d);

#pragma unroll
        for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
            if (rep >= tile_reps) {
                continue;
            }

            const int row = row_base + rep * IFAIRY64_LUT_TILE_ROWS + row_lane;
            if (row >= m) {
                continue;
            }

            const int tile = row >> 4;
            global uchar * wt = w_lut + ((tile * nb64 + wb) * IFAIRY64_LUT_WTILE_BYTES);
            const uint packed = (uint) wt[pair_lane * IFAIRY64_LUT_TILE_ROWS + row_lane];
            global half * wt_d_real = (global half *) (wt + IFAIRY64_LUT_QS_BYTES);
            global half * wt_d_imag = (global half *) (wt + IFAIRY64_LUT_QS_BYTES + 2 * IFAIRY64_LUT_TILE_ROWS);
            const float w_real = vload_half(row_lane, wt_d_real);
            const float w_imag = vload_half(row_lane, wt_d_imag);

            int sum_ac = 0;
            int sum_ad = 0;
            int sum_bc = 0;
            int sum_bd = 0;
            ifairy64_accumulate_pair_code(packed & 0x0fU, xr0, xi0, xr1, xi1, &sum_ac, &sum_ad, &sum_bc, &sum_bd);
            ifairy64_accumulate_pair_code((packed >> 4) & 0x0fU, xr2, xi2, xr3, xi3, &sum_ac, &sum_ad, &sum_bc, &sum_bd);

            acc_real[rep] += w_real * x_real * (float) sum_ac +
                             w_imag * x_imag * (float) sum_bd;
            acc_imag[rep] += w_imag * x_real * (float) sum_bc -
                             w_real * x_imag * (float) sum_ad;
        }
    }

#pragma unroll
    for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
        if (rep < tile_reps) {
            const int tmp_idx = (rep * IFAIRY64_LUT_TILE_ROWS + row_lane) * IFAIRY64_LUT_GROUP_PAIRS + pair_lane;
            tmp_real[tmp_idx] = acc_real[rep];
            tmp_imag[tmp_idx] = acc_imag[rep];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (pair_lane == 0) {
#pragma unroll
        for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
            if (rep >= tile_reps) {
                continue;
            }

            const int row = row_base + rep * IFAIRY64_LUT_TILE_ROWS + row_lane;
            if (row >= m) {
                continue;
            }

            const int tmp_base = (rep * IFAIRY64_LUT_TILE_ROWS + row_lane) * IFAIRY64_LUT_GROUP_PAIRS;
            float out_real = 0.0f;
            float out_imag = 0.0f;
#pragma unroll
            for (int i = 0; i < IFAIRY64_LUT_GROUP_PAIRS; ++i) {
                out_real += tmp_real[tmp_base + i];
                out_imag += tmp_imag[tmp_base + i];
            }

            *((global uint *) (dst + (ulong) col * nb1 + (ulong) row * nb0)) =
                ifairy64_pack_bf16_pair(out_real, out_imag);
        }
    }
}

/**
 * True LUT iFairy64 GEMV. The activation side is consumed as prebuilt
 * 16-entry LUT groups; packed weight nibbles index those groups directly.
 */
kernel void kernel_ifairy64_mul_vec_lut16_f32(
        global uchar * w_lut,
        global char  * lut,
        global float * lut_scales,
        global char  * dst,
        ulong          offsetd,
        int            k,
        int            m,
        int            n,
        ulong          nb0,
        ulong          nb1,
        int            tile_reps,
        local float  * tmp_real,
        local float  * tmp_imag
) {
    dst = dst + offsetd;

    const int row_base = get_group_id(0) * IFAIRY64_LUT_TILE_ROWS * tile_reps;
    const int col = get_group_id(1);
    const int lid = get_local_id(0);
    const int pair_lane = lid & (IFAIRY64_LUT_GROUP_PAIRS - 1);
    const int row_lane = (lid >> 4) & (IFAIRY64_LUT_TILE_ROWS - 1);
    const int nb64 = k / QK_IFAIRY64;

    if (col >= n) {
        return;
    }

    float acc_real[IFAIRY64_LUT_MAX_REPS];
    float acc_imag[IFAIRY64_LUT_MAX_REPS];
#pragma unroll
    for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
        acc_real[rep] = 0.0f;
        acc_imag[rep] = 0.0f;
    }

    for (int wb = 0; wb < nb64; ++wb) {
        const int scale_index = (col * nb64 + wb) * 2;
        const float x_real = lut_scales[scale_index + 0];
        const float x_imag = lut_scales[scale_index + 1];
        global char * lut_blk = lut + ((col * nb64 + wb) * IFAIRY64_LUT_GROUP_PAIRS * 2) * IFAIRY64_LUT_GROUP_BYTES;

#pragma unroll
        for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
            if (rep >= tile_reps) {
                continue;
            }

            const int row = row_base + rep * IFAIRY64_LUT_TILE_ROWS + row_lane;
            if (row >= m) {
                continue;
            }

            const int tile = row >> 4;
            global uchar * wt = w_lut + ((tile * nb64 + wb) * IFAIRY64_LUT_WTILE_BYTES);
            const uint packed = (uint) wt[pair_lane * IFAIRY64_LUT_TILE_ROWS + row_lane];
            global half * wt_d_real = (global half *) (wt + IFAIRY64_LUT_QS_BYTES);
            global half * wt_d_imag = (global half *) (wt + IFAIRY64_LUT_QS_BYTES + 2 * IFAIRY64_LUT_TILE_ROWS);
            const float w_real = vload_half(row_lane, wt_d_real);
            const float w_imag = vload_half(row_lane, wt_d_imag);

            const uint lo = packed & 0x0fU;
            const uint hi = (packed >> 4) & 0x0fU;
            global char * tbl0 = lut_blk + (pair_lane * 2 + 0) * IFAIRY64_LUT_GROUP_BYTES;
            global char * tbl1 = lut_blk + (pair_lane * 2 + 1) * IFAIRY64_LUT_GROUP_BYTES;

            const int sum_ac = (int) tbl0[0  + lo] + (int) tbl1[0  + hi];
            const int sum_bd = (int) tbl0[16 + lo] + (int) tbl1[16 + hi];
            const int sum_bc = (int) tbl0[32 + lo] + (int) tbl1[32 + hi];
            const int sum_ad = (int) tbl0[48 + lo] + (int) tbl1[48 + hi];

            acc_real[rep] += w_real * x_real * (float) sum_ac +
                             w_imag * x_imag * (float) sum_bd;
            acc_imag[rep] += w_imag * x_real * (float) sum_bc +
                             w_real * x_imag * (float) sum_ad;
        }
    }

#pragma unroll
    for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
        if (rep < tile_reps) {
            const int tmp_idx = (rep * IFAIRY64_LUT_TILE_ROWS + row_lane) * IFAIRY64_LUT_GROUP_PAIRS + pair_lane;
            tmp_real[tmp_idx] = acc_real[rep];
            tmp_imag[tmp_idx] = acc_imag[rep];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (pair_lane == 0) {
#pragma unroll
        for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
            if (rep >= tile_reps) {
                continue;
            }

            const int row = row_base + rep * IFAIRY64_LUT_TILE_ROWS + row_lane;
            if (row >= m) {
                continue;
            }

            const int tmp_base = (rep * IFAIRY64_LUT_TILE_ROWS + row_lane) * IFAIRY64_LUT_GROUP_PAIRS;
            float out_real = 0.0f;
            float out_imag = 0.0f;
#pragma unroll
            for (int i = 0; i < IFAIRY64_LUT_GROUP_PAIRS; ++i) {
                out_real += tmp_real[tmp_base + i];
                out_imag += tmp_imag[tmp_base + i];
            }

            *((global uint *) (dst + (ulong) col * nb1 + (ulong) row * nb0)) =
                ifairy64_pack_bf16_pair(out_real, out_imag);
        }
    }
}

/**
 * Decode-oriented true LUT GEMV with per-work-group local LUT construction.
 * This avoids writing the activation LUT to global memory, trading repeated
 * per-row-tile LUT construction for much lower LUT read/write bandwidth.
 */
kernel void kernel_ifairy64_mul_vec_lutlocal_f32(
        global uchar * w_lut,
        global char  * src,
        ulong          offset1,
        global char  * dst,
        ulong          offsetd,
        int            k,
        int            m,
        int            n,
        ulong          nb10,
        ulong          nb11,
        ulong          nb0,
        ulong          nb1,
        int            tile_reps,
        local float  * tmp_real,
        local float  * tmp_imag,
        local char   * lut_tile
) {
    src = src + offset1;
    dst = dst + offsetd;

    const int row_base = get_group_id(0) * IFAIRY64_LUT_TILE_ROWS * tile_reps;
    const int col = get_group_id(1);
    const int lid = get_local_id(0);
    const int pair_lane = lid & (IFAIRY64_LUT_GROUP_PAIRS - 1);
    const int row_lane = (lid >> 4) & (IFAIRY64_LUT_TILE_ROWS - 1);
    const int nb64 = k / QK_IFAIRY64;

    if (col >= n) {
        return;
    }

    float acc_real[IFAIRY64_LUT_MAX_REPS];
    float acc_imag[IFAIRY64_LUT_MAX_REPS];
#pragma unroll
    for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
        acc_real[rep] = 0.0f;
        acc_imag[rep] = 0.0f;
    }

    for (int wb = 0; wb < nb64; ++wb) {
        float max_real = 0.0f;
        float max_imag = 0.0f;
        if (lid < QK_IFAIRY64) {
            const int k_idx = wb * QK_IFAIRY64 + lid;
            const uint pair = *((global uint *) (src + (ulong) col * nb11 + (ulong) k_idx * nb10));
            max_real = fmax(1.0e-5f, fabs(ifairy_bf16_to_f32((ushort) (pair & 0xffffU))));
            max_imag = fmax(1.0e-5f, fabs(ifairy_bf16_to_f32((ushort) (pair >> 16))));
        }
        tmp_real[lid] = max_real;
        tmp_imag[lid] = max_imag;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int stride = QK_IFAIRY64 >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                tmp_real[lid] = fmax(tmp_real[lid], tmp_real[lid + stride]);
                tmp_imag[lid] = fmax(tmp_imag[lid], tmp_imag[lid + stride]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        const float x_real = tmp_real[0] / 63.0f;
        const float x_imag = tmp_imag[0] / 63.0f;
        const float iscale_real = 1.0f / x_real;
        const float iscale_imag = 1.0f / x_imag;

        if (lid < IFAIRY64_LUT_GROUP_PAIRS * 2) {
            const int group = lid;
            const int j0 = group * 2 + 0;
            const int j1 = group * 2 + 1;
            const int k0 = wb * QK_IFAIRY64 + j0;
            const int k1 = wb * QK_IFAIRY64 + j1;
            const uint pair0 = *((global uint *) (src + (ulong) col * nb11 + (ulong) k0 * nb10));
            const uint pair1 = *((global uint *) (src + (ulong) col * nb11 + (ulong) k1 * nb10));

            const int xr0 = ifairy_clamp_i32((int) rint(ifairy_bf16_to_f32((ushort) (pair0 & 0xffffU)) * iscale_real), -63, 63);
            const int xi0 = ifairy_clamp_i32((int) rint(ifairy_bf16_to_f32((ushort) (pair0 >> 16)) * iscale_imag), -63, 63);
            const int xr1 = ifairy_clamp_i32((int) rint(ifairy_bf16_to_f32((ushort) (pair1 & 0xffffU)) * iscale_real), -63, 63);
            const int xi1 = ifairy_clamp_i32((int) rint(ifairy_bf16_to_f32((ushort) (pair1 >> 16)) * iscale_imag), -63, 63);

            ifairy64_lut_fill_group_local(xr0, xi0, xr1, xi1, lut_tile + group * IFAIRY64_LUT_GROUP_BYTES);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
        for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
            if (rep >= tile_reps) {
                continue;
            }

            const int row = row_base + rep * IFAIRY64_LUT_TILE_ROWS + row_lane;
            if (row >= m) {
                continue;
            }

            const int tile = row >> 4;
            global uchar * wt = w_lut + ((tile * nb64 + wb) * IFAIRY64_LUT_WTILE_BYTES);
            const uint packed = (uint) wt[pair_lane * IFAIRY64_LUT_TILE_ROWS + row_lane];
            global half * wt_d_real = (global half *) (wt + IFAIRY64_LUT_QS_BYTES);
            global half * wt_d_imag = (global half *) (wt + IFAIRY64_LUT_QS_BYTES + 2 * IFAIRY64_LUT_TILE_ROWS);
            const float w_real = vload_half(row_lane, wt_d_real);
            const float w_imag = vload_half(row_lane, wt_d_imag);

            const uint lo = packed & 0x0fU;
            const uint hi = (packed >> 4) & 0x0fU;
            local char * tbl0 = lut_tile + (pair_lane * 2 + 0) * IFAIRY64_LUT_GROUP_BYTES;
            local char * tbl1 = lut_tile + (pair_lane * 2 + 1) * IFAIRY64_LUT_GROUP_BYTES;

            const int sum_ac = (int) tbl0[0  + lo] + (int) tbl1[0  + hi];
            const int sum_bd = (int) tbl0[16 + lo] + (int) tbl1[16 + hi];
            const int sum_bc = (int) tbl0[32 + lo] + (int) tbl1[32 + hi];
            const int sum_ad = (int) tbl0[48 + lo] + (int) tbl1[48 + hi];

            acc_real[rep] += w_real * x_real * (float) sum_ac +
                             w_imag * x_imag * (float) sum_bd;
            acc_imag[rep] += w_imag * x_real * (float) sum_bc +
                             w_real * x_imag * (float) sum_ad;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

#pragma unroll
    for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
        if (rep < tile_reps) {
            const int tmp_idx = (rep * IFAIRY64_LUT_TILE_ROWS + row_lane) * IFAIRY64_LUT_GROUP_PAIRS + pair_lane;
            tmp_real[tmp_idx] = acc_real[rep];
            tmp_imag[tmp_idx] = acc_imag[rep];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (pair_lane == 0) {
#pragma unroll
        for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
            if (rep >= tile_reps) {
                continue;
            }

            const int row = row_base + rep * IFAIRY64_LUT_TILE_ROWS + row_lane;
            if (row >= m) {
                continue;
            }

            const int tmp_base = (rep * IFAIRY64_LUT_TILE_ROWS + row_lane) * IFAIRY64_LUT_GROUP_PAIRS;
            float out_real = 0.0f;
            float out_imag = 0.0f;
#pragma unroll
            for (int i = 0; i < IFAIRY64_LUT_GROUP_PAIRS; ++i) {
                out_real += tmp_real[tmp_base + i];
                out_imag += tmp_imag[tmp_base + i];
            }

            *((global uint *) (dst + (ulong) col * nb1 + (ulong) row * nb0)) =
                ifairy64_pack_bf16_pair(out_real, out_imag);
        }
    }
}

/**
 * Fused iFairy64 wide-linear W2. The activation is quantized once into K64 q16
 * staging buffers, then one kernel consumes U.s0/U.s1/W.s0/W.s1 and writes the
 * final packed-BF16 complex output. U branches use w*x, W branches use
 * w*conj(x), matching the CPU fused wide-linear implementation.
 */
kernel void kernel_ifairy64_wide_linear_w2_f32_q16(
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

    const int row_base = get_group_id(0) * IFAIRY64_WIDE_TILE_M;
    const int col = get_group_id(1);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int block_slot = lid >> 4;
    const int lane = lid & 15;
    const int nb64 = k / QK_IFAIRY64;
    const int nbq = k / QK_IFAIRY64_Q16;

    if (col >= n) {
        return;
    }

    float acc_real[IFAIRY64_WIDE_TILE_M];
    float acc_imag[IFAIRY64_WIDE_TILE_M];
#pragma unroll
    for (int i = 0; i < IFAIRY64_WIDE_TILE_M; ++i) {
        acc_real[i] = 0.0f;
        acc_imag[i] = 0.0f;
    }

    for (int wb_base = 0; wb_base < nb64; wb_base += 4) {
        const int wb = wb_base + block_slot;
        const int act_block = wb;

        if (wb < nb64) {
            const int act_index = col * nbq + act_block;
            const int q_base = act_index * (2 * QK_IFAIRY64_Q16);
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int j = lane + 16 * part;
                act_real_tile[block_slot * QK_IFAIRY64 + j] = act_q[q_base + j];
                act_imag_tile[block_slot * QK_IFAIRY64 + j] = act_q[q_base + QK_IFAIRY64_Q16 + j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (wb < nb64) {
            const int act_index = col * nbq + act_block;
            const float x_real = vload_half(act_index * 2 + 0, act_d);
            const float x_imag = vload_half(act_index * 2 + 1, act_d);

#pragma unroll
            for (int tr = 0; tr < IFAIRY64_WIDE_TILE_M; ++tr) {
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
                    const int xr = (int) act_real_tile[block_slot * QK_IFAIRY64 + j];
                    const int xi = (int) act_imag_tile[block_slot * QK_IFAIRY64 + j];

                    const int2 cu0 = ifairy64_decode_code_branchless((packed_u0 >> (2 * part)) & 3U);
                    const int2 cu1 = ifairy64_decode_code_branchless((packed_u1 >> (2 * part)) & 3U);
                    const int2 cw0 = ifairy64_decode_code_branchless((packed_w0 >> (2 * part)) & 3U);
                    const int2 cw1 = ifairy64_decode_code_branchless((packed_w1 >> (2 * part)) & 3U);

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
    for (int tr = 0; tr < IFAIRY64_WIDE_TILE_M; ++tr) {
        tmp_real[tr * lsize + lid] = acc_real[tr];
        tmp_imag[tr * lsize + lid] = acc_imag[tr];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
#pragma unroll
            for (int tr = 0; tr < IFAIRY64_WIDE_TILE_M; ++tr) {
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
        for (int tr = 0; tr < IFAIRY64_WIDE_TILE_M; ++tr) {
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
                    ifairy64_pack_bf16_pair(out_real, out_imag);
            }
        }
    }
}

/**
 * Fused iFairy64 wide-linear W2 using a true activation LUT built in local
 * memory per K64 block. The LUT is shared by U0/U1/W0/W1; W branches consume
 * the conjugated-activation LUT directly, while U branches flip the imaginary
 * activation scale to recover non-conjugated w*x semantics.
 */
kernel void kernel_ifairy64_wide_linear_w2_lutlocal_f32(
        global uchar * u0_lut,
        global uchar * u1_lut,
        global uchar * w0_lut,
        global uchar * w1_lut,
        global char  * src,
        ulong          offsetx,
        global char  * bias,
        ulong          offsetb,
        int            has_bias,
        global char  * dst,
        ulong          offsetd,
        int            k,
        int            m,
        int            n,
        int            x_ne1,
        int            x_ne2,
        int            bias_ne0,
        int            bias_ne1,
        int            bias_ne2,
        int            bias_ne3,
        ulong          bias_nb0,
        ulong          bias_nb1,
        ulong          bias_nb2,
        ulong          bias_nb3,
        ulong          nb10,
        ulong          nb11,
        ulong          nb0,
        ulong          nb1,
        int            tile_reps,
        local float  * tmp_real,
        local float  * tmp_imag,
        local char   * lut_tile
) {
    src = src + offsetx;
    dst = dst + offsetd;
    bias = bias + offsetb;

    const int row_base = get_group_id(0) * IFAIRY64_LUT_TILE_ROWS * tile_reps;
    const int col = get_group_id(1);
    const int lid = get_local_id(0);
    const int pair_lane = lid & (IFAIRY64_LUT_GROUP_PAIRS - 1);
    const int row_lane = (lid >> 4) & (IFAIRY64_LUT_TILE_ROWS - 1);
    const int nb64 = k / QK_IFAIRY64;

    if (col >= n) {
        return;
    }

    float acc_real[IFAIRY64_LUT_MAX_REPS];
    float acc_imag[IFAIRY64_LUT_MAX_REPS];
#pragma unroll
    for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
        acc_real[rep] = 0.0f;
        acc_imag[rep] = 0.0f;
    }

    for (int wb = 0; wb < nb64; ++wb) {
        float max_real = 0.0f;
        float max_imag = 0.0f;
        if (lid < QK_IFAIRY64) {
            const int k_idx = wb * QK_IFAIRY64 + lid;
            const uint pair = *((global uint *) (src + (ulong) col * nb11 + (ulong) k_idx * nb10));
            max_real = fmax(1.0e-5f, fabs(ifairy_bf16_to_f32((ushort) (pair & 0xffffU))));
            max_imag = fmax(1.0e-5f, fabs(ifairy_bf16_to_f32((ushort) (pair >> 16))));
        }
        tmp_real[lid] = max_real;
        tmp_imag[lid] = max_imag;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int stride = QK_IFAIRY64 >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                tmp_real[lid] = fmax(tmp_real[lid], tmp_real[lid + stride]);
                tmp_imag[lid] = fmax(tmp_imag[lid], tmp_imag[lid + stride]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        const float x_real = tmp_real[0] / 63.0f;
        const float x_imag = tmp_imag[0] / 63.0f;
        const float iscale_real = 1.0f / x_real;
        const float iscale_imag = 1.0f / x_imag;

        if (lid < IFAIRY64_LUT_GROUP_PAIRS * 2) {
            const int group = lid;
            const int j0 = group * 2 + 0;
            const int j1 = group * 2 + 1;
            const int k0 = wb * QK_IFAIRY64 + j0;
            const int k1 = wb * QK_IFAIRY64 + j1;
            const uint pair0 = *((global uint *) (src + (ulong) col * nb11 + (ulong) k0 * nb10));
            const uint pair1 = *((global uint *) (src + (ulong) col * nb11 + (ulong) k1 * nb10));

            const int xr0 = ifairy_clamp_i32((int) rint(ifairy_bf16_to_f32((ushort) (pair0 & 0xffffU)) * iscale_real), -63, 63);
            const int xi0 = ifairy_clamp_i32((int) rint(ifairy_bf16_to_f32((ushort) (pair0 >> 16)) * iscale_imag), -63, 63);
            const int xr1 = ifairy_clamp_i32((int) rint(ifairy_bf16_to_f32((ushort) (pair1 & 0xffffU)) * iscale_real), -63, 63);
            const int xi1 = ifairy_clamp_i32((int) rint(ifairy_bf16_to_f32((ushort) (pair1 >> 16)) * iscale_imag), -63, 63);

            ifairy64_lut_fill_group_local(xr0, xi0, xr1, xi1, lut_tile + group * IFAIRY64_LUT_GROUP_BYTES);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
            if (rep >= tile_reps) {
                continue;
            }

            const int row = row_base + rep * IFAIRY64_LUT_TILE_ROWS + row_lane;
            if (row >= m) {
                continue;
            }

            const int tile = row >> 4;
            global uchar * wt_u0 = u0_lut + ((tile * nb64 + wb) * IFAIRY64_LUT_WTILE_BYTES);
            global uchar * wt_u1 = u1_lut + ((tile * nb64 + wb) * IFAIRY64_LUT_WTILE_BYTES);
            global uchar * wt_w0 = w0_lut + ((tile * nb64 + wb) * IFAIRY64_LUT_WTILE_BYTES);
            global uchar * wt_w1 = w1_lut + ((tile * nb64 + wb) * IFAIRY64_LUT_WTILE_BYTES);

            const uint packed_u0 = (uint) wt_u0[pair_lane * IFAIRY64_LUT_TILE_ROWS + row_lane];
            const uint packed_u1 = (uint) wt_u1[pair_lane * IFAIRY64_LUT_TILE_ROWS + row_lane];
            const uint packed_w0 = (uint) wt_w0[pair_lane * IFAIRY64_LUT_TILE_ROWS + row_lane];
            const uint packed_w1 = (uint) wt_w1[pair_lane * IFAIRY64_LUT_TILE_ROWS + row_lane];

            global half * u0_d_real = (global half *) (wt_u0 + IFAIRY64_LUT_QS_BYTES);
            global half * u0_d_imag = (global half *) (wt_u0 + IFAIRY64_LUT_QS_BYTES + 2 * IFAIRY64_LUT_TILE_ROWS);
            global half * u1_d_real = (global half *) (wt_u1 + IFAIRY64_LUT_QS_BYTES);
            global half * u1_d_imag = (global half *) (wt_u1 + IFAIRY64_LUT_QS_BYTES + 2 * IFAIRY64_LUT_TILE_ROWS);
            global half * w0_d_real = (global half *) (wt_w0 + IFAIRY64_LUT_QS_BYTES);
            global half * w0_d_imag = (global half *) (wt_w0 + IFAIRY64_LUT_QS_BYTES + 2 * IFAIRY64_LUT_TILE_ROWS);
            global half * w1_d_real = (global half *) (wt_w1 + IFAIRY64_LUT_QS_BYTES);
            global half * w1_d_imag = (global half *) (wt_w1 + IFAIRY64_LUT_QS_BYTES + 2 * IFAIRY64_LUT_TILE_ROWS);

            const float u0_real = vload_half(row_lane, u0_d_real);
            const float u0_imag = vload_half(row_lane, u0_d_imag);
            const float u1_real = vload_half(row_lane, u1_d_real);
            const float u1_imag = vload_half(row_lane, u1_d_imag);
            const float w0_real = vload_half(row_lane, w0_d_real);
            const float w0_imag = vload_half(row_lane, w0_d_imag);
            const float w1_real = vload_half(row_lane, w1_d_real);
            const float w1_imag = vload_half(row_lane, w1_d_imag);

            local char * tbl0 = lut_tile + (pair_lane * 2 + 0) * IFAIRY64_LUT_GROUP_BYTES;
            local char * tbl1 = lut_tile + (pair_lane * 2 + 1) * IFAIRY64_LUT_GROUP_BYTES;

#define IFAIRY64_LUT_SUMS(PACKED, AC, BD, BC, AD) \
            const uint AC##_lo = (PACKED) & 0x0fU; \
            const uint AC##_hi = ((PACKED) >> 4) & 0x0fU; \
            const int AC = (int) tbl0[0  + AC##_lo] + (int) tbl1[0  + AC##_hi]; \
            const int BD = (int) tbl0[16 + AC##_lo] + (int) tbl1[16 + AC##_hi]; \
            const int BC = (int) tbl0[32 + AC##_lo] + (int) tbl1[32 + AC##_hi]; \
            const int AD = (int) tbl0[48 + AC##_lo] + (int) tbl1[48 + AC##_hi]

            IFAIRY64_LUT_SUMS(packed_u0, u0_ac, u0_bd, u0_bc, u0_ad);
            IFAIRY64_LUT_SUMS(packed_u1, u1_ac, u1_bd, u1_bc, u1_ad);
            IFAIRY64_LUT_SUMS(packed_w0, w0_ac, w0_bd, w0_bc, w0_ad);
            IFAIRY64_LUT_SUMS(packed_w1, w1_ac, w1_bd, w1_bc, w1_ad);

#undef IFAIRY64_LUT_SUMS

            acc_real[rep] +=
                u0_real * x_real * (float) u0_ac - u0_imag * x_imag * (float) u0_bd +
                u1_real * x_real * (float) u1_ac - u1_imag * x_imag * (float) u1_bd +
                w0_real * x_real * (float) w0_ac + w0_imag * x_imag * (float) w0_bd +
                w1_real * x_real * (float) w1_ac + w1_imag * x_imag * (float) w1_bd;

            acc_imag[rep] +=
                u0_imag * x_real * (float) u0_bc - u0_real * x_imag * (float) u0_ad +
                u1_imag * x_real * (float) u1_bc - u1_real * x_imag * (float) u1_ad +
                w0_imag * x_real * (float) w0_bc + w0_real * x_imag * (float) w0_ad +
                w1_imag * x_real * (float) w1_bc + w1_real * x_imag * (float) w1_ad;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
        if (rep < tile_reps) {
            const int tmp_idx = (rep * IFAIRY64_LUT_TILE_ROWS + row_lane) * IFAIRY64_LUT_GROUP_PAIRS + pair_lane;
            tmp_real[tmp_idx] = acc_real[rep];
            tmp_imag[tmp_idx] = acc_imag[rep];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (pair_lane == 0) {
        #pragma unroll
        for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
            if (rep >= tile_reps) {
                continue;
            }

            const int row = row_base + rep * IFAIRY64_LUT_TILE_ROWS + row_lane;
            if (row >= m) {
                continue;
            }

            const int tmp_base = (rep * IFAIRY64_LUT_TILE_ROWS + row_lane) * IFAIRY64_LUT_GROUP_PAIRS;
            float out_real = 0.0f;
            float out_imag = 0.0f;
#pragma unroll
            for (int i = 0; i < IFAIRY64_LUT_GROUP_PAIRS; ++i) {
                out_real += tmp_real[tmp_base + i];
                out_imag += tmp_imag[tmp_base + i];
            }

            if (has_bias) {
                const int i1 = col % x_ne1;
                const int i2 = (col / x_ne1) % x_ne2;
                const int i3 = col / (x_ne1 * x_ne2);
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
                ifairy64_pack_bf16_pair(out_real, out_imag);
        }
    }
}

/**
 * Fused iFairy64 wide-linear W2 using a globally prebuilt activation LUT.
 * The LUT is constructed once per activation K64 block and then reused by all
 * output-row workgroups, matching the CPU LUT reuse structure more closely.
 */
kernel void kernel_ifairy64_wide_linear_w2_lutglobal_f32(
        global uchar * u0_lut,
        global uchar * u1_lut,
        global uchar * w0_lut,
        global uchar * w1_lut,
        global char  * lut,
        global float * lut_scales,
        global char  * bias,
        ulong          offsetb,
        int            has_bias,
        global char  * dst,
        ulong          offsetd,
        int            k,
        int            m,
        int            n,
        int            x_ne1,
        int            x_ne2,
        int            bias_ne0,
        int            bias_ne1,
        int            bias_ne2,
        int            bias_ne3,
        ulong          bias_nb0,
        ulong          bias_nb1,
        ulong          bias_nb2,
        ulong          bias_nb3,
        ulong          nb0,
        ulong          nb1,
        int            tile_reps,
        local float  * tmp_real,
        local float  * tmp_imag,
        local char   * lut_tile
) {
    dst = dst + offsetd;
    bias = bias + offsetb;

    const int row_base = get_group_id(0) * IFAIRY64_LUT_TILE_ROWS * tile_reps;
    const int col = get_group_id(1);
    const int lid = get_local_id(0);
    const int pair_lane = lid & (IFAIRY64_LUT_GROUP_PAIRS - 1);
    const int row_lane = (lid >> 4) & (IFAIRY64_LUT_TILE_ROWS - 1);
    const int nb64 = k / QK_IFAIRY64;

    if (col >= n) {
        return;
    }

    float acc_real[IFAIRY64_LUT_MAX_REPS];
    float acc_imag[IFAIRY64_LUT_MAX_REPS];
#pragma unroll
    for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
        acc_real[rep] = 0.0f;
        acc_imag[rep] = 0.0f;
    }

    for (int wb = 0; wb < nb64; ++wb) {
        const int scale_index = (col * nb64 + wb) * 2;
        const float x_real = lut_scales[scale_index + 0];
        const float x_imag = lut_scales[scale_index + 1];
        global char * lut_blk = lut + ((col * nb64 + wb) * IFAIRY64_LUT_GROUP_PAIRS * 2) * IFAIRY64_LUT_GROUP_BYTES;

        for (int i = lid; i < IFAIRY64_LUT_GROUP_PAIRS * 2 * IFAIRY64_LUT_GROUP_BYTES; i += get_local_size(0)) {
            lut_tile[i] = lut_blk[i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
        for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
            if (rep >= tile_reps) {
                continue;
            }

            const int row = row_base + rep * IFAIRY64_LUT_TILE_ROWS + row_lane;
            if (row >= m) {
                continue;
            }

            const int tile = row >> 4;
            global uchar * wt_u0 = u0_lut + ((tile * nb64 + wb) * IFAIRY64_LUT_WTILE_BYTES);
            global uchar * wt_u1 = u1_lut + ((tile * nb64 + wb) * IFAIRY64_LUT_WTILE_BYTES);
            global uchar * wt_w0 = w0_lut + ((tile * nb64 + wb) * IFAIRY64_LUT_WTILE_BYTES);
            global uchar * wt_w1 = w1_lut + ((tile * nb64 + wb) * IFAIRY64_LUT_WTILE_BYTES);

            const uint packed_u0 = (uint) wt_u0[pair_lane * IFAIRY64_LUT_TILE_ROWS + row_lane];
            const uint packed_u1 = (uint) wt_u1[pair_lane * IFAIRY64_LUT_TILE_ROWS + row_lane];
            const uint packed_w0 = (uint) wt_w0[pair_lane * IFAIRY64_LUT_TILE_ROWS + row_lane];
            const uint packed_w1 = (uint) wt_w1[pair_lane * IFAIRY64_LUT_TILE_ROWS + row_lane];

            global half * u0_d_real = (global half *) (wt_u0 + IFAIRY64_LUT_QS_BYTES);
            global half * u0_d_imag = (global half *) (wt_u0 + IFAIRY64_LUT_QS_BYTES + 2 * IFAIRY64_LUT_TILE_ROWS);
            global half * u1_d_real = (global half *) (wt_u1 + IFAIRY64_LUT_QS_BYTES);
            global half * u1_d_imag = (global half *) (wt_u1 + IFAIRY64_LUT_QS_BYTES + 2 * IFAIRY64_LUT_TILE_ROWS);
            global half * w0_d_real = (global half *) (wt_w0 + IFAIRY64_LUT_QS_BYTES);
            global half * w0_d_imag = (global half *) (wt_w0 + IFAIRY64_LUT_QS_BYTES + 2 * IFAIRY64_LUT_TILE_ROWS);
            global half * w1_d_real = (global half *) (wt_w1 + IFAIRY64_LUT_QS_BYTES);
            global half * w1_d_imag = (global half *) (wt_w1 + IFAIRY64_LUT_QS_BYTES + 2 * IFAIRY64_LUT_TILE_ROWS);

            const float u0_real = vload_half(row_lane, u0_d_real);
            const float u0_imag = vload_half(row_lane, u0_d_imag);
            const float u1_real = vload_half(row_lane, u1_d_real);
            const float u1_imag = vload_half(row_lane, u1_d_imag);
            const float w0_real = vload_half(row_lane, w0_d_real);
            const float w0_imag = vload_half(row_lane, w0_d_imag);
            const float w1_real = vload_half(row_lane, w1_d_real);
            const float w1_imag = vload_half(row_lane, w1_d_imag);

            local char * tbl0 = lut_tile + (pair_lane * 2 + 0) * IFAIRY64_LUT_GROUP_BYTES;
            local char * tbl1 = lut_tile + (pair_lane * 2 + 1) * IFAIRY64_LUT_GROUP_BYTES;

#define IFAIRY64_LUT_SUMS(PACKED, AC, BD, BC, AD) \
            const uint AC##_lo = (PACKED) & 0x0fU; \
            const uint AC##_hi = ((PACKED) >> 4) & 0x0fU; \
            const uint AC##_v0 = *((local uint *) (tbl0 + 4 * AC##_lo)); \
            const uint AC##_v1 = *((local uint *) (tbl1 + 4 * AC##_hi)); \
            const int AC = ifairy64_unpack_s8(AC##_v0, 0) + ifairy64_unpack_s8(AC##_v1, 0); \
            const int BD = ifairy64_unpack_s8(AC##_v0, 1) + ifairy64_unpack_s8(AC##_v1, 1); \
            const int BC = ifairy64_unpack_s8(AC##_v0, 2) + ifairy64_unpack_s8(AC##_v1, 2); \
            const int AD = ifairy64_unpack_s8(AC##_v0, 3) + ifairy64_unpack_s8(AC##_v1, 3)

            IFAIRY64_LUT_SUMS(packed_u0, u0_ac, u0_bd, u0_bc, u0_ad);
            IFAIRY64_LUT_SUMS(packed_u1, u1_ac, u1_bd, u1_bc, u1_ad);
            IFAIRY64_LUT_SUMS(packed_w0, w0_ac, w0_bd, w0_bc, w0_ad);
            IFAIRY64_LUT_SUMS(packed_w1, w1_ac, w1_bd, w1_bc, w1_ad);

#undef IFAIRY64_LUT_SUMS

            acc_real[rep] +=
                u0_real * x_real * (float) u0_ac - u0_imag * x_imag * (float) u0_bd +
                u1_real * x_real * (float) u1_ac - u1_imag * x_imag * (float) u1_bd +
                w0_real * x_real * (float) w0_ac + w0_imag * x_imag * (float) w0_bd +
                w1_real * x_real * (float) w1_ac + w1_imag * x_imag * (float) w1_bd;

            acc_imag[rep] +=
                u0_imag * x_real * (float) u0_bc - u0_real * x_imag * (float) u0_ad +
                u1_imag * x_real * (float) u1_bc - u1_real * x_imag * (float) u1_ad +
                w0_imag * x_real * (float) w0_bc + w0_real * x_imag * (float) w0_ad +
                w1_imag * x_real * (float) w1_bc + w1_real * x_imag * (float) w1_ad;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

#pragma unroll
    for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
        if (rep < tile_reps) {
            const int tmp_idx = (rep * IFAIRY64_LUT_TILE_ROWS + row_lane) * IFAIRY64_LUT_GROUP_PAIRS + pair_lane;
            tmp_real[tmp_idx] = acc_real[rep];
            tmp_imag[tmp_idx] = acc_imag[rep];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (pair_lane == 0) {
#pragma unroll
        for (int rep = 0; rep < IFAIRY64_LUT_MAX_REPS; ++rep) {
            if (rep >= tile_reps) {
                continue;
            }

            const int row = row_base + rep * IFAIRY64_LUT_TILE_ROWS + row_lane;
            if (row >= m) {
                continue;
            }

            const int tmp_base = (rep * IFAIRY64_LUT_TILE_ROWS + row_lane) * IFAIRY64_LUT_GROUP_PAIRS;
            float out_real = 0.0f;
            float out_imag = 0.0f;
#pragma unroll
            for (int i = 0; i < IFAIRY64_LUT_GROUP_PAIRS; ++i) {
                out_real += tmp_real[tmp_base + i];
                out_imag += tmp_imag[tmp_base + i];
            }

            if (has_bias) {
                const int i1 = col % x_ne1;
                const int i2 = (col / x_ne1) % x_ne2;
                const int i3 = col / (x_ne1 * x_ne2);
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
                ifairy64_pack_bf16_pair(out_real, out_imag);
        }
    }
}
