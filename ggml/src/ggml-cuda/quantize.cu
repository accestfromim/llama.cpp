#include "quantize.cuh"
#include <cstdint>


// Device function: Two-level reduction to find maximum value across 256 threads
static __device__ float block_reduce_max(float local_val) {
    // Warp-level reduction using shuffle
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Reduce within warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_val = fmaxf(local_val, __shfl_xor_sync(0xFFFFFFFF, local_val, offset));
    }

    // Shared memory: 8 values for 8 warps
    __shared__ float shared_mem[8];

    // Each warp's first thread writes its warp's max to shared memory
    if (lane_id == 0) {
        shared_mem[warp_id] = local_val;
    }

    // Synchronize to ensure all warps finish writing
    __syncthreads();

    // Only the first warp performs the final reduction
    float block_max = 0.0f;
    if (warp_id == 0) {
        block_max = lane_id < 8 ? shared_mem[lane_id] : 0.0f;

        // Reduce within the first warp
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            block_max = fmaxf(block_max, __shfl_xor_sync(0xFFFFFFFF, block_max, offset));
        }
    }

    // First thread writes the final block max to shared memory
    if (warp_id == 0 && lane_id == 0) {
        shared_mem[0] = block_max;
    }

    __syncthreads();

    return shared_mem[0];
}

// Kernel to quantize complex fp32 values to block_ifairy_q16 format
// Each block processes 256 complex numbers (512 float values)
__launch_bounds__(256, 1)
static __global__ void quantize_ifairy_q16_kernel(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t & i00 = i0;
    const int64_t & i01 = i1;
    const int64_t & i02 = i2;
    const int64_t & i03 = i3;

    const int64_t i_cont = ((i3*ne2.z + i2) * ne1 + i1) * ne0 + i0;

    block_ifairy_q16 * y = (block_ifairy_q16 *) vy;

    const int64_t ib  = i_cont / QK_K; // block index
    const int64_t iqs = i_cont % QK_K; // quant index

    // Read packed bf16 values (each float contains real in high 16 bits, imag in low 16 bits)
    const int64_t idx = i03*s03 + i02*s02 + i01*s01 + i00;

    float real_val = 0.0f;
    float imag_val = 0.0f;

    if (i00 < ne00) {
        // Unpack bf16 values from fp32
        const uint32_t packed = __float_as_uint(x[idx]);
        const uint16_t real_bf16 = packed & 0xFFFF;
        const uint16_t imag_bf16 = (packed >> 16) & 0xFFFF;

        real_val = __bfloat162float(__ushort_as_bfloat16(real_bf16));
        imag_val = __bfloat162float(__ushort_as_bfloat16(imag_bf16));
    }

    // Compute absolute values for reduction
    const float real_abs = fabsf(real_val);
    const float imag_abs = fabsf(imag_val);

    // Block-level reduction to find max values
    const float max_real = block_reduce_max(real_abs);
    const float max_imag = block_reduce_max(imag_abs);

    // Thread 0 writes scale metadata to fp16
    __shared__ float shared_scale[2];
    if (threadIdx.x == 0) {
        const float d_real = max_real / 127.0f;
        const float d_imag = max_imag / 127.0f;

        y[ib].d_real = __float2half_rn(d_real);
        y[ib].d_imag = __float2half_rn(d_imag);

        shared_scale[0] = d_real;
        shared_scale[1] = d_imag;
    }
    __syncthreads();

    // All threads quantize and store values as uint8
    const float d_real = shared_scale[0];
    const float d_imag = shared_scale[1];

    const float id_real = d_real != 0.0f ? 1.0f / d_real : 0.0f;
    const float id_imag = d_imag != 0.0f ? 1.0f / d_imag : 0.0f;

#define clamp_to_int8(v) (v < -127.0f ? -127 : (v > 127.0f ? 127 : v))

    const int8_t q_real = d_real != 0.0f ? clamp_to_int8((int8_t)rintf(real_val * id_real)) : 0;
    const int8_t q_imag = d_imag != 0.0f ? clamp_to_int8((int8_t)rintf(imag_val * id_imag)) : 0;

    y[ib].x_real[iqs] = q_real;
    y[ib].x_imag[iqs] = q_imag;
}

__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void quantize_q8_1(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t & i00 = i0;
    const int64_t & i01 = i1;
    const int64_t & i02 = i2;
    const int64_t & i03 = i3;

    const int64_t i_cont = ((i3*ne2.z + i2) * ne1 + i1) * ne0 + i0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    const float xi = i0 < ne00 ? x[i03*s03 + i02*s02 + i01*s01 + i00] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;                    // block index in channel
    const int64_t iqs = i0 % (4*QK8_1);                                             // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    // Write back 4 int8 values as a single 32 bit value for better memroy bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        const float d = 1.0f / d_inv;

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}

void quantize_row_ifairy_q16_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(!ids);  // Indirect quantization not supported
    GGML_ASSERT(ne0 % QK_K == 0);  // Must be multiple of 256

    // Grid configuration
    const uint32_t block_num_x = (ne0 + QK_K - 1) / QK_K;  // (ne0 + 255) / 256
    const uint32_t block_num_y = ne1;
    const uint32_t block_num_z = ne2 * ne3;

    const dim3 num_blocks(block_num_x, block_num_y, block_num_z);
    const dim3 block_size(256, 1, 1);

    // Initialize fast division for ne2
    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);

    quantize_ifairy_q16_kernel<<<num_blocks, block_size, 0, stream>>>(
        x, vy, ne00, s01, s02, s03, ne0, block_num_y, ne2_fastdiv);

    GGML_UNUSED(type_src0);
}

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(!ids);
    GGML_ASSERT(ne0 % QK8_1 == 0);

    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);

    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
    GGML_UNUSED(type_src0);
}

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % (4*QK8_1) == 0);

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
