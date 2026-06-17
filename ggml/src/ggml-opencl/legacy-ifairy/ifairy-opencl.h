#pragma once

#ifndef GGML_OPENCL_TARGET_VERSION
#    define GGML_OPENCL_TARGET_VERSION 120
#endif
#ifndef CL_TARGET_OPENCL_VERSION
#    define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#endif

#include <CL/cl.h>

#include <cstddef>

struct ggml_opencl_legacy_ifairy_state {
    cl_mem ifairy64_act_q_scratch = nullptr;
    cl_mem ifairy64_act_d_scratch = nullptr;
    size_t ifairy64_act_q_capacity = 0;
    size_t ifairy64_act_d_capacity = 0;

    cl_program program_ifairy_add = nullptr;
    cl_program program_ifairy_merge = nullptr;
    cl_program program_ifairy64 = nullptr;
    cl_program program_ifairy_mul = nullptr;
    cl_program program_ifairy_relu2 = nullptr;
    cl_program program_ifairy_rope = nullptr;
    cl_program program_ifairy_rms_norm = nullptr;
    cl_program program_ifairy_split = nullptr;

    cl_kernel kernel_ifairy_add = nullptr;
    cl_kernel kernel_ifairy_merge = nullptr;
    cl_kernel kernel_ifairy_q16_quantize_block127 = nullptr;
    cl_kernel kernel_ifairy64_mul_mat_f32_q16 = nullptr;
    cl_kernel kernel_ifairy64_mul_mat_f32_direct = nullptr;
    cl_kernel kernel_ifairy64_mul_vec_f32_q16 = nullptr;
    cl_kernel kernel_ifairy64_mul_vec4_f32_q16 = nullptr;
    cl_kernel kernel_ifairy_mul = nullptr;
    cl_kernel kernel_ifairy_relu2 = nullptr;
    cl_kernel kernel_ifairy_rope = nullptr;
    cl_kernel kernel_ifairy_rms_norm = nullptr;
    cl_kernel kernel_ifairy_rms_norm_mul = nullptr;
    cl_kernel kernel_ifairy_split = nullptr;
};

bool ggml_opencl_legacy_ifairy_compile_enabled(void);
const char * ggml_opencl_legacy_ifairy_runtime_env(void);
const char * ggml_opencl_legacy_ifairy64_mul_mat_impl_env(void);
