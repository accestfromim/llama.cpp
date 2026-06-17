#pragma once

#ifndef GGML_OPENCL_TARGET_VERSION
#    define GGML_OPENCL_TARGET_VERSION 120
#endif
#ifndef CL_TARGET_OPENCL_VERSION
#    define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#endif

#include <CL/cl.h>

#include <cstddef>
#include <string>

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

#ifndef GGML_OPENCL_PROGRAM_BUILDER_DEFINED
#    define GGML_OPENCL_PROGRAM_BUILDER_DEFINED
using ggml_opencl_program_builder = cl_program (*)(cl_context ctx,
                                                   cl_device_id dev,
                                                   const char * program_buffer,
                                                   const std::string & compile_opts);
#endif

bool ggml_opencl_legacy_ifairy_compile_enabled(void);
const char * ggml_opencl_legacy_ifairy_runtime_env(void);
const char * ggml_opencl_legacy_ifairy64_mul_mat_impl_env(void);

void ggml_opencl_legacy_ifairy_load_kernels(ggml_opencl_legacy_ifairy_state * state,
                                            cl_context context,
                                            cl_device_id device,
                                            const std::string & compile_opts,
                                            ggml_opencl_program_builder build_program);
void ggml_opencl_legacy_ifairy_release_scratch(ggml_opencl_legacy_ifairy_state * state);
