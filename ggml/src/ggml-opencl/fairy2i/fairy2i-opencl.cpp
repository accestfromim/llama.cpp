#include "fairy2i-opencl.h"

#include "ggml.h"
#include "ggml-impl.h"

#include <fstream>
#include <string>

#define GGML_OPENCL_CHECK(err)                                      \
    do {                                                            \
        cl_int err_ = (err);                                        \
        if (err_ != CL_SUCCESS) {                                   \
            GGML_LOG_ERROR("ggml_opencl: %s error %d at %s:%d\n",  \
                           #err, err_, __FILE__, __LINE__);         \
            GGML_ASSERT(0);                                         \
        }                                                           \
    } while (0)

static std::string ggml_opencl_read_file(const char * path) {
    std::ifstream ifs(path);
    if (!ifs) {
        return "";
    }
    std::string text;
    ifs.seekg(0, std::ios::end);
    text.resize(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    ifs.read(text.data(), text.size());
    return text;
}

bool ggml_opencl_fairy2i_compile_enabled(void) {
#ifdef GGML_USE_FAIRY2I_OPENCL
    return true;
#else
    return false;
#endif
}

const char * ggml_opencl_fairy2i_runtime_env(void) {
    return "GGML_OPENCL_FAIRY2I";
}

const char * ggml_opencl_fairy2i_tile64_mul_mat_impl_env(void) {
    return "GGML_OPENCL_FAIRY2I_TILE64_MUL_MAT_IMPL";
}

const char * ggml_opencl_fairy2i_wide_linear_w2_impl_env(void) {
    return "GGML_OPENCL_FAIRY2I_WIDE_LINEAR_W2_IMPL";
}

void ggml_opencl_fairy2i_load_kernels(ggml_opencl_fairy2i_state * state,
                                      cl_context context,
                                      cl_device_id device,
                                      const std::string & compile_opts,
                                      ggml_opencl_program_builder build_program) {
#ifdef GGML_USE_FAIRY2I_OPENCL
    cl_int err;

    // complex_add
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "complex_add.cl.h"
        };
#else
        const std::string kernel_src = ggml_opencl_read_file("complex_add.cl");
#endif
        state->program_complex_add = build_program(context, device, kernel_src.c_str(), compile_opts);
        GGML_OPENCL_CHECK((state->kernel_complex_add = clCreateKernel(state->program_complex_add, "kernel_complex_add", &err), err));
        GGML_LOG_CONT(".");
    }

    // complex_merge
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "complex_merge.cl.h"
        };
#else
        const std::string kernel_src = ggml_opencl_read_file("complex_merge.cl");
#endif
        state->program_complex_merge = build_program(context, device, kernel_src.c_str(), compile_opts);
        GGML_OPENCL_CHECK((state->kernel_complex_merge = clCreateKernel(state->program_complex_merge, "kernel_complex_merge", &err), err));
        GGML_LOG_CONT(".");
    }

    // fairy2i_tile64
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "fairy2i_tile64.cl.h"
        };
#else
        const std::string kernel_src = ggml_opencl_read_file("fairy2i_tile64.cl");
#endif
        state->program_fairy2i_tile64 = build_program(context, device, kernel_src.c_str(), compile_opts);

        GGML_OPENCL_CHECK((state->kernel_fairy2i_tile64_act_q16_64_quantize =
                               clCreateKernel(state->program_fairy2i_tile64, "kernel_fairy2i_tile64_act_q16_64_quantize", &err),
                           err));
        GGML_OPENCL_CHECK((state->kernel_fairy2i_tile64_mul_mat_f32_act_q16_64 =
                               clCreateKernel(state->program_fairy2i_tile64, "kernel_fairy2i_tile64_mul_mat_f32_act_q16_64", &err),
                           err));
        GGML_OPENCL_CHECK((state->kernel_fairy2i_tile64_mul_mat_f32_direct =
                               clCreateKernel(state->program_fairy2i_tile64, "kernel_fairy2i_tile64_mul_mat_f32_direct", &err),
                           err));
        GGML_OPENCL_CHECK((state->kernel_fairy2i_tile64_mul_vec_f32_act_q16_64 =
                               clCreateKernel(state->program_fairy2i_tile64, "kernel_fairy2i_tile64_mul_vec_f32_act_q16_64", &err),
                           err));
        GGML_OPENCL_CHECK((state->kernel_fairy2i_tile64_mul_vec4_f32_act_q16_64 =
                               clCreateKernel(state->program_fairy2i_tile64, "kernel_fairy2i_tile64_mul_vec4_f32_act_q16_64", &err),
                           err));
        GGML_OPENCL_CHECK((state->kernel_fairy2i_tile64_wide_linear_w2_f32_act_q16_64 =
                               clCreateKernel(state->program_fairy2i_tile64, "kernel_fairy2i_tile64_wide_linear_w2_f32_act_q16_64", &err),
                           err));
        GGML_LOG_CONT(".");
    }

    // complex_mul
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "complex_mul.cl.h"
        };
#else
        const std::string kernel_src = ggml_opencl_read_file("complex_mul.cl");
#endif
        state->program_complex_mul = build_program(context, device, kernel_src.c_str(), compile_opts);
        GGML_OPENCL_CHECK((state->kernel_complex_mul = clCreateKernel(state->program_complex_mul, "kernel_complex_mul", &err), err));
        GGML_LOG_CONT(".");
    }

    // complex_relu2
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "complex_relu2.cl.h"
        };
#else
        const std::string kernel_src = ggml_opencl_read_file("complex_relu2.cl");
#endif
        state->program_complex_relu2 = build_program(context, device, kernel_src.c_str(), compile_opts);
        GGML_OPENCL_CHECK((state->kernel_complex_relu2 = clCreateKernel(state->program_complex_relu2, "kernel_complex_relu2", &err), err));
        GGML_LOG_CONT(".");
    }

    // complex_rms_norm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "complex_rms_norm.cl.h"
        };
#else
        const std::string kernel_src = ggml_opencl_read_file("complex_rms_norm.cl");
#endif
        state->program_complex_rms_norm = build_program(context, device, kernel_src.c_str(), compile_opts);
        GGML_OPENCL_CHECK((state->kernel_complex_rms_norm =
                               clCreateKernel(state->program_complex_rms_norm, "kernel_complex_rms_norm", &err),
                           err));
        GGML_OPENCL_CHECK((state->kernel_complex_rms_norm_mul =
                               clCreateKernel(state->program_complex_rms_norm, "kernel_complex_rms_norm_mul", &err),
                           err));
        GGML_LOG_CONT(".");
    }

    // complex_rope
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "complex_rope.cl.h"
        };
#else
        const std::string kernel_src = ggml_opencl_read_file("complex_rope.cl");
#endif
        state->program_complex_rope = build_program(context, device, kernel_src.c_str(), compile_opts);
        GGML_OPENCL_CHECK((state->kernel_complex_rope = clCreateKernel(state->program_complex_rope, "kernel_complex_rope", &err), err));
        GGML_LOG_CONT(".");
    }

    // complex_split
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "complex_split.cl.h"
        };
#else
        const std::string kernel_src = ggml_opencl_read_file("complex_split.cl");
#endif
        state->program_complex_split = build_program(context, device, kernel_src.c_str(), compile_opts);
        GGML_OPENCL_CHECK((state->kernel_complex_split = clCreateKernel(state->program_complex_split, "kernel_complex_split", &err), err));
        GGML_LOG_CONT(".");
    }
#else
    GGML_UNUSED(state);
    GGML_UNUSED(context);
    GGML_UNUSED(device);
    GGML_UNUSED(compile_opts);
    GGML_UNUSED(build_program);
#endif
}

void ggml_opencl_fairy2i_release_scratch(ggml_opencl_fairy2i_state * state) {
    if (state->fairy2i_tile64_act_q_scratch != nullptr) {
        GGML_OPENCL_CHECK(clReleaseMemObject(state->fairy2i_tile64_act_q_scratch));
        state->fairy2i_tile64_act_q_scratch = nullptr;
    }
    if (state->fairy2i_tile64_act_d_scratch != nullptr) {
        GGML_OPENCL_CHECK(clReleaseMemObject(state->fairy2i_tile64_act_d_scratch));
        state->fairy2i_tile64_act_d_scratch = nullptr;
    }
    state->fairy2i_tile64_act_q_capacity = 0;
    state->fairy2i_tile64_act_d_capacity = 0;
}
