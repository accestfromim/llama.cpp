#include "ifairy-opencl.h"

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

bool ggml_opencl_legacy_ifairy_compile_enabled(void) {
#ifdef GGML_USE_LEGACY_IFAIRY_OPENCL
    return true;
#else
    return false;
#endif
}

const char * ggml_opencl_legacy_ifairy_runtime_env(void) {
    return "GGML_OPENCL_IFAIRY64";
}

const char * ggml_opencl_legacy_ifairy64_mul_mat_impl_env(void) {
    return "GGML_OPENCL_IFAIRY64_MUL_MAT_IMPL";
}

void ggml_opencl_legacy_ifairy_load_kernels(ggml_opencl_legacy_ifairy_state * state,
                                            cl_context context,
                                            cl_device_id device,
                                            const std::string & compile_opts,
                                            ggml_opencl_program_builder build_program) {
#ifdef GGML_USE_LEGACY_IFAIRY_OPENCL
    cl_int err;

    // ifairy_add
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "ifairy_add.cl.h"
        };
#else
        const std::string kernel_src = ggml_opencl_read_file("ifairy_add.cl");
#endif
        state->program_ifairy_add = build_program(context, device, kernel_src.c_str(), compile_opts);
        GGML_OPENCL_CHECK((state->kernel_ifairy_add = clCreateKernel(state->program_ifairy_add, "kernel_ifairy_add", &err), err));
        GGML_LOG_CONT(".");
    }

    // ifairy_merge
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "ifairy_merge.cl.h"
        };
#else
        const std::string kernel_src = ggml_opencl_read_file("ifairy_merge.cl");
#endif
        state->program_ifairy_merge = build_program(context, device, kernel_src.c_str(), compile_opts);
        GGML_OPENCL_CHECK((state->kernel_ifairy_merge = clCreateKernel(state->program_ifairy_merge, "kernel_ifairy_merge", &err), err));
        GGML_LOG_CONT(".");
    }

    // ifairy64
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "ifairy64.cl.h"
        };
#else
        const std::string kernel_src = ggml_opencl_read_file("ifairy64.cl");
#endif
        state->program_ifairy64 = build_program(context, device, kernel_src.c_str(), compile_opts);

        GGML_OPENCL_CHECK((state->kernel_ifairy_q16_quantize_block127 =
                               clCreateKernel(state->program_ifairy64, "kernel_ifairy_q16_quantize_block127", &err),
                           err));
        GGML_OPENCL_CHECK((state->kernel_ifairy64_mul_mat_f32_q16 =
                               clCreateKernel(state->program_ifairy64, "kernel_ifairy64_mul_mat_f32_q16", &err),
                           err));
        GGML_OPENCL_CHECK((state->kernel_ifairy64_mul_mat_f32_direct =
                               clCreateKernel(state->program_ifairy64, "kernel_ifairy64_mul_mat_f32_direct", &err),
                           err));
        GGML_OPENCL_CHECK((state->kernel_ifairy64_mul_vec_f32_q16 =
                               clCreateKernel(state->program_ifairy64, "kernel_ifairy64_mul_vec_f32_q16", &err),
                           err));
        GGML_OPENCL_CHECK((state->kernel_ifairy64_mul_vec4_f32_q16 =
                               clCreateKernel(state->program_ifairy64, "kernel_ifairy64_mul_vec4_f32_q16", &err),
                           err));
        GGML_LOG_CONT(".");
    }

    // ifairy_mul
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "ifairy_mul.cl.h"
        };
#else
        const std::string kernel_src = ggml_opencl_read_file("ifairy_mul.cl");
#endif
        state->program_ifairy_mul = build_program(context, device, kernel_src.c_str(), compile_opts);
        GGML_OPENCL_CHECK((state->kernel_ifairy_mul = clCreateKernel(state->program_ifairy_mul, "kernel_ifairy_mul", &err), err));
        GGML_LOG_CONT(".");
    }

    // ifairy_relu2
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "ifairy_relu2.cl.h"
        };
#else
        const std::string kernel_src = ggml_opencl_read_file("ifairy_relu2.cl");
#endif
        state->program_ifairy_relu2 = build_program(context, device, kernel_src.c_str(), compile_opts);
        GGML_OPENCL_CHECK((state->kernel_ifairy_relu2 = clCreateKernel(state->program_ifairy_relu2, "kernel_ifairy_relu2", &err), err));
        GGML_LOG_CONT(".");
    }

    // ifairy_rope
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "ifairy_rope.cl.h"
        };
#else
        const std::string kernel_src = ggml_opencl_read_file("ifairy_rope.cl");
#endif
        state->program_ifairy_rope = build_program(context, device, kernel_src.c_str(), compile_opts);
        GGML_OPENCL_CHECK((state->kernel_ifairy_rope = clCreateKernel(state->program_ifairy_rope, "kernel_ifairy_rope", &err), err));
        GGML_LOG_CONT(".");
    }

    // ifairy_rms_norm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "ifairy_rms_norm.cl.h"
        };
#else
        const std::string kernel_src = ggml_opencl_read_file("ifairy_rms_norm.cl");
#endif
        state->program_ifairy_rms_norm = build_program(context, device, kernel_src.c_str(), compile_opts);
        GGML_OPENCL_CHECK((state->kernel_ifairy_rms_norm =
                               clCreateKernel(state->program_ifairy_rms_norm, "kernel_ifairy_rms_norm", &err),
                           err));
        GGML_OPENCL_CHECK((state->kernel_ifairy_rms_norm_mul =
                               clCreateKernel(state->program_ifairy_rms_norm, "kernel_ifairy_rms_norm_mul", &err),
                           err));
        GGML_LOG_CONT(".");
    }

    // ifairy_split
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "ifairy_split.cl.h"
        };
#else
        const std::string kernel_src = ggml_opencl_read_file("ifairy_split.cl");
#endif
        state->program_ifairy_split = build_program(context, device, kernel_src.c_str(), compile_opts);
        GGML_OPENCL_CHECK((state->kernel_ifairy_split = clCreateKernel(state->program_ifairy_split, "kernel_ifairy_split", &err), err));
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

void ggml_opencl_legacy_ifairy_release_scratch(ggml_opencl_legacy_ifairy_state * state) {
    if (state->ifairy64_act_q_scratch != nullptr) {
        GGML_OPENCL_CHECK(clReleaseMemObject(state->ifairy64_act_q_scratch));
        state->ifairy64_act_q_scratch = nullptr;
    }
    if (state->ifairy64_act_d_scratch != nullptr) {
        GGML_OPENCL_CHECK(clReleaseMemObject(state->ifairy64_act_d_scratch));
        state->ifairy64_act_d_scratch = nullptr;
    }
    state->ifairy64_act_q_capacity = 0;
    state->ifairy64_act_d_capacity = 0;
}
