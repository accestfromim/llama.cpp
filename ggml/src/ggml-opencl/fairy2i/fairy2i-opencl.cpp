#include "fairy2i-opencl.h"

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
