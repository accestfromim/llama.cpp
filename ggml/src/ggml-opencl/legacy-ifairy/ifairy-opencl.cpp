#include "ifairy-opencl.h"

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
