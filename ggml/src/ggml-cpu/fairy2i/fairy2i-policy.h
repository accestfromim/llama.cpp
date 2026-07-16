#pragma once

#include "lut/ggml-fairy2i-lut.h"

#include "ggml.h"
#include "ggml-impl.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

enum ggml_fairy2i_lut_impl {
    GGML_FAIRY2I_LUT_IMPL_AUTO  = 0,
    GGML_FAIRY2I_LUT_IMPL_LUT16 = 1,
    GGML_FAIRY2I_LUT_IMPL_LUT_C = 2,
};

struct ggml_fairy2i_lut_policy {
    bool                      dbg;
    bool                      lut_enabled;
    enum ggml_fairy2i_lut_impl impl;
};

static inline enum ggml_fairy2i_lut_impl ggml_fairy2i_lut_impl_from_env(const char * env_name,
                                                                         bool         dbg,
                                                                         const char * log_prefix) {
    const char * impl_env = getenv(env_name);
    if (!impl_env || impl_env[0] == '\0' || strcmp(impl_env, "0") == 0 || strcmp(impl_env, "auto") == 0 ||
        strcmp(impl_env, "lut16") == 0) {
        return GGML_FAIRY2I_LUT_IMPL_LUT16;
    }
    if (strcmp(impl_env, "lut_c") == 0) {
        return GGML_FAIRY2I_LUT_IMPL_LUT_C;
    }
    if (dbg) {
        GGML_LOG_WARN("%s: unknown %s=%s (expected auto|lut16|lut_c)\n", log_prefix, env_name, impl_env);
    }
    return GGML_FAIRY2I_LUT_IMPL_LUT16;
}

static inline struct ggml_fairy2i_lut_policy ggml_fairy2i_lut_policy_from_env(void) {
    struct ggml_fairy2i_lut_policy policy;

    policy.dbg = ggml_fairy2i_env_enabled("GGML_FAIRY2I_LUT_DEBUG");

    const char * enabled_env = getenv("GGML_FAIRY2I_LUT");
    policy.lut_enabled       = !(enabled_env && strcmp(enabled_env, "0") == 0);
    policy.impl              = ggml_fairy2i_lut_impl_from_env("GGML_FAIRY2I_LUT_IMPL", policy.dbg, "fairy2i_lut");

    return policy;
}

static inline bool ggml_fairy2i_lut_enabled_by_policy(void) {
    return ggml_fairy2i_lut_policy_from_env().lut_enabled;
}

static inline bool ggml_fairy2i_test_require_lut(void) {
    return ggml_fairy2i_env_enabled("GGML_FAIRY2I_TEST_REQUIRE_LUT");
}
