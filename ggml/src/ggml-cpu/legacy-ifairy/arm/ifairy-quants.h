#pragma once

#include "../../quants.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

static inline bool ggml_ifairy_vecdot_act_tensor_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_IFAIRY_VEC_DOT_ACT_TENSOR");
        cached           = (env && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return cached != 0;
}

bool ggml_vec_dot_ifairy_q16_K_dotprod_available(void);

void ggml_vec_dot_ifairy_q16_K_neon(int                        n,
                                    float * GGML_RESTRICT      s,
                                    size_t                     bs,
                                    const void * GGML_RESTRICT vx,
                                    size_t                     bx,
                                    const void * GGML_RESTRICT vy,
                                    size_t                     by,
                                    int                        nrc);

void ggml_vec_dot_ifairy_q16_K_dotprod(int                        n,
                                       float * GGML_RESTRICT      s,
                                       size_t                     bs,
                                       const void * GGML_RESTRICT vx,
                                       size_t                     bx,
                                       const void * GGML_RESTRICT vy,
                                       size_t                     by,
                                       int                        nrc);
