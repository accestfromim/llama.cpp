#pragma once

#include "ggml-ifairy-lut.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline bool ggml_fairy2i_env_enabled(const char * name) {
    return ggml_ifairy_env_enabled(name);
}

static inline int ggml_fairy2i_env_get_int_nonzero(const char * name, int default_value) {
    return ggml_ifairy_env_get_int_nonzero(name, default_value);
}

void   ggml_fairy2i_lut_init(void);
void   ggml_fairy2i_lut_free(void);
bool   ggml_fairy2i_lut_can_mul_mat(const struct ggml_tensor * src0,
                                    const struct ggml_tensor * src1,
                                    const struct ggml_tensor * dst);
size_t ggml_fairy2i_lut_get_wsize(const struct ggml_tensor * src0,
                                  const struct ggml_tensor * src1,
                                  const struct ggml_tensor * dst,
                                  int                        n_threads);
bool   ggml_fairy2i_lut_transform_tensor(struct ggml_tensor * tensor, struct ggml_tensor ** index_tensor_out);

#ifdef __cplusplus
}
#endif
