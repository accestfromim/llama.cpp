#pragma once

#if !defined(QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK)
#    ifdef __cplusplus
#        define GGML_COMMON_DECL_CPP
#    else
#        define GGML_COMMON_DECL_C
#    endif
#    include "../../../ggml-common.h"
#endif

#include "ggml-backend.h"
#include "ggml.h"

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline bool ggml_fairy2i_env_enabled(const char * name) {
    const char * env = getenv(name);
    return env && strcmp(env, "0") != 0;
}

static inline int ggml_fairy2i_env_get_int_nonzero(const char * name, int default_value) {
    const char * env = getenv(name);
    if (!env || strcmp(env, "0") == 0) {
        return default_value;
    }

    errno          = 0;
    char *     end = NULL;
    const long v   = strtol(env, &end, 10);
    if (end == env) {
        return default_value;
    }

    if (v > (long) INT_MAX) {
        return INT_MAX;
    }
    if (v < (long) INT_MIN) {
        return INT_MIN;
    }
    if (errno == ERANGE) {
        return v < 0 ? INT_MIN : INT_MAX;
    }
    return (int) v;
}

struct fairy2i_lut_extra {
    uint8_t *             indexes;
    size_t                size;
    void *                packed_w;
    size_t                packed_w_size;
    struct ggml_tensor *  index_tensor;
    ggml_backend_buffer_t index_buffer;
};

#if defined(_MSC_VER)
#    define GGML_FAIRY2I_LUT_ALIGN(n) __declspec(align(n))
#else
#    define GGML_FAIRY2I_LUT_ALIGN(n) __attribute__((aligned(n)))
#endif

#define GGML_FAIRY2I_LUT_WTILE_ALIGNMENT 64

typedef block_fairy2i_tile64_v2 block_fairy2i;
typedef block_fairy2i_act_q16_64 block_fairy2i_q16;
#define QK_FAIRY2I QK_FAIRY2I_TILE64
#define QK_FAIRY2I_GROUPS_PER_BLOCK QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK

struct GGML_FAIRY2I_LUT_ALIGN(GGML_FAIRY2I_LUT_WTILE_ALIGNMENT) fairy2i_lut_wtile_16 {
    uint8_t   qs[QK_FAIRY2I_TILE64_GROUPS_PER_BLOCK / 2][16];
    ggml_half d_real[16];
    ggml_half d_imag[16];
};
typedef struct fairy2i_lut_wtile_16 fairy2i_tile64_lut_wtile_16;
typedef struct fairy2i_lut_wtile_16 fairy2i_lut_wtile_16;
static_assert(sizeof(fairy2i_lut_wtile_16) == 320, "wrong fairy2i_lut_wtile_16 size");

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

void ggml_fairy2i_lut_preprocess_ex_lut16(int          m,
                                          int          k,
                                          int          n,
                                          const void * act,
                                          size_t       act_stride,
                                          void *       lut_scales,
                                          void *       lut_buf,
                                          int          ith,
                                          int          nth);
void ggml_fairy2i_lut_preprocess_ex_lut_c(int          m,
                                          int          k,
                                          int          n,
                                          const void * act,
                                          size_t       act_stride,
                                          void *       lut_scales,
                                          void *       lut_buf,
                                          int          ith,
                                          int          nth);
void ggml_fairy2i_tile64_lut_preprocess_ex_lut16(int          m,
                                                 int          k,
                                                 int          n,
                                                 const void * act,
                                                 size_t       act_stride,
                                                 void *       lut_scales,
                                                 void *       lut_buf,
                                                 int          ith,
                                                 int          nth);
void ggml_fairy2i_tile64_lut_preprocess_ex_q16_64_lut16(int          m,
                                                        int          k,
                                                        int          n,
                                                        const void * act,
                                                        size_t       act_stride,
                                                        void *       lut_scales,
                                                        void *       lut_buf,
                                                        int          ith,
                                                        int          nth);
void ggml_fairy2i_tile64_lut_preprocess_q16_64_block_lut16(const void * act_block,
                                                           void *       lut_scales,
                                                           void *       lut_block);
void ggml_fairy2i_tile64_lut_preprocess_ex_lut_c(int          m,
                                                 int          k,
                                                 int          n,
                                                 const void * act,
                                                 size_t       act_stride,
                                                 void *       lut_scales,
                                                 void *       lut_buf,
                                                 int          ith,
                                                 int          nth);

void ggml_fairy2i_lut_qgemm_lut16(int          m,
                                  int          k,
                                  int          n,
                                  const void * packed_w,
                                  const void * lut_scales,
                                  const void * lut_buf,
                                  size_t       lut_stride,
                                  float *      dst,
                                  int          ith,
                                  int          nth);
void ggml_fairy2i_lut_qgemm_fused_lut16(int          m,
                                        int          k,
                                        int          n,
                                        const void * packed_w,
                                        const void * lut_scales,
                                        const void * lut_buf,
                                        size_t       lut_stride,
                                        void *       dst,
                                        size_t       dst_stride,
                                        int          ith,
                                        int          nth);
void ggml_fairy2i_tile64_lut_qgemm_lut16(int          m,
                                         int          k,
                                         int          n,
                                         const void * packed_w,
                                         const void * lut_scales,
                                         const void * lut_buf,
                                         size_t       lut_stride,
                                         float *      dst,
                                         int          ith,
                                         int          nth);
void ggml_fairy2i_tile64_lut_qgemm_pair_lut16(int          m,
                                              int          k,
                                              int          n,
                                              const void * packed_w0,
                                              const void * packed_w1,
                                              const void * lut_scales,
                                              const void * lut_buf,
                                              size_t       lut_stride,
                                              float *      dst0,
                                              float *      dst1,
                                              int          ith,
                                              int          nth);
void ggml_fairy2i_tile64_lut_qgemm_fused_lut16(int          m,
                                               int          k,
                                               int          n,
                                               const void * packed_w,
                                               const void * lut_scales,
                                               const void * lut_buf,
                                               size_t       lut_stride,
                                               void *       dst,
                                               size_t       dst_stride,
                                               int          ith,
                                               int          nth);
void ggml_fairy2i_lut_qgemm_lut_c(int          m,
                                  int          k,
                                  int          n,
                                  const void * packed_w,
                                  const void * lut_scales,
                                  const void * lut_buf,
                                  size_t       lut_stride,
                                  float *      dst,
                                  int          ith,
                                  int          nth);
void ggml_fairy2i_tile64_lut_qgemm_lut_c(int          m,
                                         int          k,
                                         int          n,
                                         const void * packed_w,
                                         const void * lut_scales,
                                         const void * lut_buf,
                                         size_t       lut_stride,
                                         float *      dst,
                                         int          ith,
                                         int          nth);
void ggml_fairy2i_lut_qgemm_fused_lut_c(int          m,
                                        int          k,
                                        int          n,
                                        const void * packed_w,
                                        const void * lut_scales,
                                        const void * lut_buf,
                                        size_t       lut_stride,
                                        void *       dst,
                                        size_t       dst_stride,
                                        int          ith,
                                        int          nth);
void ggml_fairy2i_tile64_lut_qgemm_fused_lut_c(int          m,
                                               int          k,
                                               int          n,
                                               const void * packed_w,
                                               const void * lut_scales,
                                               const void * lut_buf,
                                               size_t       lut_stride,
                                               void *       dst,
                                               size_t       dst_stride,
                                               int          ith,
                                               int          nth);

#ifdef __cplusplus
}
#endif
