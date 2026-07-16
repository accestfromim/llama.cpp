#pragma once

#include "fairy2i-bundle.h"

#include <stdbool.h>
#include <stddef.h>

void ggml_fairy2i_tile64_lut_qgemm_pair_cpu(int          m,
                                      int          k,
                                      int          n,
                                      const void * packed_wtiles0,
                                      const void * packed_wtiles1,
                                      const void * lut,
                                      const void * lut_scales,
                                      float *      dst,
                                      size_t       dst_col_stride,
                                      size_t       dst_row_stride,
                                      bool         pack_bf16);

bool ggml_fairy2i_tile64_lut_qgemm_two_cpu(int          m,
                                      int          k,
                                      int          n,
                                      const void * packed_u0,
                                      const void * packed_w0,
                                      const void * lut,
                                      const void * lut_scales,
                                      float *      dst,
                                      size_t       dst_col_stride,
                                      size_t       dst_row_stride,
                                      bool         pack_bf16);

bool ggml_fairy2i_tile64_lut_qgemm_four_cpu(int          m,
                                      int          k,
                                      int          n,
                                      const void * packed_u0,
                                      const void * packed_u1,
                                      const void * packed_w0,
                                      const void * packed_w1,
                                      const void * lut,
                                      const void * lut_scales,
                                      float *      dst,
                                      size_t       dst_col_stride,
                                      size_t       dst_row_stride,
                                      bool         pack_bf16);

bool ggml_fairy2i_bundle_lut_qgemm_two_cpu(int                                     m,
                                           int                                     k,
                                           int                                     n,
                                           const struct ggml_fairy2i_bundle_desc * bundle,
                                           int64_t                                 global_m16_offset,
                                           const void *                            lut,
                                           const void *                            lut_scales,
                                           float *                                 dst,
                                           size_t                                  dst_col_stride,
                                           size_t                                  dst_row_stride,
                                           bool                                    pack_bf16);

bool ggml_fairy2i_bundle_lut_qgemm_four_cpu(int                                     m,
                                            int                                     k,
                                            int                                     n,
                                            const struct ggml_fairy2i_bundle_desc * bundle,
                                            int64_t                                 global_m16_offset,
                                            const void *                            lut,
                                            const void *                            lut_scales,
                                            float *                                 dst,
                                            size_t                                  dst_col_stride,
                                            size_t                                  dst_row_stride,
                                            bool                                    pack_bf16);
