#pragma once

#include "../../quants.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool ggml_fairy2i_tile64_w2_arm_neon_available(void);
bool ggml_fairy2i_tile64_w2_arm_dotprod_available(void);
const char * ggml_fairy2i_tile64_w2_arm_path_name(void);

void ggml_fairy2i_tile64_fuse_accumulate_block_four_neon(const block_fairy2i_tile64_v2 *  u0,
                                                         const block_fairy2i_tile64_v2 *  u1,
                                                         const block_fairy2i_tile64_v2 *  w0,
                                                         const block_fairy2i_tile64_v2 *  w1,
                                                         const block_fairy2i_act_q16_64 * x,
                                                         int32_t                          sums[4][4]);

void ggml_fairy2i_tile64_fuse_accumulate_block_four_dotprod(const block_fairy2i_tile64_v2 *  u0,
                                                            const block_fairy2i_tile64_v2 *  u1,
                                                            const block_fairy2i_tile64_v2 *  w0,
                                                            const block_fairy2i_tile64_v2 *  w1,
                                                            const block_fairy2i_act_q16_64 * x,
                                                            int32_t                          sums[4][4]);

bool ggml_fairy2i_tile64_fuse_accumulate_block_four_arm(const block_fairy2i_tile64_v2 *  u0,
                                                        const block_fairy2i_tile64_v2 *  u1,
                                                        const block_fairy2i_tile64_v2 *  w0,
                                                        const block_fairy2i_tile64_v2 *  w1,
                                                        const block_fairy2i_act_q16_64 * x,
                                                        int32_t                          sums[4][4]);

#ifdef __cplusplus
}
#endif
