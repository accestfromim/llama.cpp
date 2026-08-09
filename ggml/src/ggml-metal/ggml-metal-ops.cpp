#include "ggml-metal-ops.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-metal-common.h"
#include "ggml-metal-device.h"
#include "ggml-metal-impl.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <tuple>
#include <vector>

static ggml_metal_buffer_id ggml_metal_get_buffer_id(const ggml_tensor * t) {
    if (!t) {
        return { nullptr, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t) buffer->context;

    return ggml_metal_buffer_get_id(ctx, t);
}

static ggml_metal_buffer_t ggml_metal_get_buffer(const ggml_tensor * t) {
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
    return (ggml_metal_buffer_t) buffer->context;
}

static ggml_metal_pipeline_t ggml_metal_get_pipeline_fairy2i_w2_decode_fc(ggml_metal_library_t lib,
                                                                          int32_t              blocks,
                                                                          int32_t              x_nb0,
                                                                          int32_t              dst_nb0) {
    const char * base = "kernel_fairy2i_wide_linear_w2_bf16_tile4x1_w8_full_nobias_fc_simd";
    char         name[256];

    snprintf(name, sizeof(name), "%s_blocks=%d_xnb0=%d_dstnb0=%d", base, blocks, x_nb0, dst_nb0);

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (pipeline) {
        return pipeline;
    }

    ggml_metal_cv_t cv = ggml_metal_cv_init();
    ggml_metal_cv_set_int32(cv, blocks, FC_FAIRY2I_W2_DECODE + 0);
    ggml_metal_cv_set_int32(cv, x_nb0, FC_FAIRY2I_W2_DECODE + 1);
    ggml_metal_cv_set_int32(cv, dst_nb0, FC_FAIRY2I_W2_DECODE + 2);

    pipeline = ggml_metal_library_compile_pipeline(lib, base, name, cv);

    ggml_metal_cv_free(cv);

    return pipeline;
}

static ggml_metal_pipeline_t ggml_metal_get_pipeline_fairy2i_w1_decode_fc(ggml_metal_library_t lib,
                                                                          int32_t              blocks,
                                                                          int32_t              x_nb0,
                                                                          int32_t              dst_nb0) {
    const char * base = "kernel_fairy2i_wide_linear_w1_bf16_tile8x1_w16_full_nobias_fc_simd";
    char         name[256];

    snprintf(name, sizeof(name), "%s_blocks=%d_xnb0=%d_dstnb0=%d", base, blocks, x_nb0, dst_nb0);

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (pipeline) {
        return pipeline;
    }

    ggml_metal_cv_t cv = ggml_metal_cv_init();
    ggml_metal_cv_set_int32(cv, blocks, FC_FAIRY2I_W1_DECODE + 0);
    ggml_metal_cv_set_int32(cv, x_nb0, FC_FAIRY2I_W1_DECODE + 1);
    ggml_metal_cv_set_int32(cv, dst_nb0, FC_FAIRY2I_W1_DECODE + 2);

    pipeline = ggml_metal_library_compile_pipeline(lib, base, name, cv);

    ggml_metal_cv_free(cv);

    return pipeline;
}

static ggml_metal_pipeline_t ggml_metal_get_pipeline_fairy2i_bundle_w1_decode_fc(ggml_metal_library_t lib,
                                                                                 int32_t              blocks,
                                                                                 int32_t              x_nb0,
                                                                                 int32_t              dst_nb0) {
    const char * base = "kernel_fairy2i_bundle_w1_bf16_tile8x1_w8_full_nobias_fc_simd";
    char         name[256];

    snprintf(name, sizeof(name), "%s_blocks=%d_xnb0=%d_dstnb0=%d", base, blocks, x_nb0, dst_nb0);

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (pipeline) {
        return pipeline;
    }

    ggml_metal_cv_t cv = ggml_metal_cv_init();
    ggml_metal_cv_set_int32(cv, blocks, FC_FAIRY2I_BUNDLE_W1_DECODE + 0);
    ggml_metal_cv_set_int32(cv, x_nb0, FC_FAIRY2I_BUNDLE_W1_DECODE + 1);
    ggml_metal_cv_set_int32(cv, dst_nb0, FC_FAIRY2I_BUNDLE_W1_DECODE + 2);

    pipeline = ggml_metal_library_compile_pipeline(lib, base, name, cv);

    ggml_metal_cv_free(cv);

    return pipeline;
}

static ggml_metal_pipeline_t ggml_metal_get_pipeline_fairy2i_bundle_w1_exact_decode_fc(ggml_metal_library_t lib,
                                                                                       int32_t              blocks,
                                                                                       int32_t              x_nb0,
                                                                                       int32_t              dst_nb0,
                                                                                       bool                 qat) {
    const char * base = qat ? "kernel_fairy2i_bundle_w1_bf16_bf16scale_qat_tile8x1_w8_nobias_fc_simd" :
                              "kernel_fairy2i_bundle_w1_bf16_bf16scale_exact_tile8x1_w8_nobias_fc_simd";
    char         name[256];

    snprintf(name, sizeof(name), "%s_blocks=%d_xnb0=%d_dstnb0=%d", base, blocks, x_nb0, dst_nb0);

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (pipeline) {
        return pipeline;
    }

    ggml_metal_cv_t cv = ggml_metal_cv_init();
    ggml_metal_cv_set_int32(cv, blocks, FC_FAIRY2I_BUNDLE_W1_DECODE + 0);
    ggml_metal_cv_set_int32(cv, x_nb0, FC_FAIRY2I_BUNDLE_W1_DECODE + 1);
    ggml_metal_cv_set_int32(cv, dst_nb0, FC_FAIRY2I_BUNDLE_W1_DECODE + 2);

    pipeline = ggml_metal_library_compile_pipeline(lib, base, name, cv);

    ggml_metal_cv_free(cv);

    return pipeline;
}

static ggml_metal_pipeline_t ggml_metal_get_pipeline_fairy2i_bundle_w2_decode_fc(ggml_metal_library_t lib,
                                                                                 int32_t              blocks,
                                                                                 int32_t              x_nb0,
                                                                                 int32_t              dst_nb0) {
    const char * base = "kernel_fairy2i_bundle_w2_bf16_tile4x1_w8_full_nobias_fc_simd";
    char         name[256];

    snprintf(name, sizeof(name), "%s_blocks=%d_xnb0=%d_dstnb0=%d", base, blocks, x_nb0, dst_nb0);

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (pipeline) {
        return pipeline;
    }

    ggml_metal_cv_t cv = ggml_metal_cv_init();
    ggml_metal_cv_set_int32(cv, blocks, FC_FAIRY2I_BUNDLE_W2_DECODE + 0);
    ggml_metal_cv_set_int32(cv, x_nb0, FC_FAIRY2I_BUNDLE_W2_DECODE + 1);
    ggml_metal_cv_set_int32(cv, dst_nb0, FC_FAIRY2I_BUNDLE_W2_DECODE + 2);

    pipeline = ggml_metal_library_compile_pipeline(lib, base, name, cv);

    ggml_metal_cv_free(cv);

    return pipeline;
}

static ggml_metal_pipeline_t ggml_metal_get_pipeline_fairy2i_bundle_w2_exact_m64_decode_fc(ggml_metal_library_t lib,
                                                                                           int32_t              blocks,
                                                                                           int32_t              x_nb0,
                                                                                           int32_t dst_nb0) {
    const char * base = "kernel_fairy2i_bundle_w2_bf16_bf16scale_exact_m64x1_nobias_fc_simd";
    char         name[256];

    snprintf(name, sizeof(name), "%s_blocks=%d_xnb0=%d_dstnb0=%d", base, blocks, x_nb0, dst_nb0);

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (pipeline) {
        return pipeline;
    }

    ggml_metal_cv_t cv = ggml_metal_cv_init();
    ggml_metal_cv_set_int32(cv, blocks, FC_FAIRY2I_BUNDLE_W2_DECODE + 0);
    ggml_metal_cv_set_int32(cv, x_nb0, FC_FAIRY2I_BUNDLE_W2_DECODE + 1);
    ggml_metal_cv_set_int32(cv, dst_nb0, FC_FAIRY2I_BUNDLE_W2_DECODE + 2);

    pipeline = ggml_metal_library_compile_pipeline(lib, base, name, cv);

    ggml_metal_cv_free(cv);

    return pipeline;
}

static ggml_metal_pipeline_t ggml_metal_get_pipeline_fairy2i_bundle_w1_prefill_fc(ggml_metal_library_t lib,
                                                                                  const char *         base,
                                                                                  int32_t              act_rows) {
    char name[256];

    snprintf(name, sizeof(name), "%s_actrows=%d", base, act_rows);

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (pipeline) {
        return pipeline;
    }

    ggml_metal_cv_t cv = ggml_metal_cv_init();
    ggml_metal_cv_set_int32(cv, act_rows, FC_FAIRY2I_BUNDLE_W1_PREFILL + 0);

    pipeline = ggml_metal_library_compile_pipeline(lib, base, name, cv);

    ggml_metal_cv_free(cv);

    return pipeline;
}

static ggml_metal_pipeline_t ggml_metal_get_pipeline_fairy2i_rms_norm_exact(ggml_metal_library_t lib, bool qat) {
    const char * name = qat ? "kernel_fairy2i_rms_norm_qat_f32" : "kernel_fairy2i_rms_norm_exact_f32";

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (!pipeline) {
        pipeline = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
        ggml_metal_pipeline_set_smem(pipeline, qat ? 32 : 16);
    }

    return pipeline;
}

static ggml_metal_pipeline_t ggml_metal_get_pipeline_fairy2i_elementwise_exact(ggml_metal_library_t lib,
                                                                               enum ggml_op         op,
                                                                               bool                 qat) {
    const char * name = nullptr;
    switch (op) {
        case GGML_OP_FAIRY2I_SILU_EXACT:
            name = qat ? "kernel_fairy2i_silu_qat_f32" : "kernel_fairy2i_silu_exact_f32";
            break;
        case GGML_OP_FAIRY2I_MUL_EXACT:
            name = qat ? "kernel_fairy2i_mul_qat_f32" : "kernel_fairy2i_mul_exact_f32";
            break;
        case GGML_OP_FAIRY2I_PACK_BF16_EXACT:
            name = qat ? "kernel_fairy2i_round_bf16_f32" : "kernel_fairy2i_pack_bf16_exact";
            break;
        default:
            GGML_ABORT("invalid Fairy2i exact elementwise op");
    }

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (!pipeline) {
        pipeline = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }
    return pipeline;
}

static ggml_metal_pipeline_t ggml_metal_get_pipeline_fairy2i_swiglu_qat(ggml_metal_library_t lib, bool packed_bf16) {
    const char * name = packed_bf16 ? "kernel_fairy2i_swiglu_qat_packed_bf16" : "kernel_fairy2i_swiglu_qat_f32";

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (!pipeline) {
        pipeline = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }
    return pipeline;
}

static ggml_metal_pipeline_t ggml_metal_get_pipeline_set_rows_bf16_raw(ggml_metal_library_t lib) {
    constexpr const char * name     = "kernel_set_rows_bf16_raw";
    ggml_metal_pipeline_t  pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (!pipeline) {
        pipeline = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }
    return pipeline;
}

static ggml_metal_pipeline_t ggml_metal_get_pipeline_set_rows_bf16_carrier(
        ggml_metal_library_t lib,
        bool                 elements) {
    const char * name = elements ? "kernel_set_rows_bf16_carrier_elements" : "kernel_set_rows_bf16_carrier_rows";
    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (!pipeline) {
        pipeline = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }
    return pipeline;
}

static ggml_metal_pipeline_t ggml_metal_get_pipeline_fairy2i_rope_exact(ggml_metal_library_t lib, bool qat) {
    const char * name = qat ? "kernel_fairy2i_rope_neox_qat_f32" : "kernel_fairy2i_rope_neox_exact_f32";

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (!pipeline) {
        pipeline = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }

    return pipeline;
}

static ggml_metal_pipeline_t ggml_metal_get_pipeline_fairy2i_flash_attn_exact(ggml_metal_library_t lib,
                                                                              int32_t              dk,
                                                                              int32_t              dv) {
    char name[256];
    snprintf(name, sizeof(name), "kernel_fairy2i_flash_attn_ext_exact_bf16_dk%d_dv%d", dk, dv);

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (!pipeline) {
        pipeline = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }

    return pipeline;
}

static ggml_metal_pipeline_t ggml_metal_get_pipeline_fairy2i_flash_attn_exact_decode_vec(ggml_metal_library_t lib,
                                                                                         const char *         name) {
    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (!pipeline) {
        pipeline = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }
    return pipeline;
}

struct ggml_metal_op {
    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;
    ggml_metal_encoder_t enc;
    ggml_mem_ranges_t    mem_ranges;

    ggml_cgraph * gf;

    int idx_start;
    int idx_end;

    bool use_fusion;
    bool use_concurrency;
    bool use_capture;

    int debug_graph;
    int debug_fusion;
};

ggml_metal_op_t ggml_metal_op_init(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_cgraph * gf,
        int idx_start,
        int idx_end,
        bool use_fusion,
        bool use_concurrency,
        bool use_capture,
        int debug_graph,
        int debug_fusion) {
    ggml_metal_op_t res = new ggml_metal_op();

    *res = {
        /*.dev             =*/ dev,
        /*.lib             =*/ ggml_metal_device_get_library(dev),
        /*.enc             =*/ ggml_metal_encoder_init(cmd_buf, use_concurrency),
        /*.mem_ranges      =*/ ggml_mem_ranges_init(debug_graph),
        /*.gf              =*/ gf,
        /*.idx_start       =*/ idx_start,
        /*.idx_end         =*/ idx_end,
        /*.use_fusion      =*/ use_fusion,
        /*.use_concurrency =*/ use_concurrency,
        /*.use_capture     =*/ use_capture,
        /*.debug_graph     =*/ debug_graph,
        /*.debug_fusion    =*/ debug_fusion,
    };

    return res;
}

void ggml_metal_op_free(ggml_metal_op_t ctx) {
    ggml_metal_encoder_end_encoding(ctx->enc);
    ggml_metal_encoder_free(ctx->enc);
    ggml_mem_ranges_free(ctx->mem_ranges);

    delete ctx;
}

static bool ggml_metal_op_concurrency_reset(ggml_metal_op_t ctx) {
    if (!ctx->mem_ranges) {
        return true;
    }

    ggml_metal_encoder_memory_barrier(ctx->enc);

    ggml_mem_ranges_reset(ctx->mem_ranges);

    return true;
}

static bool ggml_metal_op_concurrency_check(ggml_metal_op_t ctx, const ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return false;
    }

    return ggml_mem_ranges_check(ctx->mem_ranges, node);
}

static bool ggml_metal_op_concurrency_add(ggml_metal_op_t ctx, const ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return true;
    }

    return ggml_mem_ranges_add(ctx->mem_ranges, node);
}

static int ggml_metal_op_encode_impl(ggml_metal_op_t ctx, int idx) {
    struct ggml_cgraph * gf = ctx->gf;

    struct ggml_tensor ** nodes = ggml_graph_nodes(gf) + idx;
    struct ggml_tensor *  node  = nodes[0];

    //GGML_LOG_INFO("%s: encoding node %3d, op = %8s\n", __func__, idx, ggml_op_name(node->op));

    if (ggml_is_empty(node)) {
        return 1;
    }

    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
            {
                // noop -> next node
            } return 1;
        default:
            {
            } break;
    }

    if (!ggml_metal_device_supports_op(ctx->dev, node)) {
        GGML_LOG_ERROR("%s: error: unsupported op '%s'\n", __func__, ggml_op_desc(node));
        GGML_ABORT("unsupported op");
    }

    int n_fuse = 1;

    // check if the current node can run concurrently with other nodes before it
    // the condition is that:
    //  - the current node cannot write to any previous src or dst ranges
    //  - the current node cannot read from any previous dst ranges
    //
    // if the condition is not satisfied, we put a memory barrier and clear all ranges
    // otherwise, we add the new ranges to the encoding context and process the node concurrently
    //
    {
        const bool is_concurrent = ggml_metal_op_concurrency_check(ctx, node);

        if (!is_concurrent) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        if (ctx->debug_graph > 0) {
            GGML_LOG_DEBUG("%s: node[%5d] - %-12s %s\n", __func__, idx, ggml_op_name(node->op), is_concurrent ? "(concurrent)" : "");
        }
        if (ctx->debug_graph > 1) {
            GGML_TENSOR_LOCALS( int64_t, ne0, node->src[0], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb0, node->src[0], nb);
            GGML_TENSOR_LOCALS( int64_t, ne1, node->src[1], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb1, node->src[1], nb);
            GGML_TENSOR_LOCALS( int64_t, ne,  node,         ne);
            GGML_TENSOR_LOCALS(uint64_t, nb,  node,         nb);

            if (node->src[0]) {
                GGML_LOG_DEBUG("%s: src0 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[0]->type), ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03,
                        ggml_is_contiguous(node->src[0]), node->src[0]->name);
            }
            if (node->src[1]) {
                GGML_LOG_DEBUG("%s: src1 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[1]->type), ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13,
                        ggml_is_contiguous(node->src[1]), node->src[1]->name);
            }
            if (node) {
                GGML_LOG_DEBUG("%s: node  - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], 1, %s\n", __func__, ggml_type_name(node->type), ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                        node->name);
            }
        }
    }

    switch (node->op) {
        case GGML_OP_CONCAT:
            {
                n_fuse = ggml_metal_op_concat(ctx, idx);
            } break;
        case GGML_OP_COMPLEX_SPLIT:
        case GGML_OP_COMPLEX_MERGE:
        case GGML_OP_COMPLEX_ADD:
            {
                n_fuse = ggml_metal_op_complex(ctx, idx);
            }
            break;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            {
                n_fuse = ggml_metal_op_bin(ctx, idx);
            } break;
        case GGML_OP_ADD_ID:
            {
                n_fuse = ggml_metal_op_add_id(ctx, idx);
            } break;
        case GGML_OP_REPEAT:
            {
                n_fuse = ggml_metal_op_repeat(ctx, idx);
            } break;
        case GGML_OP_ACC:
            {
                n_fuse = ggml_metal_op_acc(ctx, idx);
            } break;
        case GGML_OP_SCALE:
            {
                n_fuse = ggml_metal_op_scale(ctx, idx);
            } break;
        case GGML_OP_CLAMP:
            {
                n_fuse = ggml_metal_op_clamp(ctx, idx);
            } break;
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
        case GGML_OP_UNARY:
            {
                n_fuse = ggml_metal_op_unary(ctx, idx);
            } break;
        case GGML_OP_GLU:
            {
                n_fuse = ggml_metal_op_glu(ctx, idx);
            } break;
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            {
                n_fuse = ggml_metal_op_sum_rows(ctx, idx);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_fuse = ggml_metal_op_soft_max(ctx, idx);
            } break;
        case GGML_OP_SSM_CONV:
            {
                n_fuse = ggml_metal_op_ssm_conv(ctx, idx);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                n_fuse = ggml_metal_op_ssm_scan(ctx, idx);
            } break;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            {
                n_fuse = ggml_metal_op_rwkv(ctx, idx);
            } break;
        case GGML_OP_MUL_MAT:
            {
                n_fuse = ggml_metal_op_mul_mat(ctx, idx);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                n_fuse = ggml_metal_op_mul_mat_id(ctx, idx);
            } break;
        case GGML_OP_FAIRY2I_WIDE_LINEAR_W1:
        case GGML_OP_FAIRY2I_WIDE_LINEAR_W2:
            {
                n_fuse = ggml_metal_op_fairy2i_wide_linear_w2(ctx, idx);
            }
            break;
        case GGML_OP_ROW4_LINEAR:
        case GGML_OP_W8A8_LINEAR:
            {
                n_fuse = ggml_metal_op_row_quant_linear(ctx, idx);
            }
            break;
        case GGML_OP_GET_ROWS:
            {
                n_fuse = ggml_metal_op_get_rows(ctx, idx);
            } break;
        case GGML_OP_SET_ROWS:
            {
                n_fuse = ggml_metal_op_set_rows(ctx, idx);
            } break;
        case GGML_OP_RMS_NORM:
            {
                n_fuse = ggml_metal_op_rms_norm(ctx, idx);
            } break;
        case GGML_OP_FAIRY2I_RMS_NORM_EXACT:
            {
                n_fuse = ggml_metal_op_fairy2i_rms_norm_exact(ctx, idx);
            }
            break;
        case GGML_OP_FAIRY2I_SILU_EXACT:
        case GGML_OP_FAIRY2I_MUL_EXACT:
        case GGML_OP_FAIRY2I_PACK_BF16_EXACT:
            {
                n_fuse = ggml_metal_op_fairy2i_elementwise_exact(ctx, idx);
            }
            break;
        case GGML_OP_L2_NORM:
            {
                n_fuse = ggml_metal_op_l2_norm(ctx, idx);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                n_fuse = ggml_metal_op_group_norm(ctx, idx);
            } break;
        case GGML_OP_NORM:
            {
                n_fuse = ggml_metal_op_norm(ctx, idx);
            } break;
        case GGML_OP_ROPE:
        case GGML_OP_FAIRY2I_ROPE_EXACT:
            {
                n_fuse = ggml_metal_op_rope(ctx, idx);
            } break;
        case GGML_OP_IM2COL:
            {
                n_fuse = ggml_metal_op_im2col(ctx, idx);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                n_fuse = ggml_metal_op_conv_transpose_1d(ctx, idx);
            } break;
        case GGML_OP_UPSCALE:
            {
                n_fuse = ggml_metal_op_upscale(ctx, idx);
            } break;
        case GGML_OP_PAD:
            {
                n_fuse = ggml_metal_op_pad(ctx, idx);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                n_fuse = ggml_metal_op_pad_reflect_1d(ctx, idx);
            } break;
        case GGML_OP_ARANGE:
            {
                n_fuse = ggml_metal_op_arange(ctx, idx);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                n_fuse = ggml_metal_op_timestep_embedding(ctx, idx);
            } break;
        case GGML_OP_ARGSORT:
            {
                n_fuse = ggml_metal_op_argsort(ctx, idx);
            } break;
        case GGML_OP_LEAKY_RELU:
            {
                n_fuse = ggml_metal_op_leaky_relu(ctx, idx);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                n_fuse = ggml_metal_op_flash_attn_ext(ctx, idx);
            } break;
        case GGML_OP_DUP:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            {
                n_fuse = ggml_metal_op_cpy(ctx, idx);
            } break;
        case GGML_OP_POOL_2D:
            {
                n_fuse = ggml_metal_op_pool_2d(ctx, idx);
            } break;
        case GGML_OP_ARGMAX:
            {
                n_fuse = ggml_metal_op_argmax(ctx, idx);
            } break;
       default:
            {
                GGML_LOG_ERROR("%s: error: node %3d, op = %8s not implemented\n", __func__, idx, ggml_op_name(node->op));
                GGML_ABORT("fatal error");
            }
    }

    if (ctx->debug_graph > 0) {
        if (n_fuse > 1) {
            GGML_LOG_DEBUG("%s:               fuse %d ops\n", __func__, n_fuse);
        }
    }

    // update the mem ranges in the encoding context
    for (int i = 0; i < n_fuse; ++i) {
        if (!ggml_metal_op_concurrency_add(ctx, nodes[i])) {
            ggml_metal_op_concurrency_reset(ctx);
        }
    }

    return n_fuse;
}

int ggml_metal_op_encode(ggml_metal_op_t ctx, int idx) {
    if (ctx->use_capture) {
        ggml_metal_encoder_debug_group_push(ctx->enc, ggml_op_desc(ggml_graph_node(ctx->gf, idx)));
    }

    int res = ggml_metal_op_encode_impl(ctx, idx);
    if (idx + res > ctx->idx_end) {
        GGML_ABORT("fusion error: nodes spanning multiple encoders have been fused. this indicates a bug in the fusion logic %s",
                "https://github.com/ggml-org/llama.cpp/pull/14849");
    }

    if (ctx->use_capture) {
        ggml_metal_encoder_debug_group_pop(ctx->enc);
    }

    return res;
}

int ggml_metal_op_concat(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t dim = ((const int32_t *) op->op_params)[0];

    ggml_metal_kargs_concat args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.dim  =*/ dim,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_base(lib, GGML_OP_CONCAT);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_complex(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    if (op->op == GGML_OP_COMPLEX_ADD) {
        GGML_TENSOR_LOCALS(int32_t, ne0, op->src[0], ne);
        GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
        GGML_TENSOR_LOCALS(int32_t, ne1, op->src[1], ne);
        GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
        GGML_TENSOR_LOCALS(int32_t, ne, op, ne);
        GGML_TENSOR_LOCALS(uint64_t, nb, op, nb);

        ggml_metal_kargs_bin args = {
            /*.ne00 =*/ne00,  /*.ne01 =*/ne01, /*.ne02 =*/ne02, /*.ne03 =*/ne03,
            /*.nb00 =*/nb00,  /*.nb01 =*/nb01, /*.nb02 =*/nb02, /*.nb03 =*/nb03,
            /*.ne10 =*/ne10,  /*.ne11 =*/ne11, /*.ne12 =*/ne12, /*.ne13 =*/ne13,
            /*.nb10 =*/nb10,  /*.nb11 =*/nb11, /*.nb12 =*/nb12, /*.nb13 =*/nb13,
            /*.ne0  =*/ne0,   /*.ne1  =*/ne1,  /*.ne2  =*/ne2,  /*.ne3  =*/ne3,
            /*.nb0  =*/nb0,   /*.nb1  =*/nb1,  /*.nb2  =*/nb2,  /*.nb3  =*/nb3,
            /*.offs =*/0,
            /*.o1   =*/{ 0 },
        };

        const char * pipeline_name = ggml_complex_add_get_qat(op) ? "kernel_complex_add_qat" : "kernel_complex_add";
        ggml_metal_pipeline_t pipeline      = ggml_metal_library_get_pipeline(lib, pipeline_name);
        if (!pipeline) {
            pipeline = ggml_metal_library_compile_pipeline(lib, pipeline_name, pipeline_name, nullptr);
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 3);

        const int nth = std::min(1024, ne0);
        ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

        return 1;
    }

    GGML_TENSOR_LOCALS(int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(int32_t, ne, op, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb, op, nb);

    ggml_metal_kargs_concat args = {
        /*.ne00 =*/ne00, /*.ne01 =*/ne01, /*.ne02 =*/ne02, /*.ne03 =*/ne03,
        /*.nb00 =*/nb00, /*.nb01 =*/nb01, /*.nb02 =*/nb02, /*.nb03 =*/nb03,
        /*.ne10 =*/0,    /*.ne11 =*/0,    /*.ne12 =*/0,    /*.ne13 =*/0,
        /*.nb10 =*/0,    /*.nb11 =*/0,    /*.nb12 =*/0,    /*.nb13 =*/0,
        /*.ne0  =*/ne0,  /*.ne1  =*/ne1,  /*.ne2  =*/ne2,  /*.ne3  =*/ne3,
        /*.nb0  =*/nb0,  /*.nb1  =*/nb1,  /*.nb2  =*/nb2,  /*.nb3  =*/nb3,
        /*.dim  =*/0,
    };

    const char * pipeline_name     = op->op == GGML_OP_COMPLEX_SPLIT ? "kernel_complex_split" : "kernel_complex_merge";
    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline(lib, pipeline_name);
    if (!pipeline) {
        pipeline = ggml_metal_library_compile_pipeline(lib, pipeline_name, pipeline_name, nullptr);
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 2);

    const int nth = std::min(1024, ne0);
    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_repeat(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_repeat(lib, op->type);

    ggml_metal_kargs_repeat args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_acc(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    const size_t pnb1 = ((const int32_t *) op->op_params)[0];
    const size_t pnb2 = ((const int32_t *) op->op_params)[1];
    const size_t pnb3 = ((const int32_t *) op->op_params)[2];
    const size_t offs = ((const int32_t *) op->op_params)[3];

    const bool inplace = (bool) ((const int32_t *) op->op_params)[4];

    if (!inplace) {
        // run a separete kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

        ggml_metal_kargs_cpy args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ pnb1,
        /*.nb02 =*/ pnb2,
        /*.nb03 =*/ pnb3,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
        /*.offs =*/ offs,
        /*.o1   =*/ { 0 },
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_bin(lib, GGML_OP_ADD, 1, false);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne11, ne12, ne13, nth, 1, 1);

    return 1;
}

int ggml_metal_op_scale(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    float scale;
    float bias;
    memcpy(&scale, ((const int32_t *) op->op_params) + 0, sizeof(float));
    memcpy(&bias,  ((const int32_t *) op->op_params) + 1, sizeof(float));

    ggml_metal_kargs_scale args = {
        /*.scale =*/ scale,
        /*.bias  =*/ bias,
    };

    int64_t n = ggml_nelements(op);

    if (n % 4 == 0) {
        n /= 4;
    }

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_unary(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);

    return 1;
}

int ggml_metal_op_clamp(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    float min;
    float max;
    memcpy(&min, ((const int32_t *) op->op_params) + 0, sizeof(float));
    memcpy(&max, ((const int32_t *) op->op_params) + 1, sizeof(float));

    ggml_metal_kargs_clamp args = {
        /*.min =*/ min,
        /*.max =*/ max,
    };

    int64_t n = ggml_nelements(op);

    if (n % 4 == 0) {
        n /= 4;
    }

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_unary(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);

    return 1;
}

int ggml_metal_op_unary(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    int64_t n = ggml_nelements(op);

    if (n % 4 == 0) {
        n /= 4;
    }

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_unary(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         1);

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);

    return 1;
}

int ggml_metal_op_glu(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    if (op->src[1]) {
        GGML_ASSERT(ggml_are_same_shape(op->src[0], op->src[1]));
    }

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_glu(lib, op);

    const int32_t swp = ggml_get_op_params_i32(op, 1);
    const float alpha = ggml_get_op_params_f32(op, 2);
    const float limit = ggml_get_op_params_f32(op, 3);

    const int32_t i00 = swp ? ne0 : 0;
    const int32_t i10 = swp ? 0 : ne0;

    ggml_metal_kargs_glu args = {
        /*.ne00 =*/ ne00,
        /*.nb01 =*/ nb01,
        /*.ne10 =*/ op->src[1] ? ne10 : ne00,
        /*.nb11 =*/ op->src[1] ? nb11 : nb01,
        /*.ne0  =*/ ne0,
        /*.nb1  =*/ nb1,
        /*.i00  =*/ op->src[1] ? 0 : i00,
        /*.i10  =*/ op->src[1] ? 0 : i10,
        /*.alpha=*/ alpha,
        /*.limit=*/ limit
    };

    const int64_t nrows = ggml_nrows(op->src[0]);

    const int32_t nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00/2);

    //[encoder setComputePipelineState:pipeline];
    //[encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
    //if (src1) {
    //    [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
    //} else {
    //    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
    //}
    //[encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
    //[encoder setBytes:&args length:sizeof(args) atIndex:3];

    //[encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_sum_rows(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    ggml_metal_kargs_sum_rows args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_sum_rows(lib, op);

    int nth = 32; // SIMD width

    while (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, ne00);

    const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

    //[encoder setComputePipelineState:pipeline];
    //[encoder setBytes:&args length:sizeof(args) atIndex:0];
    //[encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
    //[encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
    //[encoder setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];

    //[encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_get_rows(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_get_rows(lib, op->src[0]->type);

    ggml_metal_kargs_get_rows args = {
        /*.ne00 =*/ ne00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne10, ne11, ne12, 32, 1, 1);

    return 1;
}

int ggml_metal_op_set_rows(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    const int32_t mode = ggml_get_op_params_i32(op, 0);

    ggml_metal_pipeline_t pipeline;
    if (mode == GGML_SET_ROWS_BF16_CARRIER_ROWS || mode == GGML_SET_ROWS_BF16_CARRIER_ELEMENTS) {
        pipeline = ggml_metal_get_pipeline_set_rows_bf16_carrier(
            lib, mode == GGML_SET_ROWS_BF16_CARRIER_ELEMENTS);
    } else {
        GGML_ASSERT(mode == 0);
        pipeline = op->src[0]->type == GGML_TYPE_BF16 ? ggml_metal_get_pipeline_set_rows_bf16_raw(lib) :
                                                        ggml_metal_library_get_pipeline_set_rows(lib, op->type);
    }

    const int32_t nk0 = ne0/ggml_blck_size(op->type);

    ggml_metal_kargs_set_rows args = {
        /*.mode =*/ mode,
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    if (mode == GGML_SET_ROWS_BF16_CARRIER_ELEMENTS) {
        const int64_t n_elements = ggml_nelements(op->src[0]);
        const int     nth        = std::min<int64_t>(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        ggml_metal_encoder_dispatch_threadgroups(enc, (n_elements + nth - 1) / nth, 1, 1, nth, 1, 1);
        return 1;
    }

    int nth = 32; // SIMD width

    while (nth < nk0 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    int nrptg = 1;
    if (nth > nk0) {
        nrptg = (nth + nk0 - 1)/nk0;
        nth   = nk0;

        if (nrptg*nth > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nrptg--;
        }
    }

    nth = std::min(nth, nk0);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_soft_max(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    float scale;
    float max_bias;

    memcpy(&scale,    ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias, ((const int32_t *) op->op_params) + 1, sizeof(max_bias));

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // softmax

    ggml_metal_kargs_soft_max args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne11        =*/ ne11,
        /*.ne12        =*/ ne12,
        /*.ne13        =*/ ne13,
        /*.nb11        =*/ nb11,
        /*.nb12        =*/ nb12,
        /*.nb13        =*/ nb13,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.scale       =*/ scale,
        /*.max_bias    =*/ max_bias,
        /*.m0          =*/ m0,
        /*.m1          =*/ m1,
        /*.n_head_log2 =*/ n_head_log2,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_soft_max(lib, op);

    int nth = 32; // SIMD width

    if (ne00%4 == 0) {
        while (nth < ne00/4 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    } else {
        while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    }

    const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_ssm_conv(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    ggml_metal_kargs_ssm_conv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_ssm_conv(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 3);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne1, ne02, 1, 1, 1);

    return 1;
}

int ggml_metal_op_ssm_scan(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne4, op->src[4], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb4, op->src[4], nb);
    GGML_TENSOR_LOCALS( int32_t, ne5, op->src[5], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb5, op->src[5], nb);
    GGML_TENSOR_LOCALS( int32_t, ne6, op->src[6], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb6, op->src[6], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    const ggml_tensor * src3 = op->src[3];
    const ggml_tensor * src4 = op->src[4];
    const ggml_tensor * src5 = op->src[5];
    const ggml_tensor * src6 = op->src[6];

    GGML_ASSERT(src3);
    GGML_ASSERT(src4);
    GGML_ASSERT(src5);
    GGML_ASSERT(src6);

    const int64_t d_state      = ne00;
    const int64_t d_inner      = ne01;
    const int64_t n_head       = ne02;
    const int64_t n_group      = ne41;
    const int64_t n_seq_tokens = ne12;
    const int64_t n_seqs       = ne13;

    ggml_metal_kargs_ssm_scan args = {
        /*.d_state      =*/ d_state,
        /*.d_inner      =*/ d_inner,
        /*.n_head       =*/ n_head,
        /*.n_group      =*/ n_group,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seqs       =*/ n_seqs,
        /*.s_off        =*/ ggml_nelements(op->src[1]) * sizeof(float),
        /*.nb01         =*/ nb01,
        /*.nb02         =*/ nb02,
        /*.nb03         =*/ nb03,
        /*.nb11         =*/ nb11,
        /*.nb12         =*/ nb12,
        /*.nb13         =*/ nb13,
        /*.nb21         =*/ nb21,
        /*.nb22         =*/ nb22,
        /*.nb31         =*/ nb31,
        /*.nb41         =*/ nb41,
        /*.nb42         =*/ nb42,
        /*.nb43         =*/ nb43,
        /*.nb51         =*/ nb51,
        /*.nb52         =*/ nb52,
        /*.nb53         =*/ nb53,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_ssm_scan(lib, op);

    const size_t sms = ggml_metal_pipeline_get_smem(pipeline);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), 5);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), 6);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), 7);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         8);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, sms, 0);

    if (ne30 == 1) {
        // Mamba-2
        ggml_metal_encoder_dispatch_threadgroups(enc, d_inner, n_head, n_seqs, d_state, 1, 1);
    } else {
        GGML_ASSERT(d_inner == 1);
        ggml_metal_encoder_dispatch_threadgroups(enc, n_head, n_seqs, 1, d_state, 1, 1);
    }

    return 1;
}

int ggml_metal_op_rwkv(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    const int64_t B = op->op == GGML_OP_RWKV_WKV6 ? op->src[5]->ne[1] : op->src[6]->ne[1];
    const int64_t T = op->src[0]->ne[2];
    const int64_t C = op->ne[0];
    const int64_t H = op->src[0]->ne[1];

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_rwkv(lib, op);

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), ida++);
    if (op->op == GGML_OP_RWKV_WKV7) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), ida++);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &B, sizeof(B), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &T, sizeof(T), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &C, sizeof(C), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &H, sizeof(H), ida++);

    ggml_metal_encoder_dispatch_threadgroups(enc, B * H, 1, 1, C/H, 1, 1);

    return 1;
}

int ggml_metal_op_cpy(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

    GGML_ASSERT(ne00 % ggml_blck_size(op->src[0]->type) == 0);

    // TODO: support
    //const int32_t nk00 = ne00/ggml_blck_size(op->type);
    const int32_t nk00 = ne00;

    int nth = 32; // SIMD width

    while (nth < nk00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;

    // TODO: relax this constraint in the future
    if (ggml_blck_size(op->src[0]->type) == 1 && ggml_blck_size(op->type) == 1) {
        if (nth > nk00) {
            nrptg = (nth + nk00 - 1)/nk00;
            nth   = nk00;

            if (nrptg*nth > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
                nrptg--;
            }
        }
    }

    nth = std::min(nth, nk00);

    ggml_metal_kargs_cpy args = {
        /*.ne00 =*/ nk00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_pool_2d(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    const int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    const int32_t k0 = opts[1];
    const int32_t k1 = opts[2];
    const int32_t s0 = opts[3];
    const int32_t s1 = opts[4];
    const int32_t p0 = opts[5];
    const int32_t p1 = opts[6];

    const int64_t IH = op->src[0]->ne[1];
    const int64_t IW = op->src[0]->ne[0];

    const int64_t N  = op->ne[3];
    const int64_t OC = op->ne[2];
    const int64_t OH = op->ne[1];
    const int64_t OW = op->ne[0];

    const int64_t np = N * OC * OH * OW;

    ggml_metal_kargs_pool_2d args_pool_2d = {
        /* .k0 = */ k0,
        /* .k1 = */ k1,
        /* .s0 = */ s0,
        /* .s1 = */ s1,
        /* .p0 = */ p0,
        /* .p1 = */ p1,
        /* .IH = */ IH,
        /* .IW = */ IW,
        /* .OH = */ OH,
        /* .OW = */ OW,
        /* .np = */ np
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_pool_2d(lib, op, op_pool);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) np);
    const int ntg = (np + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args_pool_2d, sizeof(args_pool_2d), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_mul_mat(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    GGML_ASSERT(ne00 == ne10);

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    const int ne11_mm_min = 8;

    // first try to use small-batch mat-mv kernels
    // these should be efficient for BS [2, ~8]
    if (op->src[1]->type == GGML_TYPE_F32 && (ne00%128 == 0) &&
        (
         (
          (
           op->src[0]->type == GGML_TYPE_F32  || // TODO: helper function
           op->src[0]->type == GGML_TYPE_F16  ||
           op->src[0]->type == GGML_TYPE_Q4_0 ||
           op->src[0]->type == GGML_TYPE_Q4_1 ||
           op->src[0]->type == GGML_TYPE_Q5_0 ||
           op->src[0]->type == GGML_TYPE_Q5_1 ||
           op->src[0]->type == GGML_TYPE_Q8_0 ||
           op->src[0]->type == GGML_TYPE_MXFP4 ||
           op->src[0]->type == GGML_TYPE_IQ4_NL ||
           false) && (ne11 >= 2 && ne11 <= 8)
         ) ||
         (
          (
           op->src[0]->type == GGML_TYPE_Q4_K ||
           op->src[0]->type == GGML_TYPE_Q5_K ||
           op->src[0]->type == GGML_TYPE_Q6_K ||
           false) && (ne11 >= 4 && ne11 <= 8)
         )
        )
       ) {
        // TODO: determine the optimal parameters based on grid utilization
        //       I still don't know why we should not always use the maximum available threads:
        //
        //       nsg = pipeline.maxTotalThreadsPerThreadgroup / 32
        //
        //       my current hypothesis is that the work grid is not evenly divisible for different nsg
        //       values and there can be some tail effects when nsg is high. need to confirm this
        //
        const int nsg    = 2;                 // num simdgroups per threadgroup

        // num threads along row per simdgroup
        int16_t nxpsg = 0;
        if (ne00 % 256 == 0 && ne11 < 3) {
            nxpsg = 16;
        } else if (ne00 % 128 == 0) {
            nxpsg = 8;
        } else {
            nxpsg = 4;
        }

        const int16_t nypsg  = 32/nxpsg;          // num threads along col per simdgroup (i.e. a simdgroup processes that many src0 rows at a time)
        const int16_t r0ptg  = nypsg*nsg;         // num src0 rows per threadgroup
              int16_t r1ptg  = 4;                 // num src1 rows per threadgroup

        // note: not sure how optimal are those across all different hardware. there might be someting cleverer
        switch (ne11) {
            case 2:
                r1ptg = 2; break;
            case 3:
            case 6:
                r1ptg = 3; break;
            case 4:
            case 7:
            case 8:
                r1ptg = 4; break;
            case 5:
                r1ptg = 5; break;
            default:
                GGML_ABORT("unsupported ne11");
        };

        ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_mul_mv_ext(lib, op->src[0]->type, op->src[1]->type, nsg, nxpsg, r1ptg);

        ggml_metal_kargs_mul_mv_ext args = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne10  =*/ ne10,
            /*.ne11  =*/ ne11,
            /*.ne12  =*/ ne12,
            /*.nb10  =*/ nb10,
            /*.nb11  =*/ nb11,
            /*.nb12  =*/ nb12,
            /*.nb13  =*/ nb13,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.r2    =*/ r2,
            /*.r3    =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + r0ptg - 1)/r0ptg), ((ne11 + r1ptg - 1)/r1ptg), ne12*ne13, 32, nsg, 1);
    } else if (
        !ggml_is_transposed(op->src[0]) &&
        !ggml_is_transposed(op->src[1]) &&
        // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
        // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
        props_dev->has_simdgroup_mm &&
        op->src[1]->type == GGML_TYPE_F32 &&
        ne00 % 32 == 0 && ne00 >= 64 &&
        (ne11 > ne11_mm_min || (ggml_is_quantized(op->src[0]->type) && ne12 > 1))) {
        //printf("matrix: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        switch (op->src[0]->type) {
            case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
            case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
            case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
            default: break;
        }

        ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_mul_mm(lib, op->src[0]->type, op->src[1]->type);

        ggml_metal_kargs_mul_mm args = {
            /*.ne00 =*/ ne00,
            /*.ne02 =*/ ne02,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne11 + 31)/32), ((ne01 + 63)/64), ne12*ne13, 128, 1, 1);
    } else {
        ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op);

        ggml_metal_kargs_mul_mv args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
        };

        const int nr0 = ggml_metal_pipeline_get_nr0(pipeline);
        const int nr1 = ggml_metal_pipeline_get_nr1(pipeline);
        const int nsg = ggml_metal_pipeline_get_nsg(pipeline);

        const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (op->src[0]->type == GGML_TYPE_F32 ||
            op->src[0]->type == GGML_TYPE_F16 ||
            op->src[0]->type == GGML_TYPE_BF16 ||
            op->src[0]->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0 - 1)/(nr0)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0*nsg - 1)/(nr0*nsg)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
        }
    }

    return 1;
}

size_t ggml_metal_op_row_quant_linear_extra_act_q(const ggml_tensor * op) {
    assert(op->op == GGML_OP_ROW4_LINEAR || op->op == GGML_OP_W8A8_LINEAR);
    assert(op->src[0]);

    const size_t pad      = GGML_PAD(ggml_nbytes(op), 64) - ggml_nbytes(op);
    const size_t act_rows = (size_t) ggml_nrows(op->src[0]);
    const size_t k        = (size_t) op->src[0]->ne[0];

    const size_t quant_bytes = act_rows * k * sizeof(int8_t) + act_rows * sizeof(float);
    const bool   direct_act  = op->op == GGML_OP_ROW4_LINEAR && act_rows > 8 && act_rows % 32 == 0;

    return pad + quant_bytes +
           (direct_act ? GGML_PAD(quant_bytes, 64) - quant_bytes + act_rows * k * sizeof(ggml_fp16_t) : 0);
}

static int ggml_metal_op_row_quant_linear_impl(ggml_metal_op_t      ctx,
                                               ggml_tensor *        op,
                                               ggml_metal_buffer_id x_buffer,
                                               bool                 x_packed_bf16) {
    GGML_ASSERT(op->op == GGML_OP_ROW4_LINEAR || op->op == GGML_OP_W8A8_LINEAR);
    GGML_ASSERT(op->src[0] && op->src[1] && op->src[2]);
    GGML_ASSERT(!x_packed_bf16 || op->op == GGML_OP_ROW4_LINEAR);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * x      = op->src[0];
    const ggml_tensor * codes  = op->src[1];
    const ggml_tensor * scales = op->src[2];

    const int32_t k               = ggml_get_op_params_i32(op, 2);
    const int32_t m               = ggml_get_op_params_i32(op, 1);
    const int32_t act_rows        = (int32_t) ggml_nrows(x);
    const bool    row4_direct_act = op->op == GGML_OP_ROW4_LINEAR && act_rows > 8 && act_rows % 32 == 0;

    ggml_metal_kargs_row_quant_linear args = {
        /* .k        = */ k,
        /* .m        = */ m,
        /* .act_rows = */ act_rows,
        /* .reserved = */ row4_direct_act ? act_rows : 0,
    };

    const size_t op_pad = GGML_PAD(ggml_nbytes(op), 64) - ggml_nbytes(op);

    ggml_metal_buffer_id act_q = ggml_metal_get_buffer_id(op);
    act_q.offs += ggml_nbytes(op) + op_pad;

    ggml_metal_buffer_id act_scales = act_q;
    act_scales.offs += (size_t) act_rows * (size_t) k * sizeof(int8_t);

    ggml_metal_buffer_id act_h = act_q;
    act_h.offs += GGML_PAD((size_t) act_rows * (size_t) k * sizeof(int8_t) + (size_t) act_rows * sizeof(float), 64);

    {
        const char * pipeline_name =
            x_packed_bf16 ? "kernel_row4_quantize_activation_i8_packed_bf16" : "kernel_row4_quantize_activation_i8";
        ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline(lib, pipeline_name);
        if (!pipeline) {
            pipeline = ggml_metal_library_compile_pipeline(lib, pipeline_name, pipeline_name, nullptr);
        }

        constexpr int nth = 256;
        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, x_buffer, 1);
        ggml_metal_encoder_set_buffer(enc, act_q, 2);
        ggml_metal_encoder_set_buffer(enc, act_scales, 3);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, 8 * sizeof(float), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, act_rows, 1, 1, nth, 1, 1);
    }

    // Publish token-major I8 and scales before the optional coalesced K-major
    // half transpose or the linear dispatch.
    ggml_metal_encoder_memory_barrier(enc);

    if (row4_direct_act) {
        constexpr const char * pipeline_name = "kernel_row4_transpose_activation_i8_kmajor_half";
        ggml_metal_pipeline_t  pipeline      = ggml_metal_library_get_pipeline(lib, pipeline_name);
        if (!pipeline) {
            pipeline = ggml_metal_library_compile_pipeline(lib, pipeline_name, pipeline_name, nullptr);
        }

        constexpr int tile_size   = 32;
        constexpr int tile_stride = 33;
        constexpr int nth         = 256;
        GGML_ASSERT(k % tile_size == 0);
        GGML_ASSERT(act_rows % tile_size == 0);
        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, act_q, 1);
        ggml_metal_encoder_set_buffer(enc, act_h, 2);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, tile_size * tile_stride * sizeof(ggml_fp16_t), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, k / tile_size, act_rows / tile_size, 1, nth, 1, 1);

        // Publish the K-major half activation before direct linear device loads.
        ggml_metal_encoder_memory_barrier(enc);
    }

    enum path_kind {
        PATH_DECODE,
        PATH_SMALL_BATCH,
        PATH_PREFILL,
    };

    const path_kind path   = act_rows == 1 ? PATH_DECODE : act_rows <= 8 ? PATH_SMALL_BATCH : PATH_PREFILL;
    const bool      row4   = op->op == GGML_OP_ROW4_LINEAR;
    const char *    suffix = path == PATH_DECODE ? "decode" : path == PATH_SMALL_BATCH ? "small_batch" : "prefill";
    const bool      row4_decode_o32_o4_staged_act = row4 && path == PATH_DECODE && (k == 4096 || k == 12288);
    const bool      row4_decode_o32_segmented16   = row4 && path == PATH_DECODE && !row4_decode_o32_o4_staged_act;
    const bool      w8_decode_o128_rows16         = !row4 && path == PATH_DECODE && m % 128 == 0;
    const bool      w8_decode_o64_rows8           = !row4 && path == PATH_DECODE && !w8_decode_o128_rows16;
    const bool      row4_prefill_m64n32_native_layout_dualw_direct_act = row4_direct_act;
    const bool      row4_prefill_m64n32_native_layout_dualw = row4 && path == PATH_PREFILL && !row4_direct_act;
    const char *    pipeline_name = row4_prefill_m64n32_native_layout_dualw_direct_act ?
                                        "kernel_row4_w1a8_prefill_m64n32_ilp4_native_layout_dualw_direct_act" :
                                    row4_prefill_m64n32_native_layout_dualw ?
                                        "kernel_row4_w1a8_prefill_m64n32_ilp4_native_layout_dualw" :
                                    row4_decode_o32_o4_staged_act     ? "kernel_row4_w1a8_decode_o32_o4_staged_act" :
                                    row4_decode_o32_segmented16       ? "kernel_row4_w1a8_decode_o32_segmented16" :
                                    w8_decode_o128_rows16             ? "kernel_row8_w8a8_decode_o128_rows16" :
                                    w8_decode_o64_rows8               ? "kernel_row8_w8a8_decode_o64_rows8" :
                                    row4 && path == PATH_SMALL_BATCH  ? "kernel_row4_w1a8_small_batch" :
                                    !row4 && path == PATH_SMALL_BATCH ? "kernel_row8_w8a8_small_batch" :
                                    row4                              ? "kernel_row4_w1a8_prefill" :
                                                                        "kernel_row8_w8a8_prefill";
    const char *    row4_path_detail =
        row4_decode_o32_o4_staged_act ?
            "A8/I32 O32 one-O4-per-SIMDgroup staged-act" :
        row4_decode_o32_segmented16 ?
            "A8/I32 O32 two-O4-per-SIMDgroup segmented16" :
        path != PATH_PREFILL ?
            "A8/I32" :
        row4_prefill_m64n32_native_layout_dualw_direct_act ?
            "half MMA/F32 exact accumulate M64N32 native-layout dual-W direct-act transpose32x32 SIMD-striped" :
            "half MMA/F32 exact accumulate M64N32 native-layout dual-W SIMD-striped";

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline(lib, pipeline_name);
    if (!pipeline) {
        pipeline = ggml_metal_library_compile_pipeline(lib, pipeline_name, pipeline_name, nullptr);
    }

    using log_key = std::tuple<int, int, int32_t, int32_t, int32_t>;

    static std::mutex                 log_mutex;
    static std::set<log_key>          logged;
    thread_local std::vector<log_key> logged_local = [] {
        std::vector<log_key> result;
        result.reserve(16);
        return result;
    }();

    const int     op_index  = row4 ? 0 : 1;
    const log_key key       = std::make_tuple(op_index, (int) path, act_rows, m, k);
    bool          log_shape = false;
    if (std::find(logged_local.begin(), logged_local.end(), key) == logged_local.end()) {
        {
            const std::lock_guard<std::mutex> lock(log_mutex);
            log_shape = logged.emplace(key).second;
        }
        logged_local.emplace_back(key);
    }
    if (log_shape) {
        if (row4) {
            GGML_LOG_INFO(
                "ROW4 Metal W1A8 path: %s layout=m16k128_split8_v1 act_rows=%d O=%d K=%d (%s, BF16 boundary)\n", suffix,
                act_rows, m, k, row4_path_detail);
        } else {
            const char * w8_path_detail = path == PATH_PREFILL  ? "half MMA/F32 K1024 segments/I32 merge" :
                                          w8_decode_o128_rows16 ? "A8/I32 O128 sixteen-rows-per-SIMDgroup" :
                                          w8_decode_o64_rows8   ? "A8/I32 O64 eight-rows-per-SIMDgroup" :
                                                                  "A8/I32";
            GGML_LOG_INFO(
                "ROW8 Metal W8A8 lm_head path: %s layout=s8_m16k128_rowmajor_v1 act_rows=%d O=%d K=%d (%s, BF16 "
                "boundary)\n",
                suffix, act_rows, m, k, w8_path_detail);
        }
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(codes), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(scales), 2);
    ggml_metal_encoder_set_buffer(enc, act_q, 3);
    ggml_metal_encoder_set_buffer(enc, act_scales, 4);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 5);
    if (row4_prefill_m64n32_native_layout_dualw_direct_act) {
        ggml_metal_encoder_set_buffer(enc, act_h, 6);
    }

    if (row4_prefill_m64n32_native_layout_dualw_direct_act) {
        constexpr int row_tile  = 64;
        constexpr int col_tile  = 32;
        constexpr int k_tile    = 32;
        constexpr int out_elems = 4 * 8 * 8;
        constexpr int nth       = 128;
        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        ggml_metal_encoder_set_threadgroup_memory_size(enc, row_tile * k_tile * sizeof(ggml_fp16_t), 0);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, 0, 1);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, out_elems * sizeof(float), 2);
        ggml_metal_encoder_dispatch_threadgroups(enc, m / row_tile, act_rows / col_tile, 1, nth, 1, 1);
    } else if (row4_prefill_m64n32_native_layout_dualw) {
        constexpr int row_tile  = 64;
        constexpr int col_tile  = 32;
        constexpr int k_tile    = 32;
        constexpr int out_elems = 4 * 8 * 8;
        constexpr int nth       = 128;
        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        ggml_metal_encoder_set_threadgroup_memory_size(enc, row_tile * k_tile * sizeof(ggml_fp16_t), 0);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, k_tile * col_tile * sizeof(ggml_fp16_t), 1);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, out_elems * sizeof(float), 2);
        ggml_metal_encoder_dispatch_threadgroups(enc, m / row_tile, (act_rows + col_tile - 1) / col_tile, 1, nth, 1, 1);
    } else if (path == PATH_PREFILL) {
        constexpr int rows_per_tg = 16;
        constexpr int k_tile      = 32;
        constexpr int nth         = 128;
        ggml_metal_encoder_set_threadgroup_memory_size(enc, 16 * k_tile * sizeof(ggml_fp16_t), 0);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, k_tile * rows_per_tg * sizeof(ggml_fp16_t), 1);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, 16 * rows_per_tg * sizeof(float), 2);
        ggml_metal_encoder_dispatch_threadgroups(enc, m / 16, (act_rows + rows_per_tg - 1) / rows_per_tg, 1, nth, 1, 1);
    } else if (row4_decode_o32_o4_staged_act) {
        // Copy activation bytes once per threadgroup, then let eight
        // SIMD-groups independently consume one canonical O4 each across O32.
        constexpr int rows_per_tg = 32;
        constexpr int nth         = 8 * 32;
        GGML_ASSERT(k == 4096 || k == 12288);
        GGML_ASSERT(act_q.offs % 16 == 0);
        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        ggml_metal_encoder_set_threadgroup_memory_size(enc, k * sizeof(int8_t), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, m / rows_per_tg, 1, 1, nth, 1, 1);
    } else if (row4_decode_o32_segmented16) {
        // Four SIMD-groups cover two canonical O16 tiles.  Each SIMD-group
        // uses two sixteen-lane halves to independently reduce two O4s.
        constexpr int rows_per_tg = 32;
        constexpr int nth         = 4 * 32;
        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        ggml_metal_encoder_dispatch_threadgroups(enc, m / rows_per_tg, 1, 1, nth, 1, 1);
    } else if (w8_decode_o128_rows16) {
        // Eight SIMD-groups cover eight O16 tiles; each group reuses one
        // activation char4 across all sixteen rows of its canonical tile.
        constexpr int rows_per_tg = 128;
        constexpr int nth         = 8 * 32;
        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        ggml_metal_encoder_dispatch_threadgroups(enc, m / rows_per_tg, 1, 1, nth, 1, 1);
    } else if (w8_decode_o64_rows8) {
        // Eight SIMD-groups cover four O16 tiles; each group reuses one
        // activation char4 across eight contiguous output rows.
        constexpr int rows_per_tg = 64;
        constexpr int nth         = 8 * 32;
        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        ggml_metal_encoder_dispatch_threadgroups(enc, m / rows_per_tg, 1, 1, nth, 1, 1);
    } else {
        constexpr int lanes_per_row = 8;
        constexpr int nth           = 16 * lanes_per_row;
        ggml_metal_encoder_set_threadgroup_memory_size(enc, nth * sizeof(int32_t), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, m / 16, act_rows, 1, nth, 1, 1);
    }

    return 1;
}

int ggml_metal_op_row_quant_linear(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ggml_graph_node(ctx->gf, idx);
    GGML_ASSERT(op->src[0]);

    return ggml_metal_op_row_quant_linear_impl(ctx, op, ggml_metal_get_buffer_id(op->src[0]), false);
}

int ggml_metal_op_fairy2i_wide_linear_w2(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * x         = op->src[0];
    const bool          is_bundle = op->src[1] && op->src[1]->type == GGML_TYPE_FAIRY2I_BUNDLE_CODES;
    const ggml_tensor * codes     = is_bundle ? op->src[1] : nullptr;
    const ggml_tensor * scales    = is_bundle ? op->src[2] : nullptr;
    const ggml_tensor * u_s0      = is_bundle ? nullptr : op->src[1];
    const ggml_tensor * u_s1      = is_bundle ? nullptr : op->src[2];
    const ggml_tensor * w_s0      = is_bundle ? nullptr : op->src[3];
    const ggml_tensor * w_s1      = is_bundle ? nullptr : op->src[4];
    const ggml_tensor * bias      = is_bundle ? op->src[3] : op->src[5];

    GGML_ASSERT(x);
    if (is_bundle) {
        GGML_ASSERT((op->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W1 || op->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W2) && codes &&
                    scales);
    } else {
        GGML_ASSERT(op->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W2 && u_s0 && w_s0);
        GGML_ASSERT((u_s1 == nullptr) == (w_s1 == nullptr));
    }
    const bool                 is_w1 = is_bundle ? op->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W1 : u_s1 == nullptr;
    const bool                 is_exact_bundle_w1 = is_bundle && is_w1 && scales->type == GGML_TYPE_BF16;
    const bool                 is_exact_bundle_w2 = is_bundle && !is_w1 && scales->type == GGML_TYPE_BF16;
    const bool                 is_exact_bundle    = is_exact_bundle_w1 || is_exact_bundle_w2;
    const bool                 is_qat_bundle      = is_exact_bundle && ggml_fairy2i_wide_linear_get_qat(op);
    const ggml_metal_buffer_id exact_w1_coeff_lut =
        is_exact_bundle_w1 ? ggml_metal_buffer_get_fairy2i_w1_coeff_lut(ggml_metal_get_buffer(scales), scales) :
                             ggml_metal_buffer_id{ nullptr, 0 };
    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);
    if (is_exact_bundle) {
        GGML_ASSERT(props_dev->has_bfloat);
        GGML_ASSERT(props_dev->has_simdgroup_mm);
        GGML_ASSERT(props_dev->has_simdgroup_reduction);
    }
    int32_t strict_staged_reconstruction = is_exact_bundle_w2 ? 1 : 0;
    if (is_exact_bundle_w2) {
        const char * strict_env           = getenv("GGML_FAIRY2I_STRICT_STAGED_RECONSTRUCTION");
        strict_staged_reconstruction      = !strict_env || strcmp(strict_env, "0") != 0;
        static bool logged_reconstruction = false;
        if (!logged_reconstruction) {
            GGML_LOG_INFO("FAIRY2I W2 reconstruction: %s (BF16 codes/scales and BF16 MMA)\n",
                          strict_staged_reconstruction ? "strict_staged_bf16" : "collapsed_f32_bf16");
            logged_reconstruction = true;
        }
    }

    const int32_t                           k                        = (int32_t) x->ne[0];
    const int32_t                           m                        = (int32_t) op->ne[0];
    const int32_t                           act_rows                 = (int32_t) ggml_nrows(x);
    const bool                              use_bundle_w1_direct_act =
        is_bundle && is_w1 && (act_rows % 16 == 0 || (is_exact_bundle_w1 && act_rows != 1));
    // The direct kernel loads two N8 activation tiles. Pad the staged row stride so tail prefill batches can use the
    // same coalesced K-major path; the kernel masks stores above act_rows.
    const int32_t staged_act_rows =
        use_bundle_w1_direct_act ? (int32_t) GGML_PAD(act_rows, act_rows <= 8 ? 8 : 16) : act_rows;
    const bool use_metal3_qat_prefill =
        is_exact_bundle_w1 && is_qat_bundle && act_rows != 1 && props_dev->fairy2i_metal3_compat;
    if (is_exact_bundle_w1) {
        static std::atomic_bool logged_decode         = false;
        static std::atomic_bool logged_prefill_direct = false;
        static std::atomic_bool logged_prefill_staged = false;
        static std::atomic_bool logged_prefill_metal3 = false;
        std::atomic_bool *      logged                = act_rows == 1            ? &logged_decode :
                                                        use_metal3_qat_prefill   ? &logged_prefill_metal3 :
                                                        use_bundle_w1_direct_act ? &logged_prefill_direct :
                                                                                   &logged_prefill_staged;
        if (!logged->exchange(true, std::memory_order_relaxed)) {
            if (act_rows == 1) {
                GGML_LOG_INFO(
                    "FAIRY2I Metal bundle W1 %s path: decode "
                    "(BF16 activation/scale, scalar F32 accumulate)\n",
                    is_qat_bundle ? "QAT" : "exact");
            } else if (use_metal3_qat_prefill) {
                GGML_LOG_INFO(
                    "FAIRY2I Metal bundle W1 QAT path: prefill_metal3_compat "
                    "(BF16 activation/scale, scalar F32 accumulate)\n");
            } else {
                GGML_LOG_INFO("FAIRY2I Metal bundle W1 %s path: %s (BF16 activation/scale, BF16 MMA, F32 accumulate)\n",
                              is_qat_bundle ? "QAT" : "exact",
                              act_rows == 1            ? "decode" :
                              use_bundle_w1_direct_act ? "prefill_direct" :
                                                         "prefill_staged");
            }
        }
    }
    ggml_metal_kargs_fairy2i_wide_linear_w2 args = {
        /*.k         =*/k,
        /*.m         =*/m,
        /*.act_rows  =*/act_rows,
        /*.has_bias  =*/bias ? 1 : 0,
        /*.x_ne1     =*/(int32_t) x->ne[1],
        /*.x_ne2     =*/(int32_t) x->ne[2],
        /*.x_ne3     =*/(int32_t) x->ne[3],
        /*.x_nb0     =*/x->nb[0],
        /*.x_nb1     =*/x->nb[1],
        /*.x_nb2     =*/x->nb[2],
        /*.x_nb3     =*/x->nb[3],
        /*.bias_ne0  =*/bias ? (int32_t) bias->ne[0] : 1,
        /*.bias_ne1  =*/bias ? (int32_t) bias->ne[1] : 1,
        /*.bias_ne2  =*/bias ? (int32_t) bias->ne[2] : 1,
        /*.bias_ne3  =*/bias ? (int32_t) bias->ne[3] : 1,
        /*.bias_nb0  =*/bias ? bias->nb[0] : 0,
        /*.bias_nb1  =*/bias ? bias->nb[1] : 0,
        /*.bias_nb2  =*/bias ? bias->nb[2] : 0,
        /*.bias_nb3  =*/bias ? bias->nb[3] : 0,
        /*.dst_nb0   =*/op->nb[0],
        /*.dst_nb1   =*/op->nb[1],
        /*.dst_nb2   =*/op->nb[2],
        /*.dst_nb3   =*/op->nb[3],
        /*.strict_staged_reconstruction =*/strict_staged_reconstruction,
    };

    const size_t op_act_q_pad = GGML_PAD(ggml_nbytes(op), 32) - ggml_nbytes(op);

    ggml_metal_buffer_id act_q = ggml_metal_get_buffer_id(op);
    act_q.offs += ggml_nbytes(op) + op_act_q_pad;
    ggml_metal_buffer_id act_metrics = act_q;
    if (act_rows != 1 && !use_metal3_qat_prefill) {
        act_metrics.offs += (size_t) staged_act_rows * (size_t) (k / ggml_blck_size(GGML_TYPE_FAIRY2I_TILE64_V2)) * 2u *
                            (size_t) ggml_blck_size(GGML_TYPE_FAIRY2I_ACT_Q16_64) * sizeof(ggml_bf16_t);
    }

    if (act_rows != 1 && !use_metal3_qat_prefill) {
        const char * pipeline_name     = is_exact_bundle_w1 && use_bundle_w1_direct_act ?
                                             "kernel_fairy2i_act_bfloat_64_stage_bf16_kmajor_exact" :
                                         is_exact_bundle          ? "kernel_fairy2i_act_bfloat_64_stage_bf16_exact" :
                                         use_bundle_w1_direct_act ? "kernel_fairy2i_act_half_64_stage_bf16_kmajor" :
                                                                    "kernel_fairy2i_act_half_64_stage_bf16";
        ggml_metal_pipeline_t pipeline = nullptr;
        if (use_bundle_w1_direct_act) {
            pipeline = ggml_metal_get_pipeline_fairy2i_bundle_w1_prefill_fc(lib, pipeline_name, staged_act_rows);
        } else {
            pipeline = ggml_metal_library_get_pipeline(lib, pipeline_name);
            if (!pipeline) {
                pipeline = ggml_metal_library_compile_pipeline(lib, pipeline_name, pipeline_name, nullptr);
            }
        }

        const int nth = ggml_blck_size(GGML_TYPE_FAIRY2I_TILE64_V2);
        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x), 1);
        ggml_metal_encoder_set_buffer(enc, act_q, 2);
        if (is_exact_bundle) {
            ggml_metal_encoder_set_buffer(enc, act_metrics, 3);
            ggml_metal_encoder_set_threadgroup_memory_size(enc, 2 * sizeof(uint32_t), 0);
        }
        ggml_metal_encoder_dispatch_threadgroups(enc, k / nth, act_rows, 1, nth, 1, 1);
    } else if (is_exact_bundle_w1 && !is_qat_bundle) {
        const char *          pipeline_name = "kernel_fairy2i_act_bfloat_64_metric_bf16_exact";
        ggml_metal_pipeline_t pipeline      = ggml_metal_library_get_pipeline(lib, pipeline_name);
        if (!pipeline) {
            pipeline = ggml_metal_library_compile_pipeline(lib, pipeline_name, pipeline_name, nullptr);
        }

        const int nth = ggml_blck_size(GGML_TYPE_FAIRY2I_TILE64_V2);
        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x), 1);
        ggml_metal_encoder_set_buffer(enc, act_q, 2);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, 2 * sizeof(uint32_t), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, k / nth, 1, 1, nth, 1, 1);
    }

    if (act_rows != 1 && !use_metal3_qat_prefill && ctx->use_concurrency) {
        ggml_metal_op_concurrency_reset(ctx);
    }

    if (use_metal3_qat_prefill) {
        constexpr int          rows_per_tile = 8;
        constexpr int          block_slots   = 8;
        constexpr int          nth           = block_slots * 16;
        constexpr const char * pipeline_name = "kernel_fairy2i_bundle_w1_bf16_bf16scale_qat_tile8x1_w8_simd";
        ggml_metal_pipeline_t  pipeline      = ggml_metal_library_get_pipeline(lib, pipeline_name);
        if (!pipeline) {
            pipeline = ggml_metal_library_compile_pipeline(lib, pipeline_name, pipeline_name, nullptr);
        }

        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(codes), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(scales), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x), 3);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(bias), 4);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 5);
        // The QAT specialization does not consume activation metrics, but Metal requires a bound buffer.
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x), 6);
        ggml_metal_encoder_set_buffer(enc, exact_w1_coeff_lut, 7);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, rows_per_tile * (nth / 32) * sizeof(float), 0);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, rows_per_tile * (nth / 32) * sizeof(float), 1);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, block_slots * 16 * 4 * sizeof(uint16_t), 2);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, block_slots * 16 * sizeof(uint8_t), 3);
        ggml_metal_encoder_dispatch_threadgroups(enc, (m + rows_per_tile - 1) / rows_per_tile, act_rows, 1, nth, 1, 1);
        return 1;
    }

    if (act_rows != 1) {
        const char * pipeline_name =
            is_exact_bundle_w1 && use_bundle_w1_direct_act ?
                "kernel_fairy2i_bundle_w1_bfloat_bf16scale_exact_mma32x16_k32_direct_act" :
            is_exact_bundle_w1       ? "kernel_fairy2i_bundle_w1_bfloat_bf16scale_exact_mma32x16_k32" :
            is_exact_bundle_w2       ? "kernel_fairy2i_bundle_w2_bfloat_bf16scale_exact_mma32x16" :
            use_bundle_w1_direct_act ? "kernel_fairy2i_bundle_w1_half_mma32x16_k16_direct_act" :
            is_bundle                ? (is_w1 ? "kernel_fairy2i_bundle_w1_half_mma32x16_k16" :
                                                "kernel_fairy2i_bundle_w2_half_mma32x16") :
            is_w1                    ? "kernel_fairy2i_wide_linear_w1_half_w64scale_mma32x16_k16" :
                                       "kernel_fairy2i_wide_linear_w2_half_w64scale_mma32x16";
        ggml_metal_pipeline_t pipeline      = nullptr;
        if (use_bundle_w1_direct_act) {
            pipeline = ggml_metal_get_pipeline_fairy2i_bundle_w1_prefill_fc(lib, pipeline_name, staged_act_rows);
        } else {
            pipeline = ggml_metal_library_get_pipeline(lib, pipeline_name);
            if (!pipeline) {
                pipeline = ggml_metal_library_compile_pipeline(lib, pipeline_name, pipeline_name, nullptr);
            }
        }

        const int row_tile = 32;
        const int k_tile   = is_exact_bundle_w1 ? 32 : is_w1 ? 16 : 8;
        const int nth      = row_tile * 4;
        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        if (is_bundle) {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(codes), 1);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(scales), 2);
            ggml_metal_encoder_set_buffer(enc, act_q, 3);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(bias), 4);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 5);
            if (is_exact_bundle_w1) {
                ggml_metal_encoder_set_buffer(enc, act_metrics, 6);
                ggml_metal_encoder_set_buffer(enc, exact_w1_coeff_lut, 7);
            }
        } else if (is_w1) {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(u_s0), 1);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(w_s0), 2);
            ggml_metal_encoder_set_buffer(enc, act_q, 3);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(bias), 4);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 5);
        } else {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(u_s0), 1);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(u_s1), 2);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(w_s0), 3);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(w_s1), 4);
            ggml_metal_encoder_set_buffer(enc, act_q, 5);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(bias), 6);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 7);
        }
        const size_t mma_element_size = is_exact_bundle ? sizeof(ggml_bf16_t) : sizeof(ggml_fp16_t);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, row_tile * k_tile * mma_element_size, 0);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, row_tile * k_tile * mma_element_size, 1);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, row_tile * k_tile * mma_element_size, 2);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, row_tile * k_tile * mma_element_size, 3);
        if (use_bundle_w1_direct_act) {
            const int col_tile = staged_act_rows <= 8 ? 8 : 16;
            ggml_metal_encoder_set_threadgroup_memory_size(enc, row_tile * col_tile * 2 * sizeof(float), 4);
        } else {
            ggml_metal_encoder_set_threadgroup_memory_size(enc, k_tile * 8 * mma_element_size, 4);
            ggml_metal_encoder_set_threadgroup_memory_size(enc, k_tile * 8 * mma_element_size, 5);
            ggml_metal_encoder_set_threadgroup_memory_size(enc, k_tile * 8 * mma_element_size, 6);
            ggml_metal_encoder_set_threadgroup_memory_size(enc, k_tile * 8 * mma_element_size, 7);
            ggml_metal_encoder_set_threadgroup_memory_size(enc, row_tile * 16 * 2 * sizeof(float), 8);
        }
        const int col_tile = use_bundle_w1_direct_act && staged_act_rows <= 8 ? 8 : 16;
        ggml_metal_encoder_dispatch_threadgroups(enc, (m + row_tile - 1) / row_tile,
                                                 (act_rows + col_tile - 1) / col_tile, 1, nth, 1, 1);

        return 1;
    }

    if (is_w1) {
        const int  rows_per_tile = 8;
        const int  block_slots   = is_exact_bundle_w1 || (is_bundle && !bias) ? 8 : 16;
        const int  nth           = block_slots * 16;
        const bool full_rows     = (m % rows_per_tile) == 0;
        const int  blocks        = k / ggml_blck_size(GGML_TYPE_FAIRY2I_TILE64_V2);
        const bool use_fc_decode = full_rows && !bias && x->nb[0] == sizeof(uint32_t) &&
                                   op->nb[0] == sizeof(uint32_t) &&
                                   (k % ggml_blck_size(GGML_TYPE_FAIRY2I_TILE64_V2)) == 0;

        ggml_metal_pipeline_t pipeline = nullptr;
        if (is_exact_bundle_w1 && use_fc_decode) {
            pipeline = ggml_metal_get_pipeline_fairy2i_bundle_w1_exact_decode_fc(lib, blocks, (int32_t) x->nb[0],
                                                                                 (int32_t) op->nb[0], is_qat_bundle);
        } else if (is_exact_bundle_w1) {
            const char * pipeline_name = is_qat_bundle ?
                                             "kernel_fairy2i_bundle_w1_bf16_bf16scale_qat_tile8x1_w8_simd" :
                                             "kernel_fairy2i_bundle_w1_bf16_bf16scale_exact_tile8x1_w8_simd";
            pipeline                   = ggml_metal_library_get_pipeline(lib, pipeline_name);
            if (!pipeline) {
                pipeline = ggml_metal_library_compile_pipeline(lib, pipeline_name, pipeline_name, nullptr);
            }
        } else if (is_bundle && use_fc_decode) {
            pipeline = ggml_metal_get_pipeline_fairy2i_bundle_w1_decode_fc(lib, blocks, (int32_t) x->nb[0],
                                                                           (int32_t) op->nb[0]);
        } else if (is_bundle) {
            const char * pipeline_name = "kernel_fairy2i_bundle_w1_bf16_tile8x1_w16_full_simd";
            pipeline                   = ggml_metal_library_get_pipeline(lib, pipeline_name);
            if (!pipeline) {
                pipeline = ggml_metal_library_compile_pipeline(lib, pipeline_name, pipeline_name, nullptr);
            }
        } else if (use_fc_decode) {
            pipeline =
                ggml_metal_get_pipeline_fairy2i_w1_decode_fc(lib, blocks, (int32_t) x->nb[0], (int32_t) op->nb[0]);
        } else {
            const char * pipeline_name = !bias && full_rows ?
                                             "kernel_fairy2i_wide_linear_w1_bf16_tile8x1_w16_full_nobias_simd" :
                                         full_rows ? "kernel_fairy2i_wide_linear_w1_bf16_tile8x1_w16_full_simd" :
                                                     "kernel_fairy2i_wide_linear_w1_bf16_tile8x1_w16_simd";
            pipeline                   = ggml_metal_library_get_pipeline(lib, pipeline_name);
            if (!pipeline) {
                pipeline = ggml_metal_library_compile_pipeline(lib, pipeline_name, pipeline_name, nullptr);
            }
        }

        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(is_bundle ? codes : u_s0), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(is_bundle ? scales : w_s0), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x), 3);
        if (use_fc_decode) {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 4);
        } else {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(bias), 4);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 5);
        }
        if (is_exact_bundle_w1) {
            ggml_metal_encoder_set_buffer(enc, act_q, use_fc_decode ? 5 : 6);
            ggml_metal_encoder_set_buffer(enc, exact_w1_coeff_lut, use_fc_decode ? 6 : 7);
        }
        ggml_metal_encoder_set_threadgroup_memory_size(enc, rows_per_tile * (nth / 32) * sizeof(float), 0);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, rows_per_tile * (nth / 32) * sizeof(float), 1);
        if (is_exact_bundle_w1) {
            ggml_metal_encoder_set_threadgroup_memory_size(enc, block_slots * 16 * 4 * sizeof(uint16_t), 2);
            ggml_metal_encoder_set_threadgroup_memory_size(enc, block_slots * 16 * sizeof(uint8_t), 3);
        }
        ggml_metal_encoder_dispatch_threadgroups(enc, (m + rows_per_tile - 1) / rows_per_tile, 1, 1, nth, 1, 1);
        return 1;
    }

    const bool use_exact_m64_decode =
        is_exact_bundle_w2 && (m % 64) == 0 && (k % ggml_blck_size(GGML_TYPE_FAIRY2I_TILE64_V2)) == 0;
    const bool use_exact_m64_fc_decode =
        use_exact_m64_decode && !bias && x->nb[0] == sizeof(uint32_t) && op->nb[0] == sizeof(uint32_t);
    const int rows_per_tile = use_exact_m64_decode ? 64 : 4;
    const int block_slots   = 8;
    const int nth           = use_exact_m64_decode ? 256 : block_slots * 16;

    const bool full_rows     = (m % 4) == 0;
    const int  blocks        = k / ggml_blck_size(GGML_TYPE_FAIRY2I_TILE64_V2);
    const bool use_fc_decode = !is_exact_bundle_w2 && full_rows && !bias && x->nb[0] == sizeof(uint32_t) &&
                               op->nb[0] == sizeof(uint32_t) && (k % ggml_blck_size(GGML_TYPE_FAIRY2I_TILE64_V2)) == 0;

    ggml_metal_pipeline_t pipeline = nullptr;
    if (use_exact_m64_fc_decode) {
        pipeline = ggml_metal_get_pipeline_fairy2i_bundle_w2_exact_m64_decode_fc(lib, blocks, (int32_t) x->nb[0],
                                                                                 (int32_t) op->nb[0]);
    } else if (use_exact_m64_decode) {
        const char * pipeline_name = "kernel_fairy2i_bundle_w2_bf16_bf16scale_exact_m64x1_simd";
        pipeline                   = ggml_metal_library_get_pipeline(lib, pipeline_name);
        if (!pipeline) {
            pipeline = ggml_metal_library_compile_pipeline(lib, pipeline_name, pipeline_name, nullptr);
        }
    } else if (is_exact_bundle_w2) {
        const char * pipeline_name = "kernel_fairy2i_bundle_w2_bf16_bf16scale_exact_tile4x1_w8_simd";
        pipeline                   = ggml_metal_library_get_pipeline(lib, pipeline_name);
        if (!pipeline) {
            pipeline = ggml_metal_library_compile_pipeline(lib, pipeline_name, pipeline_name, nullptr);
        }
    } else if (is_bundle && use_fc_decode) {
        pipeline =
            ggml_metal_get_pipeline_fairy2i_bundle_w2_decode_fc(lib, blocks, (int32_t) x->nb[0], (int32_t) op->nb[0]);
    } else if (is_bundle) {
        const char * pipeline_name = "kernel_fairy2i_bundle_w2_bf16_tile4x1_w8_full_simd";
        pipeline                   = ggml_metal_library_get_pipeline(lib, pipeline_name);
        if (!pipeline) {
            pipeline = ggml_metal_library_compile_pipeline(lib, pipeline_name, pipeline_name, nullptr);
        }
    } else if (use_fc_decode) {
        pipeline = ggml_metal_get_pipeline_fairy2i_w2_decode_fc(lib, blocks, (int32_t) x->nb[0], (int32_t) op->nb[0]);
    } else {
        const char * pipeline_name = full_rows ? "kernel_fairy2i_wide_linear_w2_bf16_tile4x1_w8_full_simd" :
                                                 "kernel_fairy2i_wide_linear_w2_bf16_tile4x1_w8_simd";
        pipeline                   = ggml_metal_library_get_pipeline(lib, pipeline_name);
        if (!pipeline) {
            pipeline = ggml_metal_library_compile_pipeline(lib, pipeline_name, pipeline_name, nullptr);
        }
    }

    GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    if (is_bundle) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(codes), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(scales), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x), 3);
        if (use_fc_decode || use_exact_m64_fc_decode) {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 4);
        } else {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(bias), 4);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 5);
        }
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(u_s0), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(u_s1), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(w_s0), 3);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(w_s1), 4);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x), 5);
        if (use_fc_decode) {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 6);
        } else {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(bias), 6);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 7);
        }
    }
    if (use_exact_m64_decode) {
        constexpr int exact_decode_block_group = 4;
        ggml_metal_encoder_set_threadgroup_memory_size(enc, exact_decode_block_group * 256 * sizeof(uint32_t), 0);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, exact_decode_block_group * 256 * sizeof(uint32_t), 1);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, exact_decode_block_group * 64 * sizeof(uint32_t), 2);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, exact_decode_block_group * 4 * sizeof(uint32_t), 3);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, exact_decode_block_group * 16 * 2 * sizeof(uint16_t), 4);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, exact_decode_block_group * 32 * 2 * sizeof(uint16_t), 5);
    } else {
        ggml_metal_encoder_set_threadgroup_memory_size(enc, rows_per_tile * (nth / 32) * sizeof(float), 0);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, rows_per_tile * (nth / 32) * sizeof(float), 1);
    }
    if (is_exact_bundle_w2 && !use_exact_m64_decode) {
        ggml_metal_encoder_set_threadgroup_memory_size(enc, block_slots * 4 * 4 * 2 * sizeof(uint16_t), 2);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, block_slots * 2 * 16 * 2 * sizeof(uint16_t), 3);
    }
    ggml_metal_encoder_dispatch_threadgroups(enc, (m + rows_per_tile - 1) / rows_per_tile, 1, 1, nth, 1, 1);

    return 1;
}

size_t ggml_metal_op_fairy2i_wide_linear_w2_extra_act_q(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W1 || op->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W2);

    const ggml_tensor * x = op->src[0];

    const int64_t act_rows = ggml_nrows(x);
    if (act_rows == 1) {
        const ggml_tensor * scales = op->src[2];
        const bool          exact_bundle_w1 =
            op->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W1 && scales && scales->type == GGML_TYPE_BF16;
        if (!exact_bundle_w1) {
            return 0;
        }

        const int64_t blocks = x->ne[0] / ggml_blck_size(GGML_TYPE_FAIRY2I_TILE64_V2);
        const size_t  pad    = GGML_PAD(ggml_nbytes(op), 32) - ggml_nbytes(op);
        return pad + (size_t) blocks * sizeof(uint32_t);
    }

    const int64_t       blocks = x->ne[0] / ggml_blck_size(GGML_TYPE_FAIRY2I_TILE64_V2);
    const ggml_tensor * scales = op->src[2];
    const bool exact_bundle_w1 = op->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W1 && scales && scales->type == GGML_TYPE_BF16;
    const int64_t staged_act_rows = exact_bundle_w1 ? GGML_PAD(act_rows, act_rows <= 8 ? 8 : 16) : act_rows;

    const size_t pad = GGML_PAD(ggml_nbytes(op), 32) - ggml_nbytes(op);
    const size_t q8_size =
        (size_t) staged_act_rows * (size_t) blocks * 2u * (size_t) ggml_blck_size(GGML_TYPE_FAIRY2I_ACT_Q16_64);
    const size_t half_stage_size = q8_size * sizeof(ggml_fp16_t);
    const bool   exact_bundle    = scales && scales->type == GGML_TYPE_BF16;
    const size_t metric_size     = exact_bundle ? (size_t) act_rows * (size_t) blocks * sizeof(uint32_t) : 0;

    return pad + half_stage_size + metric_size;
}

size_t ggml_metal_op_mul_mat_id_extra_tpe(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    const int64_t ne02 = op->src[0]->ne[2]; // n_expert

    return ggml_type_size(GGML_TYPE_I32)*ne02;
}

size_t ggml_metal_op_mul_mat_id_extra_ids(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    const int64_t ne02 = op->src[0]->ne[2]; // n_expert
    const int64_t ne21 = op->src[2]->ne[1]; // n_token

    return ggml_type_size(GGML_TYPE_I32)*ne02*ne21;
}

int ggml_metal_op_mul_mat_id(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    // src2 = ids
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);

    GGML_ASSERT(!ggml_is_transposed(op->src[0]));
    GGML_ASSERT(!ggml_is_transposed(op->src[1]));

    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    const uint32_t r2 = 1;
    const uint32_t r3 = 1;

    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    // ne20 = n_used_experts
    // ne21 = n_rows (batch size)
    const int ne21_mm_id_min = 32;

    if (props_dev->has_simdgroup_mm &&
        ne00 % 32 == 0 && ne00 >= 64 &&
        (ne21 >= ne21_mm_id_min)) {
        GGML_ASSERT(ne00 % 4 == 0);

        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        switch (op->src[0]->type) {
            case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
            case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
            case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
            default: break;
        }

        // extra buffers for intermediate id mapping
        ggml_metal_buffer_id bid_tpe = bid_dst;
        bid_tpe.offs += ggml_nbytes(op);

        ggml_metal_buffer_id bid_ids = bid_tpe;
        bid_ids.offs += ggml_metal_op_mul_mat_id_extra_tpe(op);

        {
            ggml_metal_kargs_mul_mm_id_map0 args = {
                ne02,
                ne10,
                ne11, // n_expert_used (bcast)
                nb11,
                nb12,
                ne21, // n_tokens
                ne20, // n_expert_used
                nb21,
            };

            ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_mul_mm_id_map0(lib, ne02, ne20);

            const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

            GGML_ASSERT(ne02 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

            GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  2);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  3);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, ne02, 1, 1);
        }

        // this barrier is always needed because the next kernel has to wait for the id maps to be computed
        ggml_metal_op_concurrency_reset(ctx);

        {
            ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_mul_mm_id(lib, op->src[0]->type, GGML_TYPE_F16);

            ggml_metal_kargs_mul_mm_id args = {
                /*.ne00  =*/ ne00,
                /*.ne02  =*/ ne02,
                /*.nb01  =*/ nb01,
                /*.nb02  =*/ nb02,
                /*.nb03  =*/ nb03,
                /*.ne11  =*/ ne11, // n_expert_used (bcast)
                /*.nb10  =*/ nb10,
                /*.nb11  =*/ nb11,
                /*.nb12  =*/ nb12,
                /*.nb13  =*/ nb13,
                /*.ne20  =*/ ne20, // n_expert_used
                /*.ne21  =*/ ne21, // n_tokens
                /*.ne0   =*/ ne0,
                /*.ne1   =*/ ne1,
                /*.r2    =*/ r2,
                /*.r3    =*/ r3,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  3);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  4);
            ggml_metal_encoder_set_buffer  (enc, bid_dst,  5);

            const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, (ne21 + 31)/32, (ne01 + 63)/64, ne02, 128, 1, 1);
        }
    } else {
        ggml_metal_kargs_mul_mv_id args = {
            /*.nei0 =*/ ne20,
            /*.nei1 =*/ ne21,
            /*.nbi1 =*/ nb21,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.ne13 =*/ ne13,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nb1  =*/ nb1,
        };

        ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_mul_mv_id(lib, op);

        const int nr0 = ggml_metal_pipeline_get_nr0(pipeline);
        const int nr1 = ggml_metal_pipeline_get_nr1(pipeline);
        const int nsg = ggml_metal_pipeline_get_nsg(pipeline);

        const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

        if (ggml_is_quantized(op->src[0]->type)) {
            GGML_ASSERT(ne00 >= nsg*nr0);
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer(enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst,  3);
        ggml_metal_encoder_set_buffer(enc, bid_src2, 4);

        const int64_t _ne1 = 1;
        const int64_t ne123 = ne20*ne21;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (op->src[0]->type == GGML_TYPE_F32 ||
            op->src[0]->type == GGML_TYPE_F16 ||
            op->src[0]->type == GGML_TYPE_BF16 ||
            op->src[0]->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0 - 1)/(nr0), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0*nsg - 1)/(nr0*nsg), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        }
    }

    return 1;
}

int ggml_metal_op_add_id(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_kargs_add_id args = {
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb11 =*/ nb11,
        /*.nb21 =*/ nb21,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_base(lib, GGML_OP_ADD_ID);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, 1, nth, 1, 1);

    return 1;
}

bool ggml_metal_op_flash_attn_ext_use_vec(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    if (ggml_flash_attn_ext_get_fairy2i_exact(op) && !ggml_flash_attn_ext_get_fairy2i_flash3(op)) {
        return false;
    }

    const int64_t ne00 = op->src[0]->ne[0]; // head size
    const int64_t ne01 = op->src[0]->ne[1];  // batch size

    // use vec kernel if the batch size is small and if the head size is supported
    return (ne01 < 20) && (ne00 % 32 == 0);
}

static bool ggml_metal_op_flash_attn_ext_use_fairy2i_decode_vec(const ggml_tensor * op) {
    if (!ggml_flash_attn_ext_get_fairy2i_exact(op)) {
        return false;
    }

    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];

    const int64_t gqa_ratio = q->ne[2] / k->ne[2];

    return q->ne[0] == 128 && q->ne[1] == 1 && k->ne[0] == 128 && k->ne[1] >= 512 && v->ne[0] == 128 &&
           k->ne[2] == v->ne[2] && q->ne[2] > k->ne[2] && q->ne[2] % k->ne[2] == 0 && gqa_ratio <= 8;
}

size_t ggml_metal_op_flash_attn_ext_extra_tmp(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    if (ggml_flash_attn_ext_get_fairy2i_exact(op) && !ggml_flash_attn_ext_get_fairy2i_flash3(op)) {
        if (!ggml_metal_op_flash_attn_ext_use_fairy2i_decode_vec(op)) {
            return 0;
        }

        const size_t     nrows = (size_t) op->src[0]->ne[1] * op->src[0]->ne[2] * op->src[0]->ne[3];
        constexpr size_t nwg   = 32;
        constexpr size_t dv    = 128;
        return 2 * nrows * (size_t) op->src[1]->ne[1] * sizeof(ggml_bf16_t) + nrows * nwg * dv * sizeof(float);
    }

    if (!ggml_metal_op_flash_attn_ext_use_vec(op)) {
        return 0;
    }

    const int64_t nwg = 32;

    const int64_t ne01 = op->src[0]->ne[1];
    const int64_t ne02 = op->src[0]->ne[2];
    const int64_t ne03 = op->src[0]->ne[3];
    const int64_t ne20 = op->src[2]->ne[0];

    // temp buffer for writing the results from each workgroup
    // - ne20: the size of the Value head
    // -  + 2: the S and M values for each intermediate result
    return ggml_type_size(GGML_TYPE_F32) * (ne01 * ne02 * ne03 * nwg * (ne20 + 2));
}

int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS(int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS(int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS(int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS(int32_t, ne, op, ne);
    GGML_TENSOR_LOCALS(int32_t, nb, op, nb);

    const bool fairy2i_exact  = ggml_flash_attn_ext_get_fairy2i_exact(op);
    const bool fairy2i_flash3 = ggml_flash_attn_ext_get_fairy2i_flash3(op);

    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne11 % 32 == 0);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == op->src[2]->type);

    //GGML_ASSERT(ggml_are_same_shape (src1, src2));
    GGML_ASSERT(ne11 == ne21);
    GGML_ASSERT(ne12 == ne22);

    GGML_ASSERT(!op->src[3] || op->src[3]->type == (fairy2i_exact && !fairy2i_flash3 ? GGML_TYPE_F32 : GGML_TYPE_F16));
    GGML_ASSERT(!op->src[3] ||
                op->src[3]->ne[1] >= GGML_PAD(op->src[0]->ne[1], 8) &&
                    "the Flash-Attention Metal kernel requires the mask to be padded to 8 and at least n_queries big");

    float scale;
    float max_bias;
    float logit_softcap;

    memcpy(&scale, ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias, ((const int32_t *) op->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) op->op_params) + 2, sizeof(logit_softcap));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const bool has_mask  = op->src[3] != NULL;
    const bool has_sinks = op->src[4] != NULL;
    const bool has_bias  = max_bias != 0.0f;
    const bool has_scap  = logit_softcap != 0.0f;

    const uint32_t n_head      = op->src[0]->ne[2];
    const int32_t  n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    GGML_ASSERT(ne01 < 65536);

    if (fairy2i_exact && !fairy2i_flash3) {
        constexpr int32_t nqptg = 8;
        constexpr int32_t ncpsg = 8;
        constexpr int32_t nwg   = 32;

        GGML_ASSERT(props_dev->has_bfloat);
        GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(op->src[1]->type == GGML_TYPE_BF16);
        GGML_ASSERT(op->src[2]->type == GGML_TYPE_BF16);
        GGML_ASSERT(op->src[3] && op->src[3]->type == GGML_TYPE_F32);
        GGML_ASSERT(!op->src[4]);
        GGML_ASSERT(max_bias == 0.0f);
        GGML_ASSERT(logit_softcap == 0.0f);
        GGML_ASSERT(nb00 == sizeof(float));
        GGML_ASSERT(nb10 == sizeof(ggml_bf16_t));
        GGML_ASSERT(nb20 == sizeof(ggml_bf16_t));
        GGML_ASSERT(nb30 == sizeof(float));
        GGML_ASSERT(ne00 % 8 == 0 && ne20 % 8 == 0);
        GGML_ASSERT(ne11 % ncpsg == 0);

        const int32_t gqa_ratio      = ne02 / ne12;
        const bool    use_decode_gqa = ne01 == 1 && ne02 > ne12 && ne02 % ne12 == 0 && gqa_ratio <= nqptg;
        static bool   logged_fairy2i_exact_flash_attn_shape = false;
        if (!logged_fairy2i_exact_flash_attn_shape) {
            GGML_LOG_INFO("FAIRY2I exact FlashAttention: nq=%d q_heads=%d kv_heads=%d gqa=%d path=%s\n", ne01, ne02,
                          ne12, gqa_ratio, use_decode_gqa ? "decode_gqa" : "generic");
            logged_fairy2i_exact_flash_attn_shape = true;
        }

        ggml_metal_kargs_flash_attn_ext args = {
            /*.ne01          =*/ne01,
            /*.ne02          =*/ne02,
            /*.ne03          =*/ne03,
            /*.nb01          =*/nb01,
            /*.nb02          =*/nb02,
            /*.nb03          =*/nb03,
            /*.ne11          =*/ne11,
            /*.ne_12_2       =*/ne12,
            /*.ne_12_3       =*/ne13,
            /*.ns10          =*/int32_t(nb11 / nb10),
            /*.nb11          =*/nb11,
            /*.nb12          =*/nb12,
            /*.nb13          =*/nb13,
            /*.ns20          =*/int32_t(nb21 / nb20),
            /*.nb21          =*/nb21,
            /*.nb22          =*/nb22,
            /*.nb23          =*/nb23,
            /*.ne32          =*/ne32,
            /*.ne33          =*/ne33,
            /*.nb31          =*/nb31,
            /*.nb32          =*/nb32,
            /*.nb33          =*/nb33,
            /*.ne1           =*/ne1,
            /*.ne2           =*/ne2,
            /*.ne3           =*/ne3,
            /*.scale         =*/scale,
            /*.max_bias      =*/0.0f,
            /*.m0            =*/1.0f,
            /*.m1            =*/1.0f,
            /*.n_head_log2   =*/n_head_log2,
            /*.logit_softcap =*/0.0f,
        };

        if (ggml_metal_op_flash_attn_ext_use_fairy2i_decode_vec(op)) {
            static bool logged_fairy2i_exact_decode_vec = false;
            if (!logged_fairy2i_exact_decode_vec) {
                GGML_LOG_INFO(
                    "FAIRY2I exact FlashAttention: long-K decode_vec path (BF16 logits/probabilities, "
                    "parallel F32 reduction)\n");
                logged_fairy2i_exact_decode_vec = true;
            }

            ggml_metal_buffer_id bid_tmp = ggml_metal_get_buffer_id(op);
            bid_tmp.offs += ggml_nbytes(op);

            {
                constexpr const char * name = "kernel_fairy2i_flash_attn_ext_exact_bf16_decode_vec_logits_dk128_dv128";
                ggml_metal_pipeline_t pipeline = ggml_metal_get_pipeline_fairy2i_flash_attn_exact_decode_vec(lib, name);

                ggml_metal_encoder_set_pipeline(enc, pipeline);
                ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[3]), 3);
                ggml_metal_encoder_set_buffer(enc, bid_tmp, 4);

                const size_t smem =
                    GGML_PAD((size_t) nqptg * ne00 * sizeof(ggml_bf16_t) + (size_t) nqptg * ncpsg * sizeof(float), 16);
                ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
                ggml_metal_encoder_dispatch_threadgroups(enc, nwg, ne12, ne03, 32, 1, 1);
            }

            ggml_metal_op_concurrency_reset(ctx);

            {
                constexpr const char * name = "kernel_fairy2i_flash_attn_ext_exact_bf16_decode_vec_softmax_dk128_dv128";
                ggml_metal_pipeline_t pipeline = ggml_metal_get_pipeline_fairy2i_flash_attn_exact_decode_vec(lib, name);

                ggml_metal_encoder_set_pipeline(enc, pipeline);
                ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, bid_tmp, 1);
                ggml_metal_encoder_set_threadgroup_memory_size(enc, 16 * sizeof(float), 0);
                ggml_metal_encoder_dispatch_threadgroups(enc, ne02 * ne03, 1, 1, 32, 8, 1);
            }

            ggml_metal_op_concurrency_reset(ctx);

            {
                constexpr const char * name =
                    "kernel_fairy2i_flash_attn_ext_exact_bf16_decode_vec_output_partial_dk128_dv128";
                ggml_metal_pipeline_t pipeline = ggml_metal_get_pipeline_fairy2i_flash_attn_exact_decode_vec(lib, name);

                ggml_metal_encoder_set_pipeline(enc, pipeline);
                ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 1);
                ggml_metal_encoder_set_buffer(enc, bid_tmp, 2);
                ggml_metal_encoder_set_threadgroup_memory_size(
                    enc, (size_t) nqptg * ncpsg * sizeof(ggml_bf16_t) + (size_t) nqptg * ne20 * sizeof(float), 0);
                ggml_metal_encoder_dispatch_threadgroups(enc, nwg, ne12, ne03, 32, 1, 1);
            }

            ggml_metal_op_concurrency_reset(ctx);

            {
                constexpr const char * name =
                    "kernel_fairy2i_flash_attn_ext_exact_bf16_decode_vec_output_reduce_dk128_dv128";
                ggml_metal_pipeline_t pipeline = ggml_metal_get_pipeline_fairy2i_flash_attn_exact_decode_vec(lib, name);

                ggml_metal_encoder_set_pipeline(enc, pipeline);
                ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, bid_tmp, 1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 2);
                ggml_metal_encoder_dispatch_threadgroups(enc, ne02 * ne03, 1, 1, ne20, 1, 1);
            }

            return 1;
        }

        ggml_metal_pipeline_t pipeline = ggml_metal_get_pipeline_fairy2i_flash_attn_exact(lib, ne00, ne20);
        const size_t          smem =
            GGML_PAD((size_t) nqptg * ne00 * sizeof(ggml_bf16_t) + (size_t) nqptg * ncpsg * sizeof(float) +
                         (size_t) nqptg * ncpsg * sizeof(ggml_bf16_t) + (size_t) 3 * nqptg * sizeof(float) +
                         (size_t) nqptg * ne20 * sizeof(float),
                     16);

        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);
        GGML_ASSERT(32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        GGML_LOG_DEBUG("%s: Fairy2i exact FlashAttention BF16 pipeline dk=%d dv=%d nq=%d nkv=%d%s\n", __func__, ne00,
                       ne20, ne01, ne11, use_decode_gqa ? " decode_gqa" : "");

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 3);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[3]), 4);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 5);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 6);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, use_decode_gqa ? 1 : (ne01 + nqptg - 1) / nqptg,
                                                 use_decode_gqa ? ne12 : ne02, ne03, 32, 1, 1);

        return 1;
    }

    if (fairy2i_flash3) {
        static bool logged_fairy2i_flash3 = false;
        if (!logged_fairy2i_flash3) {
            GGML_LOG_INFO("FAIRY2I FlashAttention-3 path: BF16 Q/K/V with the standard online-softmax schedule\n");
            logged_fairy2i_flash3 = true;
        }
    }

    if (!ggml_metal_op_flash_attn_ext_use_vec(op)) {
        // half8x8 kernel
        const int64_t nqptg = 8;  // queries per threadgroup    !! sync with kernel template arguments !!
        const int64_t ncpsg = 64; // cache values per simdgroup !! sync with kernel template arguments !!

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg  % 8  == 0);
        GGML_ASSERT(ncpsg  % 32 == 0);

        const int is_q = ggml_is_quantized(op->src[1]->type) ? 1 : 0;

        // 2*(2*ncpsg)
        // ncpsg soft_max values + ncpsg mask values
        //
        // 16*32*(nsg)
        // the shared memory needed for the simdgroups to load the KV cache
        // each thread loads (dequantizes) 16 head elements, there are 32 threads in th SG
        //
#define FATTN_SMEM(nsg) (GGML_PAD((nqptg*(ne00 + 2*GGML_PAD(ne20, 64) + 2*(2*ncpsg)) + is_q*(16*32*(nsg)))*(sizeof(float)/2), 16))

        //int64_t nsgmax = 4;
        //
        //if (is_q) {
        //    nsgmax = 2;
        //    while (true) {
        //        const size_t smem = FATTN_SMEM(nsgmax);
        //        if (smem > props_dev->max_theadgroup_memory_size) {
        //            break;
        //        }
        //        nsgmax *= 2;
        //    }
        //    nsgmax /= 2;
        //}

        // simdgroups per threadgroup (a.k.a. warps)
        //nsg = ne01 <= nqptg ? MAX(4, MIN(nsgmax, MIN(ne11/ncpsg, (int64_t) pipeline.maxTotalThreadsPerThreadgroup/32))) : 4;
        int32_t nsg = 4;

        const size_t smem = FATTN_SMEM(nsg);

        ggml_metal_kargs_flash_attn_ext args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ int32_t(nb11/nb10),
            /*.nb11          =*/ nb11,
            /*.nb12          =*/ nb12,
            /*.nb13          =*/ nb13,
            /*.ns20          =*/ int32_t(nb21/nb20),
            /*.nb21          =*/ nb21,
            /*.nb22          =*/ nb22,
            /*.nb23          =*/ nb23,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_flash_attn_ext(lib, op, has_mask, has_sinks, has_bias, has_scap, nsg);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
        if (op->src[3]) {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[3]), 4);
        } else {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 4);
        }
        if (op->src[4]) {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[4]), 5);
        } else {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 5);
        }
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         6);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03, 32, nsg, 1);
#undef FATTN_SMEM
    } else {
        // half4x4 kernel
        const int64_t nqptg = 1;  // queries per threadgroup    !! sync with kernel template arguments !!
        const int64_t ncpsg = 32; // cache values per simdgroup !! sync with kernel template arguments !!
        const int64_t nkpsg = 1*ncpsg;

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg  % 1  == 0);
        GGML_ASSERT(ncpsg  % 32 == 0);

        // ne00 + 2*ncpsg*(nsg)
        // for each query, we load it as f16 in shared memory (ne00)
        // and store the soft_max values and the mask
        //
        // ne20*(nsg)
        // each simdgroup has a full f32 head vector in shared mem to accumulate results
        //
#define FATTN_SMEM(nsg) (GGML_PAD((nqptg*(GGML_PAD(ne00, 128) + 4*ncpsg*(nsg)) + 2*GGML_PAD(ne20, 128)*(nsg))*(sizeof(float)/2), 16))

        int64_t nsgmax = 2;
        while (true) {
            const size_t smem = FATTN_SMEM(nsgmax);
            // avoid using more than half of the threadgroup memory - can cause slow downs especially for large head sizes
            if (smem > props_dev->max_theadgroup_memory_size/2) {
                break;
            }
            nsgmax *= 2;
        }
        nsgmax /= 2;

        // simdgroups per threadgroup (a.k.a. warps)
        //const int64_t nsgt = MAX(2, MIN(nsgmax, MIN((ne11 + nkpsg - 1)/(nkpsg), (int64_t) pipeline.maxTotalThreadsPerThreadgroup/32)));
        const int64_t nsgt = MAX(2, MIN(nsgmax, MIN((ne11 + nkpsg - 1)/(nkpsg), (int64_t) 1024/32)));

        int64_t nsg = 1;
        while (nsg <= nsgt) {
            nsg *= 2;
        }
        nsg /= 2;

        // workgroups
        // each workgroup handles nsg*nkpsg cache values
        int32_t nwg = 1;
        if (false) {
            // for small KV caches, we could launch a single workgroup and write the results directly to dst/
            // however, this does not lead to significant improvement, so disabled
            nwg = 1;
            nsg = 4;
        } else {
            nwg = 32;
            nsg = 1;
            while (2*nwg*nsg*nkpsg < ne11 && nsg < 4) {
                nsg *= 2;
            }
        }

        ggml_metal_kargs_flash_attn_ext_vec args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ int32_t(nb11/nb10),
            /*.nb11          =*/ nb11,
            /*.nb12          =*/ nb12,
            /*.nb13          =*/ nb13,
            /*.ns20          =*/ int32_t(nb21/nb20),
            /*.nb21          =*/ nb21,
            /*.nb22          =*/ nb22,
            /*.nb23          =*/ nb23,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_vec(lib, op, has_mask, has_sinks, has_bias, has_scap, nsg, nwg);

        GGML_ASSERT(nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
        if (op->src[3]) {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[3]), 4);
        } else {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 4);
        }
        if (op->src[4]) {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[4]), 5);
        } else {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 5);
        }

        const size_t smem = FATTN_SMEM(nsg);

        //printf("smem: %zu, max: %zu, nsg = %d, nsgmax = %d\n", smem, props_dev->max_theadgroup_memory_size, (int) nsg, (int) nsgmax);
        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        if (nwg == 1) {
            // using 1 workgroup -> write the result directly into dst
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 6);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03*nwg, 32, nsg, 1);
        } else {
            // sanity checks
            GGML_ASSERT(ne01*ne02*ne03 == ne1*ne2*ne3);
            GGML_ASSERT((uint64_t)ne1*ne2*ne3 <= (1u << 31));

            ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);

            // write the results from each workgroup into a temp buffer
            ggml_metal_buffer_id bid_tmp = bid_dst;
            bid_tmp.offs += ggml_nbytes(op);
            ggml_metal_encoder_set_buffer(enc, bid_tmp, 6);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03*nwg, 32, nsg, 1);

            // sync the 2 kernels
            ggml_metal_op_concurrency_reset(ctx);

            // reduce the results from the workgroups
            {
                const int32_t nrows = ne1*ne2*ne3;

                ggml_metal_kargs_flash_attn_ext_vec_reduce args0 = {
                    nrows,
                };

                ggml_metal_pipeline_t pipeline0 =
                    ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, nwg, fairy2i_flash3);

                ggml_metal_encoder_set_pipeline(enc, pipeline0);
                ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

                ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, 32*nwg, 1, 1);
            }
        }
#undef FATTN_SMEM
    }

    return 1;
}

int ggml_metal_op_bin(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_tensor ** ops = ggml_graph_nodes(gf) + idx;

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const int idx_end = ctx->idx_end;

    const bool use_fusion = ctx->use_fusion;

    const int debug_fusion = ctx->debug_fusion;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[1]));

    bool bcast_row = false;

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.offs =*/ 0,
        /*.o1   =*/ { bid_src1.offs },
    };

    ggml_op fops[8];

    int n_fuse = 1;

    // c[0] = add(a,    b[0])
    // c[1] = add(c[0], b[1])
    // c[2] = add(c[1], b[2])
    // ...
    if (use_fusion) {
        fops[0] = GGML_OP_ADD;
        fops[1] = GGML_OP_ADD;
        fops[2] = GGML_OP_ADD;
        fops[3] = GGML_OP_ADD;
        fops[4] = GGML_OP_ADD;
        fops[5] = GGML_OP_ADD;
        fops[6] = GGML_OP_ADD;
        fops[7] = GGML_OP_ADD;

        // note: in metal, we sometimes encode the graph in parallel so we have to avoid fusing ops
        //       across splits. idx_end indicates the last node in the current split
        for (n_fuse = 0; n_fuse <= 6 && idx + n_fuse + 1 < idx_end; ++n_fuse) {
            if (!ggml_can_fuse(gf, idx + n_fuse, fops + n_fuse, 2)) {
                break;
            }

            if (ops[n_fuse] != ops[n_fuse + 1]->src[0]) {
                break;
            }

            // b[0] === b[1] === ...
            if (!ggml_are_same_layout(ops[n_fuse]->src[1], ops[n_fuse + 1]->src[1])) {
                break;
            }

            // only fuse ops if src1 is in the same Metal buffer
            ggml_metal_buffer_id bid_fuse = ggml_metal_get_buffer_id(ops[n_fuse + 1]->src[1]);
            if (bid_fuse.metal != bid_src1.metal) {
                break;
            }

            //ctx->fuse_cnt[ops[n_fuse + 1]->op]++;

            args.o1[n_fuse + 1] = bid_fuse.offs;
        }

        ++n_fuse;

        if (debug_fusion > 1 && n_fuse > 1) {
            GGML_LOG_DEBUG("%s: fuse: ADD x %d\n", __func__, n_fuse);
        }
    }

    // the offsets of src1 and all fused buffers are relative to the start of the src1 buffer
    bid_src1.offs = 0;

    ggml_metal_pipeline_t pipeline = nullptr;

    if (ggml_nelements(op->src[1]) == ne10 && ggml_is_contiguous(op->src[1]) && ne00 % 4 == 0 && ne10 % 4 == 0) {
        GGML_ASSERT(ggml_is_contiguous(op->src[0]));

        // src1 is a row
        GGML_ASSERT(ne11 == 1);

        pipeline = ggml_metal_library_get_pipeline_bin(lib, op->op, n_fuse, true);

        bcast_row = true;
    } else {
        pipeline = ggml_metal_library_get_pipeline_bin(lib, op->op, n_fuse, false);
    }

    if (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ops[n_fuse - 1]);

        for (int i = 1; i < n_fuse; ++i) {
            if (!ggml_metal_op_concurrency_check(ctx, ops[i])) {
                ggml_metal_op_concurrency_reset(ctx);

                break;
            }
        }
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

    if (bcast_row) {
        const int64_t n = ggml_nelements(op)/4;

        ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);
    } else {
        int nth = 32;

        while (16*nth < ne0 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nth *= 2;
        }

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    return n_fuse;
}

int ggml_metal_op_rms_norm(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const int idx_end = ctx->idx_end;

    const bool use_fusion = ctx->use_fusion;

    const int debug_fusion = ctx->debug_fusion;

    ggml_tensor ** ops = ggml_graph_nodes(gf) + idx;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_rms_norm args = {
        /*.ne00   =*/ ne00,
        /*.ne00_4 =*/ ne00/4,
        /*.nb1    =*/ nb1,
        /*.nb2    =*/ nb2,
        /*.nb3    =*/ nb3,
        /*.eps    =*/ eps,
        /*.nef1   =*/ { ne01 },
        /*.nef2   =*/ { ne02 },
        /*.nef3   =*/ { ne03 },
        /*.nbf1   =*/ { nb01 },
        /*.nbf2   =*/ { nb02 },
        /*.nbf3   =*/ { nb03 },
    };

    ggml_op fops[8];

    int n_fuse = 1;

    ggml_metal_buffer_id bid_fuse[2] = { bid_src0, bid_src0 };

    // d[0] = rms_norm(a)
    // d[1] = mul(d[0], b)
    // d[2] = add(d[1], c)
    if (use_fusion) {
        fops[0] = GGML_OP_RMS_NORM;
        fops[1] = GGML_OP_MUL;
        fops[2] = GGML_OP_ADD;

        for (n_fuse = 0; n_fuse <= 1 && idx + n_fuse + 1 < idx_end; ++n_fuse) {
            if (!ggml_can_fuse(gf, idx + n_fuse, fops + n_fuse, 2)) {
                break;
            }

            if (ops[n_fuse] != ops[n_fuse + 1]->src[0]) {
                break;
            }

            if (ops[n_fuse + 1]->src[1]->ne[0] != op->ne[0]) {
                break;
            }

            if (!ggml_is_contiguous_rows(ops[n_fuse + 1]->src[1])) {
                break;
            }

            if (ops[n_fuse + 1]->type != GGML_TYPE_F32) {
                break;
            }

            //ctx->fuse_cnt[ops[n_fuse + 1]->op]++;

            bid_fuse[n_fuse] = ggml_metal_get_buffer_id(ops[n_fuse + 1]->src[1]);

            args.nef1[n_fuse + 1] = ops[n_fuse + 1]->src[1]->ne[1];
            args.nef2[n_fuse + 1] = ops[n_fuse + 1]->src[1]->ne[2];
            args.nef3[n_fuse + 1] = ops[n_fuse + 1]->src[1]->ne[3];

            args.nbf1[n_fuse + 1] = ops[n_fuse + 1]->src[1]->nb[1];
            args.nbf2[n_fuse + 1] = ops[n_fuse + 1]->src[1]->nb[2];
            args.nbf3[n_fuse + 1] = ops[n_fuse + 1]->src[1]->nb[3];
        }

        ++n_fuse;

        if (debug_fusion > 1 && n_fuse > 1) {
            if (n_fuse == 2) {
                GGML_LOG_DEBUG("%s: fuse: RMS_NORM + MUL\n", __func__);
            }
            if (n_fuse == 3) {
                GGML_LOG_DEBUG("%s: fuse: RMS_NORM + MUL + ADD\n", __func__);
            }
        }
    }

    if (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ops[n_fuse - 1]);

        for (int i = 1; i < n_fuse; ++i) {
            if (!ggml_metal_op_concurrency_check(ctx, ops[i])) {
                ggml_metal_op_concurrency_reset(ctx);

                break;
            }
        }
    }

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_rms_norm(lib, op, n_fuse);

    int nth = 32; // SIMD width

    while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, ne00/4);

    const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_fuse[0], 2);
    ggml_metal_encoder_set_buffer  (enc, bid_fuse[1], 3);
    ggml_metal_encoder_set_buffer  (enc, bid_dst, 4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return n_fuse;
}

int ggml_metal_op_fairy2i_rms_norm_exact(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ggml_graph_node(ctx->gf, idx);

    GGML_ASSERT(op->op == GGML_OP_FAIRY2I_RMS_NORM_EXACT);
    GGML_ASSERT(op->src[0] && op->src[1]);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(op->src[0], op));
    GGML_ASSERT(ggml_can_repeat(op->src[1], op->src[0]));
    GGML_ASSERT(op->src[0]->nb[0] == sizeof(float));
    GGML_ASSERT(op->src[1]->nb[0] == sizeof(float));
    GGML_ASSERT(op->nb[0] == sizeof(float));

    float eps;
    memcpy(&eps, op->op_params, sizeof(eps));
    GGML_ASSERT(eps >= 0.0f);

    const ggml_tensor * src0   = op->src[0];
    const ggml_tensor * weight = op->src[1];

    ggml_metal_kargs_fairy2i_rms_norm_exact args = {
        /*.ne00 =*/(int32_t) src0->ne[0],
        /*.ne01 =*/(int32_t) src0->ne[1],
        /*.ne02 =*/(int32_t) src0->ne[2],
        /*.ne03 =*/(int32_t) src0->ne[3],
        /*.nb01 =*/src0->nb[1],
        /*.nb02 =*/src0->nb[2],
        /*.nb03 =*/src0->nb[3],
        /*.ne10 =*/(int32_t) weight->ne[0],
        /*.ne11 =*/(int32_t) weight->ne[1],
        /*.ne12 =*/(int32_t) weight->ne[2],
        /*.ne13 =*/(int32_t) weight->ne[3],
        /*.nb10 =*/weight->nb[0],
        /*.nb11 =*/weight->nb[1],
        /*.nb12 =*/weight->nb[2],
        /*.nb13 =*/weight->nb[3],
        /*.nb1  =*/op->nb[1],
        /*.nb2  =*/op->nb[2],
        /*.nb3  =*/op->nb[3],
        /*.eps  =*/eps,
    };

    const bool qat = ggml_fairy2i_exact_get_qat(op);

    ggml_metal_pipeline_t pipeline = ggml_metal_get_pipeline_fairy2i_rms_norm_exact(ctx->lib, qat);

    ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
    ggml_metal_encoder_set_bytes(ctx->enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(src0), 1);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(weight), 2);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(op), 3);
    ggml_metal_encoder_set_threadgroup_memory_size(ctx->enc, ggml_metal_pipeline_get_smem(pipeline), 0);
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc, src0->ne[1], src0->ne[2], src0->ne[3], qat ? 256 : 32, 1, 1);

    return 1;
}

int ggml_metal_op_fairy2i_elementwise_exact(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ggml_graph_node(ctx->gf, idx);

    GGML_ASSERT(op->op == GGML_OP_FAIRY2I_SILU_EXACT || op->op == GGML_OP_FAIRY2I_MUL_EXACT ||
                op->op == GGML_OP_FAIRY2I_PACK_BF16_EXACT);
    GGML_ASSERT(op->src[0] && op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(op->src[0], op));
    GGML_ASSERT(ggml_is_contiguous(op));

    // The Row4 gate/up projection is stored as [gate, up] in one F32 BF16-carrier
    // tensor. DFS graph order places the metadata-only up VIEW between SiLU and
    // MUL, so the generic adjacent-op fusion helper cannot recognize this chain:
    //
    //   SILU_EXACT(gate-view), VIEW(up), MUL_EXACT(silu, up-view)
    //
    // Preserve the QAT BF16 boundary inside the fused kernel and consume all
    // three graph nodes. When the sole consumer is a supported Row4 down
    // projection, hand its BF16 payloads directly to activation quantization
    // and consume that fourth node as well. Keep this deliberately specific to
    // the Row4 gate/up layout so all other exact operations retain their old
    // path.
    if (ctx->use_fusion && op->op == GGML_OP_FAIRY2I_SILU_EXACT && idx + 2 < ctx->idx_end &&
        ggml_fairy2i_exact_get_qat(op) && ggml_node_has_n_uses(ctx->gf, idx, 1)) {
        ggml_tensor * up   = ggml_graph_node(ctx->gf, idx + 1);
        ggml_tensor * mul  = ggml_graph_node(ctx->gf, idx + 2);
        ggml_tensor * gate = op->src[0];

        const bool row4_gate_up = up->op == GGML_OP_VIEW && up->view_src && gate->op == GGML_OP_VIEW &&
                                  gate->view_src && gate->view_src == up->view_src &&
                                  gate->view_src->op == GGML_OP_ROW4_LINEAR && gate->view_offs == 0 &&
                                  up->view_offs == (size_t) gate->ne[0] * sizeof(float) &&
                                  gate->view_src->ne[0] == 2 * gate->ne[0] && gate->view_src->ne[1] == gate->ne[1] &&
                                  gate->view_src->ne[2] == gate->ne[2] && gate->view_src->ne[3] == gate->ne[3];
        const bool exact_mul    = mul->op == GGML_OP_FAIRY2I_MUL_EXACT && mul->src[0] == op && mul->src[1] == up &&
                                  ggml_fairy2i_exact_get_qat(mul);
        const bool f32_layout = gate->type == GGML_TYPE_F32 && up->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                                mul->type == GGML_TYPE_F32 && gate->nb[0] == sizeof(float) &&
                                up->nb[0] == sizeof(float) && ggml_are_same_shape(gate, up) &&
                                ggml_are_same_shape(gate, op) && ggml_are_same_shape(gate, mul) &&
                                ggml_is_contiguous_1(gate) && ggml_is_contiguous_1(up) && ggml_is_contiguous(mul);

        if (row4_gate_up && exact_mul && f32_layout) {
            ggml_tensor *  down       = idx + 3 < ctx->idx_end ? ggml_graph_node(ctx->gf, idx + 3) : nullptr;
            const bool     fused_down = down && down->op == GGML_OP_ROW4_LINEAR && down->src[0] == mul &&
                                        ggml_node_has_n_uses(ctx->gf, idx + 2, 1) &&
                                        ggml_metal_device_supports_op(ctx->dev, mul) &&
                                        ggml_metal_device_supports_op(ctx->dev, down);
            const uint64_t ne         = (uint64_t) ggml_nelements(mul);
            const uint64_t ne0        = (uint64_t) mul->ne[0];
            const uint64_t rows       = ne / ne0;
            ggml_metal_kargs_fairy2i_elementwise_exact args = {
                /*.ne       =*/ne,
                /*.ne0      =*/ne0,
                /*.src0_nb1 =*/gate->nb[1],
                /*.src1_nb1 =*/up->nb[1],
            };

            ggml_metal_pipeline_t pipeline = ggml_metal_get_pipeline_fairy2i_swiglu_qat(ctx->lib, fused_down);
            const int             nth      = std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

            ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
            ggml_metal_encoder_set_bytes(ctx->enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(gate), 1);
            ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(up), 2);
            ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(mul), 3);
            ggml_metal_encoder_dispatch_threadgroups(ctx->enc, (ne0 + nth - 1) / nth, rows, 1, nth, 1, 1);

            if (fused_down) {
                // The producer writes tightly packed BF16 payloads into the
                // otherwise-dead MUL allocation. Publish them before the down
                // projection widens and quantizes those exact payloads.
                ggml_metal_encoder_memory_barrier(ctx->enc);
                ggml_metal_op_row_quant_linear_impl(ctx, down, ggml_metal_get_buffer_id(mul), true);

                if (ctx->debug_fusion > 1) {
                    GGML_LOG_DEBUG(
                        "%s: fuse Row4 QAT SILU_EXACT + VIEW + MUL_EXACT + ROW4_LINEAR (packed BF16 handoff)\n",
                        __func__);
                }
                return 4;
            }

            if (ctx->debug_fusion > 1) {
                GGML_LOG_DEBUG("%s: fuse Row4 QAT SILU_EXACT + VIEW + MUL_EXACT\n", __func__);
            }
            return 3;
        }
    }

    if (op->op == GGML_OP_FAIRY2I_MUL_EXACT) {
        GGML_ASSERT(op->src[1] && op->src[1]->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_are_same_shape(op->src[0], op->src[1]));
        GGML_ASSERT(op->src[0]->nb[0] == sizeof(float) && op->src[1]->nb[0] == sizeof(float));
        GGML_ASSERT(ggml_is_contiguous_1(op->src[0]) && ggml_is_contiguous_1(op->src[1]));
        GGML_ASSERT(op->type == GGML_TYPE_F32);
    } else if (op->op == GGML_OP_FAIRY2I_PACK_BF16_EXACT) {
        GGML_ASSERT(ggml_is_contiguous(op->src[0]));
        GGML_ASSERT(op->type == GGML_TYPE_BF16 || op->type == GGML_TYPE_F32);
    } else {
        GGML_ASSERT(op->src[0]->nb[0] == sizeof(float) && ggml_is_contiguous_1(op->src[0]));
        GGML_ASSERT(op->type == GGML_TYPE_F32);
    }

    uint64_t ne = (uint64_t) ggml_nelements(op);

    const bool qat =
        op->op == GGML_OP_FAIRY2I_PACK_BF16_EXACT ? op->type == GGML_TYPE_F32 : ggml_fairy2i_exact_get_qat(op);
    ggml_metal_pipeline_t pipeline = ggml_metal_get_pipeline_fairy2i_elementwise_exact(ctx->lib, op->op, qat);

    ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
    if (op->op == GGML_OP_FAIRY2I_PACK_BF16_EXACT) {
        ggml_metal_encoder_set_bytes(ctx->enc, &ne, sizeof(ne), 0);
    } else {
        ggml_metal_kargs_fairy2i_elementwise_exact args = {
            /*.ne       =*/ne,
            /*.ne0      =*/(uint64_t) op->ne[0],
            /*.src0_nb1 =*/op->src[0]->nb[1],
            /*.src1_nb1 =*/op->op == GGML_OP_FAIRY2I_MUL_EXACT ? op->src[1]->nb[1] : 0,
        };
        ggml_metal_encoder_set_bytes(ctx->enc, &args, sizeof(args), 0);
    }
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->op == GGML_OP_FAIRY2I_MUL_EXACT) {
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(op), 3);
    } else {
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(op), 2);
    }

    const int nth = std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc, (ne + nth - 1) / nth, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_l2_norm(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    int nth = 32; // SIMD width

    ggml_metal_kargs_l2_norm args = {
        /*.ne00   =*/ ne00,
        /*.ne00_4 =*/ ne00/4,
        /*.nb01   =*/ nb01,
        /*.eps    =*/ eps,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_l2_norm(lib, op);

    while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, ne00/4);

    const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

    const int64_t nrows = ggml_nrows(op->src[0]);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_group_norm(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    const int32_t ngrp = ((const int32_t *) op->op_params)[0];

    float eps;
    memcpy(&eps, op->op_params + 1, sizeof(float));

    ggml_metal_kargs_group_norm args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ngrp =*/ ngrp,
        /*.eps  =*/ eps,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_group_norm(lib, op);

    int nth = 32; // SIMD width
    //while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
    //    nth *= 2;
    //}

    //nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    //nth = std::min(nth, ne00/4);

    const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ngrp, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_norm(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_kargs_norm args = {
        /*.ne00   =*/ ne00,
        /*.ne00_4 =*/ ne00/4,
        /*.nb01   =*/ nb01,
        /*.eps    =*/ eps,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_norm(lib, op);

    int nth = 32; // SIMD width
    while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, ne00/4);

    const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

    const int64_t nrows = ggml_nrows(op->src[0]);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_rope(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_ASSERT(op->op == GGML_OP_ROPE || op->op == GGML_OP_FAIRY2I_ROPE_EXACT);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    // make sure we have one or more position id(ne10) per token(ne02)
    GGML_ASSERT(ne10 % ne02 == 0);
    GGML_ASSERT(ne10 >= ne02);

    const int n_past     = ((const int32_t *) op->op_params)[0];
    const int n_dims     = ((const int32_t *) op->op_params)[1];
  //const int mode       = ((const int32_t *) op->op_params)[2];
    // skip 3, n_ctx, used in GLM RoPE, unimplemented in metal
    const int n_ctx_orig = ((const int32_t *) op->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) op->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) op->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) op->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) op->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) op->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) op->op_params + 10, sizeof(float));

    // mrope
    const int sect_0 = ((const int32_t *) op->op_params)[11];
    const int sect_1 = ((const int32_t *) op->op_params)[12];
    const int sect_2 = ((const int32_t *) op->op_params)[13];
    const int sect_3 = ((const int32_t *) op->op_params)[14];

    ggml_metal_kargs_rope args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.ne03        =*/ ne03,
        /*.nb00        =*/ nb00,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne0         =*/ ne0,
        /*.ne1         =*/ ne1,
        /*.ne2         =*/ ne2,
        /*.ne3         =*/ ne3,
        /*.nb0         =*/ nb0,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.n_past      =*/ n_past,
        /*.n_dims      =*/ n_dims,
        /*.n_ctx_orig  =*/ n_ctx_orig,
        /*.freq_base   =*/ freq_base,
        /*.freq_scale  =*/ freq_scale,
        /*.ext_factor  =*/ ext_factor,
        /*.attn_factor =*/ attn_factor,
        /*.beta_fast   =*/ beta_fast,
        /*.beta_slow   =*/ beta_slow,
        /* sect_0      =*/ sect_0,
        /* sect_1      =*/ sect_1,
        /* sect_2      =*/ sect_2,
        /* sect_3      =*/ sect_3,
    };

    const bool exact = op->op == GGML_OP_FAIRY2I_ROPE_EXACT;
    const bool qat   = exact && ggml_fairy2i_exact_get_qat(op);
    if (exact) {
        const int mode = ((const int32_t *) op->op_params)[2];
        GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);
        GGML_ASSERT((mode & GGML_ROPE_TYPE_NEOX) != 0);
        GGML_ASSERT((mode & GGML_ROPE_TYPE_MROPE) == 0);
        GGML_ASSERT(mode != GGML_ROPE_TYPE_VISION);
        GGML_ASSERT(n_dims > 0 && n_dims <= ne00 && n_dims % 2 == 0);
    }

    ggml_metal_pipeline_t pipeline =
        exact ? ggml_metal_get_pipeline_fairy2i_rope_exact(lib, qat) : ggml_metal_library_get_pipeline_rope(lib, op);
    const int nth = exact && !qat ? (int) std::min<int64_t>(1024, (int64_t) ne00 * ne01) : std::min(1024, ne00);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    if (exact && !qat) {
        const size_t smem = GGML_PAD((size_t) n_dims * sizeof(float), 16);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
    }

    ggml_metal_encoder_dispatch_threadgroups(enc, exact && !qat ? 1 : ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_im2col(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t s1 = ((const int32_t *)(op->op_params))[1];
    const int32_t p0 = ((const int32_t *)(op->op_params))[2];
    const int32_t p1 = ((const int32_t *)(op->op_params))[3];
    const int32_t d0 = ((const int32_t *)(op->op_params))[4];
    const int32_t d1 = ((const int32_t *)(op->op_params))[5];

    const bool is_2D = ((const int32_t *)(op->op_params))[6] == 1;

    const int32_t N  = op->src[1]->ne[is_2D ? 3 : 2];
    const int32_t IC = op->src[1]->ne[is_2D ? 2 : 1];
    const int32_t IH = is_2D ? op->src[1]->ne[1] : 1;
    const int32_t IW =         op->src[1]->ne[0];

    const int32_t KH = is_2D ? op->src[0]->ne[1] : 1;
    const int32_t KW =         op->src[0]->ne[0];

    const int32_t OH = is_2D ? op->ne[2] : 1;
    const int32_t OW =         op->ne[1];

    const int32_t CHW = IC * KH * KW;

    const uint64_t ofs0 = op->src[1]->nb[is_2D ? 3 : 2] / 4;
    const uint64_t ofs1 = op->src[1]->nb[is_2D ? 2 : 1] / 4;


    ggml_metal_kargs_im2col args = {
        /*.ofs0 =*/ ofs0,
        /*.ofs1 =*/ ofs1,
        /*.IW   =*/ IW,
        /*.IH   =*/ IH,
        /*.CHW  =*/ CHW,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
        /*.N    =*/ N,
        /*.KH   =*/ KH,
        /*.KW   =*/ KW,
        /*.KHW  =*/ KH * KW,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_im2col(lib, op);

    const uint64_t n_threads = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), N);
    const int64_t  quotient  = N / n_threads + (N % n_threads > 0 ? 1 : 0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, quotient * CHW, OH, OW, n_threads, 1, 1);

    return 1;
}

int ggml_metal_op_conv_transpose_1d(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];

    const int32_t IC = op->src[1]->ne[1];
    const int32_t IL = op->src[1]->ne[0];

    const int32_t K  = op->src[0]->ne[0];

    const int32_t OL = op->ne[0];
    const int32_t OC = op->ne[1];

    ggml_metal_kargs_conv_transpose_1d args = {
        /*.IC  =*/ IC,
        /*.IL  =*/ IL,
        /*.K   =*/ K,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_conv_transpose_1d(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, OL, OC, 1, 1, 1, 1);

    return 1;
}

int ggml_metal_op_upscale(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    const float sf0 = (float)ne0/op->src[0]->ne[0];
    const float sf1 = (float)ne1/op->src[0]->ne[1];
    const float sf2 = (float)ne2/op->src[0]->ne[2];
    const float sf3 = (float)ne3/op->src[0]->ne[3];

    ggml_metal_kargs_upscale args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0 =*/ ne0,
        /*.ne1 =*/ ne1,
        /*.ne2 =*/ ne2,
        /*.ne3 =*/ ne3,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
        /*.nb2 =*/ nb2,
        /*.nb3 =*/ nb3,
        /*.sf0 =*/ sf0,
        /*.sf1 =*/ sf1,
        /*.sf2 =*/ sf2,
        /*.sf3 =*/ sf3
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_upscale(lib, op);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    ggml_metal_kargs_pad args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_pad(lib, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad_reflect_1d(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    ggml_metal_kargs_pad_reflect_1d args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.p0 =*/ ((const int32_t *)(op->op_params))[0],
        /*.p1 =*/ ((const int32_t *)(op->op_params))[1]
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_pad_reflect_1d(lib, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_arange(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    float start;
    float step;

    memcpy(&start, ((const int32_t *) op->op_params) + 0, sizeof(float));
    memcpy(&step,  ((const int32_t *) op->op_params) + 2, sizeof(float));

    ggml_metal_kargs_arange args = {
        /*.ne0   =*/ ne0,
        /*.start =*/ start,
        /*.step  =*/ step
    };

    const int nth = std::min(1024, ne0);

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_arange(lib, op);

    //[encoder setComputePipelineState:pipeline];
    //[encoder setBuffer:id_dst  offset:offs_dst  atIndex:0];
    //[encoder setBytes:&args length:sizeof(args) atIndex:1];

    //[encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 1);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_timestep_embedding(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    const int dim        = op->op_params[0];
    const int max_period = op->op_params[1];

    ggml_metal_kargs_timestep_embedding args = {
        /*.nb1 =*/ nb1,
        /*.dim =*/ dim,
        /*.max_period =*/ max_period,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_timestep_embedding(lib, op);

    const int nth = std::max(1, std::min(1024, dim/2));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne00, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_argmax(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    ggml_metal_kargs_argmax args = {
        /*.ne00 = */ ne00,
        /*.nb01 = */ nb01,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_argmax(lib, op);

    const int64_t nrows = ggml_nrows(op->src[0]);

    int nth = 32; // SIMD width
    while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
        nth *= 2;
    }

    const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_argsort(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    // bitonic sort requires the number of elements to be power of 2
    int64_t ne00_padded = 1;
    while (ne00_padded < ne00) {
        ne00_padded *= 2;
    }

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_argsort(lib, op);

    const int64_t nrows = ggml_nrows(op->src[0]);

    // Metal kernels require the buffer size to be multiple of 16 bytes
    // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength
    const size_t smem = GGML_PAD(ne00_padded*sizeof(int32_t), 16);

    ggml_metal_kargs_argsort args = {
        /*.ncols =*/ ne00,
        /*.ncols_pad =*/ ne00_padded
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, nrows, 1, ne00_padded, 1, 1);

    return 1;
}

int ggml_metal_op_leaky_relu(ggml_metal_op_t ctx, int idx) {
    ggml_cgraph * gf = ctx->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    float slope;
    memcpy(&slope, op->op_params, sizeof(float));

    ggml_metal_kargs_leaky_relu args = {
        /*.slope =*/ slope
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_library_get_pipeline_unary(lib, op);

    int64_t n = ggml_nelements(op);

    if (n % 4 == 0) {
        n /= 4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);

    return 1;
}
