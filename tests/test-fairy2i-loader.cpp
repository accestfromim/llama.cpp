// Fairy2i GGUF loader schema tests.

#include "../src/llama-model.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

enum class fairy2i_loader_case {
    valid_w1,
    mixed_w1_w2,
    incomplete_w1,
    valid_bundle_w1,
    valid_bundle_w1_qkv,
    valid_bundle_w1_qat,
    valid_bundle_w1_qat_qkv_bias,
    valid_bundle_w2,
    valid_bundle_w2_legacy_explicit,
    valid_bundle_w2_exact,
    valid_bundle_w2_exact_gqa,
    incomplete_bundle,
    mixed_bundle,
    invalid_bundle_shape,
    invalid_bundle_alignment,
    invalid_bundle_branch_order,
    invalid_bundle_scales_f32,
    invalid_bundle_scales_bf16,
    invalid_bundle_scales_shape,
    invalid_bundle_codes_type,
    invalid_bundle_codes_branches,
    invalid_bundle_schema,
    invalid_bundle_scale_scope,
    invalid_bundle_code_order,
    invalid_bundle_variant,
    invalid_exact_missing_scale_dtype,
    invalid_exact_missing_numeric_profile,
    invalid_exact_scale_dtype,
    invalid_exact_numeric_profile,
    invalid_exact_base_arch_llama,
    invalid_exact_missing_q_bias,
    invalid_exact_q_bias_f16,
    invalid_exact_token_embd_f16,
    invalid_exact_output_norm_f16,
    invalid_exact_attn_norm_f16,
    invalid_exact_ffn_norm_f16,
    invalid_exact_scales_f16,
    invalid_exact_file_type,
    invalid_exact_quant_format,
    invalid_exact_codebook,
    invalid_exact_obsolete_f32_contract,
    invalid_exact_legacy_contract,
    invalid_legacy_exact_profile,
    invalid_legacy_exact_contract,
    invalid_legacy_missing_numeric_contract,
    invalid_legacy_missing_scale_dtype,
    invalid_legacy_missing_numeric_profile,
    invalid_exact_w1,
    invalid_exact_attn_layout_llama,
    invalid_exact_attn_layout_qwen3,
    invalid_exact_missing_attn_layout,
    invalid_exact_tokenizer_profile,
    invalid_exact_missing_tokenizer_profile,
    invalid_exact_output_bias_f32,
    invalid_exact_output_bias_f16,
    invalid_exact_dense_f16,
    invalid_exact_dense_only,
    invalid_qat_schema,
    invalid_qat_base_arch,
    invalid_qat_attn_layout,
    invalid_qat_quant_variant,
    invalid_qat_scale_dtype,
    invalid_qat_numeric_profile,
    invalid_qat_scales_f16,
    invalid_qat_output_f16,
    invalid_qat_missing_output,
    invalid_qat_q_norm_f16,
    invalid_qat_missing_q_norm,
    invalid_qat_qkv_bias_f16,
};

static bool is_bundle_case(fairy2i_loader_case tc) {
    return tc != fairy2i_loader_case::valid_w1 && tc != fairy2i_loader_case::mixed_w1_w2 &&
           tc != fairy2i_loader_case::incomplete_w1;
}

static bool is_w1_case(fairy2i_loader_case tc) {
    switch (tc) {
        case fairy2i_loader_case::valid_bundle_w2:
        case fairy2i_loader_case::valid_bundle_w2_legacy_explicit:
        case fairy2i_loader_case::valid_bundle_w2_exact:
        case fairy2i_loader_case::valid_bundle_w2_exact_gqa:
        case fairy2i_loader_case::mixed_bundle:
        case fairy2i_loader_case::invalid_bundle_scales_f32:
        case fairy2i_loader_case::invalid_bundle_scales_bf16:
        case fairy2i_loader_case::invalid_bundle_scales_shape:
        case fairy2i_loader_case::invalid_bundle_codes_type:
        case fairy2i_loader_case::invalid_bundle_codes_branches:
        case fairy2i_loader_case::invalid_bundle_schema:
        case fairy2i_loader_case::invalid_bundle_scale_scope:
        case fairy2i_loader_case::invalid_bundle_code_order:
        case fairy2i_loader_case::invalid_bundle_variant:
        case fairy2i_loader_case::invalid_exact_missing_scale_dtype:
        case fairy2i_loader_case::invalid_exact_missing_numeric_profile:
        case fairy2i_loader_case::invalid_exact_scale_dtype:
        case fairy2i_loader_case::invalid_exact_numeric_profile:
        case fairy2i_loader_case::invalid_exact_base_arch_llama:
        case fairy2i_loader_case::invalid_exact_missing_q_bias:
        case fairy2i_loader_case::invalid_exact_q_bias_f16:
        case fairy2i_loader_case::invalid_exact_token_embd_f16:
        case fairy2i_loader_case::invalid_exact_output_norm_f16:
        case fairy2i_loader_case::invalid_exact_attn_norm_f16:
        case fairy2i_loader_case::invalid_exact_ffn_norm_f16:
        case fairy2i_loader_case::invalid_exact_scales_f16:
        case fairy2i_loader_case::invalid_exact_file_type:
        case fairy2i_loader_case::invalid_exact_quant_format:
        case fairy2i_loader_case::invalid_exact_codebook:
        case fairy2i_loader_case::invalid_exact_obsolete_f32_contract:
        case fairy2i_loader_case::invalid_exact_legacy_contract:
        case fairy2i_loader_case::invalid_legacy_exact_profile:
        case fairy2i_loader_case::invalid_legacy_exact_contract:
        case fairy2i_loader_case::invalid_legacy_missing_numeric_contract:
        case fairy2i_loader_case::invalid_legacy_missing_scale_dtype:
        case fairy2i_loader_case::invalid_legacy_missing_numeric_profile:
        case fairy2i_loader_case::invalid_exact_attn_layout_llama:
        case fairy2i_loader_case::invalid_exact_attn_layout_qwen3:
        case fairy2i_loader_case::invalid_exact_missing_attn_layout:
        case fairy2i_loader_case::invalid_exact_tokenizer_profile:
        case fairy2i_loader_case::invalid_exact_missing_tokenizer_profile:
        case fairy2i_loader_case::invalid_exact_output_bias_f32:
        case fairy2i_loader_case::invalid_exact_output_bias_f16:
        case fairy2i_loader_case::invalid_exact_dense_f16:
        case fairy2i_loader_case::invalid_exact_dense_only:
            return false;
        default:
            return true;
    }
}

static bool is_exact_case(fairy2i_loader_case tc) {
    switch (tc) {
        case fairy2i_loader_case::valid_bundle_w2_exact:
        case fairy2i_loader_case::valid_bundle_w2_exact_gqa:
        case fairy2i_loader_case::invalid_exact_missing_scale_dtype:
        case fairy2i_loader_case::invalid_exact_missing_numeric_profile:
        case fairy2i_loader_case::invalid_exact_scale_dtype:
        case fairy2i_loader_case::invalid_exact_numeric_profile:
        case fairy2i_loader_case::invalid_exact_base_arch_llama:
        case fairy2i_loader_case::invalid_exact_missing_q_bias:
        case fairy2i_loader_case::invalid_exact_q_bias_f16:
        case fairy2i_loader_case::invalid_exact_token_embd_f16:
        case fairy2i_loader_case::invalid_exact_output_norm_f16:
        case fairy2i_loader_case::invalid_exact_attn_norm_f16:
        case fairy2i_loader_case::invalid_exact_ffn_norm_f16:
        case fairy2i_loader_case::invalid_exact_scales_f16:
        case fairy2i_loader_case::invalid_exact_file_type:
        case fairy2i_loader_case::invalid_exact_quant_format:
        case fairy2i_loader_case::invalid_exact_codebook:
        case fairy2i_loader_case::invalid_exact_obsolete_f32_contract:
        case fairy2i_loader_case::invalid_exact_legacy_contract:
        case fairy2i_loader_case::invalid_exact_w1:
        case fairy2i_loader_case::invalid_exact_attn_layout_llama:
        case fairy2i_loader_case::invalid_exact_attn_layout_qwen3:
        case fairy2i_loader_case::invalid_exact_missing_attn_layout:
        case fairy2i_loader_case::invalid_exact_tokenizer_profile:
        case fairy2i_loader_case::invalid_exact_missing_tokenizer_profile:
        case fairy2i_loader_case::invalid_exact_output_bias_f32:
        case fairy2i_loader_case::invalid_exact_output_bias_f16:
        case fairy2i_loader_case::invalid_exact_dense_f16:
        case fairy2i_loader_case::invalid_exact_dense_only:
            return true;
        default:
            return false;
    }
}

static bool is_qat_case(fairy2i_loader_case tc) {
    switch (tc) {
        case fairy2i_loader_case::valid_bundle_w1_qat:
        case fairy2i_loader_case::valid_bundle_w1_qat_qkv_bias:
        case fairy2i_loader_case::invalid_qat_schema:
        case fairy2i_loader_case::invalid_qat_base_arch:
        case fairy2i_loader_case::invalid_qat_attn_layout:
        case fairy2i_loader_case::invalid_qat_quant_variant:
        case fairy2i_loader_case::invalid_qat_scale_dtype:
        case fairy2i_loader_case::invalid_qat_numeric_profile:
        case fairy2i_loader_case::invalid_qat_scales_f16:
        case fairy2i_loader_case::invalid_qat_output_f16:
        case fairy2i_loader_case::invalid_qat_missing_output:
        case fairy2i_loader_case::invalid_qat_q_norm_f16:
        case fairy2i_loader_case::invalid_qat_missing_q_norm:
        case fairy2i_loader_case::invalid_qat_qkv_bias_f16:
            return true;
        default:
            return false;
    }
}

static bool is_qwen2_legacy_case(fairy2i_loader_case tc) {
    switch (tc) {
        case fairy2i_loader_case::valid_bundle_w2_legacy_explicit:
        case fairy2i_loader_case::invalid_legacy_exact_profile:
        case fairy2i_loader_case::invalid_legacy_exact_contract:
        case fairy2i_loader_case::invalid_legacy_missing_numeric_contract:
        case fairy2i_loader_case::invalid_legacy_missing_scale_dtype:
        case fairy2i_loader_case::invalid_legacy_missing_numeric_profile:
            return true;
        default:
            return false;
    }
}

static std::string make_tmp_path(const char * label) {
    std::string       pattern = std::string("/tmp/llama-fairy2i-loader-") + label + "-XXXXXX.gguf";
    std::vector<char> path(pattern.begin(), pattern.end());
    path.push_back('\0');

    const int fd = mkstemps(path.data(), 5);
    if (fd < 0) {
        fprintf(stderr, "mkstemps failed for %s\n", pattern.c_str());
        exit(EXIT_FAILURE);
    }
    close(fd);
    unlink(path.data());

    return std::string(path.data());
}

struct tiny_fairy2i_writer {
    gguf_context *                    gguf = nullptr;
    ggml_context *                    ggml = nullptr;
    std::vector<std::vector<uint8_t>> tensor_data;

    tiny_fairy2i_writer() {
        gguf = gguf_init_empty();

        ggml_init_params params = {
            /* .mem_size   = */ 64 * ggml_tensor_overhead(),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml = ggml_init(params);
        tensor_data.reserve(32);
    }

    ~tiny_fairy2i_writer() {
        ggml_free(ggml);
        gguf_free(gguf);
    }

    void add_tensor(const char * name, ggml_type type, int64_t ne0, int64_t ne1) {
        ggml_tensor * tensor = ggml_new_tensor_2d(ggml, type, ne0, ne1);
        ggml_set_name(tensor, name);
        gguf_add_tensor(gguf, tensor);

        tensor_data.emplace_back(ggml_nbytes(tensor), uint8_t{ 0 });
        gguf_set_tensor_data(gguf, name, tensor_data.back().data());
    }

    void add_bundle_tensor(const char * name, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        const int64_t ne[4]  = { ne0, ne1, ne2, ne3 };
        ggml_tensor * tensor = ggml_new_tensor(ggml, type, 4, ne);
        ggml_set_name(tensor, name);
        gguf_add_tensor(gguf, tensor);

        tensor_data.emplace_back(ggml_nbytes(tensor), uint8_t{ 0 });
        gguf_set_tensor_data(gguf, name, tensor_data.back().data());
    }

    std::vector<uint8_t> & data(const char * name) {
        const int64_t tensor_id = gguf_find_tensor(gguf, name);
        if (tensor_id < 0 || (size_t) tensor_id >= tensor_data.size()) {
            fprintf(stderr, "missing tiny Fairy2i tensor data for %s\n", name);
            exit(EXIT_FAILURE);
        }
        return tensor_data[(size_t) tensor_id];
    }
};

static uint32_t pack_bf16_pair(float real, float imag) {
    const ggml_bf16_t pair[2] = { ggml_fp32_to_bf16(real), ggml_fp32_to_bf16(imag) };
    uint32_t          packed;
    memcpy(&packed, pair, sizeof(packed));
    return packed;
}

static void set_f32_data(tiny_fairy2i_writer & writer, const char * name, const std::vector<float> & values) {
    std::vector<uint8_t> & data = writer.data(name);
    if (data.size() != values.size() * sizeof(float)) {
        fprintf(stderr, "invalid tiny Fairy2i F32 fixture size for %s: got %zu expected %zu\n", name, values.size(),
                data.size() / sizeof(float));
        exit(EXIT_FAILURE);
    }
    memcpy(data.data(), values.data(), data.size());
    gguf_set_tensor_data(writer.gguf, name, data.data());
}

static void set_bf16_data(tiny_fairy2i_writer & writer, const char * name, const std::vector<float> & values) {
    std::vector<uint8_t> & data = writer.data(name);
    if (data.size() != values.size() * sizeof(ggml_bf16_t)) {
        fprintf(stderr, "invalid tiny Fairy2i BF16 fixture size for %s: got %zu expected %zu\n", name,
                data.size() / sizeof(ggml_bf16_t), values.size());
        exit(EXIT_FAILURE);
    }
    std::vector<ggml_bf16_t> bf16(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        bf16[i] = ggml_fp32_to_bf16(values[i]);
    }
    memcpy(data.data(), bf16.data(), data.size());
    gguf_set_tensor_data(writer.gguf, name, data.data());
}

struct tiny_fairy2i_shape {
    int64_t n_embd;
    int64_t n_ff;
    int64_t n_head;
    int64_t n_head_kv;
    int64_t n_vocab;
};

static void initialize_tiny_fairy2i_exact_data(tiny_fairy2i_writer & writer, const tiny_fairy2i_shape & shape) {
    std::vector<uint32_t> token_embd((size_t) shape.n_embd * (size_t) shape.n_vocab);
    for (int64_t token = 0; token < shape.n_vocab; ++token) {
        for (int64_t k = 0; k < shape.n_embd; ++k) {
            const float real = (float) ((token * 13 + k * 7) % 17 - 8) / 32.0f;
            const float imag = (float) ((token * 5 + k * 11) % 19 - 9) / 64.0f;
            token_embd[(size_t) token * (size_t) shape.n_embd + (size_t) k] = pack_bf16_pair(real, imag);
        }
    }
    std::vector<uint8_t> & token_data = writer.data("token_embd");
    if (token_data.size() != token_embd.size() * sizeof(uint32_t)) {
        fprintf(stderr, "invalid tiny Fairy2i token embedding fixture size\n");
        exit(EXIT_FAILURE);
    }
    memcpy(token_data.data(), token_embd.data(), token_data.size());
    gguf_set_tensor_data(writer.gguf, "token_embd", token_data.data());

    const char * norm_names[] = { "output_norm", "blk.0.attn_norm", "blk.0.ffn_norm" };
    for (const char * name : norm_names) {
        std::vector<float> norm(writer.data(name).size() / sizeof(float));
        for (size_t i = 0; i < norm.size(); ++i) {
            norm[i] = 1.0f + (float) ((int) (i % 5) - 2) / 16.0f;
        }
        set_f32_data(writer, name, norm);
    }

    const char * bias_names[] = { "blk.0.attn_q.bias", "blk.0.attn_k.bias", "blk.0.attn_v.bias" };
    for (size_t b = 0; b < sizeof(bias_names) / sizeof(bias_names[0]); ++b) {
        std::vector<float> bias(writer.data(bias_names[b]).size() / sizeof(float));
        for (size_t i = 0; i < bias.size(); ++i) {
            bias[i] = (float) ((int) ((i * 3 + b * 5) % 13) - 6) / 256.0f;
        }
        set_f32_data(writer, bias_names[b], bias);
    }

    const char * bundle_names[] = {
        "blk.0.attn_q",   "blk.0.attn_k", "blk.0.attn_v",   "blk.0.attn_output",
        "blk.0.ffn_gate", "blk.0.ffn_up", "blk.0.ffn_down", "output",
    };
    constexpr float scales[8] = {
        0x1p-5f, 0x1p-6f, 0x1p-8f, 0x1p-9f, 0x1p-6f, 0x1p-7f, 0x1p-9f, 0x1p-8f,
    };
    for (size_t linear = 0; linear < sizeof(bundle_names) / sizeof(bundle_names[0]); ++linear) {
        const std::string codes_name  = std::string(bundle_names[linear]) + ".bundle.codes";
        const std::string scales_name = std::string(bundle_names[linear]) + ".bundle.scales";

        std::vector<uint8_t> & codes = writer.data(codes_name.c_str());
        for (size_t i = 0; i < codes.size(); ++i) {
            const size_t tile = i / (64u * 4u * 16u);
            codes[i]          = (uint8_t) (0xe4u ^ (uint8_t) ((i + linear + tile) & 0x03u) * 0x55u);
        }
        gguf_set_tensor_data(writer.gguf, codes_name.c_str(), codes.data());

        std::vector<float> linear_scales(writer.data(scales_name.c_str()).size() / sizeof(ggml_bf16_t));
        for (size_t i = 0; i < linear_scales.size(); ++i) {
            const float tile_scale = 1.0f + (float) (i / 8u) * 0.125f;
            linear_scales[i]       = scales[i % 8u] * tile_scale * (linear == 7 ? 2.0f : 1.0f);
        }
        set_bf16_data(writer, scales_name.c_str(), linear_scales);
    }
}

static void add_tiny_fairy2i_metadata(gguf_context *             gguf,
                                      bool                       bundle,
                                      bool                       w1,
                                      bool                       exact,
                                      bool                       qwen2_legacy,
                                      const tiny_fairy2i_shape & shape) {
    gguf_set_val_str(gguf, "general.architecture", "fairy2i");
    gguf_set_val_str(gguf, "general.name", "tiny-fairy2i-w1");
    gguf_set_val_u32(gguf, "general.file_type",
                     bundle ? LLAMA_FTYPE_MOSTLY_FAIRY2I_BUNDLE_V1 : LLAMA_FTYPE_MOSTLY_FAIRY2I_TILE64_V2);
    if (bundle) {
        gguf_set_val_u32(gguf, "general.alignment", 64);
    }

    gguf_set_val_u32(gguf, "fairy2i.context_length", 16);
    gguf_set_val_u32(gguf, "fairy2i.embedding_length", (uint32_t) shape.n_embd);
    gguf_set_val_u32(gguf, "fairy2i.block_count", 1);
    gguf_set_val_u32(gguf, "fairy2i.feed_forward_length", (uint32_t) shape.n_ff);
    gguf_set_val_u32(gguf, "fairy2i.attention.head_count", (uint32_t) shape.n_head);
    gguf_set_val_u32(gguf, "fairy2i.attention.head_count_kv", (uint32_t) shape.n_head_kv);
    gguf_set_val_f32(gguf, "fairy2i.attention.layer_norm_rms_epsilon", 1e-6f);
    gguf_set_val_u32(gguf, "fairy2i.rope.dimension_count", (uint32_t) (2 * shape.n_embd / shape.n_head));
    gguf_set_val_f32(gguf, "fairy2i.rope.freq_base", 10000.0f);
    gguf_set_val_u32(gguf, "fairy2i.vocab_size", (uint32_t) shape.n_vocab);

    gguf_set_val_u32(gguf, "fairy2i.schema_version", exact ? 3 : bundle ? 2 : 1);
    gguf_set_val_str(gguf, "fairy2i.base_arch", w1 ? "qwen3" : exact || qwen2_legacy ? "qwen2" : "llama");
    gguf_set_val_str(gguf, "fairy2i.quant.format", "fairy2i_tile64_v2");
    gguf_set_val_u32(gguf, "fairy2i.quant.residual_steps", w1 ? 1 : 2);
    gguf_set_val_str(gguf, "fairy2i.quant.codebook", "{+/-1,+/-i}");
    gguf_set_val_str(gguf, "fairy2i.quant.variant", w1 ? "tile64_v2_w1_learned_scale" : "tile64_v2");
    gguf_set_val_str(gguf, "fairy2i.attn.layout",
                     w1                    ? "qwen3_real" :
                     exact || qwen2_legacy ? "qwen2_real" :
                                             "llama_real");
    gguf_set_val_str(gguf, "fairy2i.tokenizer.profile", w1 || exact || qwen2_legacy ? "qwen2" : "llama_bpe");
    gguf_set_val_u32(gguf, "fairy2i.quant.tile_size", 64);
    if (w1) {
        gguf_set_val_str(gguf, "fairy2i.quant.scale_source", "learned");
    } else {
        gguf_set_val_str(gguf, "fairy2i.quant.scale_stat", "dominant_mean_abs");
    }
    if (bundle) {
        gguf_set_val_str(gguf, "fairy2i.weight.layout", "bundle_m64k64_v1");
        if (exact) {
            gguf_set_val_str(gguf, "fairy2i.weight.scale_dtype", "bf16");
            gguf_set_val_str(gguf, "fairy2i.quant.numeric_profile", "script_f32reduce_bf16scale_v1");
        } else if (qwen2_legacy) {
            gguf_set_val_str(gguf, "fairy2i.weight.scale_dtype", "f16");
            gguf_set_val_str(gguf, "fairy2i.quant.numeric_profile", "legacy_f16_v1");
        }
        gguf_set_val_str(gguf, "fairy2i.weight.scale_scope", "m64_k64");
        gguf_set_val_str(gguf, "fairy2i.weight.code_order", "m16_q4_branch_lane");
        gguf_set_val_str(gguf, "fairy2i.weight.branch_order", w1 ? "U0,W0" : "U0,U1,W0,W1");
        gguf_set_val_u32(gguf, "fairy2i.weight.m_block", 64);
        gguf_set_val_u32(gguf, "fairy2i.weight.k_block", 64);
        gguf_set_val_u32(gguf, "fairy2i.weight.m_subtile", 16);
    }

    gguf_set_val_str(gguf, "tokenizer.ggml.model", "no_vocab");
}

static void add_tiny_fairy2i_linear(tiny_fairy2i_writer & writer, const char * base, bool skip_w0, bool add_s1) {
    const std::string u0 = std::string(base) + ".U.s0";
    const std::string w0 = std::string(base) + ".W.s0";
    writer.add_tensor(u0.c_str(), GGML_TYPE_FAIRY2I_TILE64_V2, 64, 64);
    if (!skip_w0) {
        writer.add_tensor(w0.c_str(), GGML_TYPE_FAIRY2I_TILE64_V2, 64, 64);
    }
    if (add_s1) {
        const std::string u1 = std::string(base) + ".U.s1";
        writer.add_tensor(u1.c_str(), GGML_TYPE_FAIRY2I_TILE64_V2, 64, 64);
    }
}

static void add_tiny_fairy2i_bundle(tiny_fairy2i_writer & writer,
                                    const char *          base,
                                    int                   codes_branches,
                                    int                   scales_branches,
                                    bool                  skip_scales,
                                    bool                  add_old_stage,
                                    ggml_type             codes_type,
                                    ggml_type             scales_type,
                                    int64_t               codes_physical_tiles,
                                    int64_t               scales_physical_tiles) {
    const std::string codes  = std::string(base) + ".bundle.codes";
    const std::string scales = std::string(base) + ".bundle.scales";
    writer.add_bundle_tensor(codes.c_str(), codes_type, 16, codes_branches, 64, codes_physical_tiles);
    if (!skip_scales) {
        writer.add_bundle_tensor(scales.c_str(), scales_type, 2, scales_branches, scales_physical_tiles, 1);
    }
    if (add_old_stage) {
        const std::string u0 = std::string(base) + ".U.s0";
        const std::string w0 = std::string(base) + ".W.s0";
        writer.add_tensor(u0.c_str(), GGML_TYPE_FAIRY2I_TILE64_V2, 64, 64);
        writer.add_tensor(w0.c_str(), GGML_TYPE_FAIRY2I_TILE64_V2, 64, 64);
        if (codes_branches == 4) {
            const std::string u1 = std::string(base) + ".U.s1";
            const std::string w1 = std::string(base) + ".W.s1";
            writer.add_tensor(u1.c_str(), GGML_TYPE_FAIRY2I_TILE64_V2, 64, 64);
            writer.add_tensor(w1.c_str(), GGML_TYPE_FAIRY2I_TILE64_V2, 64, 64);
        }
    }
}

static const char * loader_case_label(fairy2i_loader_case tc) {
    switch (tc) {
        case fairy2i_loader_case::valid_w1:
            return "valid";
        case fairy2i_loader_case::mixed_w1_w2:
            return "mixed";
        case fairy2i_loader_case::incomplete_w1:
            return "incomplete";
        case fairy2i_loader_case::valid_bundle_w1:
            return "bundle-w1";
        case fairy2i_loader_case::valid_bundle_w1_qkv:
            return "bundle-w1-qkv";
        case fairy2i_loader_case::valid_bundle_w1_qat:
            return "bundle-w1-qat";
        case fairy2i_loader_case::valid_bundle_w1_qat_qkv_bias:
            return "bundle-w1-qat-qkv-bias";
        case fairy2i_loader_case::valid_bundle_w2:
            return "bundle-w2";
        case fairy2i_loader_case::valid_bundle_w2_legacy_explicit:
            return "bundle-w2-legacy-explicit";
        case fairy2i_loader_case::valid_bundle_w2_exact:
            return "bundle-w2-exact";
        case fairy2i_loader_case::valid_bundle_w2_exact_gqa:
            return "bundle-w2-exact-gqa";
        case fairy2i_loader_case::incomplete_bundle:
            return "bundle-incomplete";
        case fairy2i_loader_case::mixed_bundle:
            return "bundle-mixed";
        case fairy2i_loader_case::invalid_bundle_shape:
            return "bundle-shape";
        case fairy2i_loader_case::invalid_bundle_alignment:
            return "bundle-alignment";
        case fairy2i_loader_case::invalid_bundle_branch_order:
            return "bundle-branches";
        case fairy2i_loader_case::invalid_bundle_scales_f32:
            return "bundle-scales-f32";
        case fairy2i_loader_case::invalid_bundle_scales_bf16:
            return "bundle-scales-bf16";
        case fairy2i_loader_case::invalid_bundle_scales_shape:
            return "bundle-scales-shape";
        case fairy2i_loader_case::invalid_bundle_codes_type:
            return "bundle-codes-type";
        case fairy2i_loader_case::invalid_bundle_codes_branches:
            return "bundle-codes-branches";
        case fairy2i_loader_case::invalid_bundle_schema:
            return "bundle-schema";
        case fairy2i_loader_case::invalid_bundle_scale_scope:
            return "bundle-scale-scope";
        case fairy2i_loader_case::invalid_bundle_code_order:
            return "bundle-code-order";
        case fairy2i_loader_case::invalid_bundle_variant:
            return "bundle-variant";
        case fairy2i_loader_case::invalid_exact_missing_scale_dtype:
            return "exact-missing-scale-dtype";
        case fairy2i_loader_case::invalid_exact_missing_numeric_profile:
            return "exact-missing-numeric-profile";
        case fairy2i_loader_case::invalid_exact_scale_dtype:
            return "exact-scale-dtype";
        case fairy2i_loader_case::invalid_exact_numeric_profile:
            return "exact-numeric-profile";
        case fairy2i_loader_case::invalid_exact_base_arch_llama:
            return "exact-base-arch-llama";
        case fairy2i_loader_case::invalid_exact_missing_q_bias:
            return "exact-missing-q-bias";
        case fairy2i_loader_case::invalid_exact_q_bias_f16:
            return "exact-q-bias-f16";
        case fairy2i_loader_case::invalid_exact_token_embd_f16:
            return "exact-token-embd-f16";
        case fairy2i_loader_case::invalid_exact_output_norm_f16:
            return "exact-output-norm-f16";
        case fairy2i_loader_case::invalid_exact_attn_norm_f16:
            return "exact-attn-norm-f16";
        case fairy2i_loader_case::invalid_exact_ffn_norm_f16:
            return "exact-ffn-norm-f16";
        case fairy2i_loader_case::invalid_exact_scales_f16:
            return "exact-scales-f16";
        case fairy2i_loader_case::invalid_exact_file_type:
            return "exact-file-type";
        case fairy2i_loader_case::invalid_exact_quant_format:
            return "exact-quant-format";
        case fairy2i_loader_case::invalid_exact_codebook:
            return "exact-codebook";
        case fairy2i_loader_case::invalid_exact_obsolete_f32_contract:
            return "exact-obsolete-f32-contract";
        case fairy2i_loader_case::invalid_exact_legacy_contract:
            return "exact-legacy-contract";
        case fairy2i_loader_case::invalid_legacy_exact_profile:
            return "legacy-exact-profile";
        case fairy2i_loader_case::invalid_legacy_exact_contract:
            return "legacy-exact-contract";
        case fairy2i_loader_case::invalid_legacy_missing_numeric_contract:
            return "legacy-missing-numeric-contract";
        case fairy2i_loader_case::invalid_legacy_missing_scale_dtype:
            return "legacy-missing-scale-dtype";
        case fairy2i_loader_case::invalid_legacy_missing_numeric_profile:
            return "legacy-missing-numeric-profile";
        case fairy2i_loader_case::invalid_exact_w1:
            return "exact-w1";
        case fairy2i_loader_case::invalid_exact_attn_layout_llama:
            return "exact-attn-layout-llama";
        case fairy2i_loader_case::invalid_exact_attn_layout_qwen3:
            return "exact-attn-layout-qwen3";
        case fairy2i_loader_case::invalid_exact_missing_attn_layout:
            return "exact-missing-attn-layout";
        case fairy2i_loader_case::invalid_exact_tokenizer_profile:
            return "exact-tokenizer-profile";
        case fairy2i_loader_case::invalid_exact_missing_tokenizer_profile:
            return "exact-missing-tokenizer-profile";
        case fairy2i_loader_case::invalid_exact_output_bias_f32:
            return "exact-output-bias-f32";
        case fairy2i_loader_case::invalid_exact_output_bias_f16:
            return "exact-output-bias-f16";
        case fairy2i_loader_case::invalid_exact_dense_f16:
            return "exact-dense-f16";
        case fairy2i_loader_case::invalid_exact_dense_only:
            return "exact-dense-only";
        case fairy2i_loader_case::invalid_qat_schema:
            return "qat-schema";
        case fairy2i_loader_case::invalid_qat_base_arch:
            return "qat-base-arch";
        case fairy2i_loader_case::invalid_qat_attn_layout:
            return "qat-attn-layout";
        case fairy2i_loader_case::invalid_qat_quant_variant:
            return "qat-quant-variant";
        case fairy2i_loader_case::invalid_qat_scale_dtype:
            return "qat-scale-dtype";
        case fairy2i_loader_case::invalid_qat_numeric_profile:
            return "qat-numeric-profile";
        case fairy2i_loader_case::invalid_qat_scales_f16:
            return "qat-scales-f16";
        case fairy2i_loader_case::invalid_qat_output_f16:
            return "qat-output-f16";
        case fairy2i_loader_case::invalid_qat_missing_output:
            return "qat-missing-output";
        case fairy2i_loader_case::invalid_qat_q_norm_f16:
            return "qat-q-norm-f16";
        case fairy2i_loader_case::invalid_qat_missing_q_norm:
            return "qat-missing-q-norm";
        case fairy2i_loader_case::invalid_qat_qkv_bias_f16:
            return "qat-qkv-bias-f16";
    }
    return "unknown";
}

static std::string write_tiny_fairy2i_model(fairy2i_loader_case tc) {
    const char *      label = loader_case_label(tc);
    const std::string path  = make_tmp_path(label);

    tiny_fairy2i_writer writer;
    const bool               bundle    = is_bundle_case(tc);
    const bool               w1        = is_w1_case(tc);
    const bool               exact     = is_exact_case(tc);
    const bool               qat       = is_qat_case(tc);
    const bool               exact_gqa = tc == fairy2i_loader_case::valid_bundle_w2_exact_gqa;
    const tiny_fairy2i_shape shape     = {
        /*.n_embd   =*/exact_gqa || qat ? 128 : 64,
        /*.n_ff     =*/64,
        /*.n_head   =*/exact_gqa ? 4 : 2,
        /*.n_head_kv=*/2,
        /*.n_vocab  =*/128,
    };
    const int64_t n_embd_kv = shape.n_embd * shape.n_head_kv / shape.n_head;
    add_tiny_fairy2i_metadata(writer.gguf, bundle, w1, exact, is_qwen2_legacy_case(tc), shape);
    if (qat) {
        gguf_set_val_u32(writer.gguf, "fairy2i.schema_version", 4);
        gguf_set_val_str(writer.gguf, "fairy2i.weight.scale_dtype", "bf16");
        gguf_set_val_str(writer.gguf, "fairy2i.quant.numeric_profile", "qat_bf16_learned_scale_v1");
    }
    if (tc == fairy2i_loader_case::invalid_bundle_alignment) {
        gguf_set_val_u32(writer.gguf, "general.alignment", 32);
    }
    if (tc == fairy2i_loader_case::invalid_bundle_branch_order) {
        gguf_set_val_str(writer.gguf, "fairy2i.weight.branch_order", "W0,U0");
    }
    if (tc == fairy2i_loader_case::invalid_bundle_schema) {
        gguf_set_val_u32(writer.gguf, "fairy2i.schema_version", 4);
    }
    if (tc == fairy2i_loader_case::invalid_bundle_scale_scope) {
        gguf_set_val_str(writer.gguf, "fairy2i.weight.scale_scope", "row_k64");
    }
    if (tc == fairy2i_loader_case::invalid_bundle_code_order) {
        gguf_set_val_str(writer.gguf, "fairy2i.weight.code_order", "row_q4_branch_lane");
    }
    if (tc == fairy2i_loader_case::invalid_bundle_variant) {
        gguf_set_val_str(writer.gguf, "fairy2i.quant.variant", "tile64_v1");
    }
    if (tc == fairy2i_loader_case::invalid_qat_schema) {
        gguf_set_val_u32(writer.gguf, "fairy2i.schema_version", 3);
    }
    if (tc == fairy2i_loader_case::invalid_qat_base_arch) {
        gguf_set_val_str(writer.gguf, "fairy2i.base_arch", "qwen2");
    }
    if (tc == fairy2i_loader_case::invalid_qat_attn_layout) {
        gguf_set_val_str(writer.gguf, "fairy2i.attn.layout", "qwen2_real");
    }
    if (tc == fairy2i_loader_case::invalid_qat_quant_variant) {
        gguf_set_val_str(writer.gguf, "fairy2i.quant.variant", "tile64_v2");
        gguf_set_val_u32(writer.gguf, "fairy2i.quant.residual_steps", 2);
        gguf_set_val_str(writer.gguf, "fairy2i.quant.scale_stat", "dominant_mean_abs");
        gguf_set_val_str(writer.gguf, "fairy2i.weight.branch_order", "U0,U1,W0,W1");
    }
    if (tc == fairy2i_loader_case::invalid_qat_scale_dtype) {
        gguf_set_val_str(writer.gguf, "fairy2i.weight.scale_dtype", "f16");
    }
    if (tc == fairy2i_loader_case::invalid_qat_numeric_profile) {
        gguf_set_val_str(writer.gguf, "fairy2i.quant.numeric_profile", "legacy_f16_v1");
    }
    if (tc == fairy2i_loader_case::invalid_exact_missing_scale_dtype) {
        gguf_remove_key(writer.gguf, "fairy2i.weight.scale_dtype");
    }
    if (tc == fairy2i_loader_case::invalid_exact_missing_numeric_profile) {
        gguf_remove_key(writer.gguf, "fairy2i.quant.numeric_profile");
    }
    if (tc == fairy2i_loader_case::invalid_exact_scale_dtype) {
        gguf_set_val_str(writer.gguf, "fairy2i.weight.scale_dtype", "f16");
    }
    if (tc == fairy2i_loader_case::invalid_exact_numeric_profile) {
        gguf_set_val_str(writer.gguf, "fairy2i.quant.numeric_profile", "legacy_f16_v1");
    }
    if (tc == fairy2i_loader_case::invalid_exact_base_arch_llama) {
        gguf_set_val_str(writer.gguf, "fairy2i.base_arch", "llama");
    }
    if (tc == fairy2i_loader_case::invalid_exact_file_type) {
        gguf_set_val_u32(writer.gguf, "general.file_type", LLAMA_FTYPE_MOSTLY_FAIRY2I_TILE64_V2);
    }
    if (tc == fairy2i_loader_case::invalid_exact_quant_format) {
        gguf_set_val_str(writer.gguf, "fairy2i.quant.format", "fairy2i_tile64_v1");
    }
    if (tc == fairy2i_loader_case::invalid_exact_codebook) {
        gguf_set_val_str(writer.gguf, "fairy2i.quant.codebook", "{0,1}");
    }
    if (tc == fairy2i_loader_case::invalid_exact_obsolete_f32_contract) {
        gguf_set_val_str(writer.gguf, "fairy2i.weight.scale_dtype", "f32");
        gguf_set_val_str(writer.gguf, "fairy2i.quant.numeric_profile", "script_bf16_f32_v1");
    }
    if (tc == fairy2i_loader_case::invalid_exact_legacy_contract) {
        gguf_set_val_str(writer.gguf, "fairy2i.weight.scale_dtype", "f16");
        gguf_set_val_str(writer.gguf, "fairy2i.quant.numeric_profile", "legacy_f16_v1");
    }
    if (tc == fairy2i_loader_case::invalid_legacy_exact_profile) {
        gguf_set_val_str(writer.gguf, "fairy2i.weight.scale_dtype", "f16");
        gguf_set_val_str(writer.gguf, "fairy2i.quant.numeric_profile", "script_f32reduce_bf16scale_v1");
    }
    if (tc == fairy2i_loader_case::invalid_legacy_exact_contract) {
        gguf_set_val_str(writer.gguf, "fairy2i.weight.scale_dtype", "bf16");
        gguf_set_val_str(writer.gguf, "fairy2i.quant.numeric_profile", "script_f32reduce_bf16scale_v1");
    }
    if (tc == fairy2i_loader_case::invalid_legacy_missing_numeric_contract) {
        gguf_remove_key(writer.gguf, "fairy2i.weight.scale_dtype");
        gguf_remove_key(writer.gguf, "fairy2i.quant.numeric_profile");
    }
    if (tc == fairy2i_loader_case::invalid_legacy_missing_scale_dtype) {
        gguf_remove_key(writer.gguf, "fairy2i.weight.scale_dtype");
    }
    if (tc == fairy2i_loader_case::invalid_legacy_missing_numeric_profile) {
        gguf_remove_key(writer.gguf, "fairy2i.quant.numeric_profile");
    }
    if (tc == fairy2i_loader_case::valid_bundle_w2_legacy_explicit) {
        gguf_set_val_str(writer.gguf, "fairy2i.weight.scale_dtype", "f16");
        gguf_set_val_str(writer.gguf, "fairy2i.quant.numeric_profile", "legacy_f16_v1");
    }
    if (tc == fairy2i_loader_case::invalid_exact_attn_layout_llama) {
        gguf_set_val_str(writer.gguf, "fairy2i.attn.layout", "llama_real");
    }
    if (tc == fairy2i_loader_case::invalid_exact_attn_layout_qwen3) {
        gguf_set_val_str(writer.gguf, "fairy2i.attn.layout", "qwen3_real");
    }
    if (tc == fairy2i_loader_case::invalid_exact_missing_attn_layout) {
        gguf_remove_key(writer.gguf, "fairy2i.attn.layout");
    }
    if (tc == fairy2i_loader_case::invalid_exact_tokenizer_profile) {
        gguf_set_val_str(writer.gguf, "fairy2i.tokenizer.profile", "llama_bpe");
    }
    if (tc == fairy2i_loader_case::invalid_exact_missing_tokenizer_profile) {
        gguf_remove_key(writer.gguf, "fairy2i.tokenizer.profile");
    }

    writer.add_tensor("token_embd",
                      tc == fairy2i_loader_case::invalid_exact_token_embd_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                      shape.n_embd, shape.n_vocab);
    writer.add_tensor("output_norm",
                      tc == fairy2i_loader_case::invalid_exact_output_norm_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                      2 * shape.n_embd, 1);
    if ((!exact || tc == fairy2i_loader_case::invalid_exact_dense_f16 ||
         tc == fairy2i_loader_case::invalid_exact_dense_only) &&
        tc != fairy2i_loader_case::invalid_qat_missing_output) {
        const ggml_type output_type =
            qat ? (tc == fairy2i_loader_case::invalid_qat_output_f16 ? GGML_TYPE_F16 : GGML_TYPE_BF16) :
            tc == fairy2i_loader_case::invalid_exact_dense_f16 ? GGML_TYPE_F16 :
                                                                 GGML_TYPE_F32;
        writer.add_tensor("output", output_type, 2 * shape.n_embd, shape.n_vocab);
    }
    writer.add_tensor("blk.0.attn_norm",
                      tc == fairy2i_loader_case::invalid_exact_attn_norm_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                      2 * shape.n_embd, 1);
    writer.add_tensor("blk.0.ffn_norm",
                      tc == fairy2i_loader_case::invalid_exact_ffn_norm_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                      2 * shape.n_embd, 1);
    if (w1) {
        if (tc != fairy2i_loader_case::invalid_qat_missing_q_norm) {
            writer.add_tensor("blk.0.attn_q_norm",
                              tc == fairy2i_loader_case::invalid_qat_q_norm_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                              2 * shape.n_embd / shape.n_head, 1);
        }
        writer.add_tensor("blk.0.attn_k_norm", GGML_TYPE_F32, 2 * shape.n_embd / shape.n_head, 1);
    }
    if (exact) {
        if (tc != fairy2i_loader_case::invalid_exact_missing_q_bias) {
            writer.add_tensor("blk.0.attn_q.bias",
                              tc == fairy2i_loader_case::invalid_exact_q_bias_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                              2 * shape.n_embd, 1);
        }
        writer.add_tensor("blk.0.attn_k.bias", GGML_TYPE_F32, 2 * n_embd_kv, 1);
        writer.add_tensor("blk.0.attn_v.bias", GGML_TYPE_F32, 2 * n_embd_kv, 1);
        if (tc == fairy2i_loader_case::invalid_exact_output_bias_f32 ||
            tc == fairy2i_loader_case::invalid_exact_output_bias_f16) {
            writer.add_tensor("blk.0.attn_output.bias",
                              tc == fairy2i_loader_case::invalid_exact_output_bias_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                              2 * shape.n_embd, 1);
        }
    }

    const char * linear_names[] = { "blk.0.attn_q",   "blk.0.attn_k", "blk.0.attn_v",  "blk.0.attn_output",
                                    "blk.0.ffn_gate", "blk.0.ffn_up", "blk.0.ffn_down" };
    const bool   merged_qkv     = tc == fairy2i_loader_case::valid_bundle_w1_qkv ||
                                  tc == fairy2i_loader_case::valid_bundle_w1_qat_qkv_bias ||
                                  tc == fairy2i_loader_case::invalid_qat_qkv_bias_f16;
    if (merged_qkv) {
        const int64_t qkv_physical_tiles = (shape.n_embd / 64) * ((shape.n_embd + 2 * n_embd_kv) / 64);
        add_tiny_fairy2i_bundle(writer, "blk.0.attn_qkv", 2, 2, false, false, GGML_TYPE_FAIRY2I_BUNDLE_CODES,
                                exact || qat ? GGML_TYPE_BF16 : GGML_TYPE_F16, qkv_physical_tiles, qkv_physical_tiles);
        if (qat) {
            writer.add_tensor("blk.0.attn_qkv.bias",
                              tc == fairy2i_loader_case::invalid_qat_qkv_bias_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                              2 * (shape.n_embd + 2 * n_embd_kv), 1);
        }
    }
    for (size_t i = 0; i < sizeof(linear_names) / sizeof(linear_names[0]); ++i) {
        if (merged_qkv && i < 3) {
            continue;
        }
        if (bundle) {
            int        codes_branches  = w1 ? 2 : 4;
            int        scales_branches = codes_branches;
            ggml_type  codes_type      = GGML_TYPE_FAIRY2I_BUNDLE_CODES;
            const bool use_bf16_scales = (exact && tc != fairy2i_loader_case::invalid_exact_legacy_contract) || qat ||
                                         tc == fairy2i_loader_case::invalid_legacy_exact_contract;
            ggml_type  scales_type     = use_bf16_scales ? GGML_TYPE_BF16 : GGML_TYPE_F16;
            if (tc == fairy2i_loader_case::invalid_exact_obsolete_f32_contract) {
                scales_type = GGML_TYPE_F32;
            }
            const int64_t linear_k[] = {
                shape.n_embd, shape.n_embd, shape.n_embd, shape.n_embd, shape.n_embd, shape.n_embd, shape.n_ff,
            };
            const int64_t linear_m[] = {
                shape.n_embd, n_embd_kv, n_embd_kv, shape.n_embd, shape.n_ff, shape.n_ff, shape.n_embd,
            };
            int64_t codes_physical_tiles  = (linear_k[i] / 64) * (linear_m[i] / 64);
            int64_t scales_physical_tiles = codes_physical_tiles;

            if (i == 0) {
                if (tc == fairy2i_loader_case::invalid_bundle_shape) {
                    codes_physical_tiles = 2;
                } else if (tc == fairy2i_loader_case::invalid_bundle_scales_f32) {
                    scales_type = GGML_TYPE_F32;
                } else if (tc == fairy2i_loader_case::invalid_bundle_scales_bf16) {
                    scales_type = GGML_TYPE_BF16;
                } else if (tc == fairy2i_loader_case::invalid_exact_scales_f16) {
                    scales_type = GGML_TYPE_F16;
                } else if (tc == fairy2i_loader_case::invalid_qat_scales_f16) {
                    scales_type = GGML_TYPE_F16;
                } else if (tc == fairy2i_loader_case::invalid_bundle_scales_shape) {
                    scales_physical_tiles = 2;
                } else if (tc == fairy2i_loader_case::invalid_bundle_codes_type) {
                    codes_type = GGML_TYPE_F16;
                } else if (tc == fairy2i_loader_case::invalid_bundle_codes_branches) {
                    codes_branches -= 1;
                }
            }

            add_tiny_fairy2i_bundle(writer, linear_names[i], codes_branches, scales_branches,
                                    tc == fairy2i_loader_case::incomplete_bundle && i == 0,
                                    tc == fairy2i_loader_case::mixed_bundle && i == 0, codes_type, scales_type,
                                    codes_physical_tiles, scales_physical_tiles);
        } else {
            add_tiny_fairy2i_linear(writer, linear_names[i], tc == fairy2i_loader_case::incomplete_w1 && i == 0,
                                    tc == fairy2i_loader_case::mixed_w1_w2 && i == 0);
        }
    }

    if (exact && tc != fairy2i_loader_case::invalid_exact_dense_only) {
        const int64_t output_physical_tiles = (shape.n_embd / 64) * ((shape.n_vocab / 2) / 64);
        add_tiny_fairy2i_bundle(writer, "output", 4, 4, false, false, GGML_TYPE_FAIRY2I_BUNDLE_CODES,
                                tc == fairy2i_loader_case::invalid_exact_legacy_contract       ? GGML_TYPE_F16 :
                                tc == fairy2i_loader_case::invalid_exact_obsolete_f32_contract ? GGML_TYPE_F32 :
                                                                                                 GGML_TYPE_BF16,
                                output_physical_tiles, output_physical_tiles);
    }
    if (tc == fairy2i_loader_case::valid_bundle_w2_exact || tc == fairy2i_loader_case::valid_bundle_w2_exact_gqa) {
        initialize_tiny_fairy2i_exact_data(writer, shape);
    }

    if (!gguf_write_to_file(writer.gguf, path.c_str(), false)) {
        fprintf(stderr, "failed to write %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    return path;
}

static bool load_model_expect(const char * label, fairy2i_loader_case tc, bool expect_success) {
    const std::string path = write_tiny_fairy2i_model(tc);

    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers       = 0;
    params.use_mmap           = false;
    params.check_tensors      = false;

    llama_model * model = llama_model_load_from_file(path.c_str(), params);
    const bool    ok    = model != nullptr;
    if (ok && expect_success && tc == fairy2i_loader_case::valid_w1 &&
        llama_model_rope_type(model) != LLAMA_ROPE_TYPE_NEOX) {
        fprintf(stderr, "%s: expected Qwen3-real Fairy2i rope type NEOX, got %d\n", label,
                (int) llama_model_rope_type(model));
        llama_model_free(model);
        unlink(path.c_str());
        return false;
    }
    if (ok && expect_success && is_exact_case(tc) &&
        (!model->fairy2i_uses_qwen2_exact_numeric_profile() || model->fairy2i_schema_version != 3)) {
        fprintf(stderr, "%s: expected schema v3 script_f32reduce_bf16scale_v1 model state\n", label);
        llama_model_free(model);
        unlink(path.c_str());
        return false;
    }
    if (ok && expect_success && is_qat_case(tc) &&
        (!model->fairy2i_uses_qwen3_qat_numeric_profile() || !model->fairy2i_uses_bf16_runtime_profile() ||
         model->fairy2i_schema_version != 4)) {
        fprintf(stderr, "%s: expected schema v4 qat_bf16_learned_scale_v1 model state\n", label);
        llama_model_free(model);
        unlink(path.c_str());
        return false;
    }
    if (ok && expect_success && !is_exact_case(tc) && !is_qat_case(tc) && is_bundle_case(tc) &&
        (model->fairy2i_uses_bf16_runtime_profile() || model->fairy2i_schema_version != 2)) {
        fprintf(stderr, "%s: expected schema v2 legacy_f16_v1 model state\n", label);
        llama_model_free(model);
        unlink(path.c_str());
        return false;
    }
    llama_model_free(model);
    unlink(path.c_str());

    if (ok != expect_success) {
        fprintf(stderr, "%s: expected load %s, got %s\n", label, expect_success ? "success" : "failure",
                ok ? "success" : "failure");
        return false;
    }

    printf("  %s: PASS\n", label);
    return true;
}

static bool load_model_with_env_expect_failure(const char * label, fairy2i_loader_case tc, const char * env_name) {
    const std::string path = write_tiny_fairy2i_model(tc);

    const char *      old_env   = getenv(env_name);
    const bool        had_env   = old_env != nullptr;
    const std::string old_value = had_env ? old_env : "";
    setenv(env_name, "1", 1);

    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers       = 0;
    params.use_mmap           = false;
    params.check_tensors      = false;

    llama_model * model  = llama_model_load_from_file(path.c_str(), params);
    const bool    loaded = model != nullptr;
    llama_model_free(model);
    unlink(path.c_str());

    if (had_env) {
        setenv(env_name, old_value.c_str(), 1);
    } else {
        unsetenv(env_name);
    }

    if (loaded) {
        fprintf(stderr, "%s: expected load failure while %s=1\n", label, env_name);
        return false;
    }

    printf("  %s: PASS\n", label);
    return true;
}

static bool init_context_expect(const char *          label,
                                fairy2i_loader_case   tc,
                                ggml_type             type_k,
                                ggml_type             type_v,
                                bool                  expect_success,
                                llama_flash_attn_type flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED,
                                int                   n_gpu_layers    = 0,
                                bool                  offload_kqv     = true,
                                const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr) {
    const std::string path = write_tiny_fairy2i_model(tc);

    llama_model_params model_params    = llama_model_default_params();
    model_params.n_gpu_layers          = n_gpu_layers;
    model_params.use_mmap              = false;
    model_params.check_tensors         = false;
    model_params.tensor_buft_overrides = tensor_buft_overrides;

    llama_model * model = llama_model_load_from_file(path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "%s: failed to load tiny model\n", label);
        unlink(path.c_str());
        return false;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx                = 8;
    context_params.n_batch              = 8;
    context_params.n_ubatch             = 8;
    context_params.flash_attn_type      = flash_attn_type;
    context_params.type_k               = type_k;
    context_params.type_v               = type_v;
    context_params.offload_kqv          = offload_kqv;

    llama_context * context     = llama_init_from_model(model, context_params);
    const bool      initialized = context != nullptr;
    llama_free(context);
    llama_model_free(model);
    unlink(path.c_str());

    if (initialized != expect_success) {
        fprintf(stderr, "%s: expected context init %s, got %s\n", label, expect_success ? "success" : "failure",
                initialized ? "success" : "failure");
        return false;
    }

    printf("  %s: PASS\n", label);
    return true;
}

static ggml_backend_buffer_type_t tensor_buffer_type(const ggml_tensor * tensor) {
    while (tensor && tensor->view_src) {
        tensor = tensor->view_src;
    }

    return tensor && tensor->buffer ? ggml_backend_buffer_get_type(tensor->buffer) : nullptr;
}

static bool buffer_type_uses_backend(ggml_backend_buffer_type_t buft, const char * backend_name) {
    ggml_backend_dev_t dev = buft ? ggml_backend_buft_get_device(buft) : nullptr;
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    return reg && strcmp(ggml_backend_reg_name(reg), backend_name) == 0;
}

static ggml_backend_dev_t find_metal_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "Metal") == 0) {
            return dev;
        }
    }
    return nullptr;
}

static bool probe_metal_exact_w1_support(ggml_backend_dev_t dev, bool & supported) {
    struct ggml_init_params params = {
        /*.mem_size   =*/64 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Fairy2i Qwen3 QAT: failed to initialize exact W1 Metal capability probe\n");
        return false;
    }

    ggml_tensor * x        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 1);
    ggml_tensor * codes    = ggml_new_tensor_4d(ctx, GGML_TYPE_FAIRY2I_BUNDLE_CODES, 16, 2, 64, 1);
    ggml_tensor * scales   = ggml_new_tensor_3d(ctx, GGML_TYPE_BF16, 2, 2, 1);
    ggml_tensor * exact_w1 = ggml_fairy2i_wide_linear_w1_bundle(ctx, x, codes, scales, nullptr, 64, 64);
    supported              = ggml_backend_dev_supports_op(dev, exact_w1);
    ggml_free(ctx);
    return true;
}

struct exact_decode_result {
    std::vector<float>   logits;
    std::vector<uint8_t> k_cur_bf16;
    std::vector<uint8_t> v_cur_bf16;
    std::vector<size_t>  k_chunk_bytes;
    std::vector<size_t>  v_chunk_bytes;
    llama_token          greedy_token = -1;
};

struct exact_eval_capture {
    const char *          label;
    exact_decode_result * result;
    bool                  valid = true;
};

static bool capture_exact_gqa_kv(ggml_tensor * tensor, bool ask, void * user_data) {
    exact_eval_capture * capture = (exact_eval_capture *) user_data;
    const bool           is_k    = strcmp(tensor->name, "Kcur_bf16_exact-0") == 0;
    const bool           is_v    = strcmp(tensor->name, "Vcur_bf16_exact-0") == 0;
    if (ask) {
        return is_k || is_v;
    }

    if ((!is_k && !is_v) || tensor->type != GGML_TYPE_BF16 || tensor->ne[0] != 64 || tensor->ne[1] != 2 ||
        tensor->ne[3] != 1) {
        fprintf(stderr, "%s: invalid exact GQA K/V callback tensor %s type=%s shape=[%lld,%lld,%lld,%lld]\n",
                capture->label, tensor->name, ggml_type_name(tensor->type), (long long) tensor->ne[0],
                (long long) tensor->ne[1], (long long) tensor->ne[2], (long long) tensor->ne[3]);
        capture->valid = false;
        return true;
    }

    std::vector<uint8_t> & values = is_k ? capture->result->k_cur_bf16 : capture->result->v_cur_bf16;
    std::vector<size_t> &  chunks = is_k ? capture->result->k_chunk_bytes : capture->result->v_chunk_bytes;
    const size_t           nbytes = ggml_nbytes(tensor);
    const size_t           offset = values.size();
    values.resize(offset + nbytes);
    ggml_backend_tensor_get(tensor, values.data() + offset, 0, nbytes);
    chunks.push_back(nbytes);
    return true;
}

static bool run_exact_decode(const char *          label,
                             llama_model *         model,
                             llama_flash_attn_type flash_attn_type,
                             exact_decode_result & result) {
    result.logits.clear();
    result.k_cur_bf16.clear();
    result.v_cur_bf16.clear();
    result.k_chunk_bytes.clear();
    result.v_chunk_bytes.clear();
    result.greedy_token        = -1;
    exact_eval_capture capture = {
        /*.label =*/label,
        /*.result=*/&result,
    };

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx                = 16;
    context_params.n_batch              = 8;
    context_params.n_ubatch             = 8;
    context_params.n_seq_max            = 1;
    context_params.n_threads            = 2;
    context_params.n_threads_batch      = 2;
    context_params.flash_attn_type      = flash_attn_type;
    context_params.type_k               = GGML_TYPE_BF16;
    context_params.type_v               = GGML_TYPE_BF16;
    context_params.offload_kqv          = true;
    context_params.no_perf              = true;
    context_params.cb_eval              = capture_exact_gqa_kv;
    context_params.cb_eval_user_data    = &capture;

    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) {
        fprintf(stderr, "%s: failed to initialize context\n", label);
        return false;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (n_vocab <= 0) {
        fprintf(stderr, "%s: invalid vocabulary size\n", label);
        llama_free(context);
        return false;
    }

    result.logits.reserve(3 * (size_t) n_vocab);
    auto capture_last_logits = [&]() {
        const float * logits = llama_get_logits_ith(context, -1);
        if (!logits) {
            fprintf(stderr, "%s: missing logits\n", label);
            return false;
        }
        for (int32_t i = 0; i < n_vocab; ++i) {
            if (!std::isfinite(logits[i])) {
                fprintf(stderr, "%s: non-finite logit at %d\n", label, i);
                return false;
            }
        }
        result.logits.insert(result.logits.end(), logits, logits + n_vocab);
        return true;
    };

    llama_token prompt_tokens[] = { 1, 7, 23, 42 };
    int32_t     decode_status   = llama_decode(context, llama_batch_get_one(prompt_tokens, 4));
    if (decode_status != 0 || !capture_last_logits()) {
        if (decode_status != 0) {
            fprintf(stderr, "%s: prefill llama_decode failed with status %d\n", label, decode_status);
        }
        llama_free(context);
        return false;
    }

    const llama_token decode_tokens[] = { 99, 17 };
    for (llama_token decode_token : decode_tokens) {
        decode_status = llama_decode(context, llama_batch_get_one(&decode_token, 1));
        if (decode_status != 0 || !capture_last_logits()) {
            if (decode_status != 0) {
                fprintf(stderr, "%s: token llama_decode failed with status %d\n", label, decode_status);
            }
            llama_free(context);
            return false;
        }
    }

    const auto final_logits_begin = result.logits.end() - n_vocab;
    result.greedy_token =
        (llama_token) std::distance(final_logits_begin, std::max_element(final_logits_begin, result.logits.end()));
    if (result.greedy_token < 0 || result.greedy_token >= n_vocab) {
        fprintf(stderr, "%s: invalid greedy token\n", label);
        llama_free(context);
        return false;
    }
    const std::vector<size_t> expected_chunks = {
        64u * 2u * 4u * sizeof(ggml_bf16_t),
        64u * 2u * sizeof(ggml_bf16_t),
        64u * 2u * sizeof(ggml_bf16_t),
    };
    if (!capture.valid || result.k_chunk_bytes != expected_chunks || result.v_chunk_bytes != expected_chunks) {
        fprintf(stderr, "%s: exact GQA K/V capture did not observe prefill plus two decode chunks\n", label);
        llama_free(context);
        return false;
    }

    llama_free(context);
    return true;
}

static bool test_exact_full_graph_cpu() {
    const std::string path = write_tiny_fairy2i_model(fairy2i_loader_case::valid_bundle_w2_exact_gqa);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = 0;
    model_params.use_mmap           = false;
    model_params.check_tensors      = false;

    llama_model * model = llama_model_load_from_file(path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Fairy2i exact full graph CPU: failed to load model\n");
        unlink(path.c_str());
        return false;
    }

    exact_decode_result first;
    exact_decode_result second;
    bool ok = run_exact_decode("Fairy2i exact CPU full graph", model, LLAMA_FLASH_ATTN_TYPE_DISABLED, first);
    ok =
        run_exact_decode("Fairy2i exact CPU deterministic repeat", model, LLAMA_FLASH_ATTN_TYPE_DISABLED, second) && ok;
    if (ok) {
        if (first.logits.empty()) {
            fprintf(stderr, "Fairy2i exact full graph CPU: missing logits\n");
            ok = false;
        } else if (const auto minmax_logits = std::minmax_element(first.logits.begin(), first.logits.end());
                   *minmax_logits.first == *minmax_logits.second) {
            fprintf(stderr, "Fairy2i exact full graph CPU: non-zero fixture produced degenerate logits\n");
            ok = false;
        } else if (first.greedy_token != second.greedy_token || first.logits.size() != second.logits.size() ||
                   memcmp(first.logits.data(), second.logits.data(), first.logits.size() * sizeof(float)) != 0 ||
                   first.k_cur_bf16 != second.k_cur_bf16 || first.v_cur_bf16 != second.v_cur_bf16) {
            fprintf(stderr,
                    "Fairy2i exact full graph GQA CPU: repeated logits/token/KV are not deterministic (%d vs %d)\n",
                    first.greedy_token, second.greedy_token);
            ok = false;
        }
    }

    llama_model_free(model);
    unlink(path.c_str());

    if (ok) {
        printf("  Fairy2i exact full graph GQA CPU prefill/two-decode/KV: PASS\n");
    }
    return ok;
}

static bool test_exact_full_graph_cpu_metal() {
    if (!find_metal_device()) {
        printf("  Fairy2i exact full graph CPU/Metal logits: SKIP (Metal unavailable)\n");
        return true;
    }

    const std::string path = write_tiny_fairy2i_model(fairy2i_loader_case::valid_bundle_w2_exact_gqa);

    llama_model_params cpu_model_params = llama_model_default_params();
    cpu_model_params.n_gpu_layers       = 0;
    cpu_model_params.use_mmap           = false;
    cpu_model_params.check_tensors      = false;

    llama_model_params metal_model_params = cpu_model_params;
    metal_model_params.n_gpu_layers       = 2;

    llama_model * cpu_model   = llama_model_load_from_file(path.c_str(), cpu_model_params);
    llama_model * metal_model = llama_model_load_from_file(path.c_str(), metal_model_params);
    if (!cpu_model || !metal_model) {
        fprintf(stderr, "Fairy2i exact full graph: failed to load CPU or Metal model\n");
        llama_model_free(metal_model);
        llama_model_free(cpu_model);
        unlink(path.c_str());
        return false;
    }

    exact_decode_result cpu;
    exact_decode_result metal_first;
    exact_decode_result metal_second;
    bool ok = run_exact_decode("Fairy2i exact CPU reference graph", cpu_model, LLAMA_FLASH_ATTN_TYPE_DISABLED, cpu);
    ok      = run_exact_decode("Fairy2i exact Metal Flash Attention graph", metal_model, LLAMA_FLASH_ATTN_TYPE_ENABLED,
                               metal_first) &&
              ok;
    ok      = run_exact_decode("Fairy2i exact Metal deterministic repeat", metal_model, LLAMA_FLASH_ATTN_TYPE_ENABLED,
                               metal_second) &&
              ok;

    if (ok && cpu.logits.size() != metal_first.logits.size()) {
        fprintf(stderr, "Fairy2i exact full graph: CPU/Metal logits size mismatch\n");
        ok = false;
    }
    if (ok && (cpu.k_cur_bf16 != metal_first.k_cur_bf16 || cpu.v_cur_bf16 != metal_first.v_cur_bf16)) {
        fprintf(stderr, "Fairy2i exact full graph GQA: CPU/Metal BF16 K/V inputs differ\n");
        ok = false;
    }

    double max_abs          = 0.0;
    double squared_error    = 0.0;
    double reference_energy = 0.0;
    double nmse             = 0.0;
    size_t max_abs_index    = 0;
    if (ok) {
        for (size_t i = 0; i < cpu.logits.size(); ++i) {
            const double error = (double) metal_first.logits[i] - (double) cpu.logits[i];
            if (std::abs(error) > max_abs) {
                max_abs       = std::abs(error);
                max_abs_index = i;
            }
            squared_error += error * error;
            reference_energy += (double) cpu.logits[i] * (double) cpu.logits[i];
        }
        const auto minmax_logits = std::minmax_element(cpu.logits.begin(), cpu.logits.end());
        nmse                     = reference_energy > 0.0 ? squared_error / reference_energy : squared_error;
        if (reference_energy == 0.0 || *minmax_logits.first == *minmax_logits.second) {
            fprintf(stderr, "Fairy2i exact full graph: non-zero fixture produced degenerate CPU logits\n");
            ok = false;
        } else if (nmse > 1e-6 || max_abs > 1e-2) {
            fprintf(
                stderr,
                "Fairy2i exact full graph: CPU/Metal mismatch NMSE=%.9g max_abs=%.9g at %zu (CPU=%.9g Metal=%.9g)\n",
                nmse, max_abs, max_abs_index, cpu.logits[max_abs_index], metal_first.logits[max_abs_index]);
            ok = false;
        }
    }

    if (ok &&
        (metal_first.greedy_token != metal_second.greedy_token ||
         metal_first.logits.size() != metal_second.logits.size() ||
         memcmp(metal_first.logits.data(), metal_second.logits.data(), metal_first.logits.size() * sizeof(float)) !=
             0 ||
         metal_first.k_cur_bf16 != metal_second.k_cur_bf16 || metal_first.v_cur_bf16 != metal_second.v_cur_bf16)) {
        fprintf(stderr,
                "Fairy2i exact full graph GQA: repeated Metal logits/token/KV are not deterministic (%d vs %d)\n",
                metal_first.greedy_token, metal_second.greedy_token);
        ok = false;
    }

    llama_model_free(metal_model);
    llama_model_free(cpu_model);
    unlink(path.c_str());

    if (ok) {
        printf("  Fairy2i exact full graph GQA CPU/Metal logits+KV: PASS (NMSE=%.9g max_abs=%.9g)\n", nmse, max_abs);
    }
    return ok;
}

struct qwen3_qat_graph_capture {
    bool q_norm                            = false;
    bool k_norm                            = false;
    bool k_bf16                            = false;
    bool v_bf16                            = false;
    bool dense_output_input_bf16           = false;
    bool dense_output_result_bf16_boundary = false;
    bool valid                             = true;
};

static bool capture_qwen3_qat_graph(ggml_tensor * tensor, bool ask, void * user_data) {
    qwen3_qat_graph_capture * capture                = (qwen3_qat_graph_capture *) user_data;
    const bool                is_q_norm              = strcmp(tensor->name, "Qcur_normed-0") == 0;
    const bool                is_k_norm              = strcmp(tensor->name, "Kcur_normed-0") == 0;
    const bool                is_k_bf16              = strcmp(tensor->name, "Kcur_bf16_exact-0") == 0;
    const bool                is_v_bf16              = strcmp(tensor->name, "Vcur_bf16_exact-0") == 0;
    const bool                is_dense_output_input  = strcmp(tensor->name, "result_output_inp_bf16") == 0;
    const bool                is_dense_output_result = strcmp(tensor->name, "result_output") == 0;
    if (ask) {
        return is_q_norm || is_k_norm || is_k_bf16 || is_v_bf16 || is_dense_output_input || is_dense_output_result;
    }

    if (is_q_norm || is_k_norm) {
        const bool norm_valid = tensor->op == GGML_OP_FAIRY2I_RMS_NORM_EXACT && tensor->type == GGML_TYPE_F32 &&
                                tensor->ne[0] == 128 && tensor->ne[1] == 2;
        capture->valid        = capture->valid && norm_valid;
        capture->q_norm       = capture->q_norm || is_q_norm;
        capture->k_norm       = capture->k_norm || is_k_norm;
    } else if (is_k_bf16 || is_v_bf16) {
        capture->valid  = capture->valid && tensor->type == GGML_TYPE_BF16 && tensor->ne[0] == 128;
        capture->k_bf16 = capture->k_bf16 || is_k_bf16;
        capture->v_bf16 = capture->v_bf16 || is_v_bf16;
    } else if (is_dense_output_input || is_dense_output_result) {
        const bool boundary_valid = is_dense_output_input ?
                                        tensor->type == GGML_TYPE_BF16 :
                                        tensor->type == GGML_TYPE_F32 && tensor->op == GGML_OP_FAIRY2I_PACK_BF16_EXACT;
        capture->valid            = capture->valid && boundary_valid;
        capture->dense_output_input_bf16 = capture->dense_output_input_bf16 || is_dense_output_input;
        capture->dense_output_result_bf16_boundary =
            capture->dense_output_result_bf16_boundary || is_dense_output_result;
    }
    return true;
}

static bool test_qwen3_qat_full_graph_metal() {
    const std::string path = write_tiny_fairy2i_model(fairy2i_loader_case::valid_bundle_w1_qat);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = 2;
    model_params.use_mmap           = false;
    model_params.check_tensors      = false;

    llama_model * model = llama_model_load_from_file(path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Fairy2i Qwen3 QAT graph: failed to load model\n");
        unlink(path.c_str());
        return false;
    }

    qwen3_qat_graph_capture capture;
    llama_context_params    context_params = llama_context_default_params();
    context_params.n_ctx                   = 8;
    context_params.n_batch                 = 8;
    context_params.n_ubatch                = 8;
    context_params.flash_attn_type         = LLAMA_FLASH_ATTN_TYPE_AUTO;
    context_params.type_k                  = GGML_TYPE_COUNT;
    context_params.type_v                  = GGML_TYPE_COUNT;
    context_params.cb_eval                 = capture_qwen3_qat_graph;
    context_params.cb_eval_user_data       = &capture;

    llama_context * context = llama_init_from_model(model, context_params);
    bool            ok      = context != nullptr;
    if (ok) {
        llama_token token = 1;
        ok                = llama_decode(context, llama_batch_get_one(&token, 1)) == 0;
    }
    if (ok && (!capture.valid || !capture.q_norm || !capture.k_norm || !capture.k_bf16 || !capture.v_bf16 ||
               !capture.dense_output_input_bf16 || !capture.dense_output_result_bf16_boundary)) {
        fprintf(stderr,
                "Fairy2i Qwen3 QAT graph: missing exact head-dim RMSNorm, BF16 K/V, or BF16 dense output boundary\n");
        ok = false;
    }

    llama_free(context);
    llama_model_free(model);
    unlink(path.c_str());
    if (ok) {
        printf("  Fairy2i Qwen3 QAT exact head-dim RMSNorm/BF16 KV+dense output boundary: PASS\n");
    }
    return ok;
}

static bool check_bundle_input_placement(const char * label, int n_gpu_layers, bool expect_metal) {
    const std::string path = write_tiny_fairy2i_model(fairy2i_loader_case::valid_bundle_w2);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = n_gpu_layers;
    model_params.use_mmap           = false;
    model_params.check_tensors      = false;

    llama_model * model = llama_model_load_from_file(path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "%s: failed to load tiny bundle model\n", label);
        unlink(path.c_str());
        return false;
    }

    ggml_backend_buffer_type_t embedding_buft = tensor_buffer_type(model->tok_embd);
    const std::string embedding_buffer = embedding_buft ? ggml_backend_buft_name(embedding_buft) : "<unallocated>";
    const bool        placement_ok     = expect_metal ? buffer_type_uses_backend(embedding_buft, "Metal") :
                                                        embedding_buft == ggml_backend_cpu_buffer_type();
    llama_model_free(model);
    unlink(path.c_str());

    if (!placement_ok) {
        fprintf(stderr, "%s: expected token_embd on %s, got %s\n", label, expect_metal ? "Metal" : "CPU",
                embedding_buffer.c_str());
        return false;
    }

    printf("  %s: PASS\n", label);
    return true;
}

int main() {
    llama_backend_init();

    bool ok = true;
    ok      = load_model_expect("Fairy2i W1 complete tensor set", fairy2i_loader_case::valid_w1, true) && ok;
    ok      = load_model_expect("Fairy2i W1 rejects mixed s1 tensor", fairy2i_loader_case::mixed_w1_w2, false) && ok;
    ok = load_model_expect("Fairy2i W1 rejects incomplete tensor set", fairy2i_loader_case::incomplete_w1, false) && ok;
    ok = load_model_expect("Fairy2i bundle W1 complete tensor set", fairy2i_loader_case::valid_bundle_w1, true) && ok;
    ok = load_model_expect("Fairy2i bundle W1 merged QKV tensor set", fairy2i_loader_case::valid_bundle_w1_qkv, true) &&
         ok;
    ok = load_model_expect("Fairy2i Qwen3 QAT schema v4 accepts optional QKV bias",
                           fairy2i_loader_case::valid_bundle_w1_qat, true) &&
         ok;
    ok = load_model_expect("Fairy2i Qwen3 QAT schema v4 accepts merged BF16-runtime QKV bias",
                           fairy2i_loader_case::valid_bundle_w1_qat_qkv_bias, true) &&
         ok;
    ok = load_model_expect("Fairy2i bundle W2 complete tensor set", fairy2i_loader_case::valid_bundle_w2, true) && ok;
    ok = load_model_expect("Fairy2i bundle W2 explicit legacy numeric profile",
                           fairy2i_loader_case::valid_bundle_w2_legacy_explicit, true) &&
         ok;
    ok = load_model_expect("Fairy2i bundle W2 exact schema v3 tensor set", fairy2i_loader_case::valid_bundle_w2_exact,
                           true) &&
         ok;
    ok = load_model_expect("Fairy2i bundle W2 exact schema v3 GQA tensor set",
                           fairy2i_loader_case::valid_bundle_w2_exact_gqa, true) &&
         ok;
    ok = load_model_expect("Fairy2i bundle rejects incomplete tensor set", fairy2i_loader_case::incomplete_bundle,
                           false) &&
         ok;
    ok = load_model_expect("Fairy2i bundle rejects legacy row-block stage tensors", fairy2i_loader_case::mixed_bundle,
                           false) &&
         ok;
    ok = load_model_expect("Fairy2i bundle rejects invalid shape", fairy2i_loader_case::invalid_bundle_shape, false) &&
         ok;
    ok = load_model_expect("Fairy2i bundle rejects insufficient alignment",
                           fairy2i_loader_case::invalid_bundle_alignment, false) &&
         ok;
    ok = load_model_expect("Fairy2i bundle rejects branch order", fairy2i_loader_case::invalid_bundle_branch_order,
                           false) &&
         ok;
    ok =
        load_model_expect("Fairy2i bundle rejects F32 scales", fairy2i_loader_case::invalid_bundle_scales_f32, false) &&
        ok;
    ok = load_model_expect("Fairy2i bundle rejects BF16 scales", fairy2i_loader_case::invalid_bundle_scales_bf16,
                           false) &&
         ok;
    ok = load_model_expect("Fairy2i bundle rejects independently invalid scale shape",
                           fairy2i_loader_case::invalid_bundle_scales_shape, false) &&
         ok;
    ok = load_model_expect("Fairy2i bundle rejects invalid codes type", fairy2i_loader_case::invalid_bundle_codes_type,
                           false) &&
         ok;
    ok = load_model_expect("Fairy2i bundle rejects invalid codes branch count",
                           fairy2i_loader_case::invalid_bundle_codes_branches, false) &&
         ok;
    ok =
        load_model_expect("Fairy2i bundle rejects schema version", fairy2i_loader_case::invalid_bundle_schema, false) &&
        ok;
    ok = load_model_expect("Fairy2i bundle rejects scale scope", fairy2i_loader_case::invalid_bundle_scale_scope,
                           false) &&
         ok;
    ok =
        load_model_expect("Fairy2i bundle rejects code order", fairy2i_loader_case::invalid_bundle_code_order, false) &&
        ok;
    ok =
        load_model_expect("Fairy2i bundle rejects quant variant", fairy2i_loader_case::invalid_bundle_variant, false) &&
        ok;
    ok = load_model_expect("Fairy2i exact schema rejects missing scale dtype",
                           fairy2i_loader_case::invalid_exact_missing_scale_dtype, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects missing numeric profile",
                           fairy2i_loader_case::invalid_exact_missing_numeric_profile, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects wrong scale dtype metadata",
                           fairy2i_loader_case::invalid_exact_scale_dtype, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects wrong numeric profile",
                           fairy2i_loader_case::invalid_exact_numeric_profile, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects non-Qwen2 base architecture",
                           fairy2i_loader_case::invalid_exact_base_arch_llama, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects missing Q bias",
                           fairy2i_loader_case::invalid_exact_missing_q_bias, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects F16 Q bias carrier",
                           fairy2i_loader_case::invalid_exact_q_bias_f16, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects F16 token embedding carrier",
                           fairy2i_loader_case::invalid_exact_token_embd_f16, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects F16 output norm carrier",
                           fairy2i_loader_case::invalid_exact_output_norm_f16, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects F16 attention norm carrier",
                           fairy2i_loader_case::invalid_exact_attn_norm_f16, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects F16 FFN norm carrier",
                           fairy2i_loader_case::invalid_exact_ffn_norm_f16, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects F16 scale tensors",
                           fairy2i_loader_case::invalid_exact_scales_f16, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects non-bundle general.file_type",
                           fairy2i_loader_case::invalid_exact_file_type, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects wrong quant format",
                           fairy2i_loader_case::invalid_exact_quant_format, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects wrong codebook", fairy2i_loader_case::invalid_exact_codebook,
                           false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects obsolete F32 scale contract",
                           fairy2i_loader_case::invalid_exact_obsolete_f32_contract, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects internally consistent legacy F16 contract",
                           fairy2i_loader_case::invalid_exact_legacy_contract, false) &&
         ok;
    ok = load_model_expect("Fairy2i legacy schema rejects exact numeric profile",
                           fairy2i_loader_case::invalid_legacy_exact_profile, false) &&
         ok;
    ok = load_model_expect("Fairy2i legacy schema rejects internally consistent exact BF16 contract",
                           fairy2i_loader_case::invalid_legacy_exact_contract, false) &&
         ok;
    ok = load_model_expect("Fairy2i legacy W2 schema rejects missing numeric contract",
                           fairy2i_loader_case::invalid_legacy_missing_numeric_contract, false) &&
         ok;
    ok = load_model_expect("Fairy2i legacy W2 schema rejects missing scale dtype",
                           fairy2i_loader_case::invalid_legacy_missing_scale_dtype, false) &&
         ok;
    ok = load_model_expect("Fairy2i legacy W2 schema rejects missing numeric profile",
                           fairy2i_loader_case::invalid_legacy_missing_numeric_profile, false) &&
         ok;
    ok =
        load_model_expect("Fairy2i exact schema rejects W1 bundle", fairy2i_loader_case::invalid_exact_w1, false) && ok;
    ok = load_model_expect("Fairy2i exact schema rejects llama attention layout",
                           fairy2i_loader_case::invalid_exact_attn_layout_llama, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects Qwen3 attention layout",
                           fairy2i_loader_case::invalid_exact_attn_layout_qwen3, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects missing attention layout",
                           fairy2i_loader_case::invalid_exact_missing_attn_layout, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects wrong tokenizer profile",
                           fairy2i_loader_case::invalid_exact_tokenizer_profile, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects missing tokenizer profile",
                           fairy2i_loader_case::invalid_exact_missing_tokenizer_profile, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects F32 attention output bias",
                           fairy2i_loader_case::invalid_exact_output_bias_f32, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects F16 attention output bias",
                           fairy2i_loader_case::invalid_exact_output_bias_f16, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects F16 dense output",
                           fairy2i_loader_case::invalid_exact_dense_f16, false) &&
         ok;
    ok = load_model_expect("Fairy2i exact schema rejects dense-only output fallback",
                           fairy2i_loader_case::invalid_exact_dense_only, false) &&
         ok;
    ok = load_model_expect("Fairy2i Qwen3 QAT rejects non-v4 schema", fairy2i_loader_case::invalid_qat_schema, false) &&
         ok;
    ok = load_model_expect("Fairy2i Qwen3 QAT rejects non-Qwen3 base architecture",
                           fairy2i_loader_case::invalid_qat_base_arch, false) &&
         ok;
    ok = load_model_expect("Fairy2i Qwen3 QAT rejects non-Qwen3 attention layout",
                           fairy2i_loader_case::invalid_qat_attn_layout, false) &&
         ok;
    ok = load_model_expect("Fairy2i Qwen3 QAT rejects non-W1 quant variant",
                           fairy2i_loader_case::invalid_qat_quant_variant, false) &&
         ok;
    ok = load_model_expect("Fairy2i Qwen3 QAT rejects non-BF16 scale metadata",
                           fairy2i_loader_case::invalid_qat_scale_dtype, false) &&
         ok;
    ok = load_model_expect("Fairy2i Qwen3 QAT rejects wrong numeric profile",
                           fairy2i_loader_case::invalid_qat_numeric_profile, false) &&
         ok;
    ok = load_model_expect("Fairy2i Qwen3 QAT rejects F16 scale tensors", fairy2i_loader_case::invalid_qat_scales_f16,
                           false) &&
         ok;
    ok = load_model_expect("Fairy2i Qwen3 QAT rejects F16 dense lm_head", fairy2i_loader_case::invalid_qat_output_f16,
                           false) &&
         ok;
    ok = load_model_expect("Fairy2i Qwen3 QAT rejects missing dense lm_head",
                           fairy2i_loader_case::invalid_qat_missing_output, false) &&
         ok;
    ok = load_model_expect("Fairy2i Qwen3 QAT rejects F16 Q RMSNorm carrier",
                           fairy2i_loader_case::invalid_qat_q_norm_f16, false) &&
         ok;
    ok = load_model_expect("Fairy2i Qwen3 QAT rejects missing Q RMSNorm",
                           fairy2i_loader_case::invalid_qat_missing_q_norm, false) &&
         ok;
    ok = load_model_expect("Fairy2i Qwen3 QAT rejects F16 merged QKV bias carrier",
                           fairy2i_loader_case::invalid_qat_qkv_bias_f16, false) &&
         ok;
    const llama_context_params default_context_params = llama_context_default_params();
    if (default_context_params.type_k != GGML_TYPE_COUNT || default_context_params.type_v != GGML_TYPE_COUNT) {
        fprintf(stderr, "Fairy2i KV AUTO default: expected GGML_TYPE_COUNT sentinels\n");
        ok = false;
    } else {
        printf("  Fairy2i KV AUTO default sentinels: PASS\n");
    }
    ok = load_model_with_env_expect_failure("Fairy2i exact schema rejects forced dense output",
                                            fairy2i_loader_case::valid_bundle_w2_exact,
                                            "LLAMA_FAIRY2I_FORCE_DENSE_OUTPUT") &&
         ok;
    ok = load_model_with_env_expect_failure("Fairy2i exact schema rejects NEON dense output",
                                            fairy2i_loader_case::valid_bundle_w2_exact, "LLAMA_FAIRY2I_OUTPUT_NEON") &&
         ok;
    ok = load_model_with_env_expect_failure("Fairy2i Qwen3 QAT schema rejects NEON dense output",
                                            fairy2i_loader_case::valid_bundle_w1_qat, "LLAMA_FAIRY2I_OUTPUT_NEON") &&
         ok;
    ok = init_context_expect("Fairy2i exact context resolves AUTO KV cache to BF16",
                             fairy2i_loader_case::valid_bundle_w2_exact, GGML_TYPE_COUNT, GGML_TYPE_COUNT, true) &&
         ok;
    ok = init_context_expect("Fairy2i exact CPU context resolves AUTO Flash Attention to reference graph",
                             fairy2i_loader_case::valid_bundle_w2_exact, GGML_TYPE_COUNT, GGML_TYPE_COUNT, true,
                             LLAMA_FLASH_ATTN_TYPE_AUTO) &&
         ok;
    ok = init_context_expect("Fairy2i exact CPU context rejects explicitly enabled Flash Attention",
                             fairy2i_loader_case::valid_bundle_w2_exact, GGML_TYPE_COUNT, GGML_TYPE_COUNT, false,
                             LLAMA_FLASH_ATTN_TYPE_ENABLED) &&
         ok;
    ok = init_context_expect("Fairy2i exact context accepts AUTO K and explicit BF16 V",
                             fairy2i_loader_case::valid_bundle_w2_exact, GGML_TYPE_COUNT, GGML_TYPE_BF16, true) &&
         ok;
    ok = init_context_expect("Fairy2i exact context rejects explicit F16 KV cache",
                             fairy2i_loader_case::valid_bundle_w2_exact, GGML_TYPE_F16, GGML_TYPE_F16, false) &&
         ok;
    ok = init_context_expect("Fairy2i exact context rejects quantized K cache",
                             fairy2i_loader_case::valid_bundle_w2_exact, GGML_TYPE_Q8_0, GGML_TYPE_BF16, false) &&
         ok;
    ok = init_context_expect("Fairy2i Qwen3 QAT context rejects explicit F16 KV cache",
                             fairy2i_loader_case::valid_bundle_w1_qat, GGML_TYPE_F16, GGML_TYPE_F16, false) &&
         ok;
    ok = init_context_expect("Fairy2i Qwen3 QAT context rejects unsupported all-CPU BF16-scale W1 graph",
                             fairy2i_loader_case::valid_bundle_w1_qat, GGML_TYPE_COUNT, GGML_TYPE_COUNT, false,
                             LLAMA_FLASH_ATTN_TYPE_AUTO) &&
         ok;
    const llama_model_tensor_buft_override exact_cpu_tensor_override[] = {
        { "blk\\.0\\.attn_norm", ggml_backend_cpu_buffer_type() },
        { nullptr,               nullptr                        },
    };
    ok = init_context_expect("Fairy2i exact all-CPU context rejects tensor buffer overrides",
                             fairy2i_loader_case::valid_bundle_w2_exact, GGML_TYPE_COUNT, GGML_TYPE_COUNT, false,
                             LLAMA_FLASH_ATTN_TYPE_DISABLED, 0, true, exact_cpu_tensor_override) &&
         ok;
    ok = test_exact_full_graph_cpu() && ok;
    if (ggml_backend_dev_t metal_dev = find_metal_device()) {
        bool       exact_w1_supported = false;
        const bool capability_probed  = probe_metal_exact_w1_support(metal_dev, exact_w1_supported);
        ok                            = capability_probed && ok;
        if (capability_probed && exact_w1_supported) {
            printf("  Fairy2i Qwen3 QAT exact W1 Metal capability: PASS\n");
            ok = init_context_expect("Fairy2i Qwen3 QAT full Metal context resolves AUTO KV cache to BF16",
                                     fairy2i_loader_case::valid_bundle_w1_qat, GGML_TYPE_COUNT, GGML_TYPE_COUNT, true,
                                     LLAMA_FLASH_ATTN_TYPE_AUTO, 2) &&
                 ok;
            ok = init_context_expect("Fairy2i Qwen3 QAT full Metal context accepts explicit BF16 KV cache",
                                     fairy2i_loader_case::valid_bundle_w1_qat, GGML_TYPE_BF16, GGML_TYPE_BF16, true,
                                     LLAMA_FLASH_ATTN_TYPE_AUTO, 2) &&
                 ok;
            ok = test_qwen3_qat_full_graph_metal() && ok;
        } else if (capability_probed) {
            ok = init_context_expect("Fairy2i Qwen3 QAT context rejects Metal without exact W1 capability",
                                     fairy2i_loader_case::valid_bundle_w1_qat, GGML_TYPE_COUNT, GGML_TYPE_COUNT, false,
                                     LLAMA_FLASH_ATTN_TYPE_AUTO, 2) &&
                 ok;
            printf("  Fairy2i Qwen3 QAT full Metal graph: SKIP (exact W1 BF16-scale op unsupported)\n");
        }
        ok = init_context_expect("Fairy2i exact full Metal context requires Flash Attention",
                                 fairy2i_loader_case::valid_bundle_w2_exact, GGML_TYPE_COUNT, GGML_TYPE_COUNT, false,
                                 LLAMA_FLASH_ATTN_TYPE_DISABLED, 2) &&
             ok;
        ok = init_context_expect("Fairy2i exact full Metal context accepts automatic Flash Attention",
                                 fairy2i_loader_case::valid_bundle_w2_exact, GGML_TYPE_COUNT, GGML_TYPE_COUNT, true,
                                 LLAMA_FLASH_ATTN_TYPE_AUTO, 2) &&
             ok;
        ok = init_context_expect("Fairy2i exact context rejects mixed CPU/Metal placement",
                                 fairy2i_loader_case::valid_bundle_w2_exact, GGML_TYPE_COUNT, GGML_TYPE_COUNT, false,
                                 LLAMA_FLASH_ATTN_TYPE_DISABLED, 1) &&
             ok;
        ok = init_context_expect("Fairy2i exact full Metal context rejects offload_kqv=false",
                                 fairy2i_loader_case::valid_bundle_w2_exact, GGML_TYPE_COUNT, GGML_TYPE_COUNT, false,
                                 LLAMA_FLASH_ATTN_TYPE_AUTO, 2, false) &&
             ok;
        ok = init_context_expect("Fairy2i exact full Metal context rejects tensor buffer overrides",
                                 fairy2i_loader_case::valid_bundle_w2_exact, GGML_TYPE_COUNT, GGML_TYPE_COUNT, false,
                                 LLAMA_FLASH_ATTN_TYPE_AUTO, 2, true, exact_cpu_tensor_override) &&
             ok;
        ok = test_exact_full_graph_cpu_metal() && ok;
        ok = check_bundle_input_placement("Fairy2i full Metal bundle keeps embedding in one split", 2, true) && ok;
        ok = check_bundle_input_placement("Fairy2i partial Metal bundle keeps embedding on CPU", 1, false) && ok;
    } else {
        printf("  Fairy2i Qwen3 QAT full Metal context/graph: SKIP (Metal unavailable)\n");
        printf("  Fairy2i Metal bundle input placement: SKIP (Metal unavailable)\n");
    }

    llama_backend_free();
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
