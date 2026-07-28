// Fairy2i GGUF loader schema tests.

#include "../src/llama-model.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <unistd.h>

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
    valid_bundle_w2,
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
};

static bool is_bundle_case(fairy2i_loader_case tc) {
    return tc != fairy2i_loader_case::valid_w1 && tc != fairy2i_loader_case::mixed_w1_w2 &&
           tc != fairy2i_loader_case::incomplete_w1;
}

static bool is_w1_case(fairy2i_loader_case tc) {
    switch (tc) {
        case fairy2i_loader_case::valid_bundle_w2:
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
            return false;
        default:
            return true;
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
};

static void add_tiny_fairy2i_metadata(gguf_context * gguf, bool bundle, bool w1) {
    gguf_set_val_str(gguf, "general.architecture", "fairy2i");
    gguf_set_val_str(gguf, "general.name", "tiny-fairy2i-w1");
    gguf_set_val_u32(gguf, "general.file_type",
                     bundle ? LLAMA_FTYPE_MOSTLY_FAIRY2I_BUNDLE_V1 : LLAMA_FTYPE_MOSTLY_FAIRY2I_TILE64_V2);
    if (bundle) {
        gguf_set_val_u32(gguf, "general.alignment", 64);
    }

    gguf_set_val_u32(gguf, "fairy2i.context_length", 16);
    gguf_set_val_u32(gguf, "fairy2i.embedding_length", 64);
    gguf_set_val_u32(gguf, "fairy2i.block_count", 1);
    gguf_set_val_u32(gguf, "fairy2i.feed_forward_length", 64);
    gguf_set_val_u32(gguf, "fairy2i.attention.head_count", 2);
    gguf_set_val_u32(gguf, "fairy2i.attention.head_count_kv", 2);
    gguf_set_val_f32(gguf, "fairy2i.attention.layer_norm_rms_epsilon", 1e-6f);
    gguf_set_val_u32(gguf, "fairy2i.rope.dimension_count", 64);
    gguf_set_val_f32(gguf, "fairy2i.rope.freq_base", 10000.0f);
    gguf_set_val_u32(gguf, "fairy2i.vocab_size", 128);

    gguf_set_val_u32(gguf, "fairy2i.schema_version", bundle ? 2 : 1);
    gguf_set_val_str(gguf, "fairy2i.base_arch", w1 ? "qwen3" : "llama");
    gguf_set_val_str(gguf, "fairy2i.quant.format", "fairy2i_tile64_v2");
    gguf_set_val_u32(gguf, "fairy2i.quant.residual_steps", w1 ? 1 : 2);
    gguf_set_val_str(gguf, "fairy2i.quant.codebook", "{+/-1,+/-i}");
    gguf_set_val_str(gguf, "fairy2i.quant.variant", w1 ? "tile64_v2_w1_learned_scale" : "tile64_v2");
    gguf_set_val_str(gguf, "fairy2i.attn.layout", w1 ? "qwen3_real" : "llama_real");
    gguf_set_val_str(gguf, "fairy2i.tokenizer.profile", w1 ? "qwen2" : "llama_bpe");
    gguf_set_val_u32(gguf, "fairy2i.quant.tile_size", 64);
    if (w1) {
        gguf_set_val_str(gguf, "fairy2i.quant.scale_source", "learned");
    } else {
        gguf_set_val_str(gguf, "fairy2i.quant.scale_stat", "dominant_mean_abs");
    }
    if (bundle) {
        gguf_set_val_str(gguf, "fairy2i.weight.layout", "bundle_m64k64_v1");
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
        case fairy2i_loader_case::valid_bundle_w2:
            return "bundle-w2";
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
    }
    return "unknown";
}

static std::string write_tiny_fairy2i_model(fairy2i_loader_case tc) {
    const char *      label = loader_case_label(tc);
    const std::string path  = make_tmp_path(label);

    tiny_fairy2i_writer writer;
    const bool          bundle = is_bundle_case(tc);
    const bool          w1     = is_w1_case(tc);
    add_tiny_fairy2i_metadata(writer.gguf, bundle, w1);
    if (tc == fairy2i_loader_case::invalid_bundle_alignment) {
        gguf_set_val_u32(writer.gguf, "general.alignment", 32);
    }
    if (tc == fairy2i_loader_case::invalid_bundle_branch_order) {
        gguf_set_val_str(writer.gguf, "fairy2i.weight.branch_order", "W0,U0");
    }
    if (tc == fairy2i_loader_case::invalid_bundle_schema) {
        gguf_set_val_u32(writer.gguf, "fairy2i.schema_version", 3);
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

    writer.add_tensor("token_embd", GGML_TYPE_F32, 64, 128);
    writer.add_tensor("output_norm", GGML_TYPE_F32, 128, 1);
    writer.add_tensor("output", GGML_TYPE_F32, 128, 128);
    writer.add_tensor("blk.0.attn_norm", GGML_TYPE_F32, 128, 1);
    writer.add_tensor("blk.0.ffn_norm", GGML_TYPE_F32, 128, 1);
    if (w1) {
        writer.add_tensor("blk.0.attn_q_norm", GGML_TYPE_F32, 64, 1);
        writer.add_tensor("blk.0.attn_k_norm", GGML_TYPE_F32, 64, 1);
    }

    const char * linear_names[] = { "blk.0.attn_q",   "blk.0.attn_k", "blk.0.attn_v",  "blk.0.attn_output",
                                    "blk.0.ffn_gate", "blk.0.ffn_up", "blk.0.ffn_down" };
    const bool   merged_qkv     = tc == fairy2i_loader_case::valid_bundle_w1_qkv;
    if (merged_qkv) {
        add_tiny_fairy2i_bundle(writer, "blk.0.attn_qkv", 2, 2, false, false, GGML_TYPE_FAIRY2I_BUNDLE_CODES,
                                GGML_TYPE_F16, 3, 3);
    }
    for (size_t i = 0; i < sizeof(linear_names) / sizeof(linear_names[0]); ++i) {
        if (merged_qkv && i < 3) {
            continue;
        }
        if (bundle) {
            int       codes_branches        = w1 ? 2 : 4;
            int       scales_branches       = codes_branches;
            ggml_type codes_type            = GGML_TYPE_FAIRY2I_BUNDLE_CODES;
            ggml_type scales_type           = GGML_TYPE_F16;
            int64_t   codes_physical_tiles  = 1;
            int64_t   scales_physical_tiles = 1;

            if (i == 0) {
                if (tc == fairy2i_loader_case::invalid_bundle_shape) {
                    codes_physical_tiles = 2;
                } else if (tc == fairy2i_loader_case::invalid_bundle_scales_f32) {
                    scales_type = GGML_TYPE_F32;
                } else if (tc == fairy2i_loader_case::invalid_bundle_scales_bf16) {
                    scales_type = GGML_TYPE_BF16;
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

static bool metal_device_available() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "Metal") == 0) {
            return true;
        }
    }
    return false;
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
    ok = load_model_expect("Fairy2i bundle W2 complete tensor set", fairy2i_loader_case::valid_bundle_w2, true) && ok;
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
    if (metal_device_available()) {
        ok = check_bundle_input_placement("Fairy2i full Metal bundle keeps embedding in one split", 2, true) && ok;
        ok = check_bundle_input_placement("Fairy2i partial Metal bundle keeps embedding on CPU", 1, false) && ok;
    } else {
        printf("  Fairy2i Metal bundle input placement: SKIP (Metal unavailable)\n");
    }

    llama_backend_free();
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
