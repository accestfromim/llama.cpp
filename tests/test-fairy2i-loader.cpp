// Fairy2i GGUF loader schema tests.

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <unistd.h>

enum class fairy2i_loader_case {
    valid_w1,
    mixed_w1_w2,
    incomplete_w1,
};

static std::string make_tmp_path(const char * label) {
    std::string pattern = std::string("/tmp/llama-fairy2i-loader-") + label + "-XXXXXX.gguf";
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
    gguf_context *                  gguf = nullptr;
    ggml_context *                  ggml = nullptr;
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

        tensor_data.emplace_back(ggml_nbytes(tensor), uint8_t{0});
        gguf_set_tensor_data(gguf, name, tensor_data.back().data());
    }
};

static void add_tiny_fairy2i_metadata(gguf_context * gguf) {
    gguf_set_val_str(gguf, "general.architecture", "fairy2i");
    gguf_set_val_str(gguf, "general.name", "tiny-fairy2i-w1");
    gguf_set_val_u32(gguf, "general.file_type", LLAMA_FTYPE_MOSTLY_FAIRY2I_TILE64_V2);

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

    gguf_set_val_u32(gguf, "fairy2i.schema_version", 1);
    gguf_set_val_str(gguf, "fairy2i.base_arch", "qwen3");
    gguf_set_val_str(gguf, "fairy2i.quant.format", "fairy2i_tile64_v2");
    gguf_set_val_u32(gguf, "fairy2i.quant.residual_steps", 1);
    gguf_set_val_str(gguf, "fairy2i.quant.codebook", "{+/-1,+/-i}");
    gguf_set_val_str(gguf, "fairy2i.quant.variant", "tile64_v2_w1_learned_scale");
    gguf_set_val_str(gguf, "fairy2i.attn.layout", "qwen3_real");
    gguf_set_val_str(gguf, "fairy2i.tokenizer.profile", "qwen2");
    gguf_set_val_u32(gguf, "fairy2i.quant.tile_size", 64);
    gguf_set_val_str(gguf, "fairy2i.quant.scale_source", "learned");

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

static std::string write_tiny_fairy2i_model(fairy2i_loader_case tc) {
    const char * label = tc == fairy2i_loader_case::valid_w1       ? "valid" :
                         tc == fairy2i_loader_case::mixed_w1_w2    ? "mixed" :
                                                                      "incomplete";
    const std::string path = make_tmp_path(label);

    tiny_fairy2i_writer writer;
    add_tiny_fairy2i_metadata(writer.gguf);

    writer.add_tensor("token_embd", GGML_TYPE_F32, 64, 128);
    writer.add_tensor("output_norm", GGML_TYPE_F32, 128, 1);
    writer.add_tensor("output", GGML_TYPE_F32, 128, 128);
    writer.add_tensor("blk.0.attn_norm", GGML_TYPE_F32, 128, 1);
    writer.add_tensor("blk.0.ffn_norm", GGML_TYPE_F32, 128, 1);
    writer.add_tensor("blk.0.attn_q_norm", GGML_TYPE_F32, 64, 1);
    writer.add_tensor("blk.0.attn_k_norm", GGML_TYPE_F32, 64, 1);

    add_tiny_fairy2i_linear(writer, "blk.0.attn_q", tc == fairy2i_loader_case::incomplete_w1,
                            tc == fairy2i_loader_case::mixed_w1_w2);
    add_tiny_fairy2i_linear(writer, "blk.0.attn_k", false, false);
    add_tiny_fairy2i_linear(writer, "blk.0.attn_v", false, false);
    add_tiny_fairy2i_linear(writer, "blk.0.attn_output", false, false);
    add_tiny_fairy2i_linear(writer, "blk.0.ffn_gate", false, false);
    add_tiny_fairy2i_linear(writer, "blk.0.ffn_up", false, false);
    add_tiny_fairy2i_linear(writer, "blk.0.ffn_down", false, false);

    if (!gguf_write_to_file(writer.gguf, path.c_str(), false)) {
        fprintf(stderr, "failed to write %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    return path;
}

static bool load_model_expect(const char * label, fairy2i_loader_case tc, bool expect_success) {
    const std::string path = write_tiny_fairy2i_model(tc);

    llama_model_params params = llama_model_default_params();
    params.use_mmap          = false;
    params.check_tensors     = false;

    llama_model * model = llama_model_load_from_file(path.c_str(), params);
    const bool    ok    = model != nullptr;
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

int main() {
    llama_backend_init();

    bool ok = true;
    ok = load_model_expect("Fairy2i W1 complete tensor set", fairy2i_loader_case::valid_w1, true) && ok;
    ok = load_model_expect("Fairy2i W1 rejects mixed s1 tensor", fairy2i_loader_case::mixed_w1_w2, false) && ok;
    ok = load_model_expect("Fairy2i W1 rejects incomplete tensor set", fairy2i_loader_case::incomplete_w1, false) && ok;

    llama_backend_free();
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
