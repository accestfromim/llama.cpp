// Qwen3 Row4 GGUF loader and tiny full-graph tests.

#include "../src/llama-model.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
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

namespace {

static bool row4_cpu_backend_available() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        return false;
    }
    const ggml_init_params params = {
        /*.mem_size   =*/256 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor * x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 1);
    ggml_tensor * codes  = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES, 64, 4, 1, 8);
    ggml_tensor * scales = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, 128);
    ggml_tensor * result = ggml_row4_linear(ctx, x, codes, scales, 128, 128);
    const bool    ok     = ggml_backend_supports_op(backend, result);

    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok;
}

enum class loader_case {
    valid,
    incomplete_metadata,
    bad_layout,
    bad_alignment,
    missing_qkv_scales,
    wrong_qkv_codes_type,
    wrong_qkv_codes_shape,
    wrong_qkv_scales_type,
    missing_lm_head,
    missing_lm_head_codes,
    missing_lm_head_scales,
    wrong_lm_head_type,
    dense_lm_head,
    separate_q_weight,
    non_aligned_ffn,
};

static const char * case_name(loader_case tc) {
    switch (tc) {
        case loader_case::valid:
            return "valid";
        case loader_case::incomplete_metadata:
            return "incomplete-metadata";
        case loader_case::bad_layout:
            return "bad-layout";
        case loader_case::bad_alignment:
            return "bad-alignment";
        case loader_case::missing_qkv_scales:
            return "missing-qkv-scales";
        case loader_case::wrong_qkv_codes_type:
            return "wrong-qkv-codes-type";
        case loader_case::wrong_qkv_codes_shape:
            return "wrong-qkv-codes-shape";
        case loader_case::wrong_qkv_scales_type:
            return "wrong-qkv-scales-type";
        case loader_case::missing_lm_head:
            return "missing-lm-head";
        case loader_case::missing_lm_head_codes:
            return "missing-lm-head-codes";
        case loader_case::missing_lm_head_scales:
            return "missing-lm-head-scales";
        case loader_case::wrong_lm_head_type:
            return "wrong-lm-head-type";
        case loader_case::dense_lm_head:
            return "dense-lm-head";
        case loader_case::separate_q_weight:
            return "separate-q-weight";
        case loader_case::non_aligned_ffn:
            return "non-aligned-ffn";
    }
    return "unknown";
}

static std::string make_tmp_path(loader_case tc) {
    std::string       pattern = std::string("/tmp/llama-row4-loader-") + case_name(tc) + "-XXXXXX.gguf";
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

struct tiny_writer {
    gguf_context *                    gguf = nullptr;
    ggml_context *                    ggml = nullptr;
    std::vector<std::vector<uint8_t>> data;

    tiny_writer() {
        gguf                          = gguf_init_empty();
        const ggml_init_params params = {
            /*.mem_size   =*/64 * ggml_tensor_overhead(),
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        ggml = ggml_init(params);
        data.reserve(32);
    }

    ~tiny_writer() {
        ggml_free(ggml);
        gguf_free(gguf);
    }

    void add(const char * name, ggml_type type, std::initializer_list<int64_t> dims) {
        int64_t ne[4] = { 1, 1, 1, 1 };
        int     nd    = 0;
        for (int64_t dim : dims) {
            ne[nd++] = dim;
        }
        ggml_tensor * tensor = ggml_new_tensor(ggml, type, nd, ne);
        ggml_set_name(tensor, name);
        gguf_add_tensor(gguf, tensor);

        data.emplace_back(ggml_nbytes(tensor), uint8_t{ 0 });
        gguf_set_tensor_data(gguf, name, data.back().data());
    }

    std::vector<uint8_t> & tensor_data(const char * name) {
        const int64_t id = gguf_find_tensor(gguf, name);
        if (id < 0 || (size_t) id >= data.size()) {
            fprintf(stderr, "missing tiny Row4 tensor %s\n", name);
            exit(EXIT_FAILURE);
        }
        return data[(size_t) id];
    }

    void refresh(const char * name) { gguf_set_tensor_data(gguf, name, tensor_data(name).data()); }
};

static void fill_bf16(tiny_writer & writer, const char * name, float base, bool varying = false) {
    std::vector<uint8_t> &   bytes = writer.tensor_data(name);
    std::vector<ggml_bf16_t> values(bytes.size() / sizeof(ggml_bf16_t));
    for (size_t i = 0; i < values.size(); ++i) {
        const float value = varying ? base + (float) ((int) (i % 17) - 8) / 256.0f : base;
        values[i]         = ggml_fp32_to_bf16(value);
    }
    memcpy(bytes.data(), values.data(), bytes.size());
    writer.refresh(name);
}

static void fill_row4_codes(tiny_writer & writer, const char * name, uint8_t seed) {
    std::vector<uint8_t> & bytes = writer.tensor_data(name);
    for (size_t i = 0; i < bytes.size(); ++i) {
        const uint8_t low  = (uint8_t) ((3 * i + seed) & 15);
        const uint8_t high = (uint8_t) ((5 * i + seed + 7) & 15);
        bytes[i]           = (uint8_t) (low | (high << 4));
    }
    writer.refresh(name);
}

static void fill_w8_codes(tiny_writer & writer, const char * name) {
    std::vector<uint8_t> & bytes = writer.tensor_data(name);
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = (uint8_t) (int8_t) (((29 * i + 3) % 255) - 127);
    }
    writer.refresh(name);
}

static void fill_f32(tiny_writer & writer, const char * name, float positive, float negative) {
    std::vector<uint8_t> & bytes = writer.tensor_data(name);
    std::vector<float>     values(bytes.size() / sizeof(float));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = (i & 1) ? negative : positive;
    }
    memcpy(bytes.data(), values.data(), bytes.size());
    writer.refresh(name);
}

static void add_metadata(tiny_writer & writer, loader_case tc, int64_t n_ff) {
    gguf_context * gguf = writer.gguf;
    gguf_set_val_str(gguf, "general.architecture", "qwen3");
    gguf_set_val_str(gguf, "general.name", "tiny-qwen3-row4");
    gguf_set_val_u32(gguf, "general.file_type", LLAMA_FTYPE_MOSTLY_ROW4);
    gguf_set_val_u32(gguf, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(gguf, "general.alignment", tc == loader_case::bad_alignment ? 32 : 64);

    gguf_set_val_u32(gguf, "qwen3.context_length", 32);
    gguf_set_val_u32(gguf, "qwen3.embedding_length", 128);
    gguf_set_val_u32(gguf, "qwen3.block_count", 1);
    gguf_set_val_u32(gguf, "qwen3.feed_forward_length", (uint32_t) n_ff);
    gguf_set_val_u32(gguf, "qwen3.attention.head_count", 1);
    gguf_set_val_u32(gguf, "qwen3.attention.head_count_kv", 1);
    gguf_set_val_f32(gguf, "qwen3.attention.layer_norm_rms_epsilon", 1.0e-6f);
    gguf_set_val_u32(gguf, "qwen3.rope.dimension_count", 128);
    gguf_set_val_f32(gguf, "qwen3.rope.freq_base", 1000000.0f);
    gguf_set_val_u32(gguf, "qwen3.vocab_size", 128);
    gguf_set_val_str(gguf, "tokenizer.ggml.model", "no_vocab");

    gguf_set_val_u32(gguf, "row4.schema_version", 1);
    gguf_set_val_str(gguf, "row4.weight_layout",
                     tc == loader_case::bad_layout ? "m16k128_adjacent_v0" : "m16k128_split8_v1");
    gguf_set_val_str(gguf, "row4.codebook", "uv_axis_v1");
    gguf_set_val_str(gguf, "row4.numeric_profile", "bf16_a8_away_i32_bf16_v1");
    gguf_set_val_str(gguf, "row4.qkv_order", "q_k_v");
    gguf_set_val_str(gguf, "row4.ffn_order", "gate_up");
    gguf_set_val_str(gguf, "row4.lm_head_layout", "s8_m16k128_rowmajor_v1");
    if (tc == loader_case::incomplete_metadata) {
        gguf_remove_key(gguf, "row4.numeric_profile");
    }
}

static void add_row4_linear(tiny_writer & writer,
                            const char *  base,
                            int64_t       o,
                            int64_t       k,
                            uint8_t       seed,
                            loader_case   tc) {
    const std::string codes_name  = std::string(base) + ".row4.codes";
    const std::string scales_name = std::string(base) + ".row4.scales";

    ggml_type codes_type  = GGML_TYPE_ROW4_CODES;
    ggml_type scales_type = GGML_TYPE_BF16;
    int64_t   ne0         = 64;
    if (strcmp(base, "blk.0.attn_qkv") == 0) {
        if (tc == loader_case::wrong_qkv_codes_type) {
            codes_type = GGML_TYPE_I8;
        }
        if (tc == loader_case::wrong_qkv_codes_shape) {
            ne0 = 63;
        }
        if (tc == loader_case::wrong_qkv_scales_type) {
            scales_type = GGML_TYPE_F32;
        }
    }

    writer.add(codes_name.c_str(), codes_type, { ne0, 4, k / 128, o / 16 });
    fill_row4_codes(writer, codes_name.c_str(), seed);
    if (strcmp(base, "blk.0.attn_qkv") != 0 || tc != loader_case::missing_qkv_scales) {
        writer.add(scales_name.c_str(), scales_type, { o });
        if (scales_type == GGML_TYPE_BF16) {
            fill_bf16(writer, scales_name.c_str(), 0.03125f);
        } else {
            fill_f32(writer, scales_name.c_str(), 0.03125f, -0.03125f);
        }
    }
}

static std::string write_model(loader_case tc) {
    const std::string path = make_tmp_path(tc);
    tiny_writer       writer;

    const int64_t n_ff = tc == loader_case::non_aligned_ffn ? 192 : 128;
    add_metadata(writer, tc, n_ff);

    writer.add("token_embd.weight", GGML_TYPE_BF16, { 128, 128 });
    writer.add("output_norm.weight", GGML_TYPE_BF16, { 128 });
    writer.add("blk.0.attn_norm.weight", GGML_TYPE_BF16, { 128 });
    writer.add("blk.0.attn_q_norm.weight", GGML_TYPE_BF16, { 128 });
    writer.add("blk.0.attn_k_norm.weight", GGML_TYPE_BF16, { 128 });
    writer.add("blk.0.ffn_norm.weight", GGML_TYPE_BF16, { 128 });

    fill_bf16(writer, "token_embd.weight", 0.0f, true);
    for (const char * name : { "output_norm.weight", "blk.0.attn_norm.weight", "blk.0.attn_q_norm.weight",
                               "blk.0.attn_k_norm.weight", "blk.0.ffn_norm.weight" }) {
        fill_bf16(writer, name, 1.0f);
    }

    add_row4_linear(writer, "blk.0.attn_qkv", 384, 128, 1, tc);
    add_row4_linear(writer, "blk.0.attn_output", 128, 128, 3, tc);
    add_row4_linear(writer, "blk.0.ffn_gate_up", 2 * n_ff, 128, 5, tc);
    add_row4_linear(writer, "blk.0.ffn_down", 128, n_ff, 7, tc);

    if (tc != loader_case::missing_lm_head && tc != loader_case::missing_lm_head_codes) {
        const ggml_type output_type = tc == loader_case::wrong_lm_head_type ? GGML_TYPE_F32 : GGML_TYPE_I8;
        writer.add("output.w8.codes", output_type, { 128, 16, 1, 8 });
        if (output_type == GGML_TYPE_I8) {
            fill_w8_codes(writer, "output.w8.codes");
        } else {
            fill_f32(writer, "output.w8.codes", 1.0f, -1.0f);
        }
    }
    if (tc != loader_case::missing_lm_head && tc != loader_case::missing_lm_head_scales) {
        writer.add("output.w8.scales", GGML_TYPE_F32, { 128 });
        fill_f32(writer, "output.w8.scales", 0.00390625f, -0.00390625f);
    }

    if (tc == loader_case::dense_lm_head) {
        writer.add("output.weight", GGML_TYPE_BF16, { 128, 128 });
        fill_bf16(writer, "output.weight", 0.03125f, true);
    }
    if (tc == loader_case::separate_q_weight) {
        writer.add("blk.0.attn_q.weight", GGML_TYPE_BF16, { 128, 128 });
        fill_bf16(writer, "blk.0.attn_q.weight", 0.03125f, true);
    }

    if (!gguf_write_to_file(writer.gguf, path.c_str(), false)) {
        fprintf(stderr, "failed to write %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    return path;
}

static llama_model * load_model(const char * path, int n_gpu_layers) {
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers       = n_gpu_layers;
    params.use_mmap           = false;
    params.check_tensors      = true;
    return llama_model_load_from_file(path, params);
}

static bool test_loader_cases() {
    bool              ok      = true;
    const loader_case cases[] = {
        loader_case::valid,
        loader_case::incomplete_metadata,
        loader_case::bad_layout,
        loader_case::bad_alignment,
        loader_case::missing_qkv_scales,
        loader_case::wrong_qkv_codes_type,
        loader_case::wrong_qkv_codes_shape,
        loader_case::wrong_qkv_scales_type,
        loader_case::missing_lm_head,
        loader_case::missing_lm_head_codes,
        loader_case::missing_lm_head_scales,
        loader_case::wrong_lm_head_type,
        loader_case::dense_lm_head,
        loader_case::separate_q_weight,
        loader_case::non_aligned_ffn,
    };

    for (loader_case tc : cases) {
        const std::string path     = write_model(tc);
        llama_model *     model    = load_model(path.c_str(), 0);
        const bool        loaded   = model != nullptr;
        const bool        expected = tc == loader_case::valid;
        if (loaded != expected) {
            fprintf(stderr, "Row4 loader %s: expected %s, got %s\n", case_name(tc), expected ? "success" : "failure",
                    loaded ? "success" : "failure");
            ok = false;
        }
        if (loaded && (!model->row4_enabled || model->row4_schema_version != 1 || !model->output_w8.codes ||
                       !model->output_w8.scales)) {
            fprintf(stderr, "Row4 loader %s: valid model state was not populated\n", case_name(tc));
            ok = false;
        }
        llama_model_free(model);
        unlink(path.c_str());
    }

    printf("  Row4 strict metadata/tensor loader matrix - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

struct log_capture {
    std::string text;
};

static void capture_log(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    static_cast<log_capture *>(user_data)->text += text;
}

static bool test_lora_rejection() {
    const std::string path  = write_model(loader_case::valid);
    llama_model *     model = load_model(path.c_str(), 0);
    if (!model) {
        fprintf(stderr, "failed to load valid tiny Row4 model for LoRA rejection test\n");
        unlink(path.c_str());
        return false;
    }

    log_capture capture;
    llama_log_set(capture_log, &capture);
    llama_adapter_lora * adapter = llama_adapter_lora_init(model, "/this/adapter/must/not/be/opened.gguf");
    llama_log_set(nullptr, nullptr);

    const bool ok = adapter == nullptr && capture.text.find("does not support LoRA adapters") != std::string::npos &&
                    capture.text.find("failed to load lora adapter file") == std::string::npos;
    if (!ok) {
        fprintf(stderr, "Row4 LoRA rejection was not reported before adapter I/O: %s\n", capture.text.c_str());
    }
    llama_adapter_lora_free(adapter);
    llama_model_free(model);
    unlink(path.c_str());
    printf("  Row4 LoRA load-time rejection - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool run_decode(llama_model * model, std::vector<float> & logits, bool use_metal) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx                = 16;
    params.n_batch              = 8;
    params.n_ubatch             = 8;
    params.flash_attn_type      = use_metal ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context * ctx = llama_init_from_model(model, params);
    if (!ctx) {
        fprintf(stderr, "failed to initialize tiny Row4 context\n");
        return false;
    }

    const int32_t n_vocab        = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto          capture_logits = [&]() {
        const float * current = llama_get_logits_ith(ctx, -1);
        if (!current) {
            return false;
        }
        for (int32_t i = 0; i < n_vocab; ++i) {
            if (!std::isfinite(current[i])) {
                return false;
            }
        }
        logits.insert(logits.end(), current, current + n_vocab);
        return true;
    };

    llama_token prompt[] = { 1, 7, 23, 42 };
    bool        ok       = llama_decode(ctx, llama_batch_get_one(prompt, 4)) == 0 && capture_logits();
    llama_token next[]   = { 99, 17 };
    for (llama_token token : next) {
        ok = llama_decode(ctx, llama_batch_get_one(&token, 1)) == 0 && capture_logits() && ok;
    }
    llama_free(ctx);
    return ok;
}

static llama_context_params tiny_context_params() {
    llama_context_params params = llama_context_default_params();
    params.n_ctx                = 16;
    params.n_batch              = 8;
    params.n_ubatch             = 8;
    return params;
}

static bool has_metal_device() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (reg && strcmp(ggml_backend_reg_name(reg), "Metal") == 0) {
            return true;
        }
    }
    return false;
}

static bool test_context_contract() {
    const std::string path      = write_model(loader_case::valid);
    llama_model *     cpu_model = load_model(path.c_str(), 0);
    bool              ok        = cpu_model != nullptr;

    if (cpu_model) {
        llama_context_params params = tiny_context_params();
        params.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        params.type_k               = GGML_TYPE_F16;
        params.type_v               = GGML_TYPE_BF16;
        llama_context * ctx         = llama_init_from_model(cpu_model, params);
        if (ctx) {
            fprintf(stderr, "Row4 accepted a non-BF16 K cache\n");
            ok = false;
        }
        llama_free(ctx);
    }
    llama_model_free(cpu_model);

    if (has_metal_device()) {
        llama_model * metal_model = load_model(path.c_str(), 99);
        if (!metal_model) {
            ok = false;
        } else {
            llama_context_params params = tiny_context_params();
            params.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            params.type_k               = GGML_TYPE_BF16;
            params.type_v               = GGML_TYPE_BF16;
            llama_context * ctx         = llama_init_from_model(metal_model, params);
            if (ctx) {
                fprintf(stderr, "Row4 accepted full Metal placement with Flash Attention disabled\n");
                ok = false;
            }
            llama_free(ctx);
        }
        llama_model_free(metal_model);
    }

    unlink(path.c_str());
    printf("  Row4 BF16 KV/Metal Flash Attention context contract - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_full_graph() {
    const std::string path      = write_model(loader_case::valid);
    llama_model *     cpu_model = load_model(path.c_str(), 0);
    if (!cpu_model) {
        fprintf(stderr, "failed to load valid tiny Row4 CPU model\n");
        unlink(path.c_str());
        return false;
    }

    std::vector<float> cpu_first;
    std::vector<float> cpu_second;
    bool               ok =
        run_decode(cpu_model, cpu_first, false) && run_decode(cpu_model, cpu_second, false) && cpu_first == cpu_second;
    llama_model_free(cpu_model);
    if (!ok) {
        fprintf(stderr, "tiny Row4 CPU prefill/two-decode was not deterministic\n");
    }

    if (ok && has_metal_device()) {
        llama_model *      metal_model = load_model(path.c_str(), 99);
        std::vector<float> metal;
        if (!metal_model || !run_decode(metal_model, metal, true) || metal.size() != cpu_first.size()) {
            fprintf(stderr, "tiny Row4 Metal prefill/two-decode failed\n");
            ok = false;
        } else {
            double squared_error = 0.0;
            double energy        = 0.0;
            double max_abs       = 0.0;
            for (size_t i = 0; i < metal.size(); ++i) {
                const double diff = (double) metal[i] - cpu_first[i];
                squared_error += diff * diff;
                energy += (double) cpu_first[i] * cpu_first[i];
                max_abs = std::max(max_abs, fabs(diff));
            }
            const double nmse = energy > 0.0 ? squared_error / energy : squared_error;
            if (nmse > 1.0e-6 || max_abs > 1.0e-2) {
                fprintf(stderr, "tiny Row4 CPU/Metal logits mismatch: NMSE=%g max_abs=%g\n", nmse, max_abs);
                ok = false;
            }
        }
        llama_model_free(metal_model);
    } else if (ok) {
        printf("  Row4 tiny full graph Metal comparison: SKIP (Metal unavailable)\n");
    }

    unlink(path.c_str());
    printf("  Row4 tiny CPU%s prefill + two decode - %s\n", has_metal_device() ? "/Metal" : "", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    printf("========================================\n");
    printf("Qwen3 Row4 Loader Tests\n");
    printf("========================================\n");

    if (!row4_cpu_backend_available()) {
        printf("Row4 loader/full-graph tests: SKIP (native CPU Row4 path unavailable in this build)\n");
        return 0;
    }

    int failed = 0;
    failed += !test_loader_cases();
    failed += !test_lora_rejection();
    failed += !test_context_contract();
    failed += !test_full_graph();

    printf("========================================\n");
    printf("%s (%d failed)\n", failed == 0 ? "All Row4 loader tests PASSED" : "Row4 loader tests FAILED", failed);
    printf("========================================\n");
    return failed == 0 ? 0 : 1;
}
