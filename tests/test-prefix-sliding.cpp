#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void decode_positions(llama_context * ctx, int first, int last, int seq_first, int n_seq, llama_token token) {
    for (int pos = first; pos < last; ++pos) {
        llama_batch batch = llama_batch_init(n_seq, 0, 1);
        batch.n_tokens    = n_seq;
        for (int i = 0; i < n_seq; ++i) {
            batch.token[i]     = token;
            batch.pos[i]       = pos;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = seq_first + i;
            batch.logits[i]    = false;
        }
        const int result = llama_decode(ctx, batch);
        llama_batch_free(batch);
        require(result == 0, "decode failed");
    }
}

static void decode_prompt(llama_context * ctx, int count, int seq_id, llama_token token) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    batch.n_tokens    = count;
    for (int pos = 0; pos < count; ++pos) {
        batch.token[pos]     = token;
        batch.pos[pos]       = pos;
        batch.n_seq_id[pos]  = 1;
        batch.seq_id[pos][0] = seq_id;
        batch.logits[pos]    = false;
    }
    const int result = llama_decode(ctx, batch);
    llama_batch_free(batch);
    require(result == 0, "chunked prompt decode failed");
}

static size_t save_sequence(llama_context * ctx, llama_seq_id seq_id, std::vector<uint8_t> & state) {
    state.resize(llama_state_seq_get_size(ctx, seq_id));
    const size_t size = llama_state_seq_get_data(ctx, state.data(), state.size(), seq_id);
    require(size > 0 && size <= state.size(), "sequence state save failed");
    state.resize(size);
    return size;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL {bf16|turbo}\n", argv[0]);
        return 2;
    }

    const std::string mode = argv[2];
    require(mode == "bf16" || mode == "turbo", "invalid KV mode");

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 999;
    llama_model * model        = llama_model_load_from_file(argv[1], mparams);
    require(model != nullptr, "model load failed");

    llama_context_params cparams      = llama_context_default_params();
    cparams.n_ctx                     = 2048;
    cparams.n_batch                   = 128;
    cparams.n_ubatch                  = 32;
    cparams.n_seq_max                 = 4;
    cparams.type_k                    = mode == "bf16" ? GGML_TYPE_BF16 : GGML_TYPE_TURBO4_0;
    cparams.type_v                    = mode == "bf16" ? GGML_TYPE_BF16 : GGML_TYPE_TURBO3_0;
    cparams.flash_attn_type           = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.offload_kqv               = true;
    cparams.kv_unified                = false;
    cparams.prefix_sliding_window     = 64;
    cparams.prefix_sliding_prefix_cap = 128;

#ifndef _WIN32
    if (mode == "turbo") {
        unsetenv("TURBO_K_MEAN_WARMUP");
        unsetenv("TURBO_KV_BOUNDARY_BF16_LAYERS");
        const auto production_state_size = [&](bool disable_centering) {
            if (disable_centering) {
                setenv("TURBO_K_MEAN_CENTER", "0", 1);
            } else {
                unsetenv("TURBO_K_MEAN_CENTER");
            }

            llama_context_params production      = cparams;
            production.n_ctx                     = 256;
            production.n_seq_max                 = 1;
            production.prefix_sliding_window     = LLAMA_ROW4_PREFIX_SLIDING_PRODUCTION_WINDOW;
            production.prefix_sliding_prefix_cap = LLAMA_ROW4_PREFIX_SLIDING_PRODUCTION_PREFIX_CAP;

            llama_context * production_ctx = llama_init_from_model(model, production);
            require(production_ctx != nullptr, "production profile context creation failed");
            require(llama_memory_seq_set_prefix(llama_get_memory(production_ctx), 0, 1),
                    "production profile prefix set failed");

            const llama_vocab * production_vocab = llama_model_get_vocab(model);
            llama_token         production_token = llama_vocab_bos(production_vocab);
            if (production_token == LLAMA_TOKEN_NULL) {
                production_token = 0;
            }
            decode_prompt(production_ctx, 1, 0, production_token);
            const size_t state_size = llama_state_seq_get_size(production_ctx, 0);
            llama_free(production_ctx);
            return state_size;
        };

        const size_t centered_state_size   = production_state_size(false);
        const size_t uncentered_state_size = production_state_size(true);
        unsetenv("TURBO_K_MEAN_CENTER");
        require(centered_state_size > uncentered_state_size, "production profile did not enable K mean centering");
    }
#endif

    const auto expect_context_failure = [&](llama_context_params invalid, const char * message) {
        llama_context * invalid_ctx = llama_init_from_model(model, invalid);
        if (invalid_ctx) {
            llama_free(invalid_ctx);
            throw std::runtime_error(message);
        }
    };
    {
        llama_context_params invalid      = cparams;
        invalid.prefix_sliding_prefix_cap = 0;
        expect_context_failure(invalid, "unpaired prefix settings were accepted");
        invalid        = cparams;
        invalid.type_k = GGML_TYPE_F16;
        invalid.type_v = GGML_TYPE_F16;
        expect_context_failure(invalid, "unsupported KV types were accepted");
        invalid                 = cparams;
        invalid.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        expect_context_failure(invalid, "disabled Flash Attention was accepted");
        invalid             = cparams;
        invalid.offload_kqv = false;
        expect_context_failure(invalid, "disabled KQV offload was accepted");
        invalid            = cparams;
        invalid.kv_unified = true;
        expect_context_failure(invalid, "unified KV was accepted");
        invalid                = cparams;
        invalid.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
        expect_context_failure(invalid, "non-causal attention was accepted");
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    require(ctx != nullptr, "context creation failed");
    llama_memory_t mem = llama_get_memory(ctx);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token         token = llama_vocab_bos(vocab);
    if (token == LLAMA_TOKEN_NULL) {
        token = 0;
    }

    for (int seq = 0; seq < 4; ++seq) {
        require(llama_memory_seq_set_prefix(mem, seq, 80), "prefix set failed");
        decode_prompt(ctx, 80, seq, token);
    }
    require(!llama_memory_seq_set_prefix(mem, 4, 80), "out-of-range sequence prefix was accepted");

    decode_positions(ctx, 80, 257, 0, 4, token);
    std::vector<uint8_t> state_at_256;
    const size_t         size_at_256 = save_sequence(ctx, 0, state_at_256);

    decode_positions(ctx, 257, 512, 0, 4, token);
    require(llama_memory_seq_pos_min(mem, 0) == 0, "prefix was overwritten");
    require(llama_memory_seq_pos_max(mem, 0) == 511, "absolute positions were renumbered");

    std::vector<uint8_t> state_at_511;
    const size_t         size_at_511 = save_sequence(ctx, 0, state_at_511);
    std::fprintf(stderr, "bounded state sizes: at 256=%zu, at 511=%zu\n", size_at_256, size_at_511);
    require(size_at_511 == size_at_256, "active KV state grew after the physical cache saturated");

    std::vector<uint8_t> whole_state(llama_state_get_size(ctx));
    const size_t         whole_size = llama_state_get_data(ctx, whole_state.data(), whole_state.size());
    require(whole_size > 0 && whole_size <= whole_state.size(), "whole state save failed");
    whole_state.resize(whole_size);
    llama_memory_clear(mem, false);
    require(llama_state_set_data(ctx, whole_state.data(), whole_state.size()) == whole_state.size(),
            "whole state restore failed");
    for (int seq = 0; seq < 4; ++seq) {
        require(llama_memory_seq_get_prefix(mem, seq) == 80, "whole state restore lost prefix metadata");
        require(llama_memory_seq_pos_max(mem, seq) == 511, "whole state restore lost a stream");
    }

    const auto expect_state_failure = [&](llama_context_params mismatched, const char * message) {
        llama_context * mismatched_ctx = llama_init_from_model(model, mismatched);
        require(mismatched_ctx != nullptr, "mismatched context creation failed");
        const size_t restored = llama_state_set_data(mismatched_ctx, whole_state.data(), whole_state.size());
        llama_free(mismatched_ctx);
        require(restored == 0, message);
    };
    {
        llama_context_params mismatched  = cparams;
        mismatched.prefix_sliding_window = 32;
        expect_state_failure(mismatched, "state with a mismatched window was accepted");
        mismatched                           = cparams;
        mismatched.prefix_sliding_prefix_cap = 96;
        expect_state_failure(mismatched, "state with a mismatched prefix cap was accepted");
        if (mode == "bf16") {
            mismatched        = cparams;
            mismatched.type_k = GGML_TYPE_TURBO4_0;
            mismatched.type_v = GGML_TYPE_TURBO3_0;
            expect_state_failure(mismatched, "state with mismatched KV types was accepted");
        }
    }

    llama_memory_seq_cp(mem, 0, 1, -1, -1);
    require(llama_memory_seq_get_prefix(mem, 1) == 80, "sequence copy lost prefix metadata");
    decode_positions(ctx, 512, 513, 1, 1, token);

    require(llama_state_seq_set_data(ctx, state_at_511.data(), state_at_511.size(), 2) == state_at_511.size(),
            "sequence state restore failed");
    require(llama_memory_seq_get_prefix(mem, 2) == 80, "state restore lost prefix metadata");
    require(llama_memory_seq_pos_max(mem, 2) == 511, "state restore lost absolute positions");

    require(llama_memory_seq_rm(mem, 2, 8, 10), "sequence removal failed");
    require(llama_memory_seq_get_prefix(mem, 2) == 8, "partial prefix removal did not truncate the prefix");

    require(llama_memory_seq_rm(mem, 3, 0, -1), "full sequence removal failed");
    require(llama_memory_seq_get_prefix(mem, 3) == -1, "full sequence removal retained prefix metadata");
    require(llama_memory_seq_pos_max(mem, 3) == -1, "full sequence removal retained KV cells");

    llama_memory_seq_keep(mem, 0);
    require(llama_memory_seq_get_prefix(mem, 0) == 80, "sequence keep changed the retained prefix");
    for (int seq = 1; seq < 4; ++seq) {
        require(llama_memory_seq_get_prefix(mem, seq) == -1, "sequence keep retained stale prefix metadata");
        require(llama_memory_seq_pos_max(mem, seq) == -1, "sequence keep retained stale KV cells");
    }

    llama_memory_clear(mem, false);
    for (int seq = 0; seq < 4; ++seq) {
        require(llama_memory_seq_get_prefix(mem, seq) == -1, "clear retained prefix metadata");
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    std::printf("prefix sliding lifecycle: OK (%s, bounded sequence state %zu bytes)\n", mode.c_str(), size_at_511);
    return 0;
}
