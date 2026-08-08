#include "arg.h"
#include "chat.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool has_spec_type_arg(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--spec-type") == 0 ||
            std::strncmp(argv[i], "--spec-type=", 12) == 0) {
            return true;
        }
    }

    return false;
}

static bool speculative_types_are_default(const std::vector<common_speculative_type> & types) {
    return types.empty() || (types.size() == 1 && types[0] == COMMON_SPECULATIVE_TYPE_NONE);
}

static bool speculative_has_type(
        const std::vector<common_speculative_type> & types,
        common_speculative_type type) {
    return std::find(types.begin(), types.end(), type) != types.end();
}

int main(int argc, char ** argv) {
    common_params params;
    const bool has_user_spec_type = has_spec_type_arg(argc, argv);

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init();

    if (!has_user_spec_type && speculative_types_are_default(params.speculative.types)) {
        params.speculative.types = {
            params.speculative.model.path.empty()
                ? COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE
                : COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE
        };
    }

    const bool has_draft_simple = speculative_has_type(params.speculative.types, COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);

    if (has_draft_simple && params.speculative.model.path.empty()) {
        LOG_ERR("%s: --model-draft is required for --spec-type draft-simple\n", __func__);
        return 1;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    common_init_result llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt.model.get();
    ctx_tgt   = llama_init_tgt.context.get();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    auto       chat_templates    = common_chat_templates_init(model_tgt, params.chat_template);
    const bool has_chat_template = common_chat_templates_was_explicit(chat_templates.get());

    if (params.conversation_mode == COMMON_CONVERSATION_MODE_AUTO) {
        if (has_chat_template) {
            LOG_INF("%s: chat template is available, enabling conversation mode (disable it with -no-cnv)\n", __func__);
            params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;
        } else {
            params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;
        }
    }

    std::string prompt = params.prompt;

    if (params.conversation_mode == COMMON_CONVERSATION_MODE_ENABLED && params.enable_chat_template) {
        if (!has_chat_template) {
            LOG_WRN("%s: chat template is not available; using the default chat template\n", __func__);
        }

        common_chat_templates_inputs inputs;
        inputs.use_jinja             = params.use_jinja;
        inputs.add_generation_prompt = !params.prompt.empty();
        inputs.reasoning_format      = params.reasoning_format;
        inputs.enable_thinking       = params.reasoning_budget != 0;
        inputs.chat_template_kwargs  = params.default_template_kwargs;

        if (!params.system_prompt.empty()) {
            common_chat_msg message;
            message.role    = "system";
            message.content = params.system_prompt;
            inputs.messages.push_back(std::move(message));
        }

        if (!params.prompt.empty()) {
            common_chat_msg message;
            message.role    = "user";
            message.content = params.prompt;
            inputs.messages.push_back(std::move(message));
        }

        prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
        LOG_INF("%s: applied chat template to the prompt\n", __func__);
    }

    common_init_result llama_init_dft;

    if (has_draft_simple) {
        common_params params_dft = params;

        params_dft.devices      = params.speculative.devices;
        params_dft.model        = params.speculative.model;
        params_dft.n_ctx        = params.speculative.n_ctx;
        params_dft.n_batch      = params.speculative.n_ctx > 0 ? params.speculative.n_ctx : params.n_batch;
        params_dft.n_gpu_layers = params.speculative.n_gpu_layers;

        if (params.speculative.cpuparams.n_threads > 0) {
            params_dft.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
        }

        params_dft.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
        params_dft.tensor_buft_overrides     = params.speculative.tensor_buft_overrides;

        llama_init_dft = common_init_from_params(params_dft);
        ctx_dft = llama_init_dft.context.get();

        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to initialize draft model\n", __func__);
            return 1;
        }

        if (!common_speculative_are_compatible(ctx_tgt, ctx_dft)) {
            LOG_INF("the draft model '%s' is not compatible with the target model '%s'. tokens will be translated between the draft and target models.\n", params.speculative.model.path.c_str(), params.model.path.c_str());
        }
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    // how many tokens to draft each time
    int n_draft     = params.speculative.n_max;
    int n_draft_min = params.speculative.n_min;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    // eval the prompt
    llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1));

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    // init the speculator
    params.speculative.ctx_tgt = ctx_tgt;
    params.speculative.ctx_dft = ctx_dft;

    struct common_speculative * spec = common_speculative_init(params.speculative, 1);
    common_speculative_begin(spec, 0, prompt_tgt);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    while (params.n_predict < 0 || n_predict < params.n_predict) {
        int n_draft_step = n_draft;
        if (params.n_predict >= 0) {
            n_draft_step = std::min(n_draft_step, params.n_predict - n_predict - 1);
            n_draft_step = std::max(n_draft_step, 0);
        }

        // optionally, generate draft tokens that can be appended to the target batch
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        llama_tokens draft;
        common_speculative_get_draft_params(spec, 0) = {
            /* .drafting = */ true,
            /* .n_max    = */ n_draft_step,
            /* .n_past   = */ n_past,
            /* .id_last  = */ id_last,
            /* .prompt   = */ &prompt_tgt,
            /* .result   = */ &draft,
        };
        common_speculative_draft(spec);

        //LOG_DBG("draft: %s\n", string_from(ctx_dft, draft).c_str());

        // always have a token to evaluate from before - id_last
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, id_last, n_past++, { 0 }, true);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            // do not waste time on small drafts
            if (draft.size() < (size_t) n_draft_min) {
                draft.clear();
            }

            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
            }

            //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());

            llama_decode(ctx_tgt, batch_tgt);
        }

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        const auto ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft);

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        common_speculative_accept(spec, 0, ids.size() - 1);

        n_past    += ids.size() - 1;
        n_drafted += draft.size(); // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
        }

        if ((params.n_predict >= 0 && n_predict >= params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", n_drafted > 0 ? 100.0f * n_accept / n_drafted : 0.0f);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");

    if (ctx_dft != nullptr) {
        llama_perf_context_print(ctx_dft);
    } else {
        LOG_INF("no draft model used\n");
    }

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    common_speculative_free(spec);
    llama_batch_free(batch_tgt);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
