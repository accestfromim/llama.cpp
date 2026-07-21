#include "speculative.h"

#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "common.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include <cstring>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

static const std::array<std::pair<const char *, common_speculative_type>, COMMON_SPECULATIVE_TYPE_COUNT> SPECULATIVE_TYPE_NAMES = {{
    { "none",          COMMON_SPECULATIVE_TYPE_NONE },
    { "draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE },
    { "ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE },
    { "ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K },
    { "ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V },
    { "ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD },
}};

const char * common_speculative_all_types_str() {
    return "none,draft-simple,ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod";
}

std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<enum common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        types.push_back(common_speculative_type_from_name(name));
    }

    return types;
}

enum common_speculative_type common_speculative_type_from_name(const std::string & name) {
    for (const auto & entry : SPECULATIVE_TYPE_NAMES) {
        if (name == entry.first) {
            return entry.second;
        }
    }

    throw std::invalid_argument(string_format("unknown speculative type: %s", name.c_str()));
}

std::string common_speculative_type_to_str(enum common_speculative_type type) {
    for (const auto & entry : SPECULATIVE_TYPE_NAMES) {
        if (type == entry.second) {
            return entry.first;
        }
    }

    return "unknown";
}

struct common_speculative_impl {
    common_speculative_type type;
    uint32_t n_seq;

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}
    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) {
        (void) seq_id;
        (void) prompt;
    }

    virtual void draft(struct common_speculative * spec, llama_seq_id seq_id, common_speculative_draft_params & params) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted) {
        (void) seq_id;
        (void) n_accepted;
    }
};

struct common_speculative {
    struct llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    struct llama_context * ctx_dft;
    struct common_sampler * smpl;

    llama_batch batch;
    bool has_draft_model_state = false;
    llama_tokens prompt_dft;
    bool vocab_dft_compatible = true; // whether retokenization is needed
    std::map<std::string, std::string> tgt_dft_replacements = {};

    common_params_speculative params;
    std::vector<common_speculative_draft_params> draft_params;
    std::vector<std::unique_ptr<common_speculative_impl>> impls;
};

struct common_speculative * common_speculative_init(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft) {
    auto * result = new common_speculative {
        /* .ctx_tgt    = */ ctx_tgt,
        /* .ctx_dft    = */ ctx_dft,
        /* .smpl       = */ nullptr,
        /* .batch      = */ llama_batch_init(llama_n_batch(ctx_dft), 0, 1),
        /* .has_draft_model_state = */ true,
        /* .prompt_dft = */ {},
        /* .vocab_dft_compatible = */ false,
    };

    // TODO: optimize or pass from outside?
#if 0
    {
        common_params_sampling params;
        params.no_perf = false;

        params.top_k = 40;
        params.top_p = 0.9;

        params.samplers = {
            COMMON_SAMPLER_TYPE_TOP_K,
            COMMON_SAMPLER_TYPE_TOP_P,
            COMMON_SAMPLER_TYPE_INFILL,
        };

        result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
    }
#else
    {
        common_params_sampling params;
        params.no_perf = false;

        params.top_k = 10;

        params.samplers = {
            COMMON_SAMPLER_TYPE_TOP_K,
        };

        result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
    }
#endif

    result->vocab_dft_compatible = common_speculative_are_compatible(ctx_tgt, ctx_dft);
    LOG_DBG("vocab_dft_compatible = %d\n", result->vocab_dft_compatible);

    return result;
}

void common_speculative_free(struct common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    if (spec->smpl) {
        common_sampler_free(spec->smpl);
    }

    if (spec->has_draft_model_state) {
        llama_batch_free(spec->batch);
    }

    delete spec;
}

bool common_speculative_are_compatible(
    const struct llama_context * ctx_tgt,
    const struct llama_context * ctx_dft) {
    const struct llama_model * model_tgt = llama_get_model(ctx_tgt);
    const struct llama_model * model_dft = llama_get_model(ctx_dft);

    const struct llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const struct llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_DBG("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_DBG("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_DBG("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(ctx_tgt, i).c_str(),
                        common_token_to_piece(ctx_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

void common_speculative_add_replacement_tgt_dft(
        struct common_speculative * spec,
        const char *source, const char *dest) {
    spec->tgt_dft_replacements[source] = dest;
}

static std::string replace_to_dft(
        struct common_speculative * spec,
        const std::string& input) {
    std::string result = input;
    for (const auto & pair : spec->tgt_dft_replacements) {
        size_t pos = result.find(pair.first);
        while (pos != std::string::npos) {
            result.replace(pos, pair.first.length(), pair.second);
            pos = result.find(pair.first, pos + pair.second.length());
        }
    }
    return result;
}

static std::string replace_to_tgt(
        struct common_speculative * spec,
        const std::string& input) {
    std::string result = input;
    for (const auto& pair : spec->tgt_dft_replacements) {
        size_t pos = result.find(pair.second);
        while (pos != std::string::npos) {
            result.replace(pos, pair.second.length(), pair.first);
            pos = result.find(pair.second, pos + pair.first.length());
        }
    }
    return result;
}

static bool common_speculative_has_type(
        const std::vector<enum common_speculative_type> & types,
        enum common_speculative_type type) {
    return std::find(types.begin(), types.end(), type) != types.end();
}

static int32_t common_speculative_draft_max(
        const common_speculative_draft_params & dp,
        int32_t fallback) {
    return dp.n_max >= 0 ? dp.n_max : fallback;
}

static void common_speculative_limit_result(
        const common_speculative_draft_params & dp) {
    if (dp.result == nullptr || dp.n_max < 0) {
        return;
    }

    if (dp.result->size() > (size_t) dp.n_max) {
        dp.result->resize(dp.n_max);
    }
}

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_speculative_impl_draft_simple(uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq) {}

    void draft(common_speculative * spec, llama_seq_id /*seq_id*/, common_speculative_draft_params & dp) override {
        if (spec->ctx_dft == nullptr || dp.prompt == nullptr || dp.result == nullptr) {
            return;
        }

        common_speculative_params params;
        params.n_draft = common_speculative_draft_max(dp, spec->params.n_max);
        params.n_reuse = llama_n_ctx(spec->ctx_dft) - params.n_draft;
        params.p_min   = spec->params.p_min;

        if (params.n_draft <= 0) {
            return;
        }

        *dp.result = common_speculative_gen_draft(spec, params, *dp.prompt, dp.id_last);
    }
};

struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq)
        , config {
            /* .size_ngram = */ params.ngram_simple.size_n,
            /* .size_mgram = */ params.ngram_simple.size_m,
        } {}

    void draft(common_speculative * /*spec*/, llama_seq_id /*seq_id*/, common_speculative_draft_params & dp) override {
        if (dp.prompt == nullptr || dp.result == nullptr) {
            return;
        }

        *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        common_speculative_limit_result(dp);
    }
};

struct common_speculative_impl_ngram_map : public common_speculative_impl {
    std::vector<common_ngram_map> maps;

    common_speculative_impl_ngram_map(
            enum common_speculative_type type,
            const common_params_speculative_ngram_map & params,
            uint32_t n_seq)
        : common_speculative_impl(type, n_seq) {
        const bool key_only = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;

        maps.reserve(n_seq);
        for (uint32_t i = 0; i < n_seq; ++i) {
            maps.emplace_back(params.size_n, params.size_m, key_only, params.min_hits);
        }
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) maps.size());

        common_ngram_map_begin(maps[seq_id], prompt);
    }

    void draft(common_speculative * /*spec*/, llama_seq_id seq_id, common_speculative_draft_params & dp) override {
        GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) maps.size());

        if (dp.prompt == nullptr || dp.result == nullptr) {
            return;
        }

        common_ngram_map_draft(maps[seq_id], *dp.prompt, dp.id_last, *dp.result);
        common_speculative_limit_result(dp);
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted) override {
        GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) maps.size());

        common_ngram_map_accept(maps[seq_id], n_accepted);
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;
    common_ngram_mod mod;

    struct seq_info {
        size_t i_last       = 0;
        size_t n_draft_last = 0;
        int    n_low        = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) sinfos.size());

        auto & sinfo = sinfos[seq_id];
        sinfo.i_last       = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        constexpr double f_thold = 0.25;
        const double f = (double) mod.get_used() / (double) mod.size();
        if (f > f_thold) {
            LOG_WRN("ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", f, f_thold);
            mod.reset();
            sinfo.i_last = 0;
        }
    }

    void draft(common_speculative * /*spec*/, llama_seq_id seq_id, common_speculative_draft_params & dp) override {
        GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) sinfos.size());

        if (dp.prompt == nullptr || dp.result == nullptr) {
            return;
        }

        auto & sinfo = sinfos[seq_id];
        auto & result = *dp.result;
        const auto & prompt = *dp.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        const size_t n = mod.get_n();
        if (cur_len < n) {
            return;
        }

        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        const int32_t n_max = common_speculative_draft_max(dp, params.n_max);
        if (n_max <= 0) {
            return;
        }

        result.resize(n + n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dp.id_last;

        for (int32_t i = 0; i < n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        sinfo.n_draft_last = result.size();
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted) override {
        GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) sinfos.size());

        auto & sinfo = sinfos[seq_id];

        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double) n_accepted / (double) sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    LOG_DBG("low acceptance streak (%d) - resetting ngram_mod\n", sinfo.n_low);
                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }
};

struct common_speculative * common_speculative_init(
        struct common_params_speculative & params,
        uint32_t n_seq) {
    auto types = params.types;
    if (types.empty()) {
        types.push_back(COMMON_SPECULATIVE_TYPE_NONE);
    }

    const bool has_draft_simple = common_speculative_has_type(types, COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);

    common_speculative * result = nullptr;
    if (has_draft_simple) {
        if (params.ctx_tgt == nullptr || params.ctx_dft == nullptr) {
            throw std::invalid_argument("--spec-type draft-simple requires initialized target and draft contexts");
        }

        result = common_speculative_init(params.ctx_tgt, params.ctx_dft);
        for (const auto & pair : params.replacements) {
            common_speculative_add_replacement_tgt_dft(result, pair.first.c_str(), pair.second.c_str());
        }
    } else {
        result = new common_speculative();
    }

    result->params = params;
    result->draft_params.resize(n_seq);

    for (const auto type : types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
                result->impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(n_seq));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                result->impls.push_back(std::make_unique<common_speculative_impl_ngram_simple>(params, n_seq));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                result->impls.push_back(std::make_unique<common_speculative_impl_ngram_map>(
                            type, params.ngram_map_k, n_seq));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                result->impls.push_back(std::make_unique<common_speculative_impl_ngram_map>(
                            type, params.ngram_map_k4v, n_seq));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                result->impls.push_back(std::make_unique<common_speculative_impl_ngram_mod>(params, n_seq));
                break;
            case COMMON_SPECULATIVE_TYPE_COUNT:
                throw std::invalid_argument("invalid speculative type");
        }
    }

    return result;
}

void common_speculative_begin(
        struct common_speculative * spec,
        llama_seq_id seq_id,
        const llama_tokens & prompt) {
    GGML_ASSERT(spec != nullptr);
    GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) spec->draft_params.size());

    for (auto & impl : spec->impls) {
        impl->begin(seq_id, prompt);
    }
}

common_speculative_draft_params & common_speculative_get_draft_params(
        struct common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec != nullptr);
    GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) spec->draft_params.size());

    return spec->draft_params[seq_id];
}

void common_speculative_draft(struct common_speculative * spec) {
    GGML_ASSERT(spec != nullptr);

    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) spec->draft_params.size(); ++seq_id) {
        auto & dp = spec->draft_params[seq_id];

        if (!dp.drafting) {
            continue;
        }

        if (dp.result != nullptr) {
            dp.result->clear();
        }

        for (auto & impl : spec->impls) {
            impl->draft(spec, seq_id, dp);

            if (dp.result != nullptr && !dp.result->empty()) {
                break;
            }
        }

        dp.drafting = false;
    }
}

void common_speculative_accept(
        struct common_speculative * spec,
        llama_seq_id seq_id,
        uint16_t n_accepted) {
    GGML_ASSERT(spec != nullptr);
    GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) spec->draft_params.size());

    for (auto & impl : spec->impls) {
        impl->accept(seq_id, n_accepted);
    }
}


llama_tokens common_speculative_gen_draft(
        struct common_speculative * spec,
        struct common_speculative_params params,
        const llama_tokens & prompt_tgt_main_model, // specified in target model vocab
        llama_token id_last) {
    auto & batch  = spec->batch;
    auto & ctx_tgt = spec->ctx_tgt;
    auto & ctx_dft = spec->ctx_dft;
    auto & smpl   = spec->smpl;
    auto & prompt_dft = spec->prompt_dft;

    auto * mem_dft = llama_get_memory(ctx_dft);

    int reuse_i = 0;
    int reuse_n = 0;

    const int n_ctx = llama_n_ctx(ctx_dft) - params.n_draft;

    llama_tokens prompt_tgt_draft_model;
    if (!spec->vocab_dft_compatible) {
        std::string text;
        text = common_detokenize(ctx_tgt, prompt_tgt_main_model, true);
        text = replace_to_dft(spec, text);
        LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());
        prompt_tgt_draft_model = common_tokenize(ctx_dft, text, false, true);

        // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
        const auto * model_tgt = llama_get_model(ctx_tgt);
        const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

        int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
        GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");
        text.resize(-n_chars);
        llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
        text = replace_to_dft(spec, text);

        LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
        id_last = common_tokenize(ctx_dft, text, false, true)[0];
    }
    // prompt_tgt's tokens will always be compatible with ctx_dft
    const llama_tokens &prompt_tgt =
        spec->vocab_dft_compatible ? prompt_tgt_main_model : prompt_tgt_draft_model;

    const int i_start = std::max<int>(0, (int) prompt_tgt.size() - n_ctx);

    // reuse as much as possible from the old draft context
    // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
    for (int i = 0; i < (int) prompt_dft.size(); ++i) {
        int cur = 0;
        while (i_start + cur < (int) prompt_tgt.size() &&
               i       + cur < (int) prompt_dft.size() &&
               prompt_tgt[i_start + cur] == prompt_dft[i + cur]) {
            cur++;
        }

        if ((cur >= params.n_reuse || n_ctx >= (int) prompt_tgt.size()) && cur > reuse_n) {
            reuse_i = i;
            reuse_n = cur;
        }
    }

    LOG_DBG("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt_dft.size());

    llama_tokens result;
    result.reserve(params.n_draft);

    if (reuse_n == 0) {
        llama_memory_clear(mem_dft, false);
        prompt_dft.clear();
    } else {
        // this happens when a previous draft has been discarded (for example, due to being too small), but the
        // target model agreed with it. in this case, we simply pass back the previous results to save compute
        if (reuse_i + reuse_n < (int) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
            for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                result.push_back(prompt_dft[i]);

                if (params.n_draft <= (int) result.size()) {
                    break;
                }
            }

            return result;
        }

        if (reuse_i > 0) {
            llama_memory_seq_rm (mem_dft, 0, 0, reuse_i);
            llama_memory_seq_add(mem_dft, 0, reuse_i, -1, -reuse_i);

            prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
        }

        if (reuse_n < (int) prompt_dft.size()) {
            llama_memory_seq_rm (mem_dft, 0, reuse_n, -1);
            prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
        }
    }

    // prepare a batch to evaluate any new tokens in the prompt
    common_batch_clear(batch);

    for (size_t i = i_start + reuse_n; i < prompt_tgt.size(); ++i) {
        //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_tgt[i]);
        common_batch_add(batch, prompt_tgt[i], i - i_start, { 0 }, false);

        prompt_dft.push_back(prompt_tgt[i]);
    }

    // we should rarely end-up here during normal decoding
    if (batch.n_tokens > 0) {
        //LOG_DBG("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());

        llama_decode(ctx_dft, batch);
    }

    const llama_pos n_past = prompt_dft.size();

    LOG_DBG("%s: n_past = %d\n", __func__, n_past);

    common_batch_clear(batch);
    common_batch_add  (batch, id_last, n_past, { 0 }, true);

    prompt_dft.push_back(id_last);

    LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

    llama_decode(ctx_dft, batch);

    common_sampler_reset(smpl);

    // sample n_draft tokens from the draft model
    for (int i = 0; i < params.n_draft; ++i) {
        common_batch_clear(batch);

        common_sampler_sample(smpl, ctx_dft, 0, true);

        const auto * cur_p = common_sampler_get_candidates(smpl, true);

        for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
            LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                    k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
        }

        // add drafted token for each sequence
        const llama_token id = cur_p->data[0].id;

        common_sampler_accept(smpl, id, true);

        result.push_back(id);

        if (params.n_draft <= (int) result.size()) {
            break;
        }

        // only collect very high-confidence draft tokens
        if (cur_p->data[0].p < params.p_min) {
            break;
        }

        common_batch_add(batch, id, n_past + i + 1, { 0 }, true);

        // evaluate the drafted tokens on the draft model
        llama_decode(ctx_dft, batch);

        prompt_dft.push_back(id);
    }

    if (!spec->vocab_dft_compatible) {
        std::string detokenized = common_detokenize(ctx_dft, result, true);
        detokenized = replace_to_tgt(spec, detokenized);
        LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
        result = common_tokenize(ctx_tgt, detokenized, false, true);
        if (result.size() > (size_t)params.n_draft) {
            result.resize(params.n_draft);
        }
    }
    return result;
}
