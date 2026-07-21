#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list of all supported speculative decoding types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

struct common_speculative_params {
    int n_draft = 16;  // max drafted tokens
    int n_reuse = 256;

    float p_min = 0.75f; // min probability required to accept a token in the draft
};

struct common_speculative_draft_params {
    bool drafting = false;

    // overrides the configured max draft length when >= 0
    int32_t n_max = -1;

    llama_pos   n_past  = 0;
    llama_token id_last = LLAMA_TOKEN_NULL;

    const llama_tokens * prompt = nullptr;
    llama_tokens       * result = nullptr;
};

struct common_speculative * common_speculative_init(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft
);

struct common_speculative * common_speculative_init(
        struct common_params_speculative & params,
        uint32_t n_seq);

void common_speculative_free(struct common_speculative * spec);

bool common_speculative_are_compatible(
        const struct llama_context * ctx_tgt,
        const struct llama_context * ctx_dft);

void common_speculative_add_replacement_tgt_dft(
        struct common_speculative * spec,
        const char *source, const char *dest);

void common_speculative_begin(
        struct common_speculative * spec,
        llama_seq_id seq_id,
        const llama_tokens & prompt);

common_speculative_draft_params & common_speculative_get_draft_params(
        struct common_speculative * spec,
        llama_seq_id seq_id);

void common_speculative_draft(struct common_speculative * spec);

void common_speculative_accept(
        struct common_speculative * spec,
        llama_seq_id seq_id,
        uint16_t n_accepted);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_gen_draft(
               struct common_speculative * spec,
        struct common_speculative_params   params,
                      const llama_tokens & prompt,
                             llama_token   id_last);
