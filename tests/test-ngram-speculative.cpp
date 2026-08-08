#include "ngram-map.h"
#include "ngram-mod.h"

#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        std::exit(1);
    }
}

static void test_ngram_simple() {
    const llama_tokens               history = { 99, 1, 2, 7, 8, 3, 4, 1 };
    const common_ngram_simple_config config  = { 2, 2 };

    const llama_tokens draft = common_ngram_simple_draft(config, history, 2);
    require(draft == llama_tokens({ 7, 8 }), "ngram-simple expected continuation");
    require(common_ngram_simple_draft(config, history, 9).empty(), "ngram-simple unexpected match");
}

static void test_ngram_map(bool key_only) {
    const llama_tokens history = { 99, 1, 2, 7, 8, 3, 4, 1 };
    common_ngram_map   map(2, 2, key_only, 1);
    common_ngram_map_begin(map, history);

    llama_tokens draft;
    common_ngram_map_draft(map, history, 2, draft);
    require(draft == llama_tokens({ 7, 8 }), "ngram-map expected continuation");
    common_ngram_map_accept(map, 2);
}

static void test_ngram_mod() {
    common_ngram_mod                mod(2, 1024);
    const common_ngram_mod::entry_t sequence[] = { 1, 2, 7 };

    require(mod.get(sequence) == common_ngram_mod::EMPTY, "ngram-mod initial state");
    mod.add(sequence);
    require(mod.get(sequence) == 7, "ngram-mod lookup");
    require(mod.get_used() == 1, "ngram-mod occupancy");
    mod.reset();
    require(mod.get(sequence) == common_ngram_mod::EMPTY, "ngram-mod reset");
}

int main() {
    test_ngram_simple();
    test_ngram_map(true);
    test_ngram_map(false);
    test_ngram_mod();
    std::puts("N-Gram speculative tests passed");
    return 0;
}
