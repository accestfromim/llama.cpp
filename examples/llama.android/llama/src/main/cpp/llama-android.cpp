#include <android/log.h>
#include <chrono>
#include <jni.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <utility>
#include <unistd.h>
#include <vector>
#include "llama.h"
#include "common.h"
#include "chat.h"
#include "ggml-cpu.h"

// Write C++ code here.
//
// Do not forget to dynamically load the C++ library into your application.
//
// For instance,
//
// In MainActivity.java:
//    static {
//       System.loadLibrary("llama-android");
//    }
//
// Or, in MainActivity.kt:
//    companion object {
//      init {
//         System.loadLibrary("llama-android")
//      }
//    }

#define TAG_LLAMA_ANDROID "LLAMA_ANDROID"
#define TAG_LLAMA_COMPLEX "LLAMA_COMPLEX"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG_LLAMA_ANDROID, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG_LLAMA_ANDROID, __VA_ARGS__)

jclass la_int_var;
jmethodID la_int_var_value;
jmethodID la_int_var_inc;

std::string cached_token_chars;

struct e2e_bench_aggregate {
    double generated_tokens_avg = 0.0;
    double prefill_ms_avg = 0.0;
    double first_token_ms_avg = 0.0;
    double decode_ms_per_token_avg = 0.0;
    double tok_s_avg = 0.0;
    double total_ms_avg = 0.0;
};

struct mobile_profile {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    bool disable_ifairy_lut;
    bool enable_ifairy_vecdot_act_tensor;
    int generation_priority;
    int batch_priority;
    int affinity_profile;
    const char * name;
};

struct runtime_overrides {
    int n_ctx = -1;
    int n_batch = -1;
    int n_ubatch = -1;
    int n_threads = -1;
    int n_threads_batch = -1;
    int disable_ifairy_lut = -1;
    int affinity_profile = -1;
    int enable_ifairy_vecdot_act_tensor = -1;
    int generation_priority = -99;
    int batch_priority = -99;
    int sched_debug = -1;
    int opencl_supports_debug = -1;
    int force_cpu = -1;
};

static runtime_overrides g_runtime_overrides;

static void enable_android_opencl_ifairy64_gate() {
#ifdef GGML_USE_OPENCL
    setenv("GGML_OPENCL_IFAIRY64", "1", 1);
    LOGi("Runtime toggle: GGML_OPENCL_IFAIRY64=1 for OpenCL build");
    setenv("LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2", "1", 1);
    LOGi("Runtime toggle: LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2=1 for OpenCL build");
#endif
}

static void configure_opencl_ifairy64_mul_mat_impl(const char * impl) {
#ifdef GGML_USE_OPENCL
    if (impl == nullptr || impl[0] == '\0' || strcmp(impl, "default") == 0) {
        unsetenv("GGML_OPENCL_IFAIRY64_MUL_MAT_IMPL");
        LOGi("Runtime toggle: GGML_OPENCL_IFAIRY64_MUL_MAT_IMPL unset");
        return;
    }

    if (
            strcmp(impl, "gemm") == 0 ||
            strcmp(impl, "gemv2") == 0 ||
            strcmp(impl, "gemv4") == 0 ||
            strcmp(impl, "gemv8") == 0 ||
            strcmp(impl, "gemv16") == 0 ||
            strcmp(impl, "direct") == 0 ||
            strcmp(impl, "lut16") == 0 ||
            strcmp(impl, "lut32") == 0 ||
            strcmp(impl, "lut64") == 0 ||
            strcmp(impl, "lutglobal16") == 0 ||
            strcmp(impl, "lutglobal32") == 0 ||
            strcmp(impl, "lutglobal64") == 0) {
        setenv("GGML_OPENCL_IFAIRY64_MUL_MAT_IMPL", impl, 1);
        LOGi("Runtime toggle: GGML_OPENCL_IFAIRY64_MUL_MAT_IMPL=%s", impl);
        return;
    }

    LOGe("Ignoring unsupported GGML_OPENCL_IFAIRY64_MUL_MAT_IMPL=%s", impl);
#else
    (void) impl;
#endif
}

static void configure_opencl_ifairy64_wide_linear_w2_impl(const char * impl) {
#ifdef GGML_USE_OPENCL
    if (impl == nullptr || impl[0] == '\0' || strcmp(impl, "default") == 0 || strcmp(impl, "q16") == 0) {
        unsetenv("GGML_OPENCL_IFAIRY64_WIDE_LINEAR_W2_IMPL");
        LOGi("Runtime toggle: GGML_OPENCL_IFAIRY64_WIDE_LINEAR_W2_IMPL unset (q16)");
        return;
    }

    if (
            strcmp(impl, "q16dot8") == 0 ||
            strcmp(impl, "dot8") == 0 ||
            strcmp(impl, "lutlocal") == 0 ||
            strcmp(impl, "lut") == 0 ||
            strcmp(impl, "lutglobal") == 0 ||
            strcmp(impl, "lutglobal16") == 0 ||
            strcmp(impl, "lutglobal32") == 0 ||
            strcmp(impl, "lutglobal64") == 0) {
        setenv("GGML_OPENCL_IFAIRY64_WIDE_LINEAR_W2_IMPL", impl, 1);
        LOGi("Runtime toggle: GGML_OPENCL_IFAIRY64_WIDE_LINEAR_W2_IMPL=%s", impl);
        return;
    }

    LOGe("Ignoring unsupported GGML_OPENCL_IFAIRY64_WIDE_LINEAR_W2_IMPL=%s", impl);
#else
    (void) impl;
#endif
}

struct effective_runtime_config {
    int n_ctx = -1;
    int n_batch = -1;
    int n_ubatch = -1;
    int n_threads = -1;
    int n_threads_batch = -1;
    int disable_ifairy_lut = -1;
    int affinity_profile = -1;
    int enable_ifairy_vecdot_act_tensor = -1;
    int generation_priority = -99;
    int batch_priority = -99;
};

static effective_runtime_config g_effective_runtime_config;

enum affinity_profile {
    AFFINITY_PROFILE_NONE = 0,
    AFFINITY_PROFILE_BIG_CORES = 1,
    AFFINITY_PROFILE_BIG_GEN_DEFAULT_BATCH = 2,
    AFFINITY_PROFILE_BIG_GEN_LITTLE_BATCH = 3,
    AFFINITY_PROFILE_STABLE_BIG_CORES = 4,
};

struct cpu_topology {
    std::vector<int> big_cores;
    std::vector<int> little_cores;
    int max_capacity = 0;
    bool detected = false;
};

struct android_llama_context {
    llama_context * context = nullptr;
    ggml_threadpool_t threadpool = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;
    common_chat_templates_ptr chat_templates;
    std::vector<common_chat_msg> chat_history;
    std::string pending_user_content;
    std::string pending_assistant_content;
    int n_past = 0;
    bool pending_chat = false;
};

static android_llama_context * from_context_handle(jlong handle) {
    return reinterpret_cast<android_llama_context *>(handle);
}

static llama_context * unwrap_context(jlong handle) {
    const auto wrapper = from_context_handle(handle);
    return wrapper != nullptr ? wrapper->context : nullptr;
}

static void clear_chat_state(android_llama_context * wrapper) {
    if (!wrapper) {
        return;
    }

    wrapper->chat_history.clear();
    wrapper->pending_user_content.clear();
    wrapper->pending_assistant_content.clear();
    wrapper->pending_chat = false;
    wrapper->n_past = 0;
}

static void commit_pending_chat(android_llama_context * wrapper) {
    if (!wrapper || !wrapper->pending_chat) {
        return;
    }

    common_chat_msg user_msg;
    user_msg.role = "user";
    user_msg.content = std::move(wrapper->pending_user_content);
    wrapper->chat_history.push_back(std::move(user_msg));

    common_chat_msg assistant_msg;
    assistant_msg.role = "assistant";
    assistant_msg.content = std::move(wrapper->pending_assistant_content);
    wrapper->chat_history.push_back(std::move(assistant_msg));

    wrapper->pending_user_content.clear();
    wrapper->pending_assistant_content.clear();
    wrapper->pending_chat = false;

    LOGi("Committed chat turn: history_messages=%zu n_past=%d",
            wrapper->chat_history.size(),
            wrapper->n_past);
}

static void log_native_memory_snapshot(const char * stage);
bool is_valid_utf8(const char * string);

static const char * affinity_profile_name(int profile) {
    switch (profile) {
        case AFFINITY_PROFILE_NONE:
            return "none";
        case AFFINITY_PROFILE_BIG_CORES:
            return "big-cores";
        case AFFINITY_PROFILE_BIG_GEN_DEFAULT_BATCH:
            return "big-gen-default-batch";
        case AFFINITY_PROFILE_BIG_GEN_LITTLE_BATCH:
            return "big-gen-little-batch";
        case AFFINITY_PROFILE_STABLE_BIG_CORES:
            return "stable-big-cores";
        default:
            return "default";
    }
}

static const char * runtime_toggle_name(int value) {
    if (value < 0) {
        return "default";
    }
    return value == 0 ? "0" : "1";
}

static int default_mobile_threads(int cpu_count) {
    return std::max(1, std::min(6, cpu_count));
}

static mobile_profile choose_mobile_profile(const llama_model * model) {
    (void) model;

    return {
            /*.n_ctx    =*/ 2048,
            /*.n_batch  =*/ 2048,
            /*.n_ubatch =*/ 512,
            /*.disable_ifairy_lut =*/ false,
            /*.enable_ifairy_vecdot_act_tensor =*/ false,
            /*.generation_priority =*/ GGML_SCHED_PRIO_NORMAL,
            /*.batch_priority =*/ GGML_SCHED_PRIO_NORMAL,
            /*.affinity_profile =*/ AFFINITY_PROFILE_NONE,
            /*.name     =*/ "uniform-default",
    };
}

static void apply_mobile_runtime_toggles(const mobile_profile & profile) {
    enable_android_opencl_ifairy64_gate();

    const bool disable_ifairy_lut = g_runtime_overrides.disable_ifairy_lut >= 0
            ? g_runtime_overrides.disable_ifairy_lut != 0
            : profile.disable_ifairy_lut;

    if (disable_ifairy_lut) {
        setenv("GGML_IFAIRY_LUT", "0", 1);
        LOGi("Runtime toggle: GGML_IFAIRY_LUT=0 for profile `%s`", profile.name);
    } else {
        setenv("GGML_IFAIRY_LUT", "1", 1);
        LOGi("Runtime toggle: GGML_IFAIRY_LUT=1 for profile `%s`", profile.name);
    }

    const bool enable_ifairy_vecdot_act_tensor = g_runtime_overrides.enable_ifairy_vecdot_act_tensor >= 0
            ? g_runtime_overrides.enable_ifairy_vecdot_act_tensor != 0
            : profile.enable_ifairy_vecdot_act_tensor;

    if (enable_ifairy_vecdot_act_tensor) {
            setenv("GGML_IFAIRY_VEC_DOT_ACT_TENSOR", "1", 1);
            LOGi("Runtime toggle: GGML_IFAIRY_VEC_DOT_ACT_TENSOR=1 for profile `%s`", profile.name);
    } else {
        if (g_runtime_overrides.enable_ifairy_vecdot_act_tensor >= 0) {
            setenv("GGML_IFAIRY_VEC_DOT_ACT_TENSOR", "0", 1);
            LOGi("Runtime toggle: GGML_IFAIRY_VEC_DOT_ACT_TENSOR=0 for profile `%s`", profile.name);
        } else {
            unsetenv("GGML_IFAIRY_VEC_DOT_ACT_TENSOR");
            LOGi("Runtime toggle: GGML_IFAIRY_VEC_DOT_ACT_TENSOR unset for profile `%s`", profile.name);
        }
    }
}

static int resolve_priority_override(int override_value, int profile_value) {
    return override_value != -99 ? override_value : profile_value;
}

static std::vector<llama_token> build_bench_prompt_tokens(
        llama_context * context,
        const int prompt_tokens
) {
    std::vector<llama_token> tokens;
    tokens.reserve(std::max(prompt_tokens, 1));

    const std::string seed = "Summarize the following sentence in a concise way. ";
    bool add_special = true;

    while ((int) tokens.size() < prompt_tokens) {
        const auto chunk = common_tokenize(context, seed, add_special, false);
        if (chunk.empty()) {
            break;
        }

        const int remaining = prompt_tokens - (int) tokens.size();
        const int take = std::min(remaining, (int) chunk.size());
        tokens.insert(tokens.end(), chunk.begin(), chunk.begin() + take);
        add_special = false;
    }

    if ((int) tokens.size() < prompt_tokens) {
        const llama_model * model = llama_get_model(context);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const llama_token fallback = llama_vocab_bos(vocab);
        while ((int) tokens.size() < prompt_tokens) {
            tokens.push_back(fallback);
        }
    }

    return tokens;
}

static int completion_init_tokens(
        llama_context * context,
        llama_batch * batch,
        const std::vector<llama_token> & tokens_list,
        const int n_len,
        const int n_past
) {
    cached_token_chars.clear();

    const int n_ctx = llama_n_ctx(context);
    const size_t n_kv_req = static_cast<size_t>(n_past) + tokens_list.size() + static_cast<size_t>(n_len);

    LOGi("n_len = %d, n_ctx = %d, n_past = %d, n_kv_req = %zu", n_len, n_ctx, n_past, n_kv_req);
    log_native_memory_snapshot("jni-before-completion-init");

    if (n_kv_req > (size_t) n_ctx) {
        LOGe("error: n_kv_req > n_ctx, the required KV cache size is not big enough");
    }

    const uint32_t n_batch = llama_n_batch(context);
    const size_t n_tokens = tokens_list.size();
    for (size_t start = 0; start < n_tokens; start += n_batch) {
        const size_t end = std::min(start + static_cast<size_t>(n_batch), n_tokens);

        common_batch_clear(*batch);
        for (size_t i = start; i < end; ++i) {
            common_batch_add(*batch, tokens_list[i], static_cast<llama_pos>(n_past + i), { 0 }, false);
        }

        batch->logits[batch->n_tokens - 1] = (end == n_tokens);

        if (llama_decode(context, *batch) != 0) {
            LOGe("llama_decode() failed while ingesting prompt chunk [%zu, %zu)", start, end);
            break;
        }
    }

    log_native_memory_snapshot("jni-after-completion-init");
    return n_past + (int) tokens_list.size();
}

static bool completion_loop_step(
        llama_context * context,
        llama_batch * batch,
        llama_sampler * sampler,
        const int n_len,
        int & n_cur,
        std::string & out_piece
) {
    out_piece.clear();

    const auto model = llama_get_model(context);
    const auto vocab = llama_model_get_vocab(model);
    const auto new_token_id = llama_sampler_sample(sampler, context, -1);

    if (llama_vocab_is_eog(vocab, new_token_id) || n_cur == n_len) {
        return false;
    }

    const auto new_token_chars = common_token_to_piece(context, new_token_id);
    cached_token_chars += new_token_chars;

    if (is_valid_utf8(cached_token_chars.c_str())) {
        out_piece = cached_token_chars;
        cached_token_chars.clear();
    }

    common_batch_clear(*batch);
    common_batch_add(*batch, new_token_id, n_cur, { 0 }, true);

    ++n_cur;

    if (llama_decode(context, *batch) != 0) {
        LOGe("llama_decode() returned null");
    }

    return true;
}

static std::string trim_copy(const std::string & value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }

    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

static std::string read_proc_value(const char * path, const char * key) {
    std::ifstream input(path);
    if (!input.is_open()) {
        return "unavailable";
    }

    std::string line;
    const std::string prefix = std::string(key) + ":";
    while (std::getline(input, line)) {
        if (line.rfind(prefix, 0) == 0) {
            return trim_copy(line.substr(prefix.size()));
        }
    }

    return "n/a";
}

static int read_int_file(const std::string & path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        return -1;
    }

    int value = -1;
    input >> value;
    return input.fail() ? -1 : value;
}

static cpu_topology detect_cpu_topology() {
    cpu_topology topology;
    const int cpu_count = std::max(1, (int) sysconf(_SC_NPROCESSORS_ONLN));

    struct cpu_metric {
        int index;
        int capacity;
    };

    std::vector<cpu_metric> metrics;
    metrics.reserve(cpu_count);

    for (int cpu = 0; cpu < cpu_count; ++cpu) {
        const std::string cpu_base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
        int capacity = read_int_file(cpu_base + "/cpu_capacity");
        if (capacity < 0) {
            capacity = read_int_file(cpu_base + "/cpufreq/cpuinfo_max_freq");
        }
        if (capacity < 0) {
            continue;
        }
        metrics.push_back({ cpu, capacity });
        topology.max_capacity = std::max(topology.max_capacity, capacity);
    }

    if (metrics.empty() || topology.max_capacity <= 0) {
        return topology;
    }

    const int threshold = std::max(1, (topology.max_capacity * 80) / 100);
    for (const auto & metric : metrics) {
        if (metric.capacity >= threshold) {
            topology.big_cores.push_back(metric.index);
        } else {
            topology.little_cores.push_back(metric.index);
        }
    }

    topology.detected = !topology.big_cores.empty();
    return topology;
}

static void clear_cpumask(bool (&mask)[GGML_MAX_N_THREADS]) {
    std::fill(std::begin(mask), std::end(mask), false);
}

static void apply_cpumask(bool (&mask)[GGML_MAX_N_THREADS], const std::vector<int> & cores) {
    clear_cpumask(mask);
    for (const int core : cores) {
        if (core >= 0 && core < GGML_MAX_N_THREADS) {
            mask[core] = true;
        }
    }
}

static std::string join_cpu_list(const std::vector<int> & cores) {
    if (cores.empty()) {
        return "none";
    }

    std::ostringstream builder;
    for (size_t i = 0; i < cores.size(); ++i) {
        if (i > 0) {
            builder << ",";
        }
        builder << cores[i];
    }
    return builder.str();
}

static void configure_threadpool_params(
        ggml_threadpool_params & tpp,
        int n_threads,
        const std::vector<int> & cores,
        bool strict_cpu,
        int priority) {
    ggml_threadpool_params_init(&tpp, n_threads);
    if (priority >= GGML_SCHED_PRIO_LOW && priority <= GGML_SCHED_PRIO_REALTIME) {
        tpp.prio = (ggml_sched_priority) priority;
    }
    if (!cores.empty()) {
        apply_cpumask(tpp.cpumask, cores);
        tpp.strict_cpu = strict_cpu;
    }
}

static bool create_affinity_threadpools(
        llama_context * context,
        int n_threads,
        int n_threads_batch,
        int generation_priority,
        int batch_priority,
        int profile,
        const cpu_topology & topology,
        android_llama_context & wrapper) {
    if (profile == AFFINITY_PROFILE_NONE) {
        LOGi("Affinity profile: none");
        return true;
    }

    if (!topology.detected) {
        LOGi("Affinity profile `%s` requested but CPU topology detection failed; falling back to default scheduling",
                affinity_profile_name(profile));
        return true;
    }

    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        LOGe("No CPU backend found for affinity threadpool setup");
        return false;
    }

    auto * reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto * ggml_threadpool_free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
    if (!ggml_threadpool_new_fn || !ggml_threadpool_free_fn) {
        LOGe("CPU backend does not expose threadpool factory functions");
        return false;
    }

    std::vector<int> generation_cores;
    std::vector<int> batch_cores;
    bool batch_default = false;

    switch (profile) {
        case AFFINITY_PROFILE_BIG_CORES:
            generation_cores = topology.big_cores;
            batch_cores = topology.big_cores;
            break;
        case AFFINITY_PROFILE_BIG_GEN_DEFAULT_BATCH:
            generation_cores = topology.big_cores;
            batch_default = true;
            break;
        case AFFINITY_PROFILE_BIG_GEN_LITTLE_BATCH:
            generation_cores = topology.big_cores;
            batch_cores = topology.little_cores;
            break;
        case AFFINITY_PROFILE_STABLE_BIG_CORES:
            generation_cores = topology.big_cores;
            if (generation_cores.size() > 1) {
                generation_cores.pop_back();
            }
            batch_cores = generation_cores;
            break;
        default:
            LOGi("Unknown affinity profile `%d`; falling back to default scheduling", profile);
            return true;
    }

    if (generation_cores.empty()) {
        LOGi("Affinity profile `%s` has no generation cores after topology detection; falling back to default scheduling",
                affinity_profile_name(profile));
        return true;
    }

    ggml_threadpool_params tpp;
    ggml_threadpool_params tpp_batch;
    configure_threadpool_params(tpp, n_threads, generation_cores, true, generation_priority);
    if (batch_default) {
        ggml_threadpool_params_init(&tpp_batch, n_threads_batch);
        if (batch_priority >= GGML_SCHED_PRIO_LOW && batch_priority <= GGML_SCHED_PRIO_REALTIME) {
            tpp_batch.prio = (ggml_sched_priority) batch_priority;
        }
    } else {
        configure_threadpool_params(tpp_batch, n_threads_batch, batch_cores, true, batch_priority);
    }

    LOGi("Affinity profile `%s`: generation cores = [%s], batch cores = [%s], generation prio = %d, batch prio = %d",
            affinity_profile_name(profile),
            join_cpu_list(generation_cores).c_str(),
            batch_default ? "default" : join_cpu_list(batch_cores).c_str(),
            tpp.prio,
            tpp_batch.prio);

    ggml_threadpool_t threadpool_batch = nullptr;
    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!threadpool_batch) {
            LOGe("Failed to create batch threadpool for affinity profile `%s`", affinity_profile_name(profile));
            return false;
        }
        tpp.paused = true;
    }

    ggml_threadpool_t threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        if (threadpool_batch) {
            ggml_threadpool_free_fn(threadpool_batch);
        }
        LOGe("Failed to create generation threadpool for affinity profile `%s`", affinity_profile_name(profile));
        return false;
    }

    wrapper.threadpool = threadpool;
    wrapper.threadpool_batch = threadpool_batch;
    llama_attach_threadpool(context, threadpool, threadpool_batch);
    return true;
}

static void log_native_memory_snapshot(const char * stage) {
    const std::string vm_rss = read_proc_value("/proc/self/status", "VmRSS");
    const std::string vm_hwm = read_proc_value("/proc/self/status", "VmHWM");
    const std::string vm_size = read_proc_value("/proc/self/status", "VmSize");
    const std::string rss_anon = read_proc_value("/proc/self/status", "RssAnon");
    const std::string rss_file = read_proc_value("/proc/self/status", "RssFile");
    const std::string mem_available = read_proc_value("/proc/meminfo", "MemAvailable");
    const std::string mem_free = read_proc_value("/proc/meminfo", "MemFree");

    LOGi(
            "MEM-NATIVE[%s] VmRSS=%s VmHWM=%s VmSize=%s RssAnon=%s RssFile=%s MemAvailable=%s MemFree=%s",
            stage,
            vm_rss.c_str(),
            vm_hwm.c_str(),
            vm_size.c_str(),
            rss_anon.c_str(),
            rss_file.c_str(),
            mem_available.c_str(),
            mem_free.c_str());
}

bool is_valid_utf8(const char * string) {
    if (!string) {
        return true;
    }

    const unsigned char * bytes = (const unsigned char *)string;
    int num;

    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            // U+0000 to U+007F
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // U+0080 to U+07FF
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // U+0800 to U+FFFF
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // U+10000 to U+10FFFF
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }

    return true;
}

static void log_callback(ggml_log_level level, const char * fmt, void * data) {
    (void) data;

    const int android_level =
            level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR :
            level == GGML_LOG_LEVEL_WARN  ? ANDROID_LOG_WARN  :
            level == GGML_LOG_LEVEL_INFO  ? ANDROID_LOG_INFO  :
            g_runtime_overrides.sched_debug >= 0 ? ANDROID_LOG_INFO :
                                            ANDROID_LOG_DEBUG;

    __android_log_write(android_level, TAG_LLAMA_COMPLEX, fmt);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_load_1model(JNIEnv *env, jobject, jstring filename) {
    llama_model_params model_params = llama_model_default_params();
    if (g_runtime_overrides.force_cpu > 0) {
        model_params.n_gpu_layers = 0;
        LOGi("Runtime override: forcing CPU model load with n_gpu_layers=0");
    }

    auto path_to_model = env->GetStringUTFChars(filename, 0);
    LOGi("Loading model from %s", path_to_model);
    log_native_memory_snapshot("jni-before-load-model");

    auto model = llama_model_load_from_file(path_to_model, model_params);
    env->ReleaseStringUTFChars(filename, path_to_model);

    if (!model) {
        LOGe("load_model() failed");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "load_model() failed");
        return 0;
    }

    LOGi("Model loaded: size = %.2f MiB, params = %.2f B",
            llama_model_size(model) / 1024.0 / 1024.0,
            llama_model_n_params(model) / 1e9);
    log_native_memory_snapshot("jni-after-load-model");

    return reinterpret_cast<jlong>(model);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_free_1model(JNIEnv *, jobject, jlong model) {
    llama_model_free(reinterpret_cast<llama_model *>(model));
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_new_1context(JNIEnv *env, jobject, jlong jmodel) {
    auto model = reinterpret_cast<llama_model *>(jmodel);

    if (!model) {
        LOGe("new_context(): model cannot be null");
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Model cannot be null");
        return 0;
    }

    const int cpu_count = std::max(1, (int) sysconf(_SC_NPROCESSORS_ONLN));
    const int default_threads_batch = default_mobile_threads(cpu_count);
    const int default_threads = default_mobile_threads(cpu_count);
    const int n_threads_batch = g_runtime_overrides.n_threads_batch > 0
            ? g_runtime_overrides.n_threads_batch
            : default_threads_batch;
    const int n_threads = g_runtime_overrides.n_threads > 0
            ? g_runtime_overrides.n_threads
            : default_threads;
    const mobile_profile profile = choose_mobile_profile(model);
    const int generation_priority = resolve_priority_override(g_runtime_overrides.generation_priority, profile.generation_priority);
    const int batch_priority = resolve_priority_override(g_runtime_overrides.batch_priority, profile.batch_priority);
    const int affinity_profile = g_runtime_overrides.affinity_profile >= 0
            ? g_runtime_overrides.affinity_profile
            : profile.affinity_profile;
    const int effective_disable_ifairy_lut = g_runtime_overrides.disable_ifairy_lut >= 0
            ? g_runtime_overrides.disable_ifairy_lut
            : (profile.disable_ifairy_lut ? 1 : 0);
    const int effective_enable_ifairy_vecdot_act_tensor = g_runtime_overrides.enable_ifairy_vecdot_act_tensor >= 0
            ? g_runtime_overrides.enable_ifairy_vecdot_act_tensor
            : (profile.enable_ifairy_vecdot_act_tensor ? 1 : 0);
    LOGi("Using %d generation threads, %d batch threads, affinity profile `%s`, generation prio %d, batch prio %d",
            n_threads,
            n_threads_batch,
            affinity_profile_name(affinity_profile),
            generation_priority,
            batch_priority);

    apply_mobile_runtime_toggles(profile);
    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx           = g_runtime_overrides.n_ctx > 0 ? g_runtime_overrides.n_ctx : profile.n_ctx;
    ctx_params.n_batch         = g_runtime_overrides.n_batch > 0 ? g_runtime_overrides.n_batch : profile.n_batch;
    ctx_params.n_ubatch        = g_runtime_overrides.n_ubatch > 0 ? g_runtime_overrides.n_ubatch : profile.n_ubatch;
    ctx_params.n_threads       = n_threads;
    ctx_params.n_threads_batch = n_threads_batch;
    ctx_params.no_perf         = true;

    g_effective_runtime_config = {
            /*.n_ctx =*/ (int) ctx_params.n_ctx,
            /*.n_batch =*/ (int) ctx_params.n_batch,
            /*.n_ubatch =*/ (int) ctx_params.n_ubatch,
            /*.n_threads =*/ n_threads,
            /*.n_threads_batch =*/ n_threads_batch,
            /*.disable_ifairy_lut =*/ effective_disable_ifairy_lut,
            /*.affinity_profile =*/ affinity_profile,
            /*.enable_ifairy_vecdot_act_tensor =*/ effective_enable_ifairy_vecdot_act_tensor,
            /*.generation_priority =*/ generation_priority,
            /*.batch_priority =*/ batch_priority,
    };

    LOGi("Creating context with profile `%s`: params = %.2f B, n_ctx = %u, n_batch = %u, n_ubatch = %u, type_k = %d, type_v = %d",
            profile.name,
            llama_model_n_params(model) / 1e9,
            ctx_params.n_ctx,
            ctx_params.n_batch,
            ctx_params.n_ubatch,
            (int) ctx_params.type_k,
            (int) ctx_params.type_v);
    LOGi("Effective runtime: threads=%d batch_threads=%d affinity=%s(%d) ifairy_lut=%s ifairy_vecdot_act_tensor=%s generation_prio=%d batch_prio=%d",
            g_effective_runtime_config.n_threads,
            g_effective_runtime_config.n_threads_batch,
            affinity_profile_name(g_effective_runtime_config.affinity_profile),
            g_effective_runtime_config.affinity_profile,
            g_effective_runtime_config.disable_ifairy_lut == 0 ? "enabled" : "disabled",
            runtime_toggle_name(g_effective_runtime_config.enable_ifairy_vecdot_act_tensor),
            g_effective_runtime_config.generation_priority,
            g_effective_runtime_config.batch_priority);
    log_native_memory_snapshot("jni-before-new-context");

    llama_context * context = llama_new_context_with_model(model, ctx_params);

    if (!context) {
        LOGe("llama_new_context_with_model() returned null)");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "llama_new_context_with_model() returned null)");
        return 0;
    }

    android_llama_context * wrapper = new android_llama_context {
        /*.context =*/ context,
        /*.threadpool =*/ nullptr,
        /*.threadpool_batch =*/ nullptr,
        /*.chat_templates =*/ nullptr,
        /*.chat_history =*/ {},
        /*.pending_user_content =*/ {},
        /*.pending_assistant_content =*/ {},
        /*.n_past =*/ 0,
        /*.pending_chat =*/ false,
    };

    try {
        wrapper->chat_templates = common_chat_templates_init(model, "");
        const bool explicit_template = common_chat_templates_was_explicit(wrapper->chat_templates.get());
        const char * source = common_chat_templates_source(wrapper->chat_templates.get());
        const int source_bytes = source != nullptr ? (int) strlen(source) : 0;
        LOGi("Chat template initialized: source=%s bytes=%d preview=`%.*s`",
                explicit_template ? "model-tokenizer.chat_template" : "common-fallback",
                source_bytes,
                std::min(source_bytes, 160),
                source != nullptr ? source : "");
    } catch (const std::exception & e) {
        LOGe("Chat template initialization failed: %s", e.what());
    }

    const cpu_topology topology = detect_cpu_topology();
    if (!create_affinity_threadpools(context, n_threads, n_threads_batch, generation_priority, batch_priority, affinity_profile, topology, *wrapper)) {
        llama_free(context);
        delete wrapper;
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "failed to configure affinity threadpools");
        return 0;
    }

    LOGi("Context created: n_ctx = %u, n_batch = %u, n_ubatch = %u, generation prio = %d, batch prio = %d",
            llama_n_ctx(context),
            llama_n_batch(context),
            llama_n_ubatch(context),
            generation_priority,
            batch_priority);
    log_native_memory_snapshot("jni-after-new-context");

    return reinterpret_cast<jlong>(wrapper);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_configure_1runtime(
        JNIEnv *,
        jobject,
        jint n_ctx,
        jint n_batch,
        jint n_ubatch,
        jint n_threads,
        jint n_threads_batch,
        jint disable_ifairy_lut,
        jint affinity_profile,
        jint enable_ifairy_vecdot_act_tensor,
        jint generation_priority,
        jint batch_priority,
        jint sched_debug,
        jint opencl_supports_debug,
        jint force_cpu
) {
    g_runtime_overrides.n_ctx = n_ctx;
    g_runtime_overrides.n_batch = n_batch;
    g_runtime_overrides.n_ubatch = n_ubatch;
    g_runtime_overrides.n_threads = n_threads;
    g_runtime_overrides.n_threads_batch = n_threads_batch;
    g_runtime_overrides.disable_ifairy_lut = disable_ifairy_lut;
    g_runtime_overrides.affinity_profile = affinity_profile;
    g_runtime_overrides.enable_ifairy_vecdot_act_tensor = enable_ifairy_vecdot_act_tensor;
    g_runtime_overrides.generation_priority = generation_priority;
    g_runtime_overrides.batch_priority = batch_priority;
    g_runtime_overrides.sched_debug = sched_debug;
    g_runtime_overrides.opencl_supports_debug = opencl_supports_debug;
    g_runtime_overrides.force_cpu = force_cpu;

    if (g_runtime_overrides.sched_debug >= 0) {
        char value[16];
        snprintf(value, sizeof(value), "%d", g_runtime_overrides.sched_debug);
        setenv("GGML_SCHED_DEBUG", value, 1);
    } else {
        unsetenv("GGML_SCHED_DEBUG");
    }

    if (g_runtime_overrides.opencl_supports_debug >= 0) {
        char value[16];
        snprintf(value, sizeof(value), "%d", g_runtime_overrides.opencl_supports_debug);
        setenv("GGML_OPENCL_SUPPORTS_DEBUG", value, 1);
        const char * debug_file = "/sdcard/Android/data/com.example.llama/files/bench/opencl_supports_debug.log";
        setenv("GGML_OPENCL_SUPPORTS_DEBUG_FILE", debug_file, 1);
        if (g_runtime_overrides.opencl_supports_debug != 0) {
            unlink(debug_file);
        }
    } else {
        unsetenv("GGML_OPENCL_SUPPORTS_DEBUG");
        unsetenv("GGML_OPENCL_SUPPORTS_DEBUG_FILE");
    }

    LOGi("Configured runtime overrides: n_ctx=%d n_batch=%d n_ubatch=%d n_threads=%d n_threads_batch=%d disable_ifairy_lut=%d affinity_profile=%d (%s) enable_ifairy_vecdot_act_tensor=%d generation_priority=%d batch_priority=%d sched_debug=%d opencl_supports_debug=%d force_cpu=%d",
            g_runtime_overrides.n_ctx,
            g_runtime_overrides.n_batch,
            g_runtime_overrides.n_ubatch,
            g_runtime_overrides.n_threads,
            g_runtime_overrides.n_threads_batch,
            g_runtime_overrides.disable_ifairy_lut,
            g_runtime_overrides.affinity_profile,
            affinity_profile_name(g_runtime_overrides.affinity_profile),
            g_runtime_overrides.enable_ifairy_vecdot_act_tensor,
            g_runtime_overrides.generation_priority,
            g_runtime_overrides.batch_priority,
            g_runtime_overrides.sched_debug,
            g_runtime_overrides.opencl_supports_debug,
            g_runtime_overrides.force_cpu);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_configure_1opencl_1mul_1mat_1impl(
        JNIEnv * env,
        jobject,
        jstring jimpl
) {
    if (jimpl == nullptr) {
        configure_opencl_ifairy64_mul_mat_impl(nullptr);
        return;
    }

    const char * impl = env->GetStringUTFChars(jimpl, nullptr);
    configure_opencl_ifairy64_mul_mat_impl(impl);
    env->ReleaseStringUTFChars(jimpl, impl);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_configure_1opencl_1wide_1linear_1w2_1impl(
        JNIEnv * env,
        jobject,
        jstring jimpl
) {
    if (jimpl == nullptr) {
        configure_opencl_ifairy64_wide_linear_w2_impl(nullptr);
        return;
    }

    const char * impl = env->GetStringUTFChars(jimpl, nullptr);
    configure_opencl_ifairy64_wide_linear_w2_impl(impl);
    env->ReleaseStringUTFChars(jimpl, impl);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_free_1context(JNIEnv *, jobject, jlong context) {
    auto * wrapper = from_context_handle(context);
    if (!wrapper) {
        return;
    }

    llama_detach_threadpool(wrapper->context);
    llama_free(wrapper->context);

    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto * reg = cpu_dev ? ggml_backend_dev_backend_reg(cpu_dev) : nullptr;
    auto * ggml_threadpool_free_fn = reg
            ? (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free")
            : nullptr;
    if (ggml_threadpool_free_fn) {
        if (wrapper->threadpool_batch) {
            ggml_threadpool_free_fn(wrapper->threadpool_batch);
        }
        if (wrapper->threadpool) {
            ggml_threadpool_free_fn(wrapper->threadpool);
        }
    }

    delete wrapper;
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_backend_1free(JNIEnv *, jobject) {
    llama_backend_free();
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_log_1to_1android(JNIEnv *, jobject) {
    llama_log_set(log_callback, NULL);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_bench_1model(
        JNIEnv *env,
        jobject,
        jlong context_pointer,
        jlong model_pointer,
        jlong batch_pointer,
        jint pp,
        jint tg,
        jint pl,
        jint nr
        ) {
    auto pp_avg = 0.0;
    auto tg_avg = 0.0;
    auto pp_std = 0.0;
    auto tg_std = 0.0;

    const auto context = unwrap_context(context_pointer);
    const auto model = reinterpret_cast<llama_model *>(model_pointer);
    const auto batch = reinterpret_cast<llama_batch *>(batch_pointer);

    const int n_ctx = llama_n_ctx(context);
    const int n_batch = (int) llama_n_batch(context);

    LOGi("n_ctx = %d, n_batch = %d", n_ctx, n_batch);

    int i, j;
    int nri;
    for (nri = 0; nri < nr; nri++) {
        LOGi("Benchmark prompt processing (pp)");

        const int n_tokens = pp;
        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp_start = ggml_time_us();
        for (int start = 0; start < n_tokens; start += n_batch) {
            const int end = std::min(start + n_batch, n_tokens);

            common_batch_clear(*batch);
            for (i = start; i < end; i++) {
                common_batch_add(*batch, 0, i, { 0 }, false);
            }

            batch->logits[batch->n_tokens - 1] = (end == n_tokens);
            if (llama_decode(context, *batch) != 0) {
                LOGi("llama_decode() failed during prompt processing chunk [%d, %d)", start, end);
                break;
            }
        }
        const auto t_pp_end = ggml_time_us();

        // bench text generation

        LOGi("Benchmark text generation (tg)");

        llama_memory_clear(llama_get_memory(context), false);
        const auto t_tg_start = ggml_time_us();
        for (i = 0; i < tg; i++) {

            common_batch_clear(*batch);
            for (j = 0; j < pl; j++) {
                common_batch_add(*batch, 0, i, { j }, true);
            }

            LOGi("llama_decode() text generation: %d", i);
            if (llama_decode(context, *batch) != 0) {
                LOGi("llama_decode() failed during text generation");
            }
        }

        const auto t_tg_end = ggml_time_us();

        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp = double(t_pp_end - t_pp_start) / 1000000.0;
        const auto t_tg = double(t_tg_end - t_tg_start) / 1000000.0;

        const auto speed_pp = double(pp) / t_pp;
        const auto speed_tg = double(pl * tg) / t_tg;

        pp_avg += speed_pp;
        tg_avg += speed_tg;

        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;

        LOGi("pp %f t/s, tg %f t/s", speed_pp, speed_tg);
    }

    pp_avg /= double(nr);
    tg_avg /= double(nr);

    if (nr > 1) {
        pp_std = sqrt(pp_std / double(nr - 1) - pp_avg * pp_avg * double(nr) / double(nr - 1));
        tg_std = sqrt(tg_std / double(nr - 1) - tg_avg * tg_avg * double(nr) / double(nr - 1));
    } else {
        pp_std = 0;
        tg_std = 0;
    }

    char model_desc[128];
    llama_model_desc(model, model_desc, sizeof(model_desc));

    const auto model_size     = double(llama_model_size(model)) / 1024.0 / 1024.0 / 1024.0;
    const auto model_n_params = double(llama_model_n_params(model)) / 1e9;

    const auto backend    = "(Android)"; // TODO: What should this be?

    std::stringstream result;
    result << std::setprecision(2);
    result << "| model | size | params | backend | test | t/s |\n";
    result << "| --- | --- | --- | --- | --- | --- |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | " << backend << " | pp " << pp << " | " << pp_avg << " ± " << pp_std << " |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | " << backend << " | tg " << tg << " | " << tg_avg << " ± " << tg_std << " |\n";

    return env->NewStringUTF(result.str().c_str());
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_new_1batch(JNIEnv *, jobject, jint n_tokens, jint embd, jint n_seq_max) {

    // Source: Copy of llama.cpp:llama_batch_init but heap-allocated.

    llama_batch *batch = new llama_batch {
        0,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
    };

    if (embd) {
        batch->embd = (float *) malloc(sizeof(float) * n_tokens * embd);
    } else {
        batch->token = (llama_token *) malloc(sizeof(llama_token) * n_tokens);
    }

    batch->pos      = (llama_pos *)     malloc(sizeof(llama_pos)      * n_tokens);
    batch->n_seq_id = (int32_t *)       malloc(sizeof(int32_t)        * n_tokens);
    batch->seq_id   = (llama_seq_id **) malloc(sizeof(llama_seq_id *) * n_tokens);
    llama_seq_id * seq_id_storage = (llama_seq_id *) malloc(sizeof(llama_seq_id) * n_tokens * n_seq_max);
    for (int i = 0; i < n_tokens; ++i) {
        batch->seq_id[i] = seq_id_storage + (i * n_seq_max);
    }
    batch->logits   = (int8_t *)        malloc(sizeof(int8_t)         * n_tokens);

    LOGi("Allocated batch: n_tokens = %d, embd = %d, n_seq_max = %d", n_tokens, embd, n_seq_max);
    log_native_memory_snapshot("jni-after-new-batch");

    return reinterpret_cast<jlong>(batch);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_free_1batch(JNIEnv *, jobject, jlong batch_pointer) {
    const auto batch = reinterpret_cast<llama_batch *>(batch_pointer);
    if (!batch) {
        return;
    }

    free(batch->token);
    free(batch->embd);
    free(batch->pos);
    free(batch->n_seq_id);
    if (batch->seq_id) {
        if (batch->seq_id[0]) {
            free(batch->seq_id[0]);
        }
    }
    free(batch->seq_id);
    free(batch->logits);
    delete batch;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_new_1sampler(JNIEnv *, jobject) {
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    return reinterpret_cast<jlong>(smpl);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_free_1sampler(JNIEnv *, jobject, jlong sampler_pointer) {
    llama_sampler_free(reinterpret_cast<llama_sampler *>(sampler_pointer));
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_backend_1init(JNIEnv *, jobject) {
    enable_android_opencl_ifairy64_gate();
    LOGi("OpenCL env before backend init: OCL_ICD_FILENAMES=%s OCL_ICD_ENABLE_TRACE=%s",
            getenv("OCL_ICD_FILENAMES") ? getenv("OCL_ICD_FILENAMES") : "<unset>",
            getenv("OCL_ICD_ENABLE_TRACE") ? getenv("OCL_ICD_ENABLE_TRACE") : "<unset>");
    llama_backend_init();
}

extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_system_1info(JNIEnv *env, jobject) {
    return env->NewStringUTF(llama_print_system_info());
}

extern "C"
JNIEXPORT jint JNICALL
Java_android_llama_cpp_LLamaAndroid_completion_1init(
        JNIEnv *env,
        jobject,
        jlong context_pointer,
        jlong batch_pointer,
        jstring jtext,
        jboolean format_chat,
        jint n_len
    ) {
    const auto text = env->GetStringUTFChars(jtext, 0);
    if (text == nullptr) {
        return 0;
    }
    auto * wrapper = from_context_handle(context_pointer);
    const auto context = wrapper != nullptr ? wrapper->context : nullptr;
    const auto batch = reinterpret_cast<llama_batch *>(batch_pointer);
    if (context == nullptr || batch == nullptr) {
        env->ReleaseStringUTFChars(jtext, text);
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "completion_init() received null state");
        return 0;
    }

    std::string prompt;
    bool add_special = true;
    const bool parse_special = (format_chat == JNI_TRUE);
    int n_past = 0;

    if (format_chat == JNI_TRUE) {
        if (!wrapper->chat_templates) {
            env->ReleaseStringUTFChars(jtext, text);
            env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "chat template is not initialized");
            return 0;
        }
        if (wrapper->pending_chat) {
            commit_pending_chat(wrapper);
        }

        common_chat_msg user_msg;
        user_msg.role = "user";
        user_msg.content = text != nullptr ? text : "";

        try {
            prompt = common_chat_format_single(
                    wrapper->chat_templates.get(),
                    wrapper->chat_history,
                    user_msg,
                    /* add_ass = */ true,
                    /* use_jinja = */ true);
        } catch (const std::exception & e) {
            env->ReleaseStringUTFChars(jtext, text);
            LOGe("Chat template formatting failed: %s", e.what());
            env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), e.what());
            return 0;
        }

        wrapper->pending_user_content = std::move(user_msg.content);
        wrapper->pending_assistant_content.clear();
        wrapper->pending_chat = true;
        n_past = wrapper->n_past;
        add_special = false;

        LOGi("Formatted chat prompt: history_messages=%zu n_past=%d bytes=%zu add_special=%d parse_special=1 preview=`%.*s`",
                wrapper->chat_history.size(),
                n_past,
                prompt.size(),
                add_special ? 1 : 0,
                (int) std::min<size_t>(prompt.size(), 240),
                prompt.c_str());
    } else {
        clear_chat_state(wrapper);
        llama_memory_clear(llama_get_memory(context), true);
        prompt = text != nullptr ? text : "";
        LOGi("Using raw prompt path: bytes=%zu parse_special=0", prompt.size());
    }

    const auto tokens_list = common_tokenize(context, prompt, add_special, parse_special);
    const int n_cur = completion_init_tokens(context, batch, tokens_list, n_len, n_past);
    if (wrapper != nullptr) {
        wrapper->n_past = n_cur;
    }

    env->ReleaseStringUTFChars(jtext, text);
    return n_cur;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_completion_1loop(
        JNIEnv * env,
        jobject,
        jlong context_pointer,
        jlong batch_pointer,
        jlong sampler_pointer,
        jint n_len,
        jobject intvar_ncur
) {
    auto * wrapper = from_context_handle(context_pointer);
    const auto context = wrapper != nullptr ? wrapper->context : nullptr;
    const auto batch   = reinterpret_cast<llama_batch   *>(batch_pointer);
    const auto sampler = reinterpret_cast<llama_sampler *>(sampler_pointer);

    if (!la_int_var) la_int_var = env->GetObjectClass(intvar_ncur);
    if (!la_int_var_value) la_int_var_value = env->GetMethodID(la_int_var, "getValue", "()I");
    if (!la_int_var_inc) la_int_var_inc = env->GetMethodID(la_int_var, "inc", "()V");

    const auto n_cur = env->CallIntMethod(intvar_ncur, la_int_var_value);
    int n_cur_mut = n_cur;
    std::string piece;
    if (!completion_loop_step(context, batch, sampler, n_len, n_cur_mut, piece)) {
        return nullptr;
    }

    if (wrapper != nullptr) {
        wrapper->n_past = n_cur_mut;
        if (wrapper->pending_chat) {
            wrapper->pending_assistant_content += piece;
        }
    }

    env->CallVoidMethod(intvar_ncur, la_int_var_inc);
    return env->NewStringUTF(piece.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_completion_1end(
        JNIEnv *,
        jobject,
        jlong context_pointer
) {
    commit_pending_chat(from_context_handle(context_pointer));
}

extern "C"
JNIEXPORT jdoubleArray JNICALL
Java_android_llama_cpp_LLamaAndroid_e2e_1bench_1model(
        JNIEnv * env,
        jobject,
        jlong context_pointer,
        jlong batch_pointer,
        jlong sampler_pointer,
        jint pp,
        jint tg,
        jint nr
) {
    const auto context = unwrap_context(context_pointer);
    const auto batch = reinterpret_cast<llama_batch *>(batch_pointer);
    const auto sampler = reinterpret_cast<llama_sampler *>(sampler_pointer);

    if (context == nullptr || batch == nullptr || sampler == nullptr) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "e2e_bench_model() received null state");
        return nullptr;
    }

    if (pp <= 0 || tg <= 0 || nr <= 0) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "e2e_bench_model() requires positive pp/tg/nr");
        return nullptr;
    }

    e2e_bench_aggregate aggregate;

    char model_desc[128];
    llama_model_desc(llama_get_model(context), model_desc, sizeof(model_desc));
    LOGi("E2E benchmark start: model=%s prompt=%d decode=%d repetitions=%d n_ctx=%d n_batch=%d n_ubatch=%d threads=%d batch_threads=%d affinity=%s(%d) ifairy_lut=%s ifairy_vecdot_act_tensor=%s",
            model_desc,
            pp,
            tg,
            nr,
            g_effective_runtime_config.n_ctx,
            g_effective_runtime_config.n_batch,
            g_effective_runtime_config.n_ubatch,
            g_effective_runtime_config.n_threads,
            g_effective_runtime_config.n_threads_batch,
            affinity_profile_name(g_effective_runtime_config.affinity_profile),
            g_effective_runtime_config.affinity_profile,
            g_effective_runtime_config.disable_ifairy_lut == 0 ? "enabled" : "disabled",
            runtime_toggle_name(g_effective_runtime_config.enable_ifairy_vecdot_act_tensor));

    for (int run = 0; run < nr; ++run) {
        llama_memory_clear(llama_get_memory(context), true);
        llama_sampler_reset(sampler);
        cached_token_chars.clear();

        const auto prompt_tokens = build_bench_prompt_tokens(context, pp);

        using clock = std::chrono::steady_clock;
        const auto request_start = clock::now();
        int n_cur = completion_init_tokens(context, batch, prompt_tokens, tg, 0);
        const int total_len = n_cur + tg;
        const auto after_prefill = clock::now();

        int generated_tokens = 0;
        bool first_token_seen = false;
        std::chrono::steady_clock::time_point first_token_at = after_prefill;

        while (generated_tokens < tg) {
            std::string piece;
            if (!completion_loop_step(context, batch, sampler, total_len, n_cur, piece)) {
                break;
            }

            ++generated_tokens;
            if (!first_token_seen) {
                first_token_seen = true;
                first_token_at = clock::now();
            }
        }

        const auto finished_at = clock::now();
        llama_memory_clear(llama_get_memory(context), true);

        const double prefill_ms = std::chrono::duration<double, std::milli>(after_prefill - request_start).count();
        const double first_token_ms = first_token_seen
                ? std::chrono::duration<double, std::milli>(first_token_at - request_start).count()
                : prefill_ms;
        const double total_ms = std::chrono::duration<double, std::milli>(finished_at - request_start).count();
        const double decode_window_ms = first_token_seen
                ? std::chrono::duration<double, std::milli>(finished_at - first_token_at).count()
                : 0.0;
        const int decode_tail_tokens = std::max(generated_tokens - 1, 0);
        const double decode_ms_per_token = decode_tail_tokens > 0
                ? decode_window_ms / (double) decode_tail_tokens
                : 0.0;
        const double tok_s = decode_ms_per_token > 0.0
                ? 1000.0 / decode_ms_per_token
                : 0.0;

        aggregate.generated_tokens_avg += (double) generated_tokens;
        aggregate.prefill_ms_avg += prefill_ms;
        aggregate.first_token_ms_avg += first_token_ms;
        aggregate.decode_ms_per_token_avg += decode_ms_per_token;
        aggregate.tok_s_avg += tok_s;
        aggregate.total_ms_avg += total_ms;

        LOGi("E2E benchmark run %d/%d: model=%s generated=%d prefill_ms=%.2f first_token_ms=%.2f decode_ms_per_token=%.2f tok_s=%.4f total_ms=%.2f",
                run + 1,
                nr,
                model_desc,
                generated_tokens,
                prefill_ms,
                first_token_ms,
                decode_ms_per_token,
                tok_s,
                total_ms);
    }

    aggregate.generated_tokens_avg /= nr;
    aggregate.prefill_ms_avg /= nr;
    aggregate.first_token_ms_avg /= nr;
    aggregate.decode_ms_per_token_avg /= nr;
    aggregate.tok_s_avg /= nr;
    aggregate.total_ms_avg /= nr;

    LOGi("E2E benchmark average: model=%s generated=%.2f prefill_ms=%.2f first_token_ms=%.2f decode_ms_per_token=%.2f tok_s=%.4f total_ms=%.2f",
            model_desc,
            aggregate.generated_tokens_avg,
            aggregate.prefill_ms_avg,
            aggregate.first_token_ms_avg,
            aggregate.decode_ms_per_token_avg,
            aggregate.tok_s_avg,
            aggregate.total_ms_avg);

    jdouble values[6] = {
            aggregate.generated_tokens_avg,
            aggregate.prefill_ms_avg,
            aggregate.first_token_ms_avg,
            aggregate.decode_ms_per_token_avg,
            aggregate.tok_s_avg,
            aggregate.total_ms_avg,
    };
    jdoubleArray result = env->NewDoubleArray(6);
    env->SetDoubleArrayRegion(result, 0, 6, values);
    return result;
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_kv_1cache_1clear(JNIEnv *, jobject, jlong context) {
    auto * wrapper = from_context_handle(context);
    if (wrapper != nullptr && wrapper->context != nullptr) {
        llama_memory_clear(llama_get_memory(wrapper->context), true);
    }
    clear_chat_state(wrapper);
    log_native_memory_snapshot("jni-after-kv-cache-clear");
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_log_1native_1memory(JNIEnv *env, jobject, jstring jstage) {
    const auto stage = env->GetStringUTFChars(jstage, nullptr);
    log_native_memory_snapshot(stage != nullptr ? stage : "unknown");
    if (stage != nullptr) {
        env->ReleaseStringUTFChars(jstage, stage);
    }
}
