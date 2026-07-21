# Stage 1-4 Implementation Notes

本文档记录 fariy2i 的 `llama.cpp` fork 当前已经完成的 speculative n-gram 迁移工作。

目标范围仍然只覆盖 `common` 层和 `examples/speculative-simple`；本阶段没有修改 `tools/server`。

## 当前提交范围

已经完成：

```text
阶段 1: common: add speculative type parameters
阶段 2: common: add ngram speculative helpers
阶段 3: common: introduce type-based speculative framework
阶段 4: examples: use speculative framework in speculative-simple
```

尚未完成：

```text
tools/server speculative 迁移
```

阶段 4 已经让 `llama-speculative-simple` 改用新接口，并根据 `--spec-type` 选择 `draft-simple` 或 n-gram 类型。

## 阶段 1: 参数层

阶段 1 的目标是让公共参数可以表达 speculative type，而不是把 speculative 固定理解成 draft model。

修改文件：

```text
common/common.h
common/arg.cpp
common/speculative.h
common/speculative.cpp
```

新增 speculative type：

```cpp
enum common_speculative_type {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_COUNT,
};
```

扩展 `common_params_speculative`：

```cpp
std::vector<enum common_speculative_type> types = { COMMON_SPECULATIVE_TYPE_NONE };

llama_context * ctx_tgt = nullptr;
llama_context * ctx_dft = nullptr;

common_params_speculative_ngram_mod ngram_mod;
common_params_speculative_ngram_map ngram_simple;
common_params_speculative_ngram_map ngram_map_k;
common_params_speculative_ngram_map ngram_map_k4v;
```

新增 type helper：

```cpp
const char * common_speculative_all_types_str();
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);
enum common_speculative_type common_speculative_type_from_name(const std::string & name);
std::string common_speculative_type_to_str(enum common_speculative_type type);
```

新增 CLI 参数：

```text
--spec-type

--spec-ngram-simple-size-n
--spec-ngram-simple-size-m
--spec-ngram-simple-min-hits

--spec-ngram-map-k-size-n
--spec-ngram-map-k-size-m
--spec-ngram-map-k-min-hits

--spec-ngram-map-k4v-size-n
--spec-ngram-map-k4v-size-m
--spec-ngram-map-k4v-min-hits

--spec-ngram-mod-n-match
--spec-ngram-mod-n-min
--spec-ngram-mod-n-max
```

注意：阶段 1 只让参数可以表达这些类型，不负责真正生成 draft tokens。

## 阶段 2: n-gram 算法层

阶段 2 的目标是把上游 llama.cpp 的 n-gram helper 放入 fariy2i 的 common 库。

新增文件：

```text
common/ngram-map.h
common/ngram-map.cpp
common/ngram-mod.h
common/ngram-mod.cpp
```

修改文件：

```text
common/CMakeLists.txt
```

引入的 helper 能力：

```text
common_ngram_simple_draft
common_ngram_map_begin
common_ngram_map_draft
common_ngram_map_accept
common_ngram_mod
```

这些文件目前只是算法能力和 common build 接入，调用入口由阶段 3 的 framework 统一封装。

## 阶段 3: type-based speculative framework

阶段 3 的目标是新增统一 speculative 框架，让 draft model 和 n-gram 都可以走同一组 common 接口。

修改文件：

```text
common/speculative.h
common/speculative.cpp
```

旧接口保留：

```cpp
common_speculative_init(ctx_tgt, ctx_dft);
common_speculative_gen_draft(...);
```

新接口新增：

```cpp
struct common_speculative_draft_params;

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);
common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);
void common_speculative_draft(common_speculative * spec);
void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted);
```

`common_speculative_draft_params` 是每个 sequence 的 draft 请求槽：

```cpp
struct common_speculative_draft_params {
    bool drafting = false;
    int32_t n_max = -1;
    llama_pos   n_past  = 0;
    llama_token id_last = LLAMA_TOKEN_NULL;
    const llama_tokens * prompt = nullptr;
    llama_tokens       * result = nullptr;
};
```

当前字段含义：

```text
drafting:
  当前 sequence 是否需要生成 draft。

n_max:
  单次 draft token 上限。>= 0 时覆盖 params.speculative.n_max。

n_past:
  预留给调用侧记录 target 已处理位置。当前 common impl 内部还没有消费它。

id_last:
  target 刚采样出的 token，也是 draft 的起点 token。

prompt:
  当前 sequence 已知 token 历史。

result:
  输出 draft tokens。
```

内部新增抽象：

```cpp
struct common_speculative_impl {
    common_speculative_type type;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt);
    virtual void draft(common_speculative * spec, llama_seq_id seq_id, common_speculative_draft_params & params) = 0;
    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted);
};
```

已接入 impl：

```text
common_speculative_impl_draft_simple
common_speculative_impl_ngram_simple
common_speculative_impl_ngram_map
common_speculative_impl_ngram_mod
```

type 到 impl 的关系：

```text
draft-simple  -> common_speculative_impl_draft_simple
ngram-simple  -> common_speculative_impl_ngram_simple
ngram-map-k   -> common_speculative_impl_ngram_map(key_only = true)
ngram-map-k4v -> common_speculative_impl_ngram_map(key_only = false)
ngram-mod     -> common_speculative_impl_ngram_mod
none          -> no impl
```

### draft-simple 的处理方式

`draft-simple` 没有复制一份 draft model 逻辑，而是在新 framework impl 内部继续调用旧函数：

```cpp
common_speculative_gen_draft(spec, params, *dp.prompt, dp.id_last);
```

这样做的目的：

```text
1. 保留旧 draft model 行为，降低阶段 3 风险。
2. 对外先切到新接口，后续 example / server 不需要再重复换入口。
3. 让 draft-simple 和 n-gram 可以按 type 统一调度。
```

因此阶段 3 后的关系是：

```text
外部调用新接口
  -> common_speculative_impl_draft_simple
      -> 旧 common_speculative_gen_draft(...)
```

旧接口还可以继续被旧调用方直接使用。

### ngram-simple 的处理方式

`ngram-simple` 使用阶段 2 引入的：

```cpp
common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
```

并在生成后根据 `dp.n_max` 限制输出长度。

### ngram-map-k / ngram-map-k4v 的处理方式

`ngram-map-k` 和 `ngram-map-k4v` 共用同一个 impl：

```cpp
common_speculative_impl_ngram_map
```

差异只在构造 `common_ngram_map` 时的 `key_only`：

```text
ngram-map-k:
  key_only = true

ngram-map-k4v:
  key_only = false
```

生命周期：

```text
begin:
  common_ngram_map_begin(...)

draft:
  common_ngram_map_draft(...)

accept:
  common_ngram_map_accept(...)
```

### ngram-mod 的处理方式

`ngram-mod` 使用：

```cpp
common_ngram_mod
```

当前实现保留了上游的核心策略：

```text
1. begin 时把 prompt 写入 ngram mod。
2. draft 时把新增 prompt 片段继续滚入 mod。
3. 根据 n_match 查找后续 token。
4. 如果命中数量不足 n_min，则返回空 draft。
5. 如果低接受率连续出现，则 reset ngram mod。
```

为了保持当前阶段简单，`common_ngram_mod` 使用固定容量：

```cpp
4*1024*1024
```

后续如果需要与上游完全对齐，可以再把容量变成参数。

## 新 framework 的推荐调用方式

阶段 4 修改 `examples/speculative-simple` 时，建议调用侧按下面方式使用：

```cpp
params.speculative.ctx_tgt = ctx_tgt;
params.speculative.ctx_dft = ctx_dft; // 只有 draft-simple 需要

common_speculative * spec = common_speculative_init(params.speculative, 1);
common_speculative_begin(spec, 0, prompt);

auto & dp = common_speculative_get_draft_params(spec, 0);
dp.drafting = true;
dp.n_max    = params.speculative.n_max;
dp.n_past   = n_past;
dp.id_last  = id_last;
dp.prompt   = &prompt;
dp.result   = &draft;

common_speculative_draft(spec);
```

target 校验接受数量之后调用：

```cpp
common_speculative_accept(spec, 0, n_accepted);
```

释放仍然使用：

```cpp
common_speculative_free(spec);
```

## 阶段 4: speculative-simple 接入

阶段 4 的目标是让 `examples/speculative-simple` 真正消费阶段 1-3 引入的新框架。

修改文件：

```text
examples/speculative-simple/speculative-simple.cpp
```

完成的改动：

```text
1. 不再无条件要求 --model-draft。
2. 如果用户没有传 --spec-type，则自动选择默认 type:
   - 有 draft model: 默认 draft-simple
   - 无 draft model: 默认 ngram-simple
3. 如果用户显式传 --spec-type none，则保留 none，不自动改成 ngram-simple。
4. 只有 type 包含 draft-simple 时才加载 draft model。
5. 纯 n-gram type 不再初始化 ctx_dft。
6. 设置 params.speculative.ctx_tgt / ctx_dft。
7. 用 common_speculative_init(params.speculative, 1) 创建 spec。
8. 用 common_speculative_begin(spec, 0, prompt_tgt) 初始化 sequence。
9. 每轮采样前设置 common_speculative_draft_params。
10. 调 common_speculative_draft(spec) 生成 draft。
11. target 校验后调 common_speculative_accept(spec, 0, n_accepted)。
12. 无 draft model 时跳过 llama_perf_context_print(ctx_dft)。
13. accept 百分比输出避免 n_drafted == 0 时除零。
```

默认 type 选择逻辑：

```cpp
if (!has_user_spec_type && speculative_types_are_default(params.speculative.types)) {
    params.speculative.types = {
        params.speculative.model.path.empty()
            ? COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE
            : COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE
    };
}
```

draft model 加载条件：

```cpp
const bool has_draft_simple = speculative_has_type(params.speculative.types, COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);

if (has_draft_simple && params.speculative.model.path.empty()) {
    LOG_ERR("%s: --model-draft is required for --spec-type draft-simple\n", __func__);
    return 1;
}
```

新 speculative 初始化方式：

```cpp
params.speculative.ctx_tgt = ctx_tgt;
params.speculative.ctx_dft = ctx_dft;

common_speculative * spec = common_speculative_init(params.speculative, 1);
common_speculative_begin(spec, 0, prompt_tgt);
```

每轮 draft 生成方式：

```cpp
llama_tokens draft;
common_speculative_get_draft_params(spec, 0) = {
    /* .drafting = */ true,
    /* .n_max    = */ n_draft,
    /* .n_past   = */ n_past,
    /* .id_last  = */ id_last,
    /* .prompt   = */ &prompt_tgt,
    /* .result   = */ &draft,
};
common_speculative_draft(spec);
```

target 接受 draft 后通知 framework：

```cpp
common_speculative_accept(spec, 0, ids.size() - 1);
```

阶段 4 完成后，下面命令可以作为最小验证目标：

```bash
llama-speculative-simple -m target.gguf --spec-type ngram-simple -p "..."
```

旧 draft model 路径也应该继续可用：

```bash
llama-speculative-simple -m target.gguf -md draft.gguf --spec-type draft-simple -p "..."
```

## 已做验证

当前阶段已做单文件编译验证：

```text
common/speculative.cpp
common/arg.cpp
common/ngram-map.cpp
common/ngram-mod.cpp
examples/speculative-simple/speculative-simple.cpp
```

同时执行：

```text
git diff --check
```

结果通过。

因为当前环境没有完整 CMake 构建链，这里没有执行完整项目构建。

## 当前边界

阶段 4 已经把 `llama-speculative-simple` 切到新 common framework。

当前代码已经具备：

```text
common 层按 type 初始化 impl
common 层按 sequence 生成 draft
common 层通知 impl 接受数量
speculative-simple 按 --spec-type 选择 draft-simple / n-gram
speculative-simple 在无 draft model 时默认 ngram-simple
```

当前仍未处理：

```text
tools/server speculative 迁移
server slot / multi-sequence speculative 生命周期
KV cache checkpoint / partial sequence removal
```
