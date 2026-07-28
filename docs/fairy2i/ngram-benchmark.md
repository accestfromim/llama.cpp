# Fairy2i N-Gram 评测

`scripts/run_fairy2i_ngram_eval.py` 用于运行可断点续跑的 Fairy2i N-Gram
评测。脚本会保存原始日志、每次运行对应的一份原子 JSON 记录，以及汇总后的
JSON 和 Markdown 报告。

## 默认评测配置

默认配置如下：

- 从 summarization、RAG、HAGRID 和 TriviaQA 中分别顺序选取前 20 条；
- 共 80 个 prompt，每个后端运行 80 次模型推理；
- 默认使用 `ngram-simple`；
- 使用 GGUF 聊天模板（`-cnv`），保留模型原生思考行为；
- 贪心采样：`temperature=0`、`top-k=1`、seed 42；
- 最大输出 512 token；
- context/batch/ubatch 为 `4096/4096/512`；
- 生成线程和 batch 线程均为 8；
- N-Gram 参数为 `N=12`、`M=48`、`min-hits=1`；
- `draft-max=16`、`draft-min=0`。

评测回答质量时不要使用 `--raw-completion`。关闭聊天模板后，模型可能把指令
当作普通文本续写，产生重复内容和虚高的 N-Gram 接受率。

## 编译

以下命令都应在对应的 `llama.cpp` 仓库根目录执行。不同后端使用不同的 build
目录，不要在同一个 build 目录中反复切换后端配置。

评测脚本需要的可执行文件是：

```text
<build-dir>/bin/llama-speculative-simple
```

### CPU

配置 Release 构建：

```bash
cmake -B build-rel-fairy2i \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_FAIRY2I=ON \
    -DGGML_FAIRY2I_CPU=ON \
    -DGGML_FAIRY2I_CPU_LUT=ON
```

编译评测 runner 和 Fairy2i 测试：

```bash
cmake --build build-rel-fairy2i \
    --target llama-speculative-simple test-fairy2i test-fairy2i-loader \
    -j "$(nproc)"
```

运行相关测试：

```bash
ctest --test-dir build-rel-fairy2i --output-on-failure -R fairy2i
```

CPU runner 路径为：

```text
build-rel-fairy2i/bin/llama-speculative-simple
```

`-DGGML_FAIRY2I_CPU_LUT=ON` 必须在配置阶段启用。评测脚本会在运行 CPU
配置时设置 `GGML_FAIRY2I_LUT=1` 和 `GGML_FAIRY2I_LUT_IMPL=lut16`，但
运行时环境变量不能启用一个没有被编译进 runner 的实现。

### OpenCL

系统必须先安装可用的 OpenCL ICD、运行时、开发头文件和驱动。只有
`clinfo` 能看到目标设备还不够，最终仍应通过 runner 的 `--list-devices`
确认设备已被 llama.cpp 注册。

OpenCL 使用独立 build 目录：

```bash
cmake -B build-opencl-fairy2i \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_FAIRY2I=ON \
    -DGGML_OPENCL=ON \
    -DGGML_FAIRY2I_OPENCL=ON
```

编译：

```bash
cmake --build build-opencl-fairy2i \
    --target llama-speculative-simple test-fairy2i test-backend-ops \
    -j "$(nproc)"
```

查看构建产物识别到的设备：

```bash
./build-opencl-fairy2i/bin/llama-speculative-simple --list-devices
```

复制输出中的 OpenCL 设备名，作为评测脚本的 `--device` 参数。不要直接假设
设备一定叫 `OpenCL0`。



### 其他后端

CUDA、HIP、MUSA、Vulkan、SYCL、CANN、WebGPU、zDNN 和 RPC 的 Fairy2i
实现不都位于当前仓库。应在提供对应 Fairy2i 算子的仓库或分支中，按照该仓库
自己的构建说明启用后端。不能只在当前仓库中打开通用后端开关，就假定该后端
已经具备 Fairy2i 原生算子。

无论使用哪个外部后端，交给评测脚本前都要完成以下检查：

```bash
/absolute/path/to/build/bin/llama-speculative-simple --list-devices
```

然后用模型做一次短生成：

```bash
/absolute/path/to/build/bin/llama-speculative-simple \
    --model /absolute/path/to/model.gguf \
    --device <从--list-devices复制的设备名> \
    -ngl 999 \
    --spec-type ngram-simple \
    --draft-max 4 \
    --spec-ngram-simple-size-n 3 \
    --spec-ngram-simple-size-m 4 \
    --temp 0 --top-k 1 \
    -n 8 \
    -p "Hello"
```

短生成成功后，还必须从后端日志或 profiler 确认 Fairy2i 关键算子确实在目标
设备执行，而不是回退到 CPU。确认后，这个 runner 才能用于正式后端评测。

### 配置评测路径

脚本不会假设模型、数据集或可执行文件的位置。请根据测试机器设置绝对路径：

```bash
MODEL=/absolute/path/to/model.gguf
DATA_ROOT=/absolute/path/to/ngram_trie_eval_data
CPU_RUNNER=/absolute/path/to/cpu-build/bin/llama-speculative-simple
```

`DATA_ROOT` 必须直接包含 `spec_bench/`、`hagrid/` 和 `triviaqa/`。
`--model`、`--data-root` 和 `--runner` 均为必填参数。

## 冒烟测试

从每个任务中取 1 条，使用 `ngram-simple` 运行：

```bash
python3 scripts/run_fairy2i_ngram_eval.py cpu \
    --model "$MODEL" \
    --data-root "$DATA_ROOT" \
    --runner "$CPU_RUNNER" \
    --mode ngram-simple \
    --samples-per-task 1 \
    --output-dir results/smoke-cpu
```

每种配置应使用独立的输出目录。脚本会保存配置指纹，并拒绝在同一目录中混入
不兼容的运行配置。

关于data_root :
例如：
`/home/wqj/projects/llama.cpp/ngram_trie_eval_data/ngram_trie_eval_data/hagrid`
则
data_root:
`/home/wqj/projects/llama.cpp/ngram_trie_eval_data/ngram_trie_eval_data`

每次调用只能通过 `--mode` 指定一种模式，不接受逗号分隔的模式列表：

```text
none
ngram-simple
ngram-map-k
ngram-map-k4v
ngram-mod
```

## 后端评测

脚本提供以下后端配置名称：

```text
cpu, blas, cuda, hip, musa, metal, vulkan, sycl, cann, opencl, webgpu,
zdnn, rpc
```

这些名称表示实验后端，不表示后端内部的 kernel。脚本不比较 CPU 内部的
ISA/LUT 路径，也不比较 OpenCL 内部的 GEMM/GEMV/direct 路径。每个后端
只使用其构建配置及正常运行时策略选中的一种实现。

后端名称只是脚本提供的配置入口。脚本不会检查对应仓库是否已经实现
Fairy2i 算子，也不会通过 profiler 自动判断关键算子是否回退到 CPU。该检查
需要运行者根据对应后端的日志或 profiler 完成。

CPU 和 BLAS 使用 `--device none -ngl 0`。加速设备使用显式指定的设备和
`-ngl 999`。CPU 固定使用项目默认的 LUT16 实现，避免调用者终端中残留的
环境变量改变实验。OpenCL 启用 Fairy2i OpenCL 运行时开关，并将内部 kernel
选择固定为 `auto`。

`blas` 配置只会禁用设备卸载，实际是否使用 BLAS 取决于 runner 的编译配置
以及具体算子是否进入 BLAS 路径，脚本不会单独验证。`rpc` 配置也只提供后端
标签、设备绑定和全层卸载参数；RPC 服务地址和设备注册仍需通过对应 runner
支持的参数配置。

使用相应后端仓库或分支编译出的 runner。例如：

```bash
python3 scripts/run_fairy2i_ngram_eval.py metal \
    --model "$MODEL" \
    --data-root "$DATA_ROOT" \
    --runner /path/to/metal-build/bin/llama-speculative-simple \
    --device Metal0 \
    --mode ngram-simple

python3 scripts/run_fairy2i_ngram_eval.py cuda \
    --model "$MODEL" \
    --data-root "$DATA_ROOT" \
    --runner /path/to/cuda-build/bin/llama-speculative-simple \
    --device CUDA0 \
    --mode ngram-simple
```

先查看 runner 暴露的设备：

```bash
/path/to/llama-speculative-simple --list-devices
```

再复制设备名称并绑定目标设备：

```bash
python3 scripts/run_fairy2i_ngram_eval.py vulkan \
    --model "$MODEL" \
    --data-root "$DATA_ROOT" \
    --runner /path/to/llama-speculative-simple \
    --device Vulkan0 \
    --mode ngram-simple
```

设备名称不可跨机器假设，必须原样复制 `--list-devices` 的输出。所有加速后端
都要求提供 `--device`，即使该 runner 只包含一个加速后端。这可以防止后端
标签与实际运行设备不一致。

正式记录结果前还应检查卸载和路径日志。程序能够完成生成并不表示 Fairy2i
关键算子原生运行在目标后端；如果关键算子回退到 CPU，就不能作为纯目标后端
性能。

当前仓库的 OpenCL 示例：

```bash
python3 scripts/run_fairy2i_ngram_eval.py opencl \
    --model "$MODEL" \
    --data-root "$DATA_ROOT" \
    --runner /absolute/path/to/opencl-build/bin/llama-speculative-simple \
    --device OpenCL0 \
    --mode ngram-simple
```

当前 master 已包含 Fairy2i 的 CPU、OpenCL 和 Metal 原生实现。其他 Fairy2i
后端应使用对应的外部仓库或分支。

## 算法参数

### 模式选择

每次运行只能通过 `--mode` 选择一种模式：

| 模式 | 工作方式 |
| --- | --- |
| `none` | 不生成 draft，执行普通逐 token 解码，用于单独运行基线 |
| `ngram-simple` | 每一步从已有 token 历史中反向搜索最近一次相同的 N-Gram，并把其后续 token 作为候选 |
| `ngram-map-k` | 使用映射和哈希索引维护 N-Gram key，找到 key 后使用其历史后续 token |
| `ngram-map-k4v` | 一个 N-Gram key 最多统计 4 种后续 M-Gram，利用出现次数判断当前 key 的后续是否足够稳定 |
| `ngram-mod` | 使用固定大小的哈希表，把长度为 `match` 的 token 序列映射到下一个 token，并沿映射逐 token 构造 draft |

这些模式都不使用额外的 draft 模型。候选 token 仍必须交给 target 模型验证；
只有验证通过的 token 才会被接受。

### N：匹配上下文长度

参数：

```text
--ngram-n
```

适用于 `ngram-simple`、`ngram-map-k` 和 `ngram-map-k4v`。它表示查找历史时
用于匹配的连续 token 数，默认值为 12。


### M：历史候选序列长度

参数：

```text
--ngram-m
```

适用于 `ngram-simple`、`ngram-map-k` 和 `ngram-map-k4v`，默认值为 48。
找到 N-Gram 后，算法最多考察其后连续 M 个历史 token，作为可供裁剪的候选
序列。

M 可以大于 `draft-max`。两者作用不同：

例如：

```text
N=12, M=48, draft-max=16(官方默认配置)
```

表示用 12-token 序列查找历史，历史候选最长可到 48 token，但当前验证步骤
最多只取前 16 个 draft token。M 较大还会提高算法开始产生 draft 所需的最短
历史长度，因此不应把 M 无限制调大。


### min-hits：最少命中次数

参数：

```text
--ngram-min-hits
```

该参数默认值为 1，但其实际作用取决于模式：

| 模式 | 是否实际使用 `min-hits` |
| --- | --- |
| `ngram-simple` | 否，当前实现直接反向搜索最近匹配 |
| `ngram-map-k` | 否，当前 key-only 分支在检查 `min-hits` 前已经生成 draft |
| `ngram-map-k4v` | 是，key 的累计命中次数达到阈值后才考虑生成 draft |
| `ngram-mod` | 不适用 |

因此，当前代码中调整 `--ngram-min-hits` 只会实质改变
`ngram-map-k4v`。阈值越高，算法越保守：产生 draft 的样本和次数可能减少，
但候选通常有更多历史证据。

### draft-max：单轮 draft 上限

参数：

```text
--draft-max
```

表示一次投机步骤最多向 target 模型提交多少个候选 token，默认值为 16。
无论 N-Gram 算法内部找到了多长的候选序列，runner 都会把结果裁剪到该上限。
临近 `--max-new-tokens` 限制时，本轮上限还会进一步缩小。

- 较大值可能用一次 target 验证覆盖更多 token，接受率高时有利于摊薄调用
  开销。
- 较大值也会增加一次验证的计算量；如果后半段候选经常被拒绝，可能得不偿失。
- 它限制的是每轮 draft 长度，不是整条样本的累计 `n_drafted`。

例如一条样本在很多轮中反复产生 draft，即使 `draft-max=16`，最终统计的
`n_drafted` 仍然可以远大于 16。

### draft-min：最短可用 draft

参数：

```text
--draft-min
```

表示 runner 接受一个候选批次所需的最少 draft token 数，默认值为 0。如果
N-Gram 算法返回的候选数小于该阈值，runner 会清空该候选并按普通解码处理
本轮。

- `draft-min=0`：不额外过滤短 draft。
- 提高该值：可以避免为了很短的候选启动投机验证，但会减少 draft 命中轮数。

它是 runner 层面的统一过滤条件，不是 N-Gram 的匹配长度。不要与
`ngram-mod` 自己的 `--ngram-mod-min` 混淆。

### ngram-mod 参数

`ngram-mod` 不使用 `N/M/min-hits`，而使用以下参数：

| 参数 | 含义 | 默认值 |
| --- | --- | ---: |
| `--ngram-mod-match` | 哈希 key 中包含的连续 token 数 | 24 |
| `--ngram-mod-min` | 沿哈希映射构造候选时必须成功得到的最少 token 数 | 0 |
| `--ngram-mod-max` | 调用方未提供本轮上限时使用的候选 token 上限 | 16 |

`ngram-mod-match` 越大，key 越具体，但历史命中会更少。生成过程中如果映射链
在达到 `ngram-mod-min` 前中断，本轮整个 draft 会被丢弃。达到最小值后中断，
则保留已经找到的候选。

在 `llama-speculative-simple` 中，runner 每轮都会显式传入由 `draft-max`
计算出的上限。因此当前 runner 实际使用的是 `--draft-max`，
`--ngram-mod-max` 只作为调用方没有提供本轮上限时的 fallback；在这份评测
脚本中通常不会生效。当前 runner 的实际本轮上限可理解为：

```text
min(draft-max, 当前剩余输出 token 数)
```

### 推荐的固定实验配置

如果目的是比较不同后端，而不是调优 N-Gram 算法，应在所有后端上固定相同的
模式和参数。例如：

```bash
python3 scripts/run_fairy2i_ngram_eval.py cpu \
    --model "$MODEL" \
    --data-root "$DATA_ROOT" \
    --runner "$CPU_RUNNER" \
    --mode ngram-simple \
    --ngram-n 12 \
    --ngram-m 48 \
    --ngram-min-hits 1 \
    --draft-max 16 \
    --draft-min 0 \
    --output-dir results/cpu-ngram-simple-n12-m48
```

对其他后端只替换后端名称、runner、device 和输出目录，不要同时修改
N-Gram 参数。这样测到的差异才主要来自后端，而不是算法配置变化。

## 数据集范围

选择任务：

```bash
--tasks summarization,rag,hagrid,triviaqa
```

`--samples-per-task` 表示从每个选中任务的 JSONL 文件开头顺序选取多少条。
例如，每个任务取前 5 条：

```bash
--samples-per-task 5
```

默认值为每个任务 20 条。设为 0 表示运行所有数据：

```bash
--samples-per-task 0
```

固定短输出吞吐评测可以使用 `--max-new-tokens 128`；原生聊天负载使用默认的
512。不要把不同最大输出长度产生的结果当作相同实验直接比较。

## 断点续跑和结果汇总

使用完全相同的命令重新运行即可续跑。已有的成功记录会被跳过，失败或不完整
的记录会重新运行。

每个输出目录包含：

```text
manifest.json
records/<task>/<mode>/*.json
logs/<task>/<mode>/*.stdout.txt
logs/<task>/<mode>/*.stderr.log
summary.json
summary.md
```

不运行推理，仅重建汇总：

```bash
python3 scripts/run_fairy2i_ngram_eval.py cpu \
    --model "$MODEL" \
    --data-root "$DATA_ROOT" \
    --runner "$CPU_RUNNER" \
    --mode ngram-simple \
    --summary-only
```

重建汇总时的参数必须与原实验一致，因为这些参数会参与配置指纹计算。

## 指标口径

汇总报告使用：

```text
Decode TPS = sum(n_predict) / sum(decode_seconds)
接受率     = sum(n_accept) / sum(n_drafted)
```

`summary.md` 当前记录：

- 后端名称和配置指纹；
- 成功运行数和预期运行数；
- 输出 token 数；
- Decode TPS；
- drafted 和 accepted token 数；
- 接受率。

`summary.json` 还记录失败运行数、样本 TPS 中位数，以及实际产生 draft 的
样本数。逐条 JSON 记录保存任务、样本 ID、命令、运行状态、计时、draft
统计、stdout/stderr 相对路径及内容哈希。

manifest 中记录模型和 runner 的路径、文件大小及修改时间，以及数据集路径、
文件哈希和选取数量。当前脚本不自动采集 git commit、编译器、CPU 型号或
设备 profiler 信息。

Decode TPS 来自 runner 的 `decoded ...` 计时，因此不包含模型加载和 prompt
prefill。逐条原始记录还保存进程 wall time，供诊断使用。
