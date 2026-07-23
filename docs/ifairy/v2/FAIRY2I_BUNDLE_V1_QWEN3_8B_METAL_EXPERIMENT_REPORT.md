# Fairy2i bundle_v1 Qwen3-8B Metal 实验报告

日期：2026-07-17
实验分支：`codex/metal`
基线提交：`98904a549d18a60d2c2c63bf56537156627816e8`

## 结论

推荐将当前 Qwen3-8B Fairy2i W1 learned-scale 模型改用 `bundle_m64k64_v1` GGUF 权重布局，并在 Metal 上直接消费该布局，不在加载时还原或复制成 `tile64_v2`。

该方案在本机 Apple M4 上满足以下目标：

- 252 个 linear 的 code 与 scale 对现有 V2 GGUF 做 canonical 变换后逐字节一致；147 个公共张量也逐字节一致。
- linear 权重由 1,085,276,160 B 降至 871,612,416 B，减少 19.6875%。
- 整个 GGUF 由 3,581,797,824 B 降至 3,368,144,832 B，减少 5.9650%。
- 最终 ABBA 基准中，pp256 从 176.7084 提升到 191.0285 tok/s，提升 8.1038%。
- tg128 的冷态峰值与 V2 同为约 25.00 tok/s，没有性能回退；完整 ABBA 均值为 25.0067 对 24.1391 tok/s，bundle 在持续负载下没有出现 V2 的后半段降频。
- 固定 512-token 文本的端到端 perplexity 完全一致：两者均为 `15.5413 +/- 3.66521`。

当前 Metal 支持范围有意限定为 W1 bundle（branch order `U0,W0`）。W2 bundle 继续走 CPU-LUT，loader 不会把尚未实现的 W2 bundle 错误放到 Metal。

## 实验环境

| 项目 | 值 |
| --- | --- |
| 主机 | Mac mini `Mac16,10` |
| SoC | Apple M4，10 CPU cores |
| 内存 | 24 GiB |
| 系统 | macOS 26.3，build 25D125 |
| 编译器 | Apple clang 21.0.0 (`clang-2100.1.1.101`) |
| Release flags | `-O3 -DNDEBUG` |
| CMake | `GGML_METAL=ON`, `GGML_METAL_EMBED_LIBRARY=ON`, `GGML_FAIRY2I_CPU=ON` |
| benchmark threads | 8 |
| 电源 governor | macOS 不提供可配置的 governor；使用系统默认电源管理 |

所有性能对比均由同一最终二进制执行，没有同时运行多个 `llama-bench`。

## 输入、输出与复现文件

输入 checkpoint：

```text
/Users/a1806/llama/qwen_1bit_scale/checkpoint-5639
```

对照 V2 GGUF：

```text
/Users/a1806/llama/qwen_1bit_scale/qwen3-fairy2i-w1-learned-scale.gguf
SHA256 5b638515fd1e64c39b4a7c9077e497bd99b0e213663a3174bac60cf7b28251c6
```

生成的 bundle GGUF：

```text
/Users/a1806/llama/qwen_1bit_scale/qwen3-fairy2i-w1-bundle-v1.gguf
SHA256 9f46dc9a82384b8ada003c79f8be635446b4fee28e9a0c4b5ee30908b23f5de5
```

转换命令：

```bash
cd gguf-py
../.venv/bin/python -u convert_fairy2i_qwen3.py \
  /Users/a1806/llama/qwen_1bit_scale/checkpoint-5639 \
  /Users/a1806/llama/qwen_1bit_scale/qwen3-fairy2i-w1-bundle-v1.gguf \
  --weight-layout bundle_v1 --verbose
```

转换脚本及打包实现保留在：

- `gguf-py/convert_fairy2i_qwen3.py`
- `gguf-py/fairy2i/quant/tile64_v2.py`
- `gguf-py/validate_fairy2i_bundle_v1.py`（本实验新增的全文件等价性验证器）

模型权重没有加入 Git。转换过程中只生成了最终 bundle GGUF，没有保留额外中间权重副本。

## bundle_v1 的具体打包实现

### 1. W1 分支与 learned scale

对实数展开矩阵的四个象限 `A11/A12/A21/A22`，converter 先恢复广义复线性的两个分支：

```text
U.real = 0.5 * (A11 + A22)
U.imag = 0.5 * (A21 - A12)
W.real = 0.5 * (A11 - A22)
W.imag = 0.5 * (A12 + A21)
```

checkpoint 中每个 linear 的 `quant_scale` 形状为 `[4, M/64, K/64]`，四个通道依次为 `U.real, U.imag, W.real, W.imag`。W1 bundle 的 branch order 固定为 `U0,W0`。

每个元素的 2-bit codebook 为：

```text
0 = -real scale
1 = +real scale
2 = -imag scale
3 = +imag scale
```

### 2. 物理 tile 与 code 顺序

一个物理 tile 覆盖 `M64 x K64`。定义：

```text
K_blocks     = K / 64
physical_tile = (m / 64) * K_blocks + (k / 64)
m16           = (m % 64) / 16
row_lane      = m % 16
q4            = (k % 64) / 4
slot          = m16 * 16 + q4
```

同一行、连续四个 K code 被装入一个 byte：

```text
byte = c[k+0] | c[k+1] << 2 | c[k+2] << 4 | c[k+3] << 6
```

最终 code 内存顺序为：

```text
[physical_tile][slot = m16_q4][branch][row_lane]
```

对应的线性 byte offset 为：

```text
((((physical_tile * 64 + slot) * B + branch) * 16) + row_lane)
```

W1 的 `B=2`。GGUF tensor 的 ggml 维度为：

```text
codes:  [ne0=16, ne1=2, ne2=64, ne3=physical_tiles]
scales: [ne0=2,  ne1=2, ne2=physical_tiles]
```

scale 内存顺序为 `[physical_tile][branch][real/imag]`，每个 M64xK64 tile、每个 branch 只保存一对 FP16 scale。

### 3. 相对 V2 的空间变化

对一个 W1 M64xK64 tile：

| 布局 | codes | scales | 合计 |
| --- | ---: | ---: | ---: |
| V2，两 branch | 2,048 B | 512 B（同一 scale 重复 64 行） | 2,560 B |
| bundle_v1 | 2,048 B | 8 B | 2,056 B |

因此 linear 权重的精确降幅是 `(2560 - 2056) / 2560 = 19.6875%`。code 没有损失或重新量化，节省量全部来自删除重复 scale。

### 4. GGUF 约束

生成文件写入以下关键 metadata：

```text
general.file_type                 = 42
general.alignment                 = 64
fairy2i.schema_version            = 2
fairy2i.weight.layout             = bundle_m64k64_v1
fairy2i.weight.scale_scope        = m64_k64
fairy2i.weight.code_order         = m16_q4_branch_lane
fairy2i.weight.branch_order       = U0,W0
fairy2i.weight.m_block            = 64
fairy2i.weight.k_block            = 64
fairy2i.weight.m_subtile          = 16
```

验证器确认 651 个 tensor 的 data offset 均满足 64-byte 对齐。

## 权重加载策略

最终实现不在加载时创建 V2 副本：

1. `llama-model.cpp` 根据 metadata 创建原生 `bundle.codes` 与 `bundle.scales` tensor。
2. W1 bundle 只允许分配到 CPU-LUT 或 Metal；W2 bundle 若被请求 offload 到 Metal，会在模型加载/建图阶段给出明确错误。
3. graph 将 `x, codes, scales, bias` 直接传入 `GGML_OP_FAIRY2I_WIDE_LINEAR_W1`，并通过 op params 传递 layout version、逻辑 M/K 与 branch count。
4. Metal 的 `supports_op` 在执行前验证类型、连续性、M/K 的 64 对齐、tensor shape、branch count 和设备能力。
5. Metal buffer 对权重的额外分配仍为 0；测试在 graph compute 前后都检查 `codes->extra == nullptr && scales->extra == nullptr`。

因此，GGUF mmap/upload 后的 byte 顺序就是 kernel 使用的顺序。只有 activation staging 和输出归约使用临时 buffer，不存在模型级权重重排或第二份常驻权重。

## Metal kernel 如何利用新布局

### Prefill：`kernel_fairy2i_bundle_w1_half_mma32x16_k16`

- 输出 tile 为 M32，activation tile 为 N16，沿 K 每次处理 16。
- 128 个线程映射为 `32 output rows x 4 q4 groups`。
- 固定 q4 时，bundle 中 16 个 row lane 连续；一个 SIMD group 的读取会落在连续 row-lane 区域。
- 每个线程读取 U/W 两个 branch 的一个 byte，并一次展开其中四个连续 K code，写入四个 32x16 threadgroup coefficient tile。
- 一个 physical tile 的 U/W FP16 scale 只按 tile 读取，不再从每行 V2 block 中重复读取。
- coefficient tile 与已有 staged half activation 通过 `simdgroup_half8x8` MMA 累加，保留原 W1 prefill 的数值路径和 BF16 packed-complex 输出。

### Decode：`kernel_fairy2i_bundle_w1_bf16_tile8x1_w8_full_nobias_fc_simd`

- 每个 threadgroup 计算 8 个输出 row，使用 8 个 K-block slots，共 128 线程。
- 每个 q4 lane 读取四个连续 activation，即 `4*q4 + {0,1,2,3}`；这是 bundle 与旧 V2 byte 语义最重要的区别。
- 对每个 branch 使用一个 `uint2` 读取 8 个连续 row code，全部 64-bit 数据读取被拆成 32-bit 提取，避免 M4 上较慢的 64-bit shift。
- U/W 的四个 FP16 scale 以一个 `half4` 读取。
- blocks、input stride 与 output stride 通过 Metal function constants 固化，消除主模型无 bias decode 中的运行时分支。
- 每个 SIMD group 先归约 K，再由小型 threadgroup buffer 合并并输出 BF16 complex pair。

带 bias 的 decode 使用 `kernel_fairy2i_bundle_w1_bf16_tile8x1_w16_full_simd`；它也直接读取 bundle，只是不采用 no-bias function-constant 专用化。

### Decode 调优搜索

同一模型、同一 M4 上的 tg128 搜索结果如下。表中值是各次 `llama-bench` 报告的均值；只有最终 8-slot 版本进入推荐实现。

| 方案 | tg128 tok/s | 结论 |
| --- | ---: | --- |
| 通用 kernel，64-bit code 提取 | 24.50 | 起点 |
| function constants，64-bit code 提取 | 24.70 | 改善但仍回退 |
| 32-bit code 提取，16 slots | 24.88 | 接近 V2 |
| scale 仅 lane0 读取再 shuffle | 23.50 | shuffle 成本过高，淘汰 |
| 4 output rows | 24.29 | threadgroup 数量增加，淘汰 |
| 32 block slots / 512 threads | 24.70 | occupancy/归约开销过高，淘汰 |
| 8 block slots / 128 threads | 25.00 | 采用 |
| 4 block slots / 64 threads | 25.00 | 无额外收益，选择循环更少的 8 slots |

## 正确性与格式验证

### 全文件 canonical 对照

验证命令：

```bash
.venv/bin/python -u gguf-py/validate_fairy2i_bundle_v1.py \
  /Users/a1806/llama/qwen_1bit_scale/qwen3-fairy2i-w1-learned-scale.gguf \
  /Users/a1806/llama/qwen_1bit_scale/qwen3-fairy2i-w1-bundle-v1.gguf
```

结果：

```text
linear_count=252
common_tensor_count=147
tensor_count=651
canonical_sha256=ae4e8890c419d637e3fd7b339ba7f730e401f14de00e5a2e89ba002cbb449709
```

验证器对每个 M64 strip 执行以下检查：

- 解码 V2 residue-lane byte order；
- 重排并重打包为 bundle 的 consecutive-K byte order；
- 与 bundle code 逐字节比较；
- 检查 V2 的 64 行 scale 确实重复；
- 与 bundle 中唯一一份 scale 以 FP16 bit pattern 比较；
- 对非布局 tensor 直接逐字节比较。

### 后端与端到端测试

| 测试 | 结果 |
| --- | --- |
| Python pack/converter tests | 9/9 PASS |
| Metal Release `test-fairy2i` | PASS；含 bundle N=1 decode、bias decode、N=17 prefill |
| Metal loader test | PASS |
| CPU-LUT Release `test-fairy2i` | PASS |
| CPU-LUT loader test | PASS |
| 512-token perplexity | V2 = bundle = `15.5413 +/- 3.66521` |
| 完整 Qwen3-8B 全层 Metal offload | PASS |

`git clang-format --diff` 无输出。当前机器没有安装 `clang-tidy`，因此未能执行该项静态检查。

## 最终性能结果

命令模板：

```bash
./build-rel-metal/bin/llama-bench \
  -m MODEL.gguf -ngl 99 -t 8 -fa 1 -p 256 -n 128 -r 5 -o json
```

顺序为 `V2-A -> bundle-A -> bundle-B -> V2-B`，每个布局共 10 个 pp 和 10 个 tg 样本，默认 warmup。

| 布局 | pp256 tok/s | tg128 tok/s | 相对 V2 |
| --- | ---: | ---: | ---: |
| tile64_v2 | 176.7084 | 24.1391 | 基线 |
| bundle_v1 | 191.0285 | 25.0067 | pp +8.1038%，tg +3.5941% |

tg 结果需要结合热状态解释：V2-A 的前三个样本约 25.00，随后下降，V2-B 的五个样本均约 23.61；两次 bundle 的十个样本保持在 24.9895–25.0754。保守结论是 bundle decode 至少追平 V2 的冷态最佳性能，并在本机持续负载下表现出更好的稳定性；不能把完整 +3.5941% 全部归因于单一指令级优化。

`llama-bench` 对 bundle 显示的 parameter count 是物理 codes/scales tensor 的元素数，不代表模型逻辑参数量；布局间不要用该列比较模型规模。

## 原始产物

本地原始日志均保留在：

```text
tmp/fairy2i-bundle-metal/convert-bundle-v1.log
tmp/fairy2i-bundle-metal/validate-bundle-v1.log
tmp/fairy2i-bundle-metal/final-test-fairy2i-metal.log
tmp/fairy2i-bundle-metal/v2-perplexity.log
tmp/fairy2i-bundle-metal/bundle-perplexity.log
tmp/fairy2i-bundle-metal/final-abba-v2-a.json
tmp/fairy2i-bundle-metal/final-abba-bundle-a.json
tmp/fairy2i-bundle-metal/final-abba-bundle-b.json
tmp/fairy2i-bundle-metal/final-abba-v2-b.json
tmp/fairy2i-bundle-metal/final-abba-summary.json
```

## 已知限制与后续工作

- 当前 Metal bundle 实现只支持 W1 (`U0,W0`)；W2 (`U0,U1,W0,W1`) 仍是 CPU-LUT-only。
- M/K 必须是 64 的倍数，且 bundle metadata/shape 必须完全匹配。
- bundle 路径仍不支持 LoRA adapter，这与当前 graph 约束一致。
- 性能结论只在 Apple M4、当前 Qwen3-8B shape 上验证；其他 Apple GPU family 应重新测量 decode slots。
- 若后续实现 W2 Metal，应该新增独立的四 branch kernel，不应在加载时把 bundle 还原为四份 V2 权重。
