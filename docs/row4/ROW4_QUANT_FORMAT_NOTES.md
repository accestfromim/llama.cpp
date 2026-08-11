# Qwen3 Row4 W1A8-INT8 v1 格式与实现笔记

> 状态：已实现的 normative v1 合同；本文描述当前 converter、GGUF、loader、graph、ARM CPU 和 Metal 必须共同遵守的格式。
>
> 更新日期：2026-08-09
>
> 验证边界：converter 的真实 checkpoint 抽样已通过独立 bit-exact oracle；operator/loader 测试已落地，最终 CPU/Metal 路径的 pp512/tg128 性能和贪心生成 smoke test 也已记录。缺失 BOS metadata 导致的早期重复/破损输出已修复，但原 H100/FA3 环境的 full-model reference 尚未捕获，CPU eager attention 与 Metal FA3 的完整模型 logits 仍有分歧，因此 logits、NLL 和 PPL **尚未签收**。

本文中的“必须”“禁止”是 v1 格式要求，不是后续优化建议。当前实现只面向本 checkpoint 的固定 Qwen3 profile 和 Apple M4 上的 ARM64 CPU/Metal 路径。

## 1. 权威结论与证据边界

1. 当前模型的部署计算是 **W1A8-INT8**。Row4 权重解码成小范围 signed INT8，activation 动态量化为 signed INT8，点积以 INT32 累加；可读的 FP8 kernel 只用于参考实现结构、padding 等细节，不能改变 INT8 判断。
2. checkpoint 是 QAT checkpoint，不是已经打包的 1-bit 文件。它保留 BF16 latent weight 和每个 Row4 projection 的 learned signed BF16 row scale；converter 离线生成 Row4 code。
3. `lm_head` 已在 converter 中离线做 W8 量化，推理必须走 W8A8 INT8 路径，不允许退回 BF16/F16 dense output。
4. 本模型所有目标维度都满足 O/K 的 128 对齐。v1 不提供非对齐 fallback，也不接受非对齐 GGUF。
5. 训练时的 `row4_qat` package 没有随 checkpoint 提供。当前独立 oracle 可锁定 code、packing、A8、INT32、scale 和 BF16 boundary，但不能冒充原训练环境的完整 Python forward oracle。

证据优先级是：已确认的 INT8 事实和 checkpoint tensor schema，随后是当前 v1 metadata/numeric contract 和独立 oracle；`~/row4_1bit_kernel` 中的 FP8 源码只作为实现参考。

## 2. checkpoint profile

### 2.1 固定模型结构

| 字段 | v1 值 |
| --- | ---: |
| architecture | Qwen3 causal LM |
| layers | 36 |
| hidden size | 4096 |
| intermediate size | 12288 |
| attention heads / KV heads | 32 / 8 |
| head dim | 128 |
| vocab size | 151936 |
| checkpoint dtype | BF16 |
| RoPE | YaRN scale factor 4，original context 32768 |
| YaRN attention factor | metadata pre-factor 1.0；effective 1.138629 |

GGUF 中的 `qwen3.rope.scaling.attn_factor=1.0` 是输入 YaRN kernel 的 pre-factor，不是最终 magnitude scale。scale factor 4 在 ggml 中对应 `freq_scale=1/4`；启用 YaRN extrapolation 时，kernel 再乘：

```text
1 + 0.1 * ln(1 / freq_scale) = 1 + 0.1 * ln(4) = 1.138629436...
```

因此 effective attention factor 是约 `1.138629`。把 metadata 的 `1.0` 直接解释为最终 factor 会漏掉 ggml 内部的 YaRN 修正。

每层的 7 个 Row4 projection 是 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj` 和 `down_proj`；本 profile 不含 projection bias。

### 2.2 tensor 与 learned scale

- safetensors 共 651 个 tensor，全部为 BF16。
- 399 个 `.weight`：每层 7 个 projection 和 4 个 norm，再加 embedding、final norm、lm_head。
- 252 个 `.weight_scale`：`36 * 7`，形状始终为 `[out_features]`。
- scale 共 1,400,832 个，统计为：

  ```text
  min         = -0.0078125
  max         =  0.1455078125
  mean        =  0.0220746489
  non-finite  = 0
  <= 0        = 15,340
  ```

- checkpoint index 记录 `total_parameters=8,192,136,192`、`total_size=16,384,272,384`。这说明源文件是保留 latent BF16 weight 的训练 checkpoint。

Row4 scale 是 learned 参数，可能为负或零。converter 必须 bit-preserving 地复制 BF16 payload；禁止 `abs`、clamp、重估或先转换为其他低精度再写回。

## 3. Row4 codebook 合同

### 3.1 4-row 分组

令 latent weight 为 `W[O,K]`。对每个输入列 `k` 和连续 4 个输出行：

```text
a = W[4g + 0, k]
b = W[4g + 1, k]
c = W[4g + 2, k]
d = W[4g + 3, k]

u_re = 0.5 * (a + d)
u_im = 0.5 * (c - b)
v_re = 0.5 * (a - d)
v_im = 0.5 * (b + c)
```

对 `u`、`v` 各保留绝对值较大的实轴或虚轴。轴编号固定为：

| axis id | 含义 |
| ---: | --- |
| 0 | `+R` |
| 1 | `-R` |
| 2 | `+I` |
| 3 | `-I` |

编码固定为：

```text
code = u_axis | (v_axis << 2)
```

必须保持两个边界规则：

- `abs(real) == abs(imag)` 时选择 imag，因为 real 分支条件是严格的 `>`。
- `+0` 和 `-0` 都选择正轴，因为负号只由 `< 0` 判定。

### 3.2 固定 16-code 解码表

每个 nibble 解码为连续 4 个输出行的 signed INT8 值：

| code | q0 | q1 | q2 | q3 |
| ---: | ---: | ---: | ---: | ---: |
| `0` | 2 | 0 | 0 | 0 |
| `1` | 0 | 0 | 0 | -2 |
| `2` | 1 | -1 | 1 | -1 |
| `3` | 1 | 1 | -1 | -1 |
| `4` | 0 | 0 | 0 | 2 |
| `5` | -2 | 0 | 0 | 0 |
| `6` | -1 | -1 | 1 | 1 |
| `7` | -1 | 1 | -1 | 1 |
| `8` | 1 | 1 | 1 | 1 |
| `9` | -1 | 1 | 1 | -1 |
| `A` | 0 | 0 | 2 | 0 |
| `B` | 0 | 2 | 0 | 0 |
| `C` | 1 | -1 | -1 | 1 |
| `D` | -1 | -1 | -1 | -1 |
| `E` | 0 | -2 | 0 | 0 |
| `F` | 0 | 0 | -2 | 0 |

有效权重为：

```text
W_eff[o,k] = row_scale[o] * decode(code[o/4,k])[o%4]
```

一个 nibble 表示 4 个原始实权重，即 code plane 为 1 bit/weight；per-row BF16 scale 是额外开销。

## 4. `bf16_a8_away_i32_bf16_v1` 数值合同

`GGML_OP_ROW4_LINEAR` 和 `GGML_OP_W8A8_LINEAR` 使用同一 activation/output 数值边界。其输入、输出 tensor 都是 F32 carrier，但 carrier 中的有效值必须先后经过 BF16 boundary。

对每个 token 的完整 K 行，计算顺序固定为：

1. `x_f32 -> BF16 RNE -> F32`，之后的 amax 和量化只能使用 round-trip 后的值。
2. 在 F32 中计算 `sx = max(amax(abs(x_bf16)) / 127, 1e-8)`。
3. `qx = clamp(round_half_away_from_zero(x_bf16 / sx), -127, 127)`，写为 signed INT8。
4. 将 Row4 code 解码为 `{0,+/-1,+/-2}` 的 INT8，或直接读取 lm_head signed INT8 code；点积使用 INT32 accumulator。
5. 依次执行两个 F32 乘法：先 `float(acc) * sx`，再乘 `row_scale[o]`。不得把两个 rescale 乘法改写为会改变舍入点的 contraction。
6. `result_f32 -> BF16 RNE -> F32 carrier` 后写入输出。

half-away-from-zero 可规范化为：

```text
round_away(x) = copysign(floor(abs(x) + 0.5), x)
```

必须覆盖 `+/-0.5`、饱和、全零 token、BF16 halfway/RNE、只在 F32 低位不同但 BF16 相同的输入，以及负 Row4 scale。全零 activation 仍使用 `sx=1e-8`，code 全为 0。

### 4.1 lm_head 的离线 W8 合同

`lm_head.weight` 先由 BF16 bit pattern widen 到 F32，再逐输出行量化：

```text
sw[o]    = amax(abs(weight[o,:])) / 127       # F32
qw[o,k]  = clamp(round_away(weight[o,k] / sw[o]), -127, 127)
```

全零行必须写全零 INT8 code 和 `sw=0.0f`。`sw` 以 F32 保存；推理时仍使用上述 A8/INT32/BF16-boundary 合同。

## 5. GGUF v1 合同

### 5.1 enum 和严格 metadata

| 项目 | 固定值 |
| --- | --- |
| `GGML_TYPE_ROW4_CODES` | 47，opaque `(block_size,type_size)=(1,1)` |
| `LLAMA_FTYPE_MOSTLY_ROW4` | 43 |
| Row4 op | `GGML_OP_ROW4_LINEAR` |
| lm_head op | `GGML_OP_W8A8_LINEAR` |
| `general.architecture` | `qwen3` |
| `general.file_type` | 43 |
| `general.alignment` | 64 |
| `general.quantization_version` | 2 |
| `tokenizer.ggml.add_bos_token` | `true` |
| `row4.schema_version` | 1 |
| `row4.weight_layout` | `m16k128_split8_v1` |
| `row4.codebook` | `uv_axis_v1` |
| `row4.numeric_profile` | `bf16_a8_away_i32_bf16_v1` |
| `row4.qkv_order` | `q_k_v` |
| `row4.ffn_order` | `gate_up` |
| `row4.lm_head_layout` | `s8_m16k128_rowmajor_v1` |

量化 GGUF 必须写 `general.quantization_version=2` 和 `tokenizer.ggml.add_bos_token=true`，converter 和独立 verifier 都对此做硬校验。本 checkpoint 的 Hugging Face chat template 以 BOS 开始；如果 GGUF 丢失 `add_bos_token`，runtime chat prompt 会与训练/模板语义不同，并在真模型生成中表现为重复或破损输出。只要文件出现任一 Row4 marker，loader 就按严格 v1 解析；Row4 descriptor 缺失、值不匹配、type/shape 不匹配或出现被禁止的 dense 对应 tensor，都必须在加载阶段报错。

`GGML_TYPE_ROW4_CODES` 是物理 byte plane，不是普通量化 matrix type。它只允许 dedicated Row4 op 和同类型 copy，禁止进入 generic `MUL_MAT`、generic dequantizer 或通用 type-traits 路径。tiled lm_head 虽使用 `GGML_TYPE_I8`，也必须由 `W8A8_LINEAR` 消费，不能因其基础 dtype 是 I8 而路由到 generic matmul。

### 5.2 Row4 `m16k128_split8_v1`

逻辑 code 形状为 `[O/4,K]`，GGML 物理 `ne` 固定为：

```text
[64, 4, K/128, O/16]
```

外层顺序为 `output_tile, k_tile, four_row_group, k16_subblock, byte`。对 `Kt=K/128`：

```text
ot      = o / 16
kt      = k / 128
g       = (o % 16) / 4
sub16   = (k % 128) / 16
j       = (k % 16) % 8

byte_offset = ((((ot * Kt + kt) * 4 + g) * 8 + sub16) * 8 + j)
```

每个 K16 子块的 8 bytes 使用 split8，而不是相邻 nibble：

```text
low  nibble = code[group, kt*128 + sub16*16 + j]
high nibble = code[group, kt*128 + sub16*16 + 8 + j]
```

因此每个 four-row group 的 K128 code 占 64 bytes，每个 M16K128 tile 占 256 bytes。已知答案 `code=0..F` 的 K16 流必须打包为 `80 91 A2 B3 C4 D5 E6 F7`。

Row4 tensor pair 命名为：

```text
<base>.row4.codes   # GGML_TYPE_ROW4_CODES
<base>.row4.scales  # BF16 [O]，signed、bit-preserving
```

### 5.3 bundle 顺序与固定 shape

converter 合并 Q/K/V 和 gate/up，以减少 tensor 数并保持完整 M16 tile 流：

| GGUF base | logical `[O,K]` | physical code `ne` | scale 顺序 |
| --- | --- | --- | --- |
| `blk.L.attn_qkv` | `[6144,4096]` | `[64,4,32,384]` | 完整 Q rows，随后 K、V |
| `blk.L.attn_output` | `[4096,4096]` | `[64,4,32,256]` | O rows |
| `blk.L.ffn_gate_up` | `[24576,4096]` | `[64,4,32,1536]` | 完整 gate rows，随后 up |
| `blk.L.ffn_down` | `[4096,12288]` | `[64,4,96,256]` | down rows |

Q、K、V 和 gate、up 的组件边界都满足 O 的 128 对齐，不会跨 M16/four-row group 边界。禁止交织 Q/K/V 或 gate/up tile。

### 5.4 lm_head `s8_m16k128_rowmajor_v1`

lm_head tensor pair 固定为：

```text
output.w8.codes   # I8, ne=[128,16,K/128,O/16]
output.w8.scales  # F32 [O]
```

本模型是 `K=4096`、`O=151936`，所以 code `ne=[128,16,32,9496]`。物理顺序为 `output_tile, k_tile, row_in_tile, k_in_tile`，每个 M16K128 tile 内是 16 个连续的 K128 signed INT8 row。它不是普通逻辑 `[K,O]` I8 tensor。

### 5.5 参考 GGUF tensor/payload 统计

固定 profile 必须产生 436 个 tensor：

```text
1 embedding
+ 36 * (4 BF16 norms + 4 Row4 code/scale pairs)
+ 1 final norm
+ 2 lm_head W8 tensors（1 code + 1 scale）
= 436
```

| payload 类别 | bytes | 约 MiB |
| --- | ---: | ---: |
| Row4 code planes | 868,220,928 | 828.00 |
| Row4 signed BF16 scales | 2,801,664 | 2.67 |
| lm_head W8 codes | 622,329,856 | 593.50 |
| lm_head F32 scales | 607,744 | 0.58 |
| BF16 embedding/norm | 1,245,276,160 | 1,187.59 |
| **tensor payload 合计** | **2,739,236,352** | **2,612.34** |

最终冻结产物是：

```text
/Users/a1806/llama/tmp/qwen3-row4-int8-v1-final-bos.gguf
size    = 2,745,197,632 bytes
SHA-256 = 306b3086b28251cf662c462e0bc2d4e153b2a517593a651cf95f3887a62b5deb
```

该文件含完整 436 个 tensor、上表的固定 payload 统计、`general.quantization_version=2` 和 `tokenizer.ggml.add_bos_token=true`。规范性检查以 tensor count、各类 payload、metadata 与 SHA-256 为准，不以文件系统稀疏占用为准。

## 6. converter

专用入口是 `gguf-py/convert_row4_qwen3.py`。源 tar 应先解压到仓库外目录；converter 需要 NumPy、PyTorch、SafeTensors 和本仓库的 `gguf` Python package。

```sh
python3 gguf-py/convert_row4_qwen3.py /path/to/extracted-checkpoint --dry-run

python3 gguf-py/convert_row4_qwen3.py \
  /path/to/extracted-checkpoint \
  /path/to/qwen3-row4-int8.gguf \
  --verbose
```

converter 的 v1 职责是：

- 校验固定 Qwen3 config、4 个 shard、完整 tensor schema、tokenizer、可用空间和 64-byte alignment；
- 直接从 BF16 latent weight 生成 canonical Row4 code，保留 signed BF16 scale bits；
- 按 `q_k_v` / `gate_up` 生成 bundle；
- 将 lm_head 离线量化为 tiled W8 + F32 scale；
- 流式写入私有临时目录，成功后以原子 no-clobber hard link 发布；拒绝覆盖已有目标；
- 对参考 profile 核对 436 tensors 和固定 payload 统计。

普通 `convert_hf_to_gguf.py` 会识别并拒绝 specialized Row4 checkpoint，`llama-quantize` 也会拒绝 Row4 GGUF。不得用通用 converter/quantizer 重解释或 requantize 此格式；必须从原 checkpoint 重新运行专用 converter。

## 7. loader 与模型 graph

### 7.1 loader/backend gate

Row4 仍使用 `general.architecture=qwen3`，但 `file_type=43` 和完整 Row4 descriptor 选择专用模型路径。loader/model/context 的严格 gate 是合同的一部分：

- 严格检查每个 code/scale pair 的名字、type、shape、O/K 和组件顺序；
- 禁止 projection dense weight、dense lm_head、bias 和缺半边 tensor pair；
- 对实际分配到的每个设备调用 `supports_op` probe，只有 native ARM CPU 或支持两个专用 op 的 Metal backend 才可继续；
- K/V cache 必须同时为 BF16；任何其他 `type_k/type_v` 都在 context 初始化时拒绝；
- 全 CPU placement 必须禁用 Flash Attention；`AUTO` 会被明确改为 disabled，显式 enabled 会报错；
- 全 Metal placement 必须设置 `offload_kqv=true` 并启用专用 BF16 exact Flash Attention，预留 graph 还会验证 FA node 没有被 scheduler 重派到 CPU；
- mixed CPU/Metal placement 和 tensor buffer override 都会被拒绝，避免形成未经验证的跨 backend route；
- 不允许将 Row4 tensor 调度到 x86 CPU、CUDA、RPC 或不支持专用 op 的设备，也不做 silent fallback。

### 7.2 graph

专用 Qwen3 Row4 graph 直接消费四类 Row4 bundle，并以 view 拆分 Q/K/V、gate/up：

- schema 固定的 BF16 token embedding 由 `GET_ROWS` 精确上转后已经满足 F32 carrier invariant，不再追加恒等 BF16 round；任意 F32 外部 embedding 仍在进入首个 residual 前做 BF16 RNE round-trip。
- QKV 保持 checkpoint 的 Q 后 K 后 V 顺序。K/V 的 F32 carrier 由 Row4 专用 BF16-carrier `SET_ROWS` 直接按 stride 写入 BF16 cache；生产 graph 不再生成 K/V `PACK`、V `CONT` 或与它们对应的中间 buffer。
- GGUF 中的 norm weight 保留 BF16；norm、Q/K RoPE、residual add、SiLU 和 multiply 使用 QAT exact primitive，线性层之间保持 F32 carrier + BF16 boundary。
- gate/up 保持 fused projection 的 strided view；Metal 的 Row4 专用 QAT SwiGLU kernel 直接消费这两个 view，在一个 dispatch 内执行 exact SiLU 与 multiply，不再生成 gate/up `CONT`。CPU 仍保持相同数值边界。
- attention output、down 和两个 fused projection 都调用 `GGML_OP_ROW4_LINEAR`。
- 最终 output 只调用 `GGML_OP_W8A8_LINEAR`，没有 dense lm_head fallback。
- LoRA 在 adapter 加载与 graph 建立时明确拒绝；control-vector adapter 也会在 Row4 graph 中报错，不能越过 BF16 contract。

## 8. backend 实现路径

### 8.1 ARM64 CPU

CPU 实现位于 dedicated Row4 runtime，提供：

- `scalar`：只用于 oracle/测试强制路径；不是生产 fallback。
- `dotprod`：单 token decode 的默认 ARM path。
- `i8mm`：可用时用于多 token；`B>=8` 使用按 token-pair/K8 排列的临时 eight-row panel，使一次 activation load 同时服务八个输出行，较小 batch 使用 direct path。

测试可用：

```sh
GGML_ROW4_TEST_FORCE_PATH=scalar  # 或 dotprod / i8mm
GGML_ROW4_CPU_DEBUG=1
```

强制一个当前 build 不支持的 ISA 必须明确失败，不能悄悄切到其他路径。debug marker 按 `(op,path,B,O,K,panel)` 在进程内去重，格式为：

```text
row4_cpu: op=<row4|w8a8> path=<scalar|dotprod|i8mm> \
layout=<m16k128_split8_v1|s8_m16k128_rowmajor_v1> \
B=<B> O=<O> K=<K> nth=<threads> \
aqpack=<bf16_rne_a8_away_v1|bf16_rne_a8_away_pairk8_v1> panel=<0|1> prepack=0
```

当前实现每次 op 量化 activation；没有持久化 weight prepack，marker 固定 `prepack=0`。

### 8.2 Metal

未被 producer fusion 覆盖的 Metal linear 首先运行：

```text
kernel_row4_quantize_activation_i8
```

随后按 token 行数和 production shape 选择 Row4 或 W8 kernel：

| token rows `T` | Row4 | lm_head W8 |
| ---: | --- | --- |
| 1 | `kernel_row4_w1a8_decode_o32_o4_staged_act` | `kernel_row8_w8a8_decode_o128_rows16` |
| 2..8 | `kernel_row4_w1a8_small_batch` | `kernel_row8_w8a8_small_batch` |
| >=9 | M64N32 native-layout dual-W direct-act prefill | `kernel_row8_w8a8_prefill` |

decode/small-batch 使用 A8/I32。本 profile 的 Row4 decode 以 O32 threadgroup 为单位，8 个 SIMDgroup 各处理一个 O4，并把 K4096/K12288 activation 一次性 staged 到 threadgroup memory。W8 decode 每个 O128 threadgroup 使用 8 个 SIMDgroup，每组计算一个 canonical O16 tile，即每个 SIMDgroup 处理 16 行。

pp512 的 Row4 production path 先把 token-major INT8 activation 用 32x32 coalesced transpose 转为 K-major half，再用 `kernel_row4_w1a8_prefill_m64n32_ilp4_native_layout_dualw_direct_act_simd_localw`。该 kernel 每个 threadgroup 计算 M64N32；每个 SIMDgroup 只 staging 自己消费的 16-row weight slice，并直接消费 transpose 后的 activation，避免 token tile 重复 staging 和跨 SIMDgroup 的 weight barrier。gate/up 的专用 producer kernel 保持相同 MMA/舍入顺序，顺序计算配对的 gate 与 up，执行 exact QAT SiLU/multiply 后把 packed BF16 直接交给 down projection。非 32 倍数的 prefill rows 保留同族 native-layout fallback。Row4 prefill 用 half MMA/F32 exact accumulation；在当前最大 `K=12288` 下极值 `2*127*K=3,121,152 < 2^24`。W8 prefill 以 K1024 分段做 F32 exact accumulation，再转/合并为 I32；完整 `K=4096` 极值为 66,064,384，不能用单个 F32 accumulator 冒充全 K 精确 INT32。

单 token production graph 还保留三组严格门控的逐 bit fusion：72 个 QAT RMSNorm→Row4 activation-quant 链共享一个 dispatch，同时仍物化原 RMS carrier；35 个 attention output 和 36 个 down projection 把 QAT residual add 合入 Row4 epilogue；36 个 QAT SwiGLU→down 链使用 packed BF16 handoff。任一节点有额外 consumer、被标记为 output、shape/layout 不匹配或 scratch 与输入/输出 allocation 重叠时都回退到分离路径。

这些 kernel 与 `s8_m16k128_rowmajor_v1` / `m16k128_split8_v1` 物理布局直接对应。Row4/W8 host 使用静态 pipeline 名，不在每个 op 重复格式化；pipeline cache 用 allocation-free 的 64-bit hash lookup，并在碰撞桶内比较完整名称。path marker 先查 thread-local shape cache，只有首见 shape 才进全局 mutex/set，避免 decode 热路径反复分配或串行化。

首次遇到每个 `(op,path,T,O,K)` shape 时打印：

```text
ROW4 Metal W1A8 path: <decode|small_batch|prefill> \
layout=m16k128_split8_v1 act_rows=<T> O=<O> K=<K> (..., BF16 boundary)

ROW8 Metal W8A8 lm_head path: <decode|small_batch|prefill> \
layout=s8_m16k128_rowmajor_v1 act_rows=<T> O=<O> K=<K> (..., BF16 boundary)
```

Q/K RMSNorm→RoPE 融合候选不属于冻结路径：它在独立 operator 试验后的真模型重复运行中出现非确定性，因此已完全回退。production graph 仍以分离的 exact RMSNorm 和 RoPE 执行 Q/K 路径。

## 9. 正确性验证与 oracle

### 9.1 Python converter/packing tests

```sh
PYTHONPATH=gguf-py python3 -m pytest gguf-py/tests/row4 -q
```

测试必须覆盖 16 codes、tie/sign-zero、split8 known answer、signed BF16 scale、lm_head W8 half-away/zero-row、bundle 顺序、tensor 统计、preflight 和写入失败的原子性。

### 9.2 C++ operator/loader tests

CPU build：

```sh
cmake -B build-rel -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF
cmake --build build-rel --target test-row4 test-row4-loader -j "$(sysctl -n hw.ncpu)"
ctest --test-dir build-rel -R '^test-row4' --output-on-failure
```

Metal build 和强制实尺寸矩阵：

```sh
cmake -B build-rel-metal -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build-rel-metal --target test-row4 -j "$(sysctl -n hw.ncpu)"

LLAMA_ROW4_REQUIRE_METAL_TESTS=1 \
LLAMA_ROW4_REAL_SHAPE_TESTS=1 \
LLAMA_ROW4_FULL_LM_HEAD_TESTS=1 \
  ./build-rel-metal/bin/test-row4
```

这些测试的 acceptance criteria 是 CPU scalar/dotprod/i8mm 与独立 reference exact 一致，Metal 在 `T=1,2,8,9` 的 dispatch 边界 exact 一致，并覆盖 Row4 `K=12288` 极值/抵消和 W8 `K=4096` 跨 K1024 分段的极值/抵消。loader tests 还必须证明 descriptor/type/shape/backend 错误被早拒绝，opaque Row4 和 tiled W8 不会逃逸到 generic matmul。

### 9.3 独立 pure-Torch oracle

入口为 `scripts/row4/row4_oracle.py`：

```sh
python3 scripts/row4/row4_oracle.py self-test

export ROW4_ORACLE_DIR=/absolute/external/row4-oracles
python3 scripts/row4/row4_oracle.py capture \
  --checkpoint /path/to/extracted-checkpoint \
  --input-ids 1,2 \
  --run-name checkpoint-primitives

python3 scripts/row4/row4_oracle.py verify \
  --checkpoint /path/to/extracted-checkpoint \
  --gguf /path/to/qwen3-row4-int8.gguf \
  --run-name gguf-verified
```

oracle 不导入缺失的 `row4_qat`、FP8 kernel 或 converter 的 quant helper。它从真实 checkpoint 读取完整的选定 O16 x K tile，独立重算 Row4/W8 code、packing、A8、INT32 和 BF16 output，并比对 metadata、packed bytes 与 scale bits。

当前已有的真实 checkpoint 证据覆盖完整 GGUF metadata/tensor inventory/payload、layer 0/35 的 QKV/O/gate_up/down 组件首尾与 scale 边界，以及 lm_head 选定 O16 x 完整 K。最终 manifest 中 99 项 metadata、inventory、packed code、signed BF16 scale 和 F32 lm_head scale 检查均为 bit-exact PASS。外部 artifact 位于：

```text
/Users/a1806/llama/tmp/row4-oracle-runs/final-bos-quality-perf
```

该 artifact 以最终 BOS GGUF 为验证对象，包含 322 个 raw tensor，以及 4 个 shard、config、tokenizer 的完整 SHA-256；它不是模型资产，禁止提交到仓库。

默认 activation 是由 `input_ids` 键控的 deterministic F32 probe，并非 checkpoint 完整 forward 的 hidden state。若要检查同一真实运行输入，可提供 F32 carrier bundle：

```sh
python3 scripts/row4/row4_oracle.py capture \
  --checkpoint /path/to/extracted-checkpoint \
  --input-ids '[151644, 9707]' \
  --activations /path/to/f32-carriers.pt \
  --run-name captured-carriers
```

### 9.4 原 checkpoint modeling 的 full-model capture

`scripts/row4/full_model.py` 为完整 forward 提供独立入口。它在导入 checkpoint 自带、未修改的 `modeling_qwen3_row4_int8.py` 前，注入与 checkpoint state-dict schema 兼容的透明 `row4_qat.Row4Int8Linear` / `Int8Linear` shim。shim 只补足缺失的量化 linear package，并严格复用第 4 节的 BF16/A8/I32 合同；它不会把缺失的训练源码伪装成已找回。

先用 tiny tensor 做 BF16 bit-level self-test：

```sh
python3 scripts/row4/full_model.py self-test
```

可签收的 reference 必须在 `transformers==5.2.0`、Accelerate、CUDA Hopper 和可工作的 Flash Attention 3 环境中生成。加载 16 GB 权重前先执行严格 preflight：

```sh
python3 scripts/row4/full_model.py preflight \
  --checkpoint /path/to/extracted-checkpoint \
  --device cuda:0 \
  --attn-implementation flash_attention_3
```

在原 H100/FA3 环境中捕获固定 prefill 与随后两个 decode step：

```sh
export ROW4_ORACLE_DIR=/absolute/external/row4-oracles
python3 scripts/row4/full_model.py capture \
  --checkpoint /path/to/extracted-checkpoint \
  --device cuda:0 \
  --attn-implementation flash_attention_3 \
  --input-ids 151643,785,4226,374 \
  --decode-ids 220,19 \
  --cache-decoded-row4 \
  --run-name fa3-prefill-decode-golden
```

capture 会导出三阶段 logits、layer 0/35 与 lm_head 的真实 F32 carrier、argmax/top-10、NLL/PPL、环境与模型 hash。只有严格 preflight 全部满足时 manifest 才会写 `reference_capture=true`；`--allow-nonreference` 产生的诊断输出不能作为 H100/FA3 golden。

当前 Apple M4 主机不是 reference 环境：本机 Transformers 5.14.1 与 checkpoint 记录的 5.2.0 不同，没有 Accelerate、CUDA/Hopper 或可检测的 FA3，且 24 GB unified memory 对 16.38 GB BF16 checkpoint 的完整加载没有安全余量。因此这里只运行了 shim self-test 与只读 preflight，**尚无** `reference_capture=true` 的 full-model artifact。

H100 reference 与 CPU/Metal runtime logits 都导出后，用同一 comparator 验收：

```sh
python3 scripts/row4/row4_oracle.py compare-logits \
  --reference /path/to/reference.pt \
  --candidate /path/to/runtime.pt \
  --output /path/to/metrics.json
```

当前已知的第一个 CPU/Metal 分叉位于 layer 0 attention：CPU eager exact sum 为 `-48.652618`，Metal FA3 为 `-48.652451`，量化级联后固定 corpus 的 KLD 约为 `0.19355`。Row4/W8 linear operator 的 bit-exact PASS 不能消除 attention schedule 的差异；在 H100 reference、CPU 和 Metal 对同一输入通过 logits/NLL/PPL 阈值前，完整模型正确性仍为 **未通过**。

### 9.5 最终 runtime 输出 smoke test

早期 CPU/Metal chat 输出的重复和破损不是 linear kernel 数值错误，而是 GGUF 缺失 `tokenizer.ggml.add_bos_token=true`：Hugging Face chat template 已把 BOS 作为 prompt 的第一个 token，而 runtime 之前以不同 token 序列开始推理。最终 GGUF 补齐该 metadata 后，在冻结代码树上用同一 code prompt 做 32-token greedy generation，CPU 与 Metal 的 stdout 逐 byte 相同，且输出连贯：

```text
/Users/a1806/llama/tmp/row4-quality/current-final/cpu_code.stdout
/Users/a1806/llama/tmp/row4-quality/current-final/metal_code.stdout
```

将生成延长到 128/512 tokens 时，CPU 与 Metal 会因 attention schedule 差异而分叉，但两者仍保持连贯。该 checkpoint 的默认 reasoning 模式在 512 tokens 内仍未走出思考并给出最终答案，因此可以结论“已能正常生成连贯语言”，但 chat UX 的 overthinking 仍未完全签收。该 smoke test 只证明 BOS 已修复原有输出破损；它不替代 H100/FA3 golden，也不构成 full-logits/NLL/PPL 验收。

## 10. 性能验收

统一 benchmark harness：

```sh
perf/scripts/run_row4_bench.sh cpu  pp /path/to/model.gguf row4-v1
perf/scripts/run_row4_bench.sh cpu  tg /path/to/model.gguf row4-v1
perf/scripts/run_row4_bench.sh metal pp /path/to/model.gguf row4-v1
perf/scripts/run_row4_bench.sh metal tg /path/to/model.gguf row4-v1
```

默认 workload 是 `pp512` / `tg128`；CPU 使用 8 threads、3 repetitions，Metal 使用 5 repetitions并执行 cooldown，K/V cache 都显式设为 BF16。harness 会核对 reference code-plane physical shape 和期望 path marker，并保存 JSON、stderr、host/runtime path 和环境 metadata。

冻结版本在 Apple M4 上的正式结果如下。“历史 production”是用户指定的上一代 8B Qwen3 量化速度目标；“严格 legacy”是同一 runtime 下重测并保留为验收目标的 legacy Metal 比较器。严格 legacy artifact 来自当时未归档完整 dirty source snapshot 的 build，因此不是可独立重建的 clean baseline；下表 Row4 final 则全部来自 clean commit `72b7164db`，四份 metadata 都记录 `git_dirty=0`，模型 SHA-256 为 `306b3086b28251cf662c462e0bc2d4e153b2a517593a651cf95f3887a62b5deb`。

| backend | workload | Row4 final tok/s | 历史 production tok/s | 相对历史 | 严格 legacy tok/s | 相对严格 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ARM CPU M4 | pp512 | 71.762123 | 43.582466 | +64.658% | — | — |
| ARM CPU M4 | tg128 | 28.823771 | 20.465132 | +40.843% | — | — |
| Metal M4 | pp512 | 232.310440 | 226.7049 | +2.473% | 229.304120 | +1.311% |
| Metal M4 | tg128 | 29.510607 | 29.0203 | +1.690% | 29.065789 | +1.530% |

因此用户给定的四个历史 production 目标全部达成，两项 Metal workload 也都超过保留的严格 legacy comparator。

正式 raw artifact prefix 为：

| backend | workload | production path marker | artifact prefix |
| --- | --- | --- | --- |
| ARM CPU M4 | pp512 | Row4 `i8mm` pairk8/eight-row panel | `/Users/a1806/llama/tmp/row4-bench/final-clean-72b/row4-final-72b.cpu.pp.20260809T090829Z` |
| ARM CPU M4 | tg128 | Row4/W8 `dotprod` decode | `/Users/a1806/llama/tmp/row4-bench/final-clean-72b/row4-final-72b.cpu.tg.20260809T090920Z` |
| Metal M4 | pp512 | Row4 M64N32 direct-act SIMD-local-W; fused gate/up SwiGLU producer | `/Users/a1806/llama/tmp/row4-bench/final-clean-72b/row4-final-72b.metal.pp.20260809T091012Z` |
| Metal M4 | tg128 | fused RMS→quant/residual/SwiGLU; Row4 O32 staged-act; W8 O128 | `/Users/a1806/llama/tmp/row4-bench/final-clean-72b/row4-final-72b.metal.tg.20260809T091105Z` |

这些 artifact 不属于模型或仓库资产，不得提交。CPU prefill 的主要收益来自 pairk8 activation packing 和八行共享 panel。Metal prefill 的主要收益来自 M64N32 direct-act SIMD-local weight staging、gate/up SwiGLU producer fusion，以及移除 graph 中的 K/V pack/cont。Metal decode 的主要收益来自 Row4 O32 staged activation、W8 O128、72 个 RMS→quant fusion、71 个 residual epilogue fusion 和 36 个 packed SwiGLU/down handoff。静态 pipeline 名、allocation-free cache lookup 与 TLS marker fastpath 进一步减少了 host 侧 decode 开销。所有结果都保留 canonical Row4/W8 path-marker 证据；性能结论不改变第 9.4 节的 full-model blocker。

## 11. 与已有量化方式的区别

| 维度 | Row4 W1A8-INT8 v1 | Fairy2i W1 learned-scale | 普通 GGML block quant |
| --- | --- | --- | --- |
| graph | 普通 real Qwen3 linear | complex widely-linear `U*x + W*conj(x)` | 普通 real matmul |
| code 分组 | 同 K 的连续 4 个 output rows | input/output 两半组成复杂 U/W 象限 | 通常单 row 的 K block |
| code density | 4 bit / 4 real weights | 约 1 bit/original real weight | 依格式而定 |
| scale | signed BF16 `[O]` | U/W tile scale | block-local scale |
| activation | 每 token 整个 K 一个 F32 scale，A8 half-away | complex real/imag 分块 A8 | 常见为 F32/F16 或格式专用 dot |
| accumulation | INT32 | Fairy2i 专用复数路径 | 格式/后端相关 |
| GGUF | opaque Row4 code + 独立 BF16 scale | Fairy2i bundle/tile type | `block_q*` 复合 block |
| op | dedicated real `ROW4_LINEAR` | dedicated widely-linear op | generic `MUL_MAT` 为主 |

Row4 与 Fairy2i 共享“两个复数轴选择”的代数来源，但 4 数的来源、scale 粒度、activation 语义、物理布局和 graph 均不同。禁止复用或重解释 Fairy2i tensor type/kernel。Row4 也不能套用普通每行独立的 `block_q*` dequant contract。

## 12. v1 明确限制与未完成项

- **硬件范围**：当前支持/验收目标仅为 Apple M4 的 native ARM64 CPU 和 Metal。x86 CPU、CUDA、RPC 以及其他未 probe 的 backend 不支持，loader 会拒绝。
- **无 fallback**：projection 和 lm_head 都必须走专用量化 op；不提供 dense、generic matmul、非对齐或跨 backend fallback。
- **固定 runtime route**：K/V cache 只能是 BF16；全 CPU 使用 Flash Attention off，全 Metal 必须 `offload_kqv=true` 并使用 BF16 exact FA3；mixed placement 和 tensor buffer override 都不支持。
- **固定对齐/profile**：O/K 必须是 128 的倍数；只支持本文列出的 36-layer Qwen3 tensor inventory、bundle 顺序和无 bias profile。
- **当前 K 上界**：Row4 production contract 锁到 `K<=12288`，W8 lm_head 锁到 `K=4096`；更大的 K 需要重新证明 accumulator/Metal 分段精确性并升级合同，不能假定 v1 自动支持。
- **无 adapter 扩展**：Row4 graph 明确拒绝 LoRA 和 control-vector adapters。
- **无通用转换链**：不能用普通 HF converter 或 `llama-quantize` 生成、copy 或 requantize Row4 v1。
- **oracle 边界**：缺少原始 `row4_qat` 源码，所以现有 bit-exact oracle 是独立 pure-Torch primitive/layout oracle；`full_model.py` 使用透明兼容 shim，且原 H100/FA3 reference 尚未捕获。
- **整模正确性**：CPU eager attention 与 Metal FA3 已观察到 KLD 分歧，logits、NLL、PPL 尚未签收。
- **生成质量**：BOS 修复后 CPU/Metal 均能生成连贯语言，32-token greedy code prompt 逐 byte 相同；但默认 reasoning 在 512 tokens 内仍未输出最终答案，chat UX 尚未完全签收。
- **拒绝的融合**：Q/K RMSNorm→RoPE 候选因真模型非确定性已完全回退，v1 不允许把该候选当作 production path。
- **性能**：clean commit `72b7164db` 的 pp512/tg128 CPU/Metal 四项均达到历史 production 目标，两项 Metal workload 也超过保留的严格 legacy comparator。后续结果仍必须保留 raw artifact 和 path marker，且性能提升不能替代 full-model logits/NLL/PPL 验收。
