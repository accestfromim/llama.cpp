# Fairy2i bundle_v1 Llama2-7B W2 Metal 迁移报告

日期：2026-07-17

机器：Apple M4，10 核 CPU，Metal（Apple9 / Metal4）

基线提交：`9c35e720`

## 1. 结论

Llama2-7B checkpoint 可以使用当前 Fairy2i 转换器完成 2-bit、两阶段残差（W2）转换。因此本轮已将
`bundle_v1` 的直接加载和 Metal kernel 支持迁移到 W2，没有保留只转换、不能在 Metal 运行的中间状态。

最终 W2 `bundle_v1` 相对旧 `tile64_v2`：

- GGUF 从 `2,551,987,840 B` 降至 `2,153,544,896 B`，减少 `398,442,944 B`（`15.61%`）；
- 224 个线性层的四个量化分支全部位级等价，公共张量也逐字节一致；
- Metal 两轮 R3 进程均值：`pp512 165.10 tok/s`、`tg128 20.29 tok/s`；
- 相对相同 fused runtime 下的旧 V2：`pp512 +15.88%`、`tg128 +2.73%`；
- Metal 直接消费 GGUF 中的 code/scale buffer，不在加载时建立重排权重副本。

## 2. 实验选项清理

布局实验已经收敛到 `m16_q4_branch_lane`。本轮删除了未采用布局的 packer、实验转换入口和 round-trip
测试，也移除了生产转换器、格式声明和校验器中的候选选择项。仓库只保留最终生产布局；历史 Markdown
报告和结果 JSON 继续作为实验记录。

删除的实验代码：

- `gguf-py/experimental/convert_fairy2i_qwen3_metal_layout.py`
- `gguf-py/fairy2i/quant/tile64_v3_metal.py`
- `gguf-py/tests/fairy2i/test_tile64_v3_metal.py`

## 3. Checkpoint 与转换

输入：

```text
/Users/a1806/llama/llama2_7b_int8_activate/v0-20260609-100315/checkpoint-5998
```

主要配置为 32 层、实数 hidden size 4096、实数 FFN size 11008、词表 32006。转换后的复数维度为
hidden 2048、FFN 5504；词表补齐到 32128。所有低比特线性层的 M/K 均满足 64 对齐。

Dry-run：

```bash
.venv/bin/python gguf-py/convert_fairy2i_llama.py \
  /Users/a1806/llama/llama2_7b_int8_activate/v0-20260609-100315/checkpoint-5998 \
  --dry-run --weight-layout bundle_v1 --residual-steps 2 --verbose
```

完整转换：

```bash
.venv/bin/python -u gguf-py/convert_fairy2i_llama.py \
  /Users/a1806/llama/llama2_7b_int8_activate/v0-20260609-100315/checkpoint-5998 \
  /Users/a1806/llama/llama2_7b_int8_activate/fairy2i-llama2-7b-checkpoint5998-bundle-v1.gguf \
  --weight-layout bundle_v1 --residual-steps 2 --verbose
```

没有使用 `--qk-permute`。对 layer 0 Q projection 的 `U.s0` 做独立检查时，checkpoint 原始排列量化后的
SHA-256 为 `9ab049b277c576ff4075c46d7536af0b684a7990b4adf44ad0d655b77298c504`，与旧 GGUF 完全一致；
undo-permute 后的 hash 不同。

输出 GGUF SHA-256：

```text
160116d3d8959cb500eaa47ffaa82c3de4cbc4ccf52c8b1604a878b921d9259d
```

模型权重存放在仓库外，没有提交 checkpoint 或 GGUF。

## 4. W2 bundle 的具体内存布局

每个逻辑线性层由原来的四个独立张量：

```text
U.s0, U.s1, W.s0, W.s1
```

合并为：

```text
<linear>.bundle.codes   GGML_TYPE_FAIRY2I_BUNDLE_CODES
<linear>.bundle.scales  F16
```

分支顺序固定为 `U0,U1,W0,W1`。对每个 M64×K64 physical tile：

- code 的物理次序是 `[m16][q4][branch][row_lane]`；
- `m16` 选择 64 行中的一个 16 行子块；
- `q4` 选择 K64 中连续的四个 K；
- 一个 byte 的四组 2-bit 位依次保存这四个连续 K code；
- GGML tensor shape 为 `ne = [16, 4, 64, physical_tiles]`；
- scale 只保存 `[real, imag] × 4 branches`，shape 为 `ne = [2, 4, physical_tiles]`。

旧 V2 的 20-byte block 在 M64 内每一行重复同一个 real/imag scale。bundle 将 scale 降为每个
M64×K64 tile、每个分支一份，因此低比特权重区从 `2,023,751,680 B` 降至 `1,625,325,568 B`，减少
`19.6875%`。所有 tensor offset 使用 64-byte alignment。

## 5. 全文件等价性验证

`validate_fairy2i_bundle_v1.py` 已从 W1 专用扩展为按 metadata 的 branch order 同时验证 W1/W2。命令：

```bash
PYTHONPATH=gguf-py .venv/bin/python -u gguf-py/validate_fairy2i_bundle_v1.py \
  /Users/a1806/llama/llama2_7b_int8_activate/fairy2i-llama2-7b-checkpoint5998-tile64-v2.gguf \
  /Users/a1806/llama/llama2_7b_int8_activate/fairy2i-llama2-7b-checkpoint5998-bundle-v1.gguf
```

结果：

```text
branch_order=U0,U1,W0,W1
linear_count=224
common_tensor_count=67
tensor_count=515
common_bytes=527450112
v2_weight_bytes=2023751680
bundle_weight_bytes=1625325568
v2_file_bytes=2551987840
bundle_file_bytes=2153544896
canonical_sha256=7128223fb1860a631faef47ada90e9f5b52c3595395431520d41fc00290ce3ca
```

校验器逐 M64 strip 将旧 V2 residue-lane code 转成最终连续 K 的 bundle 次序，并逐分支比较 code 和
FP16 scale 的位模式；公共 dense、norm、embedding、tokenizer tensor 则直接比较 dtype、shape 和数据。

## 6. Metal W2 实现

### 6.1 加载与调度

模型加载器已经能解析 W2 的四分支 descriptor 和 tensor shape。本轮放开 W2 bundle 到 Metal 的设备
路由，并让 `GGML_OP_FAIRY2I_WIDE_LINEAR_W2` 接受 `FAIRY2I_BUNDLE_CODES + F16 scales`。Metal capability
检查严格验证 layout version、分支数 4、M/K 64 对齐和 contiguous buffer。

code 和 scale tensor 保持 GGUF 原始布局映射到 Metal；测试显式检查 `tensor->extra == nullptr`，确保没有
CPU repack 或第二份 GPU 权重。

### 6.2 Prefill kernel

保留此前 W2 sweep 中性能最好的 `32 output rows × 16 tokens × K8` MMA 映射：

- 每个 threadgroup 128 threads、4 个 simdgroups；
- 一个 K8 step 需要 32×8 个系数；前 64 threads 各读取一个 bundle byte，并展开其中四个连续 K code；
- 四个分支在 tile 内固定相邻，直接组成 `uint4(U0,U1,W0,W1)`；
- 每个 physical tile 只读取 8 个 FP16 scale，而不是从四张量、每行 block 中重复读取；
- 系数转成 FP16 后进入四组 8×8 simdgroup MMA，累加保持 FP32；
- activation 仍按原有 half staging 方式协作加载，避免改变已经验证过的计算映射。

对应 kernel marker：

```text
kernel_fairy2i_bundle_w2_half_mma32x16
```

### 6.3 Decode kernel

Decode 使用 `4 output rows × 8 K-block slots`、128 threads：

- 每个 lane 对应 K64 中一个 q4 byte，直接得到四个连续激活位置；
- 四次对齐的 32-bit load 同时取得四行、四个分支的 code；
- 两次 `half4` load 取得 tile 的全部 W2 scale；
- simd reduction 后直接写 packed BF16 complex output。

无 bias、contiguous 的常见路径使用 function-constant specialization 固化 block count 和 stride；另保留带
bias 的通用 direct-bundle kernel。

```text
kernel_fairy2i_bundle_w2_bf16_tile4x1_w8_full_nobias_fc_simd
kernel_fairy2i_bundle_w2_bf16_tile4x1_w8_full_simd
```

## 7. Correctness 与加载结果

- Metal micro test：W2 bundle 无 bias decode、带 bias decode、17-token prefill 全部通过；
- `test-fairy2i`：全部测试通过，W1/W2 bundle Metal 均直接读取；
- `test-fairy2i-loader`：W1/W2 合法 bundle 加载通过，缺 tensor、混合布局、错误 shape/alignment/branch order
  均按预期拒绝；
- Python Fairy2i tests：`16 passed`（其中 bundle/tile64_v2 packer 为 `8 passed`）；
- 完整模型：32/32 repeating layers 与 output layer 全部 offload 到 Metal；
- 固定 seed 的短生成中，旧 V2 与新 bundle 都输出 `so meaningless`，作为端到端 spot check。

当前 `build-rel-metal` 没有编入 `GGML_USE_FAIRY2I_CPU_LUT`，所以 CPU-only bundle compute 会按设计拒绝并
提示启用 LUT16；这不影响本轮 Metal 验收，也没有更改现有 W2 CPU bundle 实现。

## 8. Metal 性能

命令（两种布局完全相同，仅模型路径不同）：

```bash
env LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2=1 \
  ./build-rel-metal/bin/llama-bench \
  -m <model.gguf> -ngl 99 -t 8 -fa 1 -p 512 -n 128 -r 3
```

默认 warmup，所有进程串行执行。顺序为 V2 → bundle → bundle → V2。

| 布局 / 进程 | pp512 tok/s | tg128 tok/s |
| --- | ---: | ---: |
| V2 A1 | 142.47 ± 0.04 | 19.92 ± 0.14 |
| bundle B1 | 164.95 ± 0.59 | 20.33 ± 0.10 |
| bundle B2 | 165.25 ± 0.29 | 20.25 ± 0.01 |
| V2 A2 | 142.49 ± 0.51 | 19.58 ± 0.03 |
| V2 两进程均值 | 142.48 | 19.75 |
| bundle 两进程均值 | 165.10 | 20.29 |
| bundle 相对 V2 | **+15.88%** | **+2.73%** |

另做过一次路径诊断：未设置 `LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2=1` 时，旧 V2 会走非融合回退，两次
分别为 `16.09/5.30` 和 `16.30/5.28 tok/s`；bundle 必须使用 fused op，约为 `165/20 tok/s`。该结果只
用于确认调度路径，不属于布局性能对照，也未纳入上表均值。

## 9. 最终建议

Llama2-7B W2 应使用本轮生成的 `bundle_v1`：它已通过全文件位级等价、完整模型 Metal 运行和两轮反向
顺序 benchmark，文件更小且 pp/tg 都不回退。旧 `tile64_v2` 继续作为兼容输入保留，但新转换默认应使用
`bundle_v1`。
