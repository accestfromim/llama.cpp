# ARM / Fairy2i upstream port progress

> 开发进度记录。本文只记录已在当前 `lwt/merge_master` 分支完成并验证的 A0-A2 适配；不把通用上游 ARM quant 代码直接套到 Fairy2i/iFairy 自定义布局。

## 当前状态

- 分支：`lwt/merge_master`
- 最新实现提交：`1c8e8bf8d`
- 回滚粒度：每个目标一个提交，提交顺序见下表。
- 工作树：A0-A2 实现完成；源代码提交后由 CPU 矩阵和真实模型 smoke 验证。

| 批次 | 内容 | 提交 |
|---|---|---|
| A0 | 修正 `ggml_vec_mad1_f32()` 的 SIMD FMA 参数顺序；加入长向量 `SCALE` 回归用例 | `64cd4f3bc`, `8c351ee68` |
| A1 | 适配 ARM `-mcpu`/`-march` 探测、宏验证和所需 feature flag 传递；保留 Fairy2i source split 与 DOTPROD gate | `6028b4b64` |
| A2 | 增加 Linux/aarch64 缺失 HWCAP fallback；FP16 runtime 能力要求 `HWCAP_FPHP` 与 `HWCAP_ASIMDHP` 同时存在 | `1c8e8bf8d` |

## A0 证据

- 修复前，SIMD-only build 的长向量 `SCALE` case 在 Metal backend 对比中失败，CPU 路径通过。
- 修复后执行：

  ```bash
  cmake --build build-a0-noaccel --target test-backend-ops --config Release -j 8
  ./build-a0-noaccel/bin/test-backend-ops test -b Metal -o SCALE
  ```

  结果：`2/2 backends passed`，`SCALE` 两个 case 通过。
- `tests/test-backend-ops.cpp` 的 scale case 从 `n=10` 扩展到 `n=100`，覆盖超过 `GGML_F32_STEP` 的 SIMD 分块和尾部路径。
- Fairy2i 图仍在 `src/llama-graph.cpp` 使用通用 `ggml_scale()`；因此该修复覆盖 LoRA scaling 和 attention scaling 的共享 CPU 路径，不改变 Fairy2i 自定义权重布局。

## A1 证据

- `ggml/src/ggml-cpu/CMakeLists.txt`：
  - 优先使用可验证的显式 `-march`，否则使用显式 `-mcpu`，最后回退到 `-mcpu=native`。
  - 对 `-U__ARM_FEATURE_*` 和编译器实际宏做 feature 检查，避免 host feature 泄漏到目标 variant。
  - 将 `ARCH_FLAGS` 与 `ARCH_DEFINITIONS` 分开传递，保持 Fairy2i DOTPROD source 和 compile definition gate 不变。
- 已验证配置：
  - 显式 `-march=armv8-a` baseline；
  - `GGML_CPU_ALL_VARIANTS=ON` 的 apple-m1 / apple-m2-m3 / apple-m4 variants；
  - dotprod-enabled Fairy2i direct build；
  - dotprod-disabled Fairy2i build，`test-fairy2i` 全部通过。
- 失败的 Fairy2i CMake test-only 配置是旧 target 选择问题（`tests` 子目录未被配置），随后用 `test-fairy2i` / `test-backend-ops` 目标重新配置并通过；不是源代码失败。

## A2 证据

- `ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp` 仅补充缺失宏：`HWCAP_FPHP`、`HWCAP_ASIMDHP`、`HWCAP_ASIMDDP`、`HWCAP_SVE`、`HWCAP2_SVE2`、`HWCAP2_I8MM`、`HWCAP2_SME`。
- 保留本地 backend score ABI 和 Fairy2i runtime gate；没有引入后续 `ggml-feats.h` 重构。
- 通过伪造 Linux `sys/auxv.h` 的语法/score harness 验证缺少系统 HWCAP 宏时仍可编译，并验证 FP16 / DOTPROD score 条件。

## Fairy2i / iFairy 模型兼容性

当前模型格式对应不同 CPU gate，不能把两条路径混成一个标准 quant kernel：

| 模型 | 可验证配置 | 结果 |
|---|---|---|
| `models/Fairy-plus-minus-i-700M/ifairy.gguf` | `build-ifairy-legacy`，`GGML_IFAIRY_LUT=1` | CLI 生成 smoke PASS；加载 267 tensors，输出可读文本 |
| `models/Fairy2i-W2/fairy2i-w2.gguf`（旧 `IFairy` 权重、`fairy2i` architecture） | 同时启用 `GGML_FAIRY2I_CPU=ON` 与 `GGML_LEGACY_IFAIRY_CPU=ON` 的 compatibility build，`--device none` | CLI 生成 smoke PASS；加载 966 tensors，输出可读文本 |
| Fairy2i tile64/bundle fixtures | `build-rel-fairy2i-direct` / `build-rel-fairy2i` | loader、CPU direct、LUT、Metal fixture gates PASS |

`fairy2i-w2.gguf` 不能在只启用新 Fairy2i CPU kernel、未启用 legacy iFairy CPU kernel 的配置中加载；其 tensor type 仍是 `GGML_TYPE_IFAIRY`。这属于现有格式/compile gate 约束，不由 A0-A2 隐式改变。

## CPU CI 证据

完整脚本：

```bash
bash scripts/ci-fairy2i-cpu.sh
```

结果：通过。

覆盖内容：

- CPU-only baseline `ggml-base` / `ggml-cpu` build；
- Fairy2i direct loader/test；
- Fairy2i LUT loader/test、LUT required/disabled runtime gates；
- Fairy2i W2 backend-op 14578/14578 cases；
- legacy iFairy direct 与 LUT loader/test。

另行通过的 targeted checks：

```bash
./build-a0-noaccel/bin/test-backend-ops test -b Metal -o SCALE
./build-a1-fairy2i-nodot/bin/test-fairy2i
```

两项均通过。

## 暂不移植

- 标准 Q4/Q5/Q6/Q8 ARM repack：布局不兼容 Fairy2i tile64。
- KleidiAI / SME / SME2：单独的可选 ARM 后端路线，不能从通用 benchmark 推导 Fairy2i 收益。
- 通用 RMS_NORM、FlashAttention、IQ LUT 并行化：仅作实现参考，不替换 Fairy2i 自定义算子。
- 通用 GGUF/model-quant PR：需要 Fairy2i-specific fixture 证明 tensor type、stride、scale 和 bundle metadata 后再处理。
