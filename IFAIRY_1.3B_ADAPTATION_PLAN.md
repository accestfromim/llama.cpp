# iFairy 1.3B 适配工作计划

目标：在保持 700M 版本已有实现的基础上，完成 Fairy-plus-minus-i 1.3B 模型在 llama.cpp 的端到端适配（转换、加载、推理、测试）。

## 关键参数对比（来自 config.json）

| 配置项 | 700M (`models/Fairy-plus-minus-i-700M/config.json`) | 1.3B (`models/Fairy-plus-minus-i-1.3B/config.json`) | 影响点 |
| --- | --- | --- | --- |
| `hidden_size` | 1536 | 2048 | 头维/FFN 宽度、KV 缓存大小 |
| `intermediate_size` | 4096 | 5460 | FFN 线性层形状 |
| `num_attention_heads` / `num_key_value_heads` | 16 / 16 | 32 / 32 | QKV reshape、rope 维度 |
| `num_hidden_layers` | 24 | 24 | 结构一致，可复用层级逻辑 |
| 其他（context/vocab/eps/theta/act） | 相同 | 相同 | 量化/算子逻辑可直接继承 |

## 复用 700M 现有内容
- 量化与张量格式：`GGML_TYPE_IFAIRY` + `F16_I2`（见 `ggml/src/ggml-quants.c`、`gguf-py/convert_ifairy.py`）。
- 计算图：`llm_build_ifairy` 及相关算子（`src/llama-graph.cpp`、`src/llama-model.cpp`）。
- 转换脚本：`gguf-py/convert_ifairy.py` 已支持合并存储和多分片 safetensors。
- 名称映射：`src/llama-arch.cpp` 中的 ifairy 张量命名映射。
- 测试基线：`tests/test-ifairy.cpp` + `tests/test-ifairy-ref.py`（量化/rope/matmul）。

## 逐步工作项

1) **配置与转换准备**
- 核对 1.3B 配置读取是否覆盖隐藏维/头数/FFN 宽度：`gguf-py/convert_ifairy.py` 中的 `writer.add_*` 调用已从 config 读取，但需确保新增值无硬编码依赖（例如注释中的 1536/4096/16 仅为说明，可移除或更新）。
- 处理多分片 safetensors：确认 `model.safetensors.index.json` 下的 `weight_map` 正常驱动 `quant_and_merge`/`noquant_*`，必要时增加日志或校验提示。
- 产出 GGUF：`python3 gguf-py/convert_ifairy.py models/Fairy-plus-minus-i-1.3B ifairy.gguf --verbose`。

2) **模型类型识别**
- 调整 `src/llama-model.cpp` 中 `LLM_ARCH_IFAIRY` 的类型判定，避免 1.3B 被误判为 `LLM_TYPE_700M`。建议依据 `n_embd` 或头数分支：`n_layer==24 && n_embd==2048 (&& n_head==32)` -> `LLM_TYPE_1_3B`，保留 700M 路径用于 `n_embd==1536`。
- 确认 `llama_model_type_name` 已包含 `1.3B`，无其他枚举调整需求。

3) **算子与图形状检查**
- `llm_build_ifairy` 中的 Q/K/V reshape 依赖 `n_embd_head/2`、`n_head`/`n_head_kv`，验证 2048/32 -> 64（再除 2 得 32）后 rope 和 split/merge 的维度依旧对齐。
- `kq_scale` 默认 `1/sqrt(n_embd_head/2)`：确认对 1.3B 的数值合理或按模型需要从元数据覆盖。
- 关注 FFN：`intermediate_size=5460` 非 2 的幂，检查 `ifairy_build_ffn` 使用的矩阵乘与量化路径是否仅依赖实际 shape（预期无需改动，但需要在加载时验证张量尺寸匹配）。

4) **权重加载与张量映射**
- 复用 `llama-arch.cpp` 的命名映射，检查 1.3B safetensors 是否仍使用与 700M 相同的键（`token_embd_{real,imag}`、`blk.{i}.attn_q_{real,imag}` 等）。若有新增键，需同步映射和转换脚本。
- 确认 `gguf` 写入元数据涵盖 `n_head`, `n_head_kv`, `n_ff`, `rope_theta`, `rms_norm_eps`，供运行时 hparams 计算使用。
- 对齐激活/Norm 存储格式：`noquant_and_merge`（embeddings, lm_head）与 `noquant_and_cat`（norm）逻辑是否在 2048 维下保持 F32/F16 期望。

5) **测试与验证**
- 单元测试：`python3 tests/test-ifairy-ref.py` 生成数据；`cmake --build build --target test-ifairy -j $(nproc)`；`ctest --test-dir build -R test-ifairy --output-on-failure`。
- 转换验证：`python3 gguf-py/calidate_convert.py ifairy.gguf`（或 `gguf-py/test_convert*.py`）。
- 推理冒烟：`./build/bin/llama-cli -m models/Fairy-plus-minus-i-1.3B/ifairy.gguf --gpu-layers 0 -t 4 -p "I believe life is" -n 256 -no-cnv`，对比 700M 的 tok/s 作为性能 sanity check。

6) **文档与清理**
- 在 `IFAIRY_INFERENCE_PIPELINE.md` 或相关 README 中补充 1.3B 规格（hidden_size/heads/FFN）与转换命令。
- 确认仓库中不残留临时生成物（中间 gguf/日志），仅保留最终 `ifairy.gguf`。

完成上述步骤后，再根据需要扩展性能基准（`build/bin/llama-bench`, `llama-perplexity`）与多后端验证。***

## 性能瓶颈分析（1.3B, CPU 单线程基线）

- Profiling 命令：`./build/bin/llama-cli -m models/Fairy-plus-minus-i-1.3B/ifairy.gguf --gpu-layers 0 -t 1 -p "I believe the meaning of life is" -n 256 -no-cnv`
- 观测：`ggml_vec_dot_f32` ~55%、`ggml_compute_forward_ifairy_split` ~25%（time profiler）。
- 可能原因：
  - `ggml_vec_dot_f32`：单线程 F32 点积成为所有 matmul 的主瓶颈；若构建时未启用 `GGML_SIMD`/`GGML_NATIVE`/Accelerate/BLAS，会落入标量路径；iFairy 复数 matmul会将同一输入做两次 dot（实/虚），且每层多次 split/merge 导致重复读写。
  - `ggml_compute_forward_ifairy_split`：逐元素 BF16→F32 解包 + 分离实/虚的标量循环，内层未向量化且 i0 维串行；每层多次调用（norm、FFN、QKV）导致频繁内存搬运/缓存未命中。

## 1.3B 性能提升方案（优先级从易到难）

1. **构建验证与基线复测**
   - 重新构建启用 CPU 向量化：`cmake -B build -DGGML_NATIVE=ON -DGGML_ACCELERATE=ON`（或对应 BLAS 后端），确认日志中 `GGML_SIMD`/NEON/SVE/AVX512 已开启。
   - 保持 `--gpu-layers 0 -t 1` 重跑上面命令记录 tok/s，作为后续优化对照。
2. **ggml_vec_dot_f32 内核优化**
   - 对 2048 维常见长度增加专用内核（对齐/预取/更深 unroll），确保在 AArch64 触发 NEON/SVE/FMA 路径而非标量；必要时为大于阈值的 n 调用 vDSP/CBLAS `sdot`。
   - 复数 matmul 融合：为 iFairy 增加“成对 dot”的复数 GEMV/GEMM 内核，减少实部/虚部两次遍历以及重复加载 activations。
   - 检查 F16_I2 解码是否在内层重复展开；若是，考虑增加量化专用 dot（直接读量化位宽，减少 F32 展开）。
3. **降低 split/merge 频次与向量化**
   - 将 `ifairy_build_norm` / FFN 路径改为以 split 后的张量为中间格式，避免层内反复 split→merge→split；只在需要与通用算子交互时再 merge。
   - 在 `ggml_compute_forward_ifairy_split` 中使用批量 BF16→F32 转换（如以 16 或 32 宽度的 NEON/SVE load + F32 store），并让并行划分同时覆盖 i0 维，提升 L1/L2 利用率。
   - 评估将 split 与 RoPE/RMSNorm 融合为单核，减少一次内存往返（当前分离后又立即做旋转/归一）。
4. **调度与线程利用**
   - 将 `ggml_compute_forward_ifairy_split` 采用 `ggml_parallelize` 或扩展分片策略，使 `-t N` 能在 i0/i1 维同时并行（当前仅按行均分）。
   - 评估在 matmul 前后增加简易 prefetch/缓存对齐，减少 split 输出被后续 dot 再次读入时的缓存抖动。
5. **验证与回归检查**
   - 每完成一个优化点，固定 prompt 复测 tok/s、`llama-bench`，并对比 `ctest -R ifairy` 确认数值一致；记录是否仍是 CPU 热点、火焰图是否迁移到其他算子。
