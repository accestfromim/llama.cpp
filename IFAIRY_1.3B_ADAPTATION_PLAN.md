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
- 处理多分片 safetensors：确认 `model.safetensors.index.json` 下的 `weight_map` 正常驱动 `quant_and_merge`/`noquant_*`，必要时增加日志或校验提示（已改为解析相对路径并在任意工作目录下可运行）。
- 产出 GGUF：`python3 gguf-py/convert_ifairy.py models/Fairy-plus-minus-i-1.3B ifairy.gguf --verbose`。
- 当前转换脚本调整：FFN 宽度会按 `F16_I2` block size（256）向上补齐（5460 -> 5632），并在写入元数据时使用补齐后的 feed_forward_length；对应的 `up/gate/down` 权重、`ffn_layernorm` 实/虚权重会在合并前补零以保证量化能对齐 256。转换日志会输出 padding 提示（`gguf: padding FFN dimension from ...`）。

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
- 若加载报 `check_tensor_dims: tensor 'blk.0.ffn_sub_norm' has wrong shape; expected 11264, got 10920`，原因是 feed_forward_length 已补齐到 5632（*2=11264），但 `ffn_layernorm` 实/虚权重仍保持 5460（*2=10920）；需确保转换阶段对 `ffn_layernorm.weight_{real,imag}` 同步做零填充并重新生成 GGUF。

6) **文档与清理**
- 在 `IFAIRY_INFERENCE_PIPELINE.md` 或相关 README 中补充 1.3B 规格（hidden_size/heads/FFN）与转换命令。
- 确认仓库中不残留临时生成物（中间 gguf/日志），仅保留最终 `ifairy.gguf`。

完成上述步骤后，再根据需要扩展性能基准（`build/bin/llama-bench`, `llama-perplexity`）与多后端验证。***
