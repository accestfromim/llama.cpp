# IFAIRY64 — 状态 / 性能记录

Status: Draft (2026-04-23)

本文件用于记录 `IFAIRY64` 相关的实现状态、性能数据和专项变更，避免继续混在 `IFAIRY_ARM_3W_LUT_V2_STATUS.md` 中。

相关文档：
- `IFAIRY64_LUT_IMPLEMENTATION_PLAN.md`
- `IFAIRY64_X86_ADAPTATION_EXECUTION_GUIDE.md`
- `IFAIRY_ARM_3W_LUT_V2_STATUS.md`（旧的 ARM 3W LUT V2 总状态）

---

## 变更记录（Changelog）

按日期追加（YYYY-MM-DD）：

### 2026-06-04 (working tree, Llama 7B Fairy2i conversion support)
- Scope:
  - Supports the Llama-based Fairy2i 7B checkpoint with existing `fairy2i` architecture metadata and the existing `IFAIRY64` `tile64_v2` tensor format.
  - No new runtime architecture or kernel is required; the current Fairy2i graph path uses `fairy2i.attn.layout = qwen2_real`, which matches this checkpoint's attention/RoPE layout.
- Converter:
  - Use the local repo package explicitly:
    - `.venv/bin/python gguf-py/convert_fairy2i_llama.py /home/zybi/projects/llama2_7b_new/checkpoint-24920 /tmp/llama2_7b_new.fairy2i.gguf --verbose`
  - Dry-run validation:
    - `.venv/bin/python gguf-py/convert_fairy2i_llama.py --dry-run /home/zybi/projects/llama2_7b_new/checkpoint-24920`
  - The converter pads vocab/embedding/output rows from the original Llama vocab size to a 128-token boundary and records:
    - `fairy2i.vocab.original_size`
    - `fairy2i.vocab.padded_size`
  - Added padding tokens are emitted as `UNUSED`; token ids below the original vocab size keep their original order.
  - The output projection is kept dense `F16` with zero padded rows, while transformer linear weights use `IFAIRY64`. This avoids changing output token id ordering and keeps padded logits neutral unless a future sampler-side mask is needed.
  - `general.file_type` is written as `MOSTLY_IFAIRY`.
- Runtime notes:
  - File-type guessing maps both `GGML_TYPE_IFAIRY` and `GGML_TYPE_IFAIRY64` to `LLAMA_FTYPE_MOSTLY_IFAIRY`, so converted files no longer report `unknown type ifairy64` / guessed `all F32` when metadata is absent.
  - The EOG token texts `</s>` and `<｜end▁of▁sentence｜>` are recognized directly by `llama-vocab.cpp`, avoiding the tokenizer warning fallback for this checkpoint.
- Inspection / smoke:
  - If `.venv` has an old installed `gguf` package, run GGUF tools with `PYTHONPATH=gguf-py` so dtype `42` / `IFAIRY64` is recognized:
    - `PYTHONPATH=gguf-py .venv/bin/python -m gguf.scripts.gguf_dump /tmp/llama2_7b_new.fairy2i.gguf`
  - CPU smoke:
    - `./build-rel/bin/llama-cli -m /tmp/llama2_7b_new.fairy2i.gguf --gpu-layers 0 -t 4 -p "I believe life is" -n 16 -no-cnv`

### 2026-04-22 (working tree; base build `abcaafef`)
- 变更摘要：
  - `IFAIRY64` 在 `GGML_IFAIRY_LUT=1` 时于模型加载阶段提前完成 LUT transform/prepack，避免 decode 首轮再做 transform。
  - `IFAIRY64` 的 packed weight tile 不再把每 lane scale 展开成 `f32`，改为保留 `fp16` scale 并在 kernel 内按需转 `f32`，将 packed footprint 从 `384B/block-tile` 压回 `320B/block-tile`。
- Correctness:
  - `./build-rel-lut/bin/test-ifairy --ifairy-lut-only`: PASS
- microbench（Machine: Mac16,12 / Apple M4）：
  - `./build-rel-lut/bin/ifairy-microbench --type ifairy64 --mode fused --m 3456 --k 2560 --iters 200 --warmup 20`
  - Result:
    - `ns/iter=322370.0`
- `llama-bench`（model=`~/fairy2i_32b/fairy2i_32b.gguf`; threads=4; `-dev none -ngl 0 -b 32 -ub 32 -fa 0 --no-warmup -p 0 -n 32 -r 1 -o md`）：
  - vecdot baseline:
    - `./build-rel-lut/bin/llama-bench -m ~/fairy2i_32b/fairy2i_32b.gguf ...`
    - `tg32`: `2.22 ± 0.00 tok/s`
  - explicit LUT:
    - `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 ./build-rel-lut/bin/llama-bench -m ~/fairy2i_32b/fairy2i_32b.gguf ...`
    - `tg32`: `2.46 ± 0.00 tok/s`
  - Delta vs vecdot:
    - `tg32`: `+10.8%`

### 2026-04-23 (working tree, Fairy2i 32B output-only merged IFAIRY64)
- 变更：
  - 新增 `LLAMA_FAIRY2I_MERGED_OUTPUT=1` opt-in。
  - load 阶段从 `output.{U,W}.s{0,1}` 逐行 `dequant + sum + requant` 合成两块 output-only `IFAIRY64` 权重：
    - `output.U.merged`
    - `output.W.merged`
  - 最终 lmhead 从 4 次 Fairy2i output matmul 缩成 2 次；中间层不变。
- 额外权重开销：
  - `load_tensors: prepared merged FAIRY2I output weights in 0.523 sec (2 x 58.01 MiB)`
- 验证：
  - `./build-rel-lut/bin/test-ifairy --ifairy-lut-only`: PASS
  - `LLAMA_FAIRY2I_MERGED_OUTPUT=1 ./build-rel-lut/bin/llama-cli -m ~/fairy2i_32b/fairy2i_32b.gguf -dev none -ngl 0 -t 4 -c 8196 -b 32 -ub 32 -fa off --no-warmup -no-cnv --temp 0.2 --top-k 20 --top-p 0.9 -n 1 -p '<｜begin▁of▁sentence｜> You are a helpful AI assistant. <｜User｜> Where is China?'`: smoke PASS
- `llama-bench`（model=`~/fairy2i_32b/fairy2i_32b.gguf`, threads=4, `-p 0 -n 32 -r 1`）：
  - 默认 vecdot：
    - `./build-rel-lut/bin/llama-bench ...`: `tg32=2.24 tok/s`
    - `LLAMA_FAIRY2I_MERGED_OUTPUT=1 ./build-rel-lut/bin/llama-bench ...`: `tg32=2.26 tok/s`
    - 提升：`+0.02 tok/s`（`+0.9%`）
  - 显式 LUT：
    - `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 ./build-rel-lut/bin/llama-bench ...`: `tg32=2.42 tok/s`
    - `LLAMA_FAIRY2I_MERGED_OUTPUT=1 GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 ./build-rel-lut/bin/llama-bench ...`: `tg32=2.49 tok/s`
    - 提升：`+0.07 tok/s`（`+2.9%`）
- 结论：
  - 这条 output-only 合并对 LUT 和 vecdot 都是正收益，但收益量级符合 lmhead 占比，属于小幅提速。
  - 当前只做了 smoke 和 tok/s 验证，未做质量/困惑度回归。
