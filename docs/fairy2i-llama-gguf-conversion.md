# Fairy2i Llama GGUF 转换教程

本文记录如何从零配置 `uv` 虚拟环境，并使用
`gguf-py/convert_fairy2i_llama.py` 把 Llama-based Fairy2i Hugging Face
checkpoint 转换为 GGUF。

说明：转换脚本导入的是 Python 包 `gguf`。本仓库可本地安装的 Python 包目录是
`gguf-py/`；顶层 `ggml/` 是 C/C++ 源码目录，不包含 `pyproject.toml` 或
`setup.py`，不能直接作为 Python 包安装。

## 准备虚拟环境

在仓库根目录执行：

```bash
cd ~/projects/llama.cpp

uv venv .venv --python 3.13
```

如果当前机器的 `~/.cache/uv` 不可写，可以把 uv 缓存放到 `/tmp`：

```bash
export UV_CACHE_DIR=/tmp/uv-cache
```

安装本地 `gguf` 包和转换脚本依赖：

```bash
uv pip install --python .venv/bin/python ./gguf-py
uv pip install --python .venv/bin/python torch safetensors
```

确认环境可用：

```bash
uv pip show --python .venv/bin/python gguf torch safetensors
```

如果已有 `.venv`，只需要重新安装本地 `gguf-py` 来更新 `gguf`：

```bash
uv pip install --python .venv/bin/python ./gguf-py
```

## 选择 checkpoint

转换输入目录必须包含这些文件：

- `config.json`
- `model.safetensors.index.json`
- `tokenizer.json`
- safetensors 权重分片

例如 `~/projects/llama2_7b_chat` 的顶层不是 checkpoint，实际 checkpoint 在：

```bash
~/projects/llama2_7b_chat/checkpoint-11996
```

如果同一个训练目录里有多个 checkpoint，通常选择编号最大的 checkpoint。

## Dry-run 检查

先执行 dry-run，确认配置、词表 padding 和张量索引能被识别：

```bash
MODEL_ROOT=~/projects/llama2_7b_chat
CKPT="$MODEL_ROOT/checkpoint-11996"

.venv/bin/python gguf-py/convert_fairy2i_llama.py "$CKPT" --dry-run --verbose
```

成功时会看到类似输出：

```text
Fairy2i Llama conversion: layers=32 hidden_real=4096 hidden_complex=2048 ff_complex=5504 ff_complex_padded=5632 vocab_original=32006 vocab_padded=32128 padded_tokens=122
```

## 正式转换

把输出 GGUF 放到模型目录根目录：

```bash
MODEL_ROOT=~/projects/llama2_7b_chat
CKPT="$MODEL_ROOT/checkpoint-11996"
OUT="$MODEL_ROOT/llama2_7b_chat.fairy2i.gguf"

.venv/bin/python gguf-py/convert_fairy2i_llama.py "$CKPT" "$OUT" --verbose
```

转换成功时会输出：

```text
GGUF saved to: /home/zybi/projects/llama2_7b_chat/llama2_7b_chat.fairy2i.gguf
```

注意：如果输出目录不在当前仓库可写范围内，需要确保 shell 对该目录有写权限。

默认不要加 `--qk-permute`。只有当 checkpoint 的 q/k 权重明确需要撤销 Llama
permute 布局时，才使用：

```bash
.venv/bin/python gguf-py/convert_fairy2i_llama.py "$CKPT" "$OUT" --verbose --qk-permute
```

## 验证输出

检查文件大小：

```bash
ls -lh "$OUT"
```

用本地 `gguf-py` 读取 GGUF 元数据：

```bash
PYTHONPATH=gguf-py .venv/bin/python -m gguf.scripts.gguf_dump "$OUT" | sed -n '1,90p'
```

关键字段应包含：

- `general.architecture = 'fairy2i'`
- `general.file_type = 40`
- `fairy2i.quant.variant = 'tile64_v2'`
- `fairy2i.vocab.original_size`
- `fairy2i.vocab.padded_size`
- tensor 类型中包含 `IFAIRY64`

如果已经构建了 `llama-cli`，可以做一次 CPU smoke：

```bash
./build-rel/bin/llama-cli \
    -m "$OUT" \
    --gpu-layers 0 \
    -t 4 \
    -p "I believe life is" \
    -n 1 \
    -no-cnv
```

能加载模型并完成 1 token 生成，即说明 GGUF 至少可以被当前 runtime 解析并执行。
