# Fairy2i GGUF Conversion

This guide covers the normalized Fairy2i converter entry point. The legacy
per-architecture scripts remain available for compatibility, but new workflows
should prefer the module entry point.

## Environment

Install the local `gguf-py` package and converter dependencies in your Python
environment:

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python ./gguf-py
uv pip install --python .venv/bin/python torch safetensors tokenizers pytest
```

If `uv` cache permissions are restricted, set a writable cache directory before
installing:

```bash
export UV_CACHE_DIR=/tmp/uv-cache
```

## Inputs

The model directory must contain:

- `config.json`
- `tokenizer.json`
- `model.safetensors.index.json` plus referenced shards, or one
  `*.safetensors` file

The converter detects the base architecture from `config.model_type` when
`--base-arch auto` is used.

## Unified CLI

Run a dry-run first:

```bash
PYTHONPATH=gguf-py .venv/bin/python -m fairy2i.convert \
    /path/to/checkpoint \
    --base-arch auto \
    --dry-run \
    --verbose
```

Write a GGUF:

```bash
PYTHONPATH=gguf-py .venv/bin/python -m fairy2i.convert \
    /path/to/checkpoint \
    /path/to/output.fairy2i.gguf \
    --base-arch auto \
    --quant-variant tile64_v2 \
    --verbose
```

Supported base architectures in this refactor are `llama` and `qwen2`.

## Compatibility Scripts

Existing scripts remain supported:

```bash
PYTHONPATH=gguf-py .venv/bin/python gguf-py/convert_fairy2i_llama.py \
    /path/to/llama-checkpoint \
    /path/to/output.fairy2i.gguf \
    --verbose

PYTHONPATH=gguf-py .venv/bin/python gguf-py/convert_fairy2i_qwen2.py \
    /path/to/qwen2-checkpoint \
    /path/to/output.fairy2i.gguf \
    --quant-variant tile64_v2 \
    --verbose
```

Use `--qk-permute` only when the checkpoint's q/k tensors are known to need
the Llama undo-permute transform.

## Metadata Checks

Inspect the output with:

```bash
PYTHONPATH=gguf-py .venv/bin/python -m gguf.scripts.gguf_dump /path/to/output.fairy2i.gguf | sed -n '1,120p'
```

New files should include:

- `general.architecture = 'fairy2i'`
- `fairy2i.schema_version = 1`
- `fairy2i.base_arch`
- `fairy2i.quant.format = 'ifairy64'`
- `fairy2i.quant.variant = 'tile64_v2'`
- `fairy2i.attn.layout = 'llama_real'` for Llama or `qwen2_real` for Qwen2
- `fairy2i.tokenizer.profile`

## Converter Tests

Run the Python Fairy2i converter tests with:

```bash
PYTHONPATH=gguf-py .venv/bin/python -m pytest gguf-py/tests/fairy2i -q
```
