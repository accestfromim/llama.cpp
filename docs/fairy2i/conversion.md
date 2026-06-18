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

### Unified CLI Arguments

| Argument | Values / default | Description |
| --- | --- | --- |
| `model_dir` | required | Hugging Face checkpoint directory. |
| `output_file` | optional for Llama dry-run, required for writes and Qwen2 | Output GGUF path. |
| `--base-arch` | `auto` (default), `llama`, `qwen2` | Select the base model adapter. `auto` reads `config.model_type`. |
| `--quant-variant` | `tile64_v2` | Fairy2i quant/export variant. |
| `--residual-steps` | `2` | Residual quantization steps. Only `2` is currently supported. |
| `--dry-run` | off | Validate inputs without writing GGUF. Currently useful for Llama; Qwen2 still requires `output_file`. |
| `--qk-permute` | off | Apply Llama q/k undo-permute when the checkpoint layout requires it. |
| `--verbose` | off | Print conversion progress and tensor export stages. |

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

### Llama Script Arguments

`gguf-py/convert_fairy2i_llama.py` writes Llama-based Fairy2i checkpoints.

| Argument | Values / default | Description |
| --- | --- | --- |
| `model_dir` | required | Llama-based Fairy2i checkpoint directory. |
| `output_file` | optional only with `--dry-run` | Output GGUF path. |
| `--dry-run` | off | Validate inputs and print the conversion plan without writing GGUF. |
| `--residual-steps` | `2` | Residual quantization steps. Any value other than `2` errors. |
| `--qk-permute` | off | Enable Llama q/k undo-permute during conversion. |
| `--verbose` | off | Print conversion progress. |

Llama output projection is always exported as a dense padded output tensor:

```text
output
```

The tensor is written with raw dtype `F16`. The vocabulary is padded by the
converter, so the dense output row count matches the padded vocab size.

### Qwen2 Script Arguments

`gguf-py/convert_fairy2i_qwen2.py` exposes Qwen2-specific output and bias
controls that are not forwarded by the unified CLI.

| Argument | Values / default | Description |
| --- | --- | --- |
| `model_dir` | required | Qwen2-based Fairy2i checkpoint directory. |
| `output_file` | required | Output GGUF path. |
| `--residual-steps` | `2` | Residual quantization steps. Any value other than `2` errors. |
| `--output-layer` | `wide-linear` (default), `dense`, `both` | Select output projection export format. |
| `--qk-permute` | off | Enable Llama q/k undo-permute when the source checkpoint requires it. Disabled by default for Fairy2i Qwen2. |
| `--no-attn-bias` | off | Skip optional attention bias tensors even if present in the checkpoint. |
| `--quant-variant` | `tile64_v2` | Fairy2i quant/export variant. |
| `--verbose` | off | Print conversion progress. |

Qwen2 output projection modes:

| Mode | Tensors written | Use case |
| --- | --- | --- |
| `wide-linear` | `output.U.s0`, `output.U.s1`, `output.W.s0`, `output.W.s1` | Default Fairy2i W2 output path. |
| `dense` | `output` as raw dtype `F16` | Dense compatibility path. |
| `both` | Wide-linear tensors and dense `output` | A/B checks or transitional compatibility. |

## Metadata Checks

Inspect the output with:

```bash
PYTHONPATH=gguf-py .venv/bin/python -m gguf.scripts.gguf_dump /path/to/output.fairy2i.gguf | sed -n '1,120p'
```

New files should include:

- `general.architecture = 'fairy2i'`
- `fairy2i.schema_version = 1`
- `fairy2i.base_arch`
- `fairy2i.quant.format = 'fairy2i_tile64_v2'`
- `fairy2i.quant.variant = 'tile64_v2'`
- `fairy2i.attn.layout = 'llama_real'` for Llama or `qwen2_real` for Qwen2
- `fairy2i.tokenizer.profile`

## Converter Tests

Run the Python Fairy2i converter tests with:

```bash
PYTHONPATH=gguf-py .venv/bin/python -m pytest gguf-py/tests/fairy2i -q
```

## Troubleshooting

| Error / symptom | What to check |
| --- | --- |
| `could not detect Fairy2i base architecture` | Check `config.json` has a supported `model_type`, or pass `--base-arch llama` / `--base-arch qwen2`. |
| `output_file is required for qwen2 conversion` | Qwen2 conversion requires an output path. Use the Qwen2 script with `model_dir output_file`. |
| `only --residual-steps 2 is currently supported` | Remove the flag or pass `--residual-steps 2`. |
| `tokenizer.json not found` | Use a complete Hugging Face checkpoint directory that includes tokenizer files. |
| `expected a Llama-style BPE tokenizer with byte_fallback=true` | The checkpoint does not match the Llama Fairy2i tokenizer profile. Use the Qwen2 converter when appropriate, or fix the tokenizer files. |
| Missing or unexpected attention bias tensors | For Qwen2, omit `--no-attn-bias` to export optional attention bias tensors, or pass it to force skipping them. |
