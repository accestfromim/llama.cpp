# Fairy2i Llama GGUF Conversion

This page is kept as a compatibility pointer for the original Llama-specific
conversion workflow. The maintained conversion guide is now:

- `docs/fairy2i/conversion.md`
- `docs/fairy2i/spec.md`

For Llama checkpoints, prefer the unified converter:

```bash
PYTHONPATH=gguf-py .venv/bin/python -m fairy2i.convert \
    /path/to/llama-checkpoint \
    /path/to/output.fairy2i.gguf \
    --base-arch llama \
    --quant-variant tile64_v2 \
    --verbose
```

The legacy Llama script remains available:

```bash
PYTHONPATH=gguf-py .venv/bin/python gguf-py/convert_fairy2i_llama.py \
    /path/to/llama-checkpoint \
    /path/to/output.fairy2i.gguf \
    --verbose
```

Use `--dry-run` first to validate metadata and tensor availability. Use
`--qk-permute` only when the checkpoint is known to need Llama q/k
undo-permutation.
