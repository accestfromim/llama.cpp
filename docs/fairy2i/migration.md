# Fairy2i Migration Notes

This page records the compatibility policy for the current refactor.

## Branching And Commits

Work should happen on a dedicated branch. Each independent change should be
tested and committed before starting the next change.

Commit messages should follow the repository style:

```text
<module>: concise message
```

## GGUF Metadata

New converters write the normalized schema documented in
`docs/fairy2i/spec.md`:

```text
general.architecture = "fairy2i"
fairy2i.schema_version = 1
fairy2i.base_arch = "llama" | "qwen2"
fairy2i.quant.format = "fairy2i_tile64_v2"
fairy2i.quant.variant = "tile64_v2"
fairy2i.tokenizer.profile = "llama_bpe" | "qwen2"
```

Compatibility with earlier experimental files belongs in an explicit legacy
loader or migration tool. New Fairy2i readers must not silently treat missing
schema keys as normalized Fairy2i v1 files.

## Converter Entry Points

Preferred:

```bash
PYTHONPATH=gguf-py .venv/bin/python -m fairy2i.convert ...
```

Compatibility:

```bash
gguf-py/convert_fairy2i_llama.py
gguf-py/convert_fairy2i_qwen2.py
```

The compatibility scripts now call shared Fairy2i schema helpers. Llama no
longer imports Qwen2 converter internals for tile64_v2 packing.

## CPU Build Options

Legacy-only:

```text
GGML_IFAIRY_LUT_CPU
GGML_IFAIRY_FUSE_AVX512
```

New:

```text
GGML_FAIRY2I_CPU_LUT
GGML_FAIRY2I_CPU_AVX512
```

The old names now map only to `GGML_LEGACY_IFAIRY_*` options. They do not
enable Fairy2i features. Prefer the Fairy2i names in docs, scripts, and CI.

## OpenCL Scope

OpenCL is part of the full decoupling scope. New Fairy2i OpenCL routing uses
`GGML_OPENCL_FAIRY2I`; legacy iFairy routing keeps the older
`GGML_OPENCL_IFAIRY64` gate until the legacy backend is isolated.
