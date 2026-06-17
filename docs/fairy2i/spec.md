# Fairy2i GGUF Format Specification

This document defines the stable Fairy2i GGUF envelope used by the
conversion tools and runtime. Fairy2i is a model transformation and
quantized execution format layered on top of a base architecture such as
Llama or Qwen2. It is not itself the base transformer architecture.

## Naming

- User-visible GGUF architecture: `general.architecture = "fairy2i"`.
- Base architecture: `fairy2i.base_arch`, for example `llama` or `qwen2`.
- Storage format: `GGML_TYPE_IFAIRY64` for tile64_v2 weights.
- Compatibility: readers may continue accepting older `ifairy` metadata
  where present, but new converters must write `fairy2i`.

The `ifairy64` name refers to a weight block storage format. The `fairy2i`
name refers to the model envelope and GGUF metadata namespace.

## Required Metadata

New Fairy2i GGUF files should write these keys:

```text
general.architecture = "fairy2i"
fairy2i.schema_version = 1
fairy2i.base_arch = "llama" | "qwen2"
fairy2i.quant.format = "ifairy64"
fairy2i.quant.variant = "tile64_v2"
fairy2i.quant.residual_steps = 2
fairy2i.quant.codebook = "{+/-1,+/-i}"
fairy2i.quant.tile_size = 64
fairy2i.quant.scale_stat = "dominant_mean_abs"
fairy2i.attn.layout = "llama_real" | "qwen2_real" | "legacy_complex"
fairy2i.tokenizer.profile = "llama_bpe" | "qwen2"
```

Converters may also write:

```text
fairy2i.base_model_type = "<HF config.model_type>"
fairy2i.base_architecture = "<HF architectures[0]>"
fairy2i.vocab.original_size = <uint32>
fairy2i.vocab.padded_size = <uint32>
fairy2i.vocab.padding_multiple = <uint32>
```

For compatibility with current files, readers should tolerate missing
`fairy2i.schema_version` as version `0`, missing `fairy2i.quant.format` as
`ifairy64` when `fairy2i.quant.variant = "tile64_v2"`, and missing
`fairy2i.attn.layout` as `qwen2_real` for tile64_v2 files.

## tile64_v2 Weight Blocks

The `tile64_v2` variant stores complex phase-aware weights in
`GGML_TYPE_IFAIRY64` blocks.

- Tile size is 64 complex values.
- Residual stage count is 2.
- The codebook is `{+/-1,+/-i}`.
- Each stage stores packed 2-bit phase codes plus real and imaginary fp16
  scales.
- The scale statistic is `dominant_mean_abs`.

All dimensions quantized as `tile64_v2` must be padded so the complex input
and output dimensions are divisible by 64.

## Widely Linear Tensor Naming

Dense linear weights are transformed into widely linear complex components
`U` and `W`, each with two residual stages:

```text
<base>.U.s0
<base>.U.s1
<base>.W.s0
<base>.W.s1
```

Layer tensors use the existing llama.cpp logical tensor prefixes:

```text
blk.{i}.attn_q.U.s0
blk.{i}.attn_q.U.s1
blk.{i}.attn_q.W.s0
blk.{i}.attn_q.W.s1
blk.{i}.attn_k.*
blk.{i}.attn_v.*
blk.{i}.attn_output.*
blk.{i}.ffn_gate.*
blk.{i}.ffn_up.*
blk.{i}.ffn_down.*
```

Output projection may be stored either as dense `output` or as the same
widely linear component set under `output.*`. Files that include a partial
widely linear output projection are invalid.

## Attention Layout

`fairy2i.attn.layout` describes how attention tensors are interpreted after
conversion:

- `llama_real`: Llama-style real attention layout. Converters must use this
  for Llama-based checkpoints.
- `qwen2_real`: Qwen2-style real attention layout.
- `legacy_complex`: legacy iFairy complex layout for older files.

Converters must not write a base-architecture-specific layout for a different
base architecture.

## Vocabulary Padding

Some Fairy2i files pad vocabulary-related tensors so output shapes remain
tile-aligned while token ids stay stable. When padding is used:

- Existing token ids must be preserved.
- Padded token entries must be marked unused or otherwise non-semantic.
- `fairy2i.vocab.original_size` and `fairy2i.vocab.padded_size` must be
  written.

## Compatibility Policy

The runtime should keep reading current experimental files while new tools
write the normalized schema above. Compatibility reads should be explicit and
limited:

- `fairy2i.quant.variant = "legacy"` keeps the legacy code path.
- Missing version and format keys are accepted only through documented
  defaults.
- Unknown quant variants, tile sizes, scale statistics, attention layouts, or
  base architectures must fail with actionable errors.
