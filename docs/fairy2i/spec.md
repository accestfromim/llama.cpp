# Fairy2i GGUF Format Specification

This document defines the stable Fairy2i GGUF envelope used by the
conversion tools and runtime. Fairy2i is a model transformation and
quantized execution format layered on top of a base architecture such as
Llama or Qwen2. It is not itself the base transformer architecture.

## Naming

- User-visible GGUF architecture: `general.architecture = "fairy2i"`.
- Base architecture: `fairy2i.base_arch`, for example `llama` or `qwen2`.
- Storage format: `GGML_TYPE_FAIRY2I_TILE64_V2` for tile64_v2 weights.
- Activation quantization type: `GGML_TYPE_FAIRY2I_ACT_Q16_64`.
- New Fairy2i files and execution paths must use Fairy2i or generic complex
  identifiers only. Historical experimental identifiers are legacy-only and
  are not part of this schema.

## Required Metadata

New Fairy2i GGUF files should write these keys:

```text
general.architecture = "fairy2i"
fairy2i.schema_version = 1
fairy2i.base_arch = "llama" | "qwen2"
fairy2i.quant.format = "fairy2i_tile64_v2"
fairy2i.quant.variant = "tile64_v2"
fairy2i.quant.residual_steps = 2
fairy2i.quant.codebook = "{+/-1,+/-i}"
fairy2i.quant.tile_size = 64
fairy2i.quant.scale_stat = "dominant_mean_abs"
fairy2i.attn.layout = "llama_real" | "qwen2_real"
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

New Fairy2i readers must require these keys for schema version 1 files.
Compatibility with previous experimental files belongs in an explicit legacy
loader or migration tool, not in the normalized Fairy2i schema defaults.

## tile64_v2 Weight Blocks

The `tile64_v2` variant stores complex phase-aware weights in
`GGML_TYPE_FAIRY2I_TILE64_V2` blocks.

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

Converters must not write a base-architecture-specific layout for a different
base architecture.

## Vocabulary Padding

Some Fairy2i files pad vocabulary-related tensors so output shapes remain
tile-aligned while token ids stay stable. When padding is used:

- Existing token ids must be preserved.
- Padded token entries must be marked unused or otherwise non-semantic.
- `fairy2i.vocab.original_size` and `fairy2i.vocab.padded_size` must be
  written.

## Legacy Policy

New Fairy2i tools write only the normalized schema above. Legacy compatibility
must be explicit and isolated:

- A legacy loader or migration tool may translate previous experimental files.
- The runtime must not silently treat missing Fairy2i schema keys as the new
  tile64_v2 format.
- Unknown quant variants, tile sizes, scale statistics, attention layouts, or
  base architectures must fail with actionable errors.
