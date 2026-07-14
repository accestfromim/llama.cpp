# Fairy2i Metal tile sweep and cleanup log

Date: 2026-07-14

This note records the Metal Fairy2i tile experiments before removing dead or
negative experimental kernels. The goal of the cleanup is to keep the stable
paths that are actually used for current W2/W1 models and remove old branch
combinations that make the Metal backend difficult to reason about.

## Current models used for checks

- W2 7B: `/Users/a1806/llama/llama2_7b_int8_activate/fairy2i-llama2-7b-checkpoint5998-tile64-v2.gguf`
- W1 Qwen3: `/Users/a1806/llama/model-test/gguf-models/Qwen3/qwen3-fairy2i-w1-learned-scale.gguf`
- W1 duplicate path checked by SHA256:
  `/Users/a1806/llama/qwen_1bit_scale/qwen3-fairy2i-w1-learned-scale.gguf`

The W1 GGUF metadata is `fairy2i.quant.variant=tile64_v2_w1_learned_scale`
and `fairy2i.attn.layout=qwen3_real`. It contains Qwen3 `attn_q_norm` and
`attn_k_norm` tensors, so the current complete graph is slower than an older
incomplete path that did not account for those nodes.

## Prefill kernels

### W2 prefill

Kept path:

- `kernel_fairy2i_wide_linear_w2_q8_mma8x16`
- Tile shape: 8 output rows by 16 activation columns.
- Uses one activation int8 quantization pass, then builds four half coefficient
  tiles in threadgroup memory and consumes them with `simdgroup_multiply_accumulate`.

Observed W2 7B with `-ngl 99 -fa 1 -p 256 -n 128 -r 3`:

| threads | pp256 tok/s | tg128 tok/s |
|---:|---:|---:|
| 2 | 62.60 +/- 0.04 | 16.53 +/- 0.01 |
| 4 | 62.35 +/- 0.07 | 16.57 +/- 0.01 |
| 6 | 62.40 +/- 0.07 | 16.55 +/- 0.01 |
| 8 | 61.18 +/- 0.80 | 16.54 +/- 0.02 |
| 10 | 58.42 +/- 0.53 | 16.60 +/- 0.01 |

Conclusion: CPU thread count has little effect on decode and starts to hurt
prefill above 6 threads. The prefill kernel should remain 8x16.

### W1 prefill

Kept path:

- `kernel_fairy2i_wide_linear_w1_q8_mma8x16`
- Tile shape: 8 output rows by 16 activation columns.
- Uses the W1-specific `fairy2i_mma_coeff_w1_pair()` helper, which decodes
  `U.s0` and `W.s0` together and avoids the W2 four-stage path.

Observed current complete Qwen3 W1 graph:

| command shape | threads | pp256 tok/s |
|---|---:|---:|
| `-p 256 -n 0 -r 10` | 8 | 85.23 +/- 0.11 |
| `-p 256 -n 0 -r 5 --no-warmup` | 8 | 85.00 +/- 0.17 |
| `-p 256 -n 0 -r 5` | 8 | 85.12 +/- 0.14 |

An earlier `cf25d3b6` result showed `pp256 = 102.60 +/- 0.30`, but the current
W1 GGUF has Qwen3 q/k norm tensors and the current graph computes those nodes.
The old number should be treated as an older/incomplete graph-path comparison,
not as the current semantic baseline.

Thread sweep on current W1 complete graph:

| threads | pp256 tok/s | tg128 tok/s | notes |
|---:|---:|---:|---|
| 1 | 85.30 +/- 0.06 | 4.76 +/- 0.00 | no BF16 decode env |
| 2 | 84.86 +/- 0.17 | 4.77 +/- 0.00 | no BF16 decode env |
| 2 | 85.12 +/- 1.21 | 20.52 +/- 0.07 | BF16 decode path enabled |
| 4 | 80.01 +/- 0.86 | 20.41 +/- 0.04 | BF16 decode path enabled |
| 6 | 79.08 +/- 0.13 | 20.47 +/- 0.02 | BF16 decode path enabled |
| 8 | 78.98 +/- 0.19 | 20.38 +/- 0.01 | BF16 decode path enabled |
| 10 | 79.00 +/- 0.24 | 20.30 +/- 0.09 | BF16 decode path enabled |

Conclusion: W1 prefill should keep the 8x16 MMA kernel. Decode should use the
BF16 direct path by default rather than falling back to q8-MMA-as-decode.

## Decode kernels tried

### Attempt matrix

| area | tile / shape tried | activation path | observed result | decision |
|---|---|---|---|---|
| W2 prefill | old `tile4x4` env path | activation quantized | historical experiment only; replaced by MMA path | remove dispatch/test knobs |
| W2 prefill | old `tile8x4` env path | activation quantized | historical experiment only; replaced by MMA path | remove dispatch/test knobs |
| W2 prefill | `q8_mma8x16` | one int8 activation quantization pass | stable current prefill path; pp256 around 62 tok/s on W2 7B | keep |
| W1 prefill | `q8_mma8x16` | one int8 activation quantization pass | stable current W1 prefill path; pp256 around 85 tok/s on complete Qwen3 W1 graph | keep |
| W2 decode | `q16 tile2x1_w4` | activation q16 | early direct-accumulation experiment; not current best | remove |
| W2 decode | `q16 tile4x1_w4` | activation q16 | early fallback path; extra decode quantization is not useful for current Metal decode | remove |
| W2 decode | `q16 tile8x1_w4` | activation q16 | larger row tile did not become a winner | remove |
| W2 decode | `q16 tile16x1_w4` | activation q16 | high accumulator pressure and no stable win | remove |
| W2 decode | `q16 tile16x1_w4_lut` | activation q16 + threadgroup LUT | LUT traffic/synchronization was not a stable win on Metal | remove |
| W2 decode | BF16 `tile2x1_w4` | direct BF16 activation | lower row concurrency; not retained after sweep | remove |
| W2 decode | BF16 `tile4x1_w4` | direct BF16 activation | superseded by `tile4x1_w8`; old explicit kernel was dead after helper cleanup | remove |
| W2 decode | BF16 `tile4x1_w8` | direct BF16 activation | best retained W2 decode family; earlier 7B records were `tg128 = 17.01 +/- 0.01`, `tg512 = 16.45 +/- 0.02` | keep |
| W2 decode | BF16 `tile8x1_w4` | direct BF16 activation | more rows per threadgroup did not offset pressure/latency | remove |
| W2 decode | BF16 `tile8x1_w8` | direct BF16 activation | no stable advantage over `tile4x1_w8`; no-bias variant also not retained | remove |
| W2 decode | BF16 scale-cache trial | direct BF16 activation + threadgroup scale sharing | `tg128 = 16.34 tok/s`, slower than direct scale loads | already removed |
| W1 decode | BF16 `tile4x1_w8` | direct BF16 activation | smaller tile, not retained | remove |
| W1 decode | BF16 `tile4x1_w16` | direct BF16 activation | smaller row tile did not beat `tile8x1_w16` | remove |
| W1 decode | BF16 `tile8x1_w8` | direct BF16 activation | lower block concurrency than retained path | remove |
| W1 decode | BF16 `tile8x1_w16` | direct BF16 activation | stable retained path; BF16 decode sweep gave `tg128` around 20.3-20.5 tok/s | keep |
| W1 decode | BF16 `tile16x1_w8` | direct BF16 activation | more rows per threadgroup did not show enough benefit | remove |
| W1 decode | BF16 `tile16x1_w16` | direct BF16 activation | largest register footprint; not retained | remove |

### W2 q16 / activation-quantized direct decode

Tried kernel families:

- `tile2x1_w4`
- `tile4x1_w4`
- `tile8x1_w4`
- `tile16x1_w4`

These kernels quantize activation first and then decode 2-bit weights while
accumulating. They were useful as correctness bring-up and early fallback, but
they are no longer the preferred path. They also require extra activation
quantization work in decode, which is not beneficial on Metal for the current
models.

Cleanup decision: remove these Metal kernels and route decode to BF16 direct.

### W2 LUT decode

Tried kernel family:

- `tile16x1_w4_lut`

This builds per-lane LUT data in threadgroup memory and then looks up weight
patterns. It did not become a stable winner on Metal; the extra threadgroup
traffic and synchronization are not a good match for current decode shapes.

Cleanup decision: remove the Metal LUT decode kernel and LUT helper functions.
CPU LUT code is separate and is not affected.

### W2 BF16 direct decode

Tried kernel families:

- `tile2x1_w4`
- `tile4x1_w4`
- `tile4x1_w8`
- `tile8x1_w4`
- `tile8x1_w8`
- full-row and no-bias variants where applicable

The current stable W2 decode path is BF16 direct accumulation with four packed
weight references (`U0`, `U1`, `W0`, `W1`) consumed in one helper call. The
main retained variant is:

- rows per tile: 4
- block slots: 8
- full-row fast path retained for common aligned output sizes

Cleanup decision: keep the `tile4x1_w8` BF16 direct path and remove tile2,
tile8, w4, and LUT alternatives.

### W1 BF16 direct decode

Tried kernel families:

- rows per tile: 4, 8, 16
- block slots: 8, 16
- full-row and no-bias variants

The stable W1 decode path is the BF16 direct helper with two packed weight
references (`U.s0`, `W.s0`). The main retained variant is:

- rows per tile: 8
- block slots: 16
- full-row and no-bias fast paths retained

Cleanup decision: keep the `tile8x1_w16` W1 BF16 direct path and remove tile4,
tile16, and w8 alternatives.

## Cleanup target

Keep:

- activation quantization kernel used by prefill
- W2 prefill `q8_mma8x16`
- W1 prefill `q8_mma8x16`
- W2 BF16 direct decode `tile4x1_w8`
- W1 BF16 direct decode `tile8x1_w16`

Remove from Metal:

- q16 activation-quantized W2 decode tile2/tile4/tile8/tile16 kernels
- Metal W2 LUT decode helpers and kernel
- W2 BF16 direct decode tile2/tile8/w4 variants
- W1 BF16 direct decode tile4/tile16/w8 variants
- env-only dispatch switches for removed kernels

The cleanup intentionally does not touch CPU/NEON/LUT implementations.
