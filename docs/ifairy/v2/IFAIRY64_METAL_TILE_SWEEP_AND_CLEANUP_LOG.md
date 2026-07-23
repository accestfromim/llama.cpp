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

## 2026-07-16 W2 W64-scale expansion sweep

This sweep migrated the W1 32-row expansion strategy to the four-weight W2
prefill path. Measurements used the W2 7B model with Metal source compilation,
`-ngl 99 -t 8 -p 256 -n 0`, and the W64-scale path enabled. The original W2
W64-scale `8x16/K8` implementation measured `82.00 +/- 0.10 tok/s`.

### Output-row tile sweep

The first sweep only enlarged the output-row tile. Weight codes were still
loaded repeatedly while constructing each 8-wide coefficient tile.

| row x column tile | K tile | pp256 tok/s |
|---:|---:|---:|
| `16x16` | 8 | `94.50 +/- 0.02` |
| `32x16` | 8 | `99.57 +/- 0.12` |
| `64x16` | 8 | `95.94 +/- 0.22` |

The 32-row tile was the best balance. The 64-row tile increased thread count,
threadgroup storage, and live state enough to lose performance.

### Packed weight-code preload

Each thread now loads its assigned bytes from `U.s0`, `U.s1`, `W.s0`, and
`W.s1` once per W64 block. Two `uint4` values retain the codes for the two
output rows owned by that thread; shifts extract successive 2-bit K positions.
The four coefficient matrices remain FP32 while being formed and are converted
to FP16 only when staged for the hardware MMA operation.

| row x column tile | K tile | pp256 tok/s |
|---:|---:|---:|
| `16x16` | 8 | `138.79 +/- 0.69` |
| `32x16` | 8 | `150.08 +/- 0.33` |
| `64x16` | 8 | `142.92 +/- 0.25` |

This is the main gain: the selected `32x16/K8` path is about 83% faster than
the original `8x16/K8` W64-scale path.

### K tile and layout sweep

An initial generic implementation stored coefficient matrices with a
`k_tile` stride. It regressed K8/K16/K32 to `124.18`, `125.00`, and
`123.47 tok/s`. Storing every 8x8 MMA coefficient block contiguously restored
performance.

| row x column tile | K tile | pp256 tok/s | decision |
|---:|---:|---:|---|
| `32x16` | 8 | `148.57 +/- 1.92`; repeat `150.09 +/- 0.22` | keep |
| `32x16` | 16 | `151.82 +/- 0.53`; reverse-order repeat `149.89 +/- 2.33` | no stable gain |
| `32x16` | 32 | `149.00 +/- 1.35` | remove |
| `32x16` | 64 | `119.74 +/- 0.13` | remove |

K64 expands coefficient staging to 16 KiB and keeps eight MMA substeps in one
unrolled live range. Its register/instruction-window cost dominates the saved
barriers. K16 did not reproduce a gain over K8, so the smaller and more stable
K8 tile is retained.

### Negative variants

| variant | pp256 tok/s | conclusion |
|---|---:|---|
| FP16 coefficient construction, `16x16/K8` | `128.58 +/- 5.76` | slower |
| FP16 coefficient construction, `32x16/K8` | `134.53 +/- 0.79` | slower |
| FP16 coefficient construction, `64x16/K8` | `127.86 +/- 0.14` | slower |
| SIMD-local activation copies, `32x16/K8` | `107.21 +/- 0.09` | remove |

The SIMD-local variant removed threadgroup-wide barriers by giving each of the
four SIMD groups a private activation tile. It was compared with the ordinary
path at about `125.53 tok/s` in the same thermally throttled period and was
still roughly 15% slower. Re-reading activation four times costs more than the
barriers saved, so activation remains cooperatively loaded and shared.

Sustained later runs thermally throttled the M4 from about 150 to 125 tok/s.
Those absolute hot-state values are not mixed into the cold tile ranking above.
Temporary K/SIMD-local dispatch variables and losing shader instances were
removed after the sweep. The retained W2 path is `32x16/K8`. A final adjacent
comparison measured `8x16 = 83.61 +/- 0.06`, `16x16 = 118.14 +/- 0.37`, and
`32x16 = 124.91 +/- 0.09 tok/s`. The absolute values were thermally limited,
but the ordering remained clear. The row selector was then removed.

The final prefill implementation has no Fairy2i prefill environment variables.
For `act_rows > 1`, Metal always stages BF16 activation to FP16 once, uses the
W64 scale-sharing format invariant, and dispatches the fixed W1 or W2 32x16
kernel. Decode selection is unchanged.

The final Shader Profiler capture is
`/private/tmp/fairy-prefill-w2-row32-k8-final-20260716.trace`. In the hot-state
capture, the main prefill frame contained 165 compute intervals with
`1967.929 ms` union time and a `1981.409 ms` span. Shader Profiler identified
165 dispatches of
`kernel_fairy2i_wide_linear_w2_half_w64scale_mma32x16`; it was the dominant
sampled shader. This trace is for structural inspection rather than cold-state
throughput comparison.

## Final prefill kernels

Only three Fairy2i prefill kernels remain in Metal:

- `kernel_fairy2i_act_half_64_stage_bf16`
- `kernel_fairy2i_wide_linear_w1_half_w64scale_mma32x16_k16`
- `kernel_fairy2i_wide_linear_w2_half_w64scale_mma32x16`

W1 uses a 32-row by 16-column output tile with K16 staging. W2 uses the same
output tile with K8 staging. Both cooperatively stage one activation tile,
construct four coefficient matrices, and execute them with simdgroup MMA.
The old q8, BF16-direct, half 8x16, q8-postscale, and W64 8x16/16x16 prefill
kernels and their dispatch controls were removed.

Final combined checks with `-t 8 -p 256 -n 128 -r 3` and no Fairy2i Metal
path-selection variables measured:

| model | pp256 tok/s | tg128 tok/s |
|---|---:|---:|
| W2 Llama2 7B | `150.13 +/- 0.28` | `20.13 +/- 0.64` |
| W1 Qwen3 | `173.85 +/- 4.56` | `25.01 +/- 0.23` |

## Decode kernels tried

### Attempt matrix

| area | tile / shape tried | activation path | observed result | decision |
|---|---|---|---|---|
| W2 prefill | old `tile4x4` env path | activation quantized | historical experiment only; replaced by MMA path | remove dispatch/test knobs |
| W2 prefill | old `tile8x4` env path | activation quantized | historical experiment only; replaced by MMA path | remove dispatch/test knobs |
| W2 prefill | `q8_mma8x16` | one int8 activation quantization pass | historical pp256 around 62 tok/s on W2 7B | removed |
| W1 prefill | `q8_mma8x16` | one int8 activation quantization pass | historical pp256 around 85 tok/s on complete Qwen3 W1 graph | removed |
| W2 prefill | `half_w64scale_mma32x16`, K8 | one shared FP16 activation stage | retained path; cold pp256 around 150 tok/s | keep |
| W1 prefill | `half_w64scale_mma32x16_k16` | one shared FP16 activation stage | retained path; cold pp256 around 172 tok/s | keep |
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

## Final cleanup

Keep:

- BF16-to-FP16 activation staging used by prefill
- W2 prefill `half_w64scale_mma32x16`, K8
- W1 prefill `half_w64scale_mma32x16_k16`
- W2 BF16 direct decode `tile4x1_w8`
- W1 BF16 direct decode `tile8x1_w16`
- W1/W2 full-row no-bias function-constant decode specializations

Remove from Metal:

- q8, q8-postscale, BF16-direct, old half, and smaller W64 prefill kernels
- all Fairy2i Metal path-selection environment variables
- q16 activation-quantized W2 decode tile2/tile4/tile8/tile16 kernels
- Metal W2 LUT decode helpers and kernel
- W2 BF16 direct decode tile2/tile8/w4 variants
- W1 BF16 direct decode tile4/tile16/w8 variants

Eligible decode shapes now select the function-constant specialization
automatically. Adjacent `tg128` checks measured W2 `16.61 -> 20.31 tok/s` and
W1 `25.46 -> 26.06 tok/s`; the former values are the generic decode paths.
No live `GGML_METAL_FAIRY2I_*` path-selection option remains.

The benchmark still uses `LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2=1` to select the
shared graph-level fused operation. That gate is also used by CPU backends and
is intentionally outside this Metal-only cleanup. `GGML_METAL_FORCE_SOURCE=1`
only controls shader-library loading during development.

The cleanup intentionally does not touch CPU/NEON/LUT implementations.
