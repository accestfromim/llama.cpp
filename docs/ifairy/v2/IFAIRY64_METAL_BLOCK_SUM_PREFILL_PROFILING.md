# iFairy64 Metal W2 Block-Sum Prefill Profiling

Date: 2026-07-01

Branch: `gjq/metal`

Machine: Apple M4, Metal GPU family Apple9

Model:
`/Users/a1806/.cache/huggingface/hub/models--PKU-DS-LAB--Fairy-plus-minus-i-700M/snapshots/c274e9bb0b9a82fbe0bc20eeedbf4b8a3fcd358b/ifairy.gguf`

Model size reported by `llama-bench`: `ifairy 700M IFairy`, 827.35 M params,
548.73 MiB.

## Path Under Test

This profile uses the current Metal W2 block-sum accumulation path:

```sh
GGML_METAL_FORCE_SOURCE=1
LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2=1
GGML_METAL_FAIRY2I_W2_BLOCK_SUM=1
```

The path is the two-stage Metal implementation:

1. Quantize activation to the block-sum scratch format.
2. `kernel_fairy2i_wide_linear_w2_block_sum_partial`
3. `kernel_fairy2i_wide_linear_w2_block_sum_reduce`

## Validation

```sh
cmake --build build-rel-metal --target llama-bench test-fairy2i -j 8
env GGML_METAL_FORCE_SOURCE=1 ctest --test-dir build-rel-metal -R test-fairy2i --output-on-failure
```

Result:

```text
100% tests passed, 0 tests failed out of 1
```

## Bench Results

### pp128 / tg1

Command:

```sh
env GGML_METAL_FORCE_SOURCE=1 \
    LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2=1 \
    GGML_METAL_FAIRY2I_W2_BLOCK_SUM=1 \
    ./build-rel-metal/bin/llama-bench \
    -m /Users/a1806/.cache/huggingface/hub/models--PKU-DS-LAB--Fairy-plus-minus-i-700M/snapshots/c274e9bb0b9a82fbe0bc20eeedbf4b8a3fcd358b/ifairy.gguf \
    -ngl 99 --n-prompt 128 --n-gen 1 --repetitions 1
```

Result:

```text
pp128: 140.76 tok/s
tg1:    20.18 tok/s
```

### pp512 / tg1

Command:

```sh
env GGML_METAL_FORCE_SOURCE=1 \
    LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2=1 \
    GGML_METAL_FAIRY2I_W2_BLOCK_SUM=1 \
    ./build-rel-metal/bin/llama-bench \
    -m /Users/a1806/.cache/huggingface/hub/models--PKU-DS-LAB--Fairy-plus-minus-i-700M/snapshots/c274e9bb0b9a82fbe0bc20eeedbf4b8a3fcd358b/ifairy.gguf \
    -ngl 99 --n-prompt 512 --n-gen 1 --repetitions 1
```

Result:

```text
pp512: 151.02 tok/s
tg1:    20.93 tok/s
```

The larger prompt improves prefill throughput only modestly. That suggests token
dimension amortization helps, but the current two-stage path is still dominated
by graph/dispatch granularity and intermediate memory traffic.

## Trace Collection

### pp128

```sh
xcrun xctrace record \
    --template 'Metal System Trace' \
    --output /private/tmp/fairy2i-metal-prefill-blocksum.trace \
    --env GGML_METAL_FORCE_SOURCE=1 \
    --env LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2=1 \
    --env GGML_METAL_FAIRY2I_W2_BLOCK_SUM=1 \
    --launch -- ./build-rel-metal/bin/llama-bench \
    -m /Users/a1806/.cache/huggingface/hub/models--PKU-DS-LAB--Fairy-plus-minus-i-700M/snapshots/c274e9bb0b9a82fbe0bc20eeedbf4b8a3fcd358b/ifairy.gguf \
    -ngl 99 --n-prompt 128 --n-gen 1 --repetitions 1
```

Exports:

```text
/private/tmp/fairy2i-blocksum-cmd.xml
/private/tmp/fairy2i-blocksum-encoders.xml
/private/tmp/fairy2i-blocksum-gpu.xml
/private/tmp/fairy2i-blocksum-shaders.xml
```

### pp512

```sh
xcrun xctrace record \
    --template 'Metal System Trace' \
    --output /private/tmp/fairy2i-metal-prefill512-blocksum.trace \
    --env GGML_METAL_FORCE_SOURCE=1 \
    --env LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2=1 \
    --env GGML_METAL_FAIRY2I_W2_BLOCK_SUM=1 \
    --launch -- ./build-rel-metal/bin/llama-bench \
    -m /Users/a1806/.cache/huggingface/hub/models--PKU-DS-LAB--Fairy-plus-minus-i-700M/snapshots/c274e9bb0b9a82fbe0bc20eeedbf4b8a3fcd358b/ifairy.gguf \
    -ngl 99 --n-prompt 512 --n-gen 1 --repetitions 1
```

Exports:

```text
/private/tmp/fairy2i-blocksum512-cmd.xml
/private/tmp/fairy2i-blocksum512-encoders.xml
/private/tmp/fairy2i-blocksum512-gpu.xml
/private/tmp/fairy2i-blocksum512-shaders.xml
```

## Trace Summary

The GPU interval rows below are filtered to `llama-bench` only. System GPU
activity from WindowServer and other processes is excluded.

| case | command buffers | encoder median | encoder max | target GPU intervals | target GPU active/span | target GPU median gap | target GPU max gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pp128 block-sum | 988 | 4.417 us | 813.459 us | 882 | 10.53% | 7.590 ms | 19.962 ms |
| pp512 block-sum | 987 | 4.416 us | 913.042 us | 900 | 5.52% | 13.003 ms | 71.615 ms |

Command-buffer side:

| case | command buffer span | command buffer duration sum | command buffer median | command buffer max |
| --- | ---: | ---: | ---: | ---: |
| pp128 block-sum | 2048.477 ms | 15.699 ms | 8.000 us | 836.250 us |
| pp512 block-sum | 7334.088 ms | 17.190 ms | 8.583 us | 935.417 us |

Encoder side:

| case | encoder span | encoder duration sum | encoder median | encoder max |
| --- | ---: | ---: | ---: | ---: |
| pp128 block-sum | 2048.457 ms | 10.393 ms | 4.417 us | 813.459 us |
| pp512 block-sum | 7334.069 ms | 11.454 ms | 4.416 us | 913.042 us |

Shader-profiler list exposed only generic Metal kernels in this command-line
template:

```text
kernel_mul_row_c4_fuse_1
kernel_set_rows_f16
kernel_mul_mm_f16_f32
kernel_soft_max_f32_4
kernel_cpy_f32_f32
kernel_get_rows_f32
kernel_mul_mv_f16_f32_4
```

The block-sum Fairy2i kernels did not appear in `metal-shader-profiler-shader-list`.
This limits the command-line trace to timeline-level conclusions. It does not
provide per-shader occupancy, register pressure, or memory bandwidth counters.

## Findings

1. The current block-sum path is too fragmented for prefill.

   The run submits almost 1000 command buffers/encoders. The median encoder
   encoding duration is only about 4.4 us, while GPU target intervals are
   separated by millisecond-scale gaps. This is a poor shape for saturating the
   M4 GPU.

2. Larger prompt length does not fix the issue.

   `pp512` is only about 7.3% faster than `pp128` in tok/s. If the kernel were
   mainly compute-bound and scaled well with token count, the larger prompt
   should improve utilization more clearly. Instead, command buffer and encoder
   counts remain roughly constant, and the trace still shows large gaps.

3. The two-stage block-sum design is unattractive for prefill.

   The partial stage writes an intermediate tensor sized by output rows, tokens,
   and K blocks. The reduce stage then reads it back. For 2-bit weights the
   arithmetic intensity is already low, so this extra global memory traffic can
   dominate any saving from replacing multiplies with add/sub accumulation.

4. The useful prefill direction is a larger single-kernel tile.

   A better prefill kernel should compute a tile over both output rows and prompt
   tokens, for example `rows x tokens` tiles such as `8x4`, `8x8`, or `16x4`.
   The goal is to decode/load each weight block once and consume it across
   multiple activation columns before moving on. This avoids global partial
   writes and reduces dispatch count.

5. This trace is not enough for occupancy or bandwidth claims.

   The `Metal System Trace` command-line template reports:

   ```text
   Counter Set: (null)
   Shader Timeline: Disabled
   ```

   Therefore, the current data supports a scheduling/fragmentation diagnosis,
   but not a precise statement about occupancy, register spills, or memory
   bandwidth. Those require Instruments GUI or a custom trace template with GPU
   counters and shader timeline enabled.

## Recommendation

For prefill, stop optimizing the current block-sum partial/reduce path as the
primary route. Keep it as a correctness and algorithm probe, but move performance
work to a single-kernel tiled prefill implementation:

1. Tile across multiple prompt tokens and multiple output rows.
2. Reuse each decoded 2-bit weight block across several activation columns.
3. Keep K-block accumulation inside the kernel as much as possible.
4. Avoid global partial buffers unless the split is required by register pressure.
5. Add better labels around Fairy2i Metal dispatches, because current xctrace
   exports do not identify the custom kernels well enough.

## Follow-up: Single-Kernel `tile4x4` Probe

An experimental prefill-only env gate was added:

```sh
GGML_METAL_FAIRY2I_W2_PREFILL_TILE4X4=1
```

When enabled, prefill uses:

```text
kernel_fairy2i_wide_linear_w2_f32_tile4x4
```

The kernel keeps the same 16 output accumulation slots as the existing `tile8x2`
kernel, but changes the tile shape to 4 output rows by 4 prompt tokens. The
intent was to double token reuse per decoded weight block without increasing the
number of output accumulators.

Validation:

```sh
cmake --build build-rel-metal --target ggml-metal test-fairy2i llama-bench -j 8
env GGML_METAL_FORCE_SOURCE=1 ctest --test-dir build-rel-metal -R test-fairy2i --output-on-failure
```

Result:

```text
100% tests passed, 0 tests failed out of 1
```

Bench results, 3 repetitions:

| mode | pp128 | tg1 |
| --- | ---: | ---: |
| direct prefill `tile8x2` | 140.84 +/- 1.47 tok/s | 19.69 +/- 0.90 tok/s |
| direct prefill `tile4x4` | 139.75 +/- 0.89 tok/s | 19.32 +/- 0.37 tok/s |
| block-sum prefill | 140.89 +/- 0.88 tok/s | 20.47 +/- 0.28 tok/s |
| block-sum + prefill `tile4x4` | 140.46 +/- 0.61 tok/s | 20.50 +/- 0.23 tok/s |

For longer prefill:

| mode | pp512 | tg1 |
| --- | ---: | ---: |
| block-sum prefill | 150.24 +/- 0.18 tok/s | 21.03 +/- 0.32 tok/s |
| block-sum + prefill `tile4x4` | 150.15 +/- 0.31 tok/s | 20.20 +/- 0.67 tok/s |

Conclusion: the simple `4x4` shape does not improve prefill. It reduces the row
tile from 8 rows to 4 rows while increasing token tile from 2 tokens to 4 tokens,
so total output elements per threadgroup remains 16. The lack of improvement
suggests that this kernel is not bottlenecked only by token-side activation
reloads. The next useful probes are either:

1. a larger tile that increases total work per threadgroup, such as `8x4`, if
   register pressure is still acceptable; or
2. a different mapping that keeps 8 or 16 output rows while sharing activation
   values across tokens with fewer per-thread accumulator arrays.

### `tile4x4` Trace Comparison

Additional Metal System Trace runs were collected for direct prefill:

```text
/private/tmp/fairy2i-metal-prefill-direct-tile8x2.trace
/private/tmp/fairy2i-metal-prefill-direct-tile4x4.trace
```

Both runs used:

```sh
GGML_METAL_FORCE_SOURCE=1
LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2=1
--n-prompt 128 --n-gen 1 --repetitions 1
```

The `tile4x4` run additionally used:

```sh
GGML_METAL_FAIRY2I_W2_PREFILL_TILE4X4=1
```

Exported tables:

```text
/private/tmp/fairy2i-direct-tile8x2-cmd.xml
/private/tmp/fairy2i-direct-tile8x2-encoders.xml
/private/tmp/fairy2i-direct-tile8x2-gpu.xml
/private/tmp/fairy2i-direct-tile8x2-shaders.xml
/private/tmp/fairy2i-direct-tile4x4-cmd.xml
/private/tmp/fairy2i-direct-tile4x4-encoders.xml
/private/tmp/fairy2i-direct-tile4x4-gpu.xml
/private/tmp/fairy2i-direct-tile4x4-shaders.xml
```

Summary, with GPU intervals filtered to `llama-bench` only:

| mode | command buffers | encoder median | encoder max | target GPU intervals | target GPU active/span | target GPU median gap | target GPU max gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct `tile8x2` | 988 | 4.292 us | 1447.542 us | 890 | 11.36% | 7.388 ms | 19.853 ms |
| direct `tile4x4` | 988 | 4.333 us | 1521.292 us | 911 | 11.75% | 7.352 ms | 20.721 ms |

Command-buffer side:

| mode | command buffer span | command buffer duration sum | command buffer median | command buffer max |
| --- | ---: | ---: | ---: | ---: |
| direct `tile8x2` | 2086.417 ms | 14.926 ms | 7.979 us | 1473.917 us |
| direct `tile4x4` | 2071.752 ms | 15.391 ms | 7.771 us | 1553.541 us |

Encoder side:

| mode | encoder span | encoder duration sum | encoder median | encoder max |
| --- | ---: | ---: | ---: | ---: |
| direct `tile8x2` | 2086.396 ms | 10.180 ms | 4.292 us | 1447.542 us |
| direct `tile4x4` | 2071.735 ms | 10.575 ms | 4.333 us | 1521.292 us |

The command-line shader-profiler table again exposed only generic kernels:

```text
kernel_mul_row_c4_fuse_1
kernel_set_rows_f16
kernel_mul_mm_f16_f32
kernel_soft_max_f32_4
kernel_cpy_f32_f32
kernel_get_rows_f32
kernel_mul_mv_f16_f32_4
```

The custom Fairy2i prefill kernels did not appear in the exported shader list.

Trace conclusion: `tile4x4` does not reduce graph fragmentation. It submits the
same 988 command buffers, has nearly identical encoder duration distribution,
and does not materially improve target GPU active/span. The small active/span
difference is not enough to explain a throughput win and is consistent with
normal trace variance.

### `tile8x4` Experiment

An additional direct prefill kernel is available behind:

```sh
GGML_METAL_FAIRY2I_W2_PREFILL_TILE8X4=1
```

This keeps the original 8 output-row tile while expanding the activation tile
from 2 to 4 tokens. Each threadgroup therefore produces 32 complex outputs
instead of 16. The intended upside is better weight reuse and fewer prefill
threadgroups; the main risk is doubled per-thread accumulator state and higher
register pressure.

Short 700M benchmark:

```sh
GGML_METAL_FAIRY2I_W2_PREFILL_TILE8X4=1 \
./build-rel-metal/bin/llama-bench \
  -m /Users/a1806/.cache/huggingface/hub/models--PKU-DS-LAB--Fairy-plus-minus-i-700M/snapshots/c274e9bb0b9a82fbe0bc20eeedbf4b8a3fcd358b/ifairy.gguf \
  -ngl 99 --n-prompt 128 --n-gen 1 --repetitions 3
```

Result:

| mode | pp128 | tg1 |
| --- | ---: | ---: |
| direct `tile8x4` | 141.97 +/- 1.23 t/s | 20.71 +/- 0.26 t/s |
| block-sum flag + direct `tile8x4` | 141.92 +/- 0.61 t/s | 20.55 +/- 0.11 t/s |

The result is again within noise for the 700M model. This does not rule out
larger models or larger prefill batches, but for the current small-model test it
does not change the earlier conclusion that graph fragmentation and launch gaps
dominate over this local tile-shape choice.
