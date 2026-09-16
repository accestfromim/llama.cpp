# M5 Row4: exact INT2 TensorOps

`GGML_METAL_ROW4_M5_INT2=1` enables an opt-in native **A8 × signed INT2 → INT32** GPU path on Apple GPU family 10 with macOS/iOS 27 and MSL 4.1. It has been executed and validated on an M5 Max running macOS 27.0 (26A428). Apple's [WWDC26 TensorOps session](https://developer.apple.com/videos/play/wwdc2026/330/) describes the new quantized formats; the [Metal shading language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) defines `int2b_format`.

Set the variables before Metal device initialization. These switches default to off. The original GGUF code stream, A8 quantization, signed per-output scales, and BF16 output contract are preserved.

## Exact representation

A Row4 code is `u_axis | (v_axis << 2)`, with each axis selecting `+R`, `-R`, `+I`, or `-I`. The original numeric codebook includes +2, which signed INT2 cannot represent. Instead, expand each code into four ternary columns:

```text
W_basis = [u_real, u_imag, v_real, v_imag]  // each is -1, 0, or 1
T = A8 @ W_basis                         // INT32 accumulation

q0 = T.u_real + T.v_real
q1 = T.v_imag - T.u_imag
q2 = T.u_imag + T.v_imag
q3 = T.u_real - T.v_real
```

The reconstruction happens in integer registers before `row4_finish_i32` applies either scale or BF16 rounding. The existing conservative `254*K` accumulator bound remains in force. There is one GEMM with unchanged logical dimensions.

The tested M32N128 SG4, M64N64 SG4, and M64N128 SG8 cooperative layouts keep each aligned output quartet in consecutive registers of one thread. Revalidate this implementation-dependent property and exactness tests when changing the runtime, tile geometry, or operand format.

Four ternary coefficients occupy one byte. INT2 device tensors use a 128-byte aligned allocation offset and a K-major row stride padded to 512 elements; logical O and output/scaling strides stay unchanged. Cache allocation and scratch overlap checks include this padding.

| Projection | O / K | Expanded INT4 | Expanded INT2 |
| --- | --- | ---: | ---: |
| QKV | 6144 / 4096 | 12 MiB | 6 MiB |
| Attention output | 4096 / 4096 | 8 MiB | 4 MiB |
| Gate/up | 24576 / 4096 | 48 MiB | 24 MiB |
| Down | 4096 / 12288 | 24 MiB | 12 MiB |

The checkpoint still stores its original Row4 code stream at approximately one bit per coefficient. These sizes describe the GPU expansion, not a new checkpoint quantization.

## Dispatch

Prefill uses device preexpansion at B >= 512 for eligible v1/Pair2 shapes. With native INT2, large Pair2 projections (O >= 4096, K >= 4096) can start at B128. Existing divisibility and overflow gates still apply. Full Pair2 M64 tiles retain cooperative stores, weight lookahead, and the existing fusion guards. Other shapes retain the current online or LUT kernels.

The existing `GGML_METAL_ROW4_INT4_CACHE` variable also controls INT2 prefill caching; its name is retained for compatibility. The complete 36-layer reference-model cache is 1,656 MiB, versus 3,312 MiB for expanded INT4. Cache construction is required on first use and after weight invalidation. Single-stream decode retains the compressed LUT path.

`GGML_METAL_ROW4_M5_TERNARY_BASIS=1` remains available as an INT4 control for the reconstruction, independently of native INT2. Native INT2 always implies the ternary basis. If native source compilation fails, initialization retries MSL 4.0 INT4; the existing portable fallback remains available. Runtime path markers distinguish actual A8/I2/I32 execution from the fallback.

## Validation and measurement

Build:

```sh
cmake -B build-rel -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NATIVE=ON \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build-rel --target llama-bench llama-cli test-row4 test-backend-ops -j 12
```

Require all native prefill tile markers, real projection shapes, and cache mutation/fallback tests:

```sh
GGML_METAL_ROW4_M5_INT2=1 \
LLAMA_ROW4_REQUIRE_INT2_TESTS=1 \
LLAMA_ROW4_REQUIRE_M5_TENSOROPS_TESTS=1 LLAMA_ROW4_REAL_SHAPE_TESTS=1 \
build-rel/bin/test-row4

GGML_METAL_ROW4_M5_INT2=1 \
build-rel/bin/test-backend-ops test -b Metal -o ROW4_LINEAR
```

The tests cover all 16 codes, signed/zero scales, cancellation, K128/K512/K12288, B512/B544, non-power-of-two O1152/K768, both physical layouts, fused epilogues, and cache updates. Full-model comparisons cover pp128/pp512/pp643/pp2048, with subsequent decode checks.

For prefill and single-stream decode, use `perf/scripts/run_row4_bench.sh metal pp|tg "$MODEL" "$TAG"`, with `BINARY="$PWD/build-rel/bin/llama-bench"`, `THREADS=8`, `REPS=5`, `BATCH=2048`, `UBATCH=512`, `N_PROMPT=128|512|2048`, and `N_GEN=128`. Compare INT2 off/on serially in A/B/B/A order, with warmup and the harness's 15-second cooldown. BF16 KV and FlashAttention stay enabled.

Look for `M5 MPP TensorOps exact A8/I2/I32` and `native INT2 ternary-basis/I32 reconstruct` in the actual Row4 projection logs.


On M5 Max (18 CPU / 40 GPU cores, 128 GiB), macOS 27.0 (26A428), Apple Clang 21.0.0 and SDK 27.0, serial A/B/B/A measurements gave the following mean token rates. The model is the 8.192B, 36-layer `qwen3-row4-v2-pair2.gguf`; GPU layers=99, host threads=8, BF16 KV, FlashAttention, batch=2048, ubatch=512, warmup enabled, weight cache disabled. Power was AC/automatic (`powermode=0`), without fixed clocks. The base revision was `da3c90c278d2ef36bb19129f59cad195d9a35c63`.

| Workload | Original INT4 tok/s | Native INT2 tok/s | Change |
|---|---:|---:|---:|
| pp128 | 3155.28 | 3969.22 | +25.80% |
| pp512 | 4699.01 | 4883.90 | +3.93% |
| pp2048 | 4040.88 | 4167.02 | +3.12% |
| tg128, B1 | 145.25 | 145.28 | +0.02% |

pp128/tg128 use ten samples per mode; pp512/pp2048 use six. B1 dispatch is unchanged. Raw artifacts are `prefill128-abba/` and `final-prefill-abba/` within the experiment directory below. These are the recorded measurements before arranging the existing implementation into commits; the prefill algorithm is unchanged by that arrangement.

Experiments and raw logs are saved in `/Users/1806-admin/row4-int2-opt-20260915/`, including standalone layout probes, kernel sweeps, whole-model bitwise comparisons, build/static checks, and balanced benchmarks. The Chinese report in that directory records the machine, source revision, commands, final measurements, and scope of each result. The pp128 gain includes lowering the preexpansion threshold and changing the tile/dispatch path; it is not an isolated measurement of INT2 arithmetic throughput.

The earlier 2026-09-08 INT4-only control on macOS 26.5.1 showed no speedup. That historical experiment is retained in `/Users/1806-admin/row4-int2-prefill-20260908/`; its results do not measure native INT2.

Cached multi-stream decode is an additional opt-in path described in [the cached decode guide](ROW4_M5_INT2_CACHED_DECODE.md).
