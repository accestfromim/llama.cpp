# M5 native INT2 split-K decode

This experiment extends `GGML_METAL_ROW4_M5_INT2_DECODE=1` with cached,
exact split-K kernels for the Qwen3 Row4 Pair2 QKV, O and down projections.
It targets 4–16 simultaneous decode streams. Single-stream and two-stream
decode retain their compressed LUT kernels.

## Algorithm and numerical contract

The original Row4 stream already stores each complex axis in two bits. A
native INT2 type by itself therefore does not reduce bandwidth relative to
that compressed LUT implementation. The existing native cache stores the
four real basis components `[ur, ui, vr, vi]`, each in `{-1, 0, 1}`, using
four bits per complex weight. This change reuses that cache and its existing
lifetime/invalidation rules; it does not change the GGUF format.

For small activation batches, a full-K matrix multiply exposes too few
independent output tiles. Divide the K reduction among four or eight
threadgroups, each running native A8 × INT2 → INT32 with BK512. Store integer
partials in the otherwise unused local weight-expansion scratch. A second
kernel sums the partials and reconstructs each quartet:

```
y0 = ur + vr
y1 = vi - ui
y2 = ui + vi
y3 = ur - vr
```

Only then apply the original activation scale, individual signed BF16 row
scale, and exact BF16 rounding. No floating-point partial sum or intermediate
BF16 rounding is introduced. Global coordinates are used to store cooperative
fragments; the new kernels do not assume that a quartet stays in one lane.

| O | K | Decode streams | Matrix tile | SIMDgroups | K partitions |
|---:|---:|---:|---|---:|---:|
| 6144 | 4096 | 4–8 | M8N128 | 4 | 4 |
| 6144 | 4096 | 9–16 | M16N64 | 4 | 4 |
| 4096 | 4096 | 4–16 | M16N64 | 4 | 4 |
| 4096 | 12288 | 4–8 | M8N64 | 1 | 8 |
| 4096 | 12288 | 9–16 | M16N64 | 4 | 4 |

The 24576 × 4096 gate/up projection retains the previous cached INT2 kernel.
Partial activation tiles are zero-padded using the existing quantizer, and
only valid rows are stored. Selection requires Pair2, native INT2 support,
a ready persistent cache, and the existing INT32 accumulation bound. Cache
writes, unavailable cache, unsupported runtime, other shapes and fused
single-stream residual cases retain the original paths. Pipeline probing
fails back to the original decode selector.

The benefit is improved GPU occupancy and weight reuse across decode streams,
not a demonstrated increase in the arithmetic throughput of an INT2
instruction. Experiments with online expansion, transposed-right tiles and
compact dual INT2/UINT2 views are preserved in the raw artifact directory;
they are not enabled in the production selector.

## Enable

On M5 with macOS 27 / MSL 4.1, build the embedded Metal source in Release:

```sh
cmake -B build-rel -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NATIVE=ON \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build-rel --target llama-bench llama-cli test-row4 test-backend-ops -j 12
export GGML_METAL_ROW4_M5_INT2=1
export GGML_METAL_ROW4_M5_INT2_DECODE=1
export GGML_METAL_ROW4_INT4_CACHE=1
```

`INT4_CACHE` is the historical environment-variable name and also controls
native INT2 caches. These experimental flags remain off by default. Look for
`A8/I2/I32 ... cached decode split-K4` or `split-K8` in the path log. With
`GGML_METAL_FUSION_DEBUG=2`, each selected dispatch also logs
`Row4 INT2 split-K decode: ... I32-merge`.

## Validation and performance

| 并行序列 | 修改前 tok/s | 新算法 tok/s | 变化 |
|---:|---:|---:|---:|
| 1 | 143.96 | 146.51 | +1.77% |
| 2 | 222.32 | 223.84 | +0.69% |
| 4 | 332.91 | 342.77 | +2.96% |
| 8 | 522.65 | 653.99 | +25.13% |
| 16 | 785.71 | 1113.15 | +41.67% |

B1/B2 retain identical dispatch; their differences are timing noise.

Final measurements are recorded after serial A/B/B/A comparisons against the
frozen binary from before this change, with the same earlier INT2 optimizations
enabled on both sides. See the local report and raw artifacts:

`/Users/1806-admin/row4-int2-decode-20260916/`

The benchmark uses the 8.192B Qwen3 Row4 v2 Pair2 model, 128 prompt tokens per
stream followed by 128 decode steps, BF16 KV, FlashAttention, eight CPU
threads, one warmup and three measured repetitions per process. A/B/B/A gives
six samples per variant; tok/s is aggregate throughput across streams.
No concurrent benchmark, build or static-analysis process is used.

The standalone benchmark and bitwise-logit checker source are preserved under
`probes/`; `run-abba.py`, `run-validation.py`, frozen libraries and SHA256
provenance reproduce the comparison. Compiler, power mode, machine details,
CMake flags and exact commands are included with the raw logs. Clocks are not
pinned, so small throughput changes should not be interpreted as assured gains.

Tests include exact cached-vs-compressed results, signed/zero/subnormal row
scales, all 16 codes, partial activation tiles, cache reuse and invalidation,
write aliases, graph writers, buffer reuse/lifetimes and cache fallback. The
cache tests assert that split-K dispatches actually occurred. Native and
portable Row4 suites, Metal `ROW4_LINEAR` backend checks and whole-model
bitwise logits are checked, together with scoped clang-format and clang-tidy.
