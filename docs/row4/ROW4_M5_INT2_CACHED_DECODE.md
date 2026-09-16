# M5 Row4: cached native INT2 multi-stream decode

Enable native INT2 and the persistent weight cache before Metal initialization:

```sh
export GGML_METAL_ROW4_M5_INT2=1
export GGML_METAL_ROW4_INT4_CACHE=1
export GGML_METAL_ROW4_M5_INT2_DECODE=1
```

The new decode switch defaults to off. The historical `INT4_CACHE` name also controls the INT2 cache. This extends the [exact INT2 prefill implementation](ROW4_M5_TERNARY_BASIS.md), preserving the original A8 quantization, integer dot products, signed row scales and BF16 boundary.

Full-K cached kernels cover Pair2 shapes with O divisible by 128 and K divisible by 512:

- B4–B8, O >= 16384 and K=4096: M8N128 BK512.
- B9–B16, O >= 16384 and K=4096: M32N128 BK512 with padded activation rows.
- B12–B16, other O >= 4096 and K >= 4096: M16N64 BK512.

Small batches consume only a ready buffer-owned cache. Disabled caching, writable/ineligible weights, allocation failure or cache preparation failure retain the original compressed kernels. Weights are never fully expanded on each decode step. B1–B3 retain their existing kernels. The cache is 1,656 MiB for the 36-layer reference model, versus 3,312 MiB for expanded INT4, and preserves mutation, alias, graph-write and owner-lifetime invalidation.

The additional M8N128/M16N64 cooperative layouts preserve aligned quartets within one thread. Tests cover B5/B9/B12 tails, exact output, cache updates, aliases and fallback. Runtime markers include `A8/I2/I32`, `native INT2 ternary-basis/I32 reconstruct` and `cached multi-decode`.

```sh
GGML_METAL_ROW4_M5_INT2=1 GGML_METAL_ROW4_M5_INT2_DECODE=1 \
LLAMA_ROW4_REQUIRE_INT2_TESTS=1 LLAMA_ROW4_REQUIRE_INT2_DECODE_TESTS=1 \
LLAMA_ROW4_REQUIRE_M5_TENSOROPS_TESTS=1 LLAMA_ROW4_REAL_SHAPE_TESTS=1 \
build-rel/bin/test-row4

GGML_METAL_ROW4_M5_INT2=1 GGML_METAL_ROW4_M5_INT2_DECODE=1 \
GGML_METAL_ROW4_INT4_CACHE=1 \
build-rel/bin/test-backend-ops test -b Metal -o ROW4_LINEAR
```

The prefill guide records the Release build, M5 Max, compiler, power settings and base revision. Each stream receives pp128 followed by tg128; CPU threads=8, independent sequence KV, BF16 KV, FlashAttention, ubatch=512. Serial A/B/B/A measurements use one warmup and three timed trials per process, six samples per mode, excluding cold cache construction. A uses the original INT4 path; B enables native INT2 and cached decode. Rates below are aggregate throughput across sequences.

| Streams | Original INT4 tok/s | Cached INT2 tok/s | Change |
|---:|---:|---:|---:|
| 4 | 326.97 | 335.51 | +2.61% |
| 8 | 444.93 | 522.74 | +17.49% |
| 16 | 604.62 | 783.59 | +29.60% |

These are recorded results before splitting the existing implementation into commits. Cache-enabled B1/B2 controls did not show gains; their decode paths remain unchanged. Full-model comparisons across pp128/512/2048 and B1/2/4/8/16 followed by 128 decode steps checked 646,791,552 float32 logits bitwise equal to the reference.

Raw artifacts: `/Users/1806-admin/row4-int2-opt-20260915/decode-abba/`, `decode-controls/`, `quality2-summary.json`, `path-proof.json` and `REPORT.zh-CN.md`. The benchmark driver is `run-decode-bench.py`, using `probes/bench-decode.cpp` in that directory. All GPU workloads were run serially.
