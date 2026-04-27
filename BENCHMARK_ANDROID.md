# Benchmark Android

## Verified Device

- model: `FOA-AL00`
- processor: `Snapdragon 778G 4G`
- OS: `HarmonyOS 4.2.0`
- API: `31`
- RAM: `8.0 GB`

## Verified Model

- file: `ifairy.gguf`
- size in app UI: `576 MB`
- runtime model description: `ifairy 700M IFairy`

## Built-In Benchmark Result

Captured on 2026-04-04 from the app UI after triggering:

```bash
/tmp/android-sdk/platform-tools/adb shell am start \
  -n com.example.llama/.MainActivity \
  --es codex_action bench
```

Warm-up result:

```text
| model | size | params | backend | test | t/s |
| --- | --- | --- | --- | --- | --- |
| ifairy 700M IFairy | 0.54GiB | 0.83B | (Android) | pp 8 | 0.16 ± 0 |
| ifairy 700M IFairy | 0.54GiB | 0.83B | (Android) | tg 4 | 0.64 ± 0 |
```

Additional timing:

- warm-up time: `55.887777062 seconds`
- long benchmark path was skipped because warm-up exceeded the `5.0s` threshold

## Implemented Support

The app exposes a `Bench` button backed by the existing JNI benchmark function.

Current benchmark output path:

- UI console
- `LLAMA_ANDROID` log

## Still Pending

- model load time
- first token latency
- longer sustained tok/s runs
- 3-minute stability run
- approximate memory usage
