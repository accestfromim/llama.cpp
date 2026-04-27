# Android Prototype Status

This repository now builds the `examples/llama.android` debug APK from the current `llama.cpp` tree and packages only `arm64-v8a` native libraries.

Current status on 2026-04-04:

- `:app:assembleDebug` succeeds.
- APK output: `examples/llama.android/app/build/outputs/apk/debug/app-debug.apk`
- Packaged native ABI: `arm64-v8a` only.
- Android app UI now exposes:
  - local file picker import
  - copy-to-private-storage under `filesDir/models`
  - model metadata display
  - explicit load / unload
  - prompt input
  - streaming output
  - stop request
  - benchmark trigger
- Debug native build compiles the current repository sources, including:
  - `ggml/src/ggml-cpu/arch/arm/quants-ifairy.c`
  - `ggml/src/ggml-cpu/arch/arm/quants-ifairy-dotprod.c`

Verified on physical device in this session:

- device install succeeds via `adb install -r`
- target device: `FOA-AL00`
- OS / API: `HarmonyOS 4.2.0` / `API 31`
- existing imported model under `filesDir/models/ifairy.gguf` is discovered on startup
- model load succeeds on device
- prompt generation streams successfully on device
- `Stop` interrupts generation successfully on device
- built-in benchmark runs successfully on device

Still pending:

- runtime dispatch-hit log for `selected iFairy vecdot path: ...` has not yet appeared in `adb logcat`
- longer sustained benchmark runs beyond the built-in warm-up path

## Build

Example command used successfully in this session:

```bash
cd examples/llama.android
JAVA_HOME=/tmp/jdk17 PATH=/tmp/jdk17/bin:$PATH ./gradlew :app:assembleDebug
```

If your machine does not already have Android SDK and NDK configured, see [ANDROID_BUILD_CHAIN.md](/home/zybi/projects/llama.cpp/ANDROID_BUILD_CHAIN.md).

## Install

When a device is attached:

```bash
/tmp/android-sdk/platform-tools/adb install -r \
  examples/llama.android/app/build/outputs/apk/debug/app-debug.apk
```

Verified in this session with device serial `9TM9K23517024580`.

## Demo Flow

1. Launch the app.
2. Tap `Import Model`.
3. Pick a local `.gguf` file.
4. Confirm the file is copied into the app private directory.
5. Tap `Load Model`.
6. Enter a prompt and tap `Send`.
7. Tap `Stop` during generation if needed.
8. Tap `Bench` for the built-in minimal benchmark output.

For deterministic adb-side benchmark triggering without screen taps:

```bash
/tmp/android-sdk/platform-tools/adb shell am start \
  -n com.example.llama/.MainActivity \
  --es codex_action bench
```

## Logs

App / JNI tag:

- `LLAMA_ANDROID`

Backend / dispatch / model logs:

- `LLAMA_COMPLEX`

See [ANDROID_DISPATCH_VERIFY.md](/home/zybi/projects/llama.cpp/ANDROID_DISPATCH_VERIFY.md).
