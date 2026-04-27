# Android Dispatch Verification

## What Was Verified In Build Output

The Android debug build compiles both:

- `ggml/src/ggml-cpu/arch/arm/quants-ifairy.c`
- `ggml/src/ggml-cpu/arch/arm/quants-ifairy-dotprod.c`

The dotprod implementation is compiled with:

- `-march=armv8.2-a+dotprod`

This was verified in:

- `examples/llama.android/llama/.cxx/Debug/1f4z9t2i/arm64-v8a/compile_commands.json`

## Runtime Log

In debug builds, the selected path logs once during `pthread_once` initialization:

- `LLAMA_COMPLEX dispatch: selected iFairy vecdot path: scalar`
- `LLAMA_COMPLEX dispatch: selected iFairy vecdot path: neon`
- `LLAMA_COMPLEX dispatch: selected iFairy vecdot path: dotprod`

## Logcat Commands

```bash
/tmp/android-sdk/platform-tools/adb logcat -s LLAMA_ANDROID LLAMA_COMPLEX
```

Useful expected events:

- model import
- model load start / success / failure
- generation start
- first token
- stop request
- dispatch path selection

## Current Status

- build-time inclusion verified
- one-time dispatch log added
- target device is attached and runtime inference / benchmark are verified
- actual target-device hit is still not captured in this session
- current observed gap: `adb logcat` on the successful load / generate / stop / benchmark runs did not emit `selected iFairy vecdot path: ...`
- implication: build-time dispatch inclusion is confirmed, but runtime path logging still needs stronger instrumentation closer to the hot kernel if final proof is required
