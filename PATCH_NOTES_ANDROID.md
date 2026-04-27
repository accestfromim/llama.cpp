# Patch Notes Android

## Gradle / Build

- restricted both Android modules to `arm64-v8a`
- removed forced `Release` CMake build type from the JNI module
- added `GGML_NATIVE=OFF` for Android cross-compilation stability
- added `GGML_OPENMP=OFF` for Android CPU-only simplicity
- added `examples/llama.android/local.properties` with `sdk.dir=/tmp/android-sdk` for this session

## App / Kotlin

- removed in-app model downloading UI
- added system file picker import flow
- added copy-to-private-storage under `filesDir/models`
- added explicit model metadata display
- added load / unload state handling
- added prompt / streaming output / stop / benchmark controls
- standardized Android log tag usage around `LLAMA_ANDROID`
- moved model discovery initialization into `MainActivity.onCreate()` to avoid relying on Compose recomposition
- added adb automation hook `codex_action=bench` for deterministic benchmark verification without screen tapping

## JNI / Native

- aligned JNI `backend_init` declaration with the current native signature
- standardized JNI log tags to `LLAMA_ANDROID` and `LLAMA_COMPLEX`
- routed backend log callback to Android logcat
- fixed Android build warning for `n_kv_req` logging type

## ggml / Dispatch

- ensured Android debug build compiles the custom iFairy ARM dispatch sources
- fixed `quants-ifairy-dotprod.c` Android build failure by including `simd-mappings.h`
- changed debug dispatch selection logging to a stable one-time `LLAMA_COMPLEX` info log

## Validation Completed

- `:app:assembleDebug` succeeded
- packaged APK contains only `arm64-v8a`
- Android debug compile commands include both custom iFairy ARM source files
- dotprod translation unit is compiled with `-march=armv8.2-a+dotprod`
- APK install verified on real device `FOA-AL00` (`HarmonyOS 4.2.0`, `API 31`)
- on-device model discovery / load / generation / stop verified
- built-in warm-up benchmark verified with `ifairy.gguf`
- warm-up benchmark result:
  - `pp 8 | 0.16 t/s`
  - `tg 4 | 0.64 t/s`
- runtime dispatch log is still pending capture
