# Android ARM64 Dispatch Notes

## Why Android `arm64-v8a` Cannot Enable dotprod Globally

- `arm64-v8a` means AArch64 ABI compatibility, not guaranteed support for every optional ARMv8.x feature.
- dotprod is optional across the `arm64-v8a` device set.
- If the whole native target is compiled as dotprod-only, the resulting library can become incompatible with valid `arm64-v8a` devices that do not expose dotprod.

Therefore:
- generic code must stay compatible with baseline `arm64-v8a`
- dotprod code must be isolated
- runtime feature detection must choose the right path on the device

## How CMake Handles dotprod In This Patch

- Generic ARM backend sources are compiled normally.
- The dedicated dotprod implementation source is:
  - `ggml/src/ggml-cpu/arch/arm/quants-ifairy-dotprod.c`
- Only that source receives:
  - `-march=armv8.2-a+dotprod`
- The flag is applied only for Android `arm64-v8a` in:
  - `ggml/src/ggml-cpu/CMakeLists.txt`

This avoids:
- global target-wide dotprod enablement
- accidental loss of fallback compatibility

## How Runtime Dispatch Works

- Public entry stays:
  - `ggml_vec_dot_ifairy_q16_K`
- On first use, dispatch initializes once with `pthread_once`.
- The selector checks:
  1. whether a dotprod implementation was compiled into the binary
  2. whether the running CPU reports dotprod via `getauxval(AT_HWCAP)`
  3. otherwise whether NEON is available
  4. otherwise falls back to the generic scalar path
- The chosen function pointer is cached for later calls.

## Runtime Feature Detection Location

- Runtime CPU probing is implemented in:
  - `ggml/src/ggml-cpu/ggml-cpu.c`
- On AArch64 Linux/Android it uses cached `getauxval(AT_HWCAP)` results.
- The dispatch layer reuses the public helpers:
  - `ggml_cpu_has_neon()`
  - `ggml_cpu_has_dotprod()`

## Included iFairy vecdot Paths

- scalar fallback:
  - `ggml_vec_dot_ifairy_q16_K_generic`
- ordinary NEON:
  - `ggml_vec_dot_ifairy_q16_K_neon`
- dotprod fast path:
  - `ggml_vec_dot_ifairy_q16_K_dotprod`

## How To Confirm The Selected Path

- In debug builds, the dispatch layer logs once:
  - `scalar`
  - `neon`
  - `dotprod`
- The log is emitted from the one-time dispatch initialization path, not from the inner loop.

Expected debug message shape:

```text
selected iFairy vecdot path: scalar
selected iFairy vecdot path: neon
selected iFairy vecdot path: dotprod
```

## Android-Specific Scope

- This patch only targets CPU-side dispatch inside the current `llama.android` build flow.
- It does not add:
  - GPU dispatch
  - OpenCL / Vulkan / NPU paths
  - a parallel Android project
