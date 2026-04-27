# Android Build Chain

## Entry Point

- Gradle root: `examples/llama.android`
- App module: `examples/llama.android/app`
- Native library module: `examples/llama.android/llama`
- Native CMake entry: `examples/llama.android/llama/src/main/cpp/CMakeLists.txt`

## How The Android Module Uses This Repository

The Android JNI library does not fetch another `llama.cpp` copy. It pulls the current repository root directly:

```cmake
add_subdirectory(../../../../../../ build-llama)
```

From `examples/llama.android/llama/src/main/cpp/CMakeLists.txt`, that path resolves to the repository root and builds the in-tree `llama`, `ggml`, `ggml-base`, `ggml-cpu`, and `common` targets.

## ABI Scope

Both Gradle modules now restrict NDK output to:

- `arm64-v8a`

Verified from the packaged APK:

- `lib/arm64-v8a/libggml-base.so`
- `lib/arm64-v8a/libggml-cpu.so`
- `lib/arm64-v8a/libggml.so`
- `lib/arm64-v8a/libllama.so`
- `lib/arm64-v8a/libllama-android.so`

## CMake Arguments Used

Current Android native configuration adds:

- `-DLLAMA_CURL=OFF`
- `-DLLAMA_BUILD_COMMON=ON`
- `-DGGML_LLAMAFILE=OFF`
- `-DGGML_NATIVE=OFF`
- `-DGGML_OPENMP=OFF`

Debug builds are no longer forced to `Release`.

Verified from:

- `examples/llama.android/llama/.cxx/Debug/.../hash_key.txt`

That file shows:

- `-DCMAKE_BUILD_TYPE=Debug`

## Dispatch Sources Verified In Android Compile Commands

Verified from:

- `examples/llama.android/llama/.cxx/Debug/1f4z9t2i/arm64-v8a/compile_commands.json`

The Android build compiles:

- `ggml/src/ggml-cpu/arch/arm/quants-ifairy.c`
- `ggml/src/ggml-cpu/arch/arm/quants-ifairy-dotprod.c`

The dotprod file is compiled with:

- `-march=armv8.2-a+dotprod`

That preserves multi-path coexistence without globally compiling the whole CPU backend as dotprod-only.

## Environment Used In This Session

This machine initially lacked a usable Android toolchain. For this session:

- JDK 17 was installed under `/tmp/jdk17`
- Android SDK root was installed under `/tmp/android-sdk`
- `examples/llama.android/local.properties` points `sdk.dir=/tmp/android-sdk`

## Verified Result

Successful command:

```bash
cd examples/llama.android
JAVA_HOME=/tmp/jdk17 PATH=/tmp/jdk17/bin:$PATH ./gradlew :app:assembleDebug
```

Result:

- `BUILD SUCCESSFUL`
