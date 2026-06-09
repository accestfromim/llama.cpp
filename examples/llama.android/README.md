# llama.android: Ubuntu/Debian 从零构建指南

本文面向一台没有 Java、Android SDK、NDK 的 Ubuntu/Debian Linux 机器，说明如何安装构建环境、编译 `examples/llama.android`、安装 APK，并按 CPU 或 OpenCL 后端做 smoke test。

本版只覆盖 Android APK 相关的 CPU 和 OpenCL 后端。CUDA、Metal、SYCL 不适用于普通 Android APK；Vulkan 还需要额外的 host shader 工具链和设备验证，暂不纳入本文。

不要提交这些本机产物：`local.properties`、Android SDK/NDK、OpenCL headers/library、APK、模型权重、Gradle 缓存、benchmark 输出。

## 版本基线

| 项目 | 版本 |
| --- | --- |
| OS | Ubuntu/Debian Linux |
| JDK | 17 |
| Gradle wrapper | 8.2 |
| Android Gradle Plugin | 8.2.0 |
| Kotlin Android plugin | 1.9.0 |
| compile SDK / target SDK | 34 |
| min SDK | 31 |
| Android build tools | 34.0.0 |
| Android NDK | 30.0.14904198-beta1 |
| Android SDK CMake | 3.22.1 |
| ABI | arm64-v8a |

官方参考：

- Android command-line tools and `sdkmanager`: https://developer.android.com/tools/sdkmanager
- Android Studio command-line tools download: https://developer.android.com/studio#command-tools
- Android Gradle Plugin 8.2 compatibility: https://developer.android.com/build/releases/agp-8-2-0-release-notes
- llama.cpp OpenCL backend: `docs/backend/OPENCL.md`

## 1. 安装 Linux 基础包

```bash
sudo apt-get update
sudo apt-get install -y \
  openjdk-17-jdk \
  git \
  curl \
  wget \
  unzip \
  cmake \
  ninja-build \
  python3 \
  python3-pip
```

确认 JDK：

```bash
java -version
javac -version
```

`java -version` 应显示 OpenJDK 17。不要用系统自带的 `gradle`；本工程使用 `examples/llama.android/gradlew` 指定的 Gradle wrapper。

## 2. 安装 Android command-line tools

下面把 SDK 安装到 `$HOME/Android/Sdk`。如果要换目录，只需要同步改 `ANDROID_SDK_ROOT`。

```bash
export ANDROID_SDK_ROOT="$HOME/Android/Sdk"
export ANDROID_HOME="$ANDROID_SDK_ROOT"
mkdir -p "$ANDROID_SDK_ROOT/cmdline-tools"
```

从 Android Studio 下载页获取 Linux command-line tools。当前示例使用 `commandlinetools-linux-14742923_latest.zip`；如果该文件已更新，请到官方下载页复制新的 Linux zip 文件名。

```bash
CMDLINE_TOOLS_ZIP=commandlinetools-linux-14742923_latest.zip

curl -L \
  "https://dl.google.com/android/repository/$CMDLINE_TOOLS_ZIP" \
  -o "/tmp/$CMDLINE_TOOLS_ZIP"

rm -rf /tmp/android-cmdline-tools
mkdir -p /tmp/android-cmdline-tools
unzip -q "/tmp/$CMDLINE_TOOLS_ZIP" -d /tmp/android-cmdline-tools
rm -rf "$ANDROID_SDK_ROOT/cmdline-tools/latest"
mv /tmp/android-cmdline-tools/cmdline-tools "$ANDROID_SDK_ROOT/cmdline-tools/latest"
```

配置 shell 环境：

```bash
export PATH="$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/platform-tools:$PATH"

sdkmanager --version
```

建议写入 `~/.bashrc` 或当前 shell 的 profile：

```bash
cat >> ~/.bashrc <<'EOF'
export ANDROID_SDK_ROOT="$HOME/Android/Sdk"
export ANDROID_HOME="$ANDROID_SDK_ROOT"
export PATH="$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/platform-tools:$PATH"
EOF
```

## 3. 安装 SDK / NDK / CMake

NDK `30.0.14904198-beta1` 是 beta 包，需要让 `sdkmanager` 包含 beta channel：

```bash
sdkmanager --install --channel=1 \
  "platform-tools" \
  "platforms;android-34" \
  "build-tools;34.0.0" \
  "cmake;3.22.1" \
  "ndk;30.0.14904198"

yes | sdkmanager --licenses
```

检查安装结果：

```bash
test -f "$ANDROID_SDK_ROOT/platforms/android-34/source.properties"
test -f "$ANDROID_SDK_ROOT/build-tools/34.0.0/source.properties"
test -f "$ANDROID_SDK_ROOT/cmake/3.22.1/bin/cmake"
test -f "$ANDROID_SDK_ROOT/ndk/30.0.14904198/source.properties"

grep -n "Pkg.Revision" "$ANDROID_SDK_ROOT/ndk/30.0.14904198/source.properties"
```

NDK revision 应为 `30.0.14904198-beta1`。如果 `sdkmanager` 仍看不到该 NDK，先运行 `sdkmanager --list --channel=1 | grep 30.0.14904198`，确认 beta channel 是否可见。

## 4. 获取源码并配置本机 SDK 路径

```bash
mkdir -p "$HOME/projects"
cd "$HOME/projects"
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp/examples/llama.android
```

如果已经在本仓库中：

```bash
cd /path/to/llama.cpp/examples/llama.android
```

创建 `local.properties`。这个文件只记录本机路径，不要提交：

```bash
printf 'sdk.dir=%s\n' "$ANDROID_SDK_ROOT" > local.properties
```

## 5. 后端选择

`LLAMA_ANDROID_BACKEND` 控制 `:llama` native module 的 CMake 后端：

| 后端 | 命令值 | 说明 |
| --- | --- | --- |
| CPU | `cpu` | 默认后端，最稳，适合先验证环境。 |
| OpenCL | `opencl` | 面向支持 OpenCL 2.0+ 和 FP16 的 Android GPU，主要是已验证的 Adreno 750/830 等。 |

未知值会让 Gradle 直接报错，避免静默构建出错误 APK。

无论哪个后端，Android 构建都会固定使用：

- `LLAMA_CURL=OFF`：Android app 不依赖 libcurl。
- `LLAMA_BUILD_COMMON=ON`：JNI 层需要 common/chat helper。
- `GGML_LLAMAFILE=OFF`：llamafile 不用于 Android APK。
- `GGML_NATIVE=OFF`：避免按 host CPU 生成不适合 Android 设备的 native flags。
- `GGML_OPENMP=OFF`：避免 Android OpenMP 运行库依赖问题。

CPU 后端会显式设置 `GGML_IFAIRY_LUT_CPU=ON`。这个 iFairy LUT 路线是 CPU-only，会强制关闭加速后端；因此 OpenCL 后端会显式设置 `GGML_IFAIRY_LUT_CPU=OFF`，否则 `GGML_OPENCL=ON` 会被 CMake 覆盖成 OFF。

## 6. APK 后端切换方式

当前 app 运行时不提供 CPU/OpenCL 后端切换开关。后端是在构建 APK 时由 `LLAMA_ANDROID_BACKEND` 决定的：

- CPU APK：用 `LLAMA_ANDROID_BACKEND=cpu` 构建。
- OpenCL APK：用 `LLAMA_ANDROID_BACKEND=opencl` 构建，并提供 OpenCL headers 和 Android arm64 `libOpenCL.so`。

手机上切换后端的方式是安装不同构建产物，而不是在 app UI 或 adb intent extras 中切换。建议测试时同时保留 CPU APK 作为基线和回退路径，再安装 OpenCL APK 验证 GPU 路径。

## 7. CPU 后端构建

CPU 是默认后端：

```bash
LLAMA_ANDROID_BACKEND=cpu ./gradlew :app:assembleDebug --stacktrace
```

Release APK：

```bash
LLAMA_ANDROID_BACKEND=cpu ./gradlew :app:assembleRelease --stacktrace
```

输出路径：

- Debug: `app/build/outputs/apk/debug/app-debug.apk`
- Release: `app/build/outputs/apk/release/app-release.apk`

## 8. OpenCL 后端准备

OpenCL 后端需要 CMake 找到 OpenCL headers 和 `libOpenCL.so`。Android NDK 默认不提供 OpenCL headers/library，所以需要额外准备。

已知状态见 `docs/backend/OPENCL.md`：

- 已验证：Adreno 750、Adreno 830、Adreno X85。
- 推荐模型量化：`Q4_0`，并在 `llama-quantize` 时加 `--pure`。
- Known issue：Adreno 6xx 目前不工作。

当前分支已经可以验证 Android Gradle/CMake 的 OpenCL 依赖接线和 APK packaging。完整 iFairy OpenCL kernel 行为需要后续合并 `origin/OpenCL` 或等价 OpenCL 实现后再测；依赖目录和构建命令保持一致。

准备一个外部 OpenCL 依赖目录。不要把这个目录提交到仓库：

```text
opencl-deps/
├── include/
│   └── CL/
│       └── cl.h
└── jniLibs/
    └── arm64-v8a/
        └── libOpenCL.so
```

下面示例把依赖放到 Android 示例目录下的 `opencl-deps/`。也可以放到任意仓库外路径，只要构建时设置 `LLAMA_ANDROID_OPENCL_ROOT`。

```bash
export ANDROID_NDK_ROOT="$ANDROID_SDK_ROOT/ndk/30.0.14904198"
cd /path/to/llama.cpp/examples/llama.android
export LLAMA_ANDROID_OPENCL_ROOT="$PWD/opencl-deps"
mkdir -p "$LLAMA_ANDROID_OPENCL_ROOT/include" \
  "$LLAMA_ANDROID_OPENCL_ROOT/jniLibs/arm64-v8a"
```

安装 OpenCL headers 到外部依赖目录：

```bash
git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers
cp -r OpenCL-Headers/CL \
  "$LLAMA_ANDROID_OPENCL_ROOT/include/"
```

交叉编译 OpenCL ICD loader：

```bash
git clone --depth 1 https://github.com/KhronosGroup/OpenCL-ICD-Loader

cmake -S OpenCL-ICD-Loader -B OpenCL-ICD-Loader/build-android -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DOPENCL_ICD_LOADER_HEADERS_DIR="$LLAMA_ANDROID_OPENCL_ROOT/include" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DANDROID_STL=c++_shared

cmake --build OpenCL-ICD-Loader/build-android
```

复制 Android arm64 版 `libOpenCL.so`，并检查目录布局：

```bash
cp OpenCL-ICD-Loader/build-android/libOpenCL.so \
  "$LLAMA_ANDROID_OPENCL_ROOT/jniLibs/arm64-v8a/libOpenCL.so"

test -f "$LLAMA_ANDROID_OPENCL_ROOT/include/CL/cl.h"
test -f "$LLAMA_ANDROID_OPENCL_ROOT/jniLibs/arm64-v8a/libOpenCL.so"
```

`LLAMA_ANDROID_OPENCL_ROOT` 会自动推导：

- CMake include dir: `$LLAMA_ANDROID_OPENCL_ROOT/include`
- CMake library: `$LLAMA_ANDROID_OPENCL_ROOT/jniLibs/arm64-v8a/libOpenCL.so`
- APK packaged JNI root: `$LLAMA_ANDROID_OPENCL_ROOT/jniLibs`

如果测试同事已经提供了 `opencl-deps/` 目录，不需要重新 clone/编译 Khronos 仓库，直接设置 `LLAMA_ANDROID_OPENCL_ROOT=/path/to/opencl-deps` 即可。

## 9. OpenCL 后端构建

回到 Android 示例目录：

```bash
cd /path/to/llama.cpp/examples/llama.android
```

构建 Adreno 优化版本：

```bash
LLAMA_ANDROID_BACKEND=opencl \
LLAMA_ANDROID_OPENCL_ROOT=/path/to/opencl-deps \
LLAMA_ANDROID_OPENCL_ADRENO=1 \
LLAMA_ANDROID_OPENCL_EMBED_KERNELS=1 \
LLAMA_ANDROID_OPENCL_TARGET_VERSION=300 \
  ./gradlew :app:assembleDebug --stacktrace
```

非 Adreno GPU 或想排查 Adreno kernel 问题时：

```bash
LLAMA_ANDROID_BACKEND=opencl \
LLAMA_ANDROID_OPENCL_ROOT=/path/to/opencl-deps \
LLAMA_ANDROID_OPENCL_ADRENO=0 \
  ./gradlew :app:assembleDebug --stacktrace
```

如果依赖目录不是标准布局，可以显式指定三段路径：

```bash
LLAMA_ANDROID_BACKEND=opencl \
LLAMA_ANDROID_OPENCL_INCLUDE_DIR=/path/to/include \
LLAMA_ANDROID_OPENCL_LIBRARY=/path/to/libOpenCL.so \
LLAMA_ANDROID_OPENCL_JNILIBS=/path/to/jniLibs \
  ./gradlew :app:assembleDebug --stacktrace
```

`LLAMA_ANDROID_OPENCL_JNILIBS` 和 `LLAMA_ANDROID_EXTRA_JNILIBS` 都会把 native dependency 加进 APK；OpenCL 模式下至少要有一个目录包含 `arm64-v8a/libOpenCL.so`。Gradle 会在配置阶段检查 `include/CL/cl.h` 和 `arm64-v8a/libOpenCL.so`，缺失时直接报错。

检查 APK 是否包含 native 库：

```bash
unzip -l app/build/outputs/apk/debug/app-debug.apk | grep -E 'libllama-android|libOpenCL'
```

如果从 CPU 切到 OpenCL 后 CMake 没有重新配置，先清理 Android native build 缓存：

```bash
./gradlew :app:clean
rm -rf llama/.cxx app/.cxx
```

给测试同事的交付清单：

- 当前源码 commit 或分支名。
- `opencl-deps/` 目录，且包含 `include/CL/cl.h` 和 `jniLibs/arm64-v8a/libOpenCL.so`。
- 构建命令：`LLAMA_ANDROID_BACKEND=opencl LLAMA_ANDROID_OPENCL_ROOT=/path/to/opencl-deps ./gradlew :app:assembleDebug --stacktrace`。
- APK 内容检查：确认 `libllama-android.so` 和 `libOpenCL.so` 都在 `app-debug.apk` 中。
- 设备 smoke：安装 APK 后用 `adb logcat -s LLAMA_ANDROID` 观察 OpenCL 初始化日志；若失败，先用 CPU APK 确认 app 和模型路径正常。

## 10. 打包内置模型

默认 assets root 是 `/tmp/llama-android-assets`，也可以通过 `LLAMA_ANDROID_BUNDLED_ASSETS` 指定。目录结构：

```text
/path/to/assets/
└── models/
    ├── ifairy.gguf
    ├── bitnet_b1_58_700m.gguf
    └── llama_700m.gguf
```

构建带模型的 APK：

```bash
LLAMA_ANDROID_BACKEND=cpu \
LLAMA_ANDROID_BUNDLED_ASSETS=/path/to/assets \
  ./gradlew :app:assembleRelease --stacktrace
```

当前 app 会自动识别并安装这些内置模型名：`ifairy.gguf`、`bitnet_b1_58_700m.gguf`、`llama_700m.gguf`。其他 `models/*.gguf` 可以被打进 APK，但若要自动安装或参与内置 benchmark，需要同步扩展 app 中的 bundled model 列表。

不打包模型时，可以在 app 里手动导入 `.gguf`：

1. 打开 app 的 Settings。
2. 点击 `Import`，通过 Android 文件选择器选中本地 `.gguf`。
3. app 会把模型复制到私有目录 `filesDir/models/`，不会直接从临时 picker URI 推理。
4. 在 Settings 的 internal models 列表中选择模型，点击 `Load`。

## 11. 安装与 smoke test

确认设备：

```bash
adb devices -l
```

安装 APK：

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

启动 app：

```bash
adb shell am start -n com.example.llama/.MainActivity
```

查看日志：

```bash
adb logcat -s LLAMA_ANDROID
```

触发自动 smoke：

```bash
adb shell am start \
  -n com.example.llama/.MainActivity \
  -e codex_action smoke
```

CPU smoke 重点看模型加载、context 创建、生成是否完成。OpenCL smoke 额外看 logcat 中是否出现 `ggml_opencl` 设备选择和 kernel 初始化日志；如果 OpenCL 初始化失败，先用 `LLAMA_ANDROID_BACKEND=cpu` 构建确认 app 和模型路径本身没有问题。

## 12. 自动 benchmark

短 E2E benchmark 示例：

```bash
adb shell am start \
  -n com.example.llama/.MainActivity \
  -e codex_action builtin_e2e_bench \
  -e codex_model_filter ifairy.gguf \
  -e codex_bench_preset_filter short_prompt_short_decode \
  --ei codex_bench_repetitions 1 \
  --ei codex_n_threads 6 \
  --ei codex_n_threads_batch 6 \
  --ei codex_affinity_profile 0
```

结果目录：

```bash
adb shell ls -lh /sdcard/Android/data/com.example.llama/files/bench
adb shell cat /sdcard/Android/data/com.example.llama/files/bench/builtin_models_e2e_bench.status
adb pull /sdcard/Android/data/com.example.llama/files/bench ./android-bench-results
```

## 13. 预编译 JNI 库模式

默认构建会包含 `:llama` module，并从源码编译 JNI 库。若已经有预编译 native 库，可用：

```bash
LLAMA_ANDROID_USE_PREBUILT_LLAMA=true \
LLAMA_ANDROID_PREBUILT_JNILIBS=/path/to/jniLibs \
  ./gradlew :app:assembleRelease --stacktrace
```

`/path/to/jniLibs` 应包含 ABI 子目录，例如：

```text
/path/to/jniLibs/
└── arm64-v8a/
    └── libllama-android.so
```

如果预编译库依赖 `libOpenCL.so`，同时传：

```bash
LLAMA_ANDROID_OPENCL_JNILIBS=/path/to/opencl-jniLibs
```

## Troubleshooting

- JDK 缺失或版本错误：安装 `openjdk-17-jdk`，确认 `java -version` 是 17；不要用系统 `gradle`。
- `sdkmanager` 找不到：确认 `cmdline-tools` 目录布局是 `$ANDROID_SDK_ROOT/cmdline-tools/latest/bin/sdkmanager`。
- NDK 找不到：安装时加 `--channel=1`，并确认 `ndk/30.0.14904198/source.properties` 中 revision 是 `30.0.14904198-beta1`。
- licenses 未接受：运行 `yes | sdkmanager --licenses`。
- Gradle 仍用旧 SDK：重新写 `local.properties`，内容只保留 `sdk.dir=/your/Android/Sdk`。
- OpenCL root 找不到：确认 `LLAMA_ANDROID_OPENCL_ROOT` 指向包含 `include/CL/cl.h` 和 `jniLibs/arm64-v8a/libOpenCL.so` 的目录。
- OpenCL headers 找不到：确认 `LLAMA_ANDROID_OPENCL_INCLUDE_DIR` 指向包含 `CL/cl.h` 的目录，或使用标准 `LLAMA_ANDROID_OPENCL_ROOT` 布局。
- OpenCL library 找不到：确认 `LLAMA_ANDROID_OPENCL_LIBRARY` 指向 Android arm64 版 `libOpenCL.so`，不是 x86_64 host 版。
- OpenCL 被 CMake 强制关闭：确认 OpenCL 构建命令使用 `LLAMA_ANDROID_BACKEND=opencl`，该模式会设置 `GGML_IFAIRY_LUT_CPU=OFF`。
- APK 中没有 `libOpenCL.so`：确认 `LLAMA_ANDROID_OPENCL_ROOT/jniLibs`、`LLAMA_ANDROID_OPENCL_JNILIBS` 或 `LLAMA_ANDROID_EXTRA_JNILIBS` 指向包含 `arm64-v8a/libOpenCL.so` 的目录。
- OpenCL runtime 找不到设备：确认目标 Android 设备支持 OpenCL 2.0+ 和 FP16；Adreno 6xx 目前是 known issue。
- Adreno kernel 报错：用 `LLAMA_ANDROID_OPENCL_ADRENO=0` 重新构建排查。
- APK 太大：减少 bundled models，或用 app Settings 手动导入 `.gguf`。
- 切换后端后结果不变：执行 `./gradlew :app:clean` 并删除 `llama/.cxx app/.cxx` 后重新构建。

## 维护者验收清单

CPU 路径：

```bash
cd /path/to/llama.cpp/examples/llama.android
LLAMA_ANDROID_BACKEND=cpu ./gradlew :app:assembleDebug --stacktrace
git diff --check
```

OpenCL 路径：

```bash
LLAMA_ANDROID_BACKEND=opencl \
LLAMA_ANDROID_OPENCL_ROOT=/path/to/opencl-deps \
  ./gradlew :app:assembleDebug --stacktrace

unzip -l app/build/outputs/apk/debug/app-debug.apk | grep -E 'libllama-android|libOpenCL'
```

设备验证：

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.example.llama/.MainActivity -e codex_action smoke
adb logcat -s LLAMA_ANDROID
```
