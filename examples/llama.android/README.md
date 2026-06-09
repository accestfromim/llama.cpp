# llama.android 构建与验证指南

这份文档面向 Linux/CLI 环境，用来把 `examples/llama.android` 构建成可安装的 Android APK，并可选打包 GGUF 模型、安装到设备、跑 smoke test 或内置 benchmark。Android Studio 也可以使用，但这里以命令行为主，方便把步骤交给其他人复现。

不要提交这些本机产物：`local.properties`、APK、模型权重、Gradle/Android SDK 缓存、benchmark 输出。

## 环境基线

| 项目 | 本工程要求 / 本机实测 |
| --- | --- |
| Gradle wrapper | `8.2`，由 `gradle/wrapper/gradle-wrapper.properties` 指定 |
| Android Gradle Plugin | `8.2.0` |
| Kotlin Android plugin | `1.9.0` |
| compile SDK / target SDK | `34` |
| min SDK | `31` |
| ABI | `arm64-v8a` |
| Android NDK | `30.0.14904198-beta1`，目录通常是 `ndk/30.0.14904198` |
| Android SDK CMake | `3.22.1` |
| 推荐 JDK | JDK 17；JDK 21 可作为本机验证备选，默认 Java 26 不作为推荐基线 |

本机检查到两个 SDK root：

- `/home/zybi/Android/Sdk`：包含 `platforms/android-34`、`build-tools/34.0.0`、`cmake/3.22.1`、`ndk/30.0.14904198`，适合作为 Gradle 的 `sdk.dir`。
- `/opt/android-sdk`：包含 cmdline-tools、platform-tools、build-tools 37 和 preview platform，但没有 NDK。可以提供 `sdkmanager`/`adb`，但不要直接作为本工程的 `sdk.dir`，除非先补齐 NDK 和 SDK 34。

给别人使用时，推荐统一一个 SDK root，下面用 `/path/to/Android/Sdk` 表示。

## 准备 SDK / NDK / JDK

安装或选择一个 JDK 17：

```bash
export JAVA_HOME=/path/to/jdk17
export PATH="$JAVA_HOME/bin:$PATH"
java -version
```

准备 Android SDK。若使用 Android Studio，可以在 SDK Manager 中安装同样的组件；若使用命令行工具：

```bash
export ANDROID_SDK_ROOT=/path/to/Android/Sdk
export ANDROID_HOME="$ANDROID_SDK_ROOT"
export PATH="$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/platform-tools:$PATH"

sdkmanager --install \
  "platform-tools" \
  "platforms;android-34" \
  "build-tools;34.0.0" \
  "cmake;3.22.1" \
  "ndk;30.0.14904198"

sdkmanager --licenses
```

检查关键文件：

```bash
test -f "$ANDROID_SDK_ROOT/platforms/android-34/source.properties"
test -f "$ANDROID_SDK_ROOT/cmake/3.22.1/bin/cmake"
test -f "$ANDROID_SDK_ROOT/ndk/30.0.14904198/source.properties"
grep -n "Pkg.Revision" "$ANDROID_SDK_ROOT/ndk/30.0.14904198/source.properties"
```

`source.properties` 中的 NDK revision 应为 `30.0.14904198-beta1`。如果 `sdkmanager` 看不到这个 NDK 包，优先用 Android Studio SDK Manager 安装，或从已有机器复制同版本 NDK 目录。

## 标准构建

所有 Gradle 命令都在 Android 示例目录执行：

```bash
cd /path/to/llama.cpp/examples/llama.android
```

创建本机 SDK 配置。`local.properties` 不应提交：

```bash
printf 'sdk.dir=%s\n' "$ANDROID_SDK_ROOT" > local.properties
```

Debug 构建：

```bash
./gradlew :app:assembleDebug
```

Release 构建：

```bash
./gradlew :app:assembleRelease
```

常见输出：

- Debug APK：`app/build/outputs/apk/debug/app-debug.apk`
- Release APK：`app/build/outputs/apk/release/app-release.apk`

如果需要更多诊断：

```bash
./gradlew :app:assembleDebug --stacktrace
```

## 打包内置模型 assets

默认 assets root 是 `/tmp/llama-android-assets`，也可以通过 `LLAMA_ANDROID_BUNDLED_ASSETS` 指定。目录结构必须是：

```text
/path/to/assets/
└── models/
    ├── ifairy.gguf
    ├── bitnet_b1_58_700m.gguf
    └── llama_700m.gguf
```

构建带模型的 APK：

```bash
LLAMA_ANDROID_BUNDLED_ASSETS=/path/to/assets \
  ./gradlew :app:assembleRelease
```

当前 app 会自动识别并安装这些内置模型名：`ifairy.gguf`、`bitnet_b1_58_700m.gguf`、`llama_700m.gguf`。其他 `models/*.gguf` 可以被打进 APK，但若要自动安装或参与内置 benchmark，需要同步扩展 app 中的 bundled model 列表。

`app/build.gradle.kts` 已设置 `noCompress += listOf("gguf")`，GGUF 打包时不会被压缩。模型很大时 APK 会快速膨胀；如果只是手动体验，通常用 UI 导入模型更轻便。

## 预编译 JNI 库模式

默认构建会包含 `:llama` module，并从源码编译 JNI 库。若已经有预编译 native 库，可用：

```bash
LLAMA_ANDROID_USE_PREBUILT_LLAMA=true \
LLAMA_ANDROID_PREBUILT_JNILIBS=/path/to/jniLibs \
  ./gradlew :app:assembleRelease
```

`/path/to/jniLibs` 应包含 ABI 子目录，例如：

```text
/path/to/jniLibs/
└── arm64-v8a/
    └── libllama-android.so
```

没有明确需要时，使用默认源码构建路径更简单。

## 安装与运行

确认设备：

```bash
adb devices -l
```

安装 APK：

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
# 或
adb install -r app/build/outputs/apk/release/app-release.apk
```

启动 app：

```bash
adb shell am start -n com.example.llama/.MainActivity
```

查看日志：

```bash
adb logcat -s LLAMA_ANDROID
```

使用模型有两种方式：

- 带内置模型构建：app 启动后会从 assets 安装可识别的 bundled models。
- 不带内置模型构建：进入 app 的 Settings，选择本地 `.gguf` 文件导入。

## 自动化 smoke test 和 benchmark

`MainActivity` 支持通过 intent extras 触发自动化动作：

| key | 类型 | 说明 |
| --- | --- | --- |
| `codex_action` | string | `smoke`、`bench`、`builtin_bench`、`builtin_e2e_bench`、`builtin_power_bench` |
| `codex_model_filter` | string | 逗号分隔模型文件名，例如 `ifairy.gguf,llama_700m.gguf` |
| `codex_bench_preset_filter` | string | 逗号分隔 preset，例如 `short_prompt_short_decode,long_prompt_long_decode` |
| `codex_bench_repetitions` | int | benchmark 重复次数 |
| `codex_bench_cooldown_ms` | int | bundled benchmark 模型间冷却时间 |
| `codex_n_ctx` / `codex_n_batch` / `codex_n_ubatch` | int | llama runtime context/batch override |
| `codex_n_threads` / `codex_n_threads_batch` | int | generation / batch 线程数 |
| `codex_affinity_profile` | int | 线程 affinity profile override |
| `codex_disable_ifairy_lut` | int | `0` 启用、`1` 禁用、缺省使用 profile 默认 |
| `codex_enable_ifairy_vecdot_act_tensor` | int | `0`/`1` override，缺省使用 profile 默认 |
| `codex_generation_priority` / `codex_batch_priority` | int | 线程优先级 override，缺省 sentinel 是 `-99` |
| `codex_power_duration_ms` / `codex_power_duration_minutes` | long | power benchmark 时长 |
| `codex_battery_capacity_mah` | float | 电池容量估计值，用于 power benchmark 估算 |

Smoke test：

```bash
adb shell am start \
  -n com.example.llama/.MainActivity \
  -e codex_action smoke
```

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

常见输出文件：

- `builtin_models_bench.csv`
- `builtin_models_bench.status`
- `builtin_models_e2e_bench.csv`
- `builtin_models_e2e_bench.status`
- `builtin_models_power_bench.csv`
- `builtin_models_power_bench.status`

## Troubleshooting

- NDK 版本不匹配：确认 `local.properties` 指向的 SDK root 下存在 `ndk/30.0.14904198/source.properties`，且 revision 是 `30.0.14904198-beta1`。
- SDK root 混用：`sdk.dir`、`ANDROID_SDK_ROOT`、`adb`、`sdkmanager` 最好来自同一个 SDK。若 `adb` 来自 `/opt/android-sdk` 而 Gradle 使用 `/home/zybi/Android/Sdk`，通常可行，但排查问题时先统一路径。
- licenses 未接受：运行 `sdkmanager --licenses`，或在 Android Studio SDK Manager 中接受。
- JDK 问题：优先用 JDK 17 跑 `./gradlew`，不要依赖系统 `gradle`。若出现 Gradle native cache 异常，可先用干净缓存验证：`GRADLE_USER_HOME=/tmp/llama-android-gradle ./gradlew :app:assembleDebug`。
- `adb devices` 没有设备：检查 USB 调试授权、线缆、udev/权限；必要时执行 `adb kill-server` 后重新 `adb devices -l`。
- APK 太大或安装失败：减少 bundled models，改用 UI 导入模型，或只保留一个小模型做 smoke test。
- 运行时内存不足：降低模型大小、上下文长度和生成 token 数；优先在 app 中先跑短 prompt。
- benchmark 没有目标模型：确认 bundled model 文件名在当前 app 的 bundled model 列表中，或用 `codex_model_filter` 指向已经导入到 app 私有目录的模型文件名。

## 维护者验收清单

更新这份文档或 Android 构建流程后，建议至少验证：

```bash
cd /path/to/llama.cpp/examples/llama.android
./gradlew :app:assembleDebug
git diff --check
```

可选验证：

```bash
LLAMA_ANDROID_BUNDLED_ASSETS=/tmp/llama-android-assets \
  ./gradlew :app:assembleRelease

adb devices -l
adb install -r app/build/outputs/apk/release/app-release.apk
adb shell am start -n com.example.llama/.MainActivity -e codex_action smoke
adb logcat -s LLAMA_ANDROID
```
