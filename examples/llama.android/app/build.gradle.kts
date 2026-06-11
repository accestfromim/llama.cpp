import java.io.File

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

fun envValue(name: String): String? = System.getenv(name)?.trim()?.takeIf { it.isNotEmpty() }

fun envFlag(name: String, default: Boolean): Boolean {
    return when (System.getenv(name)?.trim()?.lowercase()) {
        null, "" -> default
        "1", "true", "yes", "on" -> true
        "0", "false", "no", "off" -> false
        else -> throw IllegalArgumentException("$name must be 1/0, true/false, yes/no, or on/off")
    }
}

val bundledAssetDir = envValue("LLAMA_ANDROID_BUNDLED_ASSETS") ?: "/tmp/llama-android-assets"
val usePrebuiltLlama = envValue("LLAMA_ANDROID_USE_PREBUILT_LLAMA") == "true"
val prebuiltLlamaJniLibs = envValue("LLAMA_ANDROID_PREBUILT_JNILIBS")
val extraJniLibs = envValue("LLAMA_ANDROID_EXTRA_JNILIBS")
val llamaAndroidBackend = envValue("LLAMA_ANDROID_BACKEND")?.lowercase() ?: "cpu"
val useSystemOpenCl = envFlag("LLAMA_ANDROID_OPENCL_SYSTEM_LIBRARY", false)
val openclRoot = envValue("LLAMA_ANDROID_OPENCL_ROOT")
val openclJniLibs = envValue("LLAMA_ANDROID_OPENCL_JNILIBS")
    ?: openclRoot?.let { File(it, "jniLibs").path }
val openclPackagingJniLibs = if (llamaAndroidBackend == "opencl" && !useSystemOpenCl) openclJniLibs else null
val extraJniLibDirs = listOf(openclPackagingJniLibs, extraJniLibs).filterNotNull().distinct()

if (llamaAndroidBackend == "opencl" && !useSystemOpenCl) {
    require(extraJniLibDirs.isNotEmpty()) {
        "LLAMA_ANDROID_BACKEND=opencl requires OpenCL dependencies. Set LLAMA_ANDROID_OPENCL_ROOT=/path/to/opencl-deps, or set LLAMA_ANDROID_OPENCL_INCLUDE_DIR, LLAMA_ANDROID_OPENCL_LIBRARY, and LLAMA_ANDROID_OPENCL_JNILIBS=/path/to/jniLibs."
    }
    val hasOpenClLibrary = extraJniLibDirs.any {
        File(it, "arm64-v8a/libOpenCL.so").isFile
    }
    require(hasOpenClLibrary) {
        "LLAMA_ANDROID_BACKEND=opencl expected arm64-v8a/libOpenCL.so under one of: ${extraJniLibDirs.joinToString()}. Use an Android arm64 libOpenCL.so, not a host x86_64 library."
    }
}

android {
    namespace = "com.example.llama"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.llama"
        minSdk = 31
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            isDebuggable = false
            signingConfig = signingConfigs.getByName("debug")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    buildFeatures {
        compose = true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.1"
    }
    androidResources {
        noCompress += listOf("gguf")
    }
    packaging {
        jniLibs {
            useLegacyPackaging = true
            if (useSystemOpenCl) {
                excludes += "**/libOpenCL.so"
            }
        }
    }
    sourceSets.getByName("main").assets.srcDirs(
        "src/main/assets",
        bundledAssetDir,
    )
    extraJniLibDirs.forEach {
        sourceSets.getByName("main").jniLibs.srcDir(it)
    }
    if (usePrebuiltLlama) {
        sourceSets.getByName("main").java.srcDir("../llama/src/main/java")
        sourceSets.getByName("main").jniLibs.srcDir(
            prebuiltLlamaJniLibs ?: "${projectDir}/build/intermediates/merged_native_libs/release/out/lib",
        )
    }
}

dependencies {

    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.2")
    implementation("androidx.activity:activity-compose:1.8.2")
    implementation(platform("androidx.compose:compose-bom:2023.08.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    if (!usePrebuiltLlama) {
        implementation(project(":llama"))
    }
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
    androidTestImplementation(platform("androidx.compose:compose-bom:2023.08.00"))
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}
