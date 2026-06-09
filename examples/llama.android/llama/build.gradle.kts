plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

fun envFlag(name: String, default: Boolean): Boolean {
    return when (System.getenv(name)?.trim()?.lowercase()) {
        null, "" -> default
        "1", "true", "yes", "on" -> true
        "0", "false", "no", "off" -> false
        else -> throw IllegalArgumentException("$name must be 1/0, true/false, yes/no, or on/off")
    }
}

fun envValue(name: String): String? = System.getenv(name)?.trim()?.takeIf { it.isNotEmpty() }

val llamaAndroidBackend = envValue("LLAMA_ANDROID_BACKEND")?.lowercase() ?: "cpu"
require(llamaAndroidBackend in setOf("cpu", "opencl")) {
    "LLAMA_ANDROID_BACKEND must be one of: cpu, opencl"
}

android {
    namespace = "android.llama.cpp"
    compileSdk = 34
    ndkVersion = "30.0.14904198-beta1"

    defaultConfig {
        minSdk = 31

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
        externalNativeBuild {
            cmake {
                arguments += "-DLLAMA_CURL=OFF"
                arguments += "-DLLAMA_BUILD_COMMON=ON"
                arguments += "-DGGML_LLAMAFILE=OFF"
                arguments += "-DGGML_NATIVE=OFF"
                arguments += "-DGGML_OPENMP=OFF"

                when (llamaAndroidBackend) {
                    "cpu" -> {
                        arguments += "-DGGML_IFAIRY_LUT_CPU=ON"
                        arguments += "-DGGML_OPENCL=OFF"
                    }
                    "opencl" -> {
                        arguments += "-DGGML_IFAIRY_LUT_CPU=OFF"
                        arguments += "-DGGML_OPENCL=ON"
                        arguments += "-DGGML_OPENCL_USE_ADRENO_KERNELS=${if (envFlag("LLAMA_ANDROID_OPENCL_ADRENO", true)) "ON" else "OFF"}"
                        arguments += "-DGGML_OPENCL_EMBED_KERNELS=${if (envFlag("LLAMA_ANDROID_OPENCL_EMBED_KERNELS", true)) "ON" else "OFF"}"
                        arguments += "-DGGML_OPENCL_TARGET_VERSION=${envValue("LLAMA_ANDROID_OPENCL_TARGET_VERSION") ?: "300"}"
                        envValue("LLAMA_ANDROID_OPENCL_INCLUDE_DIR")?.let {
                            arguments += "-DOpenCL_INCLUDE_DIR=$it"
                        }
                        envValue("LLAMA_ANDROID_OPENCL_LIBRARY")?.let {
                            arguments += "-DOpenCL_LIBRARY=$it"
                        }
                    }
                }
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {

    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}
