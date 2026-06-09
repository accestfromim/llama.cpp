pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "LlamaAndroid"
include(":app")
if (System.getenv("LLAMA_ANDROID_USE_PREBUILT_LLAMA") != "true") {
    include(":llama")
}
