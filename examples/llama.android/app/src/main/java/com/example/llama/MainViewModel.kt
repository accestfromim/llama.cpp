package com.example.llama

import android.annotation.SuppressLint
import android.content.ContentResolver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.database.Cursor
import android.llama.cpp.LLamaAndroid
import android.net.Uri
import android.os.BatteryManager
import android.os.Bundle
import android.os.PowerManager
import android.os.SystemClock
import android.provider.OpenableColumns
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale

data class ImportedModel(
    val fileName: String,
    val sizeBytes: Long,
    val privatePath: String,
)

data class ChatMessage(
    val id: Long,
    val role: ChatRole,
    val text: String,
)

enum class ChatRole {
    USER,
    ASSISTANT,
}

enum class ModelLoadState(val label: String) {
    NOT_IMPORTED("未导入"),
    IMPORTING("导入中"),
    IMPORTED("未加载"),
    LOADING("加载中"),
    LOADED("已加载"),
    FAILED("加载失败"),
}

enum class GenerationLengthPreset(val label: String, val maxTokens: Int) {
    SHORT("32", 32),
    BALANCED("64", 64),
    LONG("128", 128),
}

private data class BundledModelSpec(
    val displayName: String,
    val fileName: String,
)

private data class BenchPreset(
    val label: String,
    val promptTokens: Int,
    val genTokens: Int,
    val pl: Int = 1,
    val repetitions: Int = 3,
)

private data class BenchSummary(
    val modelFileName: String,
    val presetLabel: String,
    val promptTokens: Int,
    val genTokens: Int,
    val repetitions: Int,
    val ppTps: Double,
    val tgTps: Double,
    val ppStd: Double,
    val tgStd: Double,
    val promptLatencyMs: Double,
    val decodeLatencyMsPerToken: Double,
    val decodeTotalMs: Double,
)

private data class E2eBenchSummary(
    val modelFileName: String,
    val presetLabel: String,
    val promptTokens: Int,
    val genTokens: Int,
    val repetitions: Int,
    val nThreads: Int,
    val nThreadsBatch: Int,
    val affinityProfile: Int,
    val disableIFairyLut: Int,
    val enableIFairyVecdotActTensor: Int,
    val generatedTokensAvg: Double,
    val prefillMs: Double,
    val firstTokenMs: Double,
    val decodeLatencyMsPerToken: Double,
    val tokS: Double,
    val totalMs: Double,
)

private data class BatterySnapshot(
    val percent: Double?,
    val level: Int?,
    val scale: Int?,
    val voltageMv: Int?,
    val temperatureC: Double?,
    val status: Int?,
    val plugged: Int?,
    val chargeCounterUah: Long?,
    val remainingCapacityMah: Double?,
    val energyCounterNwh: Long?,
)

private data class PowerBenchSummary(
    val modelFileName: String,
    val presetLabel: String,
    val promptTokens: Int,
    val genTokens: Int,
    val repetitions: Int,
    val targetDurationMs: Long,
    val elapsedMs: Long,
    val loops: Int,
    val refreshCount: Int,
    val refreshTotalMs: Long,
    val generatedTokensTotal: Double,
    val startBattery: BatterySnapshot,
    val endBattery: BatterySnapshot,
    val batteryDeltaPercent: Double?,
    val batteryDeltaPercentPerHour: Double?,
    val chargeDeltaUah: Long?,
    val startRemainingCapacityMah: Double?,
    val endRemainingCapacityMah: Double?,
    val remainingCapacityDeltaMah: Double?,
    val energyDeltaWh: Double?,
    val estimatedEnergyWh: Double?,
    val estimatedAveragePowerW: Double?,
    val avgPrefillMs: Double,
    val avgFirstTokenMs: Double,
    val avgDecodeMsPerToken: Double,
    val avgTokS: Double,
    val avgTotalMs: Double,
    val note: String,
)

private data class RuntimeBenchConfig(
    val nCtx: Int = -1,
    val nBatch: Int = -1,
    val nUbatch: Int = -1,
    val nThreads: Int = -1,
    val nThreadsBatch: Int = -1,
    val affinityProfile: Int = -1,
    val disableIFairyLut: Int = -1,
    val enableIFairyVecdotActTensor: Int = -1,
    val generationPriority: Int = -99,
    val batchPriority: Int = -99,
    val schedDebug: Int = -1,
    val openclSupportsDebug: Int = -1,
    val forceCpu: Int = -1,
)

class MainViewModel(
    private val llamaAndroid: LLamaAndroid = LLamaAndroid.instance(),
) : ViewModel() {
    companion object {
        private const val LOG_TAG = "LLAMA_ANDROID"
        private const val BUNDLED_MODEL_ASSET_DIR = "models"
        private const val DEFAULT_BUNDLED_MODEL_COOLDOWN_MS = 60_000L
        private const val BENCH_OUTPUT_DIR = "bench"
        private const val BENCH_OUTPUT_FILE_NAME = "builtin_models_bench.csv"
        private const val BENCH_STATUS_FILE_NAME = "builtin_models_bench.status"
        private const val E2E_BENCH_OUTPUT_FILE_NAME = "builtin_models_e2e_bench.csv"
        private const val E2E_BENCH_STATUS_FILE_NAME = "builtin_models_e2e_bench.status"
        private const val E2E_LONG_DECODE_PRESET_LABEL = "short_prompt_long_decode_64_1024"
        private const val E2E_LONG_DECODE_OUTPUT_FILE_NAME = "builtin_models_e2e_long_decode_64p_1024g.csv"
        private const val E2E_LONG_DECODE_STATUS_FILE_NAME = "builtin_models_e2e_long_decode_64p_1024g.status"
        private const val POWER_BENCH_OUTPUT_FILE_NAME = "builtin_models_power_bench.csv"
        private const val POWER_BENCH_STATUS_FILE_NAME = "builtin_models_power_bench.status"
        private const val GENERATE_STATUS_FILE_SUFFIX = ".status"
        private const val GENERATE_OUTPUT_FILE_SUFFIX = ".txt"
        private const val DEFAULT_POWER_BENCH_DURATION_MS = 30L * 60L * 1000L
        private const val DEFAULT_POWER_BENCH_THREADS = 6
        private const val DEFAULT_POWER_BENCH_AFFINITY_PROFILE = 0
        private val BUNDLED_MODELS = listOf(
            BundledModelSpec("iFairy 700M", "ifairy.gguf"),
            BundledModelSpec("BitNet b1.58 700M", "bitnet_b1_58_700m.gguf"),
            BundledModelSpec("Llama 700M", "llama_700m.gguf"),
        )
        private val BENCH_PRESETS = listOf(
            BenchPreset(
                label = "short_prompt_short_decode",
                promptTokens = 64,
                genTokens = 32,
                repetitions = 3,
            ),
            BenchPreset(
                label = "long_prompt_long_decode",
                promptTokens = 512,
                genTokens = 128,
                repetitions = 3,
            ),
        )
        private val E2E_EXTRA_PRESETS = listOf(
            BenchPreset(
                label = E2E_LONG_DECODE_PRESET_LABEL,
                promptTokens = 64,
                genTokens = 1024,
                repetitions = 1,
            ),
        )
        private val SAFE_FILE_NAME = Regex("[^A-Za-z0-9._-]")
    }

    var diagnostics by mutableStateOf(listOf<String>())
        private set

    var messages by mutableStateOf(listOf<ChatMessage>())
        private set

    var prompt by mutableStateOf("")
        private set

    var importedModel by mutableStateOf<ImportedModel?>(null)
        private set

    var modelLoadState by mutableStateOf(ModelLoadState.NOT_IMPORTED)
        private set

    var modelError by mutableStateOf<String?>(null)
        private set

    var isGenerating by mutableStateOf(false)
        private set

    var isBenchmarking by mutableStateOf(false)
        private set

    var generationLength by mutableStateOf(GenerationLengthPreset.BALANCED)
        private set

    var useCustomGenerationLength by mutableStateOf(false)
        private set

    var customGenerationLengthInput by mutableStateOf("96")
        private set

    var availableModels by mutableStateOf(listOf<ImportedModel>())
        private set

    private var generationJob: Job? = null
    private var typingJob: Job? = null
    private var pendingAutomationAction: String? = null
    private var initializationStarted = false
    private var nextMessageId = 1L
    private val streamingBuffer = StringBuilder()
    private var appContext: Context? = null
    private var bundledModelCooldownMs: Long = DEFAULT_BUNDLED_MODEL_COOLDOWN_MS
    private var benchmarkModelFilter: Set<String>? = null
    private var benchmarkPresetFilter: Set<String>? = null
    private var benchmarkRepetitionsOverride: Int? = null
    private var benchmarkPresetOverride: BenchPreset? = null
    private var powerBenchDurationMs: Long = DEFAULT_POWER_BENCH_DURATION_MS
    private var batteryCapacityMah: Double? = null
    private var automationGeneratePrompt: String = "Hello"
    private var automationGenerateMaxTokens: Int = 16
    private var automationGenerateFormatChat: Boolean = true
    private var automationGenerateOutputPrefix: String = "adb_generate"
    private var runtimeBenchConfig = RuntimeBenchConfig()
    private val powerBenchOutputFileNames = mutableMapOf<String, String>()

    override fun onCleared() {
        super.onCleared()
        generationJob?.cancel()
        typingJob?.cancel()
        viewModelScope.launch {
            runCatching { llamaAndroid.unload() }
                .onFailure { logError("unload() failed", it) }
        }
    }

    fun initialize(context: Context) {
        if (initializationStarted) {
            return
        }
        initializationStarted = true

        val appContext = context.applicationContext
        this.appContext = appContext
        llamaAndroid.configureNativeLibraryDir(appContext.applicationInfo.nativeLibraryDir)
        refreshAvailableModels(appContext)

        val bundledAssets = bundledAssetsAvailable(appContext)
        if (bundledAssets.isEmpty()) {
            maybeStartPendingAutomation()
            return
        }

        modelLoadState = ModelLoadState.IMPORTING
        modelError = null
        log("Installing bundled models: ${bundledAssets.joinToString { it.fileName }}")

        viewModelScope.launch {
            runCatching {
                installBundledModels(appContext, bundledAssets)
            }.onSuccess {
                refreshAvailableModels(appContext)
                modelLoadState = if (importedModel == null) ModelLoadState.NOT_IMPORTED else ModelLoadState.IMPORTED
                availableModels.forEach { model ->
                    log("Bundled model ready: ${model.fileName}")
                    log("Model path: ${model.privatePath}")
                    log("Model size: ${formatSize(model.sizeBytes)}")
                }
                maybeStartPendingAutomation()
            }.onFailure { throwable ->
                refreshAvailableModels(appContext)
                modelLoadState = if (importedModel == null) ModelLoadState.FAILED else ModelLoadState.IMPORTED
                modelError = throwable.message ?: "Bundled model install failed"
                logError("installBundledModels() failed", throwable)
            }
        }
    }

    fun requestAutomationAction(action: String?) {
        appContext?.let { refreshAvailableModels(it) }
        pendingAutomationAction = action?.trim()?.lowercase(Locale.US)?.takeIf { it.isNotEmpty() }
        if (pendingAutomationAction != null) {
            log("Automation action requested: $pendingAutomationAction")
        }
        maybeStartPendingAutomation()
    }

    fun applyRuntimeOverrides(extras: Bundle?) {
        if (extras == null) {
            return
        }

        val nCtx = extras.getIntOrSentinel("codex_n_ctx")
        val nBatch = extras.getIntOrSentinel("codex_n_batch")
        val nUbatch = extras.getIntOrSentinel("codex_n_ubatch")
        val nThreads = extras.getIntOrSentinel("codex_n_threads")
        val nThreadsBatch = extras.getIntOrSentinel("codex_n_threads_batch")
        val disableIFairyLut = extras.getIntOrSentinel("codex_disable_ifairy_lut")
        val affinityProfile = extras.getIntOrSentinel("codex_affinity_profile")
        val enableIFairyVecdotActTensor = extras.getIntOrSentinel("codex_enable_ifairy_vecdot_act_tensor")
        val generationPriority = extras.getIntOrSentinel("codex_generation_priority", -99)
        val batchPriority = extras.getIntOrSentinel("codex_batch_priority", -99)
        val schedDebug = extras.getIntOrSentinel("codex_sched_debug")
        val openclSupportsDebug = extras.getIntOrSentinel("codex_opencl_supports_debug")
        val forceCpu = extras.getIntOrSentinel("codex_force_cpu")
        val benchCooldownMs = extras.getIntOrSentinel("codex_bench_cooldown_ms")
        val benchRepetitions = extras.getIntOrSentinel("codex_bench_repetitions")
        val benchPromptTokens = extras.getIntOrSentinel("codex_bench_pp")
        val benchGenTokens = extras.getIntOrSentinel("codex_bench_tg")
        val benchPromptLength = extras.getIntOrSentinel("codex_bench_pl")
        val powerDurationMs = extras.getLongOrNull("codex_power_duration_ms")
        val powerDurationMinutes = extras.getLongOrNull("codex_power_duration_minutes")
        val configuredBatteryCapacityMah = extras.getDoubleOrNull("codex_battery_capacity_mah")
        val modelFilter = extras.getString("codex_model_filter")
            ?.split(",")
            ?.map { it.trim() }
            ?.filter { it.isNotEmpty() }
            ?.toSet()
        val presetFilter = extras.getString("codex_bench_preset_filter")
            ?.split(",")
            ?.map { it.trim() }
            ?.filter { it.isNotEmpty() }
            ?.toSet()
        val generatePrompt = extras.getString("codex_prompt")
        val generateMaxTokens = extras.getIntOrSentinel("codex_max_tokens")
        val generateFormatChat = extras.getIntOrSentinel("codex_format_chat")
        val generateOutputPrefix = extras.getString("codex_output_prefix")?.trim()?.takeIf { it.isNotEmpty() }
        val nextRuntimeBenchConfig = runtimeBenchConfig.copy(
            nCtx = if (nCtx >= 0) nCtx else runtimeBenchConfig.nCtx,
            nBatch = if (nBatch >= 0) nBatch else runtimeBenchConfig.nBatch,
            nUbatch = if (nUbatch >= 0) nUbatch else runtimeBenchConfig.nUbatch,
            nThreads = if (nThreads >= 0) nThreads else runtimeBenchConfig.nThreads,
            nThreadsBatch = if (nThreadsBatch >= 0) nThreadsBatch else runtimeBenchConfig.nThreadsBatch,
            affinityProfile = if (affinityProfile >= 0) affinityProfile else runtimeBenchConfig.affinityProfile,
            disableIFairyLut = if (disableIFairyLut >= 0) disableIFairyLut else runtimeBenchConfig.disableIFairyLut,
            enableIFairyVecdotActTensor = if (enableIFairyVecdotActTensor >= 0) {
                enableIFairyVecdotActTensor
            } else {
                runtimeBenchConfig.enableIFairyVecdotActTensor
            },
            generationPriority = if (generationPriority != -99) generationPriority else runtimeBenchConfig.generationPriority,
            batchPriority = if (batchPriority != -99) batchPriority else runtimeBenchConfig.batchPriority,
            schedDebug = if (schedDebug >= 0) schedDebug else runtimeBenchConfig.schedDebug,
            openclSupportsDebug = if (openclSupportsDebug >= 0) openclSupportsDebug else runtimeBenchConfig.openclSupportsDebug,
            forceCpu = if (forceCpu >= 0) forceCpu else runtimeBenchConfig.forceCpu,
        )

        if (
            listOf(
                nCtx,
                nBatch,
                nUbatch,
                nThreads,
                nThreadsBatch,
                disableIFairyLut,
                affinityProfile,
                enableIFairyVecdotActTensor,
                schedDebug,
                openclSupportsDebug,
                forceCpu,
                benchCooldownMs,
                benchRepetitions,
                benchPromptTokens,
                benchGenTokens,
                benchPromptLength,
            ).all { it < 0 } &&
            generationPriority == -99 &&
            batchPriority == -99 &&
            powerDurationMs == null &&
            powerDurationMinutes == null &&
            configuredBatteryCapacityMah == null &&
            modelFilter.isNullOrEmpty() &&
            presetFilter.isNullOrEmpty() &&
            generatePrompt == null &&
            generateMaxTokens < 0 &&
            generateFormatChat < 0 &&
            generateOutputPrefix == null
        ) {
            return
        }

        if (!modelFilter.isNullOrEmpty()) {
            benchmarkModelFilter = modelFilter
            log("Benchmark model filter: ${modelFilter.joinToString(",")}")
        }
        if (!presetFilter.isNullOrEmpty()) {
            benchmarkPresetFilter = presetFilter
            log("Benchmark preset filter: ${presetFilter.joinToString(",")}")
        }

        if (benchCooldownMs >= 0) {
            bundledModelCooldownMs = benchCooldownMs.toLong()
            log("Benchmark cooldown override: ${bundledModelCooldownMs} ms")
        }
        if (benchRepetitions > 0) {
            benchmarkRepetitionsOverride = benchRepetitions
            log("Benchmark repetitions override: $benchRepetitions")
        }
        if (benchPromptTokens > 0 || benchGenTokens > 0 || benchPromptLength > 0) {
            val promptTokens = if (benchPromptTokens > 0) benchPromptTokens else 8
            val genTokens = if (benchGenTokens > 0) benchGenTokens else 2
            val promptLength = if (benchPromptLength > 0) benchPromptLength else 1
            val repetitions = benchmarkRepetitionsOverride ?: 1
            benchmarkPresetOverride = BenchPreset(
                label = "adb_custom",
                promptTokens = promptTokens,
                genTokens = genTokens,
                pl = promptLength,
                repetitions = repetitions,
            )
            log("Benchmark custom preset: prompt=$promptTokens decode=$genTokens pl=$promptLength repetitions=$repetitions")
        }
        val resolvedPowerDurationMs = powerDurationMs ?: powerDurationMinutes?.let { it * 60L * 1000L }
        if (resolvedPowerDurationMs != null && resolvedPowerDurationMs > 0L) {
            powerBenchDurationMs = resolvedPowerDurationMs
            log("Power benchmark duration override: ${powerBenchDurationMs} ms")
        }
        if (configuredBatteryCapacityMah != null && configuredBatteryCapacityMah > 0.0) {
            batteryCapacityMah = configuredBatteryCapacityMah
            log("Battery capacity override: ${formatCsvDecimal(configuredBatteryCapacityMah)} mAh")
        }
        if (generatePrompt != null) {
            automationGeneratePrompt = generatePrompt
            log("Automation generate prompt override: ${generatePrompt.take(80)}")
        }
        if (generateMaxTokens > 0) {
            automationGenerateMaxTokens = generateMaxTokens.coerceIn(1, 512)
            log("Automation generate max tokens override: $automationGenerateMaxTokens")
        }
        if (generateFormatChat >= 0) {
            automationGenerateFormatChat = generateFormatChat != 0
            log("Automation generate format_chat override: $automationGenerateFormatChat")
        }
        if (generateOutputPrefix != null) {
            automationGenerateOutputPrefix = sanitizeFileName(generateOutputPrefix)
            log("Automation generate output prefix override: $automationGenerateOutputPrefix")
        }
        runtimeBenchConfig = nextRuntimeBenchConfig

        viewModelScope.launch {
            runCatching {
                llamaAndroid.configureRuntime(
                    nCtx = nCtx,
                    nBatch = nBatch,
                    nUbatch = nUbatch,
                    nThreads = nThreads,
                    nThreadsBatch = nThreadsBatch,
                    disableIFairyLut = disableIFairyLut,
                    affinityProfile = affinityProfile,
                    enableIFairyVecdotActTensor = enableIFairyVecdotActTensor,
                    generationPriority = generationPriority,
                    batchPriority = batchPriority,
                    schedDebug = schedDebug,
                    openclSupportsDebug = openclSupportsDebug,
                    forceCpu = forceCpu,
                )
            }.onSuccess {
                log(
                    "Runtime overrides: n_ctx=${formatOverride(nCtx)}, n_batch=${formatOverride(nBatch)}, " +
                        "n_ubatch=${formatOverride(nUbatch)}, n_threads=${formatOverride(nThreads)}, " +
                        "n_threads_batch=${formatOverride(nThreadsBatch)}, disable_ifairy_lut=${formatOverride(disableIFairyLut)}, " +
                        "affinity_profile=${formatOverride(affinityProfile)}, " +
                        "ifairy_vecdot_act_tensor=${formatOverride(enableIFairyVecdotActTensor)}, " +
                        "sched_debug=${formatOverride(schedDebug)}, " +
                        "opencl_supports_debug=${formatOverride(openclSupportsDebug)}, " +
                        "force_cpu=${formatOverride(forceCpu)}, " +
                        "bench_cooldown_ms=${formatOverride(benchCooldownMs)}, " +
                        "bench_repetitions=${formatOverride(benchRepetitions)}, " +
                        "power_duration_ms=${powerDurationMs ?: "default"}, " +
                        "battery_capacity_mah=${configuredBatteryCapacityMah?.let(::formatCsvDecimal) ?: "default"}, " +
                        "model_filter=${modelFilter?.joinToString(",") ?: "default"}, " +
                        "preset_filter=${presetFilter?.joinToString(",") ?: "default"}, " +
                        "generation_priority=${formatOverride(generationPriority, -99)}, " +
                        "batch_priority=${formatOverride(batchPriority, -99)}"
                )
            }.onFailure { throwable ->
                logError("configureRuntime() failed", throwable)
            }
        }
    }

    fun importModel(context: Context, uri: Uri) {
        if (modelLoadState == ModelLoadState.IMPORTING || isGenerating || isBenchmarking) {
            return
        }

        modelLoadState = ModelLoadState.IMPORTING
        modelError = null
        log("Importing model from $uri")

        viewModelScope.launch {
            runCatching {
                copyModelToPrivateStorage(context.applicationContext, uri)
            }.onSuccess { imported ->
                refreshAvailableModels(context.applicationContext, imported.fileName)
                modelLoadState = ModelLoadState.IMPORTED
                log("Imported ${imported.fileName}")
                log("Model path: ${imported.privatePath}")
                log("Model size: ${formatSize(imported.sizeBytes)}")
            }.onFailure { throwable ->
                refreshAvailableModels(context.applicationContext)
                modelLoadState = if (importedModel == null) ModelLoadState.FAILED else ModelLoadState.IMPORTED
                modelError = throwable.message ?: "Model import failed"
                logError("importModel() failed", throwable)
            }
        }
    }

    fun refreshModelList() {
        val context = appContext ?: return
        refreshAvailableModels(context, importedModel?.fileName)
        log("Refreshed model list: ${availableModels.size} model(s)")
    }

    fun selectAvailableModel(fileName: String) {
        if (isGenerating || isBenchmarking || modelLoadState == ModelLoadState.LOADING || modelLoadState == ModelLoadState.IMPORTING) {
            modelError = "Stop generation, loading, import, or benchmark before selecting another model"
            return
        }

        val model = availableModels.firstOrNull { it.fileName == fileName } ?: run {
            modelError = "Model not found: $fileName"
            return
        }

        if (importedModel?.privatePath == model.privatePath && modelLoadState != ModelLoadState.LOADED) {
            return
        }

        viewModelScope.launch {
            runCatching {
                if (modelLoadState == ModelLoadState.LOADED) {
                    llamaAndroid.unload()
                    log("Model unloaded")
                }
                importedModel = model
                modelLoadState = ModelLoadState.IMPORTED
                modelError = null
                log("Selected model: ${model.fileName}")
                log("Model path: ${model.privatePath}")
                log("Model size: ${formatSize(model.sizeBytes)}")
            }.onFailure { throwable ->
                modelError = throwable.message ?: "Model selection failed"
                logError("selectAvailableModel() failed", throwable)
            }
        }
    }

    fun loadImportedModel() {
        val model = importedModel ?: run {
            modelError = "No imported model available"
            modelLoadState = ModelLoadState.FAILED
            return
        }

        if (isGenerating || isBenchmarking) {
            modelError = "Stop generation or benchmarking before loading another model"
            modelLoadState = ModelLoadState.FAILED
            return
        }

        viewModelScope.launch {
            runCatching {
                loadModelInternal(model)
            }.onSuccess {
                maybeStartPendingAutomation()
            }.onFailure { throwable ->
                modelLoadState = ModelLoadState.FAILED
                modelError = throwable.message ?: "Model load failed"
                logError("loadImportedModel() failed", throwable)
            }
        }
    }

    fun unloadModel() {
        if (isGenerating) {
            stopGeneration()
        }
        if (isBenchmarking) {
            return
        }

        viewModelScope.launch {
            runCatching { unloadModelInternal() }
                .onFailure { logError("unloadModel() failed", it) }
        }
    }

    fun send() {
        val text = prompt.trim()
        if (text.isEmpty()) {
            return
        }
        if (modelLoadState != ModelLoadState.LOADED) {
            modelError = "Load a model before generating"
            return
        }
        if (isGenerating || isBenchmarking || llamaAndroid.isGenerating()) {
            modelError = "Generation or benchmark already in progress"
            return
        }

        prompt = ""
        modelError = null

        val userMessage = ChatMessage(nextMessageId++, ChatRole.USER, text)
        val assistantMessage = ChatMessage(nextMessageId++, ChatRole.ASSISTANT, "")
        messages = messages + userMessage + assistantMessage

        resetStreamingBuffer()
        generationJob = viewModelScope.launch {
            isGenerating = true
            try {
                ensureTypingJob(assistantMessage.id)
                llamaAndroid.send(text, resolvedGenerationLength())
                    .catch { throwable ->
                        modelError = throwable.message ?: "Generation failed"
                        logError("send() failed", throwable)
                    }
                    .collect { piece ->
                        appendToStreamingBuffer(piece)
                    }
            } finally {
                flushStreamingBuffer(assistantMessage.id)
                typingJob?.cancel()
                typingJob = null
                isGenerating = false
                generationJob = null
            }
        }
    }

    fun stopGeneration() {
        if (!isGenerating) {
            return
        }
        llamaAndroid.stop()
        log("Stop requested")
    }

    fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1) {
        if (modelLoadState != ModelLoadState.LOADED) {
            modelError = "Load a model before benchmark"
            return
        }
        if (isGenerating || isBenchmarking) {
            modelError = "Another benchmark or generation is in progress"
            return
        }

        val model = importedModel ?: return
        val preset = BenchPreset(
            label = "custom",
            promptTokens = pp,
            genTokens = tg,
            pl = pl,
            repetitions = nr,
        )

        viewModelScope.launch {
            isBenchmarking = true
            try {
                val summary = runBenchPreset(model, preset)
                logBenchSummary(summary)
            } catch (throwable: Throwable) {
                logError("bench() failed", throwable)
            } finally {
                isBenchmarking = false
            }
        }
    }

    fun startPowerBenchmark() {
        viewModelScope.launch {
            applyPowerBenchmarkRuntimeDefaults()
            runBundledPowerBenchmarks()
        }
    }

    fun selectGenerationLength(preset: GenerationLengthPreset) {
        generationLength = preset
        useCustomGenerationLength = false
    }

    fun selectCustomGenerationLength() {
        useCustomGenerationLength = true
    }

    fun updateCustomGenerationLength(input: String) {
        customGenerationLengthInput = input.filter { it.isDigit() }.take(4)
    }

    fun updatePrompt(newPrompt: String) {
        prompt = newPrompt
    }

    fun clearDiagnostics() {
        diagnostics = listOf()
    }

    fun log(message: String) {
        Log.i(LOG_TAG, message)
        diagnostics = diagnostics + message
    }

    fun diagnosticsText(): String = diagnostics.joinToString("\n")

    private suspend fun loadModelInternal(model: ImportedModel) {
        modelLoadState = ModelLoadState.LOADING
        modelError = null
        log("Loading model from ${model.privatePath}")
        llamaAndroid.load(model.privatePath)
        importedModel = model
        modelLoadState = ModelLoadState.LOADED
        log("Model loaded: ${model.fileName}")
    }

    private suspend fun unloadModelInternal() {
        llamaAndroid.unload()
        modelLoadState = if (importedModel == null) ModelLoadState.NOT_IMPORTED else ModelLoadState.IMPORTED
        log("Model unloaded")
    }

    private suspend fun runBundledBenchmarks() {
        if (isGenerating || isBenchmarking) {
            modelError = "Another benchmark or generation is in progress"
            return
        }

        val bundledModels = benchmarkTargetModels()
        if (bundledModels.isEmpty()) {
            modelError = "No benchmark target models are installed"
            return
        }

        isBenchmarking = true
        try {
            applyCurrentRuntimeBenchConfig()
            val context = requireNotNull(appContext) { "App context is not initialized" }
            val summaries = mutableListOf<BenchSummary>()
            writeBenchStatus(context, BENCH_STATUS_FILE_NAME, "running")
            log("Starting bundled model benchmark sweep")
            for ((index, model) in bundledModels.withIndex()) {
                loadModelInternal(model)
                warmupBenchIfNeeded()
                benchmarkTargetPresets().forEach { preset ->
                    summaries += runBenchPreset(model, preset)
                }
                unloadModelInternal()
                if (index < bundledModels.lastIndex) {
                    log("Cooling down for ${bundledModelCooldownMs / 1000}s after ${model.fileName}")
                    delay(bundledModelCooldownMs)
                }
            }
            writeBenchCsv(context, summaries)
            logCombinedBenchTable(summaries)
            writeBenchStatus(context, BENCH_STATUS_FILE_NAME, "completed")
            log("Completed bundled model benchmark sweep")
        } catch (throwable: Throwable) {
            appContext?.let { context ->
                runCatching {
                    writeBenchStatus(
                        context,
                        BENCH_STATUS_FILE_NAME,
                        buildString {
                            append("failed")
                            val message = throwable.message?.takeIf { it.isNotBlank() }
                            if (message != null) {
                                append('\n')
                                append(message)
                            }
                        },
                    )
                }
            }
            logError("runBundledBenchmarks() failed", throwable)
        } finally {
            isBenchmarking = false
        }
    }

    private suspend fun applyCurrentRuntimeBenchConfig() {
        val config = runtimeBenchConfig
        llamaAndroid.configureRuntime(
            nCtx = config.nCtx,
            nBatch = config.nBatch,
            nUbatch = config.nUbatch,
            nThreads = config.nThreads,
            nThreadsBatch = config.nThreadsBatch,
            disableIFairyLut = config.disableIFairyLut,
            affinityProfile = config.affinityProfile,
            enableIFairyVecdotActTensor = config.enableIFairyVecdotActTensor,
            generationPriority = config.generationPriority,
            batchPriority = config.batchPriority,
            schedDebug = config.schedDebug,
            openclSupportsDebug = config.openclSupportsDebug,
            forceCpu = config.forceCpu,
        )
        log(
            "Applied runtime config before benchmark load: " +
                "n_ctx=${formatOverride(config.nCtx)}, n_batch=${formatOverride(config.nBatch)}, " +
                "n_ubatch=${formatOverride(config.nUbatch)}, n_threads=${formatOverride(config.nThreads)}, " +
                "n_threads_batch=${formatOverride(config.nThreadsBatch)}, disable_ifairy_lut=${formatOverride(config.disableIFairyLut)}, " +
                "affinity_profile=${formatOverride(config.affinityProfile)}, " +
                "ifairy_vecdot_act_tensor=${formatOverride(config.enableIFairyVecdotActTensor)}, " +
                "generation_priority=${formatOverride(config.generationPriority, -99)}, " +
                "batch_priority=${formatOverride(config.batchPriority, -99)}, " +
                "sched_debug=${formatOverride(config.schedDebug)}, opencl_supports_debug=${formatOverride(config.openclSupportsDebug)}, " +
                "force_cpu=${formatOverride(config.forceCpu)}"
        )
    }

    private suspend fun runAutomationGenerate() {
        if (isGenerating || isBenchmarking || llamaAndroid.isGenerating()) {
            modelError = "Generation or benchmark already in progress"
            return
        }

        val context = requireNotNull(appContext) { "App context is not initialized" }
        val model = importedModel ?: error("No imported model available")
        val outputPrefix = sanitizeFileName(automationGenerateOutputPrefix)
        val statusFileName = outputPrefix + GENERATE_STATUS_FILE_SUFFIX
        val outputFileName = outputPrefix + GENERATE_OUTPUT_FILE_SUFFIX

        isBenchmarking = true
        try {
            writeBenchStatus(context, statusFileName, "running")
            applyCurrentRuntimeBenchConfig()
            if (modelLoadState == ModelLoadState.LOADED) {
                unloadModelInternal()
            }
            loadModelInternal(model)

            val generated = StringBuilder()
            val startMs = SystemClock.elapsedRealtime()
            llamaAndroid
                .send(
                    automationGeneratePrompt,
                    automationGenerateMaxTokens,
                    automationGenerateFormatChat,
                )
                .collect { piece ->
                    generated.append(piece)
                }
            val elapsedMs = SystemClock.elapsedRealtime() - startMs

            writeGenerateOutput(
                context = context,
                fileName = outputFileName,
                model = model,
                elapsedMs = elapsedMs,
                generatedText = generated.toString(),
            )
            writeBenchStatus(
                context,
                statusFileName,
                buildString {
                    appendLine("completed")
                    appendLine("output=$outputFileName")
                    appendLine("generated_chars=${generated.length}")
                    appendLine("elapsed_ms=$elapsedMs")
                },
            )
            log("Automation generate completed: output=$outputFileName chars=${generated.length} elapsed_ms=$elapsedMs")
        } catch (throwable: Throwable) {
            writeBenchStatus(
                context,
                statusFileName,
                buildString {
                    appendLine("failed")
                    appendLine(throwable.message ?: throwable::class.java.simpleName)
                },
            )
            logError("runAutomationGenerate() failed", throwable)
        } finally {
            isBenchmarking = false
        }
    }

    private suspend fun runBundledE2eBenchmarks() {
        if (isGenerating || isBenchmarking) {
            modelError = "Another benchmark or generation is in progress"
            return
        }

        val bundledModels = benchmarkTargetModels()
        if (bundledModels.isEmpty()) {
            modelError = "No benchmark target models are installed"
            return
        }

        isBenchmarking = true
        try {
            val context = requireNotNull(appContext) { "App context is not initialized" }
            val targetPresets = e2eBenchmarkTargetPresets()
            val statusFileName = e2eBenchStatusFileName(targetPresets)
            val summaries = mutableListOf<E2eBenchSummary>()
            writeBenchStatus(context, statusFileName, "running")
            log("Starting bundled model E2E benchmark sweep")
            for ((index, model) in bundledModels.withIndex()) {
                loadModelInternal(model)
                warmupE2eBenchIfNeeded()
                targetPresets.forEach { preset ->
                    summaries += runE2eBenchPreset(model, preset)
                }
                unloadModelInternal()
                if (index < bundledModels.lastIndex) {
                    log("Cooling down for ${bundledModelCooldownMs / 1000}s after ${model.fileName}")
                    delay(bundledModelCooldownMs)
                }
            }
            writeE2eBenchCsv(context, summaries)
            logCombinedE2eBenchTable(summaries)
            writeBenchStatus(context, statusFileName, "completed")
            log("Completed bundled model E2E benchmark sweep")
        } catch (throwable: Throwable) {
            appContext?.let { context ->
                runCatching {
                    writeBenchStatus(
                        context,
                        e2eBenchStatusFileName(e2eBenchmarkTargetPresets()),
                        buildString {
                            append("failed")
                            val message = throwable.message?.takeIf { it.isNotBlank() }
                            if (message != null) {
                                append('\n')
                                append(message)
                            }
                        },
                    )
                }
            }
            logError("runBundledE2eBenchmarks() failed", throwable)
        } finally {
            isBenchmarking = false
        }
    }

    private suspend fun runBundledPowerBenchmarks() {
        if (isGenerating || isBenchmarking) {
            modelError = "Another benchmark or generation is in progress"
            return
        }

        val targetModels = powerBenchmarkTargetModels()
        if (targetModels.isEmpty()) {
            modelError = "No power benchmark target models are installed"
            return
        }

        isBenchmarking = true
        val context = requireNotNull(appContext) { "App context is not initialized" }
        val wakeLock = acquirePowerBenchWakeLock(context)
        try {
            val summaries = mutableListOf<PowerBenchSummary>()
            powerBenchOutputFileNames.clear()
            writeBenchStatus(context, POWER_BENCH_STATUS_FILE_NAME, "running")
            log("Starting power benchmark sweep: duration=${powerBenchDurationMs / 1000}s per model/preset")
            for ((index, model) in targetModels.withIndex()) {
                if (modelLoadState == ModelLoadState.LOADED && importedModel?.privatePath == model.privatePath) {
                    log("Reusing loaded model: ${model.fileName}")
                } else {
                    loadModelInternal(model)
                }
                warmupE2eBenchIfNeeded()
                powerBenchmarkTargetPresets().forEach { preset ->
                    summaries += runPowerBenchPreset(context, model, preset)
                    writePowerBenchCsv(context, summaries)
                }
                unloadModelInternal()
                if (index < targetModels.lastIndex) {
                    log("Cooling down for ${bundledModelCooldownMs / 1000}s after ${model.fileName}")
                    delay(bundledModelCooldownMs)
                }
            }
            writePowerBenchCsv(context, summaries)
            logCombinedPowerBenchTable(summaries)
            writeBenchStatus(context, POWER_BENCH_STATUS_FILE_NAME, "completed")
            log("Completed power benchmark sweep")
        } catch (throwable: Throwable) {
            appContext?.let { context ->
                runCatching {
                    writeBenchStatus(
                        context,
                        POWER_BENCH_STATUS_FILE_NAME,
                        buildString {
                            append("failed")
                            val message = throwable.message?.takeIf { it.isNotBlank() }
                            if (message != null) {
                                append('\n')
                                append(message)
                            }
                        },
                    )
                }
            }
            logError("runBundledPowerBenchmarks() failed", throwable)
        } finally {
            if (wakeLock.isHeld) {
                wakeLock.release()
            }
            isBenchmarking = false
        }
    }

    private suspend fun applyPowerBenchmarkRuntimeDefaults() {
        runCatching {
            llamaAndroid.configureRuntime(
                nThreads = DEFAULT_POWER_BENCH_THREADS,
                nThreadsBatch = DEFAULT_POWER_BENCH_THREADS,
                affinityProfile = DEFAULT_POWER_BENCH_AFFINITY_PROFILE,
            )
        }.onSuccess {
            log(
                "Power benchmark runtime defaults: " +
                    "n_threads=$DEFAULT_POWER_BENCH_THREADS, " +
                    "n_threads_batch=$DEFAULT_POWER_BENCH_THREADS, " +
                    "affinity_profile=$DEFAULT_POWER_BENCH_AFFINITY_PROFILE"
            )
        }.onFailure { throwable ->
            logError("applyPowerBenchmarkRuntimeDefaults() failed", throwable)
        }
    }

    private suspend fun runBenchPreset(model: ImportedModel, preset: BenchPreset): BenchSummary {
        log(
            "Running benchmark: model=${model.fileName}, preset=${preset.label}, " +
                "prompt=${preset.promptTokens}, decode=${preset.genTokens}, repetitions=${preset.repetitions}"
        )

        val raw = llamaAndroid.bench(
            pp = preset.promptTokens,
            tg = preset.genTokens,
            pl = preset.pl,
            nr = preset.repetitions,
        )
        log(raw)

        return parseBenchResult(model.fileName, preset, raw)
            ?: error("Failed to parse benchmark output for ${model.fileName} (${preset.label})")
    }

    private suspend fun warmupBenchIfNeeded() {
        val preset = benchmarkPresetOverride ?: BenchPreset("warmup", 8, 4, 1, 1)
        log("Warmup benchmark: prompt=${preset.promptTokens} decode=${preset.genTokens}")
        runCatching {
            llamaAndroid.bench(pp = preset.promptTokens, tg = preset.genTokens, pl = preset.pl, nr = 1)
        }
            .onFailure { logError("warmupBenchIfNeeded() failed", it) }
    }

    private suspend fun runE2eBenchPreset(model: ImportedModel, preset: BenchPreset): E2eBenchSummary {
        log(
            "Running E2E benchmark: model=${model.fileName}, preset=${preset.label}, " +
                "prompt=${preset.promptTokens}, decode=${preset.genTokens}, repetitions=${preset.repetitions}, " +
                "threads=${formatOverride(runtimeBenchConfig.nThreads)}, " +
                "batch_threads=${formatOverride(runtimeBenchConfig.nThreadsBatch)}, " +
                "affinity=${formatOverride(runtimeBenchConfig.affinityProfile)}, " +
                "disable_ifairy_lut=${formatOverride(runtimeBenchConfig.disableIFairyLut)}, " +
                "ifairy_vecdot_act_tensor=${formatOverride(runtimeBenchConfig.enableIFairyVecdotActTensor)}"
        )

        val result = llamaAndroid.e2eBench(
            pp = preset.promptTokens,
            tg = preset.genTokens,
            nr = preset.repetitions,
        )

        return E2eBenchSummary(
            modelFileName = model.fileName,
            presetLabel = preset.label,
            promptTokens = preset.promptTokens,
            genTokens = preset.genTokens,
            repetitions = preset.repetitions,
            nThreads = runtimeBenchConfig.nThreads,
            nThreadsBatch = runtimeBenchConfig.nThreadsBatch,
            affinityProfile = runtimeBenchConfig.affinityProfile,
            disableIFairyLut = runtimeBenchConfig.disableIFairyLut,
            enableIFairyVecdotActTensor = runtimeBenchConfig.enableIFairyVecdotActTensor,
            generatedTokensAvg = result.generatedTokensAvg,
            prefillMs = result.prefillMs,
            firstTokenMs = result.firstTokenMs,
            decodeLatencyMsPerToken = result.decodeMsPerToken,
            tokS = result.tokS,
            totalMs = result.totalMs,
        )
    }

    private suspend fun runPowerBenchPreset(
        context: Context,
        model: ImportedModel,
        preset: BenchPreset,
    ): PowerBenchSummary {
        log(
            "Running power benchmark: model=${model.fileName}, preset=${preset.label}, " +
                "prompt=${preset.promptTokens}, decode=${preset.genTokens}, repetitions=${preset.repetitions}, " +
                "duration=${powerBenchDurationMs / 1000}s"
        )

        val startBattery = readBatterySnapshot(context)
        val startElapsedMs = SystemClock.elapsedRealtime()
        val deadlineMs = startElapsedMs + powerBenchDurationMs
        var loops = 0
        var generatedTokensTotal = 0.0
        var prefillMsTotal = 0.0
        var firstTokenMsTotal = 0.0
        var decodeMsPerTokenTotal = 0.0
        var tokSTotal = 0.0
        var totalMsTotal = 0.0
        var refreshCount = 0
        var refreshTotalMs = 0L

        while (true) {
            val nowMs = SystemClock.elapsedRealtime()
            if (nowMs >= deadlineMs && loops > 0) {
                break
            }

            val result = llamaAndroid.e2eBench(
                pp = preset.promptTokens,
                tg = preset.genTokens,
                nr = preset.repetitions,
            )
            loops += 1
            generatedTokensTotal += result.generatedTokensAvg * preset.repetitions
            prefillMsTotal += result.prefillMs
            firstTokenMsTotal += result.firstTokenMs
            decodeMsPerTokenTotal += result.decodeMsPerToken
            tokSTotal += result.tokS
            totalMsTotal += result.totalMs
            log(
                String.format(
                    Locale.US,
                    "Power loop %d: model=%s preset=%s elapsed_s=%.2f target_s=%.2f tok_s=%.4f total_ms=%.2f",
                    loops,
                    model.fileName,
                    preset.label,
                    (SystemClock.elapsedRealtime() - startElapsedMs) / 1000.0,
                    powerBenchDurationMs / 1000.0,
                    result.tokS,
                    result.totalMs,
                )
            )
            writePowerProgressStatus(
                context = context,
                model = model,
                preset = preset,
                loops = loops,
                refreshCount = refreshCount,
                elapsedMs = SystemClock.elapsedRealtime() - startElapsedMs,
            )

            if (SystemClock.elapsedRealtime() < deadlineMs) {
                val refreshMs = refreshLoadedModelForPowerBench(model)
                refreshCount += 1
                refreshTotalMs += refreshMs
            }
        }

        val endElapsedMs = SystemClock.elapsedRealtime()
        val endBattery = readBatterySnapshot(context)
        val elapsedMs = endElapsedMs - startElapsedMs
        val elapsedHours = elapsedMs / 3_600_000.0
        val batteryDeltaPercent = if (startBattery.percent != null && endBattery.percent != null) {
            startBattery.percent - endBattery.percent
        } else {
            null
        }
        val chargeDeltaUah = if (startBattery.chargeCounterUah != null && endBattery.chargeCounterUah != null) {
            startBattery.chargeCounterUah - endBattery.chargeCounterUah
        } else {
            null
        }
        val remainingCapacityDeltaMah =
            if (startBattery.remainingCapacityMah != null && endBattery.remainingCapacityMah != null) {
                startBattery.remainingCapacityMah - endBattery.remainingCapacityMah
            } else {
                null
            }
        val energyDeltaWh = if (startBattery.energyCounterNwh != null && endBattery.energyCounterNwh != null) {
            (startBattery.energyCounterNwh - endBattery.energyCounterNwh) / 1_000_000_000.0
        } else {
            null
        }
        val avgVoltageV = listOfNotNull(startBattery.voltageMv, endBattery.voltageMv)
            .takeIf { it.isNotEmpty() }
            ?.average()
            ?.div(1000.0)
        val estimatedEnergyWh = estimateEnergyWh(
            energyDeltaWh = energyDeltaWh,
            chargeDeltaUah = chargeDeltaUah,
            remainingCapacityDeltaMah = remainingCapacityDeltaMah,
            avgVoltageV = avgVoltageV,
            batteryDeltaPercent = batteryDeltaPercent,
        )
        val estimatedAveragePowerW = estimatedEnergyWh?.takeIf { elapsedHours > 0.0 }?.div(elapsedHours)

        return PowerBenchSummary(
            modelFileName = model.fileName,
            presetLabel = preset.label,
            promptTokens = preset.promptTokens,
            genTokens = preset.genTokens,
            repetitions = preset.repetitions,
            targetDurationMs = powerBenchDurationMs,
            elapsedMs = elapsedMs,
            loops = loops,
            refreshCount = refreshCount,
            refreshTotalMs = refreshTotalMs,
            generatedTokensTotal = generatedTokensTotal,
            startBattery = startBattery,
            endBattery = endBattery,
            batteryDeltaPercent = batteryDeltaPercent,
            batteryDeltaPercentPerHour = batteryDeltaPercent?.takeIf { elapsedHours > 0.0 }?.div(elapsedHours),
            chargeDeltaUah = chargeDeltaUah,
            startRemainingCapacityMah = startBattery.remainingCapacityMah,
            endRemainingCapacityMah = endBattery.remainingCapacityMah,
            remainingCapacityDeltaMah = remainingCapacityDeltaMah,
            energyDeltaWh = energyDeltaWh,
            estimatedEnergyWh = estimatedEnergyWh,
            estimatedAveragePowerW = estimatedAveragePowerW,
            avgPrefillMs = prefillMsTotal / loops,
            avgFirstTokenMs = firstTokenMsTotal / loops,
            avgDecodeMsPerToken = decodeMsPerTokenTotal / loops,
            avgTokS = tokSTotal / loops,
            avgTotalMs = totalMsTotal / loops,
            note = powerEstimateNote(energyDeltaWh, chargeDeltaUah, batteryDeltaPercent, avgVoltageV),
        )
    }

    private suspend fun warmupE2eBenchIfNeeded() {
        log("Warmup E2E benchmark: prompt=8 decode=4")
        runCatching { llamaAndroid.e2eBench(pp = 8, tg = 4, nr = 1) }
            .onFailure { logError("warmupE2eBenchIfNeeded() failed", it) }
    }

    private suspend fun refreshLoadedModelForPowerBench(model: ImportedModel): Long {
        val startedAt = SystemClock.elapsedRealtime()
        log("Refreshing native backend state after power bench loop: ${model.fileName}")
        llamaAndroid.refreshBackend()
        System.gc()
        loadModelInternal(model)
        val elapsedMs = SystemClock.elapsedRealtime() - startedAt
        log("Power bench backend refresh completed in ${elapsedMs} ms")
        return elapsedMs
    }

    private fun readBatterySnapshot(context: Context): BatterySnapshot {
        val intent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = intent?.getValidIntExtra(BatteryManager.EXTRA_LEVEL)
        val scale = intent?.getValidIntExtra(BatteryManager.EXTRA_SCALE)
        val voltageMv = intent?.getValidIntExtra(BatteryManager.EXTRA_VOLTAGE)
        val temperatureTenthsC = intent?.getValidIntExtra(BatteryManager.EXTRA_TEMPERATURE)
        val status = intent?.getValidIntExtra(BatteryManager.EXTRA_STATUS)
        val plugged = intent?.getValidIntExtra(BatteryManager.EXTRA_PLUGGED)
        val batteryManager = context.getSystemService(BatteryManager::class.java)

        return BatterySnapshot(
            percent = if (level != null && scale != null && scale > 0) level * 100.0 / scale else null,
            level = level,
            scale = scale,
            voltageMv = voltageMv,
            temperatureC = temperatureTenthsC?.div(10.0),
            status = status,
            plugged = plugged,
            chargeCounterUah = batteryManager?.getValidLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER),
            remainingCapacityMah = batteryManager
                ?.getValidLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)
                ?.toBatteryCapacityMah(),
            energyCounterNwh = batteryManager?.getValidLongProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER),
        )
    }

    @SuppressLint("WakelockTimeout")
    private fun acquirePowerBenchWakeLock(context: Context): PowerManager.WakeLock {
        val powerManager = context.getSystemService(PowerManager::class.java)
        val wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "$LOG_TAG:PowerBench")
        wakeLock.acquire()
        log("Power benchmark wake lock acquired")
        return wakeLock
    }

    private fun estimateEnergyWh(
        energyDeltaWh: Double?,
        chargeDeltaUah: Long?,
        remainingCapacityDeltaMah: Double?,
        avgVoltageV: Double?,
        batteryDeltaPercent: Double?,
    ): Double? {
        if (energyDeltaWh != null && energyDeltaWh > 0.0) {
            return energyDeltaWh
        }
        if (chargeDeltaUah != null && chargeDeltaUah > 0L && avgVoltageV != null) {
            return chargeDeltaUah.toBatteryCapacityMah() / 1000.0 * avgVoltageV
        }
        if (remainingCapacityDeltaMah != null && remainingCapacityDeltaMah > 0.0 && avgVoltageV != null) {
            return remainingCapacityDeltaMah / 1000.0 * avgVoltageV
        }
        val capacityMah = batteryCapacityMah
        if (batteryDeltaPercent != null && batteryDeltaPercent > 0.0 && capacityMah != null && avgVoltageV != null) {
            return batteryDeltaPercent / 100.0 * capacityMah / 1000.0 * avgVoltageV
        }
        return null
    }

    private fun powerEstimateNote(
        energyDeltaWh: Double?,
        chargeDeltaUah: Long?,
        batteryDeltaPercent: Double?,
        avgVoltageV: Double?,
    ): String {
        return when {
            energyDeltaWh != null && energyDeltaWh > 0.0 -> "energy_counter_nwh"
            chargeDeltaUah != null && chargeDeltaUah > 0L && avgVoltageV != null -> "charge_counter_capacity_mah_times_avg_voltage"
            batteryDeltaPercent != null && batteryDeltaPercent > 0.0 && batteryCapacityMah != null && avgVoltageV != null ->
                "battery_percent_times_configured_capacity_times_avg_voltage"
            else -> "percent_only_no_energy_estimate"
        }
    }

    private fun parseBenchResult(modelFileName: String, preset: BenchPreset, raw: String): BenchSummary? {
        var ppTps: Double? = null
        var tgTps: Double? = null
        var ppStd: Double? = null
        var tgStd: Double? = null

        raw.lineSequence()
            .map { it.trim() }
            .filter { it.startsWith("|") && !it.startsWith("| ---") }
            .forEach { line ->
                val columns = line.split("|")
                    .map { it.trim() }
                    .filter { it.isNotEmpty() }
                if (columns.size < 6 || columns[0] == "model") {
                    return@forEach
                }

                val test = columns[4]
                val speed = columns[5]
                val metric = Regex("""([0-9.]+)\s*±\s*([0-9.]+)""").find(speed) ?: return@forEach
                val avg = metric.groupValues[1].toDoubleOrNull() ?: return@forEach
                val std = metric.groupValues[2].toDoubleOrNull() ?: return@forEach

                when {
                    test.startsWith("pp ") -> {
                        ppTps = avg
                        ppStd = std
                    }
                    test.startsWith("tg ") -> {
                        tgTps = avg
                        tgStd = std
                    }
                }
            }

        val promptTps = ppTps ?: return null
        val decodeTps = tgTps ?: return null
        val promptStd = ppStd ?: 0.0
        val decodeStd = tgStd ?: 0.0
        val decodeTotalTokens = preset.genTokens * preset.pl

        return BenchSummary(
            modelFileName = modelFileName,
            presetLabel = preset.label,
            promptTokens = preset.promptTokens,
            genTokens = preset.genTokens,
            repetitions = preset.repetitions,
            ppTps = promptTps,
            tgTps = decodeTps,
            ppStd = promptStd,
            tgStd = decodeStd,
            promptLatencyMs = 1000.0 * preset.promptTokens / promptTps,
            decodeLatencyMsPerToken = 1000.0 / decodeTps,
            decodeTotalMs = 1000.0 * decodeTotalTokens / decodeTps,
        )
    }

    private fun logBenchSummary(summary: BenchSummary) {
        log(
            String.format(
                Locale.US,
                "LLAMA-BENCH-SUMMARY model=%s preset=%s prompt=%d decode=%d pp_tps=%.4f tg_tps=%.4f prompt_latency_ms=%.2f decode_ms_per_token=%.2f decode_total_ms=%.2f",
                summary.modelFileName,
                summary.presetLabel,
                summary.promptTokens,
                summary.genTokens,
                summary.ppTps,
                summary.tgTps,
                summary.promptLatencyMs,
                summary.decodeLatencyMsPerToken,
                summary.decodeTotalMs,
            )
        )
    }

    private fun logCombinedBenchTable(summaries: List<BenchSummary>) {
        log("BENCH-TABLE combined")
        log("| model | preset | prompt | decode | pp t/s | tg t/s | prompt latency ms | decode ms/token | decode total ms |")
        log("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        summaries.forEach { summary ->
            log(
                String.format(
                    Locale.US,
                    "| %s | %s | %d | %d | %.4f | %.4f | %.2f | %.2f | %.2f |",
                    summary.modelFileName,
                    summary.presetLabel,
                    summary.promptTokens,
                    summary.genTokens,
                    summary.ppTps,
                    summary.tgTps,
                    summary.promptLatencyMs,
                    summary.decodeLatencyMsPerToken,
                    summary.decodeTotalMs,
                )
            )
        }
    }

    private fun logCombinedE2eBenchTable(summaries: List<E2eBenchSummary>) {
        log("BENCH-TABLE-E2E combined")
        log("| model | preset | prompt | decode | gen avg | prefill ms | first token ms | decode ms/token | tok/s | total ms |")
        log("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        summaries.forEach { summary ->
            log(
                String.format(
                    Locale.US,
                    "| %s | %s | %d | %d | %.2f | %.2f | %.2f | %.2f | %.4f | %.2f |",
                    summary.modelFileName,
                    summary.presetLabel,
                    summary.promptTokens,
                    summary.genTokens,
                    summary.generatedTokensAvg,
                    summary.prefillMs,
                    summary.firstTokenMs,
                    summary.decodeLatencyMsPerToken,
                    summary.tokS,
                    summary.totalMs,
                )
            )
        }
    }

    private fun logCombinedPowerBenchTable(summaries: List<PowerBenchSummary>) {
        log("BENCH-TABLE-POWER combined")
        log("| model | preset | elapsed s | loops | battery drop % | drop %/h | energy Wh | avg power W | tok/s | note |")
        log("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        summaries.forEach { summary ->
            log(
                String.format(
                    Locale.US,
                    "| %s | %s | %.2f | %d | %s | %s | %s | %s | %.4f | %s |",
                    summary.modelFileName,
                    summary.presetLabel,
                    summary.elapsedMs / 1000.0,
                    summary.loops,
                    formatLogNullable(summary.batteryDeltaPercent),
                    formatLogNullable(summary.batteryDeltaPercentPerHour),
                    formatLogNullable(summary.estimatedEnergyWh),
                    formatLogNullable(summary.estimatedAveragePowerW),
                    summary.avgTokS,
                    summary.note,
                )
            )
        }
    }

    private fun ensureTypingJob(messageId: Long) {
        if (typingJob?.isActive == true) {
            return
        }

        typingJob = viewModelScope.launch {
            while (isGenerating || hasBufferedCharacters()) {
                val nextCharacter = popNextBufferedCharacter()
                if (nextCharacter != null) {
                    appendAssistantCharacter(messageId, nextCharacter)
                    delay(14)
                } else {
                    delay(8)
                }
            }
        }
    }

    private fun resetStreamingBuffer() {
        streamingBuffer.setLength(0)
    }

    private fun appendToStreamingBuffer(piece: String) {
        if (piece.isEmpty()) {
            return
        }
        streamingBuffer.append(piece)
    }

    private fun popNextBufferedCharacter(): String? {
        if (streamingBuffer.isEmpty()) {
            return null
        }

        val codePoint = Character.codePointAt(streamingBuffer, 0)
        val character = String(Character.toChars(codePoint))
        streamingBuffer.delete(0, Character.charCount(codePoint))
        return character
    }

    private fun hasBufferedCharacters(): Boolean {
        return streamingBuffer.isNotEmpty()
    }

    private fun flushStreamingBuffer(messageId: Long) {
        val remainder = if (streamingBuffer.isEmpty()) {
            ""
        } else {
            val pending = streamingBuffer.toString()
            streamingBuffer.setLength(0)
            pending
        }

        if (remainder.isNotEmpty()) {
            appendAssistantText(messageId, remainder)
        }
    }

    private fun appendAssistantCharacter(messageId: Long, character: String) {
        appendAssistantText(messageId, character)
    }

    private fun appendAssistantText(messageId: Long, text: String) {
        val currentMessages = messages.toMutableList()
        val lastIndex = currentMessages.indexOfLast { it.id == messageId }
        if (lastIndex < 0) {
            return
        }

        val current = currentMessages[lastIndex]
        currentMessages[lastIndex] = current.copy(text = current.text + text)
        messages = currentMessages
    }

    private fun logError(prefix: String, throwable: Throwable) {
        val message = throwable.message ?: throwable::class.java.simpleName
        Log.e(LOG_TAG, prefix, throwable)
        diagnostics = diagnostics + "$prefix: $message"
    }

    private fun Bundle.getIntOrSentinel(key: String): Int {
        return if (containsKey(key)) getInt(key) else -1
    }

    private fun Bundle.getIntOrSentinel(key: String, sentinel: Int): Int {
        return if (containsKey(key)) getInt(key) else sentinel
    }

    private fun Bundle.getLongOrNull(key: String): Long? {
        if (!containsKey(key)) {
            return null
        }
        return when (val value = get(key)) {
            is Long -> value
            is Int -> value.toLong()
            is String -> value.toLongOrNull()
            else -> null
        }
    }

    private fun Bundle.getDoubleOrNull(key: String): Double? {
        if (!containsKey(key)) {
            return null
        }
        return when (val value = get(key)) {
            is Double -> value
            is Float -> value.toDouble()
            is Long -> value.toDouble()
            is Int -> value.toDouble()
            is String -> value.toDoubleOrNull()
            else -> null
        }
    }

    private fun Intent.getValidIntExtra(name: String): Int? {
        val value = getIntExtra(name, Int.MIN_VALUE)
        return value.takeIf { it != Int.MIN_VALUE && it >= 0 }
    }

    private fun BatteryManager.getValidLongProperty(id: Int): Long? {
        return runCatching { getLongProperty(id) }
            .getOrNull()
            ?.takeIf { it > 0L && it != Long.MIN_VALUE }
    }

    private fun Long.toBatteryCapacityMah(): Double {
        // Android documents BATTERY_PROPERTY_CHARGE_COUNTER as microampere-hours,
        // but some vendor builds expose small mAh-like values. Preserve practical mAh in both cases.
        return if (this >= 100_000L) this / 1000.0 else this.toDouble()
    }

    private fun formatOverride(value: Int): String {
        return if (value >= 0) value.toString() else "default"
    }

    private fun formatOverride(value: Int, sentinel: Int): String {
        return if (value != sentinel) value.toString() else "default"
    }

    private fun resolvedGenerationLength(): Int {
        if (!useCustomGenerationLength) {
            return generationLength.maxTokens
        }

        val parsed = customGenerationLengthInput.toIntOrNull()
        require(parsed != null && parsed in 1..512) {
            "Custom reply length must be between 1 and 512"
        }
        return parsed
    }

    private fun maybeStartPendingAutomation() {
        when (pendingAutomationAction) {
            "builtin_bench",
            "builtin_e2e_bench",
            "builtin_power_bench" -> {
                val targets = if (pendingAutomationAction == "builtin_power_bench") {
                    powerBenchmarkTargetModels()
                } else {
                    benchmarkTargetModels()
                }
                if (targets.isNotEmpty()) {
                    log("Benchmark targets: ${targets.joinToString { it.fileName }}")
                    maybeRunPendingAutomation()
                } else {
                    log("No benchmark targets matched. Available models: ${availableModels.joinToString { it.fileName }}")
                }
            }
            "bench",
            "generate",
            "smoke" -> when (modelLoadState) {
                ModelLoadState.IMPORTED,
                ModelLoadState.LOADED -> maybeRunPendingAutomation()
                else -> Unit
            }
            else -> Unit
        }
    }

    private fun maybeRunPendingAutomation() {
        when (pendingAutomationAction) {
            "bench" -> {
                log("Running automation benchmark for selected model")
                if (modelLoadState != ModelLoadState.LOADED) {
                    loadImportedModel()
                } else {
                    pendingAutomationAction = null
                    val preset = benchmarkPresetOverride ?: BenchPreset("custom", 8, 4, 1, 1)
                    bench(preset.promptTokens, preset.genTokens, preset.pl, preset.repetitions)
                }
            }
            "builtin_bench" -> {
                pendingAutomationAction = null
                log("Running automation benchmark for bundled models")
                viewModelScope.launch { runBundledBenchmarks() }
            }
            "builtin_e2e_bench" -> {
                pendingAutomationAction = null
                log("Running automation E2E benchmark for bundled models")
                viewModelScope.launch { runBundledE2eBenchmarks() }
            }
            "builtin_power_bench" -> {
                pendingAutomationAction = null
                log("Running automation power benchmark")
                viewModelScope.launch { runBundledPowerBenchmarks() }
            }
            "generate" -> {
                pendingAutomationAction = null
                log("Running automation generate prompt")
                viewModelScope.launch { runAutomationGenerate() }
            }
            "smoke" -> {
                log("Running automation smoke prompt")
                if (modelLoadState != ModelLoadState.LOADED) {
                    loadImportedModel()
                } else {
                    pendingAutomationAction = null
                    updatePrompt("Hello")
                    send()
                }
            }
            else -> Unit
        }
    }

    private suspend fun copyModelToPrivateStorage(context: Context, uri: Uri): ImportedModel = withContext(Dispatchers.IO) {
        val resolver = context.contentResolver
        val metadata = resolver.queryMetadata(uri)
        val fileName = sanitizeFileName(metadata.first ?: "imported-model.gguf")
        val sizeBytes = metadata.second ?: -1L

        val destination = prepareModelDestination(context, fileName)
        resolver.openInputStream(uri)?.use { input ->
            destination.outputStream().use { output ->
                input.copyTo(output)
            }
        } ?: error("Unable to open input stream for $uri")

        ImportedModel(
            fileName = destination.name,
            sizeBytes = if (sizeBytes >= 0) sizeBytes else destination.length(),
            privatePath = destination.absolutePath,
        )
    }

    private suspend fun installBundledModels(
        context: Context,
        bundledModels: List<BundledModelSpec>,
    ): List<ImportedModel> = withContext(Dispatchers.IO) {
        bundledModels.map { spec ->
            val destination = prepareModelDestination(context, spec.fileName)
            if (!destination.exists()) {
                context.assets.open("$BUNDLED_MODEL_ASSET_DIR/${spec.fileName}").use { input ->
                    destination.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            }

            ImportedModel(
                fileName = destination.name,
                sizeBytes = destination.length(),
                privatePath = destination.absolutePath,
            )
        }
    }

    private fun refreshAvailableModels(context: Context, preferredFileName: String? = null) {
        val modelDirs = listOfNotNull(
            File(context.filesDir, "models"),
            context.getExternalFilesDir(null)?.let { File(it, "models") },
        )
        val seenPaths = mutableSetOf<String>()
        val models = modelDirs
            .flatMap { dir -> dir.listFiles()?.filter { it.isFile && it.name.endsWith(".gguf", ignoreCase = true) }.orEmpty() }
            .filter { seenPaths.add(it.absolutePath) }
            .sortedWith(compareBy({ bundledModelOrder(it.name) }, { it.name.lowercase(Locale.US) }))
            .map {
                ImportedModel(
                    fileName = it.name,
                    sizeBytes = it.length(),
                    privatePath = it.absolutePath,
                )
            }

        availableModels = models

        val selected = preferredFileName?.let { fileName ->
            models.firstOrNull { it.fileName == fileName }
        } ?: models.firstOrNull { it.fileName == importedModel?.fileName }
            ?: models.firstOrNull()

        importedModel = selected

        if (models.isEmpty()) {
            modelLoadState = ModelLoadState.NOT_IMPORTED
        } else if (modelLoadState == ModelLoadState.NOT_IMPORTED) {
            modelLoadState = ModelLoadState.IMPORTED
        }
    }

    private fun bundledAssetsAvailable(context: Context): List<BundledModelSpec> {
        val assets = context.assets.list(BUNDLED_MODEL_ASSET_DIR).orEmpty().toSet()
        return BUNDLED_MODELS.filter { it.fileName in assets }
    }

    private fun benchmarkTargetModels(): List<ImportedModel> {
        val filter = benchmarkModelFilter
        return if (filter.isNullOrEmpty()) {
            availableModels.filter { model ->
                BUNDLED_MODELS.any { it.fileName == model.fileName }
            }
        } else {
            availableModels.filter { it.fileName in filter }
        }
    }

    private fun benchmarkTargetPresets(): List<BenchPreset> {
        benchmarkPresetOverride?.let { return listOf(it) }
        val filter = benchmarkPresetFilter
        val presets = if (filter.isNullOrEmpty()) {
            BENCH_PRESETS
        } else {
            BENCH_PRESETS.filter { it.label in filter }
        }
        val repetitions = benchmarkRepetitionsOverride ?: return presets
        return presets.map { it.copy(repetitions = repetitions) }
    }

    private fun e2eBenchmarkTargetPresets(): List<BenchPreset> {
        val filter = benchmarkPresetFilter
        val e2ePresets = BENCH_PRESETS + E2E_EXTRA_PRESETS
        val presets = if (filter.isNullOrEmpty()) {
            BENCH_PRESETS
        } else {
            e2ePresets.filter { it.label in filter }
        }
        val repetitions = benchmarkRepetitionsOverride ?: return presets
        return presets.map { it.copy(repetitions = repetitions) }
    }

    private fun e2eBenchOutputFileName(summaries: List<E2eBenchSummary>): String {
        return if (summaries.isNotEmpty() && summaries.all { it.presetLabel == E2E_LONG_DECODE_PRESET_LABEL }) {
            E2E_LONG_DECODE_OUTPUT_FILE_NAME
        } else {
            E2E_BENCH_OUTPUT_FILE_NAME
        }
    }

    private fun e2eBenchStatusFileName(presets: List<BenchPreset>): String {
        return if (presets.isNotEmpty() && presets.all { it.label == E2E_LONG_DECODE_PRESET_LABEL }) {
            E2E_LONG_DECODE_STATUS_FILE_NAME
        } else {
            E2E_BENCH_STATUS_FILE_NAME
        }
    }

    private fun powerBenchmarkTargetModels(): List<ImportedModel> {
        val filter = benchmarkModelFilter
        return if (filter.isNullOrEmpty()) {
            importedModel?.let { listOf(it) }.orEmpty()
        } else {
            availableModels.filter { it.fileName in filter }
        }
    }

    private fun powerBenchmarkTargetPresets(): List<BenchPreset> {
        val presets = benchmarkTargetPresets()
        return if (benchmarkPresetFilter.isNullOrEmpty()) {
            presets.take(1)
        } else {
            presets
        }
    }

    private fun bundledModelOrder(fileName: String): Int {
        val index = BUNDLED_MODELS.indexOfFirst { it.fileName == fileName }
        return if (index >= 0) index else 1000
    }

    private suspend fun writeBenchCsv(context: Context, summaries: List<BenchSummary>) = withContext(Dispatchers.IO) {
        val benchDir = prepareBenchDirectory(context)
        val csvFile = File(benchDir, BENCH_OUTPUT_FILE_NAME)
        csvFile.writeText(
            buildString {
                appendLine("model,preset,prompt,decode,repetitions,pp_tps,tg_tps,prompt_latency_ms,decode_ms_per_token,decode_total_ms")
                summaries.forEach { summary ->
                    appendLine(
                        listOf(
                            summary.modelFileName,
                            summary.presetLabel,
                            summary.promptTokens.toString(),
                            summary.genTokens.toString(),
                            summary.repetitions.toString(),
                            formatCsvDecimal(summary.ppTps),
                            formatCsvDecimal(summary.tgTps),
                            formatCsvDecimal(summary.promptLatencyMs),
                            formatCsvDecimal(summary.decodeLatencyMsPerToken),
                            formatCsvDecimal(summary.decodeTotalMs),
                        ).joinToString(","),
                    )
                }
            },
        )
    }

    private suspend fun writeE2eBenchCsv(context: Context, summaries: List<E2eBenchSummary>) = withContext(Dispatchers.IO) {
        val benchDir = prepareBenchDirectory(context)
        val csvFile = File(benchDir, e2eBenchOutputFileName(summaries))
        csvFile.writeText(
            buildString {
                appendLine("model,preset,prompt,decode,repetitions,n_threads,n_threads_batch,affinity_profile,disable_ifairy_lut,ifairy_vecdot_act_tensor,generated_tokens_avg,prefill_ms,first_token_ms,decode_total_ms,decode_ms_per_token,tok_s,total_ms,prefill_ratio,status,note")
                summaries.forEach { summary ->
                    val decodeTotalMs = (summary.totalMs - summary.firstTokenMs).coerceAtLeast(0.0)
                    val prefillRatio = if (summary.totalMs > 0.0) summary.prefillMs / summary.totalMs else 0.0
                    val status = if (summary.generatedTokensAvg + 0.5 >= summary.genTokens) "completed" else "early_eog"
                    val note = if (status == "early_eog") "generated_tokens_less_than_target" else ""
                    appendLine(
                        listOf(
                            summary.modelFileName,
                            summary.presetLabel,
                            summary.promptTokens.toString(),
                            summary.genTokens.toString(),
                            summary.repetitions.toString(),
                            formatOverride(summary.nThreads),
                            formatOverride(summary.nThreadsBatch),
                            formatOverride(summary.affinityProfile),
                            formatOverride(summary.disableIFairyLut),
                            formatOverride(summary.enableIFairyVecdotActTensor),
                            formatCsvDecimal(summary.generatedTokensAvg),
                            formatCsvDecimal(summary.prefillMs),
                            formatCsvDecimal(summary.firstTokenMs),
                            formatCsvDecimal(decodeTotalMs),
                            formatCsvDecimal(summary.decodeLatencyMsPerToken),
                            formatCsvDecimal(summary.tokS),
                            formatCsvDecimal(summary.totalMs),
                            formatCsvDecimal(prefillRatio),
                            status,
                            note,
                        ).joinToString(","),
                    )
                }
            },
        )
    }

    private suspend fun writePowerBenchCsv(context: Context, summaries: List<PowerBenchSummary>) = withContext(Dispatchers.IO) {
        val benchDir = prepareBenchDirectory(context)
        File(benchDir, POWER_BENCH_OUTPUT_FILE_NAME).writeText(buildPowerBenchCsv(summaries))

        summaries.groupBy { it.modelFileName }.forEach { (modelFileName, modelSummaries) ->
            val outputFileName = powerBenchOutputFileNames.getOrPut(modelFileName) {
                uniquePowerBenchOutputFileName(benchDir, modelFileName)
            }
            File(benchDir, outputFileName).writeText(buildPowerBenchCsv(modelSummaries))
        }
    }

    private fun buildPowerBenchCsv(summaries: List<PowerBenchSummary>): String {
        return buildString {
            appendLine(
                "model,preset,prompt,decode,repetitions,target_duration_ms,elapsed_ms,loops,generated_tokens_total," +
                    "refresh_count,refresh_total_ms,refresh_avg_ms," +
                    "start_battery_percent,end_battery_percent,battery_delta_percent,battery_delta_percent_per_hour," +
                    "start_level,end_level,scale,start_voltage_mv,end_voltage_mv,start_temperature_c,end_temperature_c," +
                    "start_plugged,end_plugged,start_status,end_status,start_charge_uah,end_charge_uah,charge_delta_uah," +
                    "start_remaining_battery_mah,end_remaining_battery_mah,remaining_battery_delta_mah," +
                    "start_energy_nwh,end_energy_nwh,energy_delta_wh,estimated_energy_wh,estimated_avg_power_w," +
                    "avg_prefill_ms,avg_first_token_ms,avg_decode_ms_per_token,avg_tok_s,avg_total_ms,note"
            )
            summaries.forEach { summary ->
                appendLine(
                    listOf(
                        summary.modelFileName,
                        summary.presetLabel,
                        summary.promptTokens.toString(),
                        summary.genTokens.toString(),
                        summary.repetitions.toString(),
                        summary.targetDurationMs.toString(),
                        summary.elapsedMs.toString(),
                        summary.loops.toString(),
                        formatCsvDecimal(summary.generatedTokensTotal),
                        summary.refreshCount.toString(),
                        summary.refreshTotalMs.toString(),
                        formatCsvNullable(summary.refreshTotalMs.takeIf { summary.refreshCount > 0 }?.toDouble()?.div(summary.refreshCount)),
                        formatCsvNullable(summary.startBattery.percent),
                        formatCsvNullable(summary.endBattery.percent),
                        formatCsvNullable(summary.batteryDeltaPercent),
                        formatCsvNullable(summary.batteryDeltaPercentPerHour),
                        formatCsvNullable(summary.startBattery.level),
                        formatCsvNullable(summary.endBattery.level),
                        formatCsvNullable(summary.startBattery.scale),
                        formatCsvNullable(summary.startBattery.voltageMv),
                        formatCsvNullable(summary.endBattery.voltageMv),
                        formatCsvNullable(summary.startBattery.temperatureC),
                        formatCsvNullable(summary.endBattery.temperatureC),
                        formatCsvNullable(summary.startBattery.plugged),
                        formatCsvNullable(summary.endBattery.plugged),
                        formatCsvNullable(summary.startBattery.status),
                        formatCsvNullable(summary.endBattery.status),
                        formatCsvNullable(summary.startBattery.chargeCounterUah),
                        formatCsvNullable(summary.endBattery.chargeCounterUah),
                        formatCsvNullable(summary.chargeDeltaUah),
                        formatCsvNullable(summary.startRemainingCapacityMah),
                        formatCsvNullable(summary.endRemainingCapacityMah),
                        formatCsvNullable(summary.remainingCapacityDeltaMah),
                        formatCsvNullable(summary.startBattery.energyCounterNwh),
                        formatCsvNullable(summary.endBattery.energyCounterNwh),
                        formatCsvNullable(summary.energyDeltaWh),
                        formatCsvNullable(summary.estimatedEnergyWh),
                        formatCsvNullable(summary.estimatedAveragePowerW),
                        formatCsvDecimal(summary.avgPrefillMs),
                        formatCsvDecimal(summary.avgFirstTokenMs),
                        formatCsvDecimal(summary.avgDecodeMsPerToken),
                        formatCsvDecimal(summary.avgTokS),
                        formatCsvDecimal(summary.avgTotalMs),
                        summary.note,
                    ).joinToString(","),
                )
            }
        }
    }

    private fun uniquePowerBenchOutputFileName(benchDir: File, modelFileName: String): String {
        val modelStem = sanitizeFileName(modelFileName.removeSuffix(".gguf"))
        val baseName = POWER_BENCH_OUTPUT_FILE_NAME.removeSuffix(".csv")
        val candidate = "${baseName}_${modelStem}.csv"
        if (!File(benchDir, candidate).exists()) {
            return candidate
        }

        var index = 2
        while (true) {
            val indexedCandidate = "${baseName}_${modelStem}_${index}.csv"
            if (!File(benchDir, indexedCandidate).exists()) {
                return indexedCandidate
            }
            index += 1
        }
    }

    private suspend fun writePowerProgressStatus(
        context: Context,
        model: ImportedModel,
        preset: BenchPreset,
        loops: Int,
        refreshCount: Int,
        elapsedMs: Long,
    ) = writeBenchStatus(
        context,
        POWER_BENCH_STATUS_FILE_NAME,
        buildString {
            appendLine("running")
            appendLine("model=${model.fileName}")
            appendLine("preset=${preset.label}")
            appendLine("loops=$loops")
            appendLine("refresh_count=$refreshCount")
            appendLine("elapsed_ms=$elapsedMs")
            appendLine("target_duration_ms=$powerBenchDurationMs")
        },
    )

    private suspend fun writeBenchStatus(context: Context, fileName: String, status: String) = withContext(Dispatchers.IO) {
        val benchDir = prepareBenchDirectory(context)
        File(benchDir, fileName).writeText(status)
    }

    private suspend fun writeGenerateOutput(
        context: Context,
        fileName: String,
        model: ImportedModel,
        elapsedMs: Long,
        generatedText: String,
    ) = withContext(Dispatchers.IO) {
        val benchDir = prepareBenchDirectory(context)
        File(benchDir, fileName).writeText(
            buildString {
                appendLine("model=${model.fileName}")
                appendLine("prompt=$automationGeneratePrompt")
                appendLine("max_tokens=$automationGenerateMaxTokens")
                appendLine("format_chat=$automationGenerateFormatChat")
                appendLine("force_cpu=${runtimeBenchConfig.forceCpu}")
                appendLine("elapsed_ms=$elapsedMs")
                appendLine("generated_chars=${generatedText.length}")
                appendLine("output_begin")
                append(generatedText)
                if (!generatedText.endsWith('\n')) {
                    appendLine()
                }
                appendLine("output_end")
            },
        )
    }

    private fun prepareBenchDirectory(context: Context): File {
        val root = requireNotNull(context.getExternalFilesDir(null)) { "External files dir unavailable" }
        val benchDir = File(root, BENCH_OUTPUT_DIR)
        if (!benchDir.exists() && !benchDir.mkdirs()) {
            error("Failed to create ${benchDir.absolutePath}")
        }
        return benchDir
    }

    private fun ContentResolver.queryMetadata(uri: Uri): Pair<String?, Long?> {
        query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val name = cursor.getStringOrNull(OpenableColumns.DISPLAY_NAME)
                val size = cursor.getLongOrNull(OpenableColumns.SIZE)
                return Pair(name, size)
            }
        }
        return null to null
    }

    private fun Cursor.getStringOrNull(columnName: String): String? {
        val index = getColumnIndex(columnName)
        return if (index >= 0 && !isNull(index)) getString(index) else null
    }

    private fun Cursor.getLongOrNull(columnName: String): Long? {
        val index = getColumnIndex(columnName)
        return if (index >= 0 && !isNull(index)) getLong(index) else null
    }

    private fun sanitizeFileName(fileName: String): String {
        return fileName.replace(SAFE_FILE_NAME, "_")
    }

    private fun prepareModelDestination(context: Context, fileName: String): File {
        val modelsDir = File(context.filesDir, "models")
        if (!modelsDir.exists() && !modelsDir.mkdirs()) {
            error("Failed to create ${modelsDir.absolutePath}")
        }
        return File(modelsDir, fileName)
    }

    private fun formatSize(sizeBytes: Long): String {
        if (sizeBytes < 0) {
            return "unknown"
        }
        val mib = sizeBytes / (1024.0 * 1024.0)
        return String.format(Locale.US, "%.2f MiB", mib)
    }

    private fun formatCsvDecimal(value: Double): String {
        return String.format(Locale.US, "%.4f", value)
    }

    private fun formatCsvNullable(value: Double?): String {
        return value?.let(::formatCsvDecimal).orEmpty()
    }

    private fun formatCsvNullable(value: Int?): String {
        return value?.toString().orEmpty()
    }

    private fun formatCsvNullable(value: Long?): String {
        return value?.toString().orEmpty()
    }

    private fun formatLogNullable(value: Double?): String {
        return value?.let { String.format(Locale.US, "%.4f", it) } ?: "n/a"
    }
}
