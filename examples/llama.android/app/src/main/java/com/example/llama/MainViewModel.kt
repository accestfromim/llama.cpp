package com.example.llama

import android.content.ContentResolver
import android.content.Context
import android.database.Cursor
import android.llama.cpp.LLamaAndroid
import android.net.Uri
import android.os.Bundle
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
    val generatedTokensAvg: Double,
    val prefillMs: Double,
    val firstTokenMs: Double,
    val decodeLatencyMsPerToken: Double,
    val tokS: Double,
    val totalMs: Double,
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

    private var generationJob: Job? = null
    private var typingJob: Job? = null
    private var pendingAutomationAction: String? = null
    private var initializationStarted = false
    private var nextMessageId = 1L
    private val streamingBuffer = StringBuilder()
    private var availableModels: List<ImportedModel> = listOf()
    private var appContext: Context? = null
    private var bundledModelCooldownMs: Long = DEFAULT_BUNDLED_MODEL_COOLDOWN_MS

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
        val benchCooldownMs = extras.getIntOrSentinel("codex_bench_cooldown_ms")

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
                benchCooldownMs,
            ).all { it < 0 } &&
            generationPriority == -99 &&
            batchPriority == -99
        ) {
            return
        }

        if (benchCooldownMs >= 0) {
            bundledModelCooldownMs = benchCooldownMs.toLong()
            log("Benchmark cooldown override: ${bundledModelCooldownMs} ms")
        }

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
                )
            }.onSuccess {
                log(
                    "Runtime overrides: n_ctx=${formatOverride(nCtx)}, n_batch=${formatOverride(nBatch)}, " +
                        "n_ubatch=${formatOverride(nUbatch)}, n_threads=${formatOverride(nThreads)}, " +
                        "n_threads_batch=${formatOverride(nThreadsBatch)}, disable_ifairy_lut=${formatOverride(disableIFairyLut)}, " +
                        "affinity_profile=${formatOverride(affinityProfile)}, " +
                        "ifairy_vecdot_act_tensor=${formatOverride(enableIFairyVecdotActTensor)}, " +
                        "bench_cooldown_ms=${formatOverride(benchCooldownMs)}, " +
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

        val bundledModels = availableModels.filter { model ->
            BUNDLED_MODELS.any { it.fileName == model.fileName }
        }
        if (bundledModels.isEmpty()) {
            modelError = "Bundled models are not installed"
            return
        }

        isBenchmarking = true
        try {
            val context = requireNotNull(appContext) { "App context is not initialized" }
            val summaries = mutableListOf<BenchSummary>()
            writeBenchStatus(context, BENCH_STATUS_FILE_NAME, "running")
            log("Starting bundled model benchmark sweep")
            for ((index, model) in bundledModels.withIndex()) {
                loadModelInternal(model)
                warmupBenchIfNeeded()
                BENCH_PRESETS.forEach { preset ->
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

    private suspend fun runBundledE2eBenchmarks() {
        if (isGenerating || isBenchmarking) {
            modelError = "Another benchmark or generation is in progress"
            return
        }

        val bundledModels = availableModels.filter { model ->
            BUNDLED_MODELS.any { it.fileName == model.fileName }
        }
        if (bundledModels.isEmpty()) {
            modelError = "Bundled models are not installed"
            return
        }

        isBenchmarking = true
        try {
            val context = requireNotNull(appContext) { "App context is not initialized" }
            val summaries = mutableListOf<E2eBenchSummary>()
            writeBenchStatus(context, E2E_BENCH_STATUS_FILE_NAME, "running")
            log("Starting bundled model E2E benchmark sweep")
            for ((index, model) in bundledModels.withIndex()) {
                loadModelInternal(model)
                warmupE2eBenchIfNeeded()
                BENCH_PRESETS.forEach { preset ->
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
            writeBenchStatus(context, E2E_BENCH_STATUS_FILE_NAME, "completed")
            log("Completed bundled model E2E benchmark sweep")
        } catch (throwable: Throwable) {
            appContext?.let { context ->
                runCatching {
                    writeBenchStatus(
                        context,
                        E2E_BENCH_STATUS_FILE_NAME,
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
        log("Warmup benchmark: prompt=8 decode=4")
        runCatching { llamaAndroid.bench(pp = 8, tg = 4, pl = 1, nr = 1) }
            .onFailure { logError("warmupBenchIfNeeded() failed", it) }
    }

    private suspend fun runE2eBenchPreset(model: ImportedModel, preset: BenchPreset): E2eBenchSummary {
        log(
            "Running E2E benchmark: model=${model.fileName}, preset=${preset.label}, " +
                "prompt=${preset.promptTokens}, decode=${preset.genTokens}, repetitions=${preset.repetitions}"
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
            generatedTokensAvg = result.generatedTokensAvg,
            prefillMs = result.prefillMs,
            firstTokenMs = result.firstTokenMs,
            decodeLatencyMsPerToken = result.decodeMsPerToken,
            tokS = result.tokS,
            totalMs = result.totalMs,
        )
    }

    private suspend fun warmupE2eBenchIfNeeded() {
        log("Warmup E2E benchmark: prompt=8 decode=4")
        runCatching { llamaAndroid.e2eBench(pp = 8, tg = 4, nr = 1) }
            .onFailure { logError("warmupE2eBenchIfNeeded() failed", it) }
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
            "bench",
            "smoke",
            "builtin_bench",
            "builtin_e2e_bench" -> when (modelLoadState) {
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
                pendingAutomationAction = null
                log("Running automation benchmark for selected model")
                if (modelLoadState != ModelLoadState.LOADED) {
                    loadImportedModel()
                } else {
                    bench(8, 4, 1)
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
            "smoke" -> {
                pendingAutomationAction = null
                log("Running automation smoke prompt")
                if (modelLoadState != ModelLoadState.LOADED) {
                    loadImportedModel()
                } else {
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
        val modelsDir = File(context.filesDir, "models")
        val models = modelsDir.listFiles()
            ?.filter { it.isFile }
            ?.sortedWith(compareBy({ bundledModelOrder(it.name) }, { it.name.lowercase(Locale.US) }))
            ?.map {
                ImportedModel(
                    fileName = it.name,
                    sizeBytes = it.length(),
                    privatePath = it.absolutePath,
                )
            }
            .orEmpty()

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

    private fun bundledModelOrder(fileName: String): Int {
        val index = BUNDLED_MODELS.indexOfFirst { it.fileName == fileName }
        return if (index >= 0) index else Int.MAX_VALUE
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
                            BENCH_PRESETS.first { it.label == summary.presetLabel }.repetitions.toString(),
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
        val csvFile = File(benchDir, E2E_BENCH_OUTPUT_FILE_NAME)
        csvFile.writeText(
            buildString {
                appendLine("model,preset,prompt,decode,repetitions,generated_tokens_avg,prefill_ms,first_token_ms,decode_ms_per_token,tok_s,total_ms")
                summaries.forEach { summary ->
                    appendLine(
                        listOf(
                            summary.modelFileName,
                            summary.presetLabel,
                            summary.promptTokens.toString(),
                            summary.genTokens.toString(),
                            summary.repetitions.toString(),
                            formatCsvDecimal(summary.generatedTokensAvg),
                            formatCsvDecimal(summary.prefillMs),
                            formatCsvDecimal(summary.firstTokenMs),
                            formatCsvDecimal(summary.decodeLatencyMsPerToken),
                            formatCsvDecimal(summary.tokS),
                            formatCsvDecimal(summary.totalMs),
                        ).joinToString(","),
                    )
                }
            },
        )
    }

    private suspend fun writeBenchStatus(context: Context, fileName: String, status: String) = withContext(Dispatchers.IO) {
        val benchDir = prepareBenchDirectory(context)
        File(benchDir, fileName).writeText(status)
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
}
