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

class MainViewModel(
    private val llamaAndroid: LLamaAndroid = LLamaAndroid.instance(),
) : ViewModel() {
    companion object {
        private const val LOG_TAG = "LLAMA_ANDROID"
        private const val NANOS_PER_SECOND = 1_000_000_000.0
        private const val BUNDLED_MODEL_ASSET_DIR = "models"
        private const val BUNDLED_MODEL_FILE_NAME = "ifairy.gguf"
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
        val existing = findLatestImportedModel(appContext)
        if (existing != null) {
            importedModel = existing
            modelLoadState = ModelLoadState.IMPORTED
            log("Discovered existing model: ${existing.fileName}")
            log("Model path: ${existing.privatePath}")
            log("Model size: ${formatSize(existing.sizeBytes)}")
            maybeStartPendingAutomation()
            return
        }

        if (!hasBundledModelAsset(appContext)) {
            return
        }

        modelLoadState = ModelLoadState.IMPORTING
        modelError = null
        log("Installing bundled model: $BUNDLED_MODEL_FILE_NAME")

        viewModelScope.launch {
            runCatching {
                copyBundledModelToPrivateStorage(appContext)
            }.onSuccess { imported ->
                importedModel = imported
                modelLoadState = ModelLoadState.IMPORTED
                log("Installed bundled model: ${imported.fileName}")
                log("Model path: ${imported.privatePath}")
                log("Model size: ${formatSize(imported.sizeBytes)}")
                maybeStartPendingAutomation()
            }.onFailure { throwable ->
                modelLoadState = ModelLoadState.FAILED
                modelError = throwable.message ?: "Bundled model install failed"
                logError("installBundledModel() failed", throwable)
            }
        }
    }

    fun requestAutomationAction(action: String?) {
        pendingAutomationAction = action?.trim()?.lowercase()?.takeIf { it.isNotEmpty() }
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

        if (listOf(
                nCtx,
                nBatch,
                nUbatch,
                nThreads,
                nThreadsBatch,
                disableIFairyLut,
                affinityProfile,
                enableIFairyVecdotActTensor,
            ).all { it < 0 } && generationPriority == -99 && batchPriority == -99) {
            return
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
                        "generation_priority=${formatOverride(generationPriority, -99)}, " +
                        "batch_priority=${formatOverride(batchPriority, -99)}"
                )
            }.onFailure { throwable ->
                logError("configureRuntime() failed", throwable)
            }
        }
    }

    fun importModel(context: Context, uri: Uri) {
        if (modelLoadState == ModelLoadState.IMPORTING) {
            return
        }

        modelLoadState = ModelLoadState.IMPORTING
        modelError = null
        log("Importing model from $uri")

        viewModelScope.launch {
            runCatching {
                copyModelToPrivateStorage(context.applicationContext, uri)
            }.onSuccess { imported ->
                importedModel = imported
                modelLoadState = ModelLoadState.IMPORTED
                log("Imported ${imported.fileName}")
                log("Model path: ${imported.privatePath}")
                log("Model size: ${formatSize(imported.sizeBytes)}")
            }.onFailure { throwable ->
                modelLoadState = ModelLoadState.FAILED
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

        if (isGenerating) {
            modelError = "Stop generation before loading another model"
            modelLoadState = ModelLoadState.FAILED
            return
        }

        modelLoadState = ModelLoadState.LOADING
        modelError = null
        log("Loading model from ${model.privatePath}")

        viewModelScope.launch {
            runCatching {
                llamaAndroid.load(model.privatePath)
            }.onSuccess {
                modelLoadState = ModelLoadState.LOADED
                log("Model loaded: ${model.fileName}")
                maybeRunPendingAutomation()
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

        viewModelScope.launch {
            runCatching { llamaAndroid.unload() }
                .onSuccess {
                    modelLoadState = if (importedModel == null) ModelLoadState.NOT_IMPORTED else ModelLoadState.IMPORTED
                    log("Model unloaded")
                }
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
        if (isGenerating || llamaAndroid.isGenerating()) {
            modelError = "Generation already in progress"
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

        viewModelScope.launch {
            runCatching {
                val start = System.nanoTime()
                val warmupResult = llamaAndroid.bench(pp, tg, pl, nr)
                val end = System.nanoTime()

                log(warmupResult)

                val warmupSeconds = (end - start).toDouble() / NANOS_PER_SECOND
                log("Warm up time: $warmupSeconds seconds")

                if (warmupSeconds > 5.0) {
                    log("Warm up took too long, skipping long benchmark")
                } else {
                    log(llamaAndroid.bench(512, 128, 1, 3))
                }
            }.onFailure { logError("bench() failed", it) }
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
            "bench" -> when (modelLoadState) {
                ModelLoadState.IMPORTED -> loadImportedModel()
                ModelLoadState.LOADED -> maybeRunPendingAutomation()
                else -> Unit
            }
            "smoke" -> when (modelLoadState) {
                ModelLoadState.IMPORTED -> loadImportedModel()
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
                log("Running automation benchmark")
                bench(8, 4, 1)
            }
            "smoke" -> {
                pendingAutomationAction = null
                log("Running automation smoke prompt")
                updatePrompt("Hello")
                send()
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

    private suspend fun copyBundledModelToPrivateStorage(context: Context): ImportedModel = withContext(Dispatchers.IO) {
        val destination = prepareModelDestination(context, BUNDLED_MODEL_FILE_NAME)
        context.assets.open("$BUNDLED_MODEL_ASSET_DIR/$BUNDLED_MODEL_FILE_NAME").use { input ->
            destination.outputStream().use { output ->
                input.copyTo(output)
            }
        }

        ImportedModel(
            fileName = destination.name,
            sizeBytes = destination.length(),
            privatePath = destination.absolutePath,
        )
    }

    private fun findLatestImportedModel(context: Context): ImportedModel? {
        val modelsDir = File(context.filesDir, "models")
        val latest = modelsDir.listFiles()
            ?.filter { it.isFile }
            ?.maxByOrNull { it.lastModified() }
            ?: return null

        return ImportedModel(
            fileName = latest.name,
            sizeBytes = latest.length(),
            privatePath = latest.absolutePath,
        )
    }

    private fun hasBundledModelAsset(context: Context): Boolean {
        return context.assets.list(BUNDLED_MODEL_ASSET_DIR).orEmpty().contains(BUNDLED_MODEL_FILE_NAME)
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
        return String.format("%.2f MiB", mib)
    }
}
