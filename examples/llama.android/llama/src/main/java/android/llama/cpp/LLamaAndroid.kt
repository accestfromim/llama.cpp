package android.llama.cpp

import android.os.Debug
import android.util.Log
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.concurrent.thread

class LLamaAndroid {
    companion object {
        private const val LOG_TAG = "LLAMA_ANDROID"

        data class E2eBenchResult(
            val generatedTokensAvg: Double,
            val prefillMs: Double,
            val firstTokenMs: Double,
            val decodeMsPerToken: Double,
            val tokS: Double,
            val totalMs: Double,
        )

        private class IntVar(value: Int) {
            @Volatile
            var value: Int = value
                private set

            fun inc() {
                synchronized(this) {
                    value += 1
                }
            }
        }

        private sealed interface State {
            data object Idle : State
            data class Loaded(
                val path: String,
                val model: Long,
                val context: Long,
                val batch: Long,
                val sampler: Long,
            ) : State
        }

        private val instance: LLamaAndroid = LLamaAndroid()

        fun instance(): LLamaAndroid = instance
    }

    private val runLoop: CoroutineDispatcher = Executors.newSingleThreadExecutor {
        thread(start = false, name = "Llm-RunLoop") {
            Log.d(LOG_TAG, "Dedicated thread for native code: ${Thread.currentThread().name}")

            System.loadLibrary("llama-android")
            log_to_android()
            backend_init()

            Log.i(LOG_TAG, system_info())

            it.run()
        }.apply {
            uncaughtExceptionHandler = Thread.UncaughtExceptionHandler { _, exception: Throwable ->
                Log.e(LOG_TAG, "Unhandled exception", exception)
            }
        }
    }.asCoroutineDispatcher()

    private var state: State = State.Idle

    @Volatile
    private var stopRequested: Boolean = false

    @Volatile
    private var isGenerating: Boolean = false

    private external fun log_to_android()
    private external fun log_native_memory(stage: String)
    private external fun load_model(filename: String): Long
    private external fun free_model(model: Long)
    private external fun new_context(model: Long): Long
    private external fun free_context(context: Long)
    private external fun backend_init()
    private external fun backend_free()
    private external fun new_batch(nTokens: Int, embd: Int, nSeqMax: Int): Long
    private external fun free_batch(batch: Long)
    private external fun new_sampler(): Long
    private external fun free_sampler(sampler: Long)
    private external fun configure_runtime(
        nCtx: Int,
        nBatch: Int,
        nUbatch: Int,
        nThreads: Int,
        nThreadsBatch: Int,
        disableIFairyLut: Int,
        affinityProfile: Int,
        enableIFairyVecdotActTensor: Int,
        generationPriority: Int,
        batchPriority: Int,
    )
    private external fun bench_model(
        context: Long,
        model: Long,
        batch: Long,
        pp: Int,
        tg: Int,
        pl: Int,
        nr: Int
    ): String
    private external fun e2e_bench_model(
        context: Long,
        batch: Long,
        sampler: Long,
        pp: Int,
        tg: Int,
        nr: Int
    ): DoubleArray

    private external fun system_info(): String

    private external fun completion_init(
        context: Long,
        batch: Long,
        text: String,
        formatChat: Boolean,
        nLen: Int
    ): Int

    private external fun completion_loop(
        context: Long,
        batch: Long,
        sampler: Long,
        nLen: Int,
        ncur: IntVar
    ): String?

    private external fun kv_cache_clear(context: Long)

    suspend fun configureRuntime(
        nCtx: Int = -1,
        nBatch: Int = -1,
        nUbatch: Int = -1,
        nThreads: Int = -1,
        nThreadsBatch: Int = -1,
        disableIFairyLut: Int = -1,
        affinityProfile: Int = -1,
        enableIFairyVecdotActTensor: Int = -1,
        generationPriority: Int = -99,
        batchPriority: Int = -99,
    ) {
        withContext(runLoop) {
            configure_runtime(
                nCtx,
                nBatch,
                nUbatch,
                nThreads,
                nThreadsBatch,
                disableIFairyLut,
                affinityProfile,
                enableIFairyVecdotActTensor,
                generationPriority,
                batchPriority,
            )
        }
    }

    suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1): String {
        return withContext(runLoop) {
            when (val current = state) {
                is State.Loaded -> bench_model(current.context, current.model, current.batch, pp, tg, pl, nr)
                State.Idle -> throw IllegalStateException("No model loaded")
            }
        }
    }

    suspend fun e2eBench(pp: Int, tg: Int, nr: Int = 1): E2eBenchResult {
        return withContext(runLoop) {
            when (val current = state) {
                is State.Loaded -> {
                    val values = e2e_bench_model(current.context, current.batch, current.sampler, pp, tg, nr)
                    require(values.size >= 6) { "Unexpected e2e_bench_model() result length: ${values.size}" }
                    E2eBenchResult(
                        generatedTokensAvg = values[0],
                        prefillMs = values[1],
                        firstTokenMs = values[2],
                        decodeMsPerToken = values[3],
                        tokS = values[4],
                        totalMs = values[5],
                    )
                }
                State.Idle -> throw IllegalStateException("No model loaded")
            }
        }
    }

    suspend fun load(pathToModel: String) {
        withContext(runLoop) {
            when (state) {
                is State.Loaded -> unloadLocked()
                State.Idle -> Unit
            }

            logMemorySnapshot("before-load-model")
            val model = load_model(pathToModel)
            if (model == 0L) {
                throw IllegalStateException("load_model() failed")
            }

            logMemorySnapshot("after-load-model")
            val context = new_context(model)
            if (context == 0L) {
                free_model(model)
                throw IllegalStateException("new_context() failed")
            }

            logMemorySnapshot("after-new-context")
            val batch = new_batch(2048, 0, 1)
            if (batch == 0L) {
                free_context(context)
                free_model(model)
                throw IllegalStateException("new_batch() failed")
            }

            logMemorySnapshot("after-new-batch")
            val sampler = new_sampler()
            if (sampler == 0L) {
                free_batch(batch)
                free_context(context)
                free_model(model)
                throw IllegalStateException("new_sampler() failed")
            }

            logMemorySnapshot("after-new-sampler")
            Log.i(LOG_TAG, "Loaded model from $pathToModel")
            state = State.Loaded(pathToModel, model, context, batch, sampler)
        }
    }

    fun send(message: String, maxTokens: Int, formatChat: Boolean = false): Flow<String> = flow {
        val current = when (val snapshot = state) {
            is State.Loaded -> snapshot
            State.Idle -> throw IllegalStateException("No model loaded")
        }

        check(!isGenerating) { "Generation already in progress" }

        stopRequested = false
        isGenerating = true

        try {
            Log.i(LOG_TAG, "Starting generation")
            logMemorySnapshot("before-completion-init")

            val promptTokens = completion_init(current.context, current.batch, message, formatChat, maxTokens)
            val totalBudget = promptTokens + maxTokens
            val ncur = IntVar(promptTokens)
            logMemorySnapshot("after-completion-init")
            var emittedAnyToken = false

            while (ncur.value <= totalBudget) {
                currentCoroutineContext().ensureActive()

                if (stopRequested) {
                    Log.i(LOG_TAG, "Stop requested before sampling next token")
                    break
                }

                val piece = completion_loop(current.context, current.batch, current.sampler, totalBudget, ncur) ?: break
                if (piece.isNotEmpty() && !emittedAnyToken) {
                    emittedAnyToken = true
                    Log.i(LOG_TAG, "Received first token")
                }
                emit(piece)
            }

            if (stopRequested) {
                Log.i(LOG_TAG, "Generation stopped")
            } else {
                Log.i(LOG_TAG, "Generation completed")
            }
        } finally {
            kv_cache_clear(current.context)
            logMemorySnapshot("after-kv-cache-clear")
            isGenerating = false
            stopRequested = false
        }
    }.flowOn(runLoop)

    fun stop() {
        stopRequested = true
    }

    fun isGenerating(): Boolean = isGenerating

    suspend fun unload() {
        withContext(runLoop) {
            unloadLocked()
        }
    }

    suspend fun refreshBackend() {
        withContext(runLoop) {
            unloadLocked()
            Log.i(LOG_TAG, "Refreshing llama backend")
            backend_free()
            backend_init()
            Log.i(LOG_TAG, "Llama backend refreshed")
        }
    }

    private fun unloadLocked() {
        when (val current = state) {
            is State.Loaded -> {
                free_context(current.context)
                free_model(current.model)
                free_batch(current.batch)
                free_sampler(current.sampler)
                state = State.Idle
                stopRequested = false
                isGenerating = false
                logMemorySnapshot("after-unload")
                Log.i(LOG_TAG, "Unloaded model ${current.path}")
            }
            State.Idle -> Unit
        }
    }

    private fun logMemorySnapshot(stage: String) {
        val runtime = Runtime.getRuntime()
        val usedBytes = runtime.totalMemory() - runtime.freeMemory()
        val totalBytes = runtime.totalMemory()
        val maxBytes = runtime.maxMemory()
        val nativeHeapBytes = Debug.getNativeHeapAllocatedSize()

        Log.i(
            LOG_TAG,
            String.format(
                Locale.US,
                "MEM-JAVA[%s] managed_used=%.2f MiB managed_total=%.2f MiB managed_max=%.2f MiB native_heap=%.2f MiB",
                stage,
                bytesToMiB(usedBytes),
                bytesToMiB(totalBytes),
                bytesToMiB(maxBytes),
                bytesToMiB(nativeHeapBytes),
            )
        )

        log_native_memory(stage)
    }

    private fun bytesToMiB(bytes: Long): Double {
        return bytes.toDouble() / (1024.0 * 1024.0)
    }

}
