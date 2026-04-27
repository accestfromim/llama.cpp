package com.example.llama

import android.app.ActivityManager
import android.content.ClipData
import android.content.ClipboardManager
import android.os.Bundle
import android.text.format.Formatter
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.core.content.getSystemService
import com.example.llama.ui.theme.LlamaAndroidTheme

class MainActivity(
    activityManager: ActivityManager? = null,
    clipboardManager: ClipboardManager? = null,
) : ComponentActivity() {
    private val activityManager by lazy { activityManager ?: getSystemService<ActivityManager>()!! }
    private val clipboardManager by lazy { clipboardManager ?: getSystemService<ClipboardManager>()!! }
    private val viewModel: MainViewModel by viewModels()

    private fun availableMemory(): ActivityManager.MemoryInfo {
        return ActivityManager.MemoryInfo().also { memoryInfo ->
            activityManager.getMemoryInfo(memoryInfo)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val free = Formatter.formatFileSize(this, availableMemory().availMem)
        val total = Formatter.formatFileSize(this, availableMemory().totalMem)

        viewModel.log("Current memory: $free / $total")
        viewModel.log("App filesDir: $filesDir")
        viewModel.log("Model import dir: ${filesDir.resolve("models").absolutePath}")
        viewModel.applyRuntimeOverrides(intent?.extras)
        viewModel.initialize(applicationContext)
        viewModel.requestAutomationAction(intent?.getStringExtra("codex_action"))

        setContent {
            LlamaAndroidTheme(darkTheme = false, dynamicColor = false) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                ) {
                    MainScreen(
                        viewModel = viewModel,
                        clipboard = clipboardManager,
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    viewModel: MainViewModel,
    clipboard: ClipboardManager,
) {
    val context = LocalContext.current
    var showSettings by rememberSaveable { mutableStateOf(false) }
    var showDiagnostics by rememberSaveable { mutableStateOf(false) }
    val listState = rememberLazyListState()
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) {
            viewModel.log("Model import canceled")
        } else {
            viewModel.importModel(context, uri)
        }
    }

    LaunchedEffect(viewModel.messages.size) {
        if (viewModel.messages.isNotEmpty()) {
            listState.animateScrollToItem(viewModel.messages.lastIndex)
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
    ) {
        Column(
            modifier = Modifier.fillMaxSize(),
        ) {
            ChatTopBar(
                title = viewModel.importedModel?.fileName?.substringBeforeLast(".") ?: "Fairy",
                subtitle = when (viewModel.modelLoadState) {
                    ModelLoadState.LOADED -> "Ready"
                    ModelLoadState.LOADING -> "Loading"
                    ModelLoadState.IMPORTING -> "Preparing"
                    else -> "Messages"
                },
                onOpenSettings = { showSettings = true },
            )

            if (viewModel.messages.isEmpty()) {
                EmptyConversation(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
                    modelReady = viewModel.modelLoadState == ModelLoadState.LOADED,
                    onOpenSettings = { showSettings = true },
                )
            } else {
                LazyColumn(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp),
                    state = listState,
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    item { Spacer(modifier = Modifier.height(8.dp)) }
                    items(viewModel.messages, key = { it.id }) { message ->
                        MessageBubble(message = message)
                    }
                    item { Spacer(modifier = Modifier.height(8.dp)) }
                }
            }

            viewModel.modelError?.let { error ->
                Text(
                    text = error,
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(horizontal = 20.dp, vertical = 6.dp),
                )
            }

            ComposerBar(
                prompt = viewModel.prompt,
                onPromptChanged = viewModel::updatePrompt,
                onSend = viewModel::send,
                onStop = viewModel::stopGeneration,
                onOpenSettings = { showSettings = true },
                canSend = viewModel.modelLoadState == ModelLoadState.LOADED && !viewModel.isGenerating,
                isGenerating = viewModel.isGenerating,
            )
        }

        if (showSettings) {
            val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
            ModalBottomSheet(
                onDismissRequest = { showSettings = false },
                sheetState = sheetState,
                containerColor = MaterialTheme.colorScheme.surface,
            ) {
                SettingsSheet(
                    viewModel = viewModel,
                    clipboard = clipboard,
                    showDiagnostics = showDiagnostics,
                    onToggleDiagnostics = { showDiagnostics = !showDiagnostics },
                    onImportModel = { launcher.launch(arrayOf("*/*")) },
                )
            }
        }
    }
}

@Composable
private fun ChatTopBar(
    title: String,
    subtitle: String,
    onOpenSettings: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp, vertical = 16.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column {
            Text(
                text = title.replaceFirstChar { it.uppercase() },
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        TextButton(onClick = onOpenSettings) {
            Text("Settings")
        }
    }
}

@Composable
private fun EmptyConversation(
    modifier: Modifier = Modifier,
    modelReady: Boolean,
    onOpenSettings: () -> Unit,
) {
    Column(
        modifier = modifier.padding(horizontal = 24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Box(
            modifier = Modifier
                .size(68.dp)
                .clip(CircleShape)
                .background(MaterialTheme.colorScheme.secondaryContainer),
        )
        Spacer(modifier = Modifier.height(18.dp))
        Text(
            text = if (modelReady) "Start a conversation" else "Model setup required",
            style = MaterialTheme.typography.titleMedium,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = if (modelReady) {
                "Ask anything and the reply will stream into the chat."
            } else {
                "Open settings to import a model and start chatting."
            },
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(18.dp))
        Button(onClick = onOpenSettings) {
            Text(if (modelReady) "Open Settings" else "Set Up")
        }
    }
}

@Composable
private fun MessageBubble(message: ChatMessage) {
    val isUser = message.role == ChatRole.USER
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
    ) {
        Column(
            modifier = Modifier.widthIn(max = 300.dp),
            horizontalAlignment = if (isUser) Alignment.End else Alignment.Start,
        ) {
            Box(
                modifier = Modifier
                    .clip(
                        RoundedCornerShape(
                            topStart = 22.dp,
                            topEnd = 22.dp,
                            bottomStart = if (isUser) 22.dp else 8.dp,
                            bottomEnd = if (isUser) 8.dp else 22.dp,
                        ),
                    )
                    .background(
                        if (isUser) MaterialTheme.colorScheme.primary
                        else MaterialTheme.colorScheme.surfaceVariant,
                    )
                    .padding(horizontal = 16.dp, vertical = 12.dp),
            ) {
                Text(
                    text = if (message.text.isBlank() && !isUser) "..." else message.text,
                    color = if (isUser) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurface,
                    style = MaterialTheme.typography.bodyLarge,
                )
            }
        }
    }
}

@Composable
private fun ComposerBar(
    prompt: String,
    onPromptChanged: (String) -> Unit,
    onSend: () -> Unit,
    onStop: () -> Unit,
    onOpenSettings: () -> Unit,
    canSend: Boolean,
    isGenerating: Boolean,
) {
    Surface(
        tonalElevation = 2.dp,
        color = MaterialTheme.colorScheme.surface,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .navigationBarsPadding()
                .padding(horizontal = 14.dp, vertical = 12.dp),
            verticalAlignment = Alignment.Bottom,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            TextButton(
                onClick = onOpenSettings,
                modifier = Modifier.align(Alignment.CenterVertically),
            ) {
                Text("Model")
            }

            OutlinedTextField(
                value = prompt,
                onValueChange = onPromptChanged,
                modifier = Modifier.weight(1f),
                placeholder = { Text("Message") },
                shape = RoundedCornerShape(26.dp),
                enabled = !isGenerating,
                maxLines = 5,
            )

            Button(
                onClick = if (isGenerating) onStop else onSend,
                enabled = if (isGenerating) true else canSend,
                shape = CircleShape,
                modifier = Modifier.align(Alignment.CenterVertically),
            ) {
                Text(if (isGenerating) "Stop" else "Send")
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SettingsSheet(
    viewModel: MainViewModel,
    clipboard: ClipboardManager,
    showDiagnostics: Boolean,
    onToggleDiagnostics: () -> Unit,
    onImportModel: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp),
    ) {
        Text(
            text = "Settings",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.SemiBold,
        )

        Surface(
            color = MaterialTheme.colorScheme.background,
            shape = RoundedCornerShape(24.dp),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(18.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text("Model", style = MaterialTheme.typography.titleMedium)
                Text(
                    text = viewModel.importedModel?.fileName ?: "No model",
                    style = MaterialTheme.typography.bodyLarge,
                )
                Text(
                    text = "State: ${viewModel.modelLoadState.label}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                viewModel.importedModel?.let { model ->
                    Text(
                        text = Formatter.formatShortFileSize(LocalContext.current, model.sizeBytes),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = onImportModel,
                        enabled = viewModel.modelLoadState != ModelLoadState.IMPORTING && !viewModel.isGenerating,
                    ) {
                        Text(if (viewModel.modelLoadState == ModelLoadState.IMPORTING) "Installing" else "Import")
                    }
                    Button(
                        onClick = viewModel::loadImportedModel,
                        enabled = viewModel.importedModel != null &&
                            viewModel.modelLoadState != ModelLoadState.LOADING &&
                            !viewModel.isGenerating,
                    ) {
                        Text(if (viewModel.modelLoadState == ModelLoadState.LOADING) "Loading" else "Load")
                    }
                    TextButton(
                        onClick = viewModel::unloadModel,
                        enabled = viewModel.importedModel != null || viewModel.modelLoadState == ModelLoadState.LOADED,
                    ) {
                        Text("Unload")
                    }
                }
            }
        }

        Surface(
            color = MaterialTheme.colorScheme.background,
            shape = RoundedCornerShape(24.dp),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(18.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text("Reply Length", style = MaterialTheme.typography.titleMedium)
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    GenerationLengthPreset.entries.forEach { preset ->
                        FilterChip(
                            selected = !viewModel.useCustomGenerationLength && viewModel.generationLength == preset,
                            onClick = { viewModel.selectGenerationLength(preset) },
                            label = { Text(preset.label) },
                        )
                    }
                    FilterChip(
                        selected = viewModel.useCustomGenerationLength,
                        onClick = viewModel::selectCustomGenerationLength,
                        label = { Text("Custom") },
                    )
                }
                OutlinedTextField(
                    value = viewModel.customGenerationLengthInput,
                    onValueChange = viewModel::updateCustomGenerationLength,
                    modifier = Modifier.widthIn(min = 132.dp),
                    enabled = viewModel.useCustomGenerationLength,
                    singleLine = true,
                    label = { Text("Max tokens") },
                    placeholder = { Text("1-512") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    supportingText = {
                        Text(
                            if (viewModel.useCustomGenerationLength) {
                                "Custom reply length, valid range 1-512"
                            } else {
                                "Selected preset: ${viewModel.generationLength.maxTokens} tokens"
                            },
                        )
                    },
                )
            }
        }

        Surface(
            color = MaterialTheme.colorScheme.background,
            shape = RoundedCornerShape(24.dp),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(18.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text("Diagnostics", style = MaterialTheme.typography.titleMedium)
                    TextButton(onClick = onToggleDiagnostics) {
                        Text(if (showDiagnostics) "Hide" else "Show")
                    }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = { viewModel.bench(8, 4, 1) },
                        enabled = viewModel.modelLoadState == ModelLoadState.LOADED && !viewModel.isGenerating,
                    ) {
                        Text("Benchmark")
                    }
                    TextButton(onClick = viewModel::clearDiagnostics) {
                        Text("Clear")
                    }
                    TextButton(
                        onClick = {
                            clipboard.setPrimaryClip(
                                ClipData.newPlainText("llama-diagnostics", viewModel.diagnosticsText()),
                            )
                        },
                    ) {
                        Text("Copy")
                    }
                }

                if (showDiagnostics) {
                    Divider()
                    LazyColumn(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(180.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        items(viewModel.diagnostics) { line ->
                            Text(
                                text = line,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))
    }
}
