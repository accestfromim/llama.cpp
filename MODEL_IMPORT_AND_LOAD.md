# Model Import And Load

## Implemented UI Flow

The app now uses Android's document picker instead of in-app downloading.

Implemented flow:

1. Tap `Import Model`
2. Select a local model file from the system picker
3. App copies the selected file into:
   - `filesDir/models/<sanitized-name>`
4. UI shows:
   - file name
   - file size
   - private absolute path
5. Tap `Load Model`
6. JNI loads the model from the copied private path

## UI States

The app exposes these load states:

- `未导入`
- `导入中`
- `未加载`
- `加载中`
- `已加载`
- `加载失败`

Failures are surfaced both in UI and in `LLAMA_ANDROID` logs.

## Storage Requirement

The app does not run inference directly from a temporary picker URI.

It copies the model into app-private storage first, which matches the deployment rule in `CODEX.md`.

## Current Validation Status

- Import path implemented in code
- private copy path implemented in code
- explicit load button implemented in code
- load status UI implemented in code
- no physical-device validation completed yet in this session
