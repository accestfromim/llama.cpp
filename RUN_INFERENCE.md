# Run Inference

## Implemented UI

The app now exposes:

- model status section
- prompt input
- `Send`
- `Stop`
- output log area
- `Bench`

## Runtime Flow

- `Send` calls the existing `LLamaAndroid` JNI facade.
- Native completion output is emitted as a Kotlin `Flow<String>`.
- The UI appends pieces into the last output row to produce streaming text.
- `Stop` sets a stop flag checked between token steps on the dedicated native thread.

## Guard Rails

- no generation starts unless a model is loaded
- repeated `Send` while generation is active is blocked
- load while generation is active is blocked
- errors are pushed into UI console and Android logcat

## Current Validation Status

- generation pipeline is verified end to end on device
- model `ifairy.gguf` loads successfully from app-private storage
- prompt `hello` streamed output successfully on device
- `Stop` was verified from the UI and logged:
  - `Stop requested`
  - `Stop requested before sampling next token`
  - `Generation stopped`
- benchmark trigger is exposed in UI and also available through `adb`:
  - `adb shell am start -n com.example.llama/.MainActivity --es codex_action bench`
