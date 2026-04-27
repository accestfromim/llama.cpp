# Known Issues

- Runtime load, generation, stop, and warm-up benchmark are now verified on device, but the expected one-time dispatch log `selected iFairy vecdot path: ...` still has not appeared in `adb logcat`.
- `examples/llama.android/local.properties` currently points to a temporary SDK root under `/tmp/android-sdk`; this is suitable for this machine session, not as a permanent repo default.
- JNI still uses `llama_new_context_with_model`, which now compiles with a deprecation warning. It is functional for this prototype but should later move to `llama_init_from_model`.
- `free_batch()` in `llama-android.cpp` still mirrors the previous simplified heap ownership and likely leaks nested allocations across model lifetime boundaries. It is acceptable for short-lived debug prototyping, but should be cleaned up before longer burn-in testing.
- The new adb automation hook `codex_action=bench` is intended for verification only and should be treated as a debug convenience path.
