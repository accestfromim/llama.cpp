---
name: add-new-model
description: Guided workflow for adding or adapting a model architecture in this llama.cpp fork.
---

# Add or adapt a model architecture

Use the local architecture layout. This fork currently keeps model support in `src/llama-arch.*`, `src/llama-model.*`, `src/llama-graph.*`, and `src/llama-context.cpp`; do not assume the newer upstream `src/models/` layout exists.

## Step 0: scope and duplicate check

Before editing, identify the model name, source checkpoint/config, text versus multimodal scope, closest existing architecture, required GGUF keys, tensor names, and prefill/decode shapes. Check existing issues and PRs for duplicate support. Keep conversion, runtime support, chat-template work, and backend optimization separate unless the dependency is proven.

## Step 1: conversion contract

Read the existing converter and `gguf-py/gguf/` mapping code for the closest architecture. Define the tensor names, dtypes, shapes, quantization metadata, tokenizer metadata, and required defaults. Reject malformed or incomplete checkpoints before writing output. Add focused Python tests for valid metadata, invalid shapes/types, and non-finite values.

## Step 2: runtime registration

Update the smallest applicable set of:

- `src/llama-arch.h` and `src/llama-arch.cpp` for architecture and GGUF keys;
- `src/llama-model-loader.*` for required metadata and tensor checks;
- `src/llama-model.*` for tensor mapping and model loading;
- `src/llama-graph.*` and `src/llama-context.cpp` for prefill/decode graphs;
- tokenizer or chat-template code only when the model requires it.

Load-bearing metadata must fail clearly when missing. Do not silently reinterpret incompatible tensor layouts. Prefer conversion-time constant transforms over runtime graph machinery.

## Step 3: correctness gates

Add the smallest observable tests first:

1. converter and GGUF metadata tests;
2. loader success and malformed-input rejection;
3. one deterministic prefill/decode path;
4. logits or reference comparison when a source implementation is available;
5. quantized and CPU fallback paths when applicable.

Use the local build and test commands in `AGENTS.md`. Run `test-backend-ops` for changed ggml operators. Run `scripts/test-gguf-py.sh` for converter changes.

## Step 4: review boundaries

Do not add a custom RoPE implementation when `ggml_rope_ext` can express the behavior. Do not add a new public API, GGML type, backend, or shared graph abstraction as incidental model support. Stop and review the design if one is required.

Do not claim support from a compile-only result. Record the exact model/config, command, output, and failure behavior. Keep the change understandable and small enough for the contributor to defend line by line.
