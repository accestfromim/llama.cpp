---
name: code-review
description: Review local llama.cpp and Fairy2i changes for scope, correctness, security, and validation evidence.
---

# Review a change

This produces private review notes. It never writes a PR description, commit message, issue, review comment, or reviewer reply.

## Scope

1. Read the nearest `AGENTS.md` files and `CONTRIBUTING.md`.
2. Inspect `git diff --stat` and classify every changed path.
3. Check whether the change is single-purpose and whether an upstream issue or PR already covers it.
4. Identify the exact observable contract and the tests that must fail if it regresses.

Always run the general and security checks. Add the relevant area checks:

- `ggml/`: backend dispatch, operator consistency, fallback paths, and `test-backend-ops`;
- `src/`, `gguf-py/`: model metadata, tensor shapes, conversion and loader boundaries;
- `tools/server/`: request bounds, ownership, and server scope;
- `include/llama.h`: ABI stability and a working caller;
- Fairy2i/iFairy CPU paths: layout, conjugation, routing flags, ARM guards, and reference comparison;
- workflow/scripts: shell error handling, path filters, reproducibility, and missing dependencies.

## Security gates

Treat GGUF metadata, tensor dimensions, tokenizer data, server JSON, and RPC fields as hostile input. Look for:

- allocation or loop arithmetic before overflow checks;
- declared array lengths or element types used without validation;
- file-derived counts indexing fixed arrays;
- narrowing casts or signed/unsigned comparisons around bounds;
- parsed token IDs used as indices without range checks;
- `GGML_ASSERT` aborts on file-derived values where callers can report an error;
- raw pointers that outlive their owner or async work that outlives its buffers.

## C++ and backend checks

- New dispatch conditions must be no broader than the behavior they enable.
- Keep scalar, generic, and non-ARM fallbacks intact.
- Preserve `w * conj(x)` in complex/iFairy paths.
- Check tile dimensions, alignment, tails, scale dtype, packed layout, and output carrier format.
- New or changed operators need deterministic reference cases and backend coverage.
- Remove debug/profiling code and avoid unrelated formatting churn.

## Fairy2i validation

For any Fairy2i or legacy iFairy CPU change, require evidence from the applicable gates:

```sh
scripts/ci-fairy2i-cpu.sh
```

At minimum, inspect the direct test, LUT test, LUT-off test, and the focused `test-backend-ops` case. If model loading or conversion changed, also run:

```sh
scripts/test-gguf-py.sh
```

Performance claims require the exact command, CPU, compiler flags, thread count, path markers, and raw log path. Do not accept a speedup without correctness evidence.

## Report

Group findings as:

1. Blocking: correctness, security, scope, or missing required validation;
2. Review friction: conventions, unnecessary complexity, missing tests/docs/perf data;
3. Nits: optional cleanup.

For each finding, name the file and line, explain the risk, and state the smallest concrete fix. Report verified evidence separately from inference.
