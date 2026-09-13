# MTP M3b bridge notes

## Status

The production callback bridge is implemented, but bench smoke is **blocked**: this
checkout has no validated real Gemma MTP GGUF available in the test environment, and
GPU benchmark-shaped work was not run locally. The bridge therefore fails closed when
the flag, GGUF assistant inventory, prepared single-request inputs, or explicit KV
mapping is missing.

## Bridge touchpoints

- `python/freetoken/engine/engine.py`: when `EngineConfig.speculative_mtp` is enabled
  and the model path is a direct GGUF file, load `GemmaMTPDrafter.from_gguf()`.
  Construct a read-only `TargetKvProvider` from the prepared request's page-table row,
  `batch.positions`, and the existing target KV pool. The engine provides the verify,
  commit, and idempotent abort callbacks.
- `python/freetoken/scheduler/scheduler.py`: the existing opt-in hook requires the
  complete bridge and calls `run_k1_transaction`; otherwise it uses the ordinary
  `Engine.forward_batch()` path. The scheduler remains the page/cache owner.
- `python/freetoken/models/gemma4/mtp.py`: the provider performs no allocation or
  writes. It requires an explicit assistant-to-target KV mapping; no default mapping
  is enabled because the inspected llama.cpp sources do not establish one.

The mapping is not inferred from checkpoint names. The inspected Gemma assistant
block construction and target shared-KV selection in llama.cpp do not establish
assistant-to-target layer correspondence. The bridge therefore requires a separately
validated explicit mapping rather than silently using block numbers. The relevant stable
commit is
[`73159c30399a77144f59d37fde504dfd00afbea5`](https://github.com/ggml-org/llama.cpp/commit/73159c30399a77144f59d37fde504dfd00afbea5):
`src/models/gemma4-assistant.cpp` lines 84-127 and 185-200, and
`src/models/gemma4.cpp` lines 232-273. If a future checkpoint proves a different
relationship, it must provide a new explicit mapping rather than changing this
requirement silently.

Preparation performs the ordinary target `forward_batch()` exactly once, after page-table
and embedding setup, so Gemma can export the hidden state needed by the drafter. Verification
only reuses that output and does not call the target a second time. Commit is deliberately idempotent;
abort is also idempotent and does not free pages. Final request draining and resource
release remain in the scheduler's existing lifecycle.

## Bench-smoke evidence

- Focused CPU/unit tests: not run because `pytest` is not installed in this checkout.
- `py_compile` and `git diff --check`: passed with `python3`.
- Real-file load and end-to-end `--speculative-mtp` smoke: blocked by unavailable
  validated checkpoint/bench box; do not substitute a benchmark-shaped local GPU run.
