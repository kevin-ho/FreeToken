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
  writes. The mapping defaults to assistant layers `(0, 1, 2, 3)` reading target
  layers `(26, 27, 28, 29)` for the 30-layer 26B/A4B checkpoint, and can be replaced
  explicitly with `--gemma-mtp-layer-mapping`.

## Mapping evidence

The default is source-grounded, not inferred from checkpoint names. In llama.cpp commit
[`73159c30399a77144f59d37fde504dfd00afbea5`](https://github.com/ggml-org/llama.cpp/commit/73159c30399a77144f59d37fde504dfd00afbea5),
`src/models/gemma4.cpp:7-10` reads `attention.shared_kv_layers` and sets
`n_layer_kv_from_start = n_layer_all - n_kv_shared_layers`; its attention graph then
reuses KV for layers after that boundary (`src/models/gemma4.cpp:242-247`). The
assistant source reads the same shared-KV metadata (`src/models/gemma4-assistant.cpp:7-10`)
and constructs exactly four assistant blocks (`:47-77`), while its graph calls the
assistant attention with no K/V tensors (`:127-151`). For the confirmed 30-layer
checkpoint, four shared layers therefore begin at 30 - 4 = 26: the source-majority
mapping is `(0,1,2,3) -> (26,27,28,29)`. This is a mapping of assistant blocks to the
shared target KV tail, not a claim that assistant block number equals target block
number. If a future checkpoint has different shared-KV metadata, it must pass a new
explicit mapping rather than changing this fallback silently.

The same source exposes the target post-final-norm hidden state as `h_nextn`
(`src/models/gemma4.cpp:382-387`) and the assistant consumes target token embeddings
(`src/models/gemma4-assistant.cpp:108-118`), supporting the bridge's one-target-forward
preparation path.

Preparation performs the ordinary target `forward_batch()` exactly once, after page-table
and embedding setup, so Gemma can export the hidden state needed by the drafter. Verification
only reuses that output and does not call the target a second time. Commit is deliberately idempotent;
abort is also idempotent and does not free pages. Final request draining and resource
release remain in the scheduler's existing lifecycle.

## Validation evidence (M3b completion attempt)

Environment inspection on 2026-09-12:

- `git worktree list` showed only `/home/vesper/FreeToken`; no candidate checkout or
  `candidate.env` was visible under `/home` or `/mnt`. No production files were changed.
- `/home/kho/models/mtp-drafters/mtp-gemma-4-26B-A4B-it.gguf` returned `ENOENT` in this
  execution environment, so real-file load/inventory/shape/metadata validation could not
  run. No substitute checkpoint was used.
- The permitted candidate-only rebuild, server comparison (greedy flag off/on), and
  ~256-token ON generation were not run because the candidate and real file were absent.
  No full judge or benchmark-shaped work was run.
- `python3 -m compileall -q python/freetoken`: passed.
- `git diff --check`: passed.
- Focused pytest was attempted with
  `python3 -m pytest -q tests/engine/test_mtp_transaction.py tests/scheduler/test_mtp_hook.py`;
  blocked because `pytest` is not installed (`No module named pytest`).
