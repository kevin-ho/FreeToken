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
  writes. The mapping must be supplied explicitly with `--gemma-mtp-layer-mapping`;
  there is no automatic default because the available source does not prove assistant
  blocks can read target KV rows.

## Mapping evidence

The source evidence does **not** support an automatic assistant-to-target mapping. In
llama.cpp commit [`73159c30399a77144f59d37fde504dfd00afbea5`](https://github.com/ggml-org/llama.cpp/commit/73159c30399a77144f59d37fde504dfd00afbea5),
`src/models/gemma4.cpp:7-10` reads `attention.shared_kv_layers` and sets
`n_layer_kv_from_start = n_layer_all - n_kv_shared_layers`; its target attention graph
reuses earlier KV for target layers after that boundary (`src/models/gemma4.cpp:242-247`).
For a 30-layer target with four shared layers, that identifies target layers 26-29.
However, `src/models/gemma4-assistant.cpp:127-151` calls assistant attention with null
K/V tensors, and does not map assistant block 0-3 to target layers 26-29. The assistant
loader's `shared_kv_layers` read (`:7-10`) therefore cannot by itself justify that
mapping. The candidate must pass `--gemma-mtp-layer-mapping` only after a checkpoint and
runtime probe establish the required provider contract; the safe default is unset.

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

Validation evidence (Phase C attempt, 2026-09-12):

- From this session, SSH to ASTRALPLANE did not complete: attempts returned `connection
  reset by peer` and then `timed out during banner exchange`. Consequently no candidate
  command was run, no candidate checkout was synced or rebuilt, and production was not
  touched.
- The user-provided host evidence is recorded as the authoritative file fact:
  `/home/kho/models/mtp-drafters/mtp-gemma-4-26B-A4B-it.gguf` exists on ASTRALPLANE and
  is 251939328 bytes; it was not reachable from this session, so there is no claimed
  inventory/shape/metadata output here. The required command to capture when SSH is
  available is:
  `FREETOKEN_TEST_MTP_GGUF=/home/kho/models/mtp-drafters/mtp-gemma-4-26B-A4B-it.gguf
  python3 - <<'PY' ... GemmaMTPDrafter.from_gguf(...) ... PY`.
- Flag OFF/ON exactness and short ON generation were not run because candidate setup and
  the real GPU environment were unreachable. No substitute checkpoint or benchmark was
  used.
- Local source verification fetched llama.cpp commit
  `73159c30399a77144f59d37fde504dfd00afbea5`: `gemma4.cpp:7-10` derives target KV
  reuse from `shared_kv_layers`, `gemma4.cpp:242-247` reuses target KV, while
  `gemma4-assistant.cpp:127-151` passes null K/V to assistant attention. This supports
  removing the automatic `(26,27,28,29)` mapping default.
- Local checks: `python3 -m compileall -q python/freetoken` passed and `git diff --check`
  passed. Focused pytest was not run locally because pytest is unavailable.
