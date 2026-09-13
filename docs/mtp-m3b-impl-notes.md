# MTP M3b implementation notes

## Touchpoints

- `models/gemma4/model.py`: `Gemma4ForCausalLM.enable_speculative_mtp()` gates a
  post-final-norm hidden-state side channel. `forward()` still returns the same
  logits tensor and only attaches `batch.mtp_hidden_state` when enabled.
- `models/gemma4/mtp.py`: `TargetKvProvider` reads existing target pool rows;
  `GemmaMTPDrafter.draft_step()` and `draft_into_batch()` accept the measured
  hidden state, target embedding, LM head, positions, and provider.
- `models/gemma4/__init__.py`: exports the adapter and provider.
- The existing `engine.mtp.run_k1_transaction()` remains the transaction seam;
  it preserves draft -> verify -> compare/bonus -> one commit, with abort on
  exceptions. No scheduler or cache owner was added.

The command-line flag remains default-off. There is no production adapter
factory in this checkout: a real model load must establish the target hidden
state, target embedding, LM head, and checkpoint-specific shared-KV mapping
before enabling it. Consequently the flag does not silently construct a
partially-valid drafter.

## Cache-access findings

`BaseKVCachePool` exposes `k_cache(global_layer)` and `v_cache(global_layer)`.
`MHAKVCache` applies its existing global-to-dense layer map, and the shared
scheduler page table exposes token locations. This is sufficient for a narrow
read-only provider contract when the caller supplies one request's page-table
row, token positions, and the documented assistant-layer -> target-layer
mapping. `TargetKvProvider` uses only those interfaces and performs no writes
or allocation.

The checkout does not expose an assistant-specific mapping field on Gemma
configuration, nor does it establish that every real assistant checkpoint can
reuse target rows. The provider therefore requires an explicit mapping and is
not wired automatically. There is no fake K/V fallback and no second cache.

## Scheduler wiring decision

Production wiring remains disabled for M3b. The current lifecycle does not provide a
safe draft-before-verify insertion point:

- `scheduler.Scheduler._forward()` unpacks one `ForwardInput`, calls
  `engine.forward_batch(batch, sample_args)` once, writes `next_tokens_gpu` into the
  shared `token_pool`, and immediately calls `decode_manager.filter_reqs()`.
- `Engine.forward_batch()` runs the target model forward that creates
  `batch.mtp_hidden_state`; the hidden state is therefore only available *after* the
  target forward. Calling `draft_into_batch()` from `_forward()` after that point
  would be draft-after-target, not draft-before-verify.
- `overlap_loop()` and `normal_loop()` drain a whole `ForwardData` through
  `_process_last_data()`; that drain handles EOS, aborts, prefix caching, and resource
  release for every request in the batch. It has no per-request transaction or
  rollback/commit seam.
- The scheduler batches requests, whereas `TargetKvProvider` deliberately requires
  one prepared request page-table row and its positions. Choosing one row or creating
  another cache owner would be incorrect.

Consequently, constructing `GemmaMTPDrafter` from the flag alone would either use
missing checkpoint/mapping inputs or change cache and token ownership semantics. A
future hook needs an explicit single-request transaction boundary, a second target
verification forward, and scheduler-owned commit/abort handling before it can call
`run_k1_transaction`. The flag remains parsed and model hidden-state export remains
opt-in, but it does not alter the production scheduler.

## Required bench validation checklist

1. Real-file load: load a real Gemma MTP file through `GemmaMTPDrafter.from_gguf`
   and record the inventory, shapes, metadata, and mapping.
2. Hidden-state vs greedy logits argmax: compare the target's opt-in exported
   post-norm hidden-state route against ordinary target greedy logits on the
   same real forward.
3. First end-to-end k=1 transaction smoke: with `--speculative-mtp`, verify
   draft-before-verify ordering, exact greedy output, one commit, and clean
   abort/resource behavior.
