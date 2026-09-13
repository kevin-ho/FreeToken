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

## Required bench validation checklist

1. Real-file load: load a real Gemma MTP file through `GemmaMTPDrafter.from_gguf`
   and record the inventory, shapes, metadata, and mapping.
2. Hidden-state vs greedy logits argmax: compare the target's opt-in exported
   post-norm hidden-state route against ordinary target greedy logits on the
   same real forward.
3. First end-to-end k=1 transaction smoke: with `--speculative-mtp`, verify
   draft-before-verify ordering, exact greedy output, one commit, and clean
   abort/resource behavior.
