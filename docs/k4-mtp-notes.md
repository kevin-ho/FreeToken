# Gemma k=4 MTP boundary

k=4 is intentionally not enabled in this checkout. The uncommitted generic
`run_mtp_transaction` helper was removed because it only compared fabricated callback
rows; it could not stage or roll back real target KV state and would have falsely implied
that Gemma serving supported k=4. The existing k=1 transaction remains unchanged.

## Exact serving boundary

The missing lifecycle is visible at these locations:

- `scheduler/scheduler.py:_prepare_batch` calls `CacheManager.allocate_paged()` and
  constructs one input-position mapping. It does not reserve a speculative span or provide
  a transaction snapshot.
- `engine/engine.py:forward_batch` runs one model forward, immediately calls
  `Req.complete_one()` for every request, samples one token per request, and returns a
  `ForwardOutput` with one output row.
- `models/gemma4/model.py:Gemma4ForCausalLM.forward` exports the hidden state from that
  one ordinary forward only. `models/gemma4/mtp.py:GemmaMTPDrafter` consumes one hidden
  row and currently returns `MTPProposal(width=1)`.
- `scheduler/scheduler.py:_run_speculative_mtp` invokes the k=1 seam only. Its engine
  bridge in `engine/engine.py:_init_mtp_bridge` verifies the already-produced one-row
  output; it does not execute a target verification forward over proposed positions.
- `scheduler/scheduler.py:_process_last_data` appends and streams exactly one token, then
  advances/removes the request using the ordinary decode lifecycle.
- `scheduler/cache.py:CacheManager.allocate_paged`, `cache_req`, and `_free` own page
  allocation/freeing. They have no staged KV-write owner, suffix truncation operation, or
  atomic rollback covering the page table plus SWA mappings. `_cache_req_swa` additionally
  reconciles shared/tombstoned SWA entries during commit, so blindly freeing a rejected
  suffix can free a shared slot or leave a stale mapping.
- `scheduler/prefill.py:PrefillAdder` reserves the entire request output budget at
  admission. That reservation is capacity accounting, not a speculative token transaction.

## Minimal safe API needed before k=4

Implement a scheduler/cache-owned `SpeculativeReservation` for one request that:

1. reserves page-table capacity and all model-specific state for `k + 1` positions without
   exposing those positions to the shared prefix/SWA trees;
2. runs one target forward on the reserved positions and stages target KV writes and logits;
3. commits exactly the accepted draft prefix plus one target bonus token, truncating and
   returning the rejected suffix; and
4. aborts atomically, restoring request lengths, page-table entries, full KV ownership,
   SWA/shared mappings, recurrent state (if applicable), and streaming/accounting state.

The scheduler must then pass the committed token sequence to the normal output drain rather
than calling `_process_last_data` once per draft. The engine/model interfaces also need a
batched target-forward result (at least `k + 1` target rows and the hidden state needed by
the real drafter), while preserving the ordinary MTP-off path and k=1 behavior.

Until those operations exist, wiring width 4 would either require four sequential target
forwards or retain rejected KV/accounting. Both are unsafe, so MTP remains fail-closed at
k=1. No live Gemma k=4 serving, benchmark, or hardware validation is claimed.
