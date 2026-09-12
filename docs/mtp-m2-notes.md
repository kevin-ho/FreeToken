# MTP milestone 2 notes

## Implementation

M2 adds the smallest model-neutral transaction in `freetoken.engine.mtp`: the
scheduler prepares the ordinary batch and remains the sole page owner, then a
k=1 drafter runs before target verification. A matching proposal is committed;
on rejection the target token is committed as the bonus token. Exceptions call
the supplied abort/release callback. The transaction does not allocate, free,
or roll back KV pages itself.

`models/gemma4/mtp.py` is the narrow adapter boundary. `GemmaMTPDrafter`
accepts a prepared batch through `draft_into_batch` and returns proposal logits,
optional probabilities, and width one. It requires a checkpoint-specific
`draft_step`; it does not invent a second model, tensor names, attention state,
or sampling policy.

`EngineConfig.speculative_mtp` is the explicit opt-in switch and defaults to
`False`. There is deliberately no default scheduler wiring until a real Gemma
checkpoint supplies an executable drafter. Thus normal serving behavior is
unchanged.

The GGUF groundwork in `gemma4.gguf` records only the observed inventory
categories: `token_embd`, `nextn.pre_projection`, `nextn.post_projection`,
`blk.*` groups, and shared-KV metadata. It is metadata-only and does not load or
interpret an MTP tensor payload.

## Bench-box wiring

On a bench box, construct the normal Gemma engine/scheduler and pass its already
prepared single-request batch to a checkpoint-specific `draft_step`, then call
`run_k1_transaction`. Keep the target verify callback on the existing forward
path and use the existing cache manager's commit/free lifecycle in the supplied
callbacks. The flag should remain off for baseline runs; compare it with the
same prompt, seed, cache sizes, and target checkpoint only after a local
inventory identifies the draft computation.

No real checkpoint, ASTRALPLANE, or production system was accessed for M2.

## Open questions

- Which local checkpoint tensor shapes and computation define the Gemma drafter?
- Are token embedding, LM head, and KV state shared, and how does `nextn` consume
  the target hidden state?
- Does the target's prepared batch need a model-specific input view while still
  retaining one scheduler/cache owner?
- What exact sampling semantics are required once width greater than one is
  considered? M2 intentionally implements no rejection sampling, rollback,
  attention changes, or drafter internals.
