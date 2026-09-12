# MTP M3a notes

## Implemented surface

`GemmaMTPDrafter.from_gguf()` loads the observed `nextn` GGUF component without
changing the normal Gemma model or scheduler. It reads the drafter embedding,
the two projection tensors, and four decoder blocks through the existing GGUF
reader and `dequantize` machinery. `draft_step` accepts a target hidden state,
the last-token embedding, and a caller-supplied target LM head, returning logits
at the target vocabulary size. There is intentionally no `output.weight` or
second LM head in the drafter. `draft_into_batch` is a standalone convenience
adapter; speculative MTP remains default-off and has no production wiring.

The four blocks implement Gemma-style RMS norms, gated feed-forward, per-head
Q/K norms, grouped-query attention, causal RoPE, sliding-window attention in
blocks 0-2, and full attention in block 3. `layer_output_scale` is applied only
when that tensor exists in the block, matching the existing Gemma loader's
support for that optional scalar.

## Verified frozen directions

- `nextn.pre_projection.weight`: `[1024, 5632]`, consumed as
  `concat(target_hidden[2816], last_token_embedding[2816])`.
- `nextn.post_projection.weight`: `[2816, 1024]`.
- `nextn.token_embd.weight`: `[262144, 1024]`; it is the drafter's embedding
  table and is separate from the target embedding.
- No `output.weight` is accepted; logits use the target head supplied by the
  caller.

The repository's Gemma4 implementation was used for the norm, hybrid-attention,
RoPE, and scalar conventions. No contradictory upstream `nextn` implementation
exists in this worktree, so the frozen tensor structure is the source-specific
boundary rather than an inferred target-model loader.

## Box follow-up

Run the `needs_weights` real-checkpoint test with the local Gemma MTP GGUF path,
then compare a real target-head output against the checkpoint's reference
implementation. Confirm the checkpoint's exact per-block attention metadata and
RoPE parameters before enabling any production integration. Production scheduler,
cache ownership, sampling, rollback, and multi-token transaction work remain out
of scope for M3a.
