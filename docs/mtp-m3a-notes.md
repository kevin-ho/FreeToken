# MTP M3a notes

## Implemented surface

`GemmaMTPDrafter.from_gguf()` loads the observed `nextn` GGUF component without
changing the normal Gemma model or scheduler. It reads the 1024-wide assistant
embedding, the two projection tensors, and four decoder blocks through the
existing GGUF reader and `dequantize` machinery. `draft_step` accepts a target
hidden state, the target-provided 2816-wide embedding, and a caller-supplied
target LM head, returning logits at the target vocabulary size. There is intentionally no `output.weight` or
second LM head in the drafter. `draft_into_batch` is a standalone convenience
adapter; speculative MTP remains default-off and has no production wiring.

The four blocks implement Gemma-style RMS norms, gated feed-forward, per-head
Q norms, grouped-query attention, causal RoPE, and the metadata-provided
sliding-window/full attention pattern. `layer_output_scale` is applied only
when that tensor exists in the block, matching the existing Gemma loader's
support for that optional scalar.

## Verified frozen directions

The accepted inventory is exact: the five roots `token_embd.weight`,
`output_norm.weight`, `rope_freqs.weight`, `nextn.pre_projection.weight`, and
`nextn.post_projection.weight`, plus four blocks each containing exactly
`attn_norm`, `attn_output`, `attn_q`, `attn_q_norm`, `ffn_down`, `ffn_gate`,
`ffn_norm`, `ffn_up`, `layer_output_scale`, `post_attention_norm`, and
`post_ffw_norm`. K/V tensors are deliberately absent; inventory drift hard-fails.

- `nextn.pre_projection.weight`: `[1024, 5632]`, consuming the target-provided
  `[2816]` hidden component and `[2816]` embedding component concatenated to width
  5632. The caller must provide the embedding; the drafter does not invent it.
- `nextn.post_projection.weight`: `[2816, 1024]`.
- `token_embd.weight`: `[vocab, 1024]` (the vocabulary dimension is checkpoint
  metadata, not hard-coded).
- Required metadata uses the `gemma4-assistant.` prefix and includes
  `block_count`, `attention.head_count`, `attention.head_count_kv`,
  `attention.key_length`, `attention.key_length_swa`, and
  `attention.sliding_window_pattern`; missing keys hard-fail. Q widths are 4096
  for blocks 0-2 and 8192 for block 3. Head dimensions come from the two
  `key_length` values and the pattern.
- No `output.weight` is accepted; logits use the target head supplied by the
  caller, preserving tied-head and standalone scope.

Source evidence for geometry is precise: `python/freetoken/models/gemma4/gguf.py`
lines 97-146 reads `key_length`, `key_length_swa`, and
`sliding_window_pattern`, and constructs separate full/SWA attention groups;
`python/freetoken/models/gemma4/attention.py` lines 30-35 derives Q width from
head count and head dimension. No upstream/source implementation in this
worktree defines assistant K/V tensors. Their provenance is unresolved, so the
adapter requires an explicit `kv_provider(layer, hidden, head_dim, kv_heads)`
from the target rather than fabricating K/V weights.

## Box follow-up

Run the `needs_weights` real-checkpoint test with the local Gemma MTP GGUF path,
then compare a real target-head output against the checkpoint's reference
implementation. Confirm the checkpoint's exact per-block attention metadata and
RoPE parameters before enabling any production integration. Production scheduler,
cache ownership, sampling, rollback, and multi-token transaction work remain out
of scope for M3a.
