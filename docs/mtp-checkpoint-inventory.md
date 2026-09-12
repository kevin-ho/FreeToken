# Gemma-4 MTP checkpoint inventory

## Probe and search scope

The read-only probe is `scripts/probe_checkpoint_inventory.py`. It accepts a local
Hugging Face safetensors directory or a single local `.gguf` file, and reports
configuration keys, case-insensitive `mtp`/`draft`/`nextn`/`speculative` key matches,
per-shard tensor names/dtypes/shapes, prefix groups, and (for GGUF) every metadata
field. It uses `cached_load_hf_config`, `iter_weight_files`/`ShardReader`, and
`load_gguf_metadata` plus GGUF tensor descriptors; it does not call `ShardReader.get_tensor()`
or access GGUF tensor payloads or load model weights.

The cache search was read-only. Exact search root and command:

```text
root: /home/vesper/.cache/huggingface
find /home/vesper/.cache/huggingface -maxdepth 8 -type d \
  | grep -Ei 'gemma.?4|26b|mtp|drafter|assistant'
find /home/vesper/.cache/huggingface -maxdepth 8 -type f \
  | grep -Ei 'gemma.?4|26b|mtp|drafter|assistant'
```

Both searches returned no paths in this environment. No checkpoint was downloaded,
and no local Gemma-4-26B, MTP, drafter, or assistant candidate was available to
probe. Consequently there is no tensor inventory or GGUF metadata inventory to
report from the cache.

## Format verdict (mtp-port-plan section 2)

**Undetermined; retain the plan's stop condition.** The cache contains no candidate
checkpoint, so there is no evidence for target-shared embedding/LM-head/KV weights
or for a separate draft weight set. The implementation must not choose between
those formats, invent an `mtp.*` naming convention, or add loader behavior. A real
local checkpoint must be supplied and run through the probe before deciding:
reuse target components only if the inventory and computation prove that they are
shared; otherwise model the draft as an explicitly separate weight set.

This inventory-only change therefore provides the requested measurement tool and
fixture coverage but makes no engine or loader changes.
