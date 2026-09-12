# MTP M3b research card: smallest Gemma-4 k=1 design

Status: design-only, based on the current checkout and public llama.cpp source. This document does not claim that FreeToken can execute a Gemma-4 drafter today. No engine, scheduler, or Python source is changed by M3b.

## Evidence and provenance

### Current FreeToken checkout

- The current M2 transaction is deliberately model-neutral and owns no KV pages. The prepared batch is supplied by the scheduler/cache lifecycle, while callers supply commit and abort/release callbacks: `python/freetoken/engine/mtp.py:1-5,33-49`.
- The only current drafter contract is `draft_into_batch(batch) -> MTPProposal`; the proposal contains logits, optional probabilities, and an explicit width: `python/freetoken/engine/mtp.py:13-24`.
- k=1 is greedy-only: draft first, run target verification, compare first predictions, use the target token as the bonus token on rejection, then commit exactly once; exceptions call abort: `python/freetoken/engine/mtp.py:40-68`.
- Gemma-4 currently has one ordinary `Gemma4ForCausalLM`, with one embedding, one decoder-layer stack, final norm, and LM head; its forward is embedding -> all layers -> norm -> LM head: `python/freetoken/models/gemma4/model.py:57-70,101-143`. There is no assistant/NextN model in that file.
- Gemma attention consumes the global batch positions and calls the selected attention backend with the batch and per-layer `AttentionSpec`; it does not expose a draft-state or assistant-state API: `python/freetoken/models/gemma4/attention.py:17-64,80-113`.
- The scheduler has one `CacheManager`, one shared page table, and model-specific cache tiers plugged into that manager: `python/freetoken/scheduler/scheduler.py:75-90`. It allocates pages during batch preparation before the forward: `python/freetoken/scheduler/scheduler.py:762-819`.
- The forward result is drained later; only then does the scheduler append the sampled token and either free request resources or cache the prefill prefix: `python/freetoken/scheduler/scheduler.py:302-390`. This is the required ownership boundary for a speculative transaction.
- Decode reserves one page's worth of in-flight capacity per running request and emits an ordinary decode `Batch`: `python/freetoken/scheduler/decode.py:9-39`. The cache manager's free-list is page-aligned: `python/freetoken/scheduler/cache.py:32-63`.
- The local GGUF code records only an observed inventory (`token_embd`, `nextn.pre_projection`, `nextn.post_projection`, block groups, and shared-KV metadata), and explicitly says this is metadata, not an executable drafter: `python/freetoken/models/gemma4/gguf.py:32-69`.
- M2's research note records the same stop condition: no real Gemma checkpoint, ASTRALPLANE, or production system was accessed, and the open questions include the drafter shapes, shared embedding/LM head, and KV-state relationship: `docs/mtp-m2-notes.md` (Implementation, Bench-box wiring, Open questions).

### Public llama.cpp evidence

The relevant upstream implementation is current `master` as inspected on 2026-02-07. Stable commit/PR anchors are included because `master` can move:

- Initial Gemma-4 MTP support: PR [#23398](https://github.com/ggml-org/llama.cpp/pull/23398), merge commit [`04eb4c446d22b63449d5dc41c038987d4d8cc3a6`](https://github.com/ggml-org/llama.cpp/commit/04eb4c446d22b63449d5dc41c038987d4d8cc3a6), titled `llama : add Gemma4 MTP`.
- Smaller E2B/E4B assistants: PR [#24282](https://github.com/ggml-org/llama.cpp/pull/24282), commit [`7d2b45b4f7b663cda74f23fbc3ce6dc3bd4f6545`](https://github.com/ggml-org/llama.cpp/commit/7d2b45b4f7b663cda74f23fbc3ce6dc3bd4f6545), titled `mtp: support for gemma-4 E2B and E4B assistants`.
- A later Gemma assistant fix is commit [`73159c30399a77144f59d37fde504dfd00afbea5`](https://github.com/ggml-org/llama.cpp/commit/73159c30399a77144f59d37fde504dfd00afbea5), PR [#28183](https://github.com/ggml-org/llama.cpp/pull/28183).

The source facts are:

1. The target loader reads `attention.shared_kv_layers` and derives `n_layer_kv_from_start`; layers without KV reuse an earlier KV cache. See [`src/models/gemma4.cpp` lines 3-20](https://github.com/ggml-org/llama.cpp/blob/73159c30399a77144f59d37fde504dfd00afbea5/src/models/gemma4.cpp#L3-L20), and the actual no-KV branch at [lines 232-273](https://github.com/ggml-org/llama.cpp/blob/73159c30399a77144f59d37fde504dfd00afbea5/src/models/gemma4.cpp#L232-L273).
2. The target exposes the post-final-norm hidden state as `h_nextn`, explicitly for MTP draft contexts, and says this matches Transformers/vLLM/SGLang. See [`src/models/gemma4.cpp` lines 405-424](https://github.com/ggml-org/llama.cpp/blob/73159c30399a77144f59d37fde504dfd00afbea5/src/models/gemma4.cpp#L405-L424).
3. The assistant loader requires equal K/V head widths, requires `embedding_length_out` to carry the target hidden size, loads token embedding and output from the assistant inventory, and loads `nextn_proj_pre`, `nextn_proj_post`, and assistant layers. See [`src/models/gemma4-assistant.cpp` lines 3-77](https://github.com/ggml-org/llama.cpp/blob/73159c30399a77144f59d37fde504dfd00afbea5/src/models/gemma4-assistant.cpp#L3-L77).
4. The assistant graph consumes target token embeddings plus the target hidden input, concatenates them, applies `nextn_proj_pre`, runs assistant layers, then applies `nextn_proj_post` to produce the next recurrent hidden state. See [`src/models/gemma4-assistant.cpp` lines 84-127](https://github.com/ggml-org/llama.cpp/blob/73159c30399a77144f59d37fde504dfd00afbea5/src/models/gemma4-assistant.cpp#L84-L127) and [lines 185-200](https://github.com/ggml-org/llama.cpp/blob/73159c30399a77144f59d37fde504dfd00afbea5/src/models/gemma4-assistant.cpp#L185-L200).
5. The generic NextN MTP implementation seeds draft computation from target `h_nextn`, sets masked/unmasked NextN embedding modes, advances a NextN-layer offset, and reads per-row hidden states. See [`common/speculative.cpp` lines 1330-1446](https://github.com/ggml-org/llama.cpp/blob/73159c30399a77144f59d37fde504dfd00afbea5/src/common/speculative.cpp#L1330-L1446), the draft loop at [lines 1591-1679](https://github.com/ggml-org/llama.cpp/blob/73159c30399a77144f59d37fde504dfd00afbea5/src/common/speculative.cpp#L1591-L1679), and the staging API comment at [line 13](https://github.com/ggml-org/llama.cpp/blob/73159c30399a77144f59d37fde504dfd00afbea5#L13) (the latter is a source anchor for the inspected revision; the raw file's include path is `common/../src/llama-ext.h`).

### Negative-existence search and fallback

The checkout has no Gemma assistant/NextN execution path: `rg` finds only the model-neutral `MTPDrafter` seam and the GGUF metadata inventory, not a Gemma `draft_step`, assistant module, target-hidden-state export, or NextN forward. The requested upstream search also found no `gemma-moba` path or symbol in the current llama.cpp recursive tree; the relevant current paths are `src/models/gemma4.cpp`, `src/models/gemma4-assistant.cpp`, and `common/speculative.cpp`. This is a precise negative result, not evidence that no historical branch or private implementation exists.

Likewise, the public source establishes `nextn`, `gemma4-assistant`, and `shared_kv_layers` through the paths and commits above, but does not establish a FreeToken Python API, a Hugging Face tensor prefix contract for this checkout, or that the local `nextn.*` inventory has executable-compatible shapes. Therefore the permitted fallback remains: **do not invent names or APIs; keep the adapter disabled until a supplied checkpoint inventory and target-logit/hidden-state probe establish the mapping.** If the probe cannot establish that mapping, use the existing non-speculative Gemma path and leave MTP opt-in off.

## Five card questions

### 1. What is the smallest stateless k=1 design?

Use exactly the current transaction boundary, with one checkpoint-specific Gemma adapter and no new cache owner:

1. The ordinary scheduler admits one request and calls `_prepare_batch`; allocation and attention metadata preparation happen normally (`python/freetoken/scheduler/scheduler.py:762-819`).
2. The adapter receives that prepared batch through the existing `MTPDrafter.draft_into_batch` seam (`python/freetoken/engine/mtp.py:22-24`). Its only permitted model-specific input is the measured target hidden state or other measured checkpoint input; the adapter must not assume such a value is currently exported by FreeToken.
3. For the first implementation, the adapter returns one proposal logit row (`width=1`) and optional probabilities. If the probe proves the llama.cpp-style route, the computation is conceptually: target post-final-norm hidden -> assistant pre-projection with the candidate token embedding -> one assistant layer/block -> assistant post-projection/LM head. The exact tensors, layer count, shared embedding, and KV input remain checkpoint facts to be measured, not API promises.
4. Run the target's ordinary verification callback. `run_k1_transaction` compares greedy proposal and target predictions; on equality it emits the proposal, otherwise it emits the target bonus token (`python/freetoken/engine/mtp.py:51-65`).
5. Commit through the scheduler's existing drain/cache lifecycle. Do not write a second page table, allocate draft KV pages, snapshot Gemma state, or directly call cache free methods from the adapter. The scheduler remains the sole owner (`python/freetoken/engine/mtp.py:3-5,42-49`; `python/freetoken/scheduler/scheduler.py:302-390`).

This is “stateless” only in the MTP sense: the adapter owns no persistent speculative state between calls. It does not mean the target has no KV cache. The upstream evidence suggests that assistant layers may have their own attention/KV requirements, but the current FreeToken checkout does not prove how those would be represented. Until that is measured, the smallest permitted design is a drafter that either uses no assistant KV for k=1 or is rejected as an unimplemented checkpoint-specific adapter; it must not silently allocate a second cache.

### 2. What ratification decisions follow from the evidence?

Ratify the following for M3b:

- **Opt-in only.** Keep normal Gemma serving unchanged. MTP is enabled only when a real checkpoint inventory and executable adapter are supplied.
- **One scheduler/cache owner.** The scheduler owns allocation, page tables, commit, and release. This follows both the current code and the M2 contract.
- **k=1 greedy first.** Do not add rejection sampling, width policy, or `k>1` in M3b. The current proposal has probabilities but the transaction intentionally implements greedy semantics (`python/freetoken/engine/mtp.py:14-20,40-47`).
- **No invented checkpoint format.** `nextn.pre_projection`, `nextn.post_projection`, shared KV, assistant layer counts, token embedding sharing, and target hidden-state plumbing are probe outputs. The local GGUF collector is inventory-only (`python/freetoken/models/gemma4/gguf.py:32-69`).
- **No unconditional target-KV sharing claim.** Upstream's `shared_kv_layers` proves a Gemma target optimization, not that assistant KV can reuse FreeToken target pages. The assistant path must be separately validated.
- **No rollback implementation in M3b.** With one proposal and one target verify, only the existing commit/abort lifecycle is in scope. Any mutable assistant state discovered by the probe is a blocker, not a reason to copy a DeepSeek/DSpark rollback design.
- **Fallback is ordinary Gemma.** If the checkpoint or hidden-state/KV probe fails, keep the flag off and serve through `Gemma4ForCausalLM.forward` (`python/freetoken/models/gemma4/model.py:137-143`).

### 3. What are the blockers?

1. **Checkpoint provenance and tensor mapping:** no supplied real Gemma MTP checkpoint has been validated in this checkout. Need sorted names, shapes, dtypes, config fields, and duplicate/missing-group checks for HF and GGUF.
2. **Target hidden-state access:** upstream explicitly hands `h_nextn` to MTP, but FreeToken's current Gemma forward returns only LM-head logits (`python/freetoken/models/gemma4/model.py:137-143`). Adding such plumbing would be a later source change, not an M3b documentation assumption.
3. **Assistant graph contract:** no FreeToken implementation establishes how `nextn_proj_pre/post`, assistant layers, token embeddings, or output weights map to existing layer classes.
4. **KV semantics:** upstream target layers can reuse earlier KV layers, but FreeToken has full/SWA attention groups and a shared scheduler cache. The assistant's required KV state and whether it can be stateless for k=1 are unmeasured.
5. **Batch/stream ordering:** the scheduler overlaps preparation, forward, and drain (`python/freetoken/scheduler/scheduler.py:197-251`). A future implementation must prove that draft-before-verify does not read a target hidden state before the producing forward is complete.
6. **Exactness and sampling:** M2 is greedy-only. Sampling-equivalent rejection requires proposal probabilities and a specified algorithm; acceptance rate alone is not correctness.
7. **No benchmark checkpoint/hardware:** there is no measured MTP baseline, acceptance, memory, or PCIe result in the current checkout.

### 4. What does M3c need to benchmark?

M3c is a bench-only validation milestone after the blockers are resolved; it is not a source/API milestone. Use the same target checkpoint, prompt corpus, tokenizer, seed, sampling settings, GPU, driver, attention backend, page size, cache size, and batch limits for every arm.

Minimum arms:

- baseline: MTP disabled;
- candidate: MTP enabled with `k=1` and the measured adapter;
- diagnostic: draft-only timing and target-only timing, if the adapter can be isolated without changing serving semantics.

For each arm record:

- target and draft model/checkpoint identifiers plus tensor inventory hash;
- prompt length, generated length, concurrency, batch size, and cache/page geometry;
- TTFT, decode tokens/s, end-to-end latency, and p50/p95 per-token latency;
- drafted, verified, accepted, rejected, and emitted token counts, with acceptance defined as proposal equals target prediction;
- exact token sequence against the disabled-MTP greedy baseline;
- GPU memory, host memory, KV pages, and any assistant state allocation;
- Gemma MoE expert misses/fetches, bytes, copy time, and cache occupancy where applicable;
- CUDA graph/capture status and attention backend.

Required gates: k=1 output must be token-identical to the disabled greedy baseline; abort and request-finish paths must leave no page/table leak; and any speedup must be reported with the full command and hardware. A positive acceptance rate without exactness, cache-integrity, and latency data is not an M3c result.

### 5. What is the recommended next implementation step?

Do not modify the engine or scheduler in M3b. The next permitted implementation unit, after a real checkpoint is available, is a small inventory/probe fixture and a disabled Gemma adapter test. It should:

1. exercise existing config/weight readers to print names, shapes, dtypes, and metadata;
2. verify target-only Gemma logits first;
3. determine whether target post-norm hidden output and assistant inputs can be exposed without changing cache ownership;
4. use a deterministic fake or measured adapter to test the existing `run_k1_transaction` ordering and commit/abort behavior;
5. reject the hypothesis if assistant KV or mutable state needs a new owner/rollback protocol.

That keeps M3b's smallest design falsifiable and preserves the current checkout's explicit fallback rather than inventing a Gemma API.
