# Gemma-4 MTP port plan

This is a source-grounded plan for adding multi-token prediction (MTP) to Gemma-4. It is not a claim that a Gemma MTP checkpoint or API is already supported. Source facts are marked by citations; acceptance numbers and checkpoint details must be measured.

## Reference points and scope

- PR69 base: [`f50845a`](../commit/f50845a), `feat(dsv4): implement exact DSpark speculative decoding`.
- PR69 follow-up: [`2cf938b`](../commit/2cf938b), `fix(dsv4): make hybrid DSpark serving operational`.
- PR70: [`c066b38`](../commit/c066b38), `feat(moe): bound parallel expert prefetch`.
- Current branch base: [`9535656`](../commit/9535656).

Only this document should be changed for the planning commit. The seed queue is an input source for proposals, not a stopping rule: keep accepting and measuring seeds until the configured request is complete or the request is cancelled.

## 1. What PR69 provides: generic versus DSV4-only

### Reusable transaction mechanisms

The following are useful as engine/scheduler mechanisms, subject to extracting model-independent names and contracts:

1. **A verify-shaped batch.** `scheduler/scheduler.py@f50845a` marks a decode batch speculative and reserves `1 + k` token positions. The target then runs one ordinary forward over the prepared positions. This is the smallest useful transaction: proposal, target verification, commit of a prefix, and discard of the suffix.
2. **Draft-before-verify ordering.** `engine/engine.py@f50845a` (`draft_into_batch`) writes proposal token IDs into the prepared batch and retains draft probabilities/confidence for finalization. A future drafter should return distributions, not only argmax IDs, if sampling exactness is required.
3. **Prefix commit and bonus-token handling.** `engine/engine.py@f50845a` (`_finish_speculative`) and `scheduler/scheduler.py@f50845a` record the accepted prefix and target-produced next token. The scheduler appends emitted tokens and frees/reconciles resources at its normal post-forward point.
4. **A pluggable loop shape.** `models/deepseek_v4/dspark.py@f50845a` (`SpeculativeLoop`) separates `draft`, `verify`, `snapshot`, and `restore` callbacks. The callback shape is reusable; its default policy must not assume DSV4 state.
5. **Metrics hooks.** The engine's drafted/accepted counters in `engine/engine.py@f50845a` are generic enough to retain, with model-neutral names and explicit definitions (`drafted`, `verified`, `accepted`, and emitted tokens).
6. **Bounded allocation discipline.** The existing scheduler cache lifecycle (`scheduler/cache.py@9535656`) already allocates pages before a forward and reconciles them after completion/abort. A Gemma transaction should reuse this lifecycle rather than add a second page owner.

### DSV4/DSpark semantics that must not be generalized silently

These are DeepSeek-V4-specific and remain behind a DSV4 adapter:

- `DSparkDrafter`, its `mtp.*` block structure, draft-layer KV projection/catch-up, noise token, Markov head, confidence head, and DSV4 layer numbering in `models/deepseek_v4/dspark.py@f50845a`.
- DSV4 rejection sampling (`sampling_probs`, `rejection_accept`) and confidence-derived width policy in `models/deepseek_v4/dspark.py@f50845a`. A Gemma policy must be independently specified and tested.
- Compressor/indexer carry snapshots and page-local rollback in `models/deepseek_v4/rollback.py@f50845a`. Plain Gemma attention has no such source-established carry; do not copy this snapshot or infer that KV pages alone make every future state rollback-safe.
- DSV4 attention types, sparse masking, compressor geometry, cost model, and DSV4 paged pools in `attention/dsv4_*`, `kvcache/dsv4_*`, and `models/deepseek_v4/*` at `f50845a`/`2cf938b`.
- DSV4 auxiliary addressing, hybrid CPU/GPU layer routing, and operational serving fixes in `2cf938b`. The latter also changes `moe/offload_cache.py` for DSV4 layer residency; it is not evidence that Gemma has the same layout.
- `scripts/freetoken-dsv4.sh` and DSV4-specific CLI/config flags at `f50845a`/`2cf938b`.

PR70 is separately reusable infrastructure, not MTP: bounded parallel expert shard read-ahead (`expert_prefetch`, default 2), RAM admission, and provider forwarding in `models/weight.py`, `moe/expert_banks.py`, and `models/deepseek_v4/weight.py@c066b38`. It may improve startup or reduce host-memory risk, but it does not implement proposal, verification, rollback, or PCIe accounting.

## 2. Gemma-4 drafter and weight-format decision

### What the current implementation establishes

`models/gemma4/model.py@9535656` has only `Gemma4ForCausalLM`: embedding, `Gemma4DecoderLayer` stack, final norm, and LM head. `models/gemma4/config.py@9535656` parses hybrid full-attention/SWA geometry, partial rotary full layers, optional vision, and dense or MoE variants. `models/gemma4/moe.py@9535656` defines a shared gated MLP plus routed experts, with Gemma-specific per-expert scaling and dual RMSNorm/residual fusion. There is no `draft` member, MTP config field, or Gemma MTP forward API in these files.

`models/gemma4/weight.py@9535656` supports HF safetensors key renaming and qkv/gate-up merging, native modelopt NVFP4 dense-MLP handling, and separate expert-bank loading. It explicitly rejects TP greater than 1. `models/gemma4/gguf.py@9535656` supports the observed GGUF metadata/tensor names, Q4_0 routed experts through offload, and native packed non-expert weights, also TP=1. These facts constrain implementation but do not prove the format of a future MTP checkpoint.

### Decision: do not choose a format until probes identify one

The first Gemma drafter should be a separate module/config path only after a real checkpoint is inspected. Do not assume that `mtp.*`, `model.mtp.*`, a second full model, shared embeddings, a noise token, or a particular number of draft layers exists. The loader must first report:

1. architecture/config keys and any MTP/draft declaration;
2. all tensor-name prefixes, layer count, dimensions, dtypes, and tensor shapes;
3. whether draft weights are dense, routed, quantized, or shared with target weights;
4. whether the checkpoint has a target-only or draft-specific embedding/LM head;
5. whether HF safetensors and GGUF represent the same tensors.

**Validation probes before broad implementation:**

- Add a no-production-change inspection script/test fixture that runs the existing `cached_load_hf_config`, `iter_weight_files`/`ShardReader`, and (for GGUF) `load_gguf_metadata`/`iter_gguf_tensors` against a supplied local checkpoint. It must print sorted names, shapes, dtypes, and missing/duplicate groups without loading a full model.
- Construct the current target config and model with dummy weights; assert the real checkpoint's target tensors still map through the current Gemma loader. Separately attempt a draft-only name/shape inventory. A missing draft inventory is a documented stop condition, not permission to invent names.
- Run a one-token target-vs-reference logits probe and a two-token target forward with the actual attention groups and page table. Record tolerances, token IDs, and whether target KV can be reused by a prospective drafter. These are measurements, not source facts.

The format decision is therefore: **reuse target embedding/LM head and target KV only if the probe proves the checkpoint and computation require it; otherwise model the draft as an explicitly separate weight set.** Quantization must follow the observed tensor metadata and existing `weight.py`/`gguf.py` contracts, not a guessed Gemma API.

## 3. Milestones

1. **Probe and baseline.** Capture checkpoint inventory, target logits, memory, and per-layer timing. Pass only when the target-only Gemma path remains unchanged and the inventory is reviewable.
2. **Smallest end-to-end transaction.** With a dummy or measured drafter, propose one token for one request, run target verification, accept/reject it, emit exactly one token, and return to the ordinary scheduler. No sampling optimization, CUDA graph, or throughput claim yet.
3. **Exactness and sampling.** Add width `k > 1`, greedy comparison, then distributional rejection sampling only if the draft probabilities are available. Compare a speculative run with speculation disabled under identical seed, sampling parameters, prompt, and checkpoint. Verify emitted token sequence and target logits at committed positions.
4. **Rollback.** First prove that Gemma's state consists only of page-addressed target KV plus scheduler/request state for the chosen attention groups. If a mutable state is found, add a narrowly scoped snapshot/restore for that state; do not use `CarrySnapshot` unless the state has the same hazard. Test rejection at page boundaries, within a page, full rejection, full acceptance, abort, and chunk continuation.
5. **PCIe and cache instrumentation.** Use `OffloadMoeCache`'s existing `decode_miss_stats`, per-layer stats, routing stats, and prefill hit rows (`moe/offload_cache.py@9535656`) as the baseline. Add transaction labels/counters only where needed to distinguish draft and verify traffic. Measure expert misses, fetched experts, CPU experts, bytes, copy time, and cache occupancy; do not infer these from token counts. PR70's bounded `expert_prefetch` (`c066b38`) is a controlled startup variable, not a runtime MTP result.
6. **Throughput and tuning.** Compare disabled speculation, `k=1`, and larger widths across prompt lengths, batch sizes, cache sizes, MoE backends, and hit/miss regimes. Tune only after exactness and rollback pass. The result must report tokens/s, TTFT, acceptance, verified/drafted ratio, GPU memory, host memory, PCIe bytes/time, and p50/p95 latency.

## 4. Tests and acceptance criteria

- **Inventory/loader:** fixture tests for every observed target and draft tensor group; reject incomplete merges and unsupported/ambiguous formats. No test may assert an invented key prefix.
- **Architecture:** target logits match a trusted reference for dense and MoE Gemma fixtures; attention-group routing, partial rotary geometry, router scaling, and shared/routed feed-forward outputs retain current tolerances. Extend `tests/kernels/test_gemma4_fused_ops.py@9535656` and `tests/models/test_gemma4_gguf_rope.py@9535656` only with measured cases.
- **Transaction:** unit-test proposal, verify, accept-prefix, reject, bonus token, cancellation, and open-ended seed production. Assert no duplicate or skipped committed token and no second owner of a page.
- **Exactness:** greedy output is byte-for-byte/token-for-token equal to non-speculative output. For sampling, compare a fixed-seed reference distribution with the specified rejection-sampling result; acceptance alone is insufficient.
- **Cache safety:** after every accept/reject/abort/chunk boundary, assert the existing cache invariants (`tests/scheduler/test_cache_rebuild.py`, `test_commit_repoints_page_table.py`, and `tests/kvcache/test_*`) and any Gemma-specific state invariant. No negative page IDs, leaks, stale KV, or use-after-free.
- **Offload correctness:** retain `tests/moe/test_offload.py@9535656` coverage for slot remapping/materialization; add a transaction test that verifies draft and target see the intended expert rows and that counters distinguish misses from fetches.
- **Performance gate:** publish A/B measurements on the same GPU, driver, checkpoint, prompt set, batch settings, and command. Speculation is accepted only if exactness passes and the measured workload improves the agreed throughput/latency target without exceeding the agreed memory and host-RAM budgets. No threshold is declared until hardware and checkpoint are named.

## 5. First single-hypothesis code change

**Hypothesis:** Gemma can use the existing model-neutral `1+k` scheduler transaction if its drafter is exposed as a narrow `draft_into_batch`-compatible adapter, while target KV/page ownership stays in the existing scheduler. This isolates the unknown (how Gemma proposes tokens) from known transaction mechanics.

The first change should be one small, test-backed adapter boundary, not a Gemma loader rewrite: add a Gemma-side `draft_step` interface (or equivalent callback object) that accepts the already prepared request batch and returns proposal logits/probabilities plus an explicit drafted width. The adapter must be unimplemented/disabled unless the checkpoint probe supplies a real draft component; it must not manufacture a second model, names, noise token, or checkpoint format. Add a unit test with a deterministic fake drafter proving the existing one-token transaction calls draft before verify and commits exactly the accepted prefix. Keep the change out of production Gemma execution until the probe has supplied the concrete implementation.

This is deliberately narrower than adding a full MTP implementation: it tests the reusable PR69 transaction contract first, leaves DSV4 semantics in the DSV4 path, and gives the next change a falsifiable result. If the probe shows Gemma requires mutable state or a different draft input, reject this hypothesis before adding weights or kernels.
