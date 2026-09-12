"""Standalone Gemma-4 MTP drafter for the four-block ``nextn`` GGUF head.

This module is intentionally not wired into the scheduler.  A caller supplies the target
hidden state, the embedding of the last token, and the target (usually tied) LM head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .gguf import MTP_METADATA_PREFIX, mtp_gguf_metadata

import torch

from freetoken.engine.mtp import MTPProposal
from freetoken.models.gguf.dequant import dequantize
from freetoken.models.gguf.reader import iter_gguf_tensors


@dataclass
class _Block:
    attn_norm: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor | None
    post_attn_norm: torch.Tensor
    ff_norm: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    post_ff_norm: torch.Tensor | None = None
    output_scale: torch.Tensor | None = None
    head_dim: int = 128


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype) * w


def _linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x @ w.transpose(-1, -2)


def _tensor_map(path: str, prefix: str) -> dict[str, torch.Tensor]:
    result = {}
    for t in iter_gguf_tensors(path):
        if not t.name.startswith(prefix):
            continue
        # nextn projections are small enough to materialize; this still uses the canonical
        # GGUF dequantizer and does not introduce a second quantization path.
        result[t.name[len(prefix):]] = dequantize(
            t.packed().reshape(-1), t.ggml_type, torch.float32
        ).reshape(t.shape)
    return result


def _get(weights: Mapping[str, torch.Tensor], *suffixes: str) -> torch.Tensor:
    for name in suffixes:
        if name in weights:
            return weights[name]
    raise KeyError(f"missing nextn tensor (tried {suffixes!r})")


class GemmaMTPDrafter:
    """Execute the observed four-block Gemma ``nextn`` drafter in isolation.

    ``from_gguf`` loads ``nextn.pre_projection.weight`` [1024, 5632],
    ``nextn.post_projection.weight`` [2816, 1024], the drafter embedding, and
    ``nextn.blk.0`` through ``nextn.blk.3``. ``draft_step`` accepts a target hidden
    state, a last-token ID for lookup in the drafter embedding, and a target LM head.
    ``draft_into_batch`` accepts a batch-like object carrying ``hidden_state``,
    ``last_token_id`` and ``target_lm_head`` attributes. The latter is a convenience
    only; no scheduler state is inspected or modified.
    """

    def __init__(self, draft_step: Callable[..., tuple[Any, Any | None]] | None = None, *, kv_provider=None):
        self._draft_step = draft_step
        self.pre_projection = self.post_projection = self.embedding = None
        self.output_norm = None
        self.blocks: list[_Block] = []
        self.eps = 1e-6
        self.head_dim = 128
        self.num_q_heads = 22
        self.num_kv_heads = 2
        self.rope_freqs = None
        self.rope_base = 10000.0
        self.swa_rope_base = 10000.0
        self.sliding_window = None
        self.target_lm_head = None
        # The assistant inventory intentionally has no K/V tensors. The target owns them.
        self.kv_provider = kv_provider

    @classmethod
    def from_gguf(cls, model_path: str, *, target_lm_head=None, device=None):
        self = cls()
        weights = _tensor_map(model_path, "nextn.")
        root_weights = _tensor_map(model_path, "")
        self.pre_projection = _get(weights, "pre_projection.weight")
        self.post_projection = _get(weights, "post_projection.weight")
        self.output_norm = _get(root_weights, "output_norm.weight")
        # The drafter table follows the GGUF convention and is named at the root;
        # nextn.* owns the projections and blocks.
        self.embedding = _get(root_weights, "token_embd.weight")
        self.rope_freqs = root_weights.get("rope_freqs.weight")
        from freetoken.models.gguf.reader import load_gguf_metadata
        metadata = load_gguf_metadata(model_path)
        mtp_gguf_metadata(tuple(t.name for t in iter_gguf_tensors(model_path)), metadata)
        def meta(key):
            full_key = MTP_METADATA_PREFIX + key
            if full_key not in metadata:
                raise KeyError(f"missing Gemma assistant metadata: {full_key}")
            return metadata[full_key]
        block_count = int(meta("block_count"))
        if block_count != 4:
            raise ValueError(f"Gemma assistant requires 4 blocks, got {block_count}")
        q_heads = int(meta("attention.head_count"))
        kv_heads = meta("attention.head_count_kv")
        pattern = tuple(bool(x) for x in meta("attention.sliding_window_pattern"))
        if len(pattern) != 4:
            raise ValueError("Gemma assistant sliding_window_pattern must have 4 entries")
        swa_dim, full_dim = int(meta("attention.key_length_swa")), int(meta("attention.key_length"))
        self.num_q_heads, self.num_kv_heads = q_heads, int(kv_heads[0] if isinstance(kv_heads, (list, tuple)) else kv_heads)
        self.rope_base = float(metadata.get(MTP_METADATA_PREFIX + "rope.freq_base", self.rope_base))
        self.swa_rope_base = float(metadata.get(MTP_METADATA_PREFIX + "rope.freq_base_swa", self.rope_base))
        self.sliding_window = int(metadata.get(MTP_METADATA_PREFIX + "attention.sliding_window", 0)) or None
        if tuple(self.pre_projection.shape) != (1024, 5632):
            raise ValueError(f"unexpected nextn pre_projection shape {tuple(self.pre_projection.shape)}")
        if tuple(self.post_projection.shape) != (2816, 1024):
            raise ValueError(f"unexpected nextn post_projection shape {tuple(self.post_projection.shape)}")
        if self.embedding.ndim != 2 or self.embedding.shape[1] != 1024:
            raise ValueError(f"unexpected token_embd shape {tuple(self.embedding.shape)}")
        if self.output_norm.ndim != 1 or self.output_norm.shape[0] != self.pre_projection.shape[0]:
            raise ValueError(f"unexpected nextn output_norm shape {tuple(self.output_norm.shape)}")
        if "output.weight" in root_weights:
            raise ValueError("Gemma nextn must not contain output.weight")
        if device is not None:
            self.pre_projection = self.pre_projection.to(device)
            self.post_projection = self.post_projection.to(device)
            self.output_norm = self.output_norm.to(device)
            self.embedding = self.embedding.to(device)
            if self.rope_freqs is not None:
                self.rope_freqs = self.rope_freqs.to(device)
        # Metadata is not repeated in every tensor.  These are the Gemma4 MTP geometry;
        # tensor dimensions below are checked against it while loading each block.
        for layer in range(4):
            p = f"blk.{layer}."
            def g(*names): return _get(weights, *(p + n for n in names))
            block = _Block(
                g("attn_norm.weight"), g("attn_q_norm.weight"), None,
                g("post_attention_norm.weight"), g("ffn_norm.weight"),
                g("attn_q.weight"), None, None, g("attn_output.weight"),
                g("ffn_gate.weight"), g("ffn_up.weight"), g("ffn_down.weight"),
                post_ff_norm=(g("post_ffw_norm.weight") if any(p + n in weights for n in ("post_ffw_norm.weight",)) else None),
                output_scale=(g("layer_output_scale.weight") if p + "layer_output_scale.weight" in weights else None),
                head_dim=(swa_dim if pattern[layer] else full_dim),
            )
            expected_q = 4096 if layer < 3 else 8192
            if block.q.shape[0] != expected_q or block.q.shape[1] != 1024:
                raise ValueError(f"nextn block {layer} Q must have shape ({expected_q}, 1024)")
            if block.head_dim != (swa_dim if layer < 3 else full_dim):
                raise ValueError(f"nextn block {layer} has inconsistent attention geometry")
            if block.q.shape[0] % block.head_dim:
                raise ValueError(f"nextn block {layer} has incompatible attention geometry")
            if block.q.shape[1] != self.pre_projection.shape[0]:
                raise ValueError(f"nextn block {layer} projections must consume hidden size {self.pre_projection.shape[0]}")
            self.blocks.append(block)
        self.target_lm_head = target_lm_head
        return self

    def _rope(self, x, positions, layer):
        n = x.shape[-1]
        rot = n
        base = self.swa_rope_base if layer < 3 else self.rope_base
        inv = 1.0 / (base ** (torch.arange(0, rot, 2, device=x.device).float() / rot))
        if layer == 3 and self.rope_freqs is not None:
            divisors = self.rope_freqs.to(device=x.device, dtype=inv.dtype).flatten()
            if divisors.numel() < inv.numel():
                raise ValueError("rope_freqs.weight is shorter than the full-attention head")
            inv = inv / divisors[: inv.numel()]
        phase = positions[:, None].float() * inv[None]
        c, s = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
        c, s = c[:, None, :], s[:, None, :]
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2)

    def _run(self, hidden, last_embedding, positions):
        x = _linear(torch.cat((hidden, last_embedding), -1), self.pre_projection)
        for i, b in enumerate(self.blocks):
            residual = x
            h = _rms(x, b.attn_norm, self.eps)
            num_q = b.q.shape[0] // b.head_dim
            num_kv = self.num_kv_heads
            q = _linear(h, b.q).view(-1, num_q, b.head_dim)
            if self.kv_provider is None:
                raise RuntimeError("Gemma assistant requires target K/V through kv_provider")
            k, v = self.kv_provider(i, h, b.head_dim, num_kv)
            k = k.view(-1, num_kv, b.head_dim)
            v = v.view(-1, num_kv, b.head_dim)
            k = self._rope(k, positions, i)
            q = _rms(q, b.q_norm, self.eps)
            q = self._rope(q, positions, i)
            scores = torch.einsum("thd,shd->hts", q, k) / b.head_dim**0.5
            if i < 3 and self.sliding_window:
                d = positions[:, None] - positions[None, :]
                scores = scores.masked_fill((d < 0) | (d >= self.sliding_window), -torch.inf)
            else:
                scores = scores.masked_fill(positions[None, :] > positions[:, None], -torch.inf)
            probs = scores.softmax(-1)
            v = v.repeat_interleave(num_q // num_kv, 1)
            attn = torch.einsum("hts,shd->thd", probs, v).reshape_as(h)
            h = _rms(_linear(attn, b.o), b.post_attn_norm, self.eps)
            x = residual + h
            h = _rms(x, b.ff_norm, self.eps)
            ff = torch.nn.functional.gelu(_linear(h, b.gate), approximate="tanh") * _linear(h, b.up)
            ff = _linear(ff, b.down)
            if b.post_ff_norm is not None:
                ff = _rms(ff, b.post_ff_norm, self.eps)
            x = x + ff
            if b.output_scale is not None:
                x = x * b.output_scale.reshape(1, -1)
        return _linear(_rms(x, self.output_norm, self.eps), self.post_projection)

    @torch.inference_mode()
    def draft_step(self, hidden_state, last_token_id=None, target_lm_head=None, positions=None, *, embedding=None, kv_provider=None):
        if self._draft_step is not None:
            return self._draft_step(hidden_state)
        if self.pre_projection is None:
            raise RuntimeError("GemmaMTPDrafter is not loaded")
        if positions is None:
            positions = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        if embedding is None:
            raise ValueError("draft_step requires the target-provided embedding")
        last_embedding = embedding
        if kv_provider is not None:
            self.kv_provider = kv_provider
        hidden = self._run(hidden_state, last_embedding, positions)
        head = target_lm_head or self.target_lm_head
        if head is None:
            raise ValueError("draft_step requires the target LM head (Gemma has no nextn output head)")
        logits = head.forward(hidden) if hasattr(head, "forward") else head(hidden)
        return logits, None

    def draft_into_batch(self, batch: Any) -> MTPProposal:
        if self._draft_step is not None:
            logits, probabilities = self._draft_step(batch)
        else:
            get = batch.get if isinstance(batch, Mapping) else getattr
            hidden = get("hidden_state") if isinstance(batch, Mapping) else get(batch, "hidden_state")
            token_id = get("last_token_id") if isinstance(batch, Mapping) else get(batch, "last_token_id")
            head = (batch.get("target_lm_head") if isinstance(batch, Mapping) else getattr(batch, "target_lm_head", None))
            logits, probabilities = self.draft_step(hidden, token_id, head)
        return MTPProposal(logits=logits, probabilities=probabilities, width=1)


__all__ = ["GemmaMTPDrafter"]
