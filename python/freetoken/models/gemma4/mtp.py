"""Standalone Gemma-4 MTP drafter for the four-block ``nextn`` GGUF head.

This module is intentionally not wired into the scheduler.  A caller supplies the target
hidden state, the embedding of the last token, and the target (usually tied) LM head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from freetoken.engine.mtp import MTPProposal
from freetoken.models.gguf.dequant import dequantize
from freetoken.models.gguf.reader import iter_gguf_tensors


@dataclass
class _Block:
    attn_norm: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
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
    ``nextn.blk.0`` through ``nextn.blk.3``.  ``draft_step`` accepts either explicit
    tensors or a batch-like object carrying ``hidden_state``, ``last_token_embedding``
    and ``target_lm_head`` attributes.  The latter is a convenience only; no scheduler
    state is inspected or modified.
    """

    def __init__(self, draft_step: Callable[..., tuple[Any, Any | None]] | None = None):
        self._draft_step = draft_step
        self.pre_projection = self.post_projection = self.embedding = None
        self.blocks: list[_Block] = []
        self.eps = 1e-6
        self.head_dim = 128
        self.num_q_heads = 22
        self.num_kv_heads = 2
        self.sliding_window = None
        self.target_lm_head = None

    @classmethod
    def from_gguf(cls, model_path: str, *, target_lm_head=None, device=None):
        self = cls()
        weights = _tensor_map(model_path, "nextn.")
        root_weights = _tensor_map(model_path, "")
        self.pre_projection = _get(weights, "pre_projection.weight")
        self.post_projection = _get(weights, "post_projection.weight")
        # The drafter table follows the GGUF convention and is named at the root;
        # nextn.* owns the projections and blocks.
        self.embedding = _get(root_weights, "token_embd.weight")
        if tuple(self.pre_projection.shape) != (1024, 5632):
            raise ValueError(f"unexpected nextn pre_projection shape {tuple(self.pre_projection.shape)}")
        if tuple(self.post_projection.shape) != (2816, 1024):
            raise ValueError(f"unexpected nextn post_projection shape {tuple(self.post_projection.shape)}")
        if self.embedding.ndim != 2 or self.embedding.shape[1] != 1024:
            raise ValueError(f"unexpected nextn token_embd shape {tuple(self.embedding.shape)}")
        if "output.weight" in root_weights:
            raise ValueError("Gemma nextn must not contain output.weight")
        if device is not None:
            self.pre_projection = self.pre_projection.to(device)
            self.post_projection = self.post_projection.to(device)
            self.embedding = self.embedding.to(device)
        # Metadata is not repeated in every tensor.  These are the Gemma4 MTP geometry;
        # tensor dimensions below are checked against it while loading each block.
        for layer in range(4):
            p = f"blk.{layer}."
            def g(*names): return _get(weights, *(p + n for n in names))
            block = _Block(
                g("attn_norm.weight"), g("attn_q_norm.weight"), g("attn_k_norm.weight"),
                g("post_attention_norm.weight"), g("ffn_norm.weight"),
                g("attn_q.weight"), g("attn_k.weight"),
                g("attn_v.weight", "attn_k.weight"), g("attn_output.weight"),
                g("ffn_gate.weight"), g("ffn_up.weight"), g("ffn_down.weight"),
                post_ff_norm=(g("post_ffw_norm.weight") if any(p + n in weights for n in ("post_ffw_norm.weight",)) else None),
                output_scale=(g("layer_output_scale.weight") if p + "layer_output_scale.weight" in weights else None),
            )
            if block.q.shape[0] % self.head_dim or block.k.shape[0] % self.head_dim:
                raise ValueError(f"nextn block {layer} has incompatible attention geometry")
            if block.q.shape[1] != 2816 or block.k.shape[1] != 2816:
                raise ValueError(f"nextn block {layer} projections must consume hidden size 2816")
            self.blocks.append(block)
        self.target_lm_head = target_lm_head
        return self

    def _rope(self, x, positions):
        n = x.shape[-1]
        rot = n  # nextn SWA/full dimensions are fully rotary; full uses the same head width.
        inv = 1.0 / (10000 ** (torch.arange(0, rot, 2, device=x.device).float() / rot))
        phase = positions[:, None].float() * inv[None]
        c, s = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
        c, s = c[:, None, :], s[:, None, :]
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2)

    def _run(self, hidden, last_embedding, positions):
        x = _linear(torch.cat((hidden, last_embedding), -1), self.pre_projection)
        x = _linear(x, self.post_projection)
        for i, b in enumerate(self.blocks):
            residual = x
            h = _rms(x, b.attn_norm, self.eps)
            num_q = b.q.shape[0] // self.head_dim
            num_kv = b.k.shape[0] // self.head_dim
            q = _linear(h, b.q).view(-1, num_q, self.head_dim)
            k = _linear(h, b.k).view(-1, num_kv, self.head_dim)
            v = _linear(h, b.v).view(-1, num_kv, self.head_dim)
            q = _rms(q, b.q_norm, self.eps); k = _rms(k, b.k_norm, self.eps)
            q, k = self._rope(q, positions), self._rope(k, positions)
            scores = torch.einsum("thd,shd->hts", q, k) / self.head_dim**0.5
            if i < 3 and self.sliding_window:
                d = positions[:, None] - positions[None, :]
                scores = scores.masked_fill((d < 0) | (d >= self.sliding_window), -torch.inf)
            else:
                scores = scores.masked_fill(positions[None, :] > positions[:, None], -torch.inf)
            probs = scores.softmax(-1)
            v = v.repeat_interleave(num_q // num_kv, 1)
            attn = torch.einsum("hts,shd->thd", probs, v).reshape_as(h)
            x = residual + _linear(attn, b.o)
            h = _rms(x, b.post_attn_norm, self.eps)
            ff = torch.nn.functional.gelu(_linear(h, b.gate), approximate="tanh") * _linear(h, b.up)
            ff = _linear(ff, b.down)
            if b.post_ff_norm is not None:
                ff = _rms(ff, b.post_ff_norm, self.eps)
            x = x + ff
            if b.output_scale is not None:
                x = x * b.output_scale.reshape(1, -1)
        return x

    @torch.inference_mode()
    def draft_step(self, hidden_state, last_token_embedding, target_lm_head=None, positions=None):
        if self._draft_step is not None:
            return self._draft_step(hidden_state)
        if self.pre_projection is None:
            raise RuntimeError("GemmaMTPDrafter is not loaded")
        if positions is None:
            positions = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        hidden = self._run(hidden_state, last_token_embedding, positions)
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
            emb = get("last_token_embedding") if isinstance(batch, Mapping) else get(batch, "last_token_embedding")
            head = (batch.get("target_lm_head") if isinstance(batch, Mapping) else getattr(batch, "target_lm_head", None))
            logits, probabilities = self.draft_step(hidden, emb, head)
        return MTPProposal(logits=logits, probabilities=probabilities, width=1)


__all__ = ["GemmaMTPDrafter"]
