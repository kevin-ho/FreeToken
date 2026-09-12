from __future__ import annotations

from freetoken.models.gemma4 import (
    GemmaMTPDrafter,
    GemmaMTPGGUFMetadata,
    mtp_gguf_metadata,
)
from freetoken.models.gemma4.mtp import _Block


import json
import os

import pytest
import torch


def test_synthetic_gguf_mtp_inventory_groups_are_frozen():
    from freetoken.models.gemma4.gguf import MTP_BLOCK_FIELDS, MTP_ROOT_TENSORS
    names = list(MTP_ROOT_TENSORS) + [
        f"nextn.blk.{layer}.{field}.weight"
        for layer in range(4) for field in MTP_BLOCK_FIELDS
    ]
    metadata = {
        f"gemma4-assistant.{key}": ([True, True, True, False] if key == "attention.sliding_window_pattern" else [8, 8, 8, 2] if key == "attention.head_count_kv" else 4 if key == "block_count" else 32 if key == "attention.head_count" else 128 if key in ("attention.key_length", "attention.key_length_swa") else 4096 if key == "attention.sliding_window" else 10000.0)
        for key in ("block_count", "attention.head_count", "attention.head_count_kv", "attention.key_length", "attention.key_length_swa", "attention.sliding_window_pattern", "attention.sliding_window", "rope.freq_base", "rope.freq_base_swa")
    }

    inventory = mtp_gguf_metadata(names, metadata)
    assert inventory == GemmaMTPGGUFMetadata(
        token_embd=("token_embd.weight",),
        nextn_pre_projection=("nextn.pre_projection.weight",),
        nextn_post_projection=("nextn.post_projection.weight",),
        block_groups=("nextn.blk.0", "nextn.blk.1", "nextn.blk.2", "nextn.blk.3"),
        shared_kv=(),
    )


def test_gemma_mtp_inventory_rejects_drift_and_missing_metadata():
    from freetoken.models.gemma4.gguf import MTP_TENSOR_INVENTORY
    metadata = {"gemma4-assistant." + key: 1 for key in ("block_count", "attention.head_count", "attention.head_count_kv", "attention.key_length", "attention.key_length_swa", "attention.sliding_window_pattern", "attention.sliding_window", "rope.freq_base", "rope.freq_base_swa")}
    with pytest.raises(ValueError):
        mtp_gguf_metadata(MTP_TENSOR_INVENTORY | {"nextn.blk.0.attn_k.weight"}, metadata)
    with pytest.raises(KeyError):
        mtp_gguf_metadata(MTP_TENSOR_INVENTORY, {})


def test_gemma_mtp_public_adapter_delegates_without_inventing_weights():
    batch = object()
    seen = []

    def draft_step(value):
        seen.append(value)
        return "logits", "probabilities"

    proposal = GemmaMTPDrafter(draft_step).draft_into_batch(batch)

    assert seen == [batch]
    assert proposal.logits == "logits"
    assert proposal.probabilities == "probabilities"
    assert proposal.width == 1


@pytest.mark.needs_weights
def test_real_gemma_mtp_checkpoint_shape_probe_skips_when_absent():
    path = os.environ.get("FREETOKEN_TEST_MTP_GGUF")
    if not path or not os.path.isfile(path):
        pytest.skip("set FREETOKEN_TEST_MTP_GGUF to a real Gemma MTP GGUF")
    drafter = GemmaMTPDrafter.from_gguf(path)
    assert tuple(drafter.pre_projection.shape) == (1024, 5632)
    assert tuple(drafter.post_projection.shape) == (2816, 1024)
    assert tuple(drafter.embedding.shape) == (262144, 1024)


def test_mtp_final_norm_precedes_post_projection():
    drafter = GemmaMTPDrafter()
    drafter.pre_projection = torch.zeros(1024, 5632)
    drafter.pre_projection[:, :1024] = torch.eye(1024)
    drafter.post_projection = torch.zeros(2816, 1024)
    drafter.post_projection[:1024] = torch.eye(1024)
    drafter.output_norm = torch.full((1024,), 2.0)
    drafter.blocks = []
    hidden = torch.ones(1, 2816)
    result = drafter._run(hidden, torch.zeros(1, 2816), torch.zeros(1, dtype=torch.long))
    torch.testing.assert_close(result[:, :1024], torch.full((1, 1024), 2.0))
    assert torch.count_nonzero(result[:, 1024:]) == 0


def test_fake_input_forward_uses_caller_target_vocab_head():
    drafter = GemmaMTPDrafter()
    drafter.pre_projection = torch.zeros(1024, 5632)
    drafter.post_projection = torch.zeros(2816, 1024)
    drafter.output_norm = torch.ones(1024)
    drafter.embedding = torch.zeros(32, 1024)
    ones = torch.ones(1024)
    zero = torch.zeros(1024, 4096)
    q = torch.zeros(4096, 1024)
    block = _Block(ones, torch.ones(256), None, ones, ones, q, None, None,
                   torch.zeros(1024, 4096), zero, zero, zero, head_dim=256)
    drafter.blocks = [block, block, block, block]
    drafter.attention_pattern = (False, False, False, False)
    drafter.num_kv_heads_by_layer = (8, 8, 8, 2)
    hidden = torch.zeros(1, 2816)
    embedding = torch.zeros(1, 2816)
    target_head = torch.nn.Linear(2816, 17, bias=False)
    drafter.num_kv_heads = 2
    logits, probabilities = drafter.draft_step(
        hidden, target_lm_head=target_head, embedding=embedding,
        kv_provider=lambda _layer, _hidden, head_dim, kv_heads: (
            torch.zeros(1, head_dim * kv_heads), torch.zeros(1, head_dim * kv_heads)
        ),
    )
    assert logits.shape == (1, 17)
    assert probabilities is None


def test_speculative_mtp_is_off_by_default_and_parsed_from_server_args(tmp_path):
    """parse_args resolves --model-path through the hub layer, so feed it a real
    (empty) directory; the flag itself needs no weights to parse."""
    from freetoken.server.args import parse_args

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "gemma4", "architectures": ["Gemma4ForCausalLM"]})
    )
    argv = ["--model-path", str(model_dir)]

    default, _ = parse_args(argv)
    enabled, _ = parse_args(argv + ["--speculative-mtp"])

    assert default.speculative_mtp is False
    assert enabled.speculative_mtp is True
