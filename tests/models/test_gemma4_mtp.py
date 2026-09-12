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
    names = [
        "blk.1.attn_norm.weight",
        "nextn.post_projection.weight",
        "nextn.blk.0.attn_norm.weight",
        "blk.0.attn_norm.weight",
        "token_embd.weight",
        "nextn.pre_projection.weight",
        "unrelated.tensor",
    ]
    metadata = {
        "gemma4.attention.head_count_kv": 8,
        "gemma4.attention.head_count": 32,
        "other.attention.head_count_kv": 4,
    }

    inventory = mtp_gguf_metadata(names, metadata)

    assert inventory == GemmaMTPGGUFMetadata(
        token_embd=("token_embd.weight",),
        nextn_pre_projection=("nextn.pre_projection.weight",),
        nextn_post_projection=("nextn.post_projection.weight",),
        block_groups=("blk.0", "blk.1", "nextn.blk.0"),
        shared_kv=(
            "gemma4.attention.head_count_kv",
            "other.attention.head_count_kv",
        ),
    )


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
    assert tuple(drafter.embedding.shape) == (262144, 2816)


def test_mtp_final_norm_precedes_post_projection():
    drafter = GemmaMTPDrafter()
    drafter.pre_projection = torch.zeros(1024, 5632)
    drafter.pre_projection[:, :1024] = torch.eye(1024)
    drafter.post_projection = torch.zeros(2816, 1024)
    drafter.post_projection[:1024] = torch.eye(1024)
    drafter.output_norm = torch.full((2816,), 2.0)
    drafter.blocks = []
    hidden = torch.ones(1, 2816)
    result = drafter._run(hidden, torch.zeros(1, 2816), torch.zeros(1, dtype=torch.long))
    torch.testing.assert_close(result[:, :1024], torch.full((1, 1024), 2.0))
    assert torch.count_nonzero(result[:, 1024:]) == 0


def test_fake_input_forward_uses_caller_target_vocab_head():
    drafter = GemmaMTPDrafter()
    drafter.pre_projection = torch.zeros(1024, 5632)
    drafter.post_projection = torch.zeros(2816, 1024)
    drafter.output_norm = torch.ones(2816)
    drafter.embedding = torch.zeros(32, 2816)
    ones = torch.ones(2816)
    zero = torch.zeros(2816, 2816)
    q = torch.zeros(2816, 2816)
    k = torch.zeros(256, 2816)
    block = _Block(ones, torch.ones(128), torch.ones(128), ones, ones, q, k, k, zero,
                   zero, zero, zero)
    drafter.blocks = [block, block, block, block]
    hidden = torch.zeros(1, 2816)
    embedding = torch.zeros(1, 2816)
    target_head = torch.nn.Linear(2816, 17, bias=False)
    logits, probabilities = drafter.draft_step(hidden, torch.tensor([0]), target_head)
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
