from __future__ import annotations

from freetoken.models.gemma4 import (
    GemmaMTPDrafter,
    GemmaMTPGGUFMetadata,
    mtp_gguf_metadata,
)


def test_synthetic_gguf_mtp_inventory_groups_are_frozen():
    names = [
        "blk.1.attn_norm.weight",
        "nextn.post_projection.weight",
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
        block_groups=("blk.0", "blk.1"),
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


def test_speculative_mtp_is_off_by_default_and_parsed_from_server_args(tmp_path):
    """parse_args resolves --model-path through the hub layer, so feed it a real
    (empty) directory; the flag itself needs no weights to parse."""
    from freetoken.server.args import parse_args

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    argv = ["--model-path", str(model_dir)]

    default, _ = parse_args(argv)
    enabled, _ = parse_args(argv + ["--speculative-mtp"])

    assert default.speculative_mtp is False
    assert enabled.speculative_mtp is True
