from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pytest
import torch

safetensors = pytest.importorskip("safetensors")


@pytest.fixture
def probe():
    path = Path(__file__).parents[1] / "scripts" / "probe_checkpoint_inventory.py"
    spec = importlib.util.spec_from_file_location("probe_checkpoint_inventory", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_inventory_reports_config_flags_shards_and_prefixes(tmp_path, probe):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "synthetic",
                "hidden_size": 4,
                "mtp_config": {"num_draft_layers": 2},
                "architectures": ["SyntheticForCausalLM"],
            }
        )
    )
    safetensors.torch.save_file(
        {
            "model.layers.1.weight": torch.zeros((4, 4), dtype=torch.float16),
            "model.mtp.draft.weight": torch.ones((4,), dtype=torch.float32),
        },
        str(tmp_path / "model-00002-of-00002.safetensors"),
    )
    safetensors.torch.save_file(
        {"lm_head.weight": torch.zeros((8, 4), dtype=torch.bfloat16)},
        str(tmp_path / "model-00001-of-00002.safetensors"),
    )

    report = probe.inventory(str(tmp_path))

    assert report["format"] == "hf-safetensors"
    assert "mtp_config" in report["flagged_config_keys"]
    assert [shard["file"] for shard in report["shards"]] == sorted(
        shard["file"] for shard in report["shards"]
    )
    tensors = [tensor for shard in report["shards"] for tensor in shard["tensors"]]
    assert tensors == sorted(tensors, key=lambda tensor: tensor["name"])
    assert {tensor["dtype"] for tensor in tensors} == {"BF16", "F16", "F32"}
    assert report["prefix_groups"]["model"] == [
        "model.layers.1.weight",
        "model.mtp.draft.weight",
    ]


def _write_tiny_gguf(path: Path) -> None:
    def string(value: str) -> bytes:
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    metadata = b"".join(
        [
            string("general.architecture") + struct.pack("<I", 8) + string("synthetic"),
            string("model.mtp.enabled") + struct.pack("<I", 7) + b"\x01",
        ]
    )
    tensors = [("z.weight", (4, 4)), ("a.weight", (4,))]
    tensor_infos = []
    payloads = []
    offset = 0
    for name, shape in tensors:
        payload = struct.pack("<" + "f" * (shape[0] * (shape[1] if len(shape) > 1 else 1)), *range(shape[0] * (shape[1] if len(shape) > 1 else 1)))
        tensor_infos.append(string(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", dim) for dim in shape) + struct.pack("<IQ", 0, offset))
        payloads.append(payload)
        offset += len(payload)
    header = struct.pack("<4sQQQ", b"GGUF", 3, len(tensors), 2) + metadata + b"".join(tensor_infos)
    data_start = (len(header) + 31) // 32 * 32
    path.write_bytes(header + b"\0" * (data_start - len(header)) + b"".join(payloads))


def test_gguf_report_keeps_all_metadata_and_sorts_tensors(tmp_path, probe):
    pytest.importorskip("gguf")
    path = tmp_path / "synthetic.gguf"
    _write_tiny_gguf(path)

    report = probe.inventory(str(path))

    assert report["metadata"]["general.architecture"] == "synthetic"
    assert report["flagged_metadata_keys"] == ["model.mtp.enabled"]
    assert [tensor["name"] for tensor in report["tensors"]] == ["a.weight", "z.weight"]
    assert report["prefix_groups"] == {"a": ["a.weight"], "z": ["z.weight"]}
