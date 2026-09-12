#!/usr/bin/env python3
"""Inventory checkpoint structure without loading tensor payloads.

Accepts a local Hugging Face safetensors directory or one local GGUF file and
writes a deterministic JSON report to stdout.  This is deliberately an inventory
probe: it never calls ShardReader.get_tensor() and never materializes weights.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

from freetoken.models.gguf.reader import load_gguf_metadata
from freetoken.models.loader import ShardReader, iter_weight_files
from freetoken.utils import cached_load_hf_config

_FLAG_WORDS = ("mtp", "draft", "nextn", "speculative")


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _flagged_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if any(word in str(key).lower() for word in _FLAG_WORDS):
                found.append(path)
            found.extend(_flagged_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_flagged_keys(child, f"{prefix}[{index}]"))
    return sorted(set(found), key=str.casefold)


def _prefix_groups(names: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        groups[name.split(".", 1)[0]].append(name)
    return {prefix: sorted(group) for prefix, group in sorted(groups.items())}


def _safetensor_header(path: str) -> dict[str, Any]:
    with open(path, "rb") as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"{path}: truncated safetensors header size")
        size = struct.unpack("<Q", raw_size)[0]
        header = stream.read(size)
        if len(header) != size:
            raise ValueError(f"{path}: truncated safetensors header")
    return json.loads(header)


def _hf_report(path: str) -> dict[str, Any]:
    config = _json_value(cached_load_hf_config(path).to_dict())
    # These helpers resolve local directories without downloading.  ShardReader is
    # used for the authoritative name-to-shard map, but no tensor is requested.
    files = sorted(iter_weight_files(path))
    reader = ShardReader(path, device=__import__("torch").device("meta"))
    try:
        shards = []
        for file in files:
            header = _safetensor_header(file)
            names = sorted(name for name in reader.names_in(file))
            tensors = []
            for name in names:
                info = header[name]
                tensors.append({"name": name, "dtype": info["dtype"], "shape": info["shape"]})
            shards.append({"file": os.path.abspath(file), "tensors": tensors})
    finally:
        reader.close()
    names = [tensor["name"] for shard in shards for tensor in shard["tensors"]]
    return {
        "format": "hf-safetensors",
        "path": os.path.abspath(path),
        "config": config,
        "flagged_config_keys": _flagged_keys(config),
        "shards": shards,
        "prefix_groups": _prefix_groups(names),
    }


def _gguf_report(path: str) -> dict[str, Any]:
    import gguf

    metadata = _json_value(load_gguf_metadata(path))
    tensors = []
    # GGUFReader tensor descriptors contain all inventory fields. Do not access
    # tensor.data: iter_gguf_tensors intentionally exposes packed payload bytes.
    reader = gguf.GGUFReader(path)
    for tensor in reader.tensors:
        ggml_type = int(tensor.tensor_type)
        tensors.append(
            {
                "name": tensor.name,
                "dtype": tensor.tensor_type.name,
                "ggml_type": ggml_type,
                "shape": list(reversed(int(dim) for dim in tensor.shape)),
            }
        )
    names = [tensor["name"] for tensor in tensors]
    return {
        "format": "gguf",
        "path": os.path.abspath(path),
        "metadata": metadata,
        "flagged_metadata_keys": _flagged_keys(metadata),
        "tensors": sorted(tensors, key=lambda item: item["name"]),
        "prefix_groups": _prefix_groups(names),
    }


def inventory(path: str) -> dict[str, Any]:
    if os.path.isdir(path):
        return _hf_report(path)
    if os.path.isfile(path) and path.lower().endswith(".gguf"):
        return _gguf_report(path)
    raise ValueError("path must be a local safetensors directory or a .gguf file")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="local HF safetensors directory or GGUF file")
    args = parser.parse_args()
    print(json.dumps(inventory(args.path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
