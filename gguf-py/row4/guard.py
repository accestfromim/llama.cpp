from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def is_qwen3_row4_checkpoint(dir_model: Path, hparams: Mapping[str, Any]) -> bool:
    """Detect checkpoints that require the dedicated Row4 deployment converter."""

    architectures = hparams.get("architectures")
    is_qwen3 = hparams.get("model_type") == "qwen3" or (
        isinstance(architectures, list)
        and any(isinstance(arch, str) and arch.startswith("Qwen3") for arch in architectures)
    )
    if not is_qwen3:
        return False

    auto_map = hparams.get("auto_map")
    if isinstance(auto_map, dict) and any(
        isinstance(target, str) and "row4" in target.lower()
        for target in auto_map.values()
    ):
        return True

    index_path = dir_model / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        return isinstance(weight_map, dict) and any(
            isinstance(name, str) and name.endswith(".weight_scale")
            for name in weight_map
        )

    single_file = dir_model / "model.safetensors"
    if not single_file.is_file():
        return False
    try:
        with single_file.open("rb") as file:
            header_size_bytes = file.read(8)
            if len(header_size_bytes) != 8:
                return False
            header_size = int.from_bytes(header_size_bytes, "little")
            if header_size <= 0 or header_size > 100_000_000:
                return False
            header = json.loads(file.read(header_size))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(header, dict) and any(
        isinstance(name, str) and name.endswith(".weight_scale")
        for name in header
    )
