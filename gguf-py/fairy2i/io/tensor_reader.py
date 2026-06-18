from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Dict

import gguf
import numpy as np
import torch
from safetensors import safe_open


def load_weight_map(model_dir: Path) -> Dict[str, str]:
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.is_file():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"invalid weight_map in {index_file}")
        return {k: v for k, v in weight_map.items()}

    model_files = sorted(model_dir.glob("*.safetensors"))
    if len(model_files) != 1:
        raise ValueError("no shard index and cannot infer a single safetensors file")

    filename = model_files[0].name
    with safe_open(str(model_files[0]), framework="pt", device="cpu") as f:
        return {key: filename for key in f.keys()}


class TensorReader:
    def __init__(self, model_dir: Path, weight_map: Dict[str, str]):
        self.model_dir = model_dir
        self.weight_map = weight_map

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def get(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"missing tensor key: {key}")
        filename = self.weight_map[key]
        path = self.model_dir / filename
        with safe_open(str(path), framework="pt", device="cpu") as f:
            return f.get_tensor(key)


def add_optional_vector_tensor(
    writer: gguf.GGUFWriter,
    reader: TensorReader,
    hf_key: str,
    gguf_name: str,
) -> None:
    if not reader.has(hf_key):
        return

    tensor = reader.get(hf_key).to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    writer.add_tensor(gguf_name, tensor, raw_dtype=gguf.GGMLQuantizationType.F32)
    del tensor
    gc.collect()
