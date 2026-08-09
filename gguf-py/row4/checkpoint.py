from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class TensorInfo:
    shard: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def parameter_count(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class CheckpointManifest:
    weight_map: dict[str, str]
    tensors: dict[str, TensorInfo]
    shards: tuple[str, ...]
    metadata: dict[str, object]


def load_manifest(model_dir: Path, *, expected_shards: int = 4) -> CheckpointManifest:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"required sharded checkpoint index not found: {index_path}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid checkpoint index JSON in {index_path}: {exc}") from exc
    if not isinstance(index, dict):
        raise ValueError(f"invalid checkpoint index in {index_path}: expected an object")

    raw_weight_map = index.get("weight_map")
    if not isinstance(raw_weight_map, dict) or not all(
        isinstance(name, str) and isinstance(shard, str)
        for name, shard in raw_weight_map.items()
    ):
        raise ValueError(f"invalid weight_map in {index_path}")
    weight_map = dict(raw_weight_map)
    shards = tuple(sorted(set(weight_map.values())))
    if len(shards) != expected_shards:
        raise ValueError(
            f"expected exactly {expected_shards} safetensors shards, got {len(shards)}: {list(shards)}"
        )

    disk_shards = {path.name for path in model_dir.glob("model*.safetensors") if path.is_file()}
    if disk_shards != set(shards):
        raise ValueError(
            "checkpoint shard set does not match index: "
            f"missing={sorted(set(shards) - disk_shards)}, "
            f"unexpected={sorted(disk_shards - set(shards))}"
        )

    tensors: dict[str, TensorInfo] = {}
    for shard in shards:
        shard_path = model_dir / shard
        with safe_open(str(shard_path), framework="pt", device="cpu") as file:
            for name in file.keys():
                if name in tensors:
                    raise ValueError(f"duplicate tensor across checkpoint shards: {name}")
                tensor_slice = file.get_slice(name)
                tensors[name] = TensorInfo(
                    shard=shard,
                    dtype=str(tensor_slice.get_dtype()),
                    shape=tuple(int(dim) for dim in tensor_slice.get_shape()),
                )

    indexed = set(weight_map)
    discovered = set(tensors)
    wrong_shards = sorted(
        name for name in indexed & discovered if weight_map[name] != tensors[name].shard
    )
    if indexed != discovered or wrong_shards:
        raise ValueError(
            "checkpoint index/header mismatch: "
            f"missing={sorted(indexed - discovered)[:8]}, "
            f"unexpected={sorted(discovered - indexed)[:8]}, "
            f"wrong_shard={wrong_shards[:8]}"
        )

    metadata = index.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid checkpoint metadata in {index_path}")
    total_size = sum(info.parameter_count * 2 for info in tensors.values())
    declared_size = metadata.get("total_size")
    if declared_size is not None and declared_size != total_size:
        raise ValueError(
            f"checkpoint total_size mismatch: index={declared_size!r}, headers={total_size}"
        )
    return CheckpointManifest(weight_map, tensors, shards, dict(metadata))


class TensorReader:
    def __init__(self, model_dir: Path, manifest: CheckpointManifest):
        self.model_dir = model_dir
        self.manifest = manifest

    def shape(self, name: str) -> tuple[int, ...]:
        return self.manifest.tensors[name].shape

    def get(self, name: str) -> torch.Tensor:
        info = self.manifest.tensors[name]
        with safe_open(
            str(self.model_dir / info.shard),
            framework="pt",
            device="cpu",
        ) as file:
            return file.get_tensor(name)

    @contextmanager
    def open_2d(self, name: str) -> Iterator[Callable[[int, int], torch.Tensor]]:
        info = self.manifest.tensors[name]
        if len(info.shape) != 2:
            raise ValueError(f"row streaming requires a 2D tensor, got {name}={info.shape}")
        with safe_open(
            str(self.model_dir / info.shard),
            framework="pt",
            device="cpu",
        ) as file:
            tensor_slice = file.get_slice(name)

            def read_rows(start: int, end: int) -> torch.Tensor:
                return tensor_slice[start:end, :]

            yield read_rows
