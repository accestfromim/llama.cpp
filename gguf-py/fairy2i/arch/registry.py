from __future__ import annotations

from . import llama, qwen2, qwen3
from .base import Fairy2IArchInfo


ADAPTERS = (llama, qwen2, qwen3)


def get_arch_info(name: str) -> Fairy2IArchInfo:
    for adapter in ADAPTERS:
        if adapter.INFO.name == name:
            return adapter.INFO
    raise ValueError(f"unsupported Fairy2i base architecture: {name}")


def detect_arch_info(config: dict) -> Fairy2IArchInfo:
    matches = [adapter.INFO for adapter in ADAPTERS if adapter.match(config)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"could not detect Fairy2i base architecture from model_type={config.get('model_type')!r}")
    names = ", ".join(info.name for info in matches)
    raise ValueError(f"ambiguous Fairy2i base architecture: {names}")
