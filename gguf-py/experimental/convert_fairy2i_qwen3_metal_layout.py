#!/usr/bin/env python3

"""Reproduce offline Fairy2i W1 Metal layout candidates from a Qwen3 checkpoint.

Non-default layouts are experiment artifacts and are intentionally not accepted
by the cleaned production runtime. Use validate_fairy2i_bundle_v1.py to compare
their canonical codes and scales with a tile64_v2 GGUF.
"""

from __future__ import annotations

import sys
from pathlib import Path


GGUF_PY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GGUF_PY_ROOT))

from convert_fairy2i_qwen3 import main  # noqa: E402


if __name__ == "__main__":
    main(experimental_metal_layouts=True)
