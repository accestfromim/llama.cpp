from __future__ import annotations

from collections.abc import Mapping

import gguf


SCHEMA_VERSION = 1
WEIGHT_LAYOUT = "m16k128_split8_v1"
CODEBOOK = "uv_axis_v1"
NUMERIC_PROFILE = "bf16_a8_away_i32_bf16_v1"
QKV_ORDER = "q_k_v"
FFN_ORDER = "gate_up"
LM_HEAD_LAYOUT = "s8_m16k128_rowmajor_v1"

METADATA: Mapping[str, int | str] = {
    "row4.schema_version": SCHEMA_VERSION,
    "row4.weight_layout": WEIGHT_LAYOUT,
    "row4.codebook": CODEBOOK,
    "row4.numeric_profile": NUMERIC_PROFILE,
    "row4.qkv_order": QKV_ORDER,
    "row4.ffn_order": FFN_ORDER,
    "row4.lm_head_layout": LM_HEAD_LAYOUT,
}


def write_metadata(writer: gguf.GGUFWriter) -> None:
    """Write the complete, strict Row4 v1 metadata contract."""

    for key, value in METADATA.items():
        if isinstance(value, int):
            writer.add_uint32(key, value)
        else:
            writer.add_string(key, value)
