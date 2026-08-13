from __future__ import annotations

from collections.abc import Mapping

import gguf


ROW4_LAYOUT_V1 = "v1"
ROW4_LAYOUT_V2 = "v2"
ROW4_LAYOUTS = (ROW4_LAYOUT_V1, ROW4_LAYOUT_V2)

SCHEMA_VERSION = 1
WEIGHT_LAYOUT = "m16k128_split8_v1"
SCHEMA_VERSION_V2 = 2
WEIGHT_LAYOUT_V2 = "m32k256_pair2_split8_v2"
EXECUTION_PROFILE_V2 = "metal_only_v1"
CODEBOOK = "uv_axis_v1"
NUMERIC_PROFILE = "bf16_a8_away_i32_bf16_v1"
QKV_ORDER = "q_k_v"
FFN_ORDER = "gate_up"
LM_HEAD_LAYOUT = "s8_m16k128_rowmajor_v1"
ALIGNMENT_BY_LAYOUT: Mapping[str, int] = {
    ROW4_LAYOUT_V1: 64,
    ROW4_LAYOUT_V2: 128,
}

METADATA_V1: Mapping[str, int | str] = {
    "row4.schema_version": SCHEMA_VERSION,
    "row4.weight_layout": WEIGHT_LAYOUT,
    "row4.codebook": CODEBOOK,
    "row4.numeric_profile": NUMERIC_PROFILE,
    "row4.qkv_order": QKV_ORDER,
    "row4.ffn_order": FFN_ORDER,
    "row4.lm_head_layout": LM_HEAD_LAYOUT,
}

METADATA_V2: Mapping[str, int | str] = {
    **METADATA_V1,
    "row4.schema_version": SCHEMA_VERSION_V2,
    "row4.weight_layout": WEIGHT_LAYOUT_V2,
    "row4.execution_profile": EXECUTION_PROFILE_V2,
}

# Preserve the original public constant as the default v1 contract.
METADATA = METADATA_V1
METADATA_BY_LAYOUT: Mapping[str, Mapping[str, int | str]] = {
    ROW4_LAYOUT_V1: METADATA_V1,
    ROW4_LAYOUT_V2: METADATA_V2,
}


def metadata_for_layout(row4_layout: str) -> Mapping[str, int | str]:
    try:
        return METADATA_BY_LAYOUT[row4_layout]
    except KeyError as exc:
        raise ValueError(
            f"unsupported Row4 layout {row4_layout!r}; expected one of {ROW4_LAYOUTS}"
        ) from exc


def alignment_for_layout(row4_layout: str) -> int:
    try:
        return ALIGNMENT_BY_LAYOUT[row4_layout]
    except KeyError as exc:
        raise ValueError(
            f"unsupported Row4 layout {row4_layout!r}; expected one of {ROW4_LAYOUTS}"
        ) from exc


def write_metadata(
    writer: gguf.GGUFWriter,
    row4_layout: str = ROW4_LAYOUT_V1,
) -> None:
    """Write the complete, strict metadata contract for a Row4 layout."""

    for key, value in metadata_for_layout(row4_layout).items():
        if isinstance(value, int):
            writer.add_uint32(key, value)
        else:
            writer.add_string(key, value)
