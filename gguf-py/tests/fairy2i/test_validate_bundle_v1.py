from __future__ import annotations

import numbers
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gguf
import validate_fairy2i_bundle_v1 as validator
from validate_fairy2i_bundle_v1 import validate_common_tensors


BRANCH_SUFFIXES = (".U.s0", ".U.s1", ".W.s0", ".W.s1")


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.asarray(values, dtype=np.float32)
    return (f32.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def _tensor(value: int) -> SimpleNamespace:
    data = np.asarray([value], dtype=np.float32)
    return SimpleNamespace(
        tensor_type=0,
        shape=np.asarray(data.shape),
        data=data,
        n_bytes=data.nbytes,
    )


def _named_tensor(
    name: str,
    tensor_type: gguf.GGMLQuantizationType,
    shape: tuple[int, ...],
    data: np.ndarray,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        tensor_type=tensor_type,
        shape=np.asarray(shape),
        data=data,
        n_bytes=data.nbytes,
        data_offset=0,
    )


def _schema3_readers(
    *,
    omit_common: bool = False,
    add_common: bool = False,
    omit_fields: set[str] | None = None,
    field_overrides: dict[str, object] | None = None,
    common_type_overrides: dict[str, gguf.GGMLQuantizationType] | None = None,
    common_shape_overrides: dict[str, tuple[int, ...]] | None = None,
    code_shape_overrides: dict[str, tuple[int, ...]] | None = None,
    scale_shape_overrides: dict[str, tuple[int, ...]] | None = None,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    bundle_bases, common_names = validator.qwen2_schema3_tensor_sets(1)
    common_shapes = {
        "token_embd": (64, 128),
        "output_norm": (128,),
        "blk.0.attn_norm": (128,),
        "blk.0.ffn_norm": (128,),
        "blk.0.attn_q.bias": (128,),
        "blk.0.attn_k.bias": (128,),
        "blk.0.attn_v.bias": (128,),
    }
    common_type_overrides = common_type_overrides or {}
    common_shape_overrides = common_shape_overrides or {}
    code_shape_overrides = code_shape_overrides or {}
    scale_shape_overrides = scale_shape_overrides or {}
    v2_tensors = []
    bundle_tensors = []
    for base_index, base in enumerate(sorted(bundle_bases)):
        v2_tensors.extend(
            _named_tensor(
                base + suffix,
                gguf.GGMLQuantizationType.FAIRY2I_TILE64_V2,
                (64, 64),
                np.zeros(64 * 20, dtype=np.uint8),
            )
            for suffix in BRANCH_SUFFIXES
        )
        code_shape = code_shape_overrides.get(base, (16, 4, 64, 1))
        scale_shape = scale_shape_overrides.get(base, (2, 4, 1))
        code_data = (
            np.arange(np.prod(code_shape), dtype=np.uint64) + base_index
        ).astype(np.uint8).reshape(tuple(reversed(code_shape)))
        scale_data = _bf16_bits(
            np.arange(np.prod(scale_shape), dtype=np.float32) + base_index + 1
        ).reshape(tuple(reversed(scale_shape)))
        bundle_tensors.extend(
            [
                _named_tensor(
                    base + ".bundle.codes",
                    gguf.GGMLQuantizationType.FAIRY2I_BUNDLE_CODES,
                    code_shape,
                    code_data,
                ),
                _named_tensor(
                    base + ".bundle.scales",
                    gguf.GGMLQuantizationType.BF16,
                    scale_shape,
                    scale_data,
                ),
            ]
        )

    for common_name in sorted(common_names):
        shape = common_shape_overrides.get(common_name, common_shapes[common_name])
        tensor_type = common_type_overrides.get(
            common_name, gguf.GGMLQuantizationType.F32
        )
        dtype = np.float16 if tensor_type == gguf.GGMLQuantizationType.F16 else np.float32
        common = _named_tensor(
            common_name,
            tensor_type,
            shape,
            np.ones(tuple(reversed(shape)), dtype=dtype),
        )
        v2_tensors.append(common)
        if not (omit_common and common_name == "token_embd"):
            bundle_tensors.append(common)
    if add_common:
        bundle_tensors.append(
            _named_tensor(
                "unexpected",
                gguf.GGMLQuantizationType.F32,
                (1,),
                np.asarray([2.0], dtype=np.float32),
            )
        )

    field_values: dict[str, object] = {
        "general.architecture": "fairy2i",
        "general.file_type": 42,
        "general.alignment": 64,
        "fairy2i.schema_version": 3,
        "fairy2i.block_count": 1,
        "fairy2i.embedding_length": 64,
        "fairy2i.feed_forward_length": 64,
        "fairy2i.attention.head_count": 2,
        "fairy2i.attention.head_count_kv": 2,
        "fairy2i.rope.dimension_count": 64,
        "fairy2i.vocab_size": 128,
        "fairy2i.base_arch": "qwen2",
        "fairy2i.quant.format": "fairy2i_tile64_v2",
        "fairy2i.quant.variant": "tile64_v2",
        "fairy2i.quant.residual_steps": 2,
        "fairy2i.quant.codebook": "{+/-1,+/-i}",
        "fairy2i.quant.tile_size": 64,
        "fairy2i.quant.scale_stat": "dominant_mean_abs",
        "fairy2i.quant.numeric_profile": "script_f32reduce_bf16scale_v1",
        "fairy2i.attn.layout": "qwen2_real",
        "fairy2i.tokenizer.profile": "qwen2",
        "fairy2i.weight.scale_dtype": "bf16",
        "fairy2i.weight.layout": "bundle_m64k64_v1",
        "fairy2i.weight.scale_scope": "m64_k64",
        "fairy2i.weight.code_order": "m16_q4_branch_lane",
        "fairy2i.weight.branch_order": "U0,U1,W0,W1",
        "fairy2i.weight.m_block": 64,
        "fairy2i.weight.k_block": 64,
        "fairy2i.weight.m_subtile": 16,
    }
    field_values.update(field_overrides or {})
    for name in omit_fields or set():
        field_values.pop(name, None)

    def field_type(value: object) -> gguf.GGUFValueType:
        if isinstance(value, (bool, np.bool_)):
            return gguf.GGUFValueType.BOOL
        if isinstance(value, numbers.Integral):
            return gguf.GGUFValueType.UINT32
        if isinstance(value, numbers.Real):
            return gguf.GGUFValueType.FLOAT32
        if isinstance(value, str):
            return gguf.GGUFValueType.STRING
        raise TypeError(f"unsupported fake GGUF field value: {value!r}")

    fields = {
        name: SimpleNamespace(
            types=[field_type(value)],
            contents=lambda value=value: value,
        )
        for name, value in field_values.items()
    }
    return (
        SimpleNamespace(fields={}, tensors=v2_tensors),
        SimpleNamespace(fields=fields, tensors=bundle_tensors),
    )


def _validate_schema3_readers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    v2_reader: SimpleNamespace,
    bundle_reader: SimpleNamespace,
    *,
    with_reference: bool,
) -> dict[str, object]:
    v2_path = tmp_path / "v2.gguf"
    bundle_path = tmp_path / "bundle.gguf"
    v2_path.touch()
    bundle_path.touch()

    def fake_reader(path: object, mode: str) -> SimpleNamespace:
        assert mode == "r"
        return v2_reader if Path(path) == v2_path else bundle_reader

    monkeypatch.setattr(validator.gguf, "GGUFReader", fake_reader)
    return validator.validate_layout(v2_path if with_reference else None, bundle_path)


def _schema4_reader(
    *,
    field_overrides: dict[str, object] | None = None,
    output_type: gguf.GGMLQuantizationType = gguf.GGMLQuantizationType.BF16,
    scale_bits: int = 0xBF80,
) -> SimpleNamespace:
    bundle_bases, required_common, _ = validator.qwen3_schema4_tensor_sets(1)
    linear_dims = {
        "blk.0.attn_qkv": (64, 128),
        "blk.0.attn_output": (64, 64),
        "blk.0.ffn_gate": (64, 64),
        "blk.0.ffn_up": (64, 64),
        "blk.0.ffn_down": (64, 64),
    }
    common_shapes = {
        "token_embd": (64, 128),
        "output_norm": (128,),
        "output": (128, 128),
        "blk.0.attn_norm": (128,),
        "blk.0.ffn_norm": (128,),
        "blk.0.attn_q_norm": (64,),
        "blk.0.attn_k_norm": (64,),
    }
    assert set(common_shapes) == required_common

    tensors = []
    for base in sorted(bundle_bases):
        logical_k, logical_m = linear_dims[base]
        tile_count = logical_k // 64 * (logical_m // 64)
        code_shape = (16, 2, 64, tile_count)
        scale_shape = (2, 2, tile_count)
        tensors.extend(
            [
                _named_tensor(
                    base + ".bundle.codes",
                    gguf.GGMLQuantizationType.FAIRY2I_BUNDLE_CODES,
                    code_shape,
                    np.zeros(tuple(reversed(code_shape)), dtype=np.uint8),
                ),
                _named_tensor(
                    base + ".bundle.scales",
                    gguf.GGMLQuantizationType.BF16,
                    scale_shape,
                    np.full(
                        tuple(reversed(scale_shape)),
                        np.uint16(scale_bits),
                        dtype=np.uint16,
                    ),
                ),
            ]
        )
    for name, shape in common_shapes.items():
        tensor_type = output_type if name == "output" else gguf.GGMLQuantizationType.F32
        dtype = (
            np.uint16
            if tensor_type == gguf.GGMLQuantizationType.BF16
            else np.float16
            if tensor_type == gguf.GGMLQuantizationType.F16
            else np.float32
        )
        fill = np.uint16(0x3F80) if dtype == np.uint16 else 1.0
        tensors.append(
            _named_tensor(
                name,
                tensor_type,
                shape,
                np.full(tuple(reversed(shape)), fill, dtype=dtype),
            )
        )

    field_values: dict[str, object] = {
        "general.architecture": "fairy2i",
        "general.file_type": 42,
        "general.alignment": 64,
        "fairy2i.schema_version": 4,
        "fairy2i.block_count": 1,
        "fairy2i.embedding_length": 64,
        "fairy2i.feed_forward_length": 64,
        "fairy2i.attention.head_count": 2,
        "fairy2i.attention.head_count_kv": 1,
        "fairy2i.rope.dimension_count": 64,
        "fairy2i.vocab_size": 128,
        "fairy2i.base_arch": "qwen3",
        "fairy2i.quant.format": "fairy2i_tile64_v2",
        "fairy2i.quant.variant": "tile64_v2_w1_learned_scale",
        "fairy2i.quant.residual_steps": 1,
        "fairy2i.quant.codebook": "{+/-1,+/-i}",
        "fairy2i.quant.tile_size": 64,
        "fairy2i.quant.scale_source": "learned",
        "fairy2i.quant.numeric_profile": "qat_bf16_learned_scale_v1",
        "fairy2i.attn.layout": "qwen3_real",
        "fairy2i.tokenizer.profile": "qwen2",
        "fairy2i.weight.scale_dtype": "bf16",
        "fairy2i.weight.layout": "bundle_m64k64_v1",
        "fairy2i.weight.scale_scope": "m64_k64",
        "fairy2i.weight.code_order": "m16_q4_branch_lane",
        "fairy2i.weight.branch_order": "U0,W0",
        "fairy2i.weight.m_block": 64,
        "fairy2i.weight.k_block": 64,
        "fairy2i.weight.m_subtile": 16,
    }
    field_values.update(field_overrides or {})

    def field_type(value: object) -> gguf.GGUFValueType:
        if isinstance(value, numbers.Integral):
            return gguf.GGUFValueType.UINT32
        if isinstance(value, str):
            return gguf.GGUFValueType.STRING
        raise TypeError(value)

    fields = {
        name: SimpleNamespace(
            types=[field_type(value)],
            contents=lambda value=value: value,
        )
        for name, value in field_values.items()
    }
    return SimpleNamespace(fields=fields, tensors=tensors)


def _validate_schema4_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bundle_reader: SimpleNamespace,
) -> dict[str, object]:
    bundle_path = tmp_path / "schema4.gguf"
    bundle_path.touch()
    monkeypatch.setattr(
        validator.gguf,
        "GGUFReader",
        lambda path, mode: bundle_reader,
    )
    return validator.validate_layout(None, bundle_path)


def test_common_tensor_sets_must_match_exactly() -> None:
    v2_tensors = {
        "token_embd": _tensor(1),
        "blk.0.attn_q.U.s0": _tensor(2),
    }
    bundle_tensors = {
        "token_embd": _tensor(1),
        "blk.0.attn_q.bundle.codes": _tensor(3),
        "blk.0.attn_q.bundle.scales": _tensor(4),
    }

    count, n_bytes = validate_common_tensors(
        v2_tensors,
        bundle_tensors,
        BRANCH_SUFFIXES,
    )

    assert count == 1
    assert n_bytes == v2_tensors["token_embd"].n_bytes


def test_schema4_accepts_signed_bf16_scales_and_bf16_lm_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _validate_schema4_reader(
        tmp_path,
        monkeypatch,
        _schema4_reader(scale_bits=0xBF80),
    )

    assert result["schema_version"] == 4
    assert result["branch_order"] == "U0,W0"
    assert result["linear_count"] == 5
    assert result["comparison_mode"] == "exact_bundle_structural_only"


def test_schema4_rejects_unusable_tile64_reference_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v2_path = tmp_path / "legacy-v2.gguf"
    bundle_path = tmp_path / "schema4.gguf"
    v2_path.touch()
    bundle_path.touch()
    v2_reader = SimpleNamespace(fields={}, tensors=[])
    bundle_reader = _schema4_reader()

    def fake_reader(path: object, mode: str) -> SimpleNamespace:
        assert mode == "r"
        return v2_reader if Path(path) == v2_path else bundle_reader

    monkeypatch.setattr(validator.gguf, "GGUFReader", fake_reader)
    with pytest.raises(
        ValueError,
        match=r"use --exact-structural-only SCHEMA4_BUNDLE\.gguf",
    ):
        validator.validate_layout(v2_path, bundle_path)


@pytest.mark.parametrize(
    ("reader", "message"),
    [
        (
            _schema4_reader(
                field_overrides={
                    "fairy2i.quant.numeric_profile": "script_f32reduce_bf16scale_v1"
                }
            ),
            "fairy2i.quant.numeric_profile",
        ),
        (
            _schema4_reader(output_type=gguf.GGMLQuantizationType.F16),
            "lm_head must use BF16",
        ),
        (
            _schema4_reader(scale_bits=0x7F80),
            "non-finite bundle scales",
        ),
    ],
)
def test_schema4_rejects_wrong_exact_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader: SimpleNamespace,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_schema4_reader(tmp_path, monkeypatch, reader)


def test_rejects_missing_common_tensor() -> None:
    v2_tensors = {
        "token_embd": _tensor(1),
        "output_norm": _tensor(2),
        "blk.0.attn_q.U.s0": _tensor(3),
    }
    bundle_tensors = {
        "token_embd": _tensor(1),
        "blk.0.attn_q.bundle.codes": _tensor(4),
        "blk.0.attn_q.bundle.scales": _tensor(5),
    }

    with pytest.raises(
        ValueError,
        match=r"common tensor sets differ:.*output_norm",
    ):
        validate_common_tensors(v2_tensors, bundle_tensors, BRANCH_SUFFIXES)


def test_rejects_extra_common_tensor() -> None:
    v2_tensors = {
        "token_embd": _tensor(1),
        "blk.0.attn_q.U.s0": _tensor(2),
    }
    bundle_tensors = {
        "token_embd": _tensor(1),
        "unexpected": _tensor(2),
        "blk.0.attn_q.bundle.codes": _tensor(3),
        "blk.0.attn_q.bundle.scales": _tensor(4),
    }

    with pytest.raises(
        ValueError,
        match=r"common tensor sets differ:.*unexpected",
    ):
        validate_common_tensors(v2_tensors, bundle_tensors, BRANCH_SUFFIXES)


@pytest.mark.parametrize(
    ("omit_common", "add_common", "match"),
    [
        (True, False, "schema3 exact common tensor sets differ: missing=.*token_embd"),
        (False, True, "schema3 exact common tensor sets differ:.*extra=.*unexpected"),
    ],
)
def test_schema3_with_reference_rejects_common_tensor_set_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    omit_common: bool,
    add_common: bool,
    match: str,
) -> None:
    v2_reader, bundle_reader = _schema3_readers(
        omit_common=omit_common,
        add_common=add_common,
    )
    with pytest.raises(ValueError, match=match):
        _validate_schema3_readers(
            tmp_path,
            monkeypatch,
            v2_reader,
            bundle_reader,
            with_reference=True,
        )


def test_schema3_canonical_hash_is_reference_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v2_reader, bundle_reader = _schema3_readers()
    with_reference = _validate_schema3_readers(
        tmp_path,
        monkeypatch,
        v2_reader,
        bundle_reader,
        with_reference=True,
    )
    structural_only = _validate_schema3_readers(
        tmp_path,
        monkeypatch,
        v2_reader,
        bundle_reader,
        with_reference=False,
    )

    assert with_reference["canonical_sha256"] == structural_only["canonical_sha256"]


@pytest.mark.parametrize(
    "tensor_type",
    [gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32],
)
def test_schema3_rejects_non_bf16_scale_tensor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tensor_type: gguf.GGMLQuantizationType,
) -> None:
    v2_reader, bundle_reader = _schema3_readers()
    tensor = next(
        item for item in bundle_reader.tensors if item.name.endswith(".bundle.scales")
    )
    tensor.tensor_type = tensor_type

    with pytest.raises(ValueError, match="invalid bundle scale type"):
        _validate_schema3_readers(
            tmp_path,
            monkeypatch,
            v2_reader,
            bundle_reader,
            with_reference=False,
        )


@pytest.mark.parametrize(
    ("scale_bits", "message"),
    [
        (0x7F80, "non-finite bundle scales"),
        (0x7FC1, "non-finite bundle scales"),
        (0x8000, "negative bundle scales"),
        (0xBF80, "negative bundle scales"),
    ],
)
def test_schema3_rejects_invalid_bf16_scale_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scale_bits: int,
    message: str,
) -> None:
    v2_reader, bundle_reader = _schema3_readers()
    tensor = next(
        item for item in bundle_reader.tensors if item.name.endswith(".bundle.scales")
    )
    tensor.data.reshape(-1)[0] = np.uint16(scale_bits)

    with pytest.raises(ValueError, match=message):
        _validate_schema3_readers(
            tmp_path,
            monkeypatch,
            v2_reader,
            bundle_reader,
            with_reference=False,
        )


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    [
        ("fairy2i.quant.format", "fairy2i_tile64_v1"),
        ("fairy2i.quant.variant", "legacy"),
        ("fairy2i.quant.residual_steps", 1),
        ("fairy2i.quant.codebook", "{0,1}"),
        ("fairy2i.quant.tile_size", 32),
        ("fairy2i.quant.scale_stat", "mean_abs"),
        ("fairy2i.quant.numeric_profile", "script_bf16_f32_v1"),
        ("fairy2i.attn.layout", "llama_real"),
        ("fairy2i.tokenizer.profile", "llama_bpe"),
        ("fairy2i.weight.scale_dtype", "f32"),
    ],
)
def test_schema3_rejects_wrong_exact_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    wrong_value: object,
) -> None:
    v2_reader, bundle_reader = _schema3_readers(
        field_overrides={field_name: wrong_value}
    )
    with pytest.raises(ValueError, match=field_name):
        _validate_schema3_readers(
            tmp_path,
            monkeypatch,
            v2_reader,
            bundle_reader,
            with_reference=False,
        )


@pytest.mark.parametrize(
    ("omit", "override"),
    [
        (True, None),
        (False, "llama"),
    ],
)
def test_schema3_requires_fairy2i_general_architecture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    omit: bool,
    override: str | None,
) -> None:
    v2_reader, bundle_reader = _schema3_readers(
        omit_fields={"general.architecture"} if omit else None,
        field_overrides=(
            {"general.architecture": override} if override is not None else None
        ),
    )
    with pytest.raises(ValueError, match="general.architecture"):
        _validate_schema3_readers(
            tmp_path,
            monkeypatch,
            v2_reader,
            bundle_reader,
            with_reference=False,
        )


def test_schema3_rejects_missing_explicit_attention_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v2_reader, bundle_reader = _schema3_readers(
        omit_fields={"fairy2i.attn.layout"}
    )
    with pytest.raises(ValueError, match="fairy2i.attn.layout"):
        _validate_schema3_readers(
            tmp_path,
            monkeypatch,
            v2_reader,
            bundle_reader,
            with_reference=False,
        )


def test_schema3_rejects_missing_explicit_tokenizer_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v2_reader, bundle_reader = _schema3_readers(
        omit_fields={"fairy2i.tokenizer.profile"}
    )
    with pytest.raises(ValueError, match="fairy2i.tokenizer.profile"):
        _validate_schema3_readers(
            tmp_path,
            monkeypatch,
            v2_reader,
            bundle_reader,
            with_reference=False,
        )


@pytest.mark.parametrize(
    "tensor_type",
    [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16],
)
def test_schema3_rejects_attention_output_bias_outside_exact_tensor_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tensor_type: gguf.GGMLQuantizationType,
) -> None:
    v2_reader, bundle_reader = _schema3_readers()
    dtype = np.float16 if tensor_type == gguf.GGMLQuantizationType.F16 else np.float32
    output_bias = _named_tensor(
        "blk.0.attn_output.bias",
        tensor_type,
        (128,),
        np.zeros((128,), dtype=dtype),
    )
    v2_reader.tensors.append(output_bias)
    bundle_reader.tensors.append(output_bias)

    with pytest.raises(
        ValueError,
        match=r"schema3 exact common tensor sets differ:.*attn_output.bias",
    ):
        _validate_schema3_readers(
            tmp_path,
            monkeypatch,
            v2_reader,
            bundle_reader,
            with_reference=False,
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "fairy2i.block_count",
        "fairy2i.embedding_length",
        "fairy2i.feed_forward_length",
        "fairy2i.attention.head_count",
        "fairy2i.attention.head_count_kv",
        "fairy2i.vocab_size",
    ],
)
def test_schema3_rejects_missing_dimension_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_field: str,
) -> None:
    v2_reader, bundle_reader = _schema3_readers(omit_fields={missing_field})
    with pytest.raises(ValueError, match=missing_field):
        _validate_schema3_readers(
            tmp_path,
            monkeypatch,
            v2_reader,
            bundle_reader,
            with_reference=False,
        )


@pytest.mark.parametrize("invalid_value", ["64", 64.0, True])
def test_schema3_rejects_noninteger_dimension_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_value: object,
) -> None:
    v2_reader, bundle_reader = _schema3_readers(
        field_overrides={"fairy2i.embedding_length": invalid_value}
    )
    with pytest.raises(
        ValueError,
        match="fairy2i.embedding_length: expected scalar GGUF UINT32",
    ):
        _validate_schema3_readers(
            tmp_path,
            monkeypatch,
            v2_reader,
            bundle_reader,
            with_reference=False,
        )


@pytest.mark.parametrize(
    ("field_name", "expected_value"),
    [
        ("general.file_type", 42),
        ("general.alignment", 64),
        ("fairy2i.schema_version", 3),
        ("fairy2i.block_count", 1),
        ("fairy2i.embedding_length", 64),
        ("fairy2i.feed_forward_length", 64),
        ("fairy2i.attention.head_count", 2),
        ("fairy2i.attention.head_count_kv", 2),
        ("fairy2i.rope.dimension_count", 64),
        ("fairy2i.vocab_size", 128),
        ("fairy2i.quant.residual_steps", 2),
        ("fairy2i.quant.tile_size", 64),
        ("fairy2i.weight.m_block", 64),
        ("fairy2i.weight.k_block", 64),
        ("fairy2i.weight.m_subtile", 16),
    ],
)
@pytest.mark.parametrize("invalid_kind", ["float", "bool"])
def test_schema3_rejects_non_uint32_integer_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    expected_value: int,
    invalid_kind: str,
) -> None:
    invalid_value: object = (
        float(expected_value) if invalid_kind == "float" else True
    )
    v2_reader, bundle_reader = _schema3_readers(
        field_overrides={field_name: invalid_value}
    )
    with pytest.raises(
        ValueError,
        match=rf"{field_name}: expected scalar GGUF UINT32",
    ):
        _validate_schema3_readers(
            tmp_path,
            monkeypatch,
            v2_reader,
            bundle_reader,
            with_reference=False,
        )


def test_schema3_reference_requires_exact_logical_k_m_not_only_tile_count() -> None:
    with pytest.raises(
        ValueError,
        match=r"blk\.0\.attn_q expected K=128, M=64, got K=64, M=128",
    ):
        validator.validate_schema3_reference_linear_dims(
            "blk.0.attn_q",
            64,
            128,
            {"blk.0.attn_q": (128, 64)},
        )


def test_schema3_rejects_wrong_common_tensor_type_and_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v2_reader, bundle_reader = _schema3_readers(
        common_type_overrides={"blk.0.ffn_norm": gguf.GGMLQuantizationType.F16}
    )
    with pytest.raises(ValueError, match="common tensors must use F32 carriers"):
        _validate_schema3_readers(
            tmp_path,
            monkeypatch,
            v2_reader,
            bundle_reader,
            with_reference=False,
        )

    v2_reader, bundle_reader = _schema3_readers(
        common_shape_overrides={"blk.0.ffn_norm": (64,)}
    )
    with pytest.raises(ValueError, match="invalid schema3 exact common tensor shape"):
        _validate_schema3_readers(
            tmp_path,
            monkeypatch,
            v2_reader,
            bundle_reader,
            with_reference=False,
        )


@pytest.mark.parametrize(
    ("tensor_name", "packed_bits", "value"),
    [
        ("token_embd", 0x3F807F80, None),
        ("token_embd", 0x7FC13F80, None),
        ("output_norm", None, np.nan),
        ("blk.0.attn_norm", None, np.inf),
        ("blk.0.attn_q.bias", None, -np.inf),
    ],
)
def test_schema3_rejects_nonfinite_common_tensor_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tensor_name: str,
    packed_bits: int | None,
    value: float | None,
) -> None:
    v2_reader, bundle_reader = _schema3_readers()
    tensor = next(item for item in bundle_reader.tensors if item.name == tensor_name)
    if packed_bits is not None:
        tensor.data.reshape(-1).view(np.uint32)[0] = np.uint32(packed_bits)
    else:
        assert value is not None
        tensor.data.reshape(-1)[0] = np.float32(value)

    with pytest.raises(
        ValueError,
        match=rf"non-finite schema3 exact common tensor value: {tensor_name}",
    ):
        _validate_schema3_readers(
            tmp_path,
            monkeypatch,
            v2_reader,
            bundle_reader,
            with_reference=False,
        )


def test_schema3_rejects_common_tensor_not_widened_from_bf16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v2_reader, bundle_reader = _schema3_readers()
    tensor = next(
        item for item in bundle_reader.tensors if item.name == "output_norm"
    )
    tensor.data.reshape(-1)[0] = np.float32(1.0 + 2.0**-20)

    with pytest.raises(
        ValueError,
        match=(
            "schema3 exact common tensor is not a BF16-widened F32 value: "
            "output_norm"
        ),
    ):
        _validate_schema3_readers(
            tmp_path,
            monkeypatch,
            v2_reader,
            bundle_reader,
            with_reference=False,
        )


@pytest.mark.parametrize(
    ("shape_kind", "overrides", "match"),
    [
        (
            "codes",
            {"blk.0.ffn_gate": (16, 4, 64, 2)},
            "invalid bundle code shape: blk.0.ffn_gate",
        ),
        (
            "scales",
            {"blk.0.ffn_gate": (2, 4, 2)},
            "invalid bundle scale shape: blk.0.ffn_gate",
        ),
    ],
)
def test_schema3_rejects_physical_tile_count_not_derived_from_model_dims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape_kind: str,
    overrides: dict[str, tuple[int, ...]],
    match: str,
) -> None:
    kwargs = (
        {"code_shape_overrides": overrides}
        if shape_kind == "codes"
        else {"scale_shape_overrides": overrides}
    )
    v2_reader, bundle_reader = _schema3_readers(**kwargs)
    with pytest.raises(ValueError, match=match):
        _validate_schema3_readers(
            tmp_path,
            monkeypatch,
            v2_reader,
            bundle_reader,
            with_reference=False,
        )


def test_schema3_rejects_inconsistent_attention_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v2_reader, bundle_reader = _schema3_readers(
        field_overrides={
            "fairy2i.attention.head_count": 5,
            "fairy2i.attention.head_count_kv": 1,
        }
    )
    with pytest.raises(ValueError, match="attention dimensions are inconsistent"):
        _validate_schema3_readers(
            tmp_path,
            monkeypatch,
            v2_reader,
            bundle_reader,
            with_reference=False,
        )


@pytest.mark.parametrize(
    ("head_count", "head_count_kv"),
    [
        (4, 3),
        (4, 8),
    ],
)
def test_schema3_rejects_inconsistent_gqa_head_ratio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    head_count: int,
    head_count_kv: int,
) -> None:
    v2_reader, bundle_reader = _schema3_readers(
        field_overrides={
            "fairy2i.attention.head_count": head_count,
            "fairy2i.attention.head_count_kv": head_count_kv,
        }
    )
    with pytest.raises(ValueError, match="GQA dimensions are inconsistent"):
        _validate_schema3_readers(
            tmp_path,
            monkeypatch,
            v2_reader,
            bundle_reader,
            with_reference=False,
        )
