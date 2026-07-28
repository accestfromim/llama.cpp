from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

import convert_fairy2i_qwen2 as qwen2_converter
import gguf
from convert_fairy2i_qwen2 import (
    FAIRY2I_DEEPSEEK_ASSISTANT_TOKEN,
    QKV_BIAS_SPECS,
    build_qwen2_tensor_write_plan,
    expected_qwen2_tensor_shapes,
    get_qwen2_dimensions,
    load_safetensors_manifest,
    main as qwen2_main,
    print_qwen2_preflight,
    run_qwen2_preflight,
    TensorReader,
)
from fairy2i.convert import _dispatch_args
from fairy2i.spec import QUANT_VARIANT_TILE64_V2, WEIGHT_LAYOUT_BUNDLE_V1


def _tiny_config(*, hidden_size: int = 128, vocab_size: int = 128) -> dict[str, object]:
    return {
        "_name_or_path": "tiny-fairy2i-qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "bos_token_id": 127,
        "eos_token_id": 124,
        "pad_token_id": 124,
        "hidden_size": hidden_size,
        "intermediate_size": 128,
        "max_position_embeddings": 4096,
        "model_type": "qwen2",
        "num_attention_heads": 1,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 1_000_000.0},
        "tie_word_embeddings": False,
        "vocab_size": vocab_size,
    }


def _write_tokenizer(model_dir: Path) -> None:
    base_vocab = {f"token-{token_id}": token_id for token_id in range(124)}
    added_tokens = [
        {"id": 124, "content": "<｜end▁of▁sentence｜>", "special": True},
        {"id": 125, "content": "<｜User｜>", "special": True},
        {"id": 126, "content": "<｜Assistant｜>", "special": True},
        {"id": 127, "content": "<｜begin▁of▁sentence｜>", "special": True},
    ]
    tokenizer_json = {
        "model": {"type": "BPE", "vocab": base_vocab, "merges": []},
        "added_tokens": added_tokens,
    }
    (model_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json), encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<｜begin▁of▁sentence｜>",
                "eos_token": "<｜end▁of▁sentence｜>",
                "pad_token": "<｜end▁of▁sentence｜>",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "chat_template.jinja").write_text(
        "{{ '<｜begin▁of▁sentence｜>' }}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}{{ '<｜User｜>' + message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<｜Assistant｜>' + message['content'] + '<｜end▁of▁sentence｜>' }}"
        "{% endif %}{% endfor %}",
        encoding="utf-8",
    )


def _write_checkpoint(
    model_dir: Path,
    *,
    hidden_size: int = 128,
    vocab_size: int = 128,
    missing_tensor: str | None = None,
    dtype_override: tuple[str, torch.dtype] | None = None,
    shape_override: tuple[str, tuple[int, ...]] | None = None,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    model_dir.mkdir()
    config = _tiny_config(hidden_size=hidden_size, vocab_size=vocab_size)
    dimensions = get_qwen2_dimensions(config)
    tensors = {
        name: torch.zeros(shape, dtype=torch.bfloat16)
        for name, shape in expected_qwen2_tensor_shapes(dimensions).items()
    }
    if missing_tensor is not None:
        tensors.pop(missing_tensor)
    if dtype_override is not None:
        name, dtype = dtype_override
        tensors[name] = tensors[name].to(dtype)
    if shape_override is not None:
        name, shape = shape_override
        tensors[name] = torch.zeros(shape, dtype=torch.bfloat16)

    names = sorted(tensors)
    shard_names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    shard_tensors = (
        {name: tensors[name] for name in names[::2]},
        {name: tensors[name] for name in names[1::2]},
    )
    weight_map: dict[str, str] = {}
    for shard_name, shard in zip(shard_names, shard_tensors):
        safetensors_torch.save_file(shard, model_dir / shard_name, metadata={"format": "pt"})
        weight_map.update({name: shard_name for name in shard})

    total_size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    # Deliberately wrong: the converter must compute this from headers.
                    "total_parameters": 42,
                    "total_size": total_size,
                },
                "weight_map": weight_map,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    _write_tokenizer(model_dir)
    return config, tensors


def _write_raw_safetensors(path: Path, header: dict[str, object], payload: bytes = b"") -> None:
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)


def test_qwen2_dry_run_validates_headers_and_ignores_index_parameter_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir = tmp_path / "model"
    _, tensors = _write_checkpoint(model_dir)

    report = run_qwen2_preflight(
        model_dir,
        output_file=None,
        output_layer="both",
        weight_layout=WEIGHT_LAYOUT_BUNDLE_V1,
    )

    assert len(report.manifest.shards) == 2
    assert len(report.manifest.tensors) == 15
    assert report.qat_linear_count == 7
    assert report.qkv_bias_count == 3
    assert report.manifest.parameter_count == sum(tensor.numel() for tensor in tensors.values())
    assert report.manifest.tensor_bytes == sum(
        tensor.numel() * tensor.element_size() for tensor in tensors.values()
    )
    assert report.manifest.index_metadata["total_parameters"] == 42

    print_qwen2_preflight(report)
    output = capsys.readouterr().out
    assert "shards=2 tensors=15 BF16=15 qat_linears=7 qkv_biases=3" in output
    assert "index metadata total_parameters=42 ignored" in output

    qwen2_main([str(model_dir), "--dry-run", "--output-layer", "both"])
    assert not (model_dir / "model.gguf").exists()


@pytest.mark.parametrize(
    ("header", "match"),
    [
        (
            {
                "__metadata__": {"format": 1},
                "weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
            },
            "invalid __metadata__ entry",
        ),
        (
            {"weight": {"dtype": "BF16", "shape": [True], "data_offsets": [0, 2]}},
            "invalid safetensors shape",
        ),
        (
            {"weight": {"dtype": "BF16", "shape": [1], "data_offsets": [False, 2]}},
            "invalid safetensors data_offsets",
        ),
    ],
)
def test_safetensors_header_rejects_json_types_not_accepted_by_runtime(
    tmp_path: Path,
    header: dict[str, object],
    match: str,
) -> None:
    shard = tmp_path / "malformed.safetensors"
    _write_raw_safetensors(shard, header, b"\0\0")

    with pytest.raises(ValueError, match=match):
        qwen2_converter._read_safetensors_header(shard)


@pytest.mark.parametrize(("n_head", "n_head_kv"), [(4, 8), (6, 4)])
def test_qwen2_dimensions_reject_invalid_gqa_head_ratio(n_head: int, n_head_kv: int) -> None:
    config = _tiny_config(hidden_size=128)
    config["num_attention_heads"] = n_head
    config["num_key_value_heads"] = n_head_kv
    config["head_dim"] = 128 // n_head

    with pytest.raises(ValueError, match="must be divisible by num_key_value_heads"):
        get_qwen2_dimensions(config)


def test_qwen2_preflight_requires_all_bf16(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_checkpoint(
        model_dir,
        dtype_override=("model.layers.0.self_attn.q_proj.bias", torch.float16),
    )

    with pytest.raises(ValueError, match="all checkpoint tensors must be BF16"):
        run_qwen2_preflight(
            model_dir,
            output_file=None,
            output_layer="both",
            weight_layout=WEIGHT_LAYOUT_BUNDLE_V1,
        )


def test_qwen2_preflight_requires_every_qkv_bias(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    missing_bias = f"model.layers.0.{QKV_BIAS_SPECS[0][0]}"
    _write_checkpoint(model_dir, missing_tensor=missing_bias)

    with pytest.raises(ValueError, match="missing required tensors.*q_proj.bias"):
        run_qwen2_preflight(
            model_dir,
            output_file=None,
            output_layer="both",
            weight_layout=WEIGHT_LAYOUT_BUNDLE_V1,
        )


def test_qwen2_preflight_rejects_partial_m64_tiles(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir, hidden_size=192)

    with pytest.raises(ValueError, match="complete M64xK64 tiles"):
        run_qwen2_preflight(
            model_dir,
            output_file=None,
            output_layer="both",
            weight_layout=WEIGHT_LAYOUT_BUNDLE_V1,
        )


def test_qwen2_preflight_rejects_header_shape_mismatch(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_checkpoint(
        model_dir,
        shape_override=("model.layers.0.self_attn.k_proj.weight", (126, 128)),
    )

    with pytest.raises(ValueError, match="tensor shape mismatch.*k_proj.weight"):
        run_qwen2_preflight(
            model_dir,
            output_file=None,
            output_layer="both",
            weight_layout=WEIGHT_LAYOUT_BUNDLE_V1,
        )


def test_qwen2_preflight_requires_both_indexed_shards(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)
    (model_dir / "model-00002-of-00002.safetensors").unlink()

    with pytest.raises(ValueError, match="shard set does not match.*missing"):
        run_qwen2_preflight(
            model_dir,
            output_file=None,
            output_layer="both",
            weight_layout=WEIGHT_LAYOUT_BUNDLE_V1,
        )


def test_qwen2_chat_template_adds_assistant_generation_prompt(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)

    report = run_qwen2_preflight(
        model_dir,
        output_file=None,
        output_layer="both",
        weight_layout=WEIGHT_LAYOUT_BUNDLE_V1,
    )
    assert isinstance(report.tokenizer.chat_template, str)
    generation_tail = report.tokenizer.chat_template[
        report.tokenizer.chat_template.rfind("add_generation_prompt") :
    ]
    assert FAIRY2I_DEEPSEEK_ASSISTANT_TOKEN in generation_tail
    assert "<｜end▁of▁sentence｜>" not in generation_tail


def test_qwen2_bundle_plan_and_streaming_writer_use_the_same_order(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)
    output_file = tmp_path / "tiny.gguf"

    report = run_qwen2_preflight(
        model_dir,
        output_file=output_file,
        output_layer="both",
        weight_layout=WEIGHT_LAYOUT_BUNDLE_V1,
    )
    reader = TensorReader(model_dir, report.manifest.weight_map, report.manifest.tensors)
    plan = build_qwen2_tensor_write_plan(
        reader,
        report.dimensions,
        output_layer="both",
        weight_layout=WEIGHT_LAYOUT_BUNDLE_V1,
    )
    assert len(plan) == 24
    assert [entry.name for entry in plan[:5]] == [
        "token_embd",
        "output_norm",
        "output.bundle.codes",
        "output.bundle.scales",
        "output",
    ]
    assert plan[2].shape == (1, 64, 4, 16)
    assert plan[3].shape == (1, 4, 2)
    assert not any(".U.s" in entry.name or ".W.s" in entry.name for entry in plan)

    qwen2_main(
        [
            str(model_dir),
            str(output_file),
            "--output-layer",
            "both",
            "--weight-layout",
            WEIGHT_LAYOUT_BUNDLE_V1,
        ]
    )
    gguf_reader = gguf.GGUFReader(output_file)
    tensor_names = [tensor.name for tensor in gguf_reader.tensors]
    assert tensor_names == [entry.name for entry in plan]
    assert gguf_reader.alignment == 64
    assert all(tensor.data_offset % 64 == 0 for tensor in gguf_reader.tensors)
    tensor_map = {tensor.name: tensor for tensor in gguf_reader.tensors}
    assert tensor_map["output.bundle.codes"].tensor_type == gguf.GGMLQuantizationType.FAIRY2I_BUNDLE_CODES
    assert tensor_map["output.bundle.scales"].tensor_type == gguf.GGMLQuantizationType.F16
    stored_template = gguf_reader.fields["tokenizer.chat_template"].contents()
    generation_tail = stored_template[stored_template.rfind("add_generation_prompt") :]
    assert FAIRY2I_DEEPSEEK_ASSISTANT_TOKEN in generation_tail
    assert "<｜end▁of▁sentence｜>" not in generation_tail
    assert output_file.stat().st_size >= sum(entry.nbytes for entry in plan)
    assert not output_file.with_name(f".{output_file.name}.tmp").exists()


def test_qwen2_streaming_chunks_are_bounded_by_rows_and_m64(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir, vocab_size=2048)
    report = run_qwen2_preflight(
        model_dir,
        output_file=None,
        output_layer="both",
        weight_layout=WEIGHT_LAYOUT_BUNDLE_V1,
    )
    reader = TensorReader(model_dir, report.manifest.weight_map, report.manifest.tensors)
    plan = build_qwen2_tensor_write_plan(
        reader,
        report.dimensions,
        output_layer="both",
        weight_layout=WEIGHT_LAYOUT_BUNDLE_V1,
    )

    token_chunks = list(plan[0].chunks())
    output_code_chunks = list(plan[2].chunks())
    output_scale_chunks = list(plan[3].chunks())
    dense_output_chunks = list(plan[4].chunks())

    assert [chunk.shape for chunk in token_chunks] == [(1024, 64), (1024, 64)]
    assert len(output_code_chunks) == 16
    assert all(chunk.shape == (1, 64, 4, 16) for chunk in output_code_chunks)
    assert len(output_scale_chunks) == 16
    assert all(chunk.shape == (1, 4, 2) for chunk in output_scale_chunks)
    assert [chunk.shape for chunk in dense_output_chunks] == [(1024, 128), (1024, 128)]


def test_qwen2_32b_bundle_both_plan_has_1221_tensors() -> None:
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 5120,
        "intermediate_size": 27648,
        "model_type": "qwen2",
        "num_attention_heads": 40,
        "num_hidden_layers": 64,
        "num_key_value_heads": 8,
        "vocab_size": 152064,
    }
    dimensions = get_qwen2_dimensions(config)
    shapes = expected_qwen2_tensor_shapes(dimensions)

    class ShapeOnlyReader:
        @staticmethod
        def shape(key: str) -> tuple[int, ...]:
            return shapes[key]

    plan = build_qwen2_tensor_write_plan(
        ShapeOnlyReader(),  # type: ignore[arg-type]
        dimensions,
        output_layer="both",
        weight_layout=WEIGHT_LAYOUT_BUNDLE_V1,
    )

    assert len(plan) == 1221
    assert sum(entry.name.endswith(".bundle.codes") for entry in plan) == 449
    assert sum(entry.name.endswith(".bundle.scales") for entry in plan) == 449
    assert sum(entry.name == "output" for entry in plan) == 1
    assert all(".U.s" not in entry.name and ".W.s" not in entry.name for entry in plan)


def test_unified_qwen2_dispatch_forwards_dry_run_and_output_layer() -> None:
    args = argparse.Namespace(
        model_dir=Path("/checkpoint"),
        output_file=None,
        base_arch="qwen2",
        quant_variant=QUANT_VARIANT_TILE64_V2,
        residual_steps=2,
        weight_layout=WEIGHT_LAYOUT_BUNDLE_V1,
        dry_run=True,
        qk_permute=False,
        no_attn_bias=False,
        output_layer="both",
        verbose=True,
    )

    dispatched = _dispatch_args(args, "qwen2")

    assert dispatched[0] == "/checkpoint"
    assert "--dry-run" in dispatched
    assert dispatched[dispatched.index("--output-layer") + 1] == "both"
    assert dispatched[dispatched.index("--weight-layout") + 1] == WEIGHT_LAYOUT_BUNDLE_V1


def test_qwen2_rejects_legacy_row_block_layout(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        qwen2_main([str(tmp_path), "--dry-run", "--weight-layout", "tile64_v2"])


@pytest.mark.parametrize("forbidden_flag", ["--qk-permute", "--no-attn-bias"])
def test_qwen2_rejects_incompatible_checkpoint_flags(
    tmp_path: Path,
    forbidden_flag: str,
) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        qwen2_main([str(tmp_path), "--dry-run", forbidden_flag])


def test_qwen2_refuses_to_overwrite_a_checkpoint_shard(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)
    shard = model_dir / "model-00001-of-00002.safetensors"
    original = shard.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite existing output"):
        qwen2_main([str(model_dir), str(shard)])

    assert shard.read_bytes() == original


def test_qwen2_atomic_output_cleans_up_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)
    output_file = tmp_path / "failed.gguf"
    temporary_output = output_file.with_name(f".{output_file.name}.tmp")

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected tensor write failure")

    monkeypatch.setattr(qwen2_converter, "write_qwen2_tensor_write_plan", fail_write)
    with pytest.raises(RuntimeError, match="injected tensor write failure"):
        qwen2_main([str(model_dir), str(output_file), "--output-layer", "both"])

    assert not output_file.exists()
    assert not temporary_output.exists()


def test_qwen2_preflight_requires_existing_output_parent(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)

    with pytest.raises(FileNotFoundError, match="output parent directory does not exist"):
        run_qwen2_preflight(
            model_dir,
            output_file=tmp_path / "missing" / "model.gguf",
            output_layer="both",
            weight_layout=WEIGHT_LAYOUT_BUNDLE_V1,
        )


def test_qwen2_preflight_validates_runtime_config_fields(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    config, _ = _write_checkpoint(model_dir)
    config.pop("rms_norm_eps")
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="rms_norm_eps"):
        run_qwen2_preflight(
            model_dir,
            output_file=None,
            output_layer="both",
            weight_layout=WEIGHT_LAYOUT_BUNDLE_V1,
        )


def test_qwen2_preflight_rejects_empty_chat_template_choices(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)
    (model_dir / "chat_template.jinja").unlink()
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    tokenizer_config["chat_template"] = []
    (model_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")

    with pytest.raises(ValueError, match="chat template choices must not be empty"):
        run_qwen2_preflight(
            model_dir,
            output_file=None,
            output_layer="both",
            weight_layout=WEIGHT_LAYOUT_BUNDLE_V1,
        )


def test_tiny_manifest_parameter_count_is_computed_from_shapes(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _, tensors = _write_checkpoint(model_dir)

    manifest = load_safetensors_manifest(model_dir)

    assert manifest.parameter_count == sum(math.prod(tensor.shape) for tensor in tensors.values())
