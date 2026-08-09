from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.row4 import full_model, row4_oracle


def _reference_config() -> dict[str, object]:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "auto_map": {
            "AutoModel": "modeling_qwen3_row4_int8.Qwen3ForCausalLM",
            "AutoModelForCausalLM": "modeling_qwen3_row4_int8.Qwen3ForCausalLM",
        },
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "layer_types": ["full_attention"] * 36,
        "max_position_embeddings": 131072,
        "max_window_layers": 36,
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "pad_token_id": 151645,
        "rms_norm_eps": 1.0e-6,
        "rope_parameters": {
            "attn_factor": row4_oracle.REFERENCE_YARN_RAW_ATTN_FACTOR,
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "rope_theta": 1_000_000,
            "rope_type": "yarn",
        },
        "sliding_window": None,
        "tie_word_embeddings": False,
        "transformers_version": "5.2.0",
        "use_cache": False,
        "use_sliding_window": False,
        "vocab_size": 151936,
    }


class _Shape:
    def __init__(self, values: tuple[int, ...]):
        self.values = values

    def tolist(self) -> list[int]:
        return list(self.values)


def _fake_gguf_tensors() -> dict[str, SimpleNamespace]:
    return {
        name: SimpleNamespace(
            tensor_type=tensor_type,
            shape=_Shape(shape),
            n_bytes=nbytes,
        )
        for name, (tensor_type, shape, nbytes) in (
            row4_oracle.expected_gguf_tensor_inventory().items()
        )
    }


def _fake_checkpoint() -> SimpleNamespace:
    shapes = row4_oracle.expected_checkpoint_tensor_shapes()
    names = sorted(shapes)
    headers: dict[str, tuple[str, str, tuple[int, ...]]] = {}
    weight_map: dict[str, str] = {}
    offset = 0
    for shard, count in zip(
        row4_oracle.REFERENCE_SHARDS,
        row4_oracle.REFERENCE_SHARD_TENSOR_COUNTS,
    ):
        for name in names[offset : offset + count]:
            headers[name] = (shard, "BF16", shapes[name])
            weight_map[name] = shard
        offset += count
    assert offset == len(names)
    return SimpleNamespace(
        shards=row4_oracle.REFERENCE_SHARDS,
        shard_tensor_counts=row4_oracle.REFERENCE_SHARD_TENSOR_COUNTS,
        tensor_headers=headers,
        weight_map=weight_map,
        metadata={
            "total_parameters": row4_oracle.REFERENCE_CHECKPOINT_PARAMETERS,
            "total_size": row4_oracle.REFERENCE_CHECKPOINT_BYTES,
        },
    )


def test_strict_gguf_inventory_has_exact_count_and_payload() -> None:
    inventory = row4_oracle.expected_gguf_tensor_inventory()
    assert len(inventory) == row4_oracle.REFERENCE_GGUF_TENSOR_COUNT == 436
    assert sum(entry[2] for entry in inventory.values()) == 2_739_236_352
    row4_oracle.validate_gguf_tensor_inventory(_fake_gguf_tensors(), len(inventory))


def test_strict_gguf_inventory_rejects_before_sample_checks() -> None:
    tensors = _fake_gguf_tensors()
    tensors["output.w8.scales"].n_bytes -= 4
    with pytest.raises(ValueError, match="payload bytes"):
        row4_oracle.validate_gguf_tensor_inventory(tensors, len(tensors))


def test_strict_metadata_pins_yarn_pre_factor_one() -> None:
    metadata = row4_oracle.GGUFVerifier.EXPECTED_METADATA
    assert metadata["general.quantization_version"] == 2
    assert metadata["tokenizer.ggml.add_bos_token"] is True
    assert metadata["qwen3.rope.scaling.attn_factor"] == 1.0
    assert row4_oracle.REFERENCE_YARN_EFFECTIVE_MSCALE == pytest.approx(
        1.0 + 0.1 * math.log(4.0)
    )
    assert metadata["qwen3.rope.scaling.attn_factor"] != pytest.approx(
        row4_oracle.REFERENCE_YARN_RAW_ATTN_FACTOR
    )


def test_reference_checkpoint_schema_and_config_are_exact() -> None:
    assert row4_oracle.reference_config_issues(_reference_config()) == []
    assert row4_oracle.reference_checkpoint_issues(_fake_checkpoint()) == []

    drifted = _reference_config()
    drifted["transformers_version"] = "5.2.1"
    assert any(
        "transformers_version" in issue
        for issue in row4_oracle.reference_config_issues(drifted)
    )


@pytest.mark.parametrize(
    "extra_key",
    [
        "attention_factor",
        "beta_fast",
        "beta_slow",
        "truncate",
        "partial_rotary_factor",
        "mscale",
        "mscale_all_dim",
    ],
)
def test_oracle_rejects_extra_semantic_rope_keys(extra_key: str) -> None:
    config = _reference_config()
    rope = config["rope_parameters"]
    assert isinstance(rope, dict)
    rope[extra_key] = 1.0
    issues = row4_oracle.reference_config_issues(config)
    assert any(
        "rope_parameters key set differs" in issue and extra_key in issue
        for issue in issues
    )


def test_full_model_skip_hashes_and_installed_version_are_nonreference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "args.json").write_text(
        json.dumps(
            {
                "attn_impl": "flash_attention_3",
                "bf16": True,
                "fp16": False,
                "torch_dtype": "bfloat16",
                "use_cache": False,
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        checkpoint=tmp_path,
        attn_implementation="flash_attention_3",
        device="cpu",
        skip_shard_hashes=True,
    )
    monkeypatch.setattr(
        full_model,
        "package_version",
        lambda name: "5.2.1" if name == "transformers" else "1.0",
    )
    issues = full_model.reference_issues(args, _reference_config(), _fake_checkpoint())
    assert any("installed transformers version must be 5.2.0" in issue for issue in issues)
    assert any("SHA-256 hashes" in issue for issue in issues)
    reference_capture = not issues
    assert reference_capture is False


def _temporary_oracle_dir(root: Path, name: str) -> Path:
    temporary = root / f".{name}.tmp"
    temporary.mkdir()
    (temporary / "payload.txt").write_text(name, encoding="utf-8")
    (temporary / "manifest.json").write_text(
        json.dumps({"winner": name}),
        encoding="utf-8",
    )
    return temporary


def test_directory_publish_concurrently_reserves_without_overwrite(tmp_path: Path) -> None:
    final_dir = tmp_path / "golden"
    temporaries = [
        _temporary_oracle_dir(tmp_path, "first"),
        _temporary_oracle_dir(tmp_path, "second"),
    ]
    barrier = threading.Barrier(2)
    results: list[BaseException | None] = []

    def publish(temporary: Path) -> None:
        barrier.wait()
        try:
            row4_oracle._publish_directory_no_clobber(temporary, final_dir)
        except BaseException as exc:  # exercise the exact collision type below
            results.append(exc)
        else:
            results.append(None)

    threads = [threading.Thread(target=publish, args=(temporary,)) for temporary in temporaries]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(result is None for result in results) == 1
    failures = [result for result in results if result is not None]
    assert len(failures) == 1 and isinstance(failures[0], FileExistsError)
    manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    assert (final_dir / "payload.txt").read_text(encoding="utf-8") == manifest["winner"]


def test_directory_publish_does_not_replace_broken_symlink(tmp_path: Path) -> None:
    final_dir = tmp_path / "golden"
    final_dir.symlink_to(tmp_path / "missing")
    temporary = _temporary_oracle_dir(tmp_path, "candidate")
    with pytest.raises(FileExistsError):
        row4_oracle._publish_directory_no_clobber(temporary, final_dir)
    assert final_dir.is_symlink()
