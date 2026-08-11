from __future__ import annotations

import ast
import json
import os
import stat
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterator

import numpy as np
import pytest
import gguf

torch = pytest.importorskip("torch")

import convert_row4_qwen3 as converter  # noqa: E402

from convert_row4_qwen3 import (  # noqa: E402
    Qwen3Dimensions,
    REFERENCE_BOS_TOKEN,
    REFERENCE_CHAT_TEMPLATE_BOS_EXPRESSION,
    REFERENCE_NONPOSITIVE_SCALES,
    REFERENCE_PAYLOAD_BYTES,
    REFERENCE_SCALE_VALUES,
    REFERENCE_YARN_EFFECTIVE_MSCALE,
    REFERENCE_YARN_GGUF_PRE_FACTOR,
    REFERENCE_YARN_RAW_ATTN_FACTOR,
    Row4BundleWriter,
    _atomic_publish_no_clobber,
    _cleanup_private_temporary,
    _create_private_temporary,
    _load_tokenizer_files,
    _validate_tokenizer_files,
    add_model_metadata,
    bf16_payload,
    build_tensor_plan,
    register_tensor_plan,
    run_preflight,
    validate_reference_config_profile,
    write_tensor_plan,
)
from row4.quant import decode_row4_codes  # noqa: E402
from row4.guard import is_qwen3_row4_checkpoint  # noqa: E402


def _bf16_from_bits(bits: list[int]) -> torch.Tensor:
    payload = np.asarray(bits, dtype=np.uint16)
    return torch.from_numpy(payload.view(np.int16)).view(torch.bfloat16)


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
            "attn_factor": REFERENCE_YARN_RAW_ATTN_FACTOR,
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


def test_signed_bf16_scale_payload_is_bit_preserving() -> None:
    bits = [0x0000, 0x8000, 0x3F80, 0xBF80, 0x0001, 0x8001, 0x3C00]
    scale = _bf16_from_bits(bits)
    result = bf16_payload(scale, "scale")
    np.testing.assert_array_equal(result, np.asarray(bits, dtype=np.uint16))
    assert int(torch.count_nonzero(scale <= 0).item()) == 4


@pytest.mark.parametrize("bits", [[0x7F80], [0xFF80], [0x7FC1]])
def test_bf16_payload_rejects_nonfinite_values(bits: list[int]) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        bf16_payload(_bf16_from_bits(bits), "scale")


class _FakeReader:
    def __init__(self) -> None:
        self.weights: dict[str, torch.Tensor] = {}
        self.scales: dict[str, torch.Tensor] = {}

    def shape(self, name: str) -> tuple[int, ...]:
        tensor = self.weights.get(name, self.scales.get(name))
        assert tensor is not None
        return tuple(tensor.shape)

    def get(self, name: str) -> torch.Tensor:
        return self.scales[name]

    @contextmanager
    def open_2d(self, name: str) -> Iterator[Callable[[int, int], torch.Tensor]]:
        tensor = self.weights[name]
        yield lambda start, end: tensor[start:end, :]


class _PlanReader:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.tensors[name].shape)

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]

    @contextmanager
    def open_2d(self, name: str) -> Iterator[Callable[[int, int], torch.Tensor]]:
        tensor = self.tensors[name]
        yield lambda start, end: tensor[start:end, :]


def test_fused_row4_stream_preserves_matrix_order_and_exact_bytes() -> None:
    reader = _FakeReader()
    weight_names = ["q.weight", "k.weight", "v.weight"]
    scale_names = ["q.scale", "k.scale", "v.scale"]
    for code, weight_name, scale_name in zip((1, 2, 3), weight_names, scale_names):
        logical_group = decode_row4_codes(np.full(128, code, dtype=np.uint8))
        logical_weight = np.tile(logical_group, (32, 1))
        reader.weights[weight_name] = torch.from_numpy(logical_weight).to(torch.bfloat16)
        reader.scales[scale_name] = torch.full((128,), float(code), dtype=torch.bfloat16)

    bundle = Row4BundleWriter(reader, weight_names, scale_names)  # type: ignore[arg-type]
    code_chunks = list(bundle.iter_codes())
    assert [chunk.shape for chunk in code_chunks] == [(8, 1, 4, 64)] * 3
    code_stream = b"".join(chunk.tobytes() for chunk in code_chunks)
    assert len(code_stream) == 3 * 128 * 128 // 8
    assert [code_stream[offset] for offset in (0, 2048, 4096)] == [0x11, 0x22, 0x33]

    scale_stream = b"".join(chunk.tobytes() for chunk in bundle.iter_scales())
    assert len(scale_stream) == 3 * 128 * 2
    scale_bits = np.frombuffer(scale_stream, dtype=np.uint16)
    assert [int(scale_bits[offset]) for offset in (0, 128, 256)] == [0x3F80, 0x4000, 0x4040]


def test_tiny_tensor_plan_streams_a_real_64_aligned_gguf(tmp_path: Path) -> None:
    dimensions = Qwen3Dimensions(
        layers=1,
        hidden=128,
        intermediate=128,
        heads=1,
        kv_heads=1,
        head_dim=128,
        vocab=128,
    )
    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.zeros((128, 128), dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(128, dtype=torch.bfloat16),
        "lm_head.weight": torch.arange(-64, 64, dtype=torch.float32)[None, :]
        .repeat(128, 1)
        .to(torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones(128, dtype=torch.bfloat16),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(128, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_norm.weight": torch.ones(128, dtype=torch.bfloat16),
        "model.layers.0.self_attn.k_norm.weight": torch.ones(128, dtype=torch.bfloat16),
    }
    suffixes = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    for code, suffix in enumerate(suffixes, start=1):
        group = decode_row4_codes(np.full(128, code, dtype=np.uint8))
        tensors[f"model.layers.0.{suffix}.weight"] = torch.from_numpy(
            np.tile(group, (32, 1))
        ).to(torch.bfloat16)
        tensors[f"model.layers.0.{suffix}.weight_scale"] = torch.full(
            (128,), float(code), dtype=torch.bfloat16
        )

    reader = _PlanReader(tensors)
    plan = build_tensor_plan(reader, dimensions)  # type: ignore[arg-type]
    assert len(plan) == 16
    output = tmp_path / "tiny-row4.gguf"
    writer = gguf.GGUFWriter(output, arch="qwen3")
    add_model_metadata(writer, {"rms_norm_eps": 1.0e-6}, dimensions)
    register_tensor_plan(writer, plan)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    write_tensor_plan(writer, plan, verbose=False)
    writer.close()

    gguf_reader = gguf.GGUFReader(output)
    assert gguf_reader.alignment == 64
    assert gguf_reader.get_field("general.quantization_version").contents() == 2
    assert gguf_reader.get_field("tokenizer.ggml.add_bos_token").contents() is True
    assert [tensor.name for tensor in gguf_reader.tensors] == [entry.name for entry in plan]
    assert all(tensor.data_offset % 64 == 0 for tensor in gguf_reader.tensors)
    tensor_map = {tensor.name: tensor for tensor in gguf_reader.tensors}
    qkv = tensor_map["blk.0.attn_qkv.row4.codes"]
    assert qkv.tensor_type == gguf.GGMLQuantizationType.ROW4_CODES
    np.testing.assert_array_equal(qkv.shape, np.asarray([64, 4, 1, 24], dtype=np.uint64))
    qkv_bytes = np.asarray(qkv.data).reshape(-1)
    assert [int(qkv_bytes[offset]) for offset in (0, 2048, 4096)] == [0x11, 0x22, 0x33]
    output_codes = tensor_map["output.w8.codes"]
    assert output_codes.tensor_type == gguf.GGMLQuantizationType.I8
    np.testing.assert_array_equal(
        output_codes.shape,
        np.asarray([128, 16, 1, 8], dtype=np.uint64),
    )
    assert output.stat().st_size >= sum(entry.nbytes for entry in plan)


def test_reference_checkpoint_statistics_when_requested() -> None:
    checkpoint = os.environ.get("ROW4_CHECKPOINT_DIR")
    if checkpoint is None:
        pytest.skip("set ROW4_CHECKPOINT_DIR to run the 4-shard reference preflight")
    report = run_preflight(Path(checkpoint), None)
    assert report.scale_values == REFERENCE_SCALE_VALUES
    assert report.nonpositive_scales == REFERENCE_NONPOSITIVE_SCALES
    assert report.tensor_payload_bytes == REFERENCE_PAYLOAD_BYTES


def test_reference_config_profile_is_fixed_but_plan_helper_remains_generic() -> None:
    dimensions = validate_reference_config_profile(_reference_config())
    assert dimensions.reference_tuple == (36, 4096, 12288, 32, 8, 128, 151936)

    tiny = Qwen3Dimensions(1, 128, 128, 1, 1, 128, 128)
    assert tiny.reference_tuple == (1, 128, 128, 1, 1, 128, 128)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("num_hidden_layers", 35, "reference dimensions"),
        ("hidden_act", "gelu", "hidden_act"),
        ("tie_word_embeddings", True, "tie_word_embeddings"),
        ("attention_bias", True, "attention_bias"),
        ("transformers_version", "5.2.1", "transformers_version"),
        ("use_sliding_window", True, "use_sliding_window"),
    ],
)
def test_reference_config_profile_rejects_semantic_drift(
    key: str,
    value: object,
    message: str,
) -> None:
    config = _reference_config()
    config[key] = value
    with pytest.raises(ValueError, match=message):
        validate_reference_config_profile(config)


def test_reference_config_requires_all_full_attention_layers() -> None:
    config = _reference_config()
    config["layer_types"] = ["full_attention"] * 35 + ["sliding_attention"]
    with pytest.raises(ValueError, match="layer_types"):
        validate_reference_config_profile(config)


@pytest.mark.parametrize(
    ("extra_key", "value"),
    [
        ("attention_factor", 1.0),
        ("beta_fast", 32.0),
        ("beta_slow", 1.0),
        ("truncate", True),
        ("partial_rotary_factor", 0.5),
        ("mscale", 1.0),
        ("mscale_all_dim", 1.0),
    ],
)
def test_reference_config_rejects_extra_semantic_rope_keys(
    extra_key: str,
    value: object,
) -> None:
    config = _reference_config()
    rope = config["rope_parameters"]
    assert isinstance(rope, dict)
    rope[extra_key] = value
    with pytest.raises(ValueError, match="exact rope_parameters key set"):
        validate_reference_config_profile(config)


def test_yarn_gguf_value_is_pre_factor_not_effective_mscale() -> None:
    assert REFERENCE_YARN_RAW_ATTN_FACTOR == pytest.approx(
        1.0 / REFERENCE_YARN_EFFECTIVE_MSCALE
    )
    assert REFERENCE_YARN_GGUF_PRE_FACTOR == 1.0
    effective_runtime_mscale = (
        REFERENCE_YARN_GGUF_PRE_FACTOR * REFERENCE_YARN_EFFECTIVE_MSCALE
    )
    assert effective_runtime_mscale == pytest.approx(1.138629436111989)
    assert effective_runtime_mscale != pytest.approx(REFERENCE_YARN_RAW_ATTN_FACTOR)


def _write_tokenizer(
    root: Path,
    *,
    vocab: dict[str, object],
    added_tokens: list[object],
) -> None:
    (root / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": vocab}, "added_tokens": added_tokens}),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_tokenizer_preflight_accepts_unique_disjoint_ids(tmp_path: Path) -> None:
    _write_tokenizer(
        tmp_path,
        vocab={"a": 0, "b": 1},
        added_tokens=[{"id": 2, "content": "<special>", "special": True}],
    )
    tokenizer, tokenizer_config = _load_tokenizer_files(tmp_path, 4)
    assert tokenizer["added_tokens"] == [
        {"id": 2, "content": "<special>", "special": True}
    ]
    assert tokenizer_config == {}


def _write_reference_bos_contract(root: Path, chat_template: str) -> None:
    _write_tokenizer(
        root,
        vocab={"a": 0},
        added_tokens=[
            {"id": 1, "content": REFERENCE_BOS_TOKEN, "special": True},
        ],
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": REFERENCE_BOS_TOKEN}),
        encoding="utf-8",
    )
    (root / "chat_template.jinja").write_text(chat_template, encoding="utf-8")


def test_reference_bos_contract_has_one_template_bos_and_one_automatic_bos(
    tmp_path: Path,
) -> None:
    _write_reference_bos_contract(
        tmp_path,
        "{% set prompt = messages[0]['content'] %}"
        f"{REFERENCE_CHAT_TEMPLATE_BOS_EXPRESSION}{{{{ prompt }}}}",
    )
    _validate_tokenizer_files(tmp_path, 2)

    # common/chat.cpp removes this rendered leading BOS when add_bos is true;
    # the tokenizer then prepends the one automatic BOS requested by the GGUF.
    rendered = REFERENCE_BOS_TOKEN + "hello"
    prompt_for_tokenizer = rendered.removeprefix(REFERENCE_BOS_TOKEN)
    tokenized = [1, 0] if prompt_for_tokenizer == "hello" else []
    assert tokenized.count(1) == 1


@pytest.mark.parametrize(
    ("template", "message"),
    [
        ("{{ messages[0]['content'] }}", "exactly one"),
        (
            f"{REFERENCE_CHAT_TEMPLATE_BOS_EXPRESSION}"
            f"{REFERENCE_CHAT_TEMPLATE_BOS_EXPRESSION}",
            "exactly one",
        ),
        (
            "{{ messages[0]['content'] }}"
            f"{REFERENCE_CHAT_TEMPLATE_BOS_EXPRESSION}",
            "first output expression",
        ),
    ],
)
def test_reference_bos_contract_rejects_missing_or_duplicate_template_bos(
    tmp_path: Path,
    template: str,
    message: str,
) -> None:
    _write_reference_bos_contract(tmp_path, template)
    with pytest.raises(ValueError, match=message):
        _validate_tokenizer_files(tmp_path, 2)


def test_reference_bos_contract_rejects_tokenizer_override(tmp_path: Path) -> None:
    _write_reference_bos_contract(tmp_path, REFERENCE_CHAT_TEMPLATE_BOS_EXPRESSION)
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": REFERENCE_BOS_TOKEN,
                "add_bos_token": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must omit add_bos_token"):
        _validate_tokenizer_files(tmp_path, 2)


@pytest.mark.parametrize(
    ("vocab", "added_tokens", "message"),
    [
        ({"a": True}, [], "base id"),
        ({"a": 0, "b": 0}, [], "assigned to both"),
        ({"a": 0}, [{"id": False, "content": "x"}], "added token id"),
        ({"a": 0}, [{"id": 4, "content": "x"}], "must be in"),
        ({"a": 0}, [{"id": 0, "content": "x"}], "collides"),
        (
            {"a": 0},
            [{"id": 1, "content": "x"}, {"id": 1, "content": "y"}],
            "assigned to both",
        ),
    ],
)
def test_tokenizer_preflight_rejects_invalid_or_colliding_ids(
    tmp_path: Path,
    vocab: dict[str, object],
    added_tokens: list[object],
    message: str,
) -> None:
    _write_tokenizer(tmp_path, vocab=vocab, added_tokens=added_tokens)
    with pytest.raises(ValueError, match=message):
        _load_tokenizer_files(tmp_path, 4)


def test_file_publish_is_private_atomic_and_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "model.gguf"
    temporary = _create_private_temporary(output)
    assert stat.S_IMODE(temporary.directory.stat().st_mode) == 0o700
    temporary.output.write_bytes(b"complete")
    _atomic_publish_no_clobber(temporary.output, output)
    _cleanup_private_temporary(temporary)
    assert output.read_bytes() == b"complete"
    assert not temporary.directory.exists()

    contender = _create_private_temporary(output)
    contender.output.write_bytes(b"contender")
    with pytest.raises(FileExistsError):
        _atomic_publish_no_clobber(contender.output, output)
    _cleanup_private_temporary(contender)
    assert output.read_bytes() == b"complete"


def test_file_publish_concurrent_creators_do_not_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "model.gguf"
    temporaries = [_create_private_temporary(output) for _ in range(2)]
    for index, temporary in enumerate(temporaries):
        temporary.output.write_bytes(f"writer-{index}".encode())
    barrier = threading.Barrier(2)
    results: list[BaseException | None] = []

    def publish(temporary: converter.PrivateTemporaryOutput) -> None:
        barrier.wait()
        try:
            _atomic_publish_no_clobber(temporary.output, output)
        except BaseException as exc:
            results.append(exc)
        else:
            results.append(None)

    threads = [threading.Thread(target=publish, args=(temporary,)) for temporary in temporaries]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for temporary in temporaries:
        _cleanup_private_temporary(temporary)

    assert sum(result is None for result in results) == 1
    failures = [result for result in results if result is not None]
    assert len(failures) == 1 and isinstance(failures[0], FileExistsError)
    assert output.read_bytes() in (b"writer-0", b"writer-1")


def test_file_publish_does_not_replace_broken_symlink(tmp_path: Path) -> None:
    output = tmp_path / "model.gguf"
    output.symlink_to(tmp_path / "missing")
    temporary = _create_private_temporary(output)
    temporary.output.write_bytes(b"complete")
    with pytest.raises(FileExistsError):
        _atomic_publish_no_clobber(temporary.output, output)
    _cleanup_private_temporary(temporary)
    assert output.is_symlink()


def test_private_temp_rejects_hostile_writable_parent_before_creation(tmp_path: Path) -> None:
    parent = tmp_path / "world-writable"
    parent.mkdir(mode=0o777)
    parent.chmod(0o777)
    output = parent / "model.gguf"
    victim = tmp_path / "victim"
    victim.write_bytes(b"unchanged")
    (parent / ".model.gguf.tmp").symlink_to(victim)

    with pytest.raises(PermissionError, match="group/world-writable"):
        _create_private_temporary(output)
    assert not output.exists() and not output.is_symlink()
    assert list(parent.glob(".model.gguf.tmp-*")) == []
    assert victim.read_bytes() == b"unchanged"


@pytest.mark.parametrize("mode", [0o755, 0o1777])
def test_private_temp_rejects_writable_untrusted_ancestor_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: int,
) -> None:
    parent = tmp_path / "untrusted-owner"
    parent.mkdir()
    parent.chmod(mode)
    resolved_parent = parent.resolve()
    real_stat = converter.os.stat
    untrusted_uid = max(os.geteuid() + 1, 1)

    def fake_stat(path: os.PathLike[str] | str, *, follow_symlinks: bool = True):
        result = real_stat(path, follow_symlinks=follow_symlinks)
        if Path(path) == resolved_parent:
            fields = list(result)
            fields[4] = untrusted_uid
            return os.stat_result(fields)
        return result

    with monkeypatch.context() as patcher:
        patcher.setattr(converter.os, "stat", fake_stat)
        with pytest.raises(PermissionError, match="untrusted owner"):
            _create_private_temporary(parent / "model.gguf")
    assert list(parent.glob(".model.gguf.tmp-*")) == []


def test_private_temp_allows_trusted_sticky_ancestor(tmp_path: Path) -> None:
    parent = tmp_path / "trusted-sticky"
    parent.mkdir()
    parent.chmod(0o1777)
    temporary = _create_private_temporary(parent / "model.gguf")
    _cleanup_private_temporary(temporary)
    assert not temporary.directory.exists()


def test_private_temp_cleanup_does_not_delete_replacement_path(tmp_path: Path) -> None:
    output = tmp_path / "model.gguf"
    temporary = _create_private_temporary(output)
    temporary.output.write_bytes(b"private")
    moved = tmp_path / "moved-private"
    temporary.directory.rename(moved)
    temporary.directory.mkdir(mode=0o700)
    replacement_marker = temporary.directory / "concurrent-marker"
    replacement_marker.write_bytes(b"keep")

    _cleanup_private_temporary(temporary)
    assert replacement_marker.read_bytes() == b"keep"
    assert not (moved / output.name).exists()


def test_temp_cleanup_never_deletes_concurrent_public_replacement(tmp_path: Path) -> None:
    output = tmp_path / "model.gguf"
    temporary = _create_private_temporary(output)
    temporary.output.write_bytes(b"published")
    _atomic_publish_no_clobber(temporary.output, output)
    output.unlink()
    output.write_bytes(b"concurrent")

    _cleanup_private_temporary(temporary)
    assert output.read_bytes() == b"concurrent"


def test_temp_cleanup_error_leaves_published_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "model.gguf"
    temporary = _create_private_temporary(output)
    temporary.output.write_bytes(b"published")
    _atomic_publish_no_clobber(temporary.output, output)
    real_unlink = converter.os.unlink

    def fail_private_unlink(path: str, *, dir_fd: int | None = None) -> None:
        if dir_fd is not None:
            raise OSError("injected private cleanup failure")
        real_unlink(path)

    with monkeypatch.context() as patcher:
        patcher.setattr(converter.os, "unlink", fail_private_unlink)
        _cleanup_private_temporary(temporary)
    assert output.read_bytes() == b"published"
    temporary.output.unlink()
    temporary.directory.rmdir()


def test_failed_conversion_removes_private_temp_and_public_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingWriter:
        def __init__(self, path: str | None, arch: str):
            assert arch == "qwen3"
            assert path is None

        def write_header_to_file(self, path: Path) -> None:
            Path(path).write_bytes(b"incomplete")
            raise RuntimeError("injected write failure")

        def close(self) -> None:
            pass

    monkeypatch.setattr(converter.gguf, "GGUFWriter", FailingWriter)
    monkeypatch.setattr(converter, "add_model_metadata", lambda *_args: None)
    monkeypatch.setattr(converter, "add_qwen2_vocab", lambda *_args: None)
    monkeypatch.setattr(converter, "register_tensor_plan", lambda *_args: None)

    output = tmp_path / "failed.gguf"
    report = SimpleNamespace(config={}, dimensions=None)
    with pytest.raises(RuntimeError, match="injected write failure"):
        converter.write_conversion_output(
            tmp_path,
            output,
            report,
            [],
            verbose=False,
        )
    assert not output.exists() and not output.is_symlink()
    assert list(tmp_path.glob(".failed.gguf.tmp-*")) == []


def test_generic_converter_row4_rejection_exempts_vocab_only() -> None:
    converter = Path(__file__).resolve().parents[3] / "convert_hf_to_gguf.py"
    tree = ast.parse(converter.read_text(encoding="utf-8"), filename=str(converter))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "should_reject_qwen3_row4_checkpoint"
    )
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(converter), "exec"), namespace)
    guard = namespace["should_reject_qwen3_row4_checkpoint"]
    assert callable(guard)
    assert guard(detected=True, vocab_only=False) is True
    assert guard(detected=True, vocab_only=True) is False
    assert guard(detected=False, vocab_only=False) is False


def test_generic_converter_guard_detects_auto_map_or_saved_scales(tmp_path: Path) -> None:
    dense = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
    }
    assert not is_qwen3_row4_checkpoint(tmp_path, dense)

    specialized = {
        **dense,
        "auto_map": {
            "AutoModelForCausalLM": "modeling_qwen3_row4_int8.Qwen3ForCausalLM",
        },
    }
    assert is_qwen3_row4_checkpoint(tmp_path, specialized)

    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"model.layers.0.self_attn.q_proj.weight_scale":"model.safetensors"}}',
        encoding="utf-8",
    )
    assert is_qwen3_row4_checkpoint(tmp_path, dense)
    assert not is_qwen3_row4_checkpoint(tmp_path, {"model_type": "llama"})


def test_generic_converter_guard_detects_single_safetensors_header(tmp_path: Path) -> None:
    header = json.dumps(
        {
            "model.layers.0.self_attn.q_proj.weight_scale": {
                "dtype": "BF16",
                "shape": [128],
                "data_offsets": [0, 256],
            }
        }
    ).encode("utf-8")
    with (tmp_path / "model.safetensors").open("wb") as file:
        file.write(len(header).to_bytes(8, "little"))
        file.write(header)
    assert is_qwen3_row4_checkpoint(
        tmp_path,
        {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
    )
