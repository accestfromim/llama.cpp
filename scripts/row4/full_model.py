#!/usr/bin/env python3
"""Run the original Qwen3 modeling file with pure-Torch Row4/W8 shims.

This is the full-model companion to ``row4_oracle.py``.  It injects a minimal
``row4_qat`` module before importing the checkpoint's modeling source, so the
original state dict can load without the unavailable training package.  The
linear shims implement the frozen BF16/A8/I32/BF16 profile exactly; attention
is still selected by Transformers and must be FA3 for a reference capture.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - dependency diagnostic
    raise SystemExit("full_model.py requires PyTorch") from exc

if __package__:
    from .numeric import row4_linear_selected, w8_linear_selected
    from .row4_oracle import (
        DEFAULT_CHECKPOINT,
        NUMERIC_PROFILE,
        ArtifactWriter,
        CheckpointReader,
        _ensure_external_output,
        _git_revision,
        _json_object,
        _publish_directory_no_clobber,
        file_record,
        parse_int_list,
        reference_checkpoint_issues,
        reference_config_issues,
        REFERENCE_TRANSFORMERS_VERSION,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from numeric import row4_linear_selected, w8_linear_selected  # type: ignore[no-redef]
    from row4_oracle import (  # type: ignore[no-redef]
        DEFAULT_CHECKPOINT,
        NUMERIC_PROFILE,
        ArtifactWriter,
        CheckpointReader,
        _ensure_external_output,
        _git_revision,
        _json_object,
        _publish_directory_no_clobber,
        file_record,
        parse_int_list,
        reference_checkpoint_issues,
        reference_config_issues,
        REFERENCE_TRANSFORMERS_VERSION,
    )


DEFAULT_PREFILL_IDS = "151643,785,4226,374"  # BOS + "The answer is"
DEFAULT_DECODE_IDS = "220,19"  # fixed continuation: whitespace, "4"
EXPECTED_ROW4_MODULES = 36 * 7
EXPECTED_W8_MODULES = 1


@dataclass
class ShimOptions:
    rows_per_chunk: int = 128
    cache_decoded_row4: bool = False
    require_int_mm: bool = True


SHIM_OPTIONS = ShimOptions()
MATMUL_PATH_COUNTS: dict[str, int] = {"torch._int_mm": 0, "torch.matmul_i32": 0}


def configure_shims(
    *,
    rows_per_chunk: int,
    cache_decoded_row4: bool,
    require_int_mm: bool,
) -> None:
    if rows_per_chunk <= 0 or rows_per_chunk % 16:
        raise ValueError("--rows-per-chunk must be a positive multiple of 16")
    SHIM_OPTIONS.rows_per_chunk = rows_per_chunk
    SHIM_OPTIONS.cache_decoded_row4 = cache_decoded_row4
    SHIM_OPTIONS.require_int_mm = require_int_mm
    for key in MATMUL_PATH_COUNTS:
        MATMUL_PATH_COUNTS[key] = 0


def _round_half_away(values: torch.Tensor) -> torch.Tensor:
    return torch.copysign(torch.floor(torch.abs(values) + 0.5), values)


def _quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[-1] % 128:
        raise ValueError(f"Row4 activation K must be 128-aligned, got {x.shape[-1]}")
    carrier = x.to(torch.float32).to(torch.bfloat16).to(torch.float32)
    flat = carrier.reshape(-1, carrier.shape[-1])
    if not bool(torch.isfinite(flat).all()):
        raise ValueError("activation contains a non-finite value")
    amax = torch.amax(torch.abs(flat), dim=1)
    scales = torch.maximum(
        amax / 127.0,
        torch.tensor(1.0e-8, device=x.device, dtype=torch.float32),
    )
    codes = torch.clamp(_round_half_away(flat / scales[:, None]), -127, 127).to(torch.int8)
    return codes.contiguous(), scales.to(torch.float32)


def _decode_row4_weight(weight_bf16: torch.Tensor) -> torch.Tensor:
    if weight_bf16.dtype != torch.bfloat16 or weight_bf16.ndim != 2:
        raise TypeError("Row4 checkpoint weight must be two-dimensional BF16")
    out_features, in_features = weight_bf16.shape
    if out_features % 4 or in_features % 128:
        raise ValueError(f"Row4 weight must be O4/K128 aligned, got {tuple(weight_bf16.shape)}")
    weight = weight_bf16.to(torch.float32)
    grouped = weight.reshape(out_features // 4, 4, in_features)
    a, b, c, d = (grouped[:, index, :] for index in range(4))
    u_re = 0.5 * (a + d)
    u_im = 0.5 * (c - b)
    v_re = 0.5 * (a - d)
    v_im = 0.5 * (b + c)

    def axis(real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real_dominant = torch.abs(real) > torch.abs(imag)
        one_real = torch.ones_like(real, dtype=torch.int8)
        one_imag = torch.ones_like(imag, dtype=torch.int8)
        real_value = torch.where(real < 0, -one_real, one_real)
        imag_value = torch.where(imag < 0, -one_imag, one_imag)
        zero_real = torch.zeros_like(real_value)
        zero_imag = torch.zeros_like(imag_value)
        return (
            torch.where(real_dominant, real_value, zero_real),
            torch.where(real_dominant, zero_imag, imag_value),
        )

    u_real, u_imag = axis(u_re, u_im)
    v_real, v_imag = axis(v_re, v_im)
    decoded = torch.stack(
        (u_real + v_real, -u_imag + v_imag, u_imag + v_imag, u_real - v_real),
        dim=1,
    )
    return decoded.reshape(out_features, in_features).to(torch.int8).contiguous()


def _exact_int_mm(lhs_i8: torch.Tensor, rhs_i8: torch.Tensor) -> torch.Tensor:
    lhs_i8 = lhs_i8.contiguous()
    rhs_i8 = rhs_i8.contiguous()
    if lhs_i8.dtype != torch.int8 or rhs_i8.dtype != torch.int8:
        raise TypeError("exact integer GEMM requires int8 operands")
    if hasattr(torch, "_int_mm"):
        try:
            result = torch._int_mm(lhs_i8, rhs_i8)
            if result.dtype != torch.int32:
                raise RuntimeError(f"torch._int_mm returned {result.dtype}, expected int32")
            MATMUL_PATH_COUNTS["torch._int_mm"] += 1
            return result
        except RuntimeError:
            if SHIM_OPTIONS.require_int_mm:
                raise
    if SHIM_OPTIONS.require_int_mm:
        raise RuntimeError("this PyTorch build does not provide a usable torch._int_mm")
    MATMUL_PATH_COUNTS["torch.matmul_i32"] += 1
    return torch.matmul(lhs_i8.to(torch.int32), rhs_i8.to(torch.int32))


def _finish_chunk(
    accumulator: torch.Tensor,
    activation_scales: torch.Tensor,
    row_scales: torch.Tensor,
) -> torch.Tensor:
    result = accumulator.to(torch.float32) * activation_scales[:, None]
    result = result * row_scales.to(torch.float32)[None, :]
    return result.to(torch.bfloat16)


class Row4Int8Linear(nn.Linear):
    """State-dict-compatible pure-Torch replacement for row4_qat."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs: Any):
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, device=self.weight.device, dtype=torch.bfloat16)
        )
        self._oracle_decoded_weight: torch.Tensor | None = None

    def _decoded_weight(self) -> torch.Tensor | None:
        if not SHIM_OPTIONS.cache_decoded_row4:
            return None
        cached = self._oracle_decoded_weight
        if cached is None or cached.device != self.weight.device:
            chunks = []
            for start in range(0, self.out_features, SHIM_OPTIONS.rows_per_chunk):
                end = min(start + SHIM_OPTIONS.rows_per_chunk, self.out_features)
                chunks.append(_decode_row4_weight(self.weight[start:end]))
            cached = torch.cat(chunks, dim=0)
            self._oracle_decoded_weight = cached
        return cached

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            raise RuntimeError("Qwen3 Row4 v1 does not support projection bias")
        if self.weight.dtype != torch.bfloat16 or self.weight_scale.dtype != torch.bfloat16:
            raise TypeError("Row4 checkpoint weight and weight_scale must remain BF16")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Row4 input K mismatch: {x.shape[-1]} != {self.in_features}")
        activation_codes, activation_scales = _quantize_activation(x)
        cached = self._decoded_weight()
        outputs = []
        for start in range(0, self.out_features, SHIM_OPTIONS.rows_per_chunk):
            end = min(start + SHIM_OPTIONS.rows_per_chunk, self.out_features)
            decoded = cached[start:end] if cached is not None else _decode_row4_weight(self.weight[start:end])
            accumulator = _exact_int_mm(activation_codes, decoded.transpose(0, 1).contiguous())
            outputs.append(_finish_chunk(accumulator, activation_scales, self.weight_scale[start:end]))
        output = torch.cat(outputs, dim=1)
        return output.reshape(*x.shape[:-1], self.out_features)


class Int8Linear(nn.Linear):
    """Dynamic per-row W8A8 lm_head matching the checkpoint modeling note."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            raise RuntimeError("Qwen3 Row4 v1 does not support lm_head bias")
        if self.weight.dtype != torch.bfloat16:
            raise TypeError("lm_head checkpoint weight must remain BF16")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"W8 input K mismatch: {x.shape[-1]} != {self.in_features}")
        activation_codes, activation_scales = _quantize_activation(x)
        outputs = []
        for start in range(0, self.out_features, SHIM_OPTIONS.rows_per_chunk):
            end = min(start + SHIM_OPTIONS.rows_per_chunk, self.out_features)
            weight = self.weight[start:end].to(torch.float32)
            amax = torch.amax(torch.abs(weight), dim=1)
            row_scales = amax / 127.0
            nonzero = amax != 0
            normalized = torch.zeros_like(weight)
            normalized[nonzero] = weight[nonzero] / row_scales[nonzero, None]
            weight_codes = torch.clamp(_round_half_away(normalized), -127, 127).to(torch.int8)
            weight_codes[~nonzero] = 0
            row_scales[~nonzero] = 0.0
            accumulator = _exact_int_mm(activation_codes, weight_codes.transpose(0, 1).contiguous())
            outputs.append(_finish_chunk(accumulator, activation_scales, row_scales))
        output = torch.cat(outputs, dim=1)
        return output.reshape(*x.shape[:-1], self.out_features)


def install_row4_qat_shim() -> None:
    shim = types.ModuleType("row4_qat")
    shim.__path__ = []  # type: ignore[attr-defined]
    shim.Int8Linear = Int8Linear  # type: ignore[attr-defined]
    shim.Row4Int8Linear = Row4Int8Linear  # type: ignore[attr-defined]
    shim.__all__ = ["Int8Linear", "Row4Int8Linear"]  # type: ignore[attr-defined]
    liger = types.ModuleType("row4_qat.liger_int8")
    liger.LIGER_OK = False  # type: ignore[attr-defined]

    def unavailable_liger(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise RuntimeError("Liger training loss is unavailable in the inference oracle")

    liger.fused_int8_linear_cross_entropy = unavailable_liger  # type: ignore[attr-defined]
    sys.modules["row4_qat"] = shim
    sys.modules["row4_qat.liger_int8"] = liger


def import_checkpoint_modeling(checkpoint: Path):
    install_row4_qat_shim()
    source = checkpoint / "modeling_qwen3_row4_int8.py"
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint modeling source is missing: {source}")
    module_name = "qwen3_row4_oracle_checkpoint_modeling"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import checkpoint modeling source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def reference_issues(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    checkpoint: CheckpointReader,
) -> list[str]:
    issues = reference_config_issues(config)
    issues.extend(reference_checkpoint_issues(checkpoint))
    transformers_version = package_version("transformers")
    if transformers_version != REFERENCE_TRANSFORMERS_VERSION:
        issues.append(
            f"installed transformers version must be {REFERENCE_TRANSFORMERS_VERSION}, "
            f"got {transformers_version}"
        )
    if package_version("accelerate") is None:
        issues.append("reference capture requires Accelerate for low-memory direct-to-CUDA loading")
    training_args = _json_object(args.checkpoint / "args.json")
    expected_attention = training_args.get("attn_impl")
    if expected_attention != "flash_attention_3":
        issues.append(
            "checkpoint training attn_impl must be 'flash_attention_3', "
            f"got {expected_attention!r}"
        )
    if args.attn_implementation != expected_attention:
        issues.append(
            f"attention backend must be training attn_impl={expected_attention!r}, "
            f"requested {args.attn_implementation!r}"
        )
    expected_training_flags = {
        "bf16": True,
        "fp16": False,
        "torch_dtype": "bfloat16",
        "use_cache": False,
    }
    for key, expected in expected_training_flags.items():
        actual = training_args.get(key)
        if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
            issues.append(f"checkpoint training {key} must be {expected!r}, got {actual!r}")
    if getattr(args, "skip_shard_hashes", False):
        issues.append("reference capture requires SHA-256 hashes for all four checkpoint shards")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        issues.append("reference capture requires a CUDA device")
    else:
        index = device.index if device.index is not None else torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(index)
        if capability < (9, 0):
            issues.append(f"FA3 reference requires Hopper-class compute capability, got {capability}")
    if args.attn_implementation != "flash_attention_3":
        issues.append("reference capture requires flash_attention_3")
    else:
        try:
            from transformers.utils import is_flash_attn_3_available

            if not is_flash_attn_3_available():
                issues.append("Transformers cannot detect a directly installed FA3 implementation")
        except (ImportError, AttributeError):
            issues.append("installed Transformers cannot query FA3 availability")
    return issues


def environment_record(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    record: dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "accelerate": package_version("accelerate"),
        "flash_attn": package_version("flash-attn"),
        "flash_attn_3": package_version("flash-attn-3"),
        "kernels": package_version("kernels"),
        "safetensors": package_version("safetensors"),
        "device": str(device),
        "attention_implementation": args.attn_implementation,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        record.update(
            {
                "device_name": properties.name,
                "device_total_memory": properties.total_memory,
                "compute_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    elif device.type == "mps":
        record["mps_available"] = torch.backends.mps.is_available()
    return record


class CarrierCapture:
    def __init__(self, model: nn.Module, last_layer: int):
        self.phase: str | None = None
        self.records: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        self._device_primaries: dict[str, torch.Tensor] = {}
        self._seen: set[str] = set()
        self._handles: list[Any] = []
        targets: dict[str, tuple[str, str]] = {}
        for layer in (0, last_layer):
            prefix = f"model.layers.{layer}"
            targets.update(
                {
                    f"{prefix}.self_attn.q_proj": (f"layer{layer}.qkv", "primary"),
                    f"{prefix}.self_attn.k_proj": (f"layer{layer}.qkv", "equal"),
                    f"{prefix}.self_attn.v_proj": (f"layer{layer}.qkv", "equal"),
                    f"{prefix}.self_attn.o_proj": (f"layer{layer}.o", "primary"),
                    f"{prefix}.mlp.gate_proj": (f"layer{layer}.gate_up", "primary"),
                    f"{prefix}.mlp.up_proj": (f"layer{layer}.gate_up", "equal"),
                    f"{prefix}.mlp.down_proj": (f"layer{layer}.down", "primary"),
                }
            )
        targets["lm_head"] = ("lm_head", "primary")
        modules = dict(model.named_modules())
        missing = sorted(set(targets) - set(modules))
        if missing:
            raise RuntimeError(f"cannot install carrier hooks; modules are missing: {missing}")
        for module_name, (logical_name, role) in targets.items():
            self._handles.append(
                modules[module_name].register_forward_pre_hook(
                    self._make_hook(module_name, logical_name, role)
                )
            )

    def _make_hook(self, module_name: str, logical_name: str, role: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if self.phase is None:
                raise RuntimeError(f"carrier hook {module_name} fired outside a capture phase")
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"carrier hook {module_name} did not receive a tensor")
            carrier = inputs[0]
            if carrier.dtype != torch.bfloat16:
                raise TypeError(
                    f"reference carrier {module_name} must be BF16 before the profile round-trip, "
                    f"got {carrier.dtype}"
                )
            if role == "equal":
                primary = self._device_primaries.get(logical_name)
                if primary is None or primary.shape != carrier.shape or not torch.equal(primary, carrier):
                    raise RuntimeError(f"fused carrier equality failed at {module_name}")
                return
            if logical_name in self._seen:
                raise RuntimeError(f"carrier {logical_name} was captured more than once in {self.phase}")
            self._seen.add(logical_name)
            self._device_primaries[logical_name] = carrier.detach()
            flattened = carrier.detach().reshape(-1, carrier.shape[-1]).cpu().contiguous()
            self.records[self.phase][logical_name] = {
                "bf16": flattened,
                "f32": flattened.to(torch.float32),
            }

        return hook

    def begin(self, phase: str) -> None:
        if self.phase is not None:
            raise RuntimeError(f"capture phase {self.phase} is still active")
        if phase in self.records:
            raise ValueError(f"duplicate capture phase {phase}")
        self.phase = phase
        self.records[phase] = {}
        self._device_primaries.clear()
        self._seen.clear()

    def end(self) -> None:
        if self.phase is None:
            raise RuntimeError("no carrier capture phase is active")
        expected = {
            "layer0.qkv",
            "layer0.o",
            "layer0.gate_up",
            "layer0.down",
            "layer35.qkv",
            "layer35.o",
            "layer35.gate_up",
            "layer35.down",
            "lm_head",
        }
        actual = set(self.records[self.phase])
        if actual != expected:
            raise RuntimeError(
                f"carrier capture {self.phase} incomplete: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        self.phase = None
        self._device_primaries.clear()
        self._seen.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def validate_loaded_model(model: nn.Module) -> None:
    row4_modules = [module for module in model.modules() if isinstance(module, Row4Int8Linear)]
    w8_modules = [module for module in model.modules() if isinstance(module, Int8Linear)]
    if len(row4_modules) != EXPECTED_ROW4_MODULES:
        raise RuntimeError(f"loaded {len(row4_modules)} Row4 modules, expected {EXPECTED_ROW4_MODULES}")
    if len(w8_modules) != EXPECTED_W8_MODULES:
        raise RuntimeError(f"loaded {len(w8_modules)} W8 modules, expected {EXPECTED_W8_MODULES}")
    for module in row4_modules:
        if module.weight.dtype != torch.bfloat16 or module.weight_scale.dtype != torch.bfloat16:
            raise TypeError("loaded Row4 parameters are not BF16")
        if module.weight.shape[0] != module.weight_scale.numel():
            raise ValueError("loaded Row4 weight/scale shape mismatch")
    if w8_modules[0].weight.dtype != torch.bfloat16:
        raise TypeError("loaded lm_head weight is not BF16")


def load_full_model(args: argparse.Namespace, config_dict: Mapping[str, Any]) -> nn.Module:
    try:
        import accelerate  # noqa: F401
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"full-model loading requires transformers=={REFERENCE_TRANSFORMERS_VERSION} "
            "and accelerate in the execution environment"
        ) from exc
    module = import_checkpoint_modeling(args.checkpoint)
    config = Qwen3Config.from_pretrained(args.checkpoint)
    config._attn_implementation = args.attn_implementation
    model_class = module.Qwen3ForCausalLM
    device_map: dict[str, str] | None = {"": args.device}
    model = model_class.from_pretrained(
        args.checkpoint,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    )
    if bool(config_dict.get("tie_word_embeddings")):
        raise ValueError("full Row4 golden requires untied lm_head")
    model.eval()
    validate_loaded_model(model)
    return model


def run_phase(
    model: nn.Module,
    capture: CarrierCapture,
    phase: str,
    input_ids: Sequence[int],
    device: torch.device,
    past_key_values: Any | None,
) -> tuple[torch.Tensor, Any]:
    token_tensor = torch.tensor([list(input_ids)], dtype=torch.int64, device=device)
    capture.begin(phase)
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=token_tensor,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=0,
            )
    except BaseException:
        capture.phase = None
        capture._device_primaries.clear()
        capture._seen.clear()
        raise
    capture.end()
    logits = outputs.logits.detach().cpu().contiguous()
    if logits.dtype != torch.bfloat16:
        raise TypeError(f"checkpoint modeling returned logits dtype {logits.dtype}, expected BF16")
    return logits, outputs.past_key_values


def compute_nll(
    prefill_logits: torch.Tensor,
    decode_1_logits: torch.Tensor,
    prefill_ids: Sequence[int],
    decode_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    scored = torch.cat(
        (
            prefill_logits.reshape(-1, prefill_logits.shape[-1]),
            decode_1_logits.reshape(-1, decode_1_logits.shape[-1]),
        ),
        dim=0,
    )
    targets = torch.tensor([*prefill_ids[1:], *decode_ids], dtype=torch.int64)
    if scored.shape[0] != targets.numel():
        raise AssertionError(f"internal scored-logit/target mismatch: {scored.shape[0]} != {targets.numel()}")
    values = scored.to(torch.float64)
    rows = torch.arange(targets.numel())
    token_nll = torch.logsumexp(values, dim=-1) - values[rows, targets]
    mean_nll = torch.mean(token_nll).item()
    return scored, targets, {"mean_nll": mean_nll, "ppl": math.exp(mean_nll)}


def save_torch_file(path: Path, payload: Any) -> dict[str, Any]:
    torch.save(payload, path)
    record = file_record(path)
    record["path"] = path.name
    return record


def capture_full_model(args: argparse.Namespace) -> int:
    if args.checkpoint is None:
        raise ValueError("set ROW4_CHECKPOINT_DIR or pass --checkpoint")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    config_dict = _json_object(args.checkpoint / "config.json")
    checkpoint_reader = CheckpointReader(args.checkpoint)
    issues = reference_issues(args, config_dict, checkpoint_reader)
    if issues and not args.allow_nonreference:
        raise RuntimeError("reference environment preflight failed: " + "; ".join(issues))
    prefill_ids = parse_int_list(args.input_ids)
    decode_ids = parse_int_list(args.decode_ids)
    if len(decode_ids) != 2:
        raise ValueError("--decode-ids must contain exactly two fixed token ids")
    vocab = int(config_dict["vocab_size"])
    if any(token < 0 or token >= vocab for token in [*prefill_ids, *decode_ids]):
        raise ValueError(f"token id is outside [0, {vocab})")
    output_arg = args.output_dir or os.environ.get("ROW4_ORACLE_DIR")
    if not output_arg:
        raise ValueError("set ROW4_ORACLE_DIR or pass --output-dir")
    output_root = _ensure_external_output(Path(output_arg))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"qwen3-row4-full-{timestamp}"
    if not run_name or run_name in (".", "..") or "/" in run_name:
        raise ValueError(f"invalid --run-name: {run_name!r}")
    final_dir = output_root / run_name
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing full-model golden: {final_dir}")

    configure_shims(
        rows_per_chunk=args.rows_per_chunk,
        cache_decoded_row4=args.cache_decoded_row4,
        require_int_mm=not args.allow_int32_matmul_fallback,
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    model = load_full_model(args, config_dict)
    capture = CarrierCapture(model, int(config_dict["num_hidden_layers"]) - 1)
    device = torch.device(args.device)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{run_name}.tmp-", dir=output_root))
    writer = ArtifactWriter(temp_dir)
    try:
        prefill_logits, cache = run_phase(model, capture, "prefill", prefill_ids, device, None)
        decode_1_logits, cache = run_phase(model, capture, "decode_1", [decode_ids[0]], device, cache)
        decode_2_logits, _cache = run_phase(model, capture, "decode_2", [decode_ids[1]], device, cache)
        capture.close()

        logits_by_phase = {
            "prefill": prefill_logits,
            "decode_1": decode_1_logits,
            "decode_2": decode_2_logits,
        }
        logits_artifacts: dict[str, dict[str, str]] = {}
        for phase, logits in logits_by_phase.items():
            logits_artifacts[phase] = {
                "bf16": writer.tensor(f"{phase}/logits", logits),
                "f32": writer.tensor(f"{phase}/logits_carrier", logits.to(torch.float32)),
            }

        carrier_artifacts: dict[str, dict[str, dict[str, str]]] = {}
        convenience_files: dict[str, Any] = {}
        for phase, phase_records in capture.records.items():
            carrier_artifacts[phase] = {}
            f32_carriers: dict[str, torch.Tensor] = {}
            for name, tensors in phase_records.items():
                stem = f"{phase}/carriers/{name.replace('.', '_')}"
                carrier_artifacts[phase][name] = {
                    "bf16": writer.tensor(f"{stem}_bf16", tensors["bf16"]),
                    "f32": writer.tensor(f"{stem}_f32", tensors["f32"]),
                }
                f32_carriers[name] = tensors["f32"]
            convenience_path = temp_dir / f"{phase}_carriers.pt"
            convenience_files[f"{phase}_carriers"] = save_torch_file(convenience_path, f32_carriers)

        scored_logits, targets, quality = compute_nll(
            prefill_logits,
            decode_1_logits,
            prefill_ids,
            decode_ids,
        )
        token_nll_artifact = writer.tensor(
            "quality/scored_logits",
            scored_logits,
        )
        targets_artifact = writer.tensor("quality/target_ids", targets)
        scored_path = temp_dir / "scored_logits.pt"
        convenience_files["scored_logits"] = save_torch_file(
            scored_path,
            {"logits": scored_logits, "target_ids": targets},
        )
        for phase, logits in logits_by_phase.items():
            phase_path = temp_dir / f"{phase}_logits.pt"
            convenience_files[f"{phase}_logits"] = save_torch_file(
                phase_path,
                {"logits": logits},
            )

        metadata_names = (
            "config.json",
            "args.json",
            "model.safetensors.index.json",
            "modeling_qwen3_row4_int8.py",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        )
        manifest = {
            "schema": "qwen3_row4_full_model_golden_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reference_capture": not issues,
            "reference_issues": issues,
            "numeric_profile": NUMERIC_PROFILE,
            "linear_implementation": "row4_qat_compat_pure_torch_v1",
            "attention_implementation": args.attn_implementation,
            "git_revision": _git_revision(),
            "environment": environment_record(args),
            "seed": args.seed,
            "input_ids": prefill_ids,
            "decode_ids": decode_ids,
            "phase_positions": {
                "prefill": [0, len(prefill_ids)],
                "decode_1": [len(prefill_ids), len(prefill_ids) + 1],
                "decode_2": [len(prefill_ids) + 1, len(prefill_ids) + 2],
            },
            "checkpoint": {
                "path": str(args.checkpoint),
                "metadata_files": {
                    name: file_record(args.checkpoint / name)
                    for name in metadata_names
                    if (args.checkpoint / name).is_file()
                },
                "shards": {
                    shard: file_record(
                        args.checkpoint / shard,
                        hash_contents=not args.skip_shard_hashes,
                    )
                    for shard in checkpoint_reader.shards
                },
                "full_shard_hashes_recorded": not args.skip_shard_hashes,
            },
            "shim": {
                "rows_per_chunk": SHIM_OPTIONS.rows_per_chunk,
                "cache_decoded_row4": SHIM_OPTIONS.cache_decoded_row4,
                "require_int_mm": SHIM_OPTIONS.require_int_mm,
                "matmul_path_counts": dict(MATMUL_PATH_COUNTS),
            },
            "logits": logits_artifacts,
            "carriers": carrier_artifacts,
            "quality": {
                **quality,
                "scored_logits": token_nll_artifact,
                "target_ids": targets_artifact,
                "scored_positions": len(prefill_ids) + 1,
            },
            "argmax_ids": {
                phase: torch.argmax(logits.to(torch.float32), dim=-1).reshape(-1).tolist()
                for phase, logits in logits_by_phase.items()
            },
            "top10_ids": {
                phase: torch.topk(logits.to(torch.float32), 10, dim=-1).indices.reshape(-1, 10).tolist()
                for phase, logits in logits_by_phase.items()
            },
            "convenience_files": convenience_files,
            "artifacts": writer.records,
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publish_directory_no_clobber(temp_dir, final_dir)
    except BaseException:
        capture.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    finally:
        del model

    print(
        json.dumps(
            {
                "oracle_dir": str(final_dir),
                "reference_capture": not issues,
                "mean_nll": quality["mean_nll"],
                "ppl": quality["ppl"],
            }
        )
    )
    return 0


def preflight_command(args: argparse.Namespace) -> int:
    if args.checkpoint is None:
        raise ValueError("set ROW4_CHECKPOINT_DIR or pass --checkpoint")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    config = _json_object(args.checkpoint / "config.json")
    checkpoint = CheckpointReader(args.checkpoint)
    issues = reference_issues(args, config, checkpoint)
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_tensor_count": len(checkpoint.weight_map),
        "checkpoint_header_tensor_count": len(checkpoint.tensor_headers),
        "checkpoint_shards": list(checkpoint.shards),
        "checkpoint_shard_tensor_counts": list(checkpoint.shard_tensor_counts),
        "environment": environment_record(args),
        "reference_ready": not issues,
        "reference_issues": issues,
        "estimated_checkpoint_bytes": int(
            _json_object(args.checkpoint / "model.safetensors.index.json").get("metadata", {}).get(
                "total_size", 0
            )
        ),
        "cache_decoded_row4": args.cache_decoded_row4,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not issues or args.allow_nonreference else 1


def self_test_command(_args: argparse.Namespace) -> int:
    configure_shims(rows_per_chunk=16, cache_decoded_row4=False, require_int_mm=True)
    torch.manual_seed(7)
    x = (torch.arange(3 * 128, dtype=torch.float32).reshape(3, 128) - 191.5) / 32.0
    row4 = Row4Int8Linear(128, 16, bias=False, dtype=torch.bfloat16)
    latent = ((torch.arange(16 * 128).reshape(16, 128) * 17) % 257 - 128).to(torch.float32) / 64.0
    scales = torch.linspace(-0.03125, 0.03125, 16, dtype=torch.float32).to(torch.bfloat16)
    with torch.no_grad():
        row4.weight.copy_(latent.to(torch.bfloat16))
        row4.weight_scale.copy_(scales)
    expected = row4_linear_selected(x, row4.weight, row4.weight_scale)[-1].output_bf16
    actual = row4(x)
    if not torch.equal(actual, expected):
        raise AssertionError("pure-Torch Row4Int8Linear differs from the independent oracle")
    if set(row4.state_dict()) != {"weight", "weight_scale"}:
        raise AssertionError("Row4 shim state-dict schema mismatch")

    w8 = Int8Linear(128, 16, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        w8.weight.copy_(latent.flip(0).to(torch.bfloat16))
        w8.weight[0].zero_()
    expected_w8 = w8_linear_selected(x, w8.weight)[-1].output_bf16
    actual_w8 = w8(x)
    if not torch.equal(actual_w8, expected_w8):
        raise AssertionError("pure-Torch Int8Linear differs from the independent oracle")
    if set(w8.state_dict()) != {"weight"}:
        raise AssertionError("W8 shim state-dict schema mismatch")

    configure_shims(rows_per_chunk=16, cache_decoded_row4=True, require_int_mm=True)
    cached = row4(x)
    if not torch.equal(cached, expected):
        raise AssertionError("cached Row4 decoding changes numerical output")
    print("row4 full-model shim self-test: PASS")
    return 0


def add_environment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="checkpoint path; defaults to ROW4_CHECKPOINT_DIR",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="flash_attention_3")
    parser.add_argument("--cache-decoded-row4", action="store_true")
    parser.add_argument(
        "--skip-shard-hashes",
        action="store_true",
        help="omit shard hashes and force reference_ready/reference_capture=false",
    )
    parser.add_argument(
        "--allow-nonreference",
        action="store_true",
        help="permit version/device/attention mismatches and label output non-reference",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    self_test = subparsers.add_parser("self-test", help="test pure-Torch linear shims on tiny tensors")
    self_test.set_defaults(function=self_test_command)

    preflight = subparsers.add_parser("preflight", help="check reference environment without loading weights")
    add_environment_arguments(preflight)
    preflight.set_defaults(function=preflight_command)

    capture = subparsers.add_parser("capture", help="run prefill plus two fixed decode steps")
    add_environment_arguments(capture)
    capture.add_argument("--output-dir", type=Path, help="external root; defaults to ROW4_ORACLE_DIR")
    capture.add_argument("--run-name")
    capture.add_argument("--input-ids", default=DEFAULT_PREFILL_IDS)
    capture.add_argument("--decode-ids", default=DEFAULT_DECODE_IDS)
    capture.add_argument("--seed", type=int, default=42)
    capture.add_argument("--rows-per-chunk", type=int, default=128)
    capture.add_argument("--allow-int32-matmul-fallback", action="store_true")
    capture.set_defaults(function=capture_full_model)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
