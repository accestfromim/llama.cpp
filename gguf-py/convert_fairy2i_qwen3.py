#!/usr/bin/env python3

"""
Convert a Qwen3-based Fairy2i learned-scale W1 Hugging Face checkpoint to GGUF.

Qwen3 Fairy2i W1 checkpoints store one learned 64x64 scale tensor per QAT
linear layer. In the default bundle_v1 layout, the converter concatenates the
Q/K/V M64 tile streams into one attn_qkv bundle; tile64_v2 remains available
with separate U.s0/W.s0 tensors.
"""

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import gguf
from convert_fairy2i_qwen2 import (
    TensorReader,
    load_weight_map,
    pack_token_embedding,
    set_vocab_qwen2,
)
from fairy2i.quant.tile64_v2 import (
    FAIRY2I_TILE64,
    NUMERIC_PROFILE_LEGACY_F16_V1,
    NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1,
    merge_fairy2i_bundle_v1_m,
    quantize_linear_to_fairy2i_bundle_v1_w1_learned_scale,
    quantize_linear_to_fairy2i_tile64_v2_w1_learned_scale,
    round_up,
)
from fairy2i.spec import (
    QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE,
    WEIGHT_SCALE_DTYPE_BF16,
    WEIGHT_LAYOUT_BUNDLE_V1,
    WEIGHT_LAYOUT_TILE64_V2,
    Fairy2IMetadata,
    SCALE_SOURCE_LEARNED,
    write_metadata,
)


LINEAR_SPECS = [
    ("self_attn.q_proj.weight", "attn_q"),
    ("self_attn.k_proj.weight", "attn_k"),
    ("self_attn.v_proj.weight", "attn_v"),
    ("self_attn.o_proj.weight", "attn_output"),
    ("mlp.gate_proj.weight", "ffn_gate"),
    ("mlp.up_proj.weight", "ffn_up"),
    ("mlp.down_proj.weight", "ffn_down"),
]

QKV_LINEAR_SPECS = LINEAR_SPECS[:3]


def add_tensor_f32(writer: gguf.GGUFWriter, name: str, tensor: torch.Tensor) -> None:
    data = tensor.to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.F32)
    del data
    gc.collect()


def tensor_bf16_payload(tensor: torch.Tensor, name: str) -> np.ndarray:
    """Return the original finite checkpoint BF16 payload without widening."""

    if tensor.dtype != torch.bfloat16:
        raise ValueError(
            f"{name} must be stored as checkpoint BF16 for "
            f"{NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1}, got {tensor.dtype}"
        )
    payload = (
        tensor.detach()
        .contiguous()
        .view(torch.int16)
        .cpu()
        .numpy()
        .view(np.uint16)
    )
    if np.any((payload & np.uint16(0x7F80)) == np.uint16(0x7F80)):
        raise ValueError(f"{name} contains a non-finite checkpoint BF16 value")
    return np.ascontiguousarray(payload)


def add_tensor_bf16(
    writer: gguf.GGUFWriter,
    name: str,
    tensor: torch.Tensor,
    checkpoint_name: str,
) -> None:
    payload = tensor_bf16_payload(tensor, checkpoint_name)
    writer.add_tensor(name, payload, raw_dtype=gguf.GGMLQuantizationType.BF16)
    del payload
    gc.collect()


def qwen3_yarn_gguf_attn_factor(
    rope_params: dict,
    numeric_profile: str,
) -> float | None:
    """Return the GGUF pre-factor before llama.cpp applies YaRN mscale."""

    if numeric_profile != NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1:
        if "attn_factor" in rope_params:
            return float(rope_params["attn_factor"])
        if "attention_factor" in rope_params:
            return float(rope_params["attention_factor"])
        return None

    factor = float(rope_params["factor"])
    default_mscale = 1.0 + 0.1 * math.log(factor) if factor > 1.0 else 1.0
    # transformers 5.2.0 recognizes only attention_factor. In particular,
    # attn_factor is not an alias and the supplied checkpoint therefore uses
    # the default YaRN attention factor.
    training_attention_factor = float(
        rope_params.get("attention_factor", default_mscale)
    )
    pre_factor = training_attention_factor / default_mscale
    if not math.isfinite(pre_factor) or pre_factor <= 0.0:
        raise ValueError(
            "Qwen3 YaRN GGUF attention pre-factor must be positive and finite, "
            f"got {pre_factor}"
        )
    return pre_factor


def add_rope_metadata(
    config: dict,
    writer: gguf.GGUFWriter,
    *,
    numeric_profile: str = NUMERIC_PROFILE_LEGACY_F16_V1,
) -> None:
    rope_params = config.get("rope_parameters")
    if not isinstance(rope_params, dict):
        rope_params = {}

    head_dim = int(config.get("head_dim", int(config["hidden_size"]) // int(config["num_attention_heads"])))
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(float(rope_params.get("rope_theta", config.get("rope_theta", 10000.0))))

    rope_type = rope_params.get("rope_type", rope_params.get("type"))
    if rope_type == "yarn":
        writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
        writer.add_rope_scaling_factor(float(rope_params["factor"]))
        writer.add_rope_scaling_orig_ctx_len(int(rope_params["original_max_position_embeddings"]))
        attn_factor = qwen3_yarn_gguf_attn_factor(rope_params, numeric_profile)
        if attn_factor is not None:
            writer.add_rope_scaling_attn_factors(attn_factor)


def quant_scale_key(weight_key: str) -> str:
    if not weight_key.endswith(".weight"):
        raise ValueError(f"expected weight key ending in .weight, got {weight_key}")
    return weight_key[: -len(".weight")] + ".quant_scale"


def validate_checkpoint(config: dict, weight_map: dict[str, str]) -> None:
    if config.get("model_type") != "qwen3":
        raise ValueError(f"expected model_type=qwen3, got {config.get('model_type')!r}")
    if "Qwen3ForCausalLM" not in (config.get("architectures") or []):
        raise ValueError(f"expected Qwen3ForCausalLM architecture, got {config.get('architectures')!r}")

    n_layer = int(config["num_hidden_layers"])
    required = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    for il in range(n_layer):
        required.extend(
            [
                f"model.layers.{il}.input_layernorm.weight",
                f"model.layers.{il}.post_attention_layernorm.weight",
                f"model.layers.{il}.self_attn.q_norm.weight",
                f"model.layers.{il}.self_attn.k_norm.weight",
            ]
        )
        for hf_suffix, _ in LINEAR_SPECS:
            weight_key = f"model.layers.{il}.{hf_suffix}"
            required.append(weight_key)
            required.append(quant_scale_key(weight_key))

    missing = [key for key in required if key not in weight_map]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise ValueError(f"missing required Qwen3 Fairy2i tensor(s): {preview}{suffix}")


def expected_qwen3_exact_tensor_shapes(
    config: dict,
    *,
    weight_names: set[str] | None = None,
) -> dict[str, tuple[int, ...]]:
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    n_layer = int(config["num_hidden_layers"])
    n_head = int(config["num_attention_heads"])
    n_head_kv = int(config.get("num_key_value_heads", n_head))
    head_dim = int(config.get("head_dim", hidden // n_head))
    vocab = int(config["vocab_size"])
    q_dim = n_head * head_dim
    kv_dim = n_head_kv * head_dim

    expected: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (vocab, hidden),
    }
    linear_shapes = {
        "self_attn.q_proj.weight": (q_dim, hidden),
        "self_attn.k_proj.weight": (kv_dim, hidden),
        "self_attn.v_proj.weight": (kv_dim, hidden),
        "self_attn.o_proj.weight": (hidden, q_dim),
        "mlp.gate_proj.weight": (intermediate, hidden),
        "mlp.up_proj.weight": (intermediate, hidden),
        "mlp.down_proj.weight": (hidden, intermediate),
    }
    for suffix, shape in linear_shapes.items():
        if any(dimension <= 0 or dimension % (2 * FAIRY2I_TILE64) != 0 for dimension in shape):
            raise ValueError(
                "qat_bf16_learned_scale_v1 requires complete complex M64xK64 "
                f"tiles, but {suffix} has real shape {shape}"
            )

    for il in range(n_layer):
        prefix = f"model.layers.{il}"
        expected.update(
            {
                f"{prefix}.input_layernorm.weight": (hidden,),
                f"{prefix}.post_attention_layernorm.weight": (hidden,),
                f"{prefix}.self_attn.q_norm.weight": (head_dim,),
                f"{prefix}.self_attn.k_norm.weight": (head_dim,),
            }
        )
        for suffix, shape in linear_shapes.items():
            weight_name = f"{prefix}.{suffix}"
            expected[weight_name] = shape
            expected[quant_scale_key(weight_name)] = (
                4,
                shape[0] // (2 * FAIRY2I_TILE64),
                shape[1] // (2 * FAIRY2I_TILE64),
            )

        qkv_bias_shapes = {
            f"{prefix}.self_attn.q_proj.bias": (q_dim,),
            f"{prefix}.self_attn.k_proj.bias": (kv_dim,),
            f"{prefix}.self_attn.v_proj.bias": (kv_dim,),
        }
        present_biases = set(qkv_bias_shapes).intersection(weight_names or ())
        if present_biases:
            if present_biases != set(qkv_bias_shapes):
                raise ValueError(
                    f"layer {il} has an incomplete QKV bias set: {sorted(present_biases)}"
                )
            expected.update(qkv_bias_shapes)
    return expected


def validate_qwen3_exact_checkpoint_tensors(
    model_dir: Path,
    config: dict,
    weight_map: dict[str, str],
) -> None:
    """Stream-validate the complete schema4 source contract without widening."""

    expected_shapes = expected_qwen3_exact_tensor_shapes(
        config,
        weight_names=set(weight_map),
    )
    actual_names = set(weight_map)
    expected_names = set(expected_shapes)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            "invalid Qwen3 schema4 checkpoint tensor set: "
            f"missing={missing[:4]}, unexpected={unexpected[:4]}"
        )

    names_by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        names_by_shard.setdefault(shard, []).append(name)

    chunk_elements = 1 << 20
    for shard, names in sorted(names_by_shard.items()):
        path = model_dir / shard
        if not path.is_file():
            raise FileNotFoundError(f"missing safetensors shard for schema4 preflight: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as file:
            shard_names = set(file.keys())
            for name in sorted(names):
                if name not in shard_names:
                    raise ValueError(f"{name} is missing from indexed safetensors shard {shard}")
                tensor_slice = file.get_slice(name)
                actual_shape = tuple(int(value) for value in tensor_slice.get_shape())
                expected_shape = expected_shapes[name]
                if actual_shape != expected_shape:
                    raise ValueError(
                        f"{name} shape mismatch for qat_bf16_learned_scale_v1: "
                        f"expected {expected_shape}, got {actual_shape}"
                    )
                actual_dtype = str(tensor_slice.get_dtype())
                if actual_dtype != "BF16":
                    raise ValueError(
                        f"{name} dtype mismatch for qat_bf16_learned_scale_v1: "
                        f"expected BF16, got {actual_dtype}"
                    )

                row_width = int(np.prod(actual_shape[1:], dtype=np.int64))
                rows_per_chunk = max(1, chunk_elements // max(1, row_width))
                for row_start in range(0, actual_shape[0], rows_per_chunk):
                    row_end = min(actual_shape[0], row_start + rows_per_chunk)
                    chunk = tensor_slice[row_start:row_end]
                    invalid = ~torch.isfinite(chunk)
                    if torch.any(invalid).item():
                        local_index = int(invalid.reshape(-1).nonzero()[0].item())
                        flat_index = row_start * row_width + local_index
                        raise ValueError(
                            f"{name} contains a non-finite BF16 value at flat index "
                            f"{flat_index}"
                        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert Qwen3 Fairy2i W1 learned-scale weights to GGUF")
    parser.add_argument("model_dir", type=Path, help="Path to Qwen3 Fairy2i model directory")
    parser.add_argument("output_file", type=Path, nargs="?", help="Output GGUF file path")
    parser.add_argument("--residual-steps", type=int, default=1, help="Residual quantization steps (only 1 is supported)")
    parser.add_argument(
        "--quant-variant",
        choices=[QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE],
        default=QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE,
        help="Quantization/export variant for Qwen3 learned-scale W1 checkpoints.",
    )
    parser.add_argument(
        "--output-layer",
        choices=["dense"],
        default="dense",
        help="Output projection storage. Qwen3 learned-scale checkpoints currently use dense lm_head output.",
    )
    parser.add_argument(
        "--qk-permute",
        action="store_true",
        help="Unsupported for learned-scale Qwen3; scale tiles are tied to the stored Q/K row order.",
    )
    parser.add_argument(
        "--weight-layout",
        choices=[WEIGHT_LAYOUT_BUNDLE_V1, WEIGHT_LAYOUT_TILE64_V2],
        default=WEIGHT_LAYOUT_BUNDLE_V1,
        help="Fairy2i weight storage layout (default: bundle_v1)",
    )
    parser.add_argument(
        "--numeric-profile",
        choices=[
            NUMERIC_PROFILE_LEGACY_F16_V1,
            NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1,
        ],
        default=NUMERIC_PROFILE_LEGACY_F16_V1,
        help=(
            "Numeric contract for learned-scale W1 export. legacy_f16_v1 preserves "
            "the schema2/F16-scale/F16-output path; qat_bf16_learned_scale_v1 "
            "replays training-side BF16 operation boundaries and writes original "
            "BF16 learned-scale and lm_head payloads."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without writing GGUF")
    parser.add_argument("--verbose", action="store_true", help="Print conversion progress")
    args = parser.parse_args(argv)

    if args.residual_steps != 1:
        raise ValueError("Qwen3 learned-scale Fairy2i conversion supports only --residual-steps 1")
    if args.qk_permute:
        raise ValueError("--qk-permute is not supported for learned-scale Qwen3 Fairy2i checkpoints")
    if (
        args.numeric_profile == NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1
        and args.weight_layout != WEIGHT_LAYOUT_BUNDLE_V1
    ):
        raise ValueError(
            "qat_bf16_learned_scale_v1 requires --weight-layout bundle_v1; "
            "the tile64_v2 carrier stores F16 scales"
        )
    if args.output_file is None and not args.dry_run:
        raise ValueError("output_file is required unless --dry-run is set")

    model_dir: Path = args.model_dir
    output_file: Path | None = args.output_file
    verbose: bool = args.verbose

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    weight_map = load_weight_map(model_dir)
    validate_checkpoint(config, weight_map)
    if args.numeric_profile == NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1:
        validate_qwen3_exact_checkpoint_tensors(model_dir, config, weight_map)

    hidden_real = int(config["hidden_size"])
    hidden_complex = hidden_real // 2
    n_layer = int(config["num_hidden_layers"])
    n_head = int(config["num_attention_heads"])
    n_head_kv = int(config.get("num_key_value_heads", n_head))

    ff_real = int(config["intermediate_size"])
    ff_complex = ff_real // 2
    ff_complex_padded = round_up(ff_complex, FAIRY2I_TILE64)

    if verbose or args.dry_run:
        print(f"hidden_real={hidden_real}, hidden_complex={hidden_complex}")
        print(f"ff_complex={ff_complex}, ff_complex_padded={ff_complex_padded}")
        print(f"n_layer={n_layer}, n_head={n_head}, n_head_kv={n_head_kv}")
        print(f"quant_variant={args.quant_variant}, residual_steps={args.residual_steps}")
        print(f"weight_layout={args.weight_layout}")
        print(f"numeric_profile={args.numeric_profile}")
        print("output_layer=dense")

    if args.dry_run:
        print("Qwen3 Fairy2i W1 learned-scale checkpoint validation passed.")
        return

    assert output_file is not None
    reader = TensorReader(model_dir, weight_map)

    writer = gguf.GGUFWriter(str(output_file), arch="fairy2i")
    if args.weight_layout == WEIGHT_LAYOUT_BUNDLE_V1:
        writer.add_custom_alignment(64)
    writer.add_name(config.get("_name_or_path", "Fairy2i-Qwen3-W1"))
    writer.add_context_length(int(config["max_position_embeddings"]))
    writer.add_embedding_length(hidden_complex)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(ff_complex_padded)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_layer_norm_rms_eps(float(config["rms_norm_eps"]))
    add_rope_metadata(
        config,
        writer,
        numeric_profile=args.numeric_profile,
    )
    writer.add_file_type(
        gguf.LlamaFileType.MOSTLY_FAIRY2I_BUNDLE_V1
        if args.weight_layout == WEIGHT_LAYOUT_BUNDLE_V1
        else gguf.LlamaFileType.MOSTLY_FAIRY2I_TILE64_V2
    )
    writer.add_vocab_size(int(config["vocab_size"]))
    write_metadata(
        writer,
        Fairy2IMetadata(
            base_arch="qwen3",
            base_model_type=config.get("model_type"),
            base_architecture=(config.get("architectures") or [None])[0],
            attn_layout="qwen3_real",
            tokenizer_profile="qwen2",
            quant_variant=args.quant_variant,
            residual_steps=args.residual_steps,
            scale_source=SCALE_SOURCE_LEARNED,
            weight_layout=args.weight_layout,
            numeric_profile=(
                args.numeric_profile
                if args.numeric_profile == NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1
                else None
            ),
            weight_scale_dtype=(
                WEIGHT_SCALE_DTYPE_BF16
                if args.numeric_profile == NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1
                else None
            ),
        ),
    )

    if verbose:
        print("adding token embedding")
    tok_embd = reader.get("model.embed_tokens.weight")
    tok_embd_packed = pack_token_embedding(tok_embd, hidden_complex)
    writer.add_tensor("token_embd", tok_embd_packed, raw_dtype=gguf.GGMLQuantizationType.F32)
    del tok_embd, tok_embd_packed
    gc.collect()

    if verbose:
        print("adding dense output layers")
    add_tensor_f32(writer, "output_norm", reader.get("model.norm.weight"))
    output_w = reader.get("lm_head.weight")
    if args.numeric_profile == NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1:
        add_tensor_bf16(writer, "output", output_w, "lm_head.weight")
    else:
        output_dense = output_w.to(torch.float16).cpu().numpy()
        writer.add_tensor("output", output_dense, raw_dtype=gguf.GGMLQuantizationType.F16)
        del output_dense
    del output_w
    gc.collect()

    for il in range(n_layer):
        if verbose:
            print(f"layer {il + 1}/{n_layer}")

        add_tensor_f32(writer, f"blk.{il}.attn_norm", reader.get(f"model.layers.{il}.input_layernorm.weight"))
        add_tensor_f32(writer, f"blk.{il}.ffn_norm", reader.get(f"model.layers.{il}.post_attention_layernorm.weight"))
        add_tensor_f32(writer, f"blk.{il}.attn_q_norm", reader.get(f"model.layers.{il}.self_attn.q_norm.weight"))
        add_tensor_f32(writer, f"blk.{il}.attn_k_norm", reader.get(f"model.layers.{il}.self_attn.k_norm.weight"))

        linear_specs = LINEAR_SPECS
        if args.weight_layout == WEIGHT_LAYOUT_BUNDLE_V1:
            qkv_bias_keys = [
                f"model.layers.{il}.{suffix.replace('.weight', '.bias')}" for suffix, _ in QKV_LINEAR_SPECS
            ]
            present_qkv_biases = [key for key in qkv_bias_keys if key in weight_map]
            if present_qkv_biases:
                if len(present_qkv_biases) != len(qkv_bias_keys):
                    raise ValueError(f"layer {il} has an incomplete QKV bias set: {present_qkv_biases}")
                qkv_biases = [reader.get(key) for key in qkv_bias_keys]
                if any(bias.ndim != 1 or bias.numel() % 2 != 0 for bias in qkv_biases):
                    raise ValueError(f"layer {il} QKV biases must be even-sized vectors")
                halves = [bias.numel() // 2 for bias in qkv_biases]
                merged_bias = torch.cat(
                    [bias[:half] for bias, half in zip(qkv_biases, halves)]
                    + [bias[half:] for bias, half in zip(qkv_biases, halves)]
                )
                add_tensor_f32(writer, f"blk.{il}.attn_qkv.bias", merged_bias)
                del qkv_biases, merged_bias

            qkv_bundles: list[tuple[np.ndarray, np.ndarray]] = []
            for hf_suffix, _ in QKV_LINEAR_SPECS:
                hf_key = f"model.layers.{il}.{hf_suffix}"
                w = reader.get(hf_key)
                scale = reader.get(quant_scale_key(hf_key))
                out_c = w.shape[0] // 2
                in_c = w.shape[1] // 2
                qkv_bundles.append(
                    quantize_linear_to_fairy2i_bundle_v1_w1_learned_scale(
                        w,
                        scale,
                        out_c,
                        in_c,
                        numeric_profile=args.numeric_profile,
                    )
                )
                del w, scale
                gc.collect()

            qkv_codes, qkv_scales = merge_fairy2i_bundle_v1_m(qkv_bundles)
            writer.add_tensor(
                f"blk.{il}.attn_qkv.bundle.codes",
                qkv_codes,
                raw_dtype=gguf.GGMLQuantizationType.FAIRY2I_BUNDLE_CODES,
            )
            writer.add_tensor(
                f"blk.{il}.attn_qkv.bundle.scales",
                qkv_scales,
                raw_dtype=(
                    gguf.GGMLQuantizationType.BF16
                    if args.numeric_profile
                    == NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1
                    else None
                ),
            )
            del qkv_bundles, qkv_codes, qkv_scales
            linear_specs = LINEAR_SPECS[len(QKV_LINEAR_SPECS) :]

        for hf_suffix, gguf_base in linear_specs:
            hf_key = f"model.layers.{il}.{hf_suffix}"
            w = reader.get(hf_key)
            scale = reader.get(quant_scale_key(hf_key))

            out_c = w.shape[0] // 2
            in_c = w.shape[1] // 2
            out_target = ff_complex_padded if out_c == ff_complex else out_c
            in_target = ff_complex_padded if in_c == ff_complex else in_c

            if args.weight_layout == WEIGHT_LAYOUT_BUNDLE_V1:
                codes, scales = quantize_linear_to_fairy2i_bundle_v1_w1_learned_scale(
                    w,
                    scale,
                    out_target,
                    in_target,
                    numeric_profile=args.numeric_profile,
                )
                writer.add_tensor(
                    f"blk.{il}.{gguf_base}.bundle.codes",
                    codes,
                    raw_dtype=gguf.GGMLQuantizationType.FAIRY2I_BUNDLE_CODES,
                )
                writer.add_tensor(
                    f"blk.{il}.{gguf_base}.bundle.scales",
                    scales,
                    raw_dtype=(
                        gguf.GGMLQuantizationType.BF16
                        if args.numeric_profile
                        == NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1
                        else None
                    ),
                )
                del codes, scales
            else:
                packed = quantize_linear_to_fairy2i_tile64_v2_w1_learned_scale(
                    w,
                    scale,
                    out_target,
                    in_target,
                    numeric_profile=args.numeric_profile,
                )
                for stage_name, stage_data in packed.items():
                    writer.add_tensor(
                        f"blk.{il}.{gguf_base}.{stage_name}",
                        stage_data,
                        raw_dtype=gguf.GGMLQuantizationType.FAIRY2I_TILE64_V2,
                    )
                del packed

            del w, scale
            gc.collect()

        gc.collect()

    set_vocab_qwen2(model_dir, config, writer)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"GGUF saved to: {output_file}")


if __name__ == "__main__":
    main()
