#!/usr/bin/env python3

"""
Convert a Qwen3-based Fairy2i learned-scale W1 Hugging Face checkpoint to GGUF.

Qwen3 Fairy2i W1 checkpoints store one learned 64x64 scale tensor per QAT
linear layer. The converter exports only U.s0/W.s0 tensors and relies on the
runtime's generic wide-linear path for U * conj(x) + W * x.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

import gguf
from convert_fairy2i_qwen2 import (
    TensorReader,
    load_weight_map,
    pack_token_embedding,
    set_vocab_qwen2,
)
from fairy2i.quant.tile64_v2 import (
    FAIRY2I_TILE64,
    quantize_linear_to_fairy2i_bundle_v1_w1_learned_scale,
    quantize_linear_to_fairy2i_tile64_v2_w1_learned_scale,
    round_up,
)
from fairy2i.spec import (
    QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE,
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


def add_tensor_f32(writer: gguf.GGUFWriter, name: str, tensor: torch.Tensor) -> None:
    data = tensor.to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.F32)
    del data
    gc.collect()


def add_rope_metadata(config: dict, writer: gguf.GGUFWriter) -> None:
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
        if "attn_factor" in rope_params:
            writer.add_rope_scaling_attn_factors(float(rope_params["attn_factor"]))
        elif "attention_factor" in rope_params:
            writer.add_rope_scaling_attn_factors(float(rope_params["attention_factor"]))


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
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without writing GGUF")
    parser.add_argument("--verbose", action="store_true", help="Print conversion progress")
    args = parser.parse_args(argv)

    if args.residual_steps != 1:
        raise ValueError("Qwen3 learned-scale Fairy2i conversion supports only --residual-steps 1")
    if args.qk_permute:
        raise ValueError("--qk-permute is not supported for learned-scale Qwen3 Fairy2i checkpoints")
    if args.output_file is None and not args.dry_run:
        raise ValueError("output_file is required unless --dry-run is set")

    model_dir: Path = args.model_dir
    output_file: Path | None = args.output_file
    verbose: bool = args.verbose

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    weight_map = load_weight_map(model_dir)
    validate_checkpoint(config, weight_map)

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
    add_rope_metadata(config, writer)
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
    output_dense = output_w.to(torch.float16).cpu().numpy()
    writer.add_tensor("output", output_dense, raw_dtype=gguf.GGMLQuantizationType.F16)
    del output_w, output_dense
    gc.collect()

    for il in range(n_layer):
        if verbose:
            print(f"layer {il + 1}/{n_layer}")

        add_tensor_f32(writer, f"blk.{il}.attn_norm", reader.get(f"model.layers.{il}.input_layernorm.weight"))
        add_tensor_f32(writer, f"blk.{il}.ffn_norm", reader.get(f"model.layers.{il}.post_attention_layernorm.weight"))
        add_tensor_f32(writer, f"blk.{il}.attn_q_norm", reader.get(f"model.layers.{il}.self_attn.q_norm.weight"))
        add_tensor_f32(writer, f"blk.{il}.attn_k_norm", reader.get(f"model.layers.{il}.self_attn.k_norm.weight"))

        for hf_suffix, gguf_base in LINEAR_SPECS:
            hf_key = f"model.layers.{il}.{hf_suffix}"
            w = reader.get(hf_key)
            scale = reader.get(quant_scale_key(hf_key))

            out_c = w.shape[0] // 2
            in_c = w.shape[1] // 2
            out_target = ff_complex_padded if out_c == ff_complex else out_c
            in_target = ff_complex_padded if in_c == ff_complex else in_c

            if args.weight_layout == WEIGHT_LAYOUT_BUNDLE_V1:
                codes, scales = quantize_linear_to_fairy2i_bundle_v1_w1_learned_scale(
                    w, scale, out_target, in_target
                )
                writer.add_tensor(
                    f"blk.{il}.{gguf_base}.bundle.codes",
                    codes,
                    raw_dtype=gguf.GGMLQuantizationType.FAIRY2I_BUNDLE_CODES,
                )
                writer.add_tensor(f"blk.{il}.{gguf_base}.bundle.scales", scales)
                del codes, scales
            else:
                packed = quantize_linear_to_fairy2i_tile64_v2_w1_learned_scale(w, scale, out_target, in_target)
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
