#!/usr/bin/env python3

"""
Convert a Qwen2-based Fairy2i Hugging Face checkpoint to GGUF.

This script is intentionally separate from convert_fairy2i.py:
- tokenizer export follows the Qwen2/GPT-2-style path used by convert_hf_to_gguf.py
- RoPE base is read from config["rope_parameters"]["rope_theta"] when present
- optional attention biases are exported so the GGUF can carry q/k/v bias tensors
- Qwen2-based Fairy2i 32B weights are exported with a tile64_v2 layout that matches
  the training-side QAT kernel semantics
"""

import argparse
import gc
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from safetensors import safe_open

import gguf
from fairy2i.quant.tile64_v2 import (
    FAIRY2I_TILE64,
    quantize_linear_to_fairy2i_bundle_v1_stages,
    quantize_linear_to_fairy2i_tile64_v2_stages,
    round_up,
)
from fairy2i.spec import (
    WEIGHT_LAYOUT_BUNDLE_V1,
    WEIGHT_LAYOUT_TILE64_V2,
    Fairy2IMetadata,
    write_metadata,
)


QWEN2_PRETOKENIZER_HASHES = {
    # ref: convert_hf_to_gguf.py get_vocab_base_pre()
    "d4540891389ea895b53b399da6ac824becc30f2fba0e9ddbb98f92e55ca0e97c",
    "e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea",
}


def token_looks_special(token: str | bytes) -> bool:
    if isinstance(token, bytes):
        token_text = token.decode("utf-8")
    else:
        token_text = token

    seems_special = token_text in (
        "<pad>",
        "<mask>",
        "<2mass>",
        "[@BOS@]",
    )
    seems_special = seems_special or (token_text.startswith("<|") and token_text.endswith("|>"))
    seems_special = seems_special or (token_text.startswith("<｜") and token_text.endswith("｜>"))
    seems_special = seems_special or (token_text.startswith("<unused") and token_text.endswith(">"))
    return seems_special


def get_qwen2_tokenizer_pre(model_dir: Path) -> str:
    chktxt = (
        "\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n"
        "🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 "
        "3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= "
        "нещо на Български ''''''```````\"\"\"\"......!!!!!!?????? I've been 'told he's there, "
        "'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL"
    )

    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        chktok = tokenizer.encode(chktxt).ids
        chkhsh = sha256(str(chktok).encode()).hexdigest()
        if chkhsh in QWEN2_PRETOKENIZER_HASHES:
            return "qwen2"

        print(
            f"warning: unrecognized Qwen2 tokenizer pre hash {chkhsh}, falling back to tokenizer.ggml.pre=qwen2",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"warning: failed to evaluate Qwen2 tokenizer pre-tokenizer via tokenizers ({exc}), "
            "falling back to tokenizer.ggml.pre=qwen2",
            file=sys.stderr,
        )

    return "qwen2"


def set_vocab_qwen2(model_dir: Path, config: dict, writer: gguf.GGUFWriter) -> None:
    tokenizer_json_file = model_dir / "tokenizer.json"
    if not tokenizer_json_file.is_file():
        raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")

    tokenizer_json = json.loads(tokenizer_json_file.read_text(encoding="utf-8"))
    vocab_size = int(config["vocab_size"])
    vocab = tokenizer_json.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError(f"invalid vocab in {tokenizer_json_file}")
    assert max(vocab.values()) < vocab_size

    tokpre = get_qwen2_tokenizer_pre(model_dir)
    reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in vocab.items()}
    added_tokens = tokenizer_json.get("added_tokens", [])

    added_vocab: dict[str, int] = {}
    added_tokens_decoder: dict[int, dict] = {}
    if isinstance(added_tokens, list):
        for item in added_tokens:
            if not isinstance(item, dict):
                continue
            token = item.get("content")
            token_id = item.get("id")
            if not isinstance(token, str) or not isinstance(token_id, int):
                continue
            added_vocab[token] = token_id
            added_tokens_decoder[token_id] = item
            reverse_vocab[token_id] = token

    tokens: list[str] = []
    toktypes: list[int] = []

    for i in range(vocab_size):
        if i not in reverse_vocab:
            tokens.append(f"[PAD{i}]")
            toktypes.append(gguf.TokenType.UNUSED)
            continue

        token = reverse_vocab[i]
        if token in added_vocab:
            decoder_entry = added_tokens_decoder.get(i)
            is_special = bool((decoder_entry or {}).get("special", False)) or token_looks_special(token)
            toktypes.append(gguf.TokenType.CONTROL if is_special else gguf.TokenType.USER_DEFINED)
        else:
            toktypes.append(gguf.TokenType.NORMAL)

        tokens.append(token)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre(tokpre)
    writer.add_token_list(tokens)
    writer.add_token_types(toktypes)

    special_vocab = gguf.SpecialVocab(model_dir, load_merges=True)
    special_vocab.add_to_gguf(writer)

    tokenizer_config_file = model_dir / "tokenizer_config.json"
    if tokenizer_config_file.is_file():
        tokenizer_config = json.loads(tokenizer_config_file.read_text(encoding="utf-8"))
        if "add_prefix_space" in tokenizer_config:
            writer.add_add_space_prefix(tokenizer_config["add_prefix_space"])


def load_weight_map(model_dir: Path) -> Dict[str, str]:
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.is_file():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"invalid weight_map in {index_file}")
        return {k: v for k, v in weight_map.items()}

    model_files = sorted(model_dir.glob("*.safetensors"))
    if len(model_files) != 1:
        raise ValueError("no shard index and cannot infer a single safetensors file")

    filename = model_files[0].name
    with safe_open(str(model_files[0]), framework="pt", device="cpu") as f:
        return {key: filename for key in f.keys()}


class TensorReader:
    def __init__(self, model_dir: Path, weight_map: Dict[str, str]):
        self.model_dir = model_dir
        self.weight_map = weight_map

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def get(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"missing tensor key: {key}")
        filename = self.weight_map[key]
        path = self.model_dir / filename
        with safe_open(str(path), framework="pt", device="cpu") as f:
            return f.get_tensor(key)


def undo_llama_permute(weight: torch.Tensor, n_head: int) -> torch.Tensor:
    return (
        weight.reshape(n_head, 2, weight.shape[0] // n_head // 2, *weight.shape[1:])
        .swapaxes(1, 2)
        .reshape(weight.shape)
    )


def pack_token_embedding(embed: torch.Tensor, hidden_complex: int) -> np.ndarray:
    real = embed[:, :hidden_complex].to(torch.float32)
    imag = embed[:, hidden_complex:].to(torch.float32)

    real_bits = real.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32)
    imag_bits = imag.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32)

    packed = ((imag_bits << 16) | (real_bits & 0xFFFF)).to(torch.int32).view(torch.float32)
    return packed.cpu().numpy()


def add_optional_vector_tensor(
    writer: gguf.GGUFWriter,
    reader: TensorReader,
    hf_key: str,
    gguf_name: str,
) -> None:
    if not reader.has(hf_key):
        return

    tensor = reader.get(hf_key).to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    writer.add_tensor(gguf_name, tensor, raw_dtype=gguf.GGMLQuantizationType.F32)
    del tensor
    gc.collect()


def get_rope_theta(config: dict) -> float:
    rope_params = config.get("rope_parameters")
    if isinstance(rope_params, dict) and "rope_theta" in rope_params:
        return float(rope_params["rope_theta"])
    if "rope_theta" in config:
        return float(config["rope_theta"])
    return 10000.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert Qwen2-based Fairy2i Hugging Face weights to GGUF")
    parser.add_argument("model_dir", type=Path, help="Path to Qwen2-based Fairy2i model directory")
    parser.add_argument("output_file", type=Path, help="Output GGUF file path")
    parser.add_argument("--residual-steps", type=int, default=2, help="Residual quantization steps (only 2 is supported)")
    parser.add_argument(
        "--output-layer",
        choices=["wide-linear", "dense", "both"],
        default="wide-linear",
        help="Output projection storage: wide-linear (default), dense, or both (for A/B debugging)",
    )
    parser.add_argument(
        "--qk-permute",
        action="store_true",
        help="Enable Llama q/k undo-permute during conversion (disabled by default for Fairy2i)",
    )
    parser.add_argument(
        "--no-attn-bias",
        action="store_true",
        help="Do not export optional attention bias tensors even if present in the HF checkpoint",
    )
    parser.add_argument(
        "--quant-variant",
        choices=["tile64_v2"],
        default="tile64_v2",
        help="Quantization/export variant. tile64_v2 matches the training-side QAT kernel.",
    )
    parser.add_argument(
        "--weight-layout",
        choices=[WEIGHT_LAYOUT_BUNDLE_V1, WEIGHT_LAYOUT_TILE64_V2],
        default=WEIGHT_LAYOUT_BUNDLE_V1,
        help="Fairy2i weight storage layout (default: bundle_v1)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print conversion progress")
    args = parser.parse_args(argv)

    if args.residual_steps != 2:
        raise ValueError("only --residual-steps 2 is currently supported")

    model_dir: Path = args.model_dir
    output_file: Path = args.output_file
    verbose: bool = args.verbose
    output_layer_mode: str = args.output_layer
    do_qk_permute: bool = args.qk_permute
    export_attn_bias: bool = not args.no_attn_bias
    quant_variant: str = args.quant_variant

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    hidden_real = int(config["hidden_size"])
    hidden_complex = hidden_real // 2
    n_layer = int(config["num_hidden_layers"])
    n_head = int(config["num_attention_heads"])
    n_head_kv = int(config.get("num_key_value_heads", n_head))
    rope_theta = get_rope_theta(config)

    ff_real = int(config["intermediate_size"])
    ff_complex = ff_real // 2
    ff_complex_padded = round_up(ff_complex, FAIRY2I_TILE64)

    if verbose:
        print(f"hidden_real={hidden_real}, hidden_complex={hidden_complex}")
        print(f"ff_complex={ff_complex}, ff_complex_padded={ff_complex_padded}")
        print(f"rope_theta={rope_theta}")
        print(f"output_layer_mode={output_layer_mode}, do_qk_permute={do_qk_permute}")
        print(f"export_attn_bias={export_attn_bias}")
        print(f"quant_variant={quant_variant}")

    weight_map = load_weight_map(model_dir)
    reader = TensorReader(model_dir, weight_map)

    writer = gguf.GGUFWriter(str(output_file), arch="fairy2i")
    if args.weight_layout == WEIGHT_LAYOUT_BUNDLE_V1:
        writer.add_custom_alignment(64)
    writer.add_name(config.get("_name_or_path", "Fairy2i-Qwen2"))
    writer.add_context_length(int(config["max_position_embeddings"]))
    writer.add_embedding_length(hidden_complex)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(ff_complex_padded)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_layer_norm_rms_eps(float(config["rms_norm_eps"]))
    writer.add_rope_freq_base(rope_theta)
    writer.add_file_type(
        gguf.LlamaFileType.MOSTLY_FAIRY2I_BUNDLE_V1
        if args.weight_layout == WEIGHT_LAYOUT_BUNDLE_V1
        else gguf.LlamaFileType.MOSTLY_FAIRY2I_TILE64_V2
    )
    writer.add_vocab_size(int(config["vocab_size"]))
    write_metadata(
        writer,
        Fairy2IMetadata(
            base_arch="qwen2",
            base_model_type=config.get("model_type"),
            base_architecture=(config.get("architectures") or [None])[0],
            attn_layout="qwen2_real",
            tokenizer_profile="qwen2",
            quant_variant=quant_variant,
            residual_steps=args.residual_steps,
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
        print("adding output layers")
    output_norm = reader.get("model.norm.weight").to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    writer.add_tensor("output_norm", output_norm, raw_dtype=gguf.GGMLQuantizationType.F32)
    del output_norm
    gc.collect()

    output_w = reader.get("lm_head.weight")
    if output_layer_mode in ("wide-linear", "both"):
        if verbose:
            print("adding output projection (wide-linear fairy2i)")
        output_out_c = output_w.shape[0] // 2
        output_in_c = output_w.shape[1] // 2
        if args.weight_layout == WEIGHT_LAYOUT_BUNDLE_V1:
            output_codes, output_scales = quantize_linear_to_fairy2i_bundle_v1_stages(
                output_w, output_out_c, output_in_c
            )
            writer.add_tensor(
                "output.bundle.codes",
                output_codes,
                raw_dtype=gguf.GGMLQuantizationType.FAIRY2I_BUNDLE_CODES,
            )
            writer.add_tensor("output.bundle.scales", output_scales)
            del output_codes, output_scales
        else:
            output_packed = quantize_linear_to_fairy2i_tile64_v2_stages(output_w, output_out_c, output_in_c)
            for stage_name, stage_data in output_packed.items():
                writer.add_tensor(
                    f"output.{stage_name}",
                    stage_data,
                    raw_dtype=gguf.GGMLQuantizationType.FAIRY2I_TILE64_V2,
                )
            del output_packed

    if output_layer_mode in ("dense", "both"):
        if verbose:
            print("adding output projection (dense f16)")
        output_dense = output_w.to(torch.float16).cpu().numpy()
        writer.add_tensor("output", output_dense, raw_dtype=gguf.GGMLQuantizationType.F16)
        del output_dense

    del output_w
    gc.collect()

    linear_specs = [
        ("self_attn.q_proj.weight", "attn_q", "q"),
        ("self_attn.k_proj.weight", "attn_k", "k"),
        ("self_attn.v_proj.weight", "attn_v", None),
        ("self_attn.o_proj.weight", "attn_output", None),
        ("mlp.gate_proj.weight", "ffn_gate", None),
        ("mlp.up_proj.weight", "ffn_up", None),
        ("mlp.down_proj.weight", "ffn_down", None),
    ]

    bias_specs = [
        ("self_attn.q_proj.bias", "attn_q.bias"),
        ("self_attn.k_proj.bias", "attn_k.bias"),
        ("self_attn.v_proj.bias", "attn_v.bias"),
        ("self_attn.o_proj.bias", "attn_output.bias"),
    ]

    for il in range(n_layer):
        if verbose:
            print(f"layer {il + 1}/{n_layer}")

        attn_norm = reader.get(f"model.layers.{il}.input_layernorm.weight").to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
        ffn_norm = reader.get(f"model.layers.{il}.post_attention_layernorm.weight").to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
        writer.add_tensor(f"blk.{il}.attn_norm", attn_norm, raw_dtype=gguf.GGMLQuantizationType.F32)
        writer.add_tensor(f"blk.{il}.ffn_norm", ffn_norm, raw_dtype=gguf.GGMLQuantizationType.F32)
        del attn_norm, ffn_norm

        for hf_suffix, gguf_base, permute_kind in linear_specs:
            hf_key = f"model.layers.{il}.{hf_suffix}"
            w = reader.get(hf_key)

            if permute_kind == "q" and do_qk_permute:
                w = undo_llama_permute(w, n_head)
            elif permute_kind == "k" and do_qk_permute:
                w = undo_llama_permute(w, n_head_kv)

            out_c = w.shape[0] // 2
            in_c = w.shape[1] // 2
            out_target = ff_complex_padded if out_c == ff_complex else out_c
            in_target = ff_complex_padded if in_c == ff_complex else in_c

            if args.weight_layout == WEIGHT_LAYOUT_BUNDLE_V1:
                codes, scales = quantize_linear_to_fairy2i_bundle_v1_stages(w, out_target, in_target)
                writer.add_tensor(
                    f"blk.{il}.{gguf_base}.bundle.codes",
                    codes,
                    raw_dtype=gguf.GGMLQuantizationType.FAIRY2I_BUNDLE_CODES,
                )
                writer.add_tensor(f"blk.{il}.{gguf_base}.bundle.scales", scales)
                del codes, scales
            else:
                packed = quantize_linear_to_fairy2i_tile64_v2_stages(w, out_target, in_target)
                for stage_name, stage_data in packed.items():
                    writer.add_tensor(
                        f"blk.{il}.{gguf_base}.{stage_name}",
                        stage_data,
                        raw_dtype=gguf.GGMLQuantizationType.FAIRY2I_TILE64_V2,
                    )
                del packed

            del w
            gc.collect()

        if export_attn_bias:
            for hf_suffix, gguf_name in bias_specs:
                add_optional_vector_tensor(
                    writer,
                    reader,
                    f"model.layers.{il}.{hf_suffix}",
                    f"blk.{il}.{gguf_name}",
                )

        gc.collect()

    set_vocab_qwen2(model_dir, config, writer)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"GGUF saved to: {output_file}")


if __name__ == "__main__":
    main()
