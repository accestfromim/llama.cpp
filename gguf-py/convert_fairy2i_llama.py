#!/usr/bin/env python3

"""
Convert a Llama-based Fairy2i Hugging Face checkpoint to GGUF.

This converter is for checkpoints whose dense BF16 Linear weights are trained
with the QATLinearComplexPhaseV2 tile64 quantization kernel. It uses the shared
Fairy2i tile64_v2 packer, exports a Llama BPE tokenizer, and pads the
vocabulary so the optional Fairy2i output tensor shape checks stay 64-aligned
while token ids remain stable.
"""

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import gguf
import numpy as np
import torch

from fairy2i.io.tensor_reader import TensorReader, add_optional_vector_tensor, load_weight_map
from fairy2i.quant.tile64_v2 import (
    QK_IFAIRY,
    TILE64,
    quantize_linear_to_ifairy64_stages,
    round_up,
)
from fairy2i.quant.widely_linear import undo_llama_permute
from fairy2i.spec import Fairy2IMetadata, write_metadata
from fairy2i.tokenizer.chat_template import load_fairy2i_chat_template


FAIRY2I_VOCAB_PADDING_MULTIPLE = 2 * TILE64


def round_up_even_tile_vocab(vocab_size: int) -> int:
    return round_up(vocab_size, FAIRY2I_VOCAB_PADDING_MULTIPLE)


def token_looks_special(token: str | bytes) -> bool:
    if isinstance(token, bytes):
        token_text = token.decode("utf-8")
    else:
        token_text = token

    return (
        token_text in ("<unk>", "<s>", "</s>", "<pad>")
        or (token_text.startswith("<|") and token_text.endswith("|>"))
        or (token_text.startswith("<｜") and token_text.endswith("｜>"))
        or (token_text.startswith("<") and token_text.endswith(">"))
    )


def llama_token_type(token_id: int, token_text: str, special_ids: set[int]) -> gguf.TokenType:
    if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", token_text):
        return gguf.TokenType.BYTE
    return gguf.TokenType.CONTROL if token_id in special_ids else gguf.TokenType.NORMAL


def token_config_text(tokenizer_config: dict, token_name: str) -> str | None:
    token = tokenizer_config.get(f"{token_name}_token")
    if isinstance(token, str):
        return token
    if isinstance(token, dict):
        content = token.get("content")
        if isinstance(content, str):
            return content
    return None


def token_id_for_text(reverse_vocab: dict[int, str], token_text: str) -> int | None:
    for token_id, vocab_text in reverse_vocab.items():
        if vocab_text == token_text:
            return token_id
    return None


def resolve_special_token_id(
    token_name: str,
    config: dict,
    tokenizer_config: dict,
    reverse_vocab: dict[int, str],
    padded_vocab_size: int,
) -> int:
    config_token_id = config.get(f"{token_name}_token_id")
    token_text = token_config_text(tokenizer_config, token_name)

    token_id = None
    if isinstance(config_token_id, int) and config_token_id >= 0:
        if token_text is None or reverse_vocab.get(config_token_id) == token_text:
            token_id = config_token_id

    if token_id is None and token_text is not None:
        token_id = token_id_for_text(reverse_vocab, token_text)

    if token_id is None and isinstance(config_token_id, int) and config_token_id >= 0:
        token_id = config_token_id

    if token_id is None:
        raise ValueError(f"could not resolve {token_name}_token_id")
    if token_id >= padded_vocab_size:
        raise ValueError(f"{token_name}_token_id={token_id} is outside padded vocab {padded_vocab_size}")
    return token_id


def add_fairy2i_llama_tokenizer_metadata(
    model_dir: Path,
    config: dict,
    tokenizer_config: dict,
    writer: gguf.GGUFWriter,
    reverse_vocab: dict[int, str],
    padded_vocab_size: int,
) -> None:
    writer.add_bos_token_id(
        resolve_special_token_id("bos", config, tokenizer_config, reverse_vocab, padded_vocab_size)
    )
    writer.add_eos_token_id(
        resolve_special_token_id("eos", config, tokenizer_config, reverse_vocab, padded_vocab_size)
    )
    writer.add_unk_token_id(
        resolve_special_token_id("unk", config, tokenizer_config, reverse_vocab, padded_vocab_size)
    )
    writer.add_pad_token_id(
        resolve_special_token_id("pad", config, tokenizer_config, reverse_vocab, padded_vocab_size)
    )

    writer.add_add_bos_token(False)
    writer.add_add_sep_token(False)

    chat_template = load_fairy2i_chat_template(model_dir, tokenizer_config)
    if chat_template is not None:
        writer.add_chat_template(chat_template)


def set_vocab_llama_bpe_padded(
    model_dir: Path,
    config: dict,
    writer: gguf.GGUFWriter,
    padded_vocab_size: int,
) -> None:
    tokenizer_file = model_dir / "tokenizer.json"
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")

    tokenizer_json = json.loads(tokenizer_file.read_text(encoding="utf-8"))
    tokenizer_model = tokenizer_json.get("model", {})
    if tokenizer_model.get("type") != "BPE" or not tokenizer_model.get("byte_fallback", False):
        raise ValueError("expected a Llama-style BPE tokenizer with byte_fallback=true")

    original_vocab_size = int(config["vocab_size"])
    if padded_vocab_size < original_vocab_size:
        raise ValueError(f"padded vocab {padded_vocab_size} is smaller than original vocab {original_vocab_size}")

    vocab = tokenizer_model.get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError(f"invalid tokenizer model vocab in {tokenizer_file}")

    reverse_vocab: dict[int, str] = {int(token_id): token for token, token_id in vocab.items()}

    added_tokens = tokenizer_json.get("added_tokens", [])
    added_token_ids: set[int] = set()
    added_token_special: dict[int, bool] = {}
    if isinstance(added_tokens, list):
        for item in added_tokens:
            if not isinstance(item, dict):
                continue
            token = item.get("content")
            token_id = item.get("id")
            if not isinstance(token, str) or not isinstance(token_id, int):
                continue
            reverse_vocab[token_id] = token
            added_token_ids.add(token_id)
            added_token_special[token_id] = bool(item.get("special", False))

    tokenizer_config_file = model_dir / "tokenizer_config.json"
    tokenizer_config = {}
    if tokenizer_config_file.is_file():
        tokenizer_config = json.loads(tokenizer_config_file.read_text(encoding="utf-8"))

    special_ids = {token_id for token_id, is_special in added_token_special.items() if is_special}
    for token_name in ("bos_token", "eos_token", "unk_token", "pad_token"):
        token = tokenizer_config.get(token_name)
        if isinstance(token, str):
            for token_id, token_text in reverse_vocab.items():
                if token_text == token:
                    special_ids.add(token_id)

    tokens: list[bytes] = []
    scores: list[float] = []
    toktypes: list[int] = []

    for token_id in range(original_vocab_size):
        token = reverse_vocab.get(token_id)
        if token is None:
            tokens.append(f"[PAD{token_id}]".encode("utf-8"))
            scores.append(-1000.0)
            toktypes.append(gguf.TokenType.UNUSED)
            continue

        tokens.append(token.encode("utf-8"))
        scores.append(-1000.0)

        if token_id in added_token_ids and token_id >= len(vocab):
            is_special = token_id in special_ids or token_looks_special(token)
            toktypes.append(gguf.TokenType.CONTROL if is_special else gguf.TokenType.USER_DEFINED)
        else:
            toktypes.append(llama_token_type(token_id, token, special_ids))

    for token_id in range(original_vocab_size, padded_vocab_size):
        tokens.append(f"[PAD_FAIRY2I_{token_id}]".encode("utf-8"))
        scores.append(-1000.0)
        toktypes.append(gguf.TokenType.UNUSED)

    writer.add_tokenizer_model("llama")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)
    add_fairy2i_llama_tokenizer_metadata(
        model_dir,
        config,
        tokenizer_config,
        writer,
        reverse_vocab,
        padded_vocab_size,
    )

    if "add_prefix_space" in tokenizer_config:
        writer.add_add_space_prefix(tokenizer_config["add_prefix_space"])


def pack_token_embedding_padded(embed: torch.Tensor, hidden_complex: int, padded_vocab_size: int) -> np.ndarray:
    if embed.shape[1] != hidden_complex * 2:
        raise ValueError(f"token embedding shape {tuple(embed.shape)} does not match hidden_complex={hidden_complex}")

    real = embed[:, :hidden_complex].to(torch.float32)
    imag = embed[:, hidden_complex:].to(torch.float32)

    real_bits = real.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32)
    imag_bits = imag.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32)
    packed = ((imag_bits << 16) | (real_bits & 0xFFFF)).to(torch.int32).view(torch.float32).cpu().numpy()

    if packed.shape[0] == padded_vocab_size:
        return packed
    if packed.shape[0] > padded_vocab_size:
        raise ValueError(f"cannot pad token embedding rows from {packed.shape[0]} down to {padded_vocab_size}")

    out = np.zeros((padded_vocab_size, packed.shape[1]), dtype=np.float32)
    out[: packed.shape[0], :] = packed
    return out


def dense_output_padded(output_w: torch.Tensor, padded_vocab_size: int) -> np.ndarray:
    output = output_w.to(torch.float16).cpu().numpy()
    if output.shape[0] == padded_vocab_size:
        return output
    if output.shape[0] > padded_vocab_size:
        raise ValueError(f"cannot pad output rows from {output.shape[0]} down to {padded_vocab_size}")

    out = np.zeros((padded_vocab_size, output.shape[1]), dtype=np.float16)
    out[: output.shape[0], :] = output
    return out


def add_rope_metadata(config: dict, writer: gguf.GGUFWriter) -> None:
    rope_params = config.get("rope_parameters")
    if not isinstance(rope_params, dict):
        rope_params = {}

    writer.add_rope_freq_base(float(rope_params.get("rope_theta", config.get("rope_theta", 10000.0))))

    rope_type = rope_params.get("rope_type", rope_params.get("type"))
    if rope_type == "yarn":
        writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
        writer.add_rope_scaling_factor(float(rope_params["factor"]))
        writer.add_rope_scaling_orig_ctx_len(int(rope_params["original_max_position_embeddings"]))
        writer.add_rope_scaling_yarn_beta_fast(float(rope_params.get("beta_fast", 32.0)))
        writer.add_rope_scaling_yarn_beta_slow(float(rope_params.get("beta_slow", 1.0)))
        if "attention_factor" in rope_params:
            writer.add_rope_scaling_attn_factors(float(rope_params["attention_factor"]))


def validate_checkpoint(config: dict, weight_map: dict[str, str]) -> None:
    if config.get("model_type") != "llama":
        raise ValueError(f"expected model_type=llama, got {config.get('model_type')!r}")
    if config.get("architectures") != ["LlamaForCausalLM"]:
        raise ValueError(f"expected LlamaForCausalLM architecture, got {config.get('architectures')!r}")

    required = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    n_layer = int(config["num_hidden_layers"])
    for il in range(n_layer):
        required.extend(
            [
                f"model.layers.{il}.input_layernorm.weight",
                f"model.layers.{il}.post_attention_layernorm.weight",
                f"model.layers.{il}.self_attn.q_proj.weight",
                f"model.layers.{il}.self_attn.k_proj.weight",
                f"model.layers.{il}.self_attn.v_proj.weight",
                f"model.layers.{il}.self_attn.o_proj.weight",
                f"model.layers.{il}.mlp.gate_proj.weight",
                f"model.layers.{il}.mlp.up_proj.weight",
                f"model.layers.{il}.mlp.down_proj.weight",
            ]
        )

    missing = [key for key in required if key not in weight_map]
    if missing:
        preview = ", ".join(missing[:8])
        raise KeyError(f"missing {len(missing)} required tensors: {preview}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert Llama-based Fairy2i HF weights to GGUF")
    parser.add_argument("model_dir", type=Path, help="Path to the Llama-based Fairy2i model directory")
    parser.add_argument("output_file", type=Path, nargs="?", help="Output GGUF file path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print conversion plan without writing GGUF",
    )
    parser.add_argument(
        "--residual-steps",
        type=int,
        default=2,
        help="Residual quantization steps (only 2 is supported)",
    )
    parser.add_argument("--qk-permute", action="store_true", help="Enable Llama q/k undo-permute during conversion")
    parser.add_argument("--verbose", action="store_true", help="Print conversion progress")
    args = parser.parse_args(argv)

    if args.residual_steps != 2:
        raise ValueError("only --residual-steps 2 is currently supported")
    if not args.dry_run and args.output_file is None:
        raise ValueError("output_file is required unless --dry-run is set")

    model_dir: Path = args.model_dir
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    weight_map = load_weight_map(model_dir)
    validate_checkpoint(config, weight_map)

    hidden_real = int(config["hidden_size"])
    if hidden_real % 2 != 0:
        raise ValueError(f"hidden_size must be even, got {hidden_real}")
    hidden_complex = hidden_real // 2
    n_layer = int(config["num_hidden_layers"])
    n_head = int(config["num_attention_heads"])
    n_head_kv = int(config.get("num_key_value_heads", n_head))
    vocab_original = int(config["vocab_size"])
    vocab_padded = round_up_even_tile_vocab(vocab_original)
    vocab_pad = vocab_padded - vocab_original

    ff_real = int(config["intermediate_size"])
    if ff_real % 2 != 0:
        raise ValueError(f"intermediate_size must be even, got {ff_real}")
    ff_complex = ff_real // 2
    ff_complex_padded = round_up(ff_complex, QK_IFAIRY)

    print(
        "Fairy2i Llama conversion: "
        f"layers={n_layer} hidden_real={hidden_real} hidden_complex={hidden_complex} "
        f"ff_complex={ff_complex} ff_complex_padded={ff_complex_padded} "
        f"vocab_original={vocab_original} vocab_padded={vocab_padded} padded_tokens={vocab_pad}"
    )

    if args.dry_run:
        return

    reader = TensorReader(model_dir, weight_map)
    output_file = args.output_file
    assert output_file is not None

    writer = gguf.GGUFWriter(str(output_file), arch="fairy2i")
    writer.add_name(config.get("_name_or_path", model_dir.name))
    writer.add_context_length(int(config["max_position_embeddings"]))
    writer.add_embedding_length(hidden_complex)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(ff_complex_padded)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_layer_norm_rms_eps(float(config["rms_norm_eps"]))
    add_rope_metadata(config, writer)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_IFAIRY)
    writer.add_vocab_size(vocab_padded)
    write_metadata(
        writer,
        Fairy2IMetadata(
            base_arch="llama",
            base_model_type=config.get("model_type"),
            base_architecture=(config.get("architectures") or [None])[0],
            attn_layout="llama_real",
            tokenizer_profile="llama_bpe",
            residual_steps=args.residual_steps,
            vocab_original_size=vocab_original,
            vocab_padded_size=vocab_padded,
            vocab_padding_multiple=FAIRY2I_VOCAB_PADDING_MULTIPLE,
        ),
    )

    if args.verbose:
        print("adding token embedding")
    tok_embd = reader.get("model.embed_tokens.weight")
    tok_embd_packed = pack_token_embedding_padded(tok_embd, hidden_complex, vocab_padded)
    writer.add_tensor("token_embd", tok_embd_packed, raw_dtype=gguf.GGMLQuantizationType.F32)
    del tok_embd, tok_embd_packed
    gc.collect()

    if args.verbose:
        print("adding output norm and dense padded output")
    output_norm = reader.get("model.norm.weight").to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    writer.add_tensor("output_norm", output_norm, raw_dtype=gguf.GGMLQuantizationType.F32)
    del output_norm
    gc.collect()

    output_w = reader.get("lm_head.weight")
    output_dense = dense_output_padded(output_w, vocab_padded)
    writer.add_tensor("output", output_dense, raw_dtype=gguf.GGMLQuantizationType.F16)
    del output_w, output_dense
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
        if args.verbose:
            print(f"layer {il + 1}/{n_layer}")

        attn_norm = (
            reader.get(f"model.layers.{il}.input_layernorm.weight")
            .to(torch.float32)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        ffn_norm = (
            reader.get(f"model.layers.{il}.post_attention_layernorm.weight")
            .to(torch.float32)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        writer.add_tensor(f"blk.{il}.attn_norm", attn_norm, raw_dtype=gguf.GGMLQuantizationType.F32)
        writer.add_tensor(f"blk.{il}.ffn_norm", ffn_norm, raw_dtype=gguf.GGMLQuantizationType.F32)
        del attn_norm, ffn_norm

        for hf_suffix, gguf_base, permute_kind in linear_specs:
            hf_key = f"model.layers.{il}.{hf_suffix}"
            w = reader.get(hf_key)

            if permute_kind == "q" and args.qk_permute:
                w = undo_llama_permute(w, n_head)
            elif permute_kind == "k" and args.qk_permute:
                w = undo_llama_permute(w, n_head_kv)

            out_c = w.shape[0] // 2
            in_c = w.shape[1] // 2
            out_target = ff_complex_padded if out_c == ff_complex else out_c
            in_target = ff_complex_padded if in_c == ff_complex else in_c

            packed = quantize_linear_to_ifairy64_stages(w, out_target, in_target)
            for stage_name, stage_data in packed.items():
                writer.add_tensor(
                    f"blk.{il}.{gguf_base}.{stage_name}",
                    stage_data,
                    raw_dtype=gguf.GGMLQuantizationType.IFAIRY64,
                )

            del w, packed
            gc.collect()

        for hf_suffix, gguf_name in bias_specs:
            add_optional_vector_tensor(writer, reader, f"model.layers.{il}.{hf_suffix}", f"blk.{il}.{gguf_name}")

        gc.collect()

    set_vocab_llama_bpe_padded(model_dir, config, writer, vocab_padded)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"GGUF saved to: {output_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
