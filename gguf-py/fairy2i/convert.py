from __future__ import annotations

import argparse
import json
from pathlib import Path

from .arch.registry import detect_arch_info, get_arch_info
from .spec import QUANT_VARIANT_TILE64_V2, QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE
from .spec import WEIGHT_LAYOUT_BUNDLE_V1, WEIGHT_LAYOUT_TILE64_V2


def _dispatch_args(args: argparse.Namespace, base_arch: str) -> list[str]:
    script_args = [str(args.model_dir)]
    if args.output_file is not None:
        script_args.append(str(args.output_file))

    if args.dry_run:
        script_args.append("--dry-run")
    if args.verbose:
        script_args.append("--verbose")
    if args.qk_permute:
        script_args.append("--qk-permute")
    if args.residual_steps is not None:
        script_args.extend(["--residual-steps", str(args.residual_steps)])
    script_args.extend(["--weight-layout", args.weight_layout])

    if base_arch == "qwen2":
        if args.output_file is None:
            raise ValueError("output_file is required for qwen2 conversion")
        script_args.extend(["--quant-variant", args.quant_variant])
    if base_arch == "qwen3":
        if args.output_file is None and not args.dry_run:
            raise ValueError("output_file is required for qwen3 conversion")
        script_args.extend(["--quant-variant", args.quant_variant])

    return script_args


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert Fairy2i Hugging Face weights to GGUF")
    parser.add_argument("model_dir", type=Path, help="Path to the Fairy2i model directory")
    parser.add_argument("output_file", type=Path, nargs="?", help="Output GGUF file path")
    parser.add_argument("--base-arch", choices=["auto", "llama", "qwen2", "qwen3"], default="auto")
    parser.add_argument(
        "--quant-variant",
        choices=[QUANT_VARIANT_TILE64_V2, QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE],
        default=None,
    )
    parser.add_argument("--residual-steps", type=int)
    parser.add_argument(
        "--weight-layout",
        choices=[WEIGHT_LAYOUT_BUNDLE_V1, WEIGHT_LAYOUT_TILE64_V2],
        default=WEIGHT_LAYOUT_BUNDLE_V1,
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without writing GGUF")
    parser.add_argument("--qk-permute", action="store_true", help="Enable Llama q/k undo-permute when supported")
    parser.add_argument("--verbose", action="store_true", help="Print conversion progress")
    args = parser.parse_args(argv)

    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    arch_info = detect_arch_info(config) if args.base_arch == "auto" else get_arch_info(args.base_arch)

    if args.quant_variant is None:
        args.quant_variant = (
            QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE if arch_info.name == "qwen3" else QUANT_VARIANT_TILE64_V2
        )
    if args.residual_steps is None:
        args.residual_steps = 1 if args.quant_variant == QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE else 2

    if arch_info.name == "llama":
        if args.quant_variant != QUANT_VARIANT_TILE64_V2:
            raise ValueError("Llama Fairy2i conversion currently supports only --quant-variant tile64_v2")
        from convert_fairy2i_llama import main as llama_main

        llama_main(_dispatch_args(args, arch_info.name))
        return

    if arch_info.name == "qwen2":
        from convert_fairy2i_qwen2 import main as qwen2_main

        qwen2_main(_dispatch_args(args, arch_info.name))
        return

    if arch_info.name == "qwen3":
        if args.quant_variant != QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE:
            raise ValueError("Qwen3 Fairy2i conversion requires --quant-variant tile64_v2_w1_learned_scale")
        from convert_fairy2i_qwen3 import main as qwen3_main

        qwen3_main(_dispatch_args(args, arch_info.name))
        return

    raise ValueError(f"unsupported Fairy2i base architecture: {arch_info.name}")


if __name__ == "__main__":
    main()
