#!/usr/bin/env python3
"""
tools/merge_lora_safetensors.py

Merges a LoRA adapter into a base model by processing one safetensors shard
at a time (peak memory ≈ one shard + adapter, never the full model).

Writes the merged model to an output directory, then creates an Ollama
Modelfile so you can run:

    ollama create my-model -f <output_dir>/Modelfile

Ollama quantizes during 'create' (--quantize Q4_K_M is handled by Ollama).

Usage:
    python tools/merge_lora_safetensors.py \\
        --adapter models/adapters/Minar0/my-ministral-3-8B-Instruct-2512-bnb-4bit-v1.0/checkpoint-150 \\
        --base   models/base/unsloth/Ministral-3-8B-Instruct-2512 \\
        --output models/merged/my-ministral-checkpoint-150 \\
        --name   my-ministral-150
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import torch
import safetensors.torch as st
from safetensors.torch import save_file


_SYSTEM_PROMPT = (
    "You are Romain. You write casually, mix French and English naturally, "
    "and prefer direct pragmatic answers. Reply as yourself — not as an AI. "
    "(Remove this line if you want to use the model's default system prompt.)"
)

_MISTRAL_TEMPLATE = '''\
TEMPLATE """{{ if .System }}[SYSTEM_PROMPT]{{ .System }}[/SYSTEM_PROMPT]{{ end }}\
[INST] {{ .Prompt }} [/INST]"""
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
PARAMETER stop "[SYSTEM_PROMPT]"
PARAMETER stop "[/SYSTEM_PROMPT]"
'''

# Adapter tensor prefix to strip to get the base model tensor name
# e.g. "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight"
#   →  "language_model.model.layers.0.mlp.down_proj"  (lora_A)
_ADAPTER_PREFIX_RE = re.compile(
    r"^base_model\.model\.model\.(.*)\.(lora_([AB]))\.weight$"
)


def load_adapter(adapter_dir: Path) -> tuple[dict[str, tuple], float, int]:
    """Load adapter weights.

    Returns:
        lora_map: base_tensor_name → (lora_A, lora_B)
        lora_alpha: alpha value
        lora_rank: rank r
    """
    adapter_file = adapter_dir / "adapter_model.safetensors"
    config_file = adapter_dir / "adapter_config.json"

    cfg = json.loads(config_file.read_text()) if config_file.exists() else {}
    lora_alpha = float(cfg.get("lora_alpha", 64))
    lora_rank = int(cfg.get("r", 32))

    print(f"LoRA config: rank={lora_rank}, alpha={lora_alpha}, scale={lora_alpha / lora_rank:.4f}")

    raw = st.load_file(str(adapter_file), device="cpu")
    print(f"Adapter tensors loaded: {len(raw)}")

    # Build map: base_weight_name → (lora_A tensor, lora_B tensor)
    lora_map: dict[str, dict[str, torch.Tensor]] = {}
    skipped = 0
    for name, tensor in raw.items():
        m = _ADAPTER_PREFIX_RE.match(name)
        if not m:
            skipped += 1
            continue
        base_name = m.group(1) + ".weight"   # e.g. "language_model.model.layers.0.mlp.down_proj.weight"
        ab = m.group(3)                       # "A" or "B"
        lora_map.setdefault(base_name, {})[ab] = tensor.float()

    # Pair up A and B
    paired: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for base_name, parts in lora_map.items():
        if "A" in parts and "B" in parts:
            paired[base_name] = (parts["A"], parts["B"])
        else:
            print(f"  ⚠  Incomplete LoRA pair for {base_name} — skipping", file=sys.stderr)

    print(f"  Paired LoRA weights: {len(paired)}  Skipped unparseable: {skipped}")
    return paired, lora_alpha, lora_rank


def apply_lora(
    weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Merge LoRA delta into a weight tensor.

    delta = lora_B @ lora_A  (scaled by alpha/rank)
    merged = weight + delta

    Always outputs bfloat16 — uint8/int8 (BNB-quantized) base weights cannot
    be meaningfully merged with float LoRA deltas, and llama.cpp rejects U8.
    """
    if weight.dtype in (torch.uint8, torch.int8):
        raise ValueError(
            f"Base model weight has BNB-quantized dtype {weight.dtype}. "
            "Use the fp16/bfloat16 version of the base model (without -bnb-4bit suffix)."
        )
    out_dtype = torch.bfloat16 if weight.dtype == torch.bfloat16 else torch.float16
    w = weight.float()
    delta = (lora_B @ lora_A) * scale
    return (w + delta).to(out_dtype)


def merge_shards(
    base_dir: Path,
    output_dir: Path,
    lora_paired: dict[str, tuple[torch.Tensor, torch.Tensor]],
    lora_scale: float,
) -> None:
    """Process each safetensors shard, merge LoRA where applicable."""
    index_file = base_dir / "model.safetensors.index.json"

    if index_file.exists():
        index = json.loads(index_file.read_text())
        weight_map = index["weight_map"]
        # Get ordered list of unique shards
        shards = list(dict.fromkeys(weight_map.values()))
    else:
        # Single-file model
        shards = ["model.safetensors"]
        weight_map = None

    merged_count = 0

    for shard_name in shards:
        shard_path = base_dir / shard_name
        print(f"\nProcessing shard: {shard_name} ({shard_path.stat().st_size / 1e9:.2f} GB)")

        tensors = st.load_file(str(shard_path), device="cpu")
        merged_tensors: dict[str, torch.Tensor] = {}

        for tensor_name, tensor in tensors.items():
            if tensor_name in lora_paired:
                lora_A, lora_B = lora_paired[tensor_name]
                merged_tensors[tensor_name] = apply_lora(tensor, lora_A, lora_B, lora_scale)
                merged_count += 1
            else:
                merged_tensors[tensor_name] = tensor

        out_shard = output_dir / shard_name
        save_file(merged_tensors, str(out_shard))
        size_gb = out_shard.stat().st_size / 1e9
        print(f"  Saved: {out_shard} ({size_gb:.2f} GB)")

        # Free shard memory
        del tensors, merged_tensors
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\nTotal LoRA merges applied: {merged_count}")

    # Copy index file
    if index_file.exists():
        shutil.copy2(index_file, output_dir / "model.safetensors.index.json")


def copy_metadata(base_dir: Path, adapter_dir: Path, output_dir: Path) -> None:
    """Copy config, tokenizer, and other metadata files."""
    copy_patterns = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "processor_config.json",
        "chat_template.jinja",
        "params.json",
    ]
    for pattern in copy_patterns:
        src = base_dir / pattern
        if src.exists():
            shutil.copy2(src, output_dir / pattern)

    # Use adapter's tokenizer/chat template if available (may be updated during training)
    for pattern in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
        src = adapter_dir / pattern
        if src.exists():
            shutil.copy2(src, output_dir / pattern)
            print(f"  Using adapter's {pattern}")


def write_modelfile(
    output_dir: Path,
    system_prompt: str | None,
    quant: str | None,
) -> Path:
    lines = ["FROM .\n"]
    if system_prompt:
        escaped = system_prompt.replace('"""', '\\"\\"\\"')
        lines.append(f'\nSYSTEM """{escaped}"""\n')
    lines.append(f"\n{_MISTRAL_TEMPLATE}")
    p = output_dir / "Modelfile"
    p.write_text("".join(lines))
    return p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model safetensors (shard by shard)"
    )
    parser.add_argument("--adapter", required=True,
                        help="Path to adapter directory (adapter_model.safetensors)")
    parser.add_argument("--base", required=True,
                        help="Path to base model directory (safetensors)")
    parser.add_argument("--output", required=True,
                        help="Output directory for merged model + Modelfile")
    parser.add_argument("--name", default=None,
                        help="Ollama model name (used in the printed ollama create command)")
    parser.add_argument("--quant", default="Q4_K_M",
                        help="Quantization for Ollama import (default: Q4_K_M). "
                             "Set to '' to skip quantization.")
    parser.add_argument("--system-prompt", default=_SYSTEM_PROMPT,
                        help="System prompt embedded in Modelfile")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter)
    base_dir = Path(args.base)
    output_dir = Path(args.output)

    for d, label in [(adapter_dir, "--adapter"), (base_dir, "--base")]:
        if not d.exists():
            print(f"ERROR: {label} path not found: {d}", file=sys.stderr)
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load adapter (fits in RAM, ~340 MB)
    print("=== Loading adapter ===")
    lora_paired, lora_alpha, lora_rank = load_adapter(adapter_dir)
    lora_scale = lora_alpha / lora_rank

    # Step 2: Merge shards one at a time
    print("\n=== Merging shards ===")
    merge_shards(base_dir, output_dir, lora_paired, lora_scale)

    # Step 3: Copy metadata
    print("\n=== Copying metadata ===")
    copy_metadata(base_dir, adapter_dir, output_dir)

    # Step 4: Write Modelfile
    quant = args.quant if args.quant else None
    system_prompt = args.system_prompt if args.system_prompt else None
    modelfile = write_modelfile(output_dir, system_prompt, quant)
    print(f"Modelfile written: {modelfile}")

    # Print command
    ollama_name = args.name or output_dir.name
    quant_flag = f" --quantize {quant}" if quant else ""
    print(f"\n✅ Done! Run:\n")
    print(f"  ollama create {ollama_name}{quant_flag} -f {output_dir}/Modelfile\n")
    print(f"Then test:\n  ollama run {ollama_name}")


if __name__ == "__main__":
    main()
