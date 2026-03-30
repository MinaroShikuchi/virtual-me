#!/usr/bin/env python3
"""
tools/make_modelfile.py

Converts a HuggingFace LoRA adapter (safetensors) to a GGUF LoRA adapter
and writes an Ollama Modelfile so you can run:

    ollama create my-model -f <output_dir>/Modelfile

Usage:
    python tools/make_modelfile.py \
        --adapter models/adapters/Minar0_my-ministral-3-8B-Instruct-2512-bnb-4bit-v1.0/checkpoint-150 \
        --base ministral-3:8b \
        --output models/gguf/my-model-checkpoint-150 \
        --name my-model-150
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

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

# HuggingFace module name → GGUF tensor name fragment
_MODULE_MAP = {
    "self_attn.q_proj":   "attn_q",
    "self_attn.k_proj":   "attn_k",
    "self_attn.v_proj":   "attn_v",
    "self_attn.o_proj":   "attn_output",
    "mlp.gate_proj":      "ffn_gate",
    "mlp.up_proj":        "ffn_up",
    "mlp.down_proj":      "ffn_down",
}

# Regex to parse HF LoRA tensor names (handles Mistral3 multimodal prefix)
# Groups: (layer_idx, module_path, lora_ab)
_HF_PATTERN = re.compile(
    r"^(?:base_model\.model\.model\.)?(?:language_model\.)?layers\.(\d+)\."
    r"([\w.]+)\."
    r"lora_([AB])\.weight$"
)


def _hf_name_to_gguf(hf_name: str) -> str | None:
    """Convert a HuggingFace LoRA tensor name to GGUF format.

    Returns None if the tensor should be skipped (e.g. vision tower weights).
    """
    m = _HF_PATTERN.match(hf_name)
    if not m:
        return None

    layer_idx, module_path, ab = m.group(1), m.group(2), m.group(3)

    gguf_module = _MODULE_MAP.get(module_path)
    if gguf_module is None:
        print(f"  ⚠  Unknown module '{module_path}' — skipping", file=sys.stderr)
        return None

    lora_ab = "lora_a" if ab == "A" else "lora_b"
    return f"blk.{layer_idx}.{gguf_module}.weight.{lora_ab}"


def convert_adapter_to_gguf(adapter_dir: Path, output_path: Path, lora_alpha: float) -> None:
    """Convert adapter_model.safetensors → GGUF LoRA adapter file."""
    import safetensors.torch
    import gguf

    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.exists():
        print(f"ERROR: {adapter_file} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading adapter: {adapter_file}")
    tensors = safetensors.torch.load_file(str(adapter_file))
    print(f"  {len(tensors)} tensors")

    writer = gguf.GGUFWriter(str(output_path), arch="llama")
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, lora_alpha)

    converted = skipped = 0
    for hf_name, tensor in tensors.items():
        gguf_name = _hf_name_to_gguf(hf_name)
        if gguf_name is None:
            skipped += 1
            continue
        arr = tensor.to(dtype=tensor.dtype).float().numpy().astype(np.float32)
        writer.add_tensor(gguf_name, arr)
        converted += 1

    print(f"  Converted: {converted}  Skipped: {skipped}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"  GGUF adapter written: {output_path} ({size_mb:.1f} MB)")


def write_modelfile(
    output_dir: Path,
    base: str,
    adapter_filename: str | None,
    system_prompt: str | None,
) -> Path:
    """Write Ollama Modelfile to output_dir/Modelfile."""
    lines = [f"FROM {base}\n"]

    if adapter_filename:
        lines.append(f"ADAPTER ./{adapter_filename}\n")

    if system_prompt:
        escaped = system_prompt.replace('"""', '\\"\\"\\"')
        lines.append(f'\nSYSTEM """{escaped}"""\n')

    lines.append(f"\n{_MISTRAL_TEMPLATE}")

    modelfile_path = output_dir / "Modelfile"
    modelfile_path.write_text("".join(lines))
    print(f"Modelfile written: {modelfile_path}")
    return modelfile_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LoRA adapter to GGUF and write Ollama Modelfile"
    )
    parser.add_argument(
        "--adapter", required=True,
        help="Path to adapter directory (containing adapter_model.safetensors)",
    )
    parser.add_argument(
        "--base", default="ministral-3:8b",
        help="Base model: Ollama model name (e.g. ministral-3:8b) or path to .gguf file. "
             "Default: ministral-3:8b",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for adapter.gguf + Modelfile",
    )
    parser.add_argument(
        "--name", default=None,
        help="Ollama model name to register (printed as the ollama create command)",
    )
    parser.add_argument(
        "--system-prompt", default=_SYSTEM_PROMPT,
        help="System prompt to embed in Modelfile (set to '' to omit)",
    )
    parser.add_argument(
        "--no-convert", action="store_true",
        help="Skip adapter conversion (use if adapter.gguf already exists in --output)",
    )
    args = parser.parse_args()

    adapter_dir = Path(args.adapter)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load adapter config for lora_alpha
    config_path = adapter_dir / "adapter_config.json"
    lora_alpha = 64.0  # fallback default
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        lora_alpha = float(cfg.get("lora_alpha", lora_alpha))
        print(f"LoRA config: rank={cfg.get('r')}, alpha={lora_alpha}")

    # Convert adapter safetensors → GGUF
    adapter_gguf_name = "adapter.gguf"
    adapter_gguf_path = output_dir / adapter_gguf_name

    if not args.no_convert:
        convert_adapter_to_gguf(adapter_dir, adapter_gguf_path, lora_alpha)
    else:
        if not adapter_gguf_path.exists():
            print(f"ERROR: --no-convert set but {adapter_gguf_path} not found", file=sys.stderr)
            sys.exit(1)
        print(f"Skipping conversion, using existing: {adapter_gguf_path}")

    # Write Modelfile
    system_prompt = args.system_prompt if args.system_prompt else None
    write_modelfile(output_dir, args.base, adapter_gguf_name, system_prompt)

    # Print ollama create command
    ollama_name = args.name or output_dir.name
    print(f"\n✅ Done! Run:\n")
    print(f"  ollama create {ollama_name} -f {output_dir}/Modelfile\n")
    print(f"Then test with:\n")
    print(f"  ollama run {ollama_name}")


if __name__ == "__main__":
    main()
