#!/usr/bin/env python3
"""
tools/finetune_lora.py — LoRA fine-tuning on exported conversation data.

Reads a JSONL file of user/assistant pairs (from export_finetune.py) and
fine-tunes a base model using LoRA (Low-Rank Adaptation) via the
``transformers`` + ``peft`` + ``trl`` stack.

Usage:
    python tools/finetune_lora.py \
        --base-model meta-llama/Llama-3-8B \
        --data ./data/facebook/finetune_data.jsonl \
        --output ./models/my-lora \
        --epochs 3 --batch-size 2 --lr 2e-4 --lora-rank 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# ── Ollama → HuggingFace model mapping ────────────────────────────────────────
# Maps common Ollama model names (with colon tags) to their HuggingFace repo IDs.
# This allows users to select from their local Ollama models and have the script
# automatically resolve to the correct HuggingFace weights for fine-tuning.

_OLLAMA_TO_HF: dict[str, str] = {
    # Llama 3
    "llama3:8b":                    "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3:8b-instruct":           "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3:70b":                   "meta-llama/Meta-Llama-3-70B-Instruct",
    # Llama 3.1
    "llama3.1:8b":                  "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1:70b":                 "meta-llama/Llama-3.1-70B-Instruct",
    # Llama 3.2
    "llama3.2:1b":                  "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2:3b":                  "meta-llama/Llama-3.2-3B-Instruct",
    # Mistral
    "mistral:7b":                   "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral:latest":               "mistralai/Mistral-7B-Instruct-v0.3",
    # Mixtral
    "mixtral:8x7b":                 "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # Qwen 2.5
    "qwen2.5:0.5b":                "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5:1.5b":                "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5:3b":                  "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5:7b":                  "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:14b":                 "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5:32b":                 "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5:72b":                 "Qwen/Qwen2.5-72B-Instruct",
    # Phi
    "phi3:mini":                    "microsoft/Phi-3-mini-4k-instruct",
    "phi3:medium":                  "microsoft/Phi-3-medium-4k-instruct",
    # Gemma
    "gemma:2b":                     "google/gemma-2b-it",
    "gemma:7b":                     "google/gemma-7b-it",
    "gemma2:2b":                    "google/gemma-2-2b-it",
    "gemma2:9b":                    "google/gemma-2-9b-it",
    "gemma2:27b":                   "google/gemma-2-27b-it",
}


def _resolve_model_name(model_name: str) -> str:
    """Resolve an Ollama model name to a HuggingFace repo ID.

    If the name contains a colon (Ollama format like ``mistral:7b``),
    look it up in the mapping table.  If not found, try stripping the
    tag and matching the base name.  If still not found, return as-is
    (assumes it's already a HuggingFace ID).
    """
    if ":" not in model_name and "/" in model_name:
        # Already a HuggingFace repo ID (e.g. "meta-llama/Llama-3-8B")
        return model_name

    # Direct lookup
    lower = model_name.lower().strip()
    if lower in _OLLAMA_TO_HF:
        resolved = _OLLAMA_TO_HF[lower]
        print(f"  Resolved Ollama model '{model_name}' → HuggingFace '{resolved}'", flush=True)
        return resolved

    # Try without tag (e.g. "mistral:latest" → "mistral")
    base = lower.split(":")[0]
    for key, val in _OLLAMA_TO_HF.items():
        if key.startswith(base + ":"):
            print(f"  Resolved Ollama model '{model_name}' → HuggingFace '{val}' (fuzzy match)", flush=True)
            return val

    # Not found — return as-is and let HuggingFace handle the error
    print(f"  WARNING: Could not resolve '{model_name}' to a HuggingFace ID. "
          f"Using as-is.", flush=True)
    return model_name


def run_finetune(
    base_model: str,
    data_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    max_seq_length: int = 512,
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.03,
    save_steps: int = 100,
    logging_steps: int = 10,
    use_4bit: bool = True,
    progress_callback=None,
):
    """
    Run LoRA fine-tuning on a base model using conversation data.

    Parameters
    ----------
    base_model : str
        HuggingFace model ID or local path (e.g. "meta-llama/Llama-3-8B",
        "unsloth/llama-3-8b-bnb-4bit").
    data_path : str
        Path to the JSONL training data (output of export_finetune.py).
    output_dir : str
        Directory to save the LoRA adapter weights.
    epochs : int
        Number of training epochs.
    batch_size : int
        Per-device training batch size.
    learning_rate : float
        Learning rate (2e-4 is typical for LoRA).
    lora_rank : int
        LoRA rank (8 is a good starting point).
    lora_alpha : int
        LoRA alpha scaling factor (typically 2x rank).
    lora_dropout : float
        Dropout for LoRA layers.
    max_seq_length : int
        Maximum sequence length for tokenization.
    gradient_accumulation_steps : int
        Gradient accumulation steps (effective batch = batch_size * this).
    warmup_ratio : float
        Warmup ratio for learning rate scheduler.
    save_steps : int
        Save checkpoint every N steps.
    logging_steps : int
        Log metrics every N steps.
    use_4bit : bool
        Use 4-bit quantization (QLoRA) for memory efficiency.
    progress_callback : callable or None
        Called with (step, total_steps) during training.
    """
    # Resolve Ollama model names (e.g. "mistral:7b") to HuggingFace IDs
    base_model = _resolve_model_name(base_model)

    print(f"[finetune_lora] Starting LoRA fine-tuning", flush=True)
    print(f"  Base model: {base_model}", flush=True)
    print(f"  Data: {data_path}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}", flush=True)
    print(f"  LoRA rank: {lora_rank}, alpha: {lora_alpha}, dropout: {lora_dropout}", flush=True)
    print(f"  Max seq length: {max_seq_length}, 4-bit: {use_4bit}", flush=True)
    print(flush=True)

    # ── Validate data file ─────────────────────────────────────────────────
    data_file = Path(data_path)
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_path}", flush=True)
        sys.exit(1)

    # Count examples
    with open(data_file, encoding="utf-8") as f:
        num_examples = sum(1 for line in f if line.strip())
    print(f"  Training examples: {num_examples:,}", flush=True)

    if num_examples == 0:
        print("ERROR: No training examples found in data file", flush=True)
        sys.exit(1)

    # ── Import training libraries ──────────────────────────────────────────
    print("\nLoading libraries…", flush=True)

    try:
        import torch
        from datasets import Dataset
        from transformers import (
            TrainerCallback,
            TrainingArguments,
            logging as hf_logging,
        )
        from trl import SFTTrainer
        hf_logging.set_verbosity_info()
        hf_logging.enable_default_handler()
        hf_logging.enable_explicit_format()
    except ImportError as e:
        print(f"\nERROR: Missing required library: {e}", flush=True)
        print("Install with: pip install transformers peft trl datasets bitsandbytes", flush=True)
        sys.exit(1)

    # Try Unsloth first (faster training, lower memory)
    try:
        from unsloth import FastLanguageModel
        _use_unsloth = True
        print("  Backend: Unsloth (fast path)", flush=True)
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        _use_unsloth = False
        print("  Backend: HuggingFace (install 'unsloth' for faster training)", flush=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading training data…", flush=True)

    examples = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                messages = obj.get("messages", [])
                # Format as chat template
                text_parts = []
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "user":
                        text_parts.append(f"<|user|>\n{content}")
                    elif role == "assistant":
                        text_parts.append(f"<|assistant|>\n{content}")
                if text_parts:
                    examples.append({"text": "\n".join(text_parts) + "\n<|end|>"})
            except (json.JSONDecodeError, KeyError):
                continue

    dataset = Dataset.from_list(examples)
    print(f"  Loaded {len(dataset):,} examples into dataset", flush=True)

    # ── Load model + configure LoRA ────────────────────────────────────────
    print(f"\nLoading base model: {base_model}…", flush=True)

    if _use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=use_4bit,
            dtype=None,  # auto-detect
        )
        print(f"  Model loaded: {model.config.model_type}", flush=True)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

        print(f"\nConfiguring LoRA (rank={lora_rank}, alpha={lora_alpha})…", flush=True)
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        if use_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model)
        else:
            device_map = "auto" if torch.cuda.is_available() else None
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            if not torch.cuda.is_available() and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dtype = torch.float32  # MPS doesn't support float16 well for training
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"  Model loaded: {model.config.model_type}", flush=True)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

        print(f"\nConfiguring LoRA (rank={lora_rank}, alpha={lora_alpha})…", flush=True)
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)", flush=True)

    # ── Training arguments ─────────────────────────────────────────────────
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        bf16=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\nStarting training ({epochs} epochs, {num_examples} examples)…", flush=True)

    class _ProgressCallback(TrainerCallback):
        def on_step_end(self, _args, state, _control, **_kwargs):
            if state.max_steps:
                pct = state.global_step / state.max_steps * 100
                print(
                    f"  Step {state.global_step}/{state.max_steps} ({pct:.1f}%)"
                    + (f"  loss={state.log_history[-1]['loss']:.4f}" if state.log_history else ""),
                    flush=True,
                )
                if progress_callback is not None:
                    progress_callback(state.global_step, state.max_steps)

    callbacks = [_ProgressCallback()]

    import inspect
    _trl_new_api = "processing_class" in inspect.signature(SFTTrainer.__init__).parameters

    if _trl_new_api:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=training_args,
            callbacks=callbacks or None,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            max_seq_length=max_seq_length,
            dataset_text_field="text",
            packing=False,
            callbacks=callbacks or None,
        )

    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────
    print(f"\nSaving LoRA adapter to {output_dir}…", flush=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"\n✅ Fine-tuning complete!", flush=True)
    print(f"  Adapter saved to: {output_dir}", flush=True)
    print(f"  To use: load base model + apply LoRA adapter from {output_dir}", flush=True)

    return {
        "base_model": base_model,
        "output_dir": str(output_path),
        "epochs": epochs,
        "examples": num_examples,
        "trainable_params": trainable,
        "total_params": total,
        "lora_rank": lora_rank,
    }


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning on conversation data")
    parser.add_argument("--base-model", required=True,
                        help="HuggingFace model ID (e.g. meta-llama/Llama-3-8B)")
    parser.add_argument("--data", default="./data/facebook/finetune_data.jsonl",
                        help="Path to JSONL training data")
    parser.add_argument("--output", default="./models/my-lora",
                        help="Output directory for LoRA adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization")
    args = parser.parse_args()

    run_finetune(
        base_model=args.base_model,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_seq_length=args.max_seq_length,
        gradient_accumulation_steps=args.grad_accum,
        use_4bit=not args.no_4bit,
    )


if __name__ == "__main__":
    main()
