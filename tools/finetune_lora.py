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
import os
import sys
from pathlib import Path

# Workaround for PyTorch flex_attention bug (torch >= 2.5):
# "expected scalar type Float but found Half" in sdpa_dense_backward.
# Must be set BEFORE torch is imported.
os.environ.setdefault("TORCH_CUDNN_SDPA_ENABLED", "0")

try:
    from tools.finetune_utils import (
        resolve_model_name,
        load_jsonl_data,
        load_model_and_tokenizer,
        prepare_dataset,
        build_training_args,
        save_and_export,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.finetune_utils import (
        resolve_model_name,
        load_jsonl_data,
        load_model_and_tokenizer,
        prepare_dataset,
        build_training_args,
        save_and_export,
    )

# Re-export for backward compatibility (ui/models.py imports this)
_resolve_model_name = resolve_model_name


def run_finetune(
    base_model: str,
    data_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    max_seq_length: int = 512,
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.03,
    save_steps: int = 100,
    logging_steps: int = 10,
    use_4bit: bool = True,
    progress_callback=None,
    ollama_name: str = None,
    resume: bool = False,
    system_prompt: str | None = None,
):
    """Run LoRA fine-tuning on a base model using conversation data.

    See ``tools/finetune_utils.py`` for the implementation of each step.
    """
    # ── Resolve model name ────────────────────────────────────────────────
    base_model = resolve_model_name(base_model)

    print(f"[finetune_lora] Starting LoRA fine-tuning", flush=True)
    print(f"  Base model: {base_model}", flush=True)
    print(f"  Data: {data_path}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}", flush=True)
    print(f"  LoRA rank: {lora_rank}, alpha: {lora_alpha}, dropout: {lora_dropout}",
          flush=True)
    print(f"  Max seq length: {max_seq_length}, 4-bit: {use_4bit}", flush=True)
    print(flush=True)

    # ── Load data ─────────────────────────────────────────────────────────
    raw_examples, num_examples = load_jsonl_data(data_path)

    # ── Import training libraries (lazy — heavy deps) ─────────────────────
    print("\nLoading libraries…", flush=True)

    try:
        from unsloth import FastLanguageModel  # noqa: F401
        _use_unsloth = True
        print("  Backend: Unsloth (fast path)", flush=True)
    except ImportError:
        _use_unsloth = False
        print("  Backend: HuggingFace (install 'unsloth' for faster training)",
              flush=True)

    try:
        import warnings
        import torch
        from trl import SFTTrainer

        warnings.filterwarnings("ignore", message=".*pin_memory.*")
    except ImportError as e:
        print(f"\nERROR: Missing required library: {e}", flush=True)
        print("Install with: pip install transformers peft trl datasets bitsandbytes",
              flush=True)
        sys.exit(1)

    # ── Load model + LoRA ─────────────────────────────────────────────────
    model, tokenizer, trainable, total = load_model_and_tokenizer(
        base_model=base_model,
        max_seq_length=max_seq_length,
        use_4bit=use_4bit,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_unsloth=_use_unsloth,
        torch_module=torch,
    )

    # ── Prepare dataset ───────────────────────────────────────────────────
    dataset, tokenizer, stats = prepare_dataset(
        raw_examples=raw_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        system_prompt=system_prompt,
        use_unsloth=_use_unsloth,
    )

    # ── Build training config ─────────────────────────────────────────────
    training_args, output_path = build_training_args(
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        max_seq_length=max_seq_length,
        num_examples=num_examples,
        torch_module=torch,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\nStarting training ({epochs} epochs, {num_examples} examples)…",
          flush=True)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # ── Completion-only: mask user/system tokens ──────────────────────────
    # Only compute loss on assistant responses so the model doesn't learn
    # to predict user messages ("autocomplete mode").
    if _use_unsloth:
        from unsloth import train_on_responses_only
        # Detect the instruction/response markers from the first example
        sample = dataset[0]["text"]
        # Llama-3 style (Unsloth default)
        if "<|start_header_id|>assistant<|end_header_id|>" in sample:
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
            )
        # Mistral style
        elif "[/INST]" in sample:
            trainer = train_on_responses_only(
                trainer,
                instruction_part="[INST]",
                response_part="[/INST]",
            )
        # ChatML style
        elif "<|im_start|>assistant" in sample:
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )
        else:
            print("  ⚠️  Could not detect response markers — training on ALL tokens",
                  flush=True)
        print("  Completion-only: ENABLED (train_on_responses_only)", flush=True)
    else:
        print("  ⚠️  Completion-only training requires Unsloth — training on ALL tokens",
              flush=True)

    trainer.train(resume_from_checkpoint=resume)

    # ── Save & export ─────────────────────────────────────────────────────
    save_and_export(model, tokenizer, output_path, ollama_name)

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
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning on conversation data",
    )
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
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=100,
                        help="Save a checkpoint every N training steps (default: 100)")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization")
    parser.add_argument("--ollama-name", default=None,
                        help="If provided, automatically export the adapter to GGUF "
                             "and register it in Ollama under this name.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint in output_dir "
                             "if available.")
    parser.add_argument("--system-prompt", default=None,
                        help="System prompt to prepend to each example. By default, "
                             "no system prompt is injected and the model's native "
                             "template default is used.")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("COMMAND LINE:", flush=True)
    print("  " + " ".join(sys.argv), flush=True)
    print("SCRIPT PARAMETERS:", flush=True)
    for k, v in vars(args).items():
        print(f"  {k}: {v}", flush=True)
    print("=" * 60, flush=True)

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
        save_steps=args.save_steps,
        use_4bit=not args.no_4bit,
        ollama_name=args.ollama_name,
        resume=args.resume,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    main()
