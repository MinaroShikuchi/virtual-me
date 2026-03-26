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
# This script uses the live HuggingFace Hub (HfApi) to automatically resolve 
# custom Ollama models to their corresponding HuggingFace repositories.


def _resolve_model_name(model_name: str) -> str:
    """Resolve an Ollama model name to a HuggingFace repo ID via live API search."""
    if "huggingface.co/" in model_name:
        # Extract repo ID from URL (e.g. https://huggingface.co/meta-llama/Llama-2-7b-hf -> meta-llama/Llama-2-7b-hf)
        repo_id = model_name.split("huggingface.co/")[-1].strip("/")
        # Remove potential suffixes like /tree/main or /blob/main
        repo_id = repo_id.split("/tree/")[0].split("/blob/")[0]
        return repo_id

    if ":" not in model_name and "/" in model_name:
        # Already a HuggingFace repo ID (e.g. "meta-llama/Llama-3-8B")
        return model_name

    lower = model_name.lower().strip()

    # Try finding it on HuggingFace Hub dynamically!
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        clean_name = lower.replace(":", "-").replace("_", "-")
        # Search for this model string in the Hub
        models = list(api.list_models(search=clean_name, sort="downloads", direction=-1, limit=10))
        if models:
            for m in models:
                # Prefer official text-generation models, avoid gguf/awq/gptq quantized ones
                name_l = m.id.lower()
                if "gguf" not in name_l and "awq" not in name_l and "gptq" not in name_l:
                    print(f"  Dynamic Resolution: Ollama '{model_name}' → HuggingFace '{m.id}'", flush=True)
                    return m.id
            
            # Fallback to the first one if all were quantized
            print(f"  Dynamic Resolution: Ollama '{model_name}' → HuggingFace '{models[0].id}'", flush=True)
            return models[0].id
    except Exception as e:
         print(f"  WARNING: Hub search failed: {e}", flush=True)

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

    # Try Unsloth first (faster training, lower memory)
    # Unsloth MUST be imported before transformers / trl
    try:
        from unsloth import FastLanguageModel
        _use_unsloth = True
        print("  Backend: Unsloth (fast path)", flush=True)
    except (ImportError, NotImplementedError):
        _use_unsloth = False
        print("  Backend: HuggingFace (install 'unsloth' for faster training)", flush=True)

    try:
        import warnings
        import torch
        from datasets import Dataset
        from transformers import (
            TrainerCallback,
            logging as hf_logging,
        )
        from trl import SFTConfig, SFTTrainer
        hf_logging.set_verbosity_info()
        hf_logging.enable_default_handler()
        hf_logging.enable_explicit_format()

        # Suppress the noisy "pin_memory not supported on MPS" warning
        warnings.filterwarnings("ignore", message=".*pin_memory.*")
        
        if not _use_unsloth:
            from transformers import (
                AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                AutoTokenizer, BitsAndBytesConfig, AutoConfig,
            )
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        print(f"\nERROR: Missing required library: {e}", flush=True)
        print("Install with: pip install transformers peft trl datasets bitsandbytes", flush=True)
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading training data structures…", flush=True)

    raw_examples = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                messages = obj.get("messages", [])
                if messages:
                    raw_examples.append({"messages": messages})
            except (json.JSONDecodeError, KeyError):
                continue

    # ── Load model + configure LoRA ────────────────────────────────────────
    print(f"\nLoading base model: {base_model}…", flush=True)

    # Guard: pre-quantized BNB models require CUDA
    _is_bnb_model = any(tag in base_model.lower() for tag in ["bnb-4bit", "bnb-8bit", "bnb_4bit", "bnb_8bit"])
    if _is_bnb_model and not _use_unsloth and not torch.cuda.is_available():
        raise RuntimeError(
            f"The model '{base_model}' is a pre-quantized bitsandbytes model "
            f"that requires a CUDA GPU. On Apple Silicon / CPU, use a non-quantized "
            f"model instead (e.g. 'mistralai/Ministral-8B-Instruct-2410' or "
            f"'meta-llama/Llama-3.1-8B')."
        )

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
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except ValueError:
                # Multimodal / conditional-generation models (e.g. Mistral3)
                # aren't registered under AutoModelForCausalLM.  Load the
                # architecture-specific class directly so we get a model that
                # supports .generate() and LoRA.
                print("  AutoModelForCausalLM unsupported — loading architecture "
                      "class directly", flush=True)
                _cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
                _arch = getattr(_cfg, "architectures", [None])[0]
                if _arch:
                    import transformers
                    _cls = getattr(transformers, _arch, None)
                    if _cls is None:
                        raise ValueError(
                            f"Architecture '{_arch}' not found in transformers "
                            f"{transformers.__version__}. Try upgrading: "
                            f"pip install -U transformers"
                        )
                    model = _cls.from_pretrained(
                        base_model,
                        quantization_config=bnb_config,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                else:
                    raise
            model = prepare_model_for_kbit_training(model)
        else:
            device_map = "auto" if torch.cuda.is_available() else None
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # Apple Silicon (MPS): use bfloat16 instead of float32 to halve
            # memory usage.  MPS supports bfloat16 from PyTorch 2.1+.
            _has_mps = (
                not torch.cuda.is_available()
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
            if _has_mps:
                dtype = torch.bfloat16
                print("  MPS detected — using bfloat16 to reduce memory", flush=True)

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                )
            except ValueError:
                # Multimodal / conditional-generation models (e.g. Mistral3)
                # aren't registered under AutoModelForCausalLM.  Load the
                # architecture-specific class directly so we get a model that
                # supports .generate() and LoRA.
                print("  AutoModelForCausalLM unsupported — loading architecture "
                      "class directly", flush=True)
                _cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
                _arch = getattr(_cfg, "architectures", [None])[0]
                if _arch:
                    import transformers
                    _cls = getattr(transformers, _arch, None)
                    if _cls is None:
                        raise ValueError(
                            f"Architecture '{_arch}' not found in transformers "
                            f"{transformers.__version__}. Try upgrading: "
                            f"pip install -U transformers"
                        )
                    model = _cls.from_pretrained(
                        base_model,
                        torch_dtype=dtype,
                        device_map=device_map,
                        trust_remote_code=True,
                    )
                else:
                    raise

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Base models often lack a chat template — set a sensible default
        # so apply_chat_template() works during dataset formatting.
        if tokenizer.chat_template is None:
            print("  No chat template found — applying Llama-3 style default", flush=True)
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'}}"
                "{% if message['content'] is string %}"
                "{{ message['content'] }}"
                "{% else %}"
                "{% for block in message['content'] %}"
                "{% if block['type'] == 'text' %}{{ block['text'] }}{% endif %}"
                "{% endfor %}"
                "{% endif %}"
                "{{ '<|eot_id|>' }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}{{'<|start_header_id|>assistant<|end_header_id|>\n\n'}}{% endif %}"
            )

        print(f"  Model loaded: {model.config.model_type}", flush=True)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

        # Enable gradient checkpointing to trade compute for memory —
        # stores only a fraction of activations during the forward pass
        # and recomputes the rest during backward.
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled", flush=True)

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

    # ── Process dataset with native chat template ────────────────────────
    print("Formatting dataset with model's native chat template…", flush=True)
    
    if _use_unsloth:
        from unsloth import get_chat_template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="llama-3", # Defaulting to Llama-3 style as requested (<|user|>)
        )

    examples = []
    num_filtered = 0
    raw_total = len(raw_examples)

    for ex in raw_examples:
        # Standardize for multi-modal or picky processors: content must be a list of blocks
        sanitized = []
        for m in ex["messages"]:
            content = m.get("content", "")
            if isinstance(content, str):
                sanitized.append({"role": m["role"], "content": [{"type": "text", "text": content}]})
            else:
                sanitized.append(m)

        # Get the formatted text first (safer than direct binary tokenization)
        text = tokenizer.apply_chat_template(
            sanitized, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Now count tokens exactly
        # Note: some processors (like Pixtral) don't have .encode directly, use .tokenizer
        _tokenizer_obj = getattr(tokenizer, "tokenizer", tokenizer)
        token_ids = _tokenizer_obj.encode(text, add_special_tokens=False)
        
        if len(token_ids) > max_seq_length:
            num_filtered += 1
            continue
            
        examples.append({"text": text})
        
    dataset = Dataset.from_list(examples)
    print(f"  Processed {raw_total:,} examples:", flush=True)
    print(f"    - Kept: {len(dataset):,}", flush=True)
    if num_filtered > 0:
        print(f"    - Filtered (too long > {max_seq_length}): {num_filtered:,} ⚠️", flush=True)

    # ── Training arguments ─────────────────────────────────────────────────
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Detect whether we should use bf16 (Ampere+ GPUs / MPS) or fp16 (older CUDA)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        _use_bf16 = True
        _use_fp16 = False
    elif not torch.cuda.is_available() and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _use_bf16 = True
        _use_fp16 = False
    else:
        _use_bf16 = False
        _use_fp16 = torch.cuda.is_available()

    # Compute warmup_steps from warmup_ratio (warmup_ratio is deprecated
    # in transformers ≥ v5.2).
    total_steps = max(1, (num_examples // (batch_size * gradient_accumulation_steps)) * epochs)
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        fp16=_use_fp16,
        bf16=_use_bf16,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=True,              # avoid tqdm progress bar spam in log viewer
        dataloader_pin_memory=False,    # not supported on MPS, avoids warning
        # SFT-specific settings (moved from SFTTrainer constructor)
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
        dataset_num_proc=1,  # Force single process to avoid pickling ConfigModuleInstance
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\nStarting training ({epochs} epochs, {num_examples} examples)…", flush=True)

    # Note: Using simple string printing, avoiding local unpicklable classes in SFTTrainer
    if progress_callback:
        # If external progress tracking is needed, we would need a global callback class
        pass

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer_stats = trainer.train(resume_from_checkpoint=resume)

    # ── Save ───────────────────────────────────────────────────────────────
    print(f"\nSaving LoRA adapter to {output_dir}…", flush=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"\n✅ Fine-tuning complete!", flush=True)
    print(f"  Adapter saved to: {output_dir}", flush=True)
    print(f"  To use: load base model + apply LoRA adapter from {output_dir}", flush=True)

    # ── [OPTIONAL] Auto-Export to Ollama ───────────────────────────
    if ollama_name:
        try:
            from tools.export_to_ollama import export_model
            print("\n" + "="*60, flush=True)
            print(f"🚀 Automated pipeline: Exporting and registering as '{ollama_name}' in Ollama...", flush=True)
            print("="*60 + "\n", flush=True)
            # The export script handles GGUF conversion and ollama create
            export_model(
                adapter_path=str(output_path),
                out_path=f"{output_dir}-gguf",
                quant_method="q4_k_m",
                ollama_name=ollama_name
            )
        except Exception as e:
            print(f"\n⚠️ Auto-export to Ollama failed: {e}", flush=True)
            print("You can manually export it later.", flush=True)

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
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=100,
                        help="Save a checkpoint every N training steps (default: 100)")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization")
    parser.add_argument("--ollama-name", default=None,
                        help="If provided, automatically export the adapter to GGUF and register it in Ollama under this name.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint in output_dir if available.")
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
    )


if __name__ == "__main__":
    main()
