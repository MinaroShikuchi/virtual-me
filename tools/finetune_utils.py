"""
tools/finetune_utils.py — Shared utilities for LoRA fine-tuning.

Extracted from finetune_lora.py to keep the main training script lean.
Contains: model resolution, data loading, dataset preparation, model loading,
training config, and save/export helpers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


# ── Data helpers ───────────────────────────────────────────────────────────────


def merge_consecutive_roles(messages: list[dict]) -> list[dict]:
    """Merge consecutive messages with the same role into one.

    Many chat templates (Mistral, Gemma, etc.) require strict role alternation
    (user→assistant→user→…).  The JSONL data from export_finetune.py preserves
    burst-messaging (e.g. two user messages in a row), which is valid for some
    templates but crashes others (e.g. Mistral's Jinja2 template).

    Consecutive messages are joined with a space separator.
    """
    if not messages:
        return messages
    merged: list[dict] = [dict(messages[0])]
    for m in messages[1:]:
        if m["role"] == merged[-1]["role"]:
            merged[-1]["content"] += " " + m["content"]
        else:
            merged.append(dict(m))
    return merged


def sanitize_message_content(message: dict) -> dict:
    """Ensure message content is a plain string.

    Block lists (``[{"type": "text", ...}]``) are not rendered correctly by
    most chat templates — flatten them back to a plain string.
    """
    content = message.get("content", "")
    if isinstance(content, list):
        content = " ".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return {"role": message["role"], "content": content}


# ── Ollama → HuggingFace model mapping ────────────────────────────────────────


def resolve_model_name(model_name: str) -> str:
    """Resolve an Ollama model name to a HuggingFace repo ID via live API search."""
    if "huggingface.co/" in model_name:
        repo_id = model_name.split("huggingface.co/")[-1].strip("/")
        repo_id = repo_id.split("/tree/")[0].split("/blob/")[0]
        return repo_id

    if ":" not in model_name and "/" in model_name:
        return model_name

    lower = model_name.lower().strip()

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        clean_name = lower.replace(":", "-").replace("_", "-")
        models = list(api.list_models(
            search=clean_name, sort="downloads", direction=-1, limit=10,
        ))
        if models:
            for m in models:
                name_l = m.id.lower()
                if "gguf" not in name_l and "awq" not in name_l and "gptq" not in name_l:
                    print(f"  Dynamic Resolution: Ollama '{model_name}' → HuggingFace '{m.id}'",
                          flush=True)
                    return m.id
            print(f"  Dynamic Resolution: Ollama '{model_name}' → HuggingFace '{models[0].id}'",
                  flush=True)
            return models[0].id
    except Exception as e:
        print(f"  WARNING: Hub search failed: {e}", flush=True)

    print(f"  WARNING: Could not resolve '{model_name}' to a HuggingFace ID. "
          f"Using as-is.", flush=True)
    return model_name


# ── Data loading ──────────────────────────────────────────────────────────────


def load_jsonl_data(data_path: str | Path) -> tuple[list[dict], int]:
    """Load and validate a JSONL training data file.

    Returns
    -------
    raw_examples : list[dict]
        Each element is ``{"messages": [...]}``.
    num_lines : int
        Total number of non-empty lines in the file (for progress reporting).
    """
    data_file = Path(data_path)
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_path}", flush=True)
        sys.exit(1)

    # Count lines first
    with open(data_file, encoding="utf-8") as f:
        num_lines = sum(1 for line in f if line.strip())

    if num_lines == 0:
        print("ERROR: No training examples found in data file", flush=True)
        sys.exit(1)

    # Parse examples
    raw_examples: list[dict] = []
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

    print(f"  Training examples: {num_lines:,}", flush=True)
    return raw_examples, num_lines


# ── Model + LoRA loading ─────────────────────────────────────────────────────


def load_model_and_tokenizer(
    base_model: str,
    max_seq_length: int,
    use_4bit: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    use_unsloth: bool,
    torch_module,
):
    """Load the base model, tokenizer, and configure LoRA.

    Parameters
    ----------
    torch_module : module
        The ``torch`` module (passed in to avoid re-importing).

    Returns
    -------
    model, tokenizer, trainable_params, total_params
    """
    torch = torch_module

    # Mistral3 (e.g. Mistral-Small-3.1) only ships ForConditionalGeneration;
    # register it under AutoModelForCausalLM so from_pretrained works.
    if not use_unsloth:
        from transformers import AutoModelForCausalLM
        try:
            from transformers.models.mistral3 import (
                Mistral3Config,
                Mistral3ForConditionalGeneration,
            )
            AutoModelForCausalLM.register(Mistral3Config, Mistral3ForConditionalGeneration)
        except (ImportError, AttributeError):
            pass

    print(f"\nLoading base model: {base_model}…", flush=True)

    if use_unsloth:
        from unsloth import FastLanguageModel

        # Use bfloat16 when available to avoid Float/Half mismatch in
        # sdpa_dense_backward (torch flex_attention bug with float16).
        _unsloth_dtype = None
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            _unsloth_dtype = torch.bfloat16

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=use_4bit,
            dtype=_unsloth_dtype,
        )

        # Workaround for PyTorch flex_attention bug: "expected scalar type
        # Float but found Half" in sdpa_dense_backward.  Force eager
        # attention implementation which doesn't use flex_attention.
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"
        # Also disable flash/mem-efficient SDP as a safety net
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        print(f"  Model loaded: {model.config.model_type}", flush=True)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

        print(f"\nConfiguring LoRA (rank={lora_rank}, alpha={lora_alpha})…", flush=True)
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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

            _has_mps = (
                not torch.cuda.is_available()
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
            if _has_mps:
                dtype = torch.bfloat16
                print("  MPS detected — using bfloat16 to reduce memory", flush=True)

            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"  Model loaded: {model.config.model_type}", flush=True)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled", flush=True)

        print(f"\nConfiguring LoRA (rank={lora_rank}, alpha={lora_alpha})…", flush=True)
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)", flush=True)

    return model, tokenizer, trainable, total


# ── Dataset preparation ──────────────────────────────────────────────────────


def prepare_dataset(
    raw_examples: list[dict],
    tokenizer,
    max_seq_length: int,
    system_prompt: str | None,
    use_unsloth: bool,
):
    """Sanitize, merge, filter, and build a HuggingFace Dataset.

    Returns
    -------
    dataset : datasets.Dataset
        With a ``text`` column (pre-formatted chat template strings).
    tokenizer
        The tokenizer (may have been modified by Unsloth get_chat_template).
    stats : dict
        Counts for merged, filtered, template_errors.
    """
    from datasets import Dataset

    # NOTE: We intentionally do NOT override the chat template here.
    # The model's native tokenizer already has the correct template
    # (e.g. Mistral uses [INST]/[/INST], Llama-3 uses <|start_header_id|>).
    # Overriding with get_chat_template("llama-3") would cause a mismatch
    # for non-Llama models.

    print("Formatting dataset with model's native chat template…", flush=True)

    examples: list[dict] = []
    num_filtered = 0
    num_merged = 0
    num_template_errors = 0
    num_split = 0          # multi-turn examples that were auto-split
    num_split_kept = 0     # single-turn pairs recovered from splits
    too_long_details: list[str] = []  # details of examples still too long
    raw_total = len(raw_examples)

    _tokenizer_obj = getattr(tokenizer, "tokenizer", tokenizer)

    def _try_format_and_add(messages: list[dict]) -> bool:
        """Try to format messages with chat template and add if within length.

        Returns True if the example was added, False if too long or errored.
        """
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        except Exception:
            return False

        token_ids = _tokenizer_obj.encode(text, add_special_tokens=False)
        if len(token_ids) > max_seq_length:
            return False

        examples.append({"text": text})
        return True

    def _split_into_single_turns(messages: list[dict]) -> list[list[dict]]:
        """Split a multi-turn conversation into individual user/assistant pairs.

        Preserves the system message (if any) in each split.
        """
        system_msgs = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]

        pairs: list[list[dict]] = []
        i = 0
        while i < len(non_system) - 1:
            if non_system[i]["role"] == "user" and non_system[i + 1]["role"] == "assistant":
                pair = list(system_msgs) + [non_system[i], non_system[i + 1]]
                pairs.append(pair)
                i += 2
            else:
                i += 1
        return pairs

    preview_printed = False
    for ex in raw_examples:
        sanitized = [sanitize_message_content(m) for m in ex["messages"]]

        # Merge consecutive same-role messages
        merged = merge_consecutive_roles(sanitized)
        if len(merged) != len(sanitized):
            num_merged += 1
            sanitized = merged

        # Optionally inject system prompt
        has_system = any(m["role"] == "system" for m in sanitized)
        if not has_system and system_prompt is not None:
            sanitized.insert(0, {"role": "system", "content": system_prompt})

        # Validate with chat template
        try:
            text = tokenizer.apply_chat_template(
                sanitized, tokenize=False, add_generation_prompt=False,
            )
        except Exception as e:
            num_template_errors += 1
            if num_template_errors <= 3:
                print(f"  ⚠️  Template error (skipping): {e}", flush=True)
            continue

        # Preview first valid example
        roles = {m["role"] for m in sanitized}
        if not preview_printed and "user" in roles and "assistant" in roles:
            print("\n── Chat template preview (first user/assistant example) ────",
                  flush=True)
            print(text[:1000] + ("…" if len(text) > 1000 else ""), flush=True)
            print("────────────────────────────────────────────────────────────\n",
                  flush=True)
            preview_printed = True

        # Check token length
        token_ids = _tokenizer_obj.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_seq_length:
            examples.append({"text": text})
            continue

        # ── Too long: try auto-splitting multi-turn into single-turn pairs ──
        non_system = [m for m in sanitized if m["role"] != "system"]
        user_assistant_pairs = sum(
            1 for i in range(len(non_system) - 1)
            if non_system[i]["role"] == "user" and non_system[i + 1]["role"] == "assistant"
        )

        if user_assistant_pairs > 1:
            # Multi-turn: split and try each pair individually
            num_split += 1
            pairs = _split_into_single_turns(sanitized)
            any_kept = False
            for pair in pairs:
                if _try_format_and_add(pair):
                    num_split_kept += 1
                    any_kept = True
                else:
                    # Even a single-turn pair is too long
                    pair_preview = " | ".join(
                        f"{m['role']}: {m['content'][:60]}…" if len(m['content']) > 60
                        else f"{m['role']}: {m['content']}"
                        for m in pair if m["role"] != "system"
                    )
                    pair_tokens = len(_tokenizer_obj.encode(
                        tokenizer.apply_chat_template(pair, tokenize=False, add_generation_prompt=False),
                        add_special_tokens=False,
                    ))
                    too_long_details.append(
                        f"  [{pair_tokens:,} tokens] (split from multi-turn) "
                        f"{ex.get('conversation', '?')}: {pair_preview}"
                    )
                    num_filtered += 1
            if not any_kept:
                # All pairs were too long
                pass
        else:
            # Single-turn but still too long
            num_filtered += 1
            preview = " | ".join(
                f"{m['role']}: {m['content'][:60]}…" if len(m['content']) > 60
                else f"{m['role']}: {m['content']}"
                for m in sanitized if m["role"] != "system"
            )
            too_long_details.append(
                f"  [{len(token_ids):,} tokens] {ex.get('conversation', '?')}: {preview}"
            )

    dataset = Dataset.from_list(examples).shuffle(seed=42)

    # Print stats
    print(f"  Processed {raw_total:,} examples:", flush=True)
    print(f"    - Kept: {len(dataset):,}", flush=True)
    if num_merged > 0:
        print(f"    - Merged consecutive roles: {num_merged:,}", flush=True)
    if num_split > 0:
        print(f"    - Auto-split multi-turn → single-turn: {num_split:,} "
              f"(recovered {num_split_kept:,} pairs)", flush=True)
    if num_filtered > 0:
        print(f"    - Filtered (too long > {max_seq_length}): {num_filtered:,} ⚠️",
              flush=True)
    if num_template_errors > 0:
        print(f"    - Template errors (skipped): {num_template_errors:,} ⚠️",
              flush=True)

    # Show ALL examples that are still too long
    if too_long_details:
        print(f"\n  ── Examples exceeding {max_seq_length} tokens (even after splitting) ──",
              flush=True)
        for detail in too_long_details:
            print(detail, flush=True)
        print(f"  ── End of too-long examples ({len(too_long_details):,} total) ──\n",
              flush=True)

    stats = {
        "total": raw_total,
        "kept": len(dataset),
        "merged": num_merged,
        "split": num_split,
        "split_kept": num_split_kept,
        "filtered": num_filtered,
        "template_errors": num_template_errors,
    }
    return dataset, tokenizer, stats


# ── Training config ──────────────────────────────────────────────────────────


def build_training_args(
    output_dir: str,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_ratio: float,
    logging_steps: int,
    save_steps: int,
    max_seq_length: int,
    num_examples: int,
    torch_module,
):
    """Build an SFTConfig for text-based SFT training.

    Returns
    -------
    training_args : SFTConfig
    output_path : Path
    """
    from trl import SFTConfig
    torch = torch_module

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Detect precision
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        use_bf16, use_fp16 = True, False
    elif (not torch.cuda.is_available()
          and hasattr(torch.backends, "mps")
          and torch.backends.mps.is_available()):
        use_bf16, use_fp16 = True, False
    else:
        use_bf16 = False
        use_fp16 = torch.cuda.is_available()

    total_steps = max(
        1, (num_examples // (batch_size * gradient_accumulation_steps)) * epochs,
    )
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
        save_total_limit=5,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=True,
        dataloader_pin_memory=False,
        # SFT-specific
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
        dataset_num_proc=1,
    )

    return training_args, output_path


# ── Save & export ─────────────────────────────────────────────────────────────


def save_and_export(
    model,
    tokenizer,
    output_path: Path,
):
    """Save LoRA adapter weights and tokenizer."""
    print(f"\nSaving LoRA adapter to {output_path}…", flush=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"\n✅ Fine-tuning complete!", flush=True)
    print(f"  Adapter saved to: {output_path}", flush=True)
    print(f"  To register in Ollama, use the Models page (Step 2 → Step 3).",
          flush=True)
