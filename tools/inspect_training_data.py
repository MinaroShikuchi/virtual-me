#!/usr/bin/env python3
"""
tools/inspect_training_data.py — Inspect fine-tuning data after chat-template formatting.

Loads a JSONL file (from export_finetune.py) and a tokenizer (no model weights),
applies the **exact same** formatting pipeline used by finetune_lora.py, and
displays the resulting text so you can verify the model sees proper role
delimiters, EOS tokens, etc.

This is a diagnostic tool — run it *before* training to catch formatting issues
that would cause the model to learn autocomplete instead of conversation.

Usage:
    python tools/inspect_training_data.py \
        --model meta-llama/Llama-3-8B \
        --data  ./data/facebook/finetune_data.jsonl \
        --num-examples 5 \
        --max-seq-length 512
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

try:
    from tools.finetune_utils import (
        merge_consecutive_roles,
        sanitize_message_content,
    )
except ModuleNotFoundError:
    # When run directly (python tools/inspect_training_data.py), the parent
    # directory isn't on sys.path.  Add it so `tools.*` imports work.
    import os as _os
    _os.sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.finetune_utils import (
        merge_consecutive_roles,
        sanitize_message_content,
    )


def _role_sequence(messages: list[dict]) -> str:
    """Return a compact role sequence string like 'user→assistant→user→assistant'."""
    return "→".join(m["role"] for m in messages)


# Common system prompt marker patterns across template families
_SYSTEM_PROMPT_RE = re.compile(
    r"(\[SYSTEM_PROMPT\])(.*?)(\[/SYSTEM_PROMPT\])"
    r"|"
    r"(<\|im_start\|>system\n)(.*?)(<\|im_end\|>)"
    r"|"
    r"(<<SYS>>\n)(.*?)(\n<</SYS>>)",
    re.DOTALL,
)


def _truncate_system_prompt(text: str, max_chars: int = 40) -> str:
    """Truncate the system prompt section in formatted text for display.

    Replaces the full system prompt content with a truncated version like:
        [SYSTEM_PROMPT]You are a helpful....[/SYSTEM_PROMPT]
    """
    def _replacer(m: re.Match) -> str:
        # Find which group matched (3 alternatives in the regex)
        for i in range(0, 9, 3):
            open_tag, content, close_tag = m.group(i + 1), m.group(i + 2), m.group(i + 3)
            if open_tag is not None:
                if len(content) > max_chars:
                    content = content[:max_chars].rstrip() + "…"
                return f"{open_tag}{content}{close_tag}"
        return m.group(0)  # fallback

    return _SYSTEM_PROMPT_RE.sub(_replacer, text)


# ── Diagnostic checks ─────────────────────────────────────────────────────────

def _run_diagnostics(
    formatted_texts: list[str],
    all_messages: list[list[dict]],
    tokenizer,
) -> list[str]:
    """Analyse formatted texts and return a list of warning strings."""
    warnings: list[str] = []

    # Gather special tokens from the tokenizer for detection
    special_tokens = set()
    if hasattr(tokenizer, "all_special_tokens"):
        special_tokens = set(tokenizer.all_special_tokens)

    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    # 1. Check if any example ends with a user message (nothing for the model to learn)
    user_ending = 0
    for msgs in all_messages:
        if msgs and msgs[-1]["role"] == "user":
            user_ending += 1
    if user_ending > 0:
        warnings.append(
            f"⚠️  {user_ending} example(s) end with a 'user' message — the model has "
            f"no assistant response to learn from in these."
        )

    # 2. Check for missing role markers in formatted text
    # Look for common role marker patterns across template families
    role_markers = [
        "<|start_header_id|>",   # Llama-3
        "<|im_start|>",          # ChatML (Qwen, Mistral, etc.)
        "[INST]",                # Llama-2 / Mistral instruct
        "<|user|>",              # Some custom templates
        "### Human:",            # Alpaca-style
        "<|role|>",              # Generic
    ]
    has_any_marker = False
    for text in formatted_texts[:10]:  # sample first 10
        for marker in role_markers:
            if marker in text:
                has_any_marker = True
                break
        if has_any_marker:
            break

    if not has_any_marker and formatted_texts:
        warnings.append(
            "⚠️  No known role markers found in formatted text! The chat template "
            "may be producing plain concatenation — the model won't learn turn structure."
        )

    # 3. Check for EOS / end-of-turn tokens
    # Some templates use a different end-of-turn token than the tokenizer's EOS
    # (e.g. Llama-3 uses <|eot_id|> while the tokenizer EOS is </s>).
    eot_candidates = [eos_token] if eos_token else []
    for tok in ["<|eot_id|>", "<|im_end|>", "</s>", "[/INST]"]:
        if tok not in eot_candidates:
            eot_candidates.append(tok)

    if formatted_texts:
        has_any_eot = False
        detected_eot = None
        for tok in eot_candidates:
            if tok and any(tok in t for t in formatted_texts[:10]):
                has_any_eot = True
                detected_eot = tok
                break

        if not has_any_eot:
            warnings.append(
                "⚠️  No end-of-turn token found in formatted examples. "
                "The model may not learn when to stop generating."
            )

    # 4. Check for examples with no special tokens at all
    if special_tokens and formatted_texts:
        no_special = 0
        for text in formatted_texts:
            if not any(tok in text for tok in special_tokens if tok.strip()):
                no_special += 1
        if no_special > len(formatted_texts) * 0.5:
            warnings.append(
                f"⚠️  {no_special}/{len(formatted_texts)} examples contain zero special tokens. "
                f"The template may not be applying correctly."
            )

    # 5. Check for consecutive same-role messages without clear separation
    consecutive_issues = 0
    for msgs in all_messages:
        for i in range(1, len(msgs)):
            if msgs[i]["role"] == msgs[i - 1]["role"]:
                consecutive_issues += 1
                break  # count per-example, not per-pair
    if consecutive_issues > 0:
        warnings.append(
            f"ℹ️  {consecutive_issues} example(s) have consecutive same-role messages "
            f"(e.g. two 'user' messages in a row). This is valid for trl/HuggingFace "
            f"but verify the template handles it correctly."
        )

    return warnings


# ── Main ───────────────────────────────────────────────────────────────────────

def inspect(
    model_name: str,
    data_path: str,
    num_examples: int = 5,
    max_seq_length: int = 512,
    system_prompt: str | None = None,
) -> None:
    """Load tokenizer + data, format, and display diagnostic output."""

    data_file = Path(data_path)
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load JSONL ─────────────────────────────────────────────────────────
    raw_examples: list[list[dict]] = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                messages = obj.get("messages", [])
                if messages:
                    raw_examples.append(messages)
            except (json.JSONDecodeError, KeyError):
                continue

    if not raw_examples:
        print("ERROR: No valid examples found in data file.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(raw_examples):,} examples from {data_path}\n")

    # ── Load tokenizer (no model weights) ──────────────────────────────────
    print(f"Loading tokenizer for: {model_name} …")

    # Try Unsloth first — mirrors finetune_lora.py logic.
    # Unsloth can fail at import time (ImportError) or at runtime
    # (NotImplementedError on Apple Silicon / non-NVIDIA systems).
    _use_unsloth = False
    try:
        from unsloth import FastLanguageModel, get_chat_template
        _use_unsloth = True
        print("  Backend: Unsloth detected")
    except Exception:
        pass  # will fall through to HuggingFace path below

    if _use_unsloth:
        try:
            # Unsloth's from_pretrained returns (model, tokenizer) but we only
            # need the tokenizer.  Load with minimal settings.
            _model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=True,
                dtype=None,
            )
            tokenizer = get_chat_template(
                tokenizer,
                chat_template="llama-3",
            )
            del _model  # free memory immediately
            print("  Chat template: llama-3 (via Unsloth)")
        except Exception as e:
            print(f"  Unsloth failed at runtime: {e}")
            _use_unsloth = False

    if not _use_unsloth:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        template_name = getattr(tokenizer, "chat_template", None)
        if template_name:
            preview = str(template_name)[:80] + ("…" if len(str(template_name)) > 80 else "")
            print(f"  Chat template (model-native): {preview}")
        else:
            print("  Chat template: (none / default)")
        print()
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║  NOTE: Using the model's NATIVE chat template.             ║")
        print("  ║  During training with Unsloth, finetune_lora.py overrides  ║")
        print("  ║  this with get_chat_template('llama-3'). If you train on   ║")
        print("  ║  a GPU with Unsloth, the actual formatting may differ.     ║")
        print("  ║  What you see here is what HuggingFace transformers would  ║")
        print("  ║  produce — which IS what's used when Unsloth is absent.    ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")

    _tokenizer_obj = getattr(tokenizer, "tokenizer", tokenizer)

    # ── Format all examples ────────────────────────────────────────────────
    print("\nFormatting all examples with chat template …\n")

    formatted_texts: list[str] = []
    all_sanitized: list[list[dict]] = []
    token_counts: list[int] = []
    filtered_count = 0
    merged_count = 0       # examples that needed consecutive-role merging
    template_errors = 0    # examples that failed even after merging

    for messages in raw_examples:
        sanitized = [sanitize_message_content(m) for m in messages]

        # Merge consecutive same-role messages (e.g. two user messages in a row).
        # Many templates (Mistral, Gemma) require strict role alternation.
        merged = merge_consecutive_roles(sanitized)
        if len(merged) != len(sanitized):
            merged_count += 1
            sanitized = merged

        # Prepend a system message to prevent the template from injecting
        # its own verbose default system prompt (mirrors finetune_lora.py logic).
        has_system = any(m["role"] == "system" for m in sanitized)
        if not has_system and system_prompt is not None:
            sanitized.insert(0, {
                "role": "system",
                "content": system_prompt,
            })

        try:
            text = tokenizer.apply_chat_template(
                sanitized,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            template_errors += 1
            if template_errors <= 3:
                roles = " → ".join(m["role"] for m in sanitized)
                print(f"  ⚠️  Template error on example (roles: {roles}): {e}")
            continue  # skip this example entirely

        all_sanitized.append(sanitized)
        formatted_texts.append(text)

        ids = _tokenizer_obj.encode(text, add_special_tokens=False)
        token_counts.append(len(ids))

        if len(ids) > max_seq_length:
            filtered_count += 1

    if merged_count > 0:
        print(f"  ℹ️  {merged_count:,} example(s) had consecutive same-role messages")
        print(f"     that were auto-merged (joined with space) for template compatibility.\n")
    if template_errors > 0:
        print(f"  ⚠️  {template_errors:,} example(s) failed template formatting even after merging.\n")

    # ── Display individual examples ────────────────────────────────────────
    display_count = min(num_examples, len(raw_examples))
    print("=" * 72)
    print(f"  FORMATTED EXAMPLES  ({display_count} of {len(raw_examples):,})")
    print("=" * 72)

    for i in range(display_count):
        sanitized = all_sanitized[i]
        text = formatted_texts[i]
        n_tokens = token_counts[i]
        over = n_tokens > max_seq_length

        print(f"\n{'─' * 72}")
        print(f"  Example {i + 1}")
        print(f"{'─' * 72}")

        # Raw messages (skip system role — it's template boilerplate)
        non_system = [m for m in sanitized if m["role"] != "system"]
        print(f"  Roles: {_role_sequence(non_system)}")
        print(f"  Tokens: {n_tokens:,}  {'⚠️ EXCEEDS max_seq_length' if over else '✓'}")
        print()

        for j, m in enumerate(non_system):
            content_preview = m["content"][:120]
            if len(m["content"]) > 120:
                content_preview += "…"
            print(f"    [{m['role']}] {content_preview}")

        # Formatted output — truncate the system prompt section for readability
        print(f"\n  Formatted text (what the model sees during training):")
        print()
        display_text = _truncate_system_prompt(text)
        print(display_text)
        print()

    # ── Aggregate statistics ───────────────────────────────────────────────
    print(f"\n\n{'=' * 72}")
    print(f"  AGGREGATE STATISTICS")
    print(f"{'=' * 72}\n")

    kept = len(raw_examples) - filtered_count
    print(f"  Total examples:    {len(raw_examples):,}")
    print(f"  Kept (≤ {max_seq_length} tokens): {kept:,}")
    if filtered_count:
        print(f"  Filtered (> {max_seq_length}):  {filtered_count:,} ⚠️")
    print()

    if token_counts:
        print(f"  Token counts:")
        print(f"    Min:    {min(token_counts):,}")
        print(f"    Max:    {max(token_counts):,}")
        print(f"    Mean:   {mean(token_counts):,.1f}")
        print(f"    Median: {median(token_counts):,.1f}")
        print()

    # Role sequence distribution
    seq_counter: Counter[str] = Counter()
    for sanitized in all_sanitized:
        seq_counter[_role_sequence(sanitized)] += 1

    print(f"  Role sequence patterns (top 10):")
    for seq, count in seq_counter.most_common(10):
        pct = 100 * count / len(all_sanitized)
        print(f"    {count:>6,}  ({pct:5.1f}%)  {seq}")
    print()

    # Message count distribution
    len_counter: Counter[int] = Counter()
    for sanitized in all_sanitized:
        len_counter[len(sanitized)] += 1

    print(f"  Messages per example:")
    for length in sorted(len_counter.keys()):
        count = len_counter[length]
        pct = 100 * count / len(all_sanitized)
        print(f"    {length} messages: {count:>6,}  ({pct:5.1f}%)")
    print()

    # ── Diagnostics ────────────────────────────────────────────────────────
    print(f"{'=' * 72}")
    print(f"  DIAGNOSTICS")
    print(f"{'=' * 72}\n")

    diag_warnings = _run_diagnostics(formatted_texts, all_sanitized, tokenizer)

    if diag_warnings:
        for w in diag_warnings:
            print(f"  {w}")
        print()
    else:
        print("  ✅ No issues detected.\n")



def main():
    parser = argparse.ArgumentParser(
        description="Inspect fine-tuning data after chat-template formatting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick check with 5 examples
  python tools/inspect_training_data.py \\
      --model meta-llama/Llama-3-8B \\
      --data ./data/facebook/finetune_data.jsonl

  # Show more examples, custom max length
  python tools/inspect_training_data.py \\
      --model mistralai/Mistral-7B-Instruct-v0.3 \\
      --data ./data/facebook/finetune_data.jsonl \\
      --num-examples 20 --max-seq-length 1024
        """,
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model ID (e.g. meta-llama/Llama-3-8B). "
             "Only the tokenizer is loaded — no GPU needed.",
    )
    parser.add_argument(
        "--data", default="./data/facebook/finetune_data.jsonl",
        help="Path to JSONL training data (default: ./data/facebook/finetune_data.jsonl)",
    )
    parser.add_argument(
        "--num-examples", type=int, default=5,
        help="Number of formatted examples to display (default: 5)",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=512,
        help="Max sequence length for filtering stats (default: 512)",
    )
    parser.add_argument(
        "--system-prompt", default=None,
        help="System prompt to prepend. By default, no system prompt is injected "
             "and the model's native template default is used.",
    )
    args = parser.parse_args()

    inspect(
        model_name=args.model,
        data_path=args.data,
        num_examples=args.num_examples,
        max_seq_length=args.max_seq_length,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    main()
