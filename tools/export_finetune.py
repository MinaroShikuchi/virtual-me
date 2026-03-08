"""
tools/export_finetune.py — Export Facebook messages as fine-tuning data.

Reads the extracted ``facebook_messages.json``, filters by date range,
language, and quality, then structures conversations as user/assistant
pairs suitable for LLM fine-tuning.

Output format (JSONL):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Where:
    - "user" = the other person's message
    - "assistant" = your (self) reply
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from config import DATA_DIR


# ── System message detection (mirrors extract_facebook.py) ─────────────────────

_SYSTEM_PATTERNS = [
    r"^You (sent|are now connected|named the group|set the nickname|created a poll)",
    r"^.+ (sent|named the group|set the nickname|created a poll|set the emoji)",
    r"^.+ (joined|left) the (group|call|video chat)",
    r"^.+ (started|ended) (a |the )?(call|video chat|sharing)",
    r"^(Audio|Video) call",
    r"^You (missed|started|ended) (a |the )?(call|video chat)",
    r"^.+ reacted .+ to your message",
    r"^.+ unsent a message",
    r"^.+ changed the (theme|chat theme|group photo)",
    r"^This poll is no longer available",
    r"^.+ pinned a message",
    r"^Liked a message$",
]
_SYSTEM_RE = re.compile("|".join(_SYSTEM_PATTERNS), re.IGNORECASE)


def _is_system_message(text: str) -> bool:
    """Check if a message is a system/automated message."""
    return bool(_SYSTEM_RE.match(text))


# ── URL / noise patterns ──────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

# Emoji and special character ranges
_EMOJI_CHARS = (
    r"\U0001F600-\U0001F64F"  # emoticons
    r"\U0001F300-\U0001F5FF"  # symbols & pictographs
    r"\U0001F680-\U0001F6FF"  # transport & map
    r"\U0001F1E0-\U0001F1FF"  # flags
    r"\U00002702-\U000027B0"  # dingbats
    r"\U0000FE00-\U0000FE0F"  # variation selectors
    r"\U0001FA00-\U0001FA6F"  # chess symbols
    r"\U0001FA70-\U0001FAFF"  # symbols extended-A
    r"\U00002600-\U000026FF"  # misc symbols
    r"\U0000200D"             # zero-width joiner
    r"\U00002764"             # heavy heart
    r"\U0000FE0F"             # variation selector-16
    r"\U00002665"             # heart suit
    r"\U00002763"             # heavy heart exclamation
    r"\U0001F90D-\U0001F9FF"  # supplemental symbols
)
_EMOJI_ONLY_RE = re.compile(rf"^[{_EMOJI_CHARS}\s♥❤💕💗💖💘💝💞💟♡]+$")

# Patterns for low-quality / non-conversational messages
_LOW_QUALITY_PATTERNS = re.compile(
    r"^(you ♥|♥|❤|💕|💗|haha|lol|ok|okay|yes|no|yeah|yep|nope|hmm|"
    r"ah|oh|wow|omg|idk|brb|ttyl|k|kk|thx|ty|np|gg|lmao|"
    r"😂|😊|😍|🥰|😘|👍|👌|🙏|❤️|💕|💖|"
    r"\?+|!+|\.+)$",
    re.IGNORECASE,
)

# Facebook-specific artifacts that slip through system message detection
_FB_ARTIFACT_RE = re.compile(
    r"(sent a photo|sent a video|sent a sticker|sent a GIF|"
    r"sent an attachment|liked a message|"
    r"You sent \$|You sent a link|"
    r"is now connected on Messenger|"
    r"Say hi to your new Facebook friend|"
    r"waved at you|"
    r"This person is unavailable)",
    re.IGNORECASE,
)


def _clean_message(text: str) -> str:
    """Clean a message for fine-tuning: strip URLs, normalize whitespace."""
    text = _URL_RE.sub("", text).strip()
    # Collapse multiple spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_emoji(text: str) -> str:
    """Remove emoji characters from text to count actual words."""
    return re.sub(rf"[{_EMOJI_CHARS}♥❤💕💗💖💘💝💞💟♡]", "", text).strip()


def _is_low_quality(text: str, min_words: int = 2) -> bool:
    """Check if a message is too short, emoji-heavy, or otherwise low quality."""
    if not text:
        return True
    # Pure emoji / hearts
    if _EMOJI_ONLY_RE.match(text):
        return True
    # Common low-quality single-word/reaction messages
    if _LOW_QUALITY_PATTERNS.match(text.strip()):
        return True
    # Facebook artifacts
    if _FB_ARTIFACT_RE.search(text):
        return True
    # Strip emoji and check actual word count
    stripped = _strip_emoji(text)
    words = stripped.split()
    if len(words) < min_words:
        return True
    # If more than 70% of the text is emoji/special chars, skip
    if len(stripped) < len(text) * 0.3:
        return True
    return False


# ── Main export function ──────────────────────────────────────────────────────

def export_finetune_data(
    json_path: str | Path | None = None,
    output_path: str | Path | None = None,
    years: int = 3,
    language: str = "en",
    min_words: int = 2,
    max_words: int = 200,
    max_turns: int = 1,
    progress_callback=None,
) -> dict:
    """
    Export Facebook messages as fine-tuning data in JSONL format.

    Filters conversations to the last *years* years, keeps only *language*
    messages, removes system messages, and structures as user/assistant pairs.

    Parameters
    ----------
    json_path : path to facebook_messages.json
    output_path : path for the output JSONL file
    years : number of years to look back (default 3)
    language : language code to keep (default "en")
    min_words : minimum words per message (default 2)
    max_words : maximum words for the assistant's reply (default 200).
        Replies longer than this are skipped — this is conversational
        fine-tuning, not long-form memory.
    max_turns : number of exchange turns per training example (default 1).
        1 = single user→assistant pair.
        2+ = multi-turn: up to N user→assistant exchanges per example.
    progress_callback : callable(current, total) or None

    Returns
    -------
    dict with stats: total_messages, filtered_messages, pairs_exported,
    conversations_used, skipped_too_long, output_file
    """
    if json_path is None:
        json_path = DATA_DIR / "facebook" / "facebook_messages.json"
    json_path = Path(json_path)

    if output_path is None:
        output_path = DATA_DIR / "facebook" / "finetune_data.jsonl"
    output_path = Path(output_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Messages file not found: {json_path}")

    # Load messages
    with open(json_path, encoding="utf-8") as f:
        all_messages = json.load(f)

    total_messages = len(all_messages)

    # Detect self name (most frequent sender)
    sender_counts: dict[str, int] = defaultdict(int)
    for m in all_messages:
        sender_counts[m.get("sender_name", "")] += 1
    self_name = max(sender_counts, key=sender_counts.get) if sender_counts else ""

    # Date cutoff
    cutoff = datetime.now() - timedelta(days=years * 365)
    cutoff_str = cutoff.isoformat()

    # Filter messages
    filtered: list[dict] = []
    for m in all_messages:
        # Date filter
        date_str = m.get("date", "")
        if date_str < cutoff_str:
            continue

        # Language filter
        if m.get("language", "en") != language:
            continue

        # System message filter
        text = m.get("text", "").strip()
        if _is_system_message(text):
            continue

        # Clean the text
        cleaned = _clean_message(text)
        if _is_low_quality(cleaned, min_words):
            continue

        filtered.append({**m, "text": cleaned})

    filtered_count = len(filtered)
    print(f"[export_finetune] {total_messages} total → {filtered_count} after filtering "
          f"(last {years}y, lang={language}, min_words={min_words})")

    # Group by conversation, sorted by date
    conv_msgs: dict[str, list[dict]] = defaultdict(list)
    for m in filtered:
        conv_msgs[m.get("conversation", "")].append(m)

    for conv in conv_msgs:
        conv_msgs[conv].sort(key=lambda m: m.get("date", ""))

    # Build user/assistant pairs
    # "user" = other person's message, "assistant" = self's reply
    pairs_exported = 0
    skipped_too_long = 0
    conversations_used = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_convs = len(conv_msgs)

    with open(output_path, "w", encoding="utf-8") as out:
        for ci, (conv, msgs) in enumerate(conv_msgs.items()):
            if len(msgs) < 2:
                continue

            # Build exchange pairs within this conversation
            # Walk through messages looking for other→self sequences
            i = 0
            while i < len(msgs):
                # Find a message from someone else (= "user")
                if msgs[i].get("sender_name") == self_name:
                    i += 1
                    continue

                # Collect up to max_turns of user→assistant exchanges
                turns: list[dict] = []
                j = i

                for _turn in range(max_turns):
                    # Need a "user" message (other person)
                    if j >= len(msgs) or msgs[j].get("sender_name") == self_name:
                        break
                    user_text = msgs[j].get("text", "")
                    j += 1

                    # Collect consecutive messages from the same sender as one turn
                    while j < len(msgs) and msgs[j].get("sender_name") != self_name:
                        user_text += " " + msgs[j].get("text", "")
                        j += 1

                    if j >= len(msgs):
                        break

                    # Need an "assistant" message (self)
                    if msgs[j].get("sender_name") != self_name:
                        break
                    assistant_text = msgs[j].get("text", "")
                    j += 1

                    # Collect consecutive self messages as one turn
                    while j < len(msgs) and msgs[j].get("sender_name") == self_name:
                        assistant_text += " " + msgs[j].get("text", "")
                        j += 1

                    asst_stripped = assistant_text.strip()
                    # Skip if assistant reply is too long
                    if max_words > 0 and len(asst_stripped.split()) > max_words:
                        skipped_too_long += 1
                        break

                    turns.append({"role": "user", "content": user_text.strip()})
                    turns.append({"role": "assistant", "content": asst_stripped})

                if turns and len(turns) >= 2:
                    # Validate all turns have content
                    if all(t["content"] for t in turns):
                        example = {"messages": turns}
                        out.write(json.dumps(example, ensure_ascii=False) + "\n")
                        pairs_exported += 1
                        conversations_used.add(conv)

                i = max(i + 1, j)  # advance past what we consumed

            if progress_callback:
                progress_callback(ci + 1, total_convs)

    stats = {
        "total_messages": total_messages,
        "filtered_messages": filtered_count,
        "pairs_exported": pairs_exported,
        "skipped_too_long": skipped_too_long,
        "conversations_used": len(conversations_used),
        "self_name": self_name,
        "output_file": str(output_path),
        "years": years,
        "language": language,
        "max_words": max_words,
    }

    print(f"[export_finetune] Exported {pairs_exported} training examples "
          f"from {len(conversations_used)} conversations → {output_path}")
    if skipped_too_long:
        print(f"  Skipped {skipped_too_long} pairs with assistant reply > {max_words} words")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export fine-tuning data from Facebook messages")
    parser.add_argument("--input", default=None, help="Path to facebook_messages.json")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--years", type=int, default=3, help="Years to look back")
    parser.add_argument("--language", default="en", help="Language filter")
    parser.add_argument("--min-words", type=int, default=2, help="Min words per message")
    parser.add_argument("--max-words", type=int, default=200, help="Max words for assistant reply")
    parser.add_argument("--max-turns", type=int, default=1, help="Max turns per example")
    args = parser.parse_args()

    stats = export_finetune_data(
        json_path=args.input,
        output_path=args.output,
        years=args.years,
        language=args.language,
        min_words=args.min_words,
        max_words=args.max_words,
        max_turns=args.max_turns,
    )
    print(f"\nStats: {json.dumps(stats, indent=2)}")
