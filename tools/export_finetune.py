"""
tools/export_finetune.py — Export Facebook messages as fine-tuning data.

Reads the extracted ``facebook_messages.json``, filters by date range,
language, type, and quality, then structures conversations as user/assistant
pairs suitable for LLM fine-tuning.

Messages with ``"type": "system"`` or ``"type": "reaction"`` (set during
extraction in ``extract_facebook.py``) are excluded — only ``"type": "text"``
messages are kept.

**Consecutive messages**: When the same person sends several messages in a
row (e.g. two quick texts before the other person replies), they are merged
into a single turn separated by a space.  This guarantees strict
user → assistant alternation required for fine-tuning.

A **desync guard** skips user→assistant exchanges where the assistant's first
reply timestamp is more than 30 minutes after the user's last message, which
indicates a crossed / out-of-order exchange rather than a genuine reply.

Output format (JSONL):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Where:
    - "user" = the other person's message(s)
    - "assistant" = your (self) reply/replies
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from config import DATA_DIR


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

# Facebook-specific artifacts that slip through system message detection.
# Exposed as a list so the UI can let users edit them.
DEFAULT_ARTIFACT_PATTERNS: list[str] = [
    r"sent a photo",
    r"sent a video",
    r"sent a sticker",
    r"sent a GIF",
    r"sent an attachment",
    r"liked a message",
    r"You sent \$",
    r"You sent a link",
    r"is now connected on Messenger",
    r"Say hi to your new Facebook friend",
    r"waved at you",
    r"This person is unavailable",
    r"Click for audio",
    r"Click for video",
    r"The call ended\.",
    r".+ joined the (?:audio|video) call",
    r".*\breacted\b .+ to your message",
    r"You reacted",
    r"Download file:",
]

def _compile_artifact_re(patterns: list[str] | None = None) -> re.Pattern:
    """Compile artifact patterns into a single regex."""
    pats = patterns if patterns is not None else DEFAULT_ARTIFACT_PATTERNS
    combined = "|".join(f"({p})" for p in pats if p.strip())
    return re.compile(combined, re.IGNORECASE) if combined else re.compile(r"(?!)")  # never-match

_FB_ARTIFACT_RE = _compile_artifact_re()

# Facebook group-management / system sentences that get merged into message
# content when consecutive messages are concatenated.  These are stripped
# inline by _clean_message() so they don't pollute training data.
# Name pattern: 1-6 capitalized words (or single CamelCase username).
_FB_NAME = r"(?:[A-Z]\w+(?:[- ][A-Z]\w+){0,5})"

_FB_SYSTEM_SENTENCE_RE = re.compile(
    _FB_NAME + r"\s+"
    r"(?:"
        r"created the group|"
        r"named the group\b[^.]*|"
        r"changed the group (?:name|photo)|"
        r"added " + _FB_NAME + r" to the group|"
        r"removed " + _FB_NAME + r" from the group|"
        r"left the group|"
        r"set the nickname for " + _FB_NAME + r" to \S+(?:\s+\S+){0,5}|"
        r"set your nickname to \S+(?:\s+\S+){0,5}|"
        r"set the emoji to \S+|"
        r"changed the (?:chat )?theme|"
        r"pinned a message|"
        r"started (?:a |an )?(?:audio |video )?call|"
        r"sent a live location"
    r")"
    r"\.?",
    re.UNICODE,
)


# Facebook reaction annotations: emoji + Name (optional timestamp)
# e.g. "❤Henry Philips (Jul 06, 2023 8:10:32 pm)" or "👍Romain Wllptr"
_FB_REACTION_ANNOTATION_RE = re.compile(
    rf"[{_EMOJI_CHARS}♥❤💕💗💖💘💝💞💟♡👍🎉😂😢😮😡]+"
    r"\s*[A-Z]\w+(?:\s+[A-Z]\w+){0,4}"
    r"(?:\s*\([A-Z][a-z]{2,8}\s+\d{1,2},?\s+\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm|AM|PM)?\))?",
    re.UNICODE,
)


# Facebook file attachment references from e2ee_cutover messages
# e.g. "Download file: your_facebook_activity/messages/e2ee_cutover/.../files/808135204369678.pdf"
_FB_DOWNLOAD_FILE_RE = re.compile(
    r"Download file:\s*\S+",
    re.IGNORECASE,
)


def _clean_message(text: str) -> str:
    """Clean a message for fine-tuning: strip URLs, FB system lines, normalize whitespace."""
    text = _URL_RE.sub("", text).strip()
    # Strip Facebook file download references (e2ee_cutover attachments)
    text = _FB_DOWNLOAD_FILE_RE.sub("", text).strip()
    # Strip Facebook "(edited)" annotation
    text = re.sub(r"\s*\(edited\)\s*$", "", text, flags=re.IGNORECASE).strip()
    # Strip Facebook reaction annotations (emoji + name + optional timestamp)
    text = _FB_REACTION_ANNOTATION_RE.sub(" ", text).strip()
    # Strip Facebook group-management / system sentences embedded in content
    text = _FB_SYSTEM_SENTENCE_RE.sub(" ", text).strip()
    # Collapse multiple spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_emoji(text: str) -> str:
    """Remove emoji characters from text to count actual words."""
    return re.sub(rf"[{_EMOJI_CHARS}♥❤💕💗💖💘💝💞💟♡]", "", text).strip()


def _is_low_quality(text: str, min_words: int = 2,
                    artifact_re: re.Pattern | None = None) -> bool:
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
    _art_re = artifact_re if artifact_re is not None else _FB_ARTIFACT_RE
    if _art_re.search(text):
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
    max_reply_gap_min: int = 30,
    filter_reactions: bool = True,
    system_prompt: str | None = None,
    artifact_patterns: list[str] | None = None,
    progress_callback=None,
) -> dict:
    """
    Export Facebook messages as fine-tuning data in JSONL format.

    Filters conversations to the last *years* years, keeps only *language*
    ``"type": "text"`` messages (skipping ``"system"`` and ``"reaction"``),
    removes low-quality noise, and structures as user/assistant pairs.

    A **desync guard** skips pairs where the time gap between the last user
    message and the first assistant reply exceeds *max_reply_gap_min* minutes.

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
    max_turns : soft limit on exchange turns per training example (default 1).
        1 = single user→assistant pair.
        2+ = multi-turn: up to N user→assistant exchanges per example.
        If the assistant's last reply ends with a question mark, the
        conversation keeps extending beyond this limit so it isn't cut
        off mid-flow (hard-capped at ``max(max_turns * 3, 10)``).
    max_reply_gap_min : maximum minutes between the user's last message and
        the assistant's first reply (default 30).  Pairs exceeding this are
        skipped as likely desync / crossed messages.
    progress_callback : callable(current, total) or None

    Returns
    -------
    dict with stats: total_messages, filtered_messages, pairs_exported,
    conversations_used, skipped_too_long, skipped_desync, output_file
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

    # Compile custom artifact patterns if provided
    _art_re = _compile_artifact_re(artifact_patterns) if artifact_patterns is not None else _FB_ARTIFACT_RE

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

        # Type filter — only keep "text" messages (skip system & reaction)
        if m.get("type", "text") != "text":
            continue
            
        # Ignore reactions natively isolated during extraction
        if filter_reactions and m.get("type", "text") == "reaction":
            continue

        # Clean the text
        text = m.get("text", "").strip()
        cleaned = _clean_message(text)
        if _is_low_quality(cleaned, min_words, artifact_re=_art_re):
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
    skipped_desync = 0
    conversations_used = set()
    max_gap = timedelta(minutes=max_reply_gap_min)
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

                # Collect up to max_turns of user→assistant exchanges.
                # A "turn" is one block of user message(s) followed by one
                # block of assistant message(s).  Consecutive messages from
                # the same sender are kept as separate entries (not merged).
                # If the assistant's last reply ends with a question mark,
                # keep extending beyond max_turns so the conversation isn't
                # cut off mid-flow.  A hard cap prevents runaway loops.
                turns: list[dict] = []
                j = i
                turn_count = 0
                hard_cap = max(max_turns * 3, 10)  # safety limit

                while turn_count < hard_cap:
                    # Past the soft limit — only continue if the last
                    # assistant reply ended with a question
                    if turn_count >= max_turns and turns:
                        last_asst = turns[-1]["content"]
                        if not last_asst.rstrip().endswith("?"):
                            break

                    # Need at least one "user" message (other person)
                    if j >= len(msgs) or msgs[j].get("sender_name") == self_name:
                        break

                    # Collect all consecutive user messages as separate entries
                    user_entries: list[dict] = []
                    while j < len(msgs) and msgs[j].get("sender_name") != self_name:
                        user_entries.append(msgs[j])
                        j += 1

                    if j >= len(msgs):
                        break

                    # Need at least one "assistant" message (self)
                    if msgs[j].get("sender_name") != self_name:
                        break

                    # Desync guard: check time gap between last user msg and
                    # first assistant reply
                    user_date = user_entries[-1].get("date", "")
                    asst_date = msgs[j].get("date", "")
                    if user_date and asst_date:
                        try:
                            dt_user = datetime.fromisoformat(user_date)
                            dt_asst = datetime.fromisoformat(asst_date)
                            if dt_asst - dt_user > max_gap:
                                skipped_desync += 1
                                break
                        except ValueError:
                            pass

                    # Collect all consecutive assistant messages as separate entries
                    asst_entries: list[dict] = []
                    while j < len(msgs) and msgs[j].get("sender_name") == self_name:
                        asst_entries.append(msgs[j])
                        j += 1

                    # Skip if total user or assistant word count is too long
                    total_user_words = sum(
                        len(e.get("text", "").split()) for e in user_entries
                    )
                    total_asst_words = sum(
                        len(e.get("text", "").split()) for e in asst_entries
                    )
                    if max_words > 0 and (total_user_words > max_words or total_asst_words > max_words):
                        skipped_too_long += 1
                        break

                    # Merge consecutive user messages into one turn (strict alternation)
                    user_text = " ".join(
                        ue.get("text", "").strip() for ue in user_entries
                        if ue.get("text", "").strip()
                    )
                    if user_text:
                        turns.append({"role": "user", "content": user_text})

                    # Merge consecutive assistant messages into one turn (strict alternation)
                    asst_text = " ".join(
                        ae.get("text", "").strip() for ae in asst_entries
                        if ae.get("text", "").strip()
                    )
                    if asst_text:
                        turns.append({"role": "assistant", "content": asst_text})

                    turn_count += 1

                if turns and len(turns) >= 2:
                    # Validate: must have at least one user and one assistant with content
                    has_user = any(t["role"] == "user" for t in turns)
                    has_asst = any(t["role"] == "assistant" for t in turns)
                    if has_user and has_asst and all(t["content"] for t in turns):
                        messages = turns
                        if system_prompt:
                            messages = [{"role": "system", "content": system_prompt}] + turns
                        out.write(json.dumps({"conversation": conv, "messages": messages}, ensure_ascii=False) + "\n")
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
        "skipped_desync": skipped_desync,
        "conversations_used": len(conversations_used),
        "self_name": self_name,
        "output_file": str(output_path),
        "years": years,
        "language": language,
        "max_words": max_words,
        "max_reply_gap_min": max_reply_gap_min,
    }

    print(f"[export_finetune] Exported {pairs_exported} training examples "
          f"from {len(conversations_used)} conversations → {output_path}")
    if skipped_too_long:
        print(f"  Skipped {skipped_too_long} pairs with assistant reply > {max_words} words")
    if skipped_desync:
        print(f"  Skipped {skipped_desync} pairs with reply gap > {max_reply_gap_min} min (desync)")

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
    parser.add_argument("--max-reply-gap", type=int, default=30,
                        help="Max minutes between user msg and assistant reply (desync guard)")
    parser.add_argument("--no-filter-reactions", action="store_true", help="Do not filter out reactions")
    parser.add_argument("--system-prompt", default=None, help="System prompt prepended to every training example")
    args = parser.parse_args()

    stats = export_finetune_data(
        json_path=args.input,
        output_path=args.output,
        years=args.years,
        language=args.language,
        min_words=args.min_words,
        max_words=args.max_words,
        max_turns=args.max_turns,
        max_reply_gap_min=args.max_reply_gap,
        filter_reactions=not args.no_filter_reactions,
        system_prompt=args.system_prompt,
    )
    print(f"\nStats: {json.dumps(stats, indent=2)}")
