"""
Facebook HTML Export Message Extractor

Extracts personal messages from Facebook HTML export files and structures them
into JSON format for ChromaDB ingestion.

Output format:
{
    "date": "YYYY-MM-DDTHH:MM:SS",
    "source": "facebook",
    "text": "message content",
    "conversation": "conversation name",
    "language": "en"
}

Language is detected at the conversation level during extraction using
``langdetect``.  Each conversation's longest messages are sampled and
majority-voted to assign ``"en"`` or ``"fr"`` to every message in that
conversation.
"""

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup


def parse_facebook_timestamp(timestamp_str: str) -> Optional[str]:
    """
    Convert Facebook timestamp to ISO 8601 format.
    
    Facebook format examples:
    - "Jun 18, 2015 9:35:16 pm"
    - "May 28, 2015 2:50:06 pm"
    - "Feb 07, 2015 10:49:17 pm"
    
    Args:
        timestamp_str: Raw timestamp string from Facebook HTML
        
    Returns:
        ISO 8601 formatted timestamp string or None if parsing fails
    """
    try:
        # Parse the Facebook timestamp format
        dt = datetime.strptime(timestamp_str.strip(), "%b %d, %Y %I:%M:%S %p")
        # Convert to ISO 8601 format
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except (ValueError, AttributeError) as e:
        print(f"Warning: Failed to parse timestamp '{timestamp_str}': {e}", flush=True)
        return None


def extract_conversation_name(filepath: Path) -> str:
    """
    Extract conversation name from the file path.
    
    Args:
        filepath: Path to the HTML file
        
    Returns:
        Conversation name (parent directory name)
    """
    return filepath.parent.name


# Pre-compiled regex for system message detection.  Covers all known
# Facebook Messenger automated notifications (English locale export).
_SYSTEM_RE = re.compile(
    r"^(?:"
    # ── "You …" actions ──
    r"You reacted"
    r"|You joined"
    r"|You left"
    r"|You added"
    r"|You removed"
    r"|You changed"
    r"|You named"
    r"|You are now connected"
    r"|You waved at"
    r"|You can now message and call"
    r"|You started a (?:video )?call"
    r"|You called"
    r"|You missed a (?:video )?call"
    r"|You sent an? "
    r"|You pinned a message"
    r"|You unpinned a message"
    r"|You created the group"
    r"|You set the (?:quick reaction|nickname)"
    r"|You set your nickname"
    r"|You turned (?:on|off)"
    r"|You accepted"
    r"|You declined"
    r"|You voted"
    r"|You took a screenshot"
    # ── Third-person / impersonal ──
    r"|This message was unsent"
    r"|This poll is no longer available"
    r"|Audio call"
    r"|Video call"
    r"|Missed (?:video )?call"
    r"|Say hi to"
    r"|Liked a message"
    r")",
    re.IGNORECASE,
)


def is_system_message(text: str) -> bool:
    """
    Check if a message is a system message that should be ignored.

    Covers Facebook Messenger automated notifications such as call logs,
    attachment placeholders, unsent messages, polls, reactions, group
    management, and connection announcements.

    Args:
        text: Message text to check

    Returns:
        True if message is a system message, False otherwise
    """
    return bool(_SYSTEM_RE.match(text))


def extract_messages_from_html(
    filepath: Path,
    target_user: str
) -> List[Dict[str, str]]:
    """
    Extract messages from a single Facebook HTML export file.
    
    Args:
        filepath: Path to the HTML file
        target_user: Name of the user whose messages to extract
        
    Returns:
        List of message dictionaries
    """
    messages = []
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
    except (IOError, UnicodeDecodeError) as e:
        print(f"Error reading file {filepath}: {e}", flush=True)
        return messages
    
    conversation_name = extract_conversation_name(filepath)
    
    # Find all message sections
    # Structure: <section class="_a6-g" aria-labelledby="...">
    #   <h2 class="_2ph_ _a6-h _a6-i">Sender Name</h2>
    #   <div class="_2ph_ _a6-p">
    #     <div><div></div><div>Message content</div>...
    #   </div>
    #   <footer class="_3-94 _a6-o">
    #     <div class="_a72d">Timestamp</div>
    #   </footer>
    # </section>
    
    sections = soup.find_all("section", class_="_a6-g")
    
    for section in sections:
        try:
            # Extract sender name from h2 header
            sender_header = section.find("h2", class_="_a6-h")
            if not sender_header:
                continue
            
            sender = sender_header.get_text(strip=True)
            
            # Extract timestamp from footer
            footer = section.find("footer", class_="_a6-o")
            if not footer:
                continue
            
            timestamp_div = footer.find("div", class_="_a72d")
            if not timestamp_div:
                continue
            
            timestamp_str = timestamp_div.get_text(strip=True)
            iso_timestamp = parse_facebook_timestamp(timestamp_str)
            
            if not iso_timestamp:
                continue
            
            # Extract message content
            content_div = section.find("div", class_="_a6-p")
            if not content_div:
                continue
            
            # The message text is in nested divs
            # Structure: <div class="_a6-p"><div><div></div><div>TEXT</div>...
            nested_divs = content_div.find_all("div", recursive=True)
            
            # Extract text from all divs, filtering out empty ones
            text_parts = []
            for div in nested_divs:
                # Get direct text content only (not from nested elements)
                text = div.get_text(separator=" ", strip=True)
                
                # Skip empty text and IP addresses (metadata)
                if text and not text.startswith("IP Address:"):
                    # Avoid duplicates by checking if this text is already captured
                    if text not in text_parts:
                        text_parts.append(text)
            
            # Join all text parts
            message_text = " ".join(text_parts).strip()
            
            # Skip empty messages
            if not message_text:
                continue
            
            # Skip system messages
            if is_system_message(message_text):
                continue
            
            # Create message object
            message = {
                "date": iso_timestamp,
                "sender_name": sender,
                "source": "facebook",
                "text": message_text,
                "conversation": conversation_name
            }
            
            messages.append(message)
            
        except Exception as e:
            print(f"Error processing section in {filepath}: {e}", flush=True)
            continue
    
    return messages


# Regex to strip URLs from message text before language detection.
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

# Patterns that indicate a message is a Facebook export artefact or link
# preview rather than actual user prose.  These are skipped during language
# detection because they are always in English regardless of conversation
# language.
_LANGDETECT_SKIP_RE = re.compile(
    r"^(?:"
    r"Download file:"
    r"|magnet:\?"
    r"|Liked a message"
    r")",
    re.IGNORECASE,
)


def _clean_for_langdetect(text: str) -> str:
    """Strip URLs and collapse whitespace so ``langdetect`` sees only prose."""
    cleaned = _URL_RE.sub("", text).strip()
    # Collapse multiple spaces left by URL removal
    return re.sub(r"\s{2,}", " ", cleaned)


def _detect_conversation_language(
    texts: List[str],
    sample_size: int = 30,
    min_chars: int = 30,
) -> str:
    """
    Detect the language of a single conversation by majority-voting sampled
    messages.

    1. Filter out system messages (English export artefacts).
    2. Strip URLs from every text (they confuse ``langdetect``).
    3. Keep only texts with ≥ *min_chars* of actual prose.
    4. Sort by length (longest first) and take up to *sample_size*.
    5. Run ``langdetect.detect()`` on each, majority-vote ``"en"`` vs ``"fr"``.
       Ties → ``"en"``.

    Args:
        texts: All message texts belonging to one conversation.
        sample_size: Max messages to sample (longest first, after cleaning).
        min_chars: Minimum character count after URL removal to consider a
            message for sampling.

    Returns:
        ``"en"`` or ``"fr"``
    """
    from langdetect import detect, LangDetectException
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0  # deterministic results

    if not texts:
        return "en"

    # Filter system messages, export artefacts, and clean texts
    cleaned = []
    for t in texts:
        if is_system_message(t) or _LANGDETECT_SKIP_RE.match(t):
            continue
        c = _clean_for_langdetect(t)
        if len(c) >= min_chars:
            cleaned.append(c)

    if not cleaned:
        return "en"

    # Pick the longest cleaned messages for more reliable detection
    sorted_texts = sorted(cleaned, key=len, reverse=True)[:sample_size]

    votes: Dict[str, int] = {"en": 0, "fr": 0}
    for text in sorted_texts:
        try:
            lang = detect(text)
            key = "fr" if lang == "fr" else "en"
            votes[key] += 1
        except LangDetectException:
            votes["en"] += 1

    return "fr" if votes.get("fr", 0) > votes.get("en", 0) else "en"


def _assign_languages(
    messages: List[Dict[str, str]],
    sample_size: int = 30,
) -> None:
    """
    Detect language per conversation and add a ``"language"`` field to every
    message dict **in place**.

    Groups messages by ``conversation``, detects language for each group,
    then stamps every message with the result.

    Args:
        messages: List of message dicts (mutated in place).
        sample_size: Passed to ``_detect_conversation_language`` (default 30).
    """
    # Group texts by conversation
    conv_texts: Dict[str, List[str]] = {}
    conv_indices: Dict[str, List[int]] = {}
    for i, m in enumerate(messages):
        conv = m.get("conversation", "unknown")
        conv_texts.setdefault(conv, []).append(m.get("text", ""))
        conv_indices.setdefault(conv, []).append(i)

    total_convs = len(conv_texts)
    en_convs = 0
    fr_convs = 0

    for idx, (conv_id, texts) in enumerate(conv_texts.items(), 1):
        lang = _detect_conversation_language(texts, sample_size=sample_size)
        if lang == "fr":
            fr_convs += 1
        else:
            en_convs += 1

        # Stamp every message in this conversation
        for msg_idx in conv_indices[conv_id]:
            messages[msg_idx]["language"] = lang

        if idx % 200 == 0 or idx == total_convs:
            print(
                f"  Language detection: {idx:,}/{total_convs:,} conversations "
                f"({en_convs} EN / {fr_convs} FR)",
                flush=True,
            )

    print(
        f"  🌍 Language detection complete: "
        f"{en_convs} EN / {fr_convs} FR conversations",
        flush=True,
    )


def extract_all_messages(
    data_dir: str,
    target_user: str,
    output_file: Optional[str] = None,
    detect_languages: bool = True,
) -> List[Dict[str, str]]:
    """
    Extract all messages from Facebook HTML export directory.

    Args:
        data_dir: Path to the Facebook export messages directory
        target_user: Name of the user whose messages to extract
        output_file: Optional path to save JSON output
        detect_languages: If True, detect language per conversation and add
            a ``"language"`` field to every message (default True).

    Returns:
        List of all extracted messages (each dict includes ``"language"``
        when *detect_languages* is True).
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_messages = []
    html_files = list(data_path.rglob("*.html"))

    print(f"Found {len(html_files)} HTML files to process", flush=True)

    for i, filepath in enumerate(html_files, 1):
        if i % 50 == 0:
            print(f"Processing file {i}/{len(html_files)}...", flush=True)

        messages = extract_messages_from_html(filepath, target_user)
        all_messages.extend(messages)

    # Sort messages by date
    all_messages.sort(key=lambda x: x["date"])

    print(f"\nExtracted {len(all_messages)} messages from {target_user}", flush=True)

    # Detect language per conversation
    if detect_languages and all_messages:
        print("\nDetecting conversation languages…", flush=True)
        _assign_languages(all_messages)

    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_messages, f, indent=2, ensure_ascii=False)

        print(f"Saved messages to {output_file}", flush=True)

    return all_messages


def main():
    """Main entry point — supports both CLI args and hardcoded defaults."""
    parser = argparse.ArgumentParser(
        description="Extract Facebook HTML messages to JSON",
    )
    parser.add_argument(
        "--input", "-i",
        dest="input_dir",
        default=None,
        help="Path to the Facebook export folder (containing messages/inbox/…)",
    )
    parser.add_argument(
        "--output", "-o",
        dest="output_file",
        default=None,
        help="Path to write the output JSON file",
    )
    parser.add_argument(
        "--user", "-u",
        dest="target_user",
        default="Romain Wllptr",
        help="Name of the user whose messages to extract (default: Romain Wllptr)",
    )
    parser.add_argument(
        "--no-language",
        action="store_true",
        help="Skip conversation-level language detection",
    )
    args = parser.parse_args()

    # If CLI args provided, use simple single-directory mode
    if args.input_dir and args.output_file:
        extract_all_messages(
            data_dir=args.input_dir,
            target_user=args.target_user,
            output_file=args.output_file,
            detect_languages=not args.no_language,
        )
        return

    # ── Legacy hardcoded multi-folder mode ────────────────────────────────
    BASE_DIR = "/home/minaro/Documents/facebook-romainwllprt-2026-02-13-MTCuceph/your_facebook_activity/messages"
    TARGET_USER = args.target_user
    OUTPUT_FILE = args.output_file or "/home/minaro/Git/virtual-me/facebook_messages.json"

    FOLDERS_TO_PROCESS = [
        "inbox",
        "archived_threads",
        "e2ee_cutover",
        "filtered_threads",
        "message_requests",
    ]

    all_messages = []

    print("=" * 80, flush=True)
    print("Facebook Message Extraction - All Folders", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)

    for folder in FOLDERS_TO_PROCESS:
        folder_path = f"{BASE_DIR}/{folder}"

        if not Path(folder_path).exists():
            print(f"⚠️  Folder not found, skipping: {folder}", flush=True)
            continue

        print(f"📁 Processing: {folder}", flush=True)
        print("-" * 80, flush=True)

        folder_messages = extract_all_messages(
            data_dir=folder_path,
            target_user=TARGET_USER,
            output_file=None,
            detect_languages=False,  # detect once after merging all folders
        )

        all_messages.extend(folder_messages)
        print(flush=True)

    # Sort all messages by date
    print("=" * 80, flush=True)
    print("Sorting and saving all messages...", flush=True)
    all_messages.sort(key=lambda x: x["date"])

    # Detect languages across all conversations
    if not args.no_language and all_messages:
        print("\nDetecting conversation languages…", flush=True)
        _assign_languages(all_messages)

    # Save to file
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_messages, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(all_messages)} total messages to {OUTPUT_FILE}", flush=True)

    # Print summary
    print("\n" + "=" * 80, flush=True)
    print("Summary:", flush=True)
    print("=" * 80, flush=True)

    conv_counts = Counter(msg["conversation"] for msg in all_messages)

    print(f"Total messages: {len(all_messages)}", flush=True)
    print(f"Unique conversations: {len(conv_counts)}", flush=True)
    if all_messages:
        print(f"Date range: {all_messages[0]['date']} to {all_messages[-1]['date']}", flush=True)

        # Language summary
        lang_counts = Counter(msg.get("language", "?") for msg in all_messages)
        print(f"Languages: {dict(lang_counts)}", flush=True)

    # Print sample messages
    if all_messages:
        print("\n" + "=" * 80, flush=True)
        print("Sample messages (first 3):", flush=True)
        print("=" * 80, flush=True)
        for msg in all_messages[:3]:
            print(json.dumps(msg, indent=2, ensure_ascii=False), flush=True)
            print("-" * 80, flush=True)


if __name__ == "__main__":
    main()
