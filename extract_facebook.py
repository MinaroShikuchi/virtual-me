"""
Facebook HTML Export Message Extractor

Extracts personal messages from Facebook HTML export files and structures them
into JSON format for ChromaDB ingestion.

Output format:
{
    "date": "YYYY-MM-DDTHH:MM:SS",
    "source": "facebook",
    "text": "message content",
    "conversation": "conversation name"
}
"""

import json
import os
import re
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
        print(f"Warning: Failed to parse timestamp '{timestamp_str}': {e}")
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


def is_system_message(text: str) -> bool:
    """
    Check if a message is a system message that should be ignored.
    
    System messages include:
    - "You reacted"
    - "You joined the group"
    - Other automated notifications
    
    Args:
        text: Message text to check
        
    Returns:
        True if message is a system message, False otherwise
    """
    system_patterns = [
        r"^You reacted",
        r"^You joined",
        r"^You left",
        r"^You added",
        r"^You removed",
        r"^You changed",
        r"^You named",
        r"^You are now connected",
        r"^You waved at",
    ]
    
    for pattern in system_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    
    return False


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
        print(f"Error reading file {filepath}: {e}")
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
            print(f"Error processing section in {filepath}: {e}")
            continue
    
    return messages


def extract_all_messages(
    data_dir: str,
    target_user: str,
    output_file: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Extract all messages from Facebook HTML export directory.
    
    Args:
        data_dir: Path to the Facebook export messages directory
        target_user: Name of the user whose messages to extract
        output_file: Optional path to save JSON output
        
    Returns:
        List of all extracted messages
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    all_messages = []
    html_files = list(data_path.rglob("*.html"))
    
    print(f"Found {len(html_files)} HTML files to process")
    
    for i, filepath in enumerate(html_files, 1):
        if i % 50 == 0:
            print(f"Processing file {i}/{len(html_files)}...")
        
        messages = extract_messages_from_html(filepath, target_user)
        all_messages.extend(messages)
    
    # Sort messages by date
    all_messages.sort(key=lambda x: x["date"])
    
    print(f"\nExtracted {len(all_messages)} messages from {target_user}")
    
    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_messages, f, indent=2, ensure_ascii=False)
        
        print(f"Saved messages to {output_file}")
    
    return all_messages


def main():
    """Main entry point for the script."""
    # Configuration
    BASE_DIR = "/home/minaro/Documents/facebook-romainwllprt-2026-02-13-MTCuceph/your_facebook_activity/messages"
    TARGET_USER = "Romain Wllptr"  # The user whose messages to extract
    OUTPUT_FILE = "/home/minaro/Git/virtual-me/facebook_messages.json"
    
    # Facebook message folders to process
    FOLDERS_TO_PROCESS = [
        "inbox",              # Regular conversations
        "archived_threads",   # Archived conversations
        "e2ee_cutover",      # End-to-end encrypted conversations
        "filtered_threads",   # Filtered/spam messages
        "message_requests",   # Message requests from non-friends
    ]
    
    all_messages = []
    
    print("=" * 80)
    print("Facebook Message Extraction - All Folders")
    print("=" * 80)
    print()
    
    # Process each folder
    for folder in FOLDERS_TO_PROCESS:
        folder_path = f"{BASE_DIR}/{folder}"
        
        # Check if folder exists
        from pathlib import Path
        if not Path(folder_path).exists():
            print(f"⚠️  Folder not found, skipping: {folder}")
            continue
        
        print(f"📁 Processing: {folder}")
        print("-" * 80)
        
        # Extract messages from this folder
        folder_messages = extract_all_messages(
            data_dir=folder_path,
            target_user=TARGET_USER,
            output_file=None  # Don't save individual folder files
        )
        
        all_messages.extend(folder_messages)
        print()
    
    # Sort all messages by date
    print("=" * 80)
    print("Sorting and saving all messages...")
    all_messages.sort(key=lambda x: x["date"])
    
    # Save to file
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_messages, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(all_messages)} total messages to {OUTPUT_FILE}")
    
    # Print summary by folder
    print("\n" + "=" * 80)
    print("Summary by source:")
    print("=" * 80)
    
    # Count messages by conversation to show distribution
    from collections import Counter
    conv_counts = Counter(msg["conversation"] for msg in all_messages)
    
    print(f"Total messages: {len(all_messages)}")
    print(f"Unique conversations: {len(conv_counts)}")
    print(f"Date range: {all_messages[0]['date']} to {all_messages[-1]['date']}")
    
    # Print sample messages
    if all_messages:
        print("\n" + "=" * 80)
        print("Sample messages (first 3):")
        print("=" * 80)
        for msg in all_messages[:3]:
            print(json.dumps(msg, indent=2, ensure_ascii=False))
            print("-" * 80)


if __name__ == "__main__":
    main()
