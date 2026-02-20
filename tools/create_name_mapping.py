#!/usr/bin/env python3
"""
Create a mapping from Facebook conversation IDs to friend names.
This helps the RAG system provide more contextual responses by knowing who you're talking to.
"""

import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Dict

def parse_friends_html(friends_html_path: str) -> Dict[str, str]:
    """
    Parse the Facebook friends HTML file to extract friend names.
    
    Returns:
        Dictionary mapping normalized friend names to actual friend names
    """
    with open(friends_html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    friends = {}
    friends_normalized = {}
    
    # Find all friend sections
    sections = soup.find_all('section', class_='_a6-g')
    
    for section in sections:
        # Get the friend name from the h2 tag
        h2 = section.find('h2', class_='_a6-h')
        if h2:
            friend_name = h2.get_text(strip=True)
            friends[friend_name] = friend_name
            # Create normalized version (lowercase, no spaces, no hyphens, no special chars)
            normalized = re.sub(r'[^a-z]', '', friend_name.lower())
            friends_normalized[normalized] = friend_name
    
    print(f"Found {len(friends)} friends")
    return friends, friends_normalized


def extract_conversation_ids_from_folders(messages_base_dir: str) -> Dict[str, str]:
    """
    Extract conversation IDs from the message folder structure.
    
    Returns:
        Dictionary mapping conversation IDs to folder names
    """
    conversation_ids = {}
    
    folders = [
        "inbox",
        "archived_threads",
        "e2ee_cutover",
        "filtered_threads",
        "message_requests"
    ]
    
    for folder in folders:
        folder_path = Path(messages_base_dir) / folder
        if not folder_path.exists():
            continue
        
        # List all conversation folders
        for conv_folder in folder_path.iterdir():
            if conv_folder.is_dir():
                folder_name = conv_folder.name
                conversation_ids[folder_name] = folder_name
    
    print(f"Found {len(conversation_ids)} conversation folders")
    return conversation_ids


def create_name_mapping(friends: Dict[str, str], friends_normalized: Dict[str, str], conversation_ids: Dict[str, str]) -> Dict[str, str]:
    """
    Create a mapping from conversation IDs to friend names.
    
    Uses exact matching against the friends list for best accuracy.
    """
    mapping = {}
    
    for conv_id in conversation_ids.keys():
        # Format examples:
        # - gabrielplomion_10155425206428687
        # - marie_orier_10155660950233687
        # - gabijavasaityte_10156382417018687 (concatenated)
        
        # Remove the numeric ID at the end
        parts = conv_id.rsplit('_', 1)
        if len(parts) != 2:
            mapping[conv_id] = conv_id
            continue
        
        name_part = parts[0]
        
        # Normalize the conversation name (remove underscores, lowercase, remove special chars)
        normalized_conv = re.sub(r'[^a-z]', '', name_part.lower())
        
        # Try exact match with normalized friends
        if normalized_conv in friends_normalized:
            mapping[conv_id] = friends_normalized[normalized_conv]
            continue
        
        # Try partial matching - check if normalized conv name is contained in any friend name
        best_match = None
        best_match_len = 0
        
        for norm_friend, actual_friend in friends_normalized.items():
            # Check if the conversation name matches the beginning of the friend name
            if norm_friend.startswith(normalized_conv) and len(normalized_conv) > best_match_len:
                best_match = actual_friend
                best_match_len = len(normalized_conv)
            # Or if the friend name starts with the conversation name
            elif normalized_conv.startswith(norm_friend) and len(norm_friend) > best_match_len:
                best_match = actual_friend
                best_match_len = len(norm_friend)
        
        if best_match and best_match_len >= 5:  # Require at least 5 character match
            mapping[conv_id] = best_match
        else:
            # Fall back to simple formatting: replace underscores with spaces and title case
            if '_' in name_part:
                mapping[conv_id] = name_part.replace('_', ' ').title()
            else:
                # For concatenated names without match, just capitalize
                mapping[conv_id] = name_part.capitalize()
    
    print(f"Created {len(mapping)} conversation ID to name mappings")
    return mapping


def main():
    # Paths
    FRIENDS_HTML = "/home/minaro/Documents/facebook-romainwllprt-2026-02-13-MTCuceph/connections/friends/your_friends.html"
    MESSAGES_BASE = "/home/minaro/Documents/facebook-romainwllprt-2026-02-13-MTCuceph/your_facebook_activity/messages"
    OUTPUT_FILE = "/home/minaro/Git/virtual-me/conversation_names.json"
    
    # Parse friends
    friends, friends_normalized = parse_friends_html(FRIENDS_HTML)
    
    # Extract conversation IDs
    conversation_ids = extract_conversation_ids_from_folders(MESSAGES_BASE)
    
    # Create mapping
    mapping = create_name_mapping(friends, friends_normalized, conversation_ids)
    
    # Save to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved conversation name mapping to {OUTPUT_FILE}")
    
    # Print some examples
    print("\nExample mappings:")
    test_cases = [
        'gabijavasaityte_10156382417018687',
        'gabrielplomion_10155425206428687',
        'antoinebussi_10155425206428687',
        'orlaneslamaii_10155671442548687',
        'maisonsdumonde_10157700537768687'
    ]
    
    for conv_id in test_cases:
        if conv_id in mapping:
            print(f"  {conv_id} → {mapping[conv_id]}")


if __name__ == "__main__":
    main()
