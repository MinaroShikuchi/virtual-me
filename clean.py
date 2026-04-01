import re
import json

def clean_french_chat_text(text):
    """
    Standardize spacing and punctuation for French chat 
    while preserving slang.
    """
    # 1. PROTECT CONTEXT: Placeholder for any existing code or blocks
    # Even if tags aren't there yet, we keep the logic for any markdown code
    protected_blocks = []
    
    def protect(match):
        protected_blocks.append(match.group(0))
        return f" __PROTECTED_{len(protected_blocks)-1}__ "

    # Protect any existing markdown code blocks (if any)
    text = re.sub(r'```[\s\S]*?```', protect, text)

    # 2. CLEANING THE HUMAN TEXT
    
    # Fix commas: no space before, exactly one space after
    text = re.sub(r'\s*,\s*', ', ', text)
    
    # French Punctuation Rules: Space before and after ! ? : ;
    # This makes it much easier for the tokenizer to read
    text = re.sub(r'\s*([!?;:])\s*', r' \1 ', text)
    
    # Fix multi-dots (keeping exactly 3 for ellipses)
    text = re.sub(r'\.{4,}', '...', text)
    
    # Handle apostrophes (don't add spaces around them, e.g., "t'es", "j'ai")
    text = re.sub(r"\s*'\s*", "'", text)

    # Collapse multiple spaces into one
    text = re.sub(r' {2,}', ' ', text)

    # 3. RESTORE PROTECTED BLOCKS
    for i, block in enumerate(protected_blocks):
        text = text.replace(f" __PROTECTED_{i}__ ", block)

    return text.strip()

def process_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_count = 0
    
    # Assuming your data is a list of objects with a "text" or "messages" key
    for entry in data:
        if "text" in entry:
            original = entry["text"]
            entry["text"] = clean_french_chat_text(original)
            if original != entry["text"]:
                cleaned_count += 1
        elif "messages" in entry:
            # If using OpenAI/Unsloth format
            for msg in entry["messages"]:
                if "content" in msg and msg["content"]:
                    msg["content"] = clean_french_chat_text(msg["content"])
            cleaned_count += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Nettoyage terminé !")
    print(f"📝 Entrées modifiées : {cleaned_count}")
    print(f"💾 Sauvegardé dans : {output_file}")

# --- EXECUTION ---
if __name__ == "__main__":
    process_dataset('dataset.json', 'dataset_clean.json')