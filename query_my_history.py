"""
Query your personal Facebook history using RAG with Ollama.
Includes Smart Filtering:
- "With/To [Name]" -> Restricts search to that specific conversation.
- "About [Name]"   -> Searches ALL conversations for mentions of that name.
"""

import chromadb
from chromadb.utils import embedding_functions
import ollama
import json
import re
from pathlib import Path

# --- CONFIGURATION ---
CHROMA_PATH = "./chroma_data"
COLLECTION_NAME = "romain_brain"
OLLAMA_MODEL = "deepseek-r1:14b"
NAME_MAPPING_FILE = "./conversation_names.json"

# --- SETUP ---

def load_mappings():
    """Load mappings and create a reverse lookup (Name -> ID)."""
    id_to_name = {}
    name_to_id = {}
    
    try:
        if Path(NAME_MAPPING_FILE).exists():
            with open(NAME_MAPPING_FILE, 'r', encoding='utf-8') as f:
                id_to_name = json.load(f)
                # Create reverse mapping: lowercase name -> ID
                for cid, name in id_to_name.items():
                    if name:
                        name_to_id[name.lower()] = cid
    except Exception as e:
        print(f"Warning: Could not load name mapping: {e}")
        
    return id_to_name, name_to_id

NAME_MAPPING, REVERSE_NAME_MAPPING = load_mappings()

# Embedding function (Must match ingestion - multilingual for French)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

try:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )
    print(f"✅ Connected to ChromaDB: {collection.count()} messages loaded\n")
except Exception as e:
    print(f"❌ Error connecting to ChromaDB: {e}")
    exit(1)

# --- CORE LOGIC ---

def detect_smart_filter(question):
    """
    Decides whether to apply a strict conversation filter or a global search.
    
    Logic:
    1. Scan for known names (supports partial matching).
    2. If found, check the word immediately BEFORE the name.
    3. If "with", "to", "from" -> Strict Filter (Conversation ID).
    4. Otherwise -> No Filter (Global Semantic Search).
    """
    question_lower = question.lower()
    
    # Try exact match first, then partial match
    matched_name = None
    matched_id = None
    
    # First pass: exact match
    for name, conversation_id in REVERSE_NAME_MAPPING.items():
        if name in question_lower:
            matched_name = name
            matched_id = conversation_id
            break
    
    # Second pass: partial match (e.g., "lois" matches "loisnormand")
    if not matched_name:
        # Normalize the question: remove spaces and lowercase for better matching
        question_normalized = question_lower.replace(' ', '')
        
        for name, conversation_id in REVERSE_NAME_MAPPING.items():
            # Check if the normalized name appears in the normalized question
            name_normalized = name.replace(' ', '')
            if len(name_normalized) >= 3 and name_normalized in question_normalized:
                matched_name = name
                matched_id = conversation_id
                break
            
            # Also try matching individual parts of the name
            name_parts = name.split()
            for part in name_parts:
                if len(part) >= 3 and part in question_lower:
                    matched_name = name
                    matched_id = conversation_id
                    break
            if matched_name:
                break
    
    if matched_name and matched_id:
        # Check for "with/to/from [name]" pattern using Regex
        # Try both full name and partial matches
        patterns = [
            fr"\b(with|to|from|chat)\s+{re.escape(matched_name)}\b",
            fr"\b(with|to|from|chat)\s+\w*{re.escape(matched_name.split()[0][:4])}\w*\b"  # Partial match
        ]
        
        for pattern in patterns:
            if re.search(pattern, question_lower):
                return {"conversation": matched_id}, matched_name, "Strict (Conversation)"
        
        # If name is present but no preposition, allow global search
        return None, matched_name, "Global (Mention)"
    
    return None, None, "None"

def ask_facebook_history(question: str, show_sources: bool = True):
    print(f"\n🔍 Searching memories for: '{question}'...")
    
    # 1. PRE-PROCESSING (Smart Filter)
    where_filter, friend_name, strategy = detect_smart_filter(question)
    
    if strategy == "Strict (Conversation)":
        print(f"   ↳ 🎯 Context: Loading ENTIRE conversation with '{friend_name}'")
    elif strategy == "Global (Mention)":
        print(f"   ↳ 🌎 Context: Global search about '{friend_name}'")
    
    # 2. RETRIEVE
    try:
        if strategy == "Strict (Conversation)" and where_filter:
            # Load the ENTIRE conversation with this person (no semantic search)
            # Get all messages from this conversation, sorted by date
            results = collection.get(
                where=where_filter,
                include=['metadatas', 'documents']
            )
            
            # Sort by date and take the most recent messages (or all if < 100)
            if results['metadatas']:
                # Combine documents and metadata, then sort by date
                messages = list(zip(results['documents'], results['metadatas']))
                messages.sort(key=lambda x: x[1].get('date', ''), reverse=True)
                
                # Take up to 100 most recent messages
                max_messages = 100
                messages = messages[:max_messages]
                
                # Unpack back into separate lists
                results['documents'] = [msg[0] for msg in messages]
                results['metadatas'] = [msg[1] for msg in messages]
                
                print(f"   ↳ 📚 Loaded {len(results['documents'])} messages from this conversation")
            else:
                results['documents'] = []
                results['metadatas'] = []
        else:
            # Use semantic search for global queries
            results = collection.query(
                query_texts=[question],
                n_results=50,  # Increased from 10 to 50 for better context
                where=where_filter  # Only applied if "with/to" was detected
            )
            # Flatten the nested structure from query()
            if results['documents'] and len(results['documents']) > 0:
                results['documents'] = results['documents'][0]
                results['metadatas'] = results['metadatas'][0]
            else:
                results['documents'] = []
                results['metadatas'] = []
                
    except Exception as e:
        print(f"❌ Search Error: {e}")
        return

    if not results['documents']:
        print("\n❌ No relevant messages found.")
        return

    # 3. AUGMENT
    context_messages = []
    documents = results['documents']
    metadatas = results['metadatas']

    for doc, meta in zip(documents, metadatas):
        date = meta.get('date', 'Unknown')
        sender = meta.get('sender_name', 'Unknown Sender')
        conv_id = meta.get('conversation', 'Unknown')
        
        # If we are doing a global search, it's helpful to see WHICH friend the chat was with
        chat_partner = NAME_MAPPING.get(conv_id, "Unknown Chat")
        
        context_messages.append(f"[{date}] [Chat: {chat_partner}] {sender}: {doc}")
    
    context_data = "\n".join(context_messages)
    
    if show_sources:
        print(f"\n📚 Retrieved messages:")
        print("-" * 60)
        for i, msg in enumerate(context_messages, 1):
            print(f"{i}. {msg[:120]}..." if len(msg) > 120 else f"{i}. {msg}")
        print("-" * 60 + "\n")
    
    # 4. GENERATE
    system_prompt = (
        "You are an AI assistant analyzing personal Facebook history. "
        "Use the provided messages to answer the user's question. "
        "Pay attention to who the chat is with (indicated in [Chat: ...]). "
        f"CONTEXT:\n{context_data}"
    )

    print("🤖 Answer:\n")
    try:
        # Use streaming to get response and track tokens
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': question},
            ],
            stream=False,  # Disable streaming to get token counts
            options={
                'num_ctx': 32768  # 32k context window
            }
        )
        
        # Display thinking process if available (DeepSeek-R1 feature)
        message_content = response['message']['content']
        
        # DeepSeek-R1 wraps thinking in <think>...</think> tags
        if '<think>' in message_content and '</think>' in message_content:
            # Extract thinking and answer parts
            think_start = message_content.find('<think>') + 7
            think_end = message_content.find('</think>')
            thinking = message_content[think_start:think_end].strip()
            answer = message_content[think_end + 8:].strip()
            
            # Display thinking process in a collapsed/expandable format
            print("💭 Thinking Process:")
            print("─" * 60)
            print(thinking)
            print("─" * 60)
            print("\n📝 Final Answer:")
            print(answer)
        else:
            # No thinking tags, just print the answer
            print(message_content)
        
        print("\n")
        
        # Display token usage
        if 'prompt_eval_count' in response and 'eval_count' in response:
            prompt_tokens = response.get('prompt_eval_count', 0)
            completion_tokens = response.get('eval_count', 0)
            total_tokens = prompt_tokens + completion_tokens
            
            # Create a visual token usage bar (out of 32k)
            max_tokens = 32768
            bar_width = 40
            filled = int((total_tokens / max_tokens) * bar_width)
            bar = '█' * filled + '░' * (bar_width - filled)
            percentage = (total_tokens / max_tokens) * 100
            
            print(f"📊 Token Usage: [{bar}] {percentage:.1f}%")
            print(f"   Prompt: {prompt_tokens:,} | Response: {completion_tokens:,} | Total: {total_tokens:,} / {max_tokens:,}")
            print()
            
    except Exception as e:
        print(f"❌ Ollama Error: {e}")

# --- MAIN LOOP ---
if __name__ == "__main__":
    print(f"Loaded {len(REVERSE_NAME_MAPPING)} friend names.")
    print("Logic: 'with [Name]' -> One Chat | 'about [Name]' -> All Chats")
    while True:
        user_query = input("💬 Ask (or 'q'): ").strip()
        if user_query.lower() in ['q', 'quit']:
            break
        if user_query:
            ask_facebook_history(user_query)