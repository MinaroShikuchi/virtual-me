"""
Query your personal Facebook history using RAG with Ollama.
Includes Smart Filtering:
- "With/To [Name]" -> Restricts search to that specific conversation.
- "About [Name]"   -> Searches ALL conversations for mentions of that name.
"""

import chromadb
from chromadb.utils import embedding_functions
import ollama
from neo4j import GraphDatabase
import json
import re
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL

# --- CONFIGURATION ---
CHROMA_PATH = "./.chroma_data"
COLLECTION_NAME = "virtual_me_knowledge"
EPISODIC_COLLECTION_NAME = "episodic_memory"
OLLAMA_HOST = "http://192.168.32.1:11434"
OLLAMA_MODEL = "deepseek-r1:14b"
NAME_MAPPING_FILE = "./conversation_names.json"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "nostalgia")

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

# Embedding function (must match ingestion model)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

# Connect to collections
try:
    chroma_path_expanded = os.path.expanduser(CHROMA_PATH)
    client = chromadb.PersistentClient(path=chroma_path_expanded)
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )
    
    # Try to get episodic memory, but don't fail if it doesn't exist yet
    try:
        episodic_collection = client.get_or_create_collection(
            name=EPISODIC_COLLECTION_NAME,
            embedding_function=embedding_func
        )
        print(f"✅ Connected to ChromaDB:")
        print(f"   - Main Memory: {collection.count()} messages")
        print(f"   - Episodic Memory: {episodic_collection.count()} episodes")
    except Exception:
        episodic_collection = None
        print(f"✅ Connected to ChromaDB: {collection.count()} messages loaded")
        print(f"⚠️  Episodic memory not found. Run 'episodic_memory.py' to generate.")

except Exception as e:
    print(f"❌ Error connecting to ChromaDB: {e}")
    exit(1)

# Connect to Neo4j
neo4j_driver = None
try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    # Verify connection
    with neo4j_driver.session() as session:
        count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
        print(f"✅ Connected to Neo4j: {count} nodes in Knowledge Graph\n")
except Exception as e:
    print(f"⚠️  Could not connect to Neo4j: {e}")
    print(f"    Graph context will be unavailable.\n")



# --- CORE LOGIC ---

def query_graph_context(person_name):
    """Query Neo4j for social scale context about a person."""
    if not neo4j_driver or not person_name:
        return ""
    
    try:
        with neo4j_driver.session() as session:
            # Stats: Message count and avg sentiment
            query = """
                MATCH (p:Person)
                WHERE p.normalized_name CONTAINS toLower($name)
                OPTIONAL MATCH (p)-[:SENT]->(m:Message)
                RETURN p.name as name, count(m) as msg_count, avg(m.polarity) as sentiment
                LIMIT 1
            """
            result = session.run(query, name=person_name).single()
            
            if result and result["name"]:
                name = result["name"]
                count = result["msg_count"]
                sentiment = result["sentiment"]
                
                # Interpret sentiment
                mood = "Neutral"
                if sentiment > 0.1: mood = "Positive"
                if sentiment > 0.3: mood = "Very Positive"
                if sentiment < -0.1: mood = "Negative"
                if sentiment < -0.3: mood = "Very Negative"
                
                return f"[Graph Info] {name}: {count} messages sent. Overall Vibe: {mood} ({sentiment:.2f})"
    except Exception as e:
        print(f"⚠️  Graph Query Error: {e}")
    
    return ""

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
        # Check for explicitly "about [name]" -> Global Search
        # This allows searching across ALL conversations for mentions of this person
        if re.search(fr"\b(about|mentioning)\s+{re.escape(matched_name)}\b", question_lower):
            return None, matched_name, "Global (Mention)"
            
        # Default behavior: If a name is mentioned, assume we want to query that conversation
        # This matches the user request: "retrieve the associated conversation"
        return {"conversation": matched_id}, matched_name, "Strict (Conversation)"
    
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
    context_messages = []
    context_episodes = []
    context_graph = ""
    
    # C. Retrieve Graph Context (Social Stats)
    if friend_name:
        context_graph = query_graph_context(friend_name)
        if context_graph:
            print(f"   ↳ 🕸️  Graph: Found social context for '{friend_name}'")

    try:
        # A. Retrieve Episodes (High-level memory)
        # We always query episodic memory for global context, unless strict conversation filter is active
        # (Though even then, maybe relevant?)
        if episodic_collection:
            # For now, simplistic query - just use the question
            ep_results = episodic_collection.query(
                query_texts=[question],
                n_results=5
            )
            
            if ep_results['documents'] and len(ep_results['documents']) > 0:
                ep_docs = ep_results['documents'][0]
                ep_metas = ep_results['metadatas'][0]
                
                for doc, meta in zip(ep_docs, ep_metas):
                    date = meta.get('date', 'Unknown')
                    emotion = meta.get('emotion', '')
                    if emotion and emotion != 'neutral':
                        doc += f" (Emotion: {emotion})"
                    context_episodes.append(f"[{date}] {doc}")
                    
        # B. Retrieve Messages (Granular memory)
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

    if not results['documents'] and not context_episodes:
        print("\n❌ No relevant messages found.")
        return

    # 3. AUGMENT
    documents = results['documents']
    metadatas = results['metadatas']

    for doc, meta in zip(documents, metadatas):
        date = meta.get('date', 'Unknown')
        sender = meta.get('sender_name', 'Unknown Sender')
        conv_id = meta.get('conversation', 'Unknown')
        
        # If we are doing a global search, it's helpful to see WHICH friend the chat was with
        chat_partner = NAME_MAPPING.get(conv_id, "Unknown Chat")
        
        context_messages.append(f"[{date}] [Chat: {chat_partner}] {sender}: {doc}")
    
    # Combine episodes, graph, and messages
    context_data = ""
    
    if context_graph:
        context_data += f"{context_graph}\n\n"
    
    if context_episodes:
        context_data += "=== SIGNIFICANT LIFE EVENTS (EPISODIC MEMORY) ===\n"
        context_data += "\n".join(context_episodes)
        context_data += "\n\n"
        
    context_data += "=== CHAT LOGS (DETAILED MEMORY) ===\n"
    context_data += "\n".join(context_messages)
    
    if show_sources:
        if context_episodes:
            print(f"\n🧠 Retrieved Episodes:")
            print("-" * 60)
            for i, ep in enumerate(context_episodes, 1):
                print(f"{i}. {ep}")
            print("-" * 60)
            
        print(f"\n📚 Retrieved Chat Messages:")
        print("-" * 60)
        for i, msg in enumerate(context_messages[:10], 1): # Show first 10
            print(f"{i}. {msg[:120]}..." if len(msg) > 120 else f"{i}. {msg}")
        if len(context_messages) > 10:
            print(f"... and {len(context_messages) - 10} more.")
        print("-" * 60 + "\n")
    
    # 4. GENERATE
    system_prompt = (
        "You are an AI assistant analyzing personal Facebook history. "
        "You recall both significant life events (Episodic Memory) and detailed chat logs. "
        "Use the provided messages to answer the user's question. "
        "Pay attention to dates and who the chat is with. "
        f"CONTEXT:\n{context_data}"
    )

    print("🤖 Answer:\n")
    try:
        # ... (rest of generation logic)
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.chat(
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
        # ... (rest of print logic)
        
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