"""
Ingest Facebook messages into ChromaDB.

This script loads the extracted Facebook messages and inserts them
into a persistent ChromaDB instance running in Docker.
"""

import json
import os
import sys

# Ensure project root is on sys.path so `config` is importable when run standalone
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import chromadb
from chromadb.utils import embedding_functions
from config import COLLECTION_NAME, CHROMA_PATH, EMBEDDING_MODEL  # noqa: E402

def ingest_messages(
    json_file: str = "facebook_messages.json",
    chroma_path: str = ".chroma_data",
    collection_name: str = "virtual_me_knowledge",
    batch_size: int = 1000,
    session_gap_seconds: int = 8 * 3600,
    max_msgs_per_doc: int = 150,
    reset: bool = True,
):
    """
    Ingest Facebook messages into ChromaDB.
    
    Args:
        json_file: Path to the JSON file with extracted messages
        chroma_path: Path to store ChromaDB data locally
        collection_name: Name of the collection to create/use
        batch_size: Number of messages to insert per batch
    """
    # Load messages
    print(f"Loading messages from {json_file}...")
    json_path = os.path.abspath(json_file)
    if not os.path.isfile(json_path):
        print(f"\n❌ File not found: {json_path}")
        print("   Please run the extraction step first (Step 1 in the Vector page)")
        print("   or check that the path is correct.")
        sys.exit(1)
    with open(json_path) as f:
        messages = json.load(f)
    
    print(f"Loaded {len(messages)} messages")
    
    # --- CONVERSATION SESSION GROUPING ---
    from datetime import datetime
    from collections import defaultdict
    
    # Group all messages by conversation ID first
    print("Grouping messages by conversation...")
    conv_groups = defaultdict(list)
    for msg in messages:
        if not msg.get("text"):
            continue
        conv_groups[msg.get("conversation", "unknown")].append(msg)
    
    # Sort each conversation's messages by date
    for conv in conv_groups:
        conv_groups[conv].sort(key=lambda x: x.get("date", ""))
    
    grouped_docs = []
    
    # Session gap: 8 hours. If two consecutive messages in the SAME conversation
    # are more than 8 hours apart, they're in a different "session".
    SESSION_GAP_SECONDS = session_gap_seconds
    MAX_MSGS_PER_DOC    = max_msgs_per_doc
    
    print(f"Creating session documents (session gap={SESSION_GAP_SECONDS//3600}h, max={MAX_MSGS_PER_DOC} msgs)...")
    
    for conv_id, msgs in conv_groups.items():
        # Small conversations: keep entirely as one document
        if len(msgs) <= MAX_MSGS_PER_DOC:
            doc_text = "\n".join(
                f"[{m.get('sender_name', 'Unknown')}]: {m.get('text', '')}"
                for m in msgs
            )
            grouped_docs.append({
                "text": doc_text,
                "date": msgs[0].get("date", ""),
                "conversation": conv_id,
                "message_count": len(msgs)
            })
            continue
        
        # Large conversations: split on 8-hour session gaps
        session = []
        session_start_date = None
        last_date = None
        
        for msg in msgs:
            try:
                msg_date = datetime.fromisoformat(msg.get("date", ""))
            except Exception:
                msg_date = datetime.now()
            
            is_new_session = (
                last_date is not None and
                (msg_date - last_date).total_seconds() > SESSION_GAP_SECONDS
            ) or (len(session) >= MAX_MSGS_PER_DOC)
            
            if is_new_session and session:
                doc_text = "\n".join(session)
                grouped_docs.append({
                    "text": doc_text,
                    "date": session_start_date,
                    "conversation": conv_id,
                    "message_count": len(session)
                })
                session = []
                session_start_date = None
            
            if not session:
                session_start_date = msg.get("date", "")
            
            session.append(f"[{msg.get('sender_name', 'Unknown')}]: {msg.get('text', '')}")
            last_date = msg_date
        
        # Flush last session
        if session:
            grouped_docs.append({
                "text": "\n".join(session),
                "date": session_start_date,
                "conversation": conv_id,
                "message_count": len(session)
            })
    
    print(f"Created {len(grouped_docs)} conversation session documents")

    # Embedding model — BAAI/bge-m3 (must match retrieval model)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    # Connect to ChromaDB (persistent local storage)
    chroma_path_expanded = os.path.expanduser(chroma_path)
    print(f"Connecting to ChromaDB at {chroma_path_expanded}...")
    client = chromadb.PersistentClient(path=chroma_path_expanded)
    
    # Get or create collection
    # Note: If schema changed (metadata fields), might want to reset collection or use a new name
    # For this refactor, let's use a new collection name or just overwrite if user is okay?
    # User didn't specify, but safer to use a new name or just append "_v2" if we want to preserve old
    # But usually "refactor" implies replacing. 
    # Let's delete existing documents? Or just add to the same collection (ids will differ).
    # Since documents are totally different format (windows vs single msgs), mixing them is bad.
    # We should probably reset the collection.
    
    print(f"{'Resetting' if reset else 'Reusing'} collection: {collection_name}")
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )
    
    print(f"Using collection: {collection_name}")
    
    # Insert in batches
    total_batches = (len(grouped_docs) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(grouped_docs), batch_size):
        batch_docs = grouped_docs[batch_idx:batch_idx + batch_size]
        
        # Prepare batch data
        documents = [doc["text"] for doc in batch_docs]
        metadatas = [
            {
                "date": doc["date"],
                "conversation": doc["conversation"],
                "message_count": doc["message_count"],
                "source": "facebook_windowed"
            }
            for doc in batch_docs
        ]
        ids = [f"window_{batch_idx + i}" for i in range(len(batch_docs))]
        
        # Insert batch
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        batch_num = (batch_idx // batch_size) + 1
        print(f"Inserted batch {batch_num}/{total_batches} ({len(batch_docs)} windows)")
    
    print(f"\n✅ Brain successfully installed!")
    print(f"   Total conversation windows: {len(grouped_docs)}")
    print(f"   Collection: {collection_name}")
    print(f"   Collection count: {collection.count()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Facebook messages into ChromaDB")
    parser.add_argument("--json-file",    default="facebook_messages.json",
                        help="Path to extracted messages JSON")
    parser.add_argument("--chroma-path",  default=".chroma_data",
                        help="Path to ChromaDB persistent storage")
    parser.add_argument("--collection",   default="virtual_me_knowledge",
                        help="ChromaDB collection name")
    parser.add_argument("--batch-size",   type=int,   default=1000)
    parser.add_argument("--session-gap",  type=int,   default=8*3600,
                        help="Session gap in seconds (default: 8h = 28800)")
    parser.add_argument("--max-msgs",     type=int,   default=150,
                        help="Max messages per document chunk")
    parser.add_argument("--reset",        action="store_true",
                        help="Delete and recreate the collection before ingesting")
    args = parser.parse_args()

    ingest_messages(
        json_file=args.json_file,
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        batch_size=args.batch_size,
        session_gap_seconds=args.session_gap,
        max_msgs_per_doc=args.max_msgs,
        reset=args.reset,
    )
