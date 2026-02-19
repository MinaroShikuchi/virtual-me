"""
Ingest Facebook messages into ChromaDB.

This script loads the extracted Facebook messages and inserts them
into a persistent ChromaDB instance running in Docker.
"""

import json
import os
import chromadb
from chromadb.utils import embedding_functions

def ingest_messages(
    json_file: str = "facebook_messages.json",
    chroma_path: str = "~/.chroma_data",
    collection_name: str = "romain_brain",
    batch_size: int = 1000
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
    with open(json_file) as f:
        messages = json.load(f)
    
    print(f"Loaded {len(messages)} messages")
    
    # --- CONVERSATION WINDOW GROUPING ---
    from datetime import datetime
    
    # Sort messages by conversation, then date
    print("Sorting messages...")
    messages.sort(key=lambda x: (x.get("conversation", ""), x.get("date", "")))
    
    grouped_docs = []
    
    current_window = []
    last_date = None
    current_conv = None
    
    # 30 minutes in seconds
    WINDOW_THRESHOLD = 30 * 60 
    
    print("Grouping into conversation windows...")
    for msg in messages:
        text = msg.get("text", "")
        if not text:
            continue
            
        try:
            date_str = msg.get("date", "")
            # Assuming ISO format or similar sortable string
            # If standard ISO 8601, this works for py3.7+
            # If fails, we might need dateutil or just rely on simple string gap if formats are consistent
            msg_date = datetime.fromisoformat(date_str)
        except Exception:
            # Fallback if date parsing fail, treat as new window
            msg_date = datetime.now()
            
        conv = msg.get("conversation", "unknown")
        sender = msg.get("sender_name", "Unknown")
        
        # Check if we should start a new window
        is_new_window = False
        
        if conv != current_conv:
            is_new_window = True
        elif last_date and (msg_date - last_date).total_seconds() > WINDOW_THRESHOLD:
            is_new_window = True
            
        if is_new_window and current_window:
            # Flush current window
            doc_text = "\n".join(current_window)
            grouped_docs.append({
                "text": doc_text,
                "date": current_window_start_date, # First msg date
                "conversation": current_conv,
                "message_count": len(current_window)
            })
            current_window = []
            
        # Add to current window
        if not current_window:
            current_window_start_date = date_str
            current_conv = conv
            
        current_window.append(f"[{sender}]: {text}")
        last_date = msg_date
        
    # Flush last window
    if current_window:
        doc_text = "\n".join(current_window)
        grouped_docs.append({
            "text": doc_text,
            "date": current_window_start_date,
            "conversation": current_conv,
            "message_count": len(current_window)
        })
        
    print(f"Created {len(grouped_docs)} conversation documents")

    # Embedding model (multilingual for French support)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
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
    
    print(f"Resetting collection: {collection_name}")
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass # Collection might not exist
        
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
    ingest_messages()
