"""
Ingest Facebook messages into ChromaDB.

This script loads the extracted Facebook messages and inserts them
into a persistent ChromaDB instance running in Docker.
"""

import json
import chromadb
from chromadb.utils import embedding_functions

def ingest_messages(
    json_file: str = "facebook_messages.json",
    chroma_path: str = "./chroma_data",
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
    
    # Embedding model (multilingual for French support)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Connect to ChromaDB (persistent local storage)
    print(f"Connecting to ChromaDB at {chroma_path}...")
    client = chromadb.PersistentClient(path=chroma_path)
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )
    
    print(f"Using collection: {collection_name}")
    
    # Insert in batches for performance
    total_batches = (len(messages) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(messages), batch_size):
        batch_messages = messages[batch_idx:batch_idx + batch_size]
        
        # Prepare batch data
        documents = [msg["text"] for msg in batch_messages]
        metadatas = [
            {
                "date": msg["date"],
                "sender_name": msg.get("sender_name", "Unknown"),
                "source": msg["source"],
                "conversation": msg["conversation"]
            }
            for msg in batch_messages
        ]
        ids = [f"fb_{batch_idx + i}" for i in range(len(batch_messages))]
        
        # Insert batch
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        batch_num = (batch_idx // batch_size) + 1
        print(f"Inserted batch {batch_num}/{total_batches} ({len(batch_messages)} messages)")
    
    print(f"\n✅ Brain successfully installed!")
    print(f"   Total messages: {len(messages)}")
    print(f"   Collection: {collection_name}")
    print(f"   Collection count: {collection.count()}")


if __name__ == "__main__":
    ingest_messages()
