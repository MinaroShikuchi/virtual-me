"""
Ingest Facebook messages into ChromaDB.

This script loads the extracted Facebook messages and inserts them
into a persistent ChromaDB instance running in Docker.
"""

import json
import os
import sys

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

# Ensure project root is on sys.path so `config` is importable when run standalone
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import chromadb
from chromadb.utils import embedding_functions

# Set PyTorch memory alloc conf to help prevent OOM on 8GB cards
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from config import COLLECTION_NAME, CHROMA_PATH, EMBEDDING_MODEL  # noqa: E402

def ingest_messages(
    json_file: str = "facebook_messages.json",
    chroma_path: str = ".chroma_data",
    collection_name: str = "virtual_me_knowledge",
    batch_size: int = 64,
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
    print(f"Loading messages from {json_file}...", flush=True)
    json_path = os.path.abspath(json_file)
    if not os.path.isfile(json_path):
        print(f"\n❌ File not found: {json_path}", flush=True)
        print("   Please run the extraction step first (Step 1 in the Vector page)", flush=True)
        print("   or check that the path is correct.", flush=True)
        sys.exit(1)
    with open(json_path) as f:
        messages = json.load(f)
    
    print(f"Loaded {len(messages)} messages", flush=True)
    
    # --- CONVERSATION SESSION GROUPING (DYNAMIC TEMPO) ---
    print("Grouping messages by conversation using dynamic tempo...", flush=True)
    df = pd.DataFrame(messages)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['conversation', 'date']).reset_index(drop=True)

    df['gap'] = df.groupby('conversation')['date'].diff().dt.total_seconds() / 60
    
    df['local_tempo'] = df.groupby('conversation')['gap'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    base_threshold = session_gap_seconds / 60
    
    def is_new_session(row):
        if pd.isna(row['gap']): return True 
        if row['gap'] > (row['local_tempo'] * 5) and row['gap'] > 15:
            return True
        if row['gap'] > base_threshold:
            return True
        return False

    print("Calculating session boundaries...")
    tqdm.pandas(desc="Session boundaries", mininterval=0.5, maxinterval=2.0)
    df['new_session_gap'] = df.progress_apply(is_new_session, axis=1)
    df['session_id'] = df.groupby('conversation')['new_session_gap'].cumsum()

    def format_block(group):
        idx = group.index
        conv_id = group['conversation'].iloc[0]
        
        # Pull overlapping context
        overlap = 2
        start_idx = max(0, idx[0] - overlap)
        prev_msgs = df.iloc[start_idx:idx[0]]
        prev_msgs = prev_msgs[prev_msgs['conversation'] == conv_id]
        
        prefix = "\n".join([f"{r['sender_name']}: {r['text']} [CONTEXT]" for _, r in prev_msgs.iterrows()])
        
        current_text = "\n".join([f"{row['sender_name']}: {row['text']}" for _, row in group.iterrows()])
        full_display_text = f"{prefix}\n{current_text}".strip()
        
        return {
            'text': full_display_text,
            'date': group['date'].min().isoformat(),
            'conversation': conv_id,
            'message_count': len(group)
        }

    print("Formatting message blocks...", flush=True)
    groups = [group for _, group in df.groupby(['conversation', 'session_id'])]
    with ThreadPoolExecutor() as executor:
        grouped_docs = list(tqdm(
            executor.map(format_block, groups), 
            total=len(groups), 
            desc="Formatting blocks",
            mininterval=0.5,
            maxinterval=2.0
        ))
        
    print(f"\n--- Grouping Summary ---")
    print(f"Total conversation session documents created: {len(grouped_docs)}")
    print(f"------------------------\n", flush=True)

    print("Saving intermediate grouped data for UI...", flush=True)
    with open("facebook_messages_grouped_ui.json", "w") as f:
        json.append_mode = False # Just ensuring it's not somehow append
        json.dump(grouped_docs, f, indent=2)
    print("✅ Saved facebook_messages_grouped_ui.json\n", flush=True)

    # Embedding model — BAAI/bge-m3 (must match retrieval model)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding device: {device.upper()}", flush=True)

    # Use FP16 if on GPU to significantly reduce VRAM usage
    model_kwargs = {}
    # if device == "cuda":
    #     model_kwargs = {"torch_dtype": torch.float16}
    #     print("Using FP16 (Half Precision) to save VRAM", flush=True)
    
    base_embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL, device=device, model_kwargs=model_kwargs
    )
    
    class ChunkedEmbeddingFunction:
        """Forces PyTorch to calculate embeddings in extremely small VRAM chunks to prevent OOM
        while allowing ChromaDB to write to disk in much larger, efficient batches."""
        def __init__(self, base_func, chunk_size=4):
            self.base_func = base_func
            self.chunk_size = chunk_size
            
        def __call__(self, input):
            res = []
            import torch
            for i in range(0, len(input), self.chunk_size):
                res.extend(self.base_func(input[i:i+self.chunk_size]))
            return res
            
        def name(self) -> str:
            return "sentence_transformer"
            
    embedding_func = ChunkedEmbeddingFunction(base_embedding_func, chunk_size=4)
    # Connect to ChromaDB (persistent local storage)
    chroma_path_expanded = os.path.expanduser(chroma_path)
    print(f"Connecting to ChromaDB at {chroma_path_expanded}...", flush=True)
    client = chromadb.PersistentClient(path=chroma_path_expanded)
    
    # Get or create collection
    # Note: If schema changed (metadata fields), might want to reset collection or use a new name
    # For this refactor, let's use a new collection name or just overwrite if user is okay?
    # User didn't specify, but safer to use a new name or just append "_v2" if we want to preserve old
    # But usually "refactor" implies replacing. 
    # Let's delete existing documents? Or just add to the same collection (ids will differ).
    # Since documents are totally different format (windows vs single msgs), mixing them is bad.
    # We should probably reset the collection.
    
    print(f"{'Resetting' if reset else 'Reusing'} collection: {collection_name}", flush=True)
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )
    
    print(f"Using collection: {collection_name}", flush=True)
    
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
        
        # Log before the slow embedding step
        batch_num = (batch_idx // batch_size) + 1
        print(f"Embedding batch {batch_num}/{total_batches} ({len(batch_docs)} windows)...", flush=True)

        # Upsert batch (this triggers embedding + write — the slow part)
        # Using upsert allows resuming if the script crashes midway (just run without --reset)
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        # Aggressively clear VRAM to prevent creeping fragmentation over 2000+ batches
        if device == "cuda":
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            # Log VRAM usage to identify memory leaks
            alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            cache_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            print(f"[VRAM] Allocated: {alloc_mb:.1f} MB | Cached: {cache_mb:.1f} MB", flush=True)
            
        pct = int(batch_num / total_batches * 100)
        print(f"PROGRESS: {pct}% | {batch_num}/{total_batches}", flush=True)
        print(f"✅ Batch {batch_num}/{total_batches} done", flush=True)
        print(f"Inserted batch {batch_num}/{total_batches} ({len(batch_docs)} windows)", flush=True)
    
    print(f"\n✅ Brain successfully installed!", flush=True)
    print(f"   Total conversation windows: {len(grouped_docs)}", flush=True)
    print(f"   Collection: {collection_name}", flush=True)
    print(f"   Collection count: {collection.count()}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Facebook messages into ChromaDB")
    parser.add_argument("--json-file",    default="facebook_messages.json",
                        help="Path to extracted messages JSON")
    parser.add_argument("--chroma-path",  default=".chroma_data",
                        help="Path to ChromaDB persistent storage")
    parser.add_argument("--collection",   default="virtual_me_knowledge",
                        help="ChromaDB collection name")
    parser.add_argument("--batch-size",   type=int,   default=8)
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
