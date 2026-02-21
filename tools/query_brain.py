"""
Query the personal brain stored in ChromaDB.

This script demonstrates how to perform semantic search on your
Facebook messages using natural language queries.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.utils import embedding_functions
from config import EMBEDDING_MODEL
from datetime import datetime


def query_brain(
    query: str,
    n_results: int = 5,
    chroma_path: str = "./.chroma_data",
    collection_name: str = "virtual_me_knowledge"
):
    """
    Query the personal brain with semantic search.
    
    Args:
        query: Natural language query
        n_results: Number of results to return
        chroma_path: Path to ChromaDB data
        collection_name: Name of the collection
    """
    # Embedding model (must match the one used for ingestion)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=chroma_path)
    
    # Get collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )
    
    print(f"🧠 Querying brain: '{query}'")
    print(f"   Collection: {collection_name} ({collection.count()} messages)\n")
    
    # Perform semantic search
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    # Display results
    print(f"📝 Top {n_results} results:\n")
    print("=" * 80)
    
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
        # Parse date for better display
        date_obj = datetime.fromisoformat(metadata['date'])
        date_str = date_obj.strftime("%Y-%m-%d %H:%M")
        
        print(f"\n{i}. [{date_str}] in {metadata['conversation']}")
        print(f"   {doc}")
        print("-" * 80)


def main():
    """Run example queries."""
    
    # Example queries
    queries = [
        "What did I say about Paris?",
        "conversations about work or projects",
        "messages about meeting up with friends",
    ]
    
    print("🚀 Personal Brain Query Demo\n")
    
    for query in queries:
        query_brain(query, n_results=3)
        print("\n" + "=" * 80 + "\n")
    
    # Interactive mode
    print("\n💬 Interactive Mode (type 'quit' to exit)\n")
    while True:
        user_query = input("Query: ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
        if user_query:
            query_brain(user_query, n_results=5)
            print()


if __name__ == "__main__":
    main()
