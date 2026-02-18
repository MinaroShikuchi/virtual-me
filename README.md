# Virtual Me - Personal Data Extraction & Brain

Extract and structure personal data from various sources for ChromaDB ingestion.

## Features

- **Facebook Message Extraction**: Extract messages from Facebook HTML exports
- **ChromaDB Integration**: Persistent vector database for semantic search
- **Production Quality**: Type hints, error handling, modular code

## Setup

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Extract Facebook Messages

Edit `extract_facebook.py` to set your paths:

```python
DATA_DIR = "/path/to/facebook/export/messages/inbox"
TARGET_USER = "Your Name"
OUTPUT_FILE = "facebook_messages.json"
```

Run extraction:

```bash
python extract_facebook.py
```

### 3. Ingest into ChromaDB

```bash
python ingest_facebook_messages.py
```

This creates a local ChromaDB instance in `./chroma_data/` (no Docker required).

### 4. Query Your Brain

**Option A: Simple Semantic Search**

```bash
python query_brain.py
```

Returns the most relevant messages without AI interpretation.

**Option B: RAG with Ollama (Recommended)**

First, install and run Ollama:

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull qwen2.5:7b

# Start Ollama server (in a separate terminal)
ollama serve
```

Then run the RAG query:

```bash
python query_my_history.py
```

This combines semantic search with an LLM to provide natural language answers about your message history.

## Usage

### Simple Semantic Search

```python
import chromadb
from chromadb.utils import embedding_functions

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_data")

# Get collection
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_collection(
    name="romain_brain",
    embedding_function=embedding_func
)

# Query
results = collection.query(
    query_texts=["What did I say about Paris?"],
    n_results=5
)

for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
    print(f"[{metadata['date']}] {doc}")
```

### RAG with Ollama

The `query_my_history.py` script provides natural language Q&A:

```bash
$ python query_my_history.py

🧠 Personal Facebook History RAG System
================================================================================
Model: qwen2.5:7b
Messages: 29448
================================================================================

Example questions:
  • What did I talk about with [person]?
  • When did I last mention Paris?
  • What were my thoughts on work in 2020?
  • Summarize my conversations about projects

💬 Ask about your history (or 'quit'): What did I say about Paris in 2016?

🔍 Searching memories for: 'What did I say about Paris in 2016?'...

📚 Retrieved messages:
--------------------------------------------------------------------------------
1. [2016-08-25] A Paris
2. [2016-09-12] OK met message instant
3. [2015-01-28] Paris 13
--------------------------------------------------------------------------------

🤖 Answer:

Based on your Facebook messages, on August 25, 2016, you mentioned "A Paris"...
```

## Project Structure

```
.
├── extract_facebook.py          # Extract messages from Facebook HTML
├── ingest_facebook_messages.py  # Ingest into ChromaDB
├── query_brain.py               # Simple semantic search
├── query_my_history.py          # RAG with Ollama
├── docker-compose.yml           # Optional Docker setup
├── requirements.txt             # Python dependencies
├── facebook_messages.json       # Extracted messages (generated)
└── chroma_data/                 # ChromaDB persistent storage (generated)
```

## Output Format

Each message is structured as:

```json
{
  "date": "2015-06-18T21:35:16",
  "source": "facebook",
  "text": "message content",
  "conversation": "conversation_name"
}
```
