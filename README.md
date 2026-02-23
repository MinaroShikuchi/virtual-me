# Virtual Me — Personal AI Brain

A full-stack **Reflex** application that ingests your personal data (Facebook messages, Spotify history, Google locations, LinkedIn, Strava) into a vector database and knowledge graph, then lets you chat with an AI that answers as *you* — using RAG retrieval, an Inner Deliberation Committee of psychological personas, and a Neo4j-backed semantic memory.

## Features

- **RAG Chat** — Hybrid semantic + BM25 search over your memories, with Ollama LLM integration and thinking-token support
- **Inner Deliberation Committee** — Multiple IFS-inspired personas (The Self, The Protector, The Inner Child, etc.) deliberate before synthesizing a final answer
- **Knowledge Graph** — Neo4j-backed entity extraction and relationship mapping from your data
- **Dashboard** — Overview of all data sources, ChromaDB stats, and Neo4j graph statistics
- **Vector Store** — Ingest and manage Facebook messages in ChromaDB with windowed chunking
- **Entity Browser** — Search and explore entities in the knowledge graph
- **RAG Explorer** — Debug and inspect the retrieval pipeline with metadata filters
- **Settings** — Configurable LLM model, context window, embedding model, RAG parameters, and Neo4j connection

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** (required by Reflex for the frontend build)
- **Ollama** running locally (or remote) with at least one model pulled
- **Neo4j** (optional, for knowledge graph features)

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-user/virtual-me.git
cd virtual-me
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file (loaded automatically by Reflex):

```env
# Ollama
OLLAMA_HOST=http://localhost:11434

# Neo4j (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Embedding model (default: BAAI/bge-m3)
# EMBEDDING_MODEL=BAAI/bge-m3

# Knowledge graph self-identity node name
# SELF_NAME=ME
```

### 3. Start backend services

```bash
# Start Neo4j and ChromaDB (optional — app works without them)
docker compose up -d

# Pull an Ollama model
ollama pull qwen2.5:7b
```

### 4. Run the app

```bash
reflex run
```

The app will be available at **http://localhost:3000**.

For production:

```bash
reflex run --env prod
```

## Project Structure

```
.
├── config.py                    # Central configuration and data-source registry
├── rxconfig.py                  # Reflex framework configuration
├── docker-compose.yml           # Neo4j + ChromaDB containers
├── requirements.txt             # Python dependencies
├── conversation_names.json      # Facebook conversation ID → name mapping
│
├── virtual_me/                  # Reflex application
│   ├── virtual_me.py            # App entry point and routing
│   ├── components/
│   │   ├── layout.py            # Sidebar, nav links, status badges
│   │   └── settings_dialog.py   # Settings dialog with draft/commit pattern
│   ├── pages/
│   │   ├── chat.py              # Chat page (streaming, deliberation, info bar)
│   │   ├── dashboard.py         # Dashboard page (stats, charts)
│   │   ├── entity_browser.py    # Entity browser page
│   │   ├── graph.py             # Knowledge graph extractor page
│   │   ├── rag_explorer.py      # RAG debug/explorer page
│   │   └── vector.py            # Vector store ingestor page
│   └── state/
│       ├── app_state.py         # Root state (settings, connections)
│       ├── chat_state.py        # Chat state (messages, streaming)
│       ├── dashboard_state.py   # Dashboard state (stats)
│       ├── entity_browser_state.py
│       ├── graph_state.py       # Graph extractor state
│       ├── rag_explorer_state.py
│       └── vector_state.py      # Vector ingestor state
│
├── rag/                         # RAG pipeline (framework-agnostic)
│   ├── retrieval.py             # Hybrid search (semantic + BM25 + RRF + rerank)
│   ├── llm.py                   # Ollama LLM calls, deliberation, thinking tokens
│   └── graph_retrieval.py       # Neo4j fact retrieval for RAG context
│
├── services/                    # Singleton service layer (replaces Streamlit caching)
│   ├── bm25_service.py          # BM25 keyword index
│   ├── chroma_service.py        # ChromaDB client and collections
│   ├── embedding_service.py     # Sentence-transformer embedding function
│   ├── mapping_service.py       # Conversation name mappings
│   └── reranker_service.py      # Cross-encoder reranker
│
├── graph/                       # Knowledge graph layer
│   ├── neo4j_client.py          # Neo4j driver wrapper (MERGE, stats, search)
│   └── constants.py             # Entity labels, relationship types, colors
│
├── tools/                       # CLI data extractors and ingestors
│   ├── ingest_facebook_messages.py
│   ├── build_knowledge_graph.py
│   ├── create_name_mapping.py
│   ├── episodic_memory.py
│   └── extractors/              # Per-source extractors
│       ├── facebook_messages.py
│       ├── facebook_contacts.py
│       ├── google_timeline.py
│       ├── linkedin_connections.py
│       ├── linkedin_education.py
│       ├── linkedin_positions.py
│       ├── spotify.py
│       └── strava.py
│
├── assets/                      # Static assets (favicon, etc.)
├── data/                        # Data exports (gitignored)
└── plans/                       # Migration planning docs
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Reflex Frontend                 │
│  (React, compiled from Python component tree)    │
├─────────────────────────────────────────────────┤
│              Reflex State Classes                │
│  AppState → ChatState, DashboardState, etc.      │
├──────────┬──────────┬───────────────────────────┤
│ services/│   rag/   │        graph/             │
│ (singletons)│ (pipeline)│  (Neo4j client)       │
├──────────┴──────────┴───────────────────────────┤
│  ChromaDB    │    Ollama    │      Neo4j         │
└──────────────┴──────────────┴────────────────────┘
```

## Data Sources

| Source | Type | Description |
|--------|------|-------------|
| Facebook Messages | Chat history | Windowed chunks from Messenger export |
| Google Locations | Location history | Places from Google Takeout |
| Spotify | Listening history | Extended streaming history |
| LinkedIn | Professional | Connections, positions, education |
| Strava | Fitness | Activity data |

## Configuration

All defaults are in [`config.py`](config.py). Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `DEFAULT_MODEL` | `qwen2.5:7b` | Primary Ollama model |
| `DEFAULT_INTENT_MODEL` | `llama3.2:3b` | Intent analysis model |
| `DEFAULT_OLLAMA` | `http://localhost:11434` | Ollama API endpoint |
| `DEFAULT_CTX` | `32768` | Context window size |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Sentence-transformer model |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |

All settings can be changed at runtime via the Settings dialog in the app.

## License

Private project.
