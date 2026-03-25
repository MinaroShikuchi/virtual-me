# Virtual Me — CLAUDE.md

## What this project is

A personal AI "digital twin" built on top of your own data (Facebook messages, Spotify, Steam, Google locations, Strava, etc.). It ingests personal data into ChromaDB (vector store) and Neo4j (knowledge graph), then exposes a Streamlit UI for chat, RAG exploration, graph browsing, persona simulation, and LoRA fine-tuning.

## How to run

```bash
source .venv/bin/activate
streamlit run app.py
```

Requires Ollama running locally at `http://localhost:11434`.

## Project layout

```
app.py                  # Streamlit entry point, page routing
config.py               # All constants: paths, model defaults, SOURCES registry, IDENTITIES
requirements.txt

rag/
  resources.py          # @st.cache_resource loaders (ChromaDB, BM25, embeddings, reranker)
  rag_retrieval.py      # Hybrid search: semantic + BM25 + RRF + rerank
  llm.py                # Ollama LLM call (supports thinking tokens)
  graph_retrieval.py    # Neo4j-backed graph retrieval

ui/                     # Page implementations (actual logic lives here)
  chat.py
  dashboard.py
  ingest.py
  models.py             # LoRA fine-tuning UI + dataset preview
  personas.py
  rag_explorer.py
  settings.py
  sidebar.py
  graph.py
  entity_browser.py
  extract.py
  components/

pages/                  # Thin wrappers that call ui/ — required by st.navigation()

tools/
  extract_facebook.py           # Parse Facebook HTML export
  ingest_facebook_messages.py   # Ingest into ChromaDB
  export_finetune.py            # Export JSONL for fine-tuning
  finetune_lora.py              # LoRA fine-tuning (HuggingFace/Unsloth)
  export_to_ollama.py           # GGUF export + ollama create
  build_knowledge_graph.py      # Build Neo4j graph from data
  episodic_memory.py
  persona_clustering.py
  extractors/                   # Per-source extractors (Spotify, Steam, Strava, etc.)

data/                   # Raw personal data files (gitignored)
  facebook/
  google/
  spotify/
  steam/
  strava/

models/                 # LoRA adapter outputs (gitignored)
.chroma_data/           # ChromaDB persistent storage (gitignored)
neo4j_data/             # Neo4j data (gitignored)
```

## Key config (config.py)

- `CHROMA_PATH` — ChromaDB directory (`./.chroma_data`)
- `EMBEDDING_MODEL` — override with `EMBEDDING_MODEL` env var; default `BAAI/bge-m3`. **Changing this requires re-ingesting all collections.**
- `DEFAULT_MODEL` — Ollama model for chat (`qwen2.5:7b`)
- `SOURCES` — registry of data sources; each entry drives ingestion, stats, and graph building
- `IDENTITIES` — IFS-inspired persona prompts (The Self, The Protector, The Planner, etc.)
- `NEO4J_URI/USER/PASSWORD` — override via env vars

## Data flow

1. **Extract** — `tools/extractors/` parse raw exports into normalized JSON under `data/`
2. **Ingest** — tools ingest into ChromaDB collections (one per source) and optionally Neo4j
3. **RAG** — `rag/rag_retrieval.py` does hybrid search (semantic + BM25 + RRF + optional reranker)
4. **Chat** — `rag/llm.py` calls Ollama with retrieved context + persona system prompt
5. **Fine-tune** — `tools/export_finetune.py` exports JSONL, `tools/finetune_lora.py` trains LoRA

## Fine-tuning notes

- Training data format: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
- Content must be **plain strings** when passed to `apply_chat_template` — not content-block lists (`[{"type": "text", "text": "..."}]`), which most chat templates don't render properly
- Unsloth is used when available (faster); falls back to HuggingFace transformers + peft + trl
- LoRA adapter outputs go to `models/`; export to Ollama via `tools/export_to_ollama.py`

## Architecture notes

- `pages/` are thin wrappers — all real logic is in `ui/`. Never put business logic in `pages/`.
- `rag/resources.py` uses `@st.cache_resource` — loaders run once per session. Don't call them outside Streamlit context.
- The `SOURCES` list in `config.py` is the single source of truth for what data sources exist. Add new sources there first.
- Conversations are multilingual (primarily French) — embedding model must support multilingual content.
