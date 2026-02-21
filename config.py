"""
config.py — Central configuration and data-source registry for Virtual Me.
"""
import os
from pathlib import Path

# ── ChromaDB ──────────────────────────────────
CHROMA_PATH       = "./.chroma_data"
COLLECTION_NAME   = "virtual_me_knowledge"
EPISODIC_NAME     = "episodic_memory"
NAME_MAPPING_FILE = "./conversation_names.json"

# ── Data folder ───────────────────────────────
DATA_DIR = Path("./data")

# ── Ollama defaults ───────────────────────────
DEFAULT_MODEL  = "deepseek-r1:14b"
DEFAULT_OLLAMA = "http://localhost:11434"
DEFAULT_CTX    = 32768

# ── Embedding ─────────────────────────────────
# NOTE: Changing this requires re-ingesting all ChromaDB collections.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

# Popular multilingual / English embedding models (shown in Settings picker).
EMBEDDING_MODELS = [
    "BAAI/bge-m3",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-large",
    "intfloat/e5-large-v2",
    "intfloat/e5-base-v2",
    "nomic-ai/nomic-embed-text-v1.5",
]

# ── Neo4j ─────────────────────────────────────
NEO4J_URI      = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

# ── Knowledge graph self-identity ─────────────
# The name used to anchor "you" in the graph (most-frequent sender auto-detected
# in extractors, but can be overridden with --self-name CLI arg)
SELF_NAME      = os.environ.get("SELF_NAME", "ME")

# ── Known data sources ────────────────────────
SOURCES = [
    {
        "id":            "facebook",
        "label":         "Facebook Messages",
        "icon":          "chat",
        "color":         "#4267B2",
        "chroma_source": "facebook_windowed",
        "data_folder":   "facebook",
        "file_patterns": ["*.json"],
        "stat_label":    "messages",
        "approx_total":  400_000,
        "description":   "Chat history ingested from Facebook Messenger export",
    },
    {
        "id":            "google",
        "label":         "Google Locations",
        "icon":          "map",
        "color":         "#34A853",
        "chroma_source": "google_locations",
        "data_folder":   "google",
        "file_patterns": ["Records.json", "*.json"],
        "stat_label":    "locations",
        "approx_total":  600,
        "description":   "Location history from Google Takeout",
    },
    {
        "id":            "spotify",
        "label":         "Spotify Sessions",
        "icon":          "music_note",
        "color":         "#1DB954",
        "chroma_source": "spotify",
        "data_folder":   "spotify",
        "file_patterns": ["Streaming_History_Audio_*.json"],
        "stat_label":    "sessions",
        "approx_total":  40_000,
        "description":   "Extended streaming history from Spotify privacy request",
    },
]
