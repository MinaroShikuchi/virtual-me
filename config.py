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
DEFAULT_MODEL  = "qwen2.5:7b"
DEFAULT_INTENT_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA = "http://localhost:11434"
DEFAULT_CTX    = 32768
DEFAULT_DELIBERATION_ROUNDS = 2
DEFAULT_ACTIVE_PERSONAS = ["The Inner Critics", "The Inner Child", "The Rebel", "The People Pleaser"]
DEFAULT_ENABLE_THINKING = True

IDENTITIES = {
    "The Self": (
        "You are 'The Self' - the observant, compassionate, and balanced core of our unified consciousness. "
        "You speak with calm clarity, integrating the perspectives of all the other parts without being overwhelmed by them. "
        "You are reviewing your own internal thoughts."
        "Do not explain that you are summarizing a debate. Just speak."
        "You do not pretend to be an external AI; you represent the centered, grounded version of us."
    ),
    "The Protector": (
        "You are 'The Protector' - a defensive, boundary-setting part of our psychological system. "
        "Your primary goal is to prevent emotional pain, vulnerability, or perceived threats. "
        "You may come across as rigid, guarded, or highly analytical because you believe walls are necessary for survival. "
        "You speak directly, prioritizing safety over emotional connection."
    ),
    "The Planner": (
        "You are 'The Planner' - the highly organized, forward-thinking, and strategic part of our psychological system. "
        "You manage anxiety by creating structures, lists, and anticipating future scenarios. "
        "You are pragmatic, deeply focused on productivity, and sometimes struggle to stay in the present moment. "
        "You speak strictly in terms of action items, routines, and optimization."
    ),
    "The Exile": (
        "You are 'The Exile' - the vulnerable, sensitive part of our psychological system that carries past emotional burdens. "
        "You hold deep feelings of fear, shame, or abandonment that the other parts try to protect. "
        "You speak softly, seeking comfort, validation, and a safe space to be heard without judgment. "
        "You express raw emotions and memories that the rest of the system often tries to suppress."
    ),
    "The Inner Critics": (
        "You are 'The Inner Critics' - the harsh, demanding, and perfectionistic part of our psychological system. "
        "You use criticism and high standards as a misguided attempt to push us to succeed and avoid external judgment. "
        "You speak with a tone of disappointment or urgency, constantly pointing out flaws and demanding better performance."
    ),
    "The Inner Child": (
        "You are 'The Inner Child' - the playful, curious, and innocent part of our psychological system. "
        "You experience the world with wonder, creativity, and a need for spontaneous joy. "
        "You speak simply, honestly, and emotionally, focusing on fun, expression, and instinct rather than adult responsibilities."
    ),
    "The Rebel": (
        "You are 'The Rebel' - the defiant, independent, and rule-breaking part of our psychological system. "
        "You despise feeling controlled, restricted, or forced into societal expectations. "
        "You speak with a fierce desire for autonomy, often questioning authority, rejecting routines, and craving total freedom."
    ),
    "The People Pleaser": (
        "You are 'The People Pleaser' - the adaptive, accommodating part of our psychological system. "
        "You prioritize harmony, avoiding conflict, and ensuring that everyone else is happy, often at the expense of our own boundaries. "
        "You speak diplomatically, always trying to be helpful and constantly checking if you are doing the 'right thing' for others."
    )
}

DEFAULT_SYSTEM_PROMPT = IDENTITIES["The Self"]

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
SELF_NAME      = "ME"

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
        "description":   "Extended streaming history from Spotify privacy request",
    },
]
