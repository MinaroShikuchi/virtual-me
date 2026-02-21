"""
graph/constants.py — Centralised schema constants for the Virtual-Me knowledge graph.

All entity labels, relationship types, colour palettes, icon mappings, and
interest keyword taxonomies live here so every module shares a single source
of truth.
"""

# ── Entity labels recognised by the schema ────────────────────────────────────
ENTITY_LABELS: list[str] = [
    "Person", "Place", "Song", "Artist", "Company",
    "Game", "Activity", "Interest", "School", "Device",
]

# ── Relationship types ────────────────────────────────────────────────────────
REL_TYPES: list[str] = [
    "PARTNER_OF", "FRIEND_OF", "FAMILY_OF", "COLLEAGUE_OF", "MET",
    "LIVES_IN", "VISITED", "TRAVELLED_TO",
    "WORKS_AT", "LISTENED_TO", "INTERESTED_IN",
    "KNOWS", "STUDIED_AT", "USED_DEVICE",
]

# ── Entity colour palette (hex) ──────────────────────────────────────────────
LABEL_COLORS: dict[str, str] = {
    "Person":   "#6366f1",
    "Place":    "#22c55e",
    "Song":     "#f59e0b",
    "Artist":   "#ec4899",
    "Company":  "#0ea5e9",
    "Game":     "#8b5cf6",
    "Activity": "#14b8a6",
    "Interest": "#f97316",
    "School":   "#a855f7",
    "Device":   "#64748b",
}

# ── Relationship → Material Symbols icon name ────────────────────────────────
REL_ICONS: dict[str, str] = {
    "PARTNER_OF":    "favorite",
    "FRIEND_OF":     "handshake",
    "FAMILY_OF":     "family_restroom",
    "COLLEAGUE_OF":  "badge",
    "MET":           "waving_hand",
    "LIVES_IN":      "home",
    "VISITED":       "pin_drop",
    "TRAVELLED_TO":  "flight",
    "WORKS_AT":      "apartment",
    "LISTENED_TO":   "music_note",
    "INTERESTED_IN": "star",
    "KNOWS":         "person_add",
    "STUDIED_AT":    "school",
    "USED_DEVICE":   "devices",
}

# ── Interest keyword taxonomy (used by facebook_messages extractor) ───────────
INTEREST_KEYWORDS: dict[str, list[str]] = {
    "gaming":  ["jeu", "jeux", "game", "gaming", "steam", "ps4", "ps5", "xbox",
                "nintendo", "lol", "league of legends", "fortnite", "minecraft",
                "valorant", "gta", "fifa", "cod", "call of duty", "overwatch"],
    "music":   ["musique", "music", "concert", "festival", "rap", "hip-hop",
                "techno", "playlist", "spotify", "album", "chanson", "track"],
    "sport":   ["sport", "foot", "football", "soccer", "basket", "basketball",
                "tennis", "natation", "running", "course", "velo", "vélo",
                "gym", "fitness", "musculation", "rugby", "handball"],
    "travel":  ["voyage", "travel", "vacances", "holiday", "trip", "paris",
                "london", "barcelona", "new york", "tokyo", "avion", "airport",
                "hotel", "airbnb", "roadtrip"],
    "cinema":  ["film", "movie", "cinema", "série", "netflix", "amazon prime",
                "disney", "marvel", "anime", "manga"],
    "food":    ["restaurant", "bouffe", "manger", "nourriture", "food", "pizza",
                "sushi", "burger", "cuisine", "recette", "recipe"],
    "tech":    ["tech", "code", "programmation", "dev", "développeur",
                "developer", "python", "javascript", "react", "startup",
                "intelligence artificielle", "ia", "ai"],
    "party":   ["fête", "soirée", "party", "bar", "club", "alcool",
                "bière", "beer", "vin"],
    "nature":  ["randonnée", "hiking", "montagne", "mountain", "forêt", "forest",
                "plage", "beach", "camping", "nature", "trek"],
}
