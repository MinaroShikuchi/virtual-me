"""
rag/skills.py — Specialized high-precision retrieval functions (Skills) for the Digital Twin.
Exposes Ollama tool definitions so the LLM can call these functions natively.
"""
from graph.neo4j_client import get_client

def get_top_played_games(limit=5):
    """Retrieves the top N most frequent games/activities from Neo4j."""
    facts = []
    try:
        with get_client() as client:
            with client.driver.session() as s:
                result = s.run(
                    """
                    MATCH (u {name: 'ME'})-[r:PLAYED]->(g:Game)
                    RETURN g.name, r.total_hours, r.session_count
                    ORDER BY r.total_hours DESC
                    LIMIT $limit
                    """,
                    limit=limit
                )
                for record in result:
                    facts.append(f"Game: {record['g.name']} ({record['r.total_hours']}) [played {record['r.session_count']} times]")
    except Exception as e:
        print(f"Skill error (get_top_played_games): {e}")
    return facts

def get_top_played_music(limit=5):
    """Retrieves the top N most played songs/artists."""
    facts = []
    try:
        with get_client() as client:
            with client.driver.session() as s:
                result = s.run(
                    """
                    MATCH (u {name: 'ME'})-[r:LISTENED_TO]->(a:Song)
                    RETURN a.name, r.play_count
                    ORDER BY r.play_count DESC
                    LIMIT $limit
                    """,
                    limit=limit
                )
                for record in result:
                    facts.append(f"Top Music: '{record['song']}' by {record['artist']} [{record['p_count']} plays]")
    except Exception as e:
        print(f"Skill error (get_top_played_music): {e}")
    return facts

def get_top_listened_artists(limit=5):
    """Retrieves the top N most listened-to artists."""
    facts = []
    try:
        with get_client() as client:
            with client.driver.session() as s:
                result = s.run(
                    """
                    MATCH (u {name: 'ME'})-[r:LISTENED_TO]->(a:Artist)
                    RETURN a.name, r.play_count
                    ORDER BY r.play_count DESC
                    LIMIT $limit
                    """,
                    limit=limit
                )
                for record in result:
                    facts.append(f"Top Artist: '{record['artist']}' [{record['p_count']} plays]")
    except Exception as e:
        print(f"Skill error (get_top_listened_artists): {e}")
    return facts

# ── Registry ──────────────────────────────────────────────────────────────────
SKILLS_REGISTRY = {
    "retrieve_most_played_game": get_top_played_games,
    "retrieve_top_music": get_top_played_music,
    "retrieve_top_listened_artists": get_top_listened_artists,
}

# ── Intent Router metadata (used for prompt injection) ────────────────────────
SKILLS_INFO = {
    "retrieve_most_played_game": "Finds the most frequently played video games or physical activities using graph play counts.",
    "retrieve_top_music": "Retrieves the user's most listened-to songs and artists from Spotify session data.",
    "retrieve_top_listened_artists": "Retrieves the user's most listened-to artists from Spotify session data.",
}

# ── Ollama Tool schemas (used for LLM native tool calling) ────────────────────
OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_most_played_game",
            "description": "Fetch the top most-played games or activities from the personal knowledge graph. Use this when the user asks about their favorite games, what games they play, or gaming preferences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of top results to return. Default is 5.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_top_music",
            "description": "Fetch the most listened-to songs and artists from Spotify history in the personal knowledge graph. Use this when the user asks about their favorite music, songs, or artists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of top results to return. Default is 5.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_top_listened_artists",
            "description": "Fetch the top most-listened-to artists from Spotify history in the personal knowledge graph. Use this when the user asks about their favorite artists or music.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of top results to return. Default is 5.",
                    }
                },
                "required": [],
            },
        },
    },
]
