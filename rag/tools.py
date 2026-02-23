"""
rag/tools.py — Memory tools for persona agents.

Defines tool schemas (JSON Schema for Ollama's native tool-calling API)
and a ToolExecutor class that wraps existing retrieval pipelines.
"""
import logging

log = logging.getLogger(__name__)

# ── Tool JSON Schemas (Ollama format) ──────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_memories",
            "description": (
                "Search conversation logs and episodic memories. "
                "Use this when you need to recall specific conversations, "
                "events, or experiences from the past."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_graph",
            "description": (
                "Search the knowledge graph for facts about people, "
                "places, and relationships. Use this when you need to "
                "know about specific people or locations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "people": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of people to look up",
                    },
                    "locations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of locations to look up",
                    },
                },
                "required": [],
            },
        },
    },
]


# ── Tool Executor ──────────────────────────────────────────────────────


class ToolExecutor:
    """Executes tool calls using existing retrieval infrastructure.

    Wraps ``rag.retrieval.retrieve`` and ``rag.graph_retrieval.retrieve_facts``
    so that persona agents can call them via Ollama's tool-calling protocol.

    Call :meth:`set_baseline` after construction to register documents and
    facts already provided via upfront retrieval.  Subsequent tool calls
    will automatically deduplicate against this baseline so agents only
    receive *new* information.
    """

    def __init__(
        self,
        collection,
        episodic,
        id_to_name: dict,
        name_to_id: dict,
        intent_model: str,
        ollama_host: str,
        n_results: int = 20,
        top_k: int = 10,
        do_rerank: bool = True,
        hybrid: bool = True,
    ):
        self.collection = collection
        self.episodic = episodic
        self.id_to_name = id_to_name
        self.name_to_id = name_to_id
        self.intent_model = intent_model
        self.ollama_host = ollama_host
        self.n_results = n_results
        self.top_k = top_k
        self.do_rerank = do_rerank
        self.hybrid = hybrid

        # Baseline sets — populated via set_baseline()
        self._baseline_contents: set[str] = set()
        self._baseline_episode_contents: set[str] = set()
        self._baseline_facts: set[str] = set()

    # ── baseline registration ─────────────────────────────────────────

    def set_baseline(
        self,
        docs: list[dict],
        episodes: list[dict],
        facts: list[str],
    ) -> None:
        """Register documents, episodes and facts from upfront retrieval.

        Any subsequent tool call returning the same content will be
        filtered out so agents only see genuinely new information.
        """
        self._baseline_contents = {d.get("content", "") for d in docs}
        self._baseline_episode_contents = {e.get("content", "") for e in episodes}
        self._baseline_facts = set(facts)
        log.info(
            "ToolExecutor baseline: %d docs, %d episodes, %d facts",
            len(self._baseline_contents),
            len(self._baseline_episode_contents),
            len(self._baseline_facts),
        )

    # ── public dispatch ────────────────────────────────────────────────

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Dispatch a tool call by name and return a formatted string result."""
        if tool_name == "search_memories":
            return self._search_memories(arguments.get("query", ""))
        if tool_name == "search_knowledge_graph":
            return self._search_graph(
                arguments.get("people", []),
                arguments.get("locations", []),
            )
        return f"Unknown tool: {tool_name}"

    # ── private implementations ────────────────────────────────────────

    def _search_memories(self, query: str) -> str:
        from rag.retrieval import retrieve

        docs, episodes, _intent = retrieve(
            query,
            self.n_results,
            self.collection,
            self.episodic,
            self.id_to_name,
            self.name_to_id,
            self.intent_model,
            self.ollama_host,
            top_k=self.top_k,
            do_rerank=self.do_rerank,
            hybrid=self.hybrid,
        )

        # Deduplicate against baseline
        if self._baseline_contents:
            before = len(docs)
            docs = [d for d in docs if d.get("content", "") not in self._baseline_contents]
            if before != len(docs):
                log.info("    → dedup: %d→%d docs (removed %d baseline duplicates)",
                         before, len(docs), before - len(docs))
        if self._baseline_episode_contents:
            before = len(episodes)
            episodes = [e for e in episodes if e.get("content", "") not in self._baseline_episode_contents]
            if before != len(episodes):
                log.info("    → dedup: %d→%d episodes (removed %d baseline duplicates)",
                         before, len(episodes), before - len(episodes))

        log.info("    → search_memories returned %d new docs, %d new episodes", len(docs), len(episodes))

        parts: list[str] = []
        if docs:
            parts.append("=== CONVERSATION LOGS ===")
            for d in docs:
                parts.append(
                    f"[{d['date']}] [Chat: {d['friend']}]\n{d['content']}"
                )
        if episodes:
            parts.append("=== EPISODIC MEMORIES ===")
            for e in episodes:
                parts.append(f"[{e['date']}] {e['content']}")
        return "\n\n".join(parts) if parts else "No new memories found (already in baseline context)."

    def _search_graph(self, people: list, locations: list) -> str:
        from rag.graph_retrieval import retrieve_facts

        intent = {"people": people, "locations": locations}
        facts = retrieve_facts(intent)

        # Deduplicate against baseline
        if self._baseline_facts:
            before = len(facts)
            facts = [f for f in facts if f not in self._baseline_facts]
            if before != len(facts):
                log.info("    → dedup: %d→%d facts (removed %d baseline duplicates)",
                         before, len(facts), before - len(facts))

        log.info("    → search_knowledge_graph returned %d new facts", len(facts))
        if facts:
            return (
                "=== KNOWLEDGE GRAPH FACTS ===\n"
                + "\n".join(f"- {f}" for f in facts)
            )
        return "No new facts found (already in baseline context)."
