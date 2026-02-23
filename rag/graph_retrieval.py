"""
rag/graph_retrieval.py — Lookup semantic facts in Neo4j based on LLM intent.
"""
from graph.neo4j_client import get_client

def _format_rel_label(rel: str) -> str:
    """Converts SNAKE_CASE relationships like WAS_FRIENDS_WITH into natural language with tense info."""
    clean = rel.replace("_", " ").lower()
    if rel.startswith("WAS_"):
        return f"{clean} (PAST relationship, no longer current)"
    if rel.startswith("IS_"):
        return f"{clean} (CURRENT relationship)"
    return clean

def retrieve_facts(intent: dict) -> list[str]:
    """
    Given an intent dictionary (people, locations, time_periods),
    queries Neo4j for semantic facts involving those entities.
    Returns human-readable semantic facts.
    """
    facts = []
    people = intent.get("people", [])
    locations = intent.get("locations", [])
    
    # We only bother querying if we have people or locations
    if not people and not locations:
        return facts
        
    try:
        with get_client() as client:
            with client.driver.session() as s:
                # 1. Query facts about identified people
                if people:
                    for person in people:
                        # Find relationships involving this person
                        # We use toLower to handle case insensitivity
                        result = s.run(
                            "MATCH (n {name: $name})-[r]-(m) "
                            "WITH type(r) AS rel, labels(m)[0] AS label, m.name AS target_name, properties(r) AS props "
                            "LIMIT 10 "
                            "RETURN rel, label, target_name, coalesce(props.since, '') AS since, coalesce(props.weight, '') AS weight",
                            name=person
                        )
                        for record in result:
                            r = dict(record)
                            rel_text = _format_rel_label(r['rel'])
                            factStr = f"{person} {rel_text} {r['target_name']} ({r['label']})"
                            if r['since']: factStr += f" since {r['since']}"
                            if r['weight']: factStr += f" (weight: {r['weight']})"
                            facts.append(factStr)
                            
                # 2. Query facts about locations
                if locations:
                    for loc in locations:
                        result = s.run(
                            "MATCH (n {name: $name})-[r]-(m) "
                            "WITH type(r) AS rel, labels(m)[0] AS label, m.name AS target_name, properties(r) AS props "
                            "LIMIT 10 "
                            "RETURN rel, label, target_name, coalesce(props.since, '') AS since",
                            name=loc
                        )
                        for record in result:
                            r = dict(record)
                            rel_text = _format_rel_label(r['rel'])
                            factStr = f"{loc} {rel_text} {r['target_name']} ({r['label']})"
                            if r['since']: factStr += f" since {r['since']}"
                            facts.append(factStr)
                            
    except Exception as e:
        print(f"Failed to retrieve graph facts: {e}")
        
    # Deduplicate facts but preserve order
    seen = set()
    dedup_facts = []
    for f in facts:
        if f not in seen:
            seen.add(f)
            dedup_facts.append(f)
            
    return dedup_facts
