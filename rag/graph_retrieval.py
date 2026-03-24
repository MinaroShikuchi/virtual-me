"""
rag/graph_retrieval.py — Lookup semantic facts in Neo4j based on LLM intent.
"""
from graph.neo4j_client import get_client
from rag.skills import SKILLS_REGISTRY

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
    Given an intent dictionary (people, locations, time_periods, entities, skills),
    queries Neo4j for semantic facts involving those entities OR specialized skill functions.
    Returns human-readable semantic facts.
    """
    facts = []
    
    # 0. Execute specialized Skills if identified by the intent router
    requested_skills = intent.get("skills", [])
    if isinstance(requested_skills, list) and requested_skills:
        from rag.skills import SKILLS_INFO
        for skill_id in requested_skills:
            if skill_id in SKILLS_REGISTRY:
                desc = SKILLS_INFO.get(skill_id, "Specialized retrieval results")
                skill_results = SKILLS_REGISTRY[skill_id]()
                print(f"[SKILL] Executing {skill_id}... Found {len(skill_results)} results.")
                if skill_results:
                    facts.append(f"🔍 {desc}:")
                    facts.extend([f"  - {r}" for r in skill_results])
    
    # Collect all search terms from the intent
    search_terms = []
    search_terms.extend([(p, "Person") for p in intent.get("people", [])])
    search_terms.extend([(l, "Location") for l in intent.get("locations", [])])
    search_terms.extend([(e, "Entity") for e in intent.get("entities", [])])
    
    if not search_terms:
        return facts
        
    try:
        with get_client() as client:
            with client.driver.session() as s:
                for term, source_type in search_terms:
                    # 💡 Precision Search:
                    # 1. Look for Exact matches (highest precision)
                    # 2. Look for Category/Activity partial matches (for topics like 'gaming' -> 'Video Games')
                    result = s.run(
                        """
                        MATCH (n)-[r]-(m)
                        WHERE toLower(n.name) = toLower($name)
                           OR ((n:Game OR n:category OR n:Activity) AND toLower(n.name) CONTAINS toLower($name))
                        WITH n, r, m, type(r) AS rel, labels(m)[0] AS label, m.name AS target_name,
                             CASE WHEN toLower(n.name) = toLower($name) THEN 1 ELSE 2 END as score,
                             coalesce(toInteger(r.sessions), toInteger(r.count), 0) AS p_count,
                             coalesce(toInteger(r.weight), 0) AS p_weight,
                             coalesce(toInteger(m.sessions), toInteger(m.count), 0) AS m_count
                        ORDER BY score ASC, p_count DESC, p_weight DESC, m_count DESC
                        LIMIT 15
                        RETURN n.name AS origin_name, rel, label, target_name,
                               coalesce(r.since, '') AS since, coalesce(r.weight, '') AS weight, 
                               coalesce(r.sessions, r.count, '') AS count
                        """,
                        name=term
                    )
                    
                    for record in result:
                        r = dict(record)
                        origin = r['origin_name'] or term
                        rel_text = _format_rel_label(r['rel'])
                        
                        # Build a natural language fact string
                        factStr = f"{origin} {rel_text} {r['target_name']} ({r['label']})"
                        
                        # Add extra metadata if present
                        meta = []
                        if r['since']: meta.append(f"since {r['since']}")
                        if r['weight']: meta.append(f"weight: {r['weight']}")
                        if r['count']:  meta.append(f"count: {r['count']}")
                        
                        if meta:
                            factStr += f" [{' | '.join(meta)}]"
                            
                        facts.append(factStr)
                            
    except Exception as e:
        print(f"Failed to retrieve graph facts for {search_terms}: {e}")
        
    # Deduplicate facts but preserve order
    seen = set()
    dedup_facts = []
    for f in facts:
        if f not in seen:
            seen.add(f)
            dedup_facts.append(f)
            
    return dedup_facts
