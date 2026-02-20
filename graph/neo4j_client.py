"""
graph/neo4j_client.py — Neo4j driver, schema constraints, and MERGE helpers.

Designed to be imported by extractor scripts and the Streamlit UI alike.
All writes use MERGE so re-running extractors is safe (idempotent).
"""
import sys
from contextlib import contextmanager

from neo4j import GraphDatabase, exceptions as neo4j_exc

# ── Entity labels recognised by the schema ────────────────────────────────────
ENTITY_LABELS = [
    "Person", "Place", "Song", "Artist", "Company",
    "Game", "Activity", "Interest",
]

# ── Relationship types ────────────────────────────────────────────────────────
REL_TYPES = [
    "PARTNER_OF", "FRIEND_OF", "FAMILY_OF", "COLLEAGUE_OF", "MET",
    "LIVES_IN", "VISITED", "TRAVELLED_TO",
    "WORKS_AT", "LISTENED_TO", "INTERESTED_IN",
]


class Neo4jClient:
    """Thin wrapper around the Neo4j driver with MERGE helpers and schema setup."""

    def __init__(self, uri: str, user: str, password: str):
        # Suppress "unrecognized label" notifications in Neo4j 5.x
        self.driver = GraphDatabase.driver(
            uri, auth=(user, password),
            notifications_min_severity="OFF"
        )

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def verify(self) -> bool:
        """Returns True if the connection is alive."""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def ensure_constraints(self):
        """Create UNIQUE constraints for every entity type (idempotent)."""
        with self.driver.session() as s:
            for label in ENTITY_LABELS:
                s.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.name IS UNIQUE"
                )
        print("✅ Schema constraints ensured.", flush=True)

    # ── Write helpers ─────────────────────────────────────────────────────────

    def merge_entity(self, label: str, name: str, extra_props: dict | None = None) -> None:
        """MERGE a node by (label, name) and optionally set extra properties."""
        props = extra_props or {}
        with self.driver.session() as s:
            s.run(
                f"MERGE (n:{label} {{name: $name}}) "
                f"ON CREATE SET n += $props "
                f"ON MATCH  SET n += $props",
                name=name.strip(), props=props,
            )

    def merge_relation(
        self,
        from_label: str, from_name: str,
        rel_type: str,
        to_label: str,   to_name: str,
        props: dict | None = None,
    ) -> None:
        """
        MERGE a relationship between two named entities.
        Both nodes are also MERGEd so they don't need to exist beforehand.
        """
        p = props or {}
        since = p.get("since", "")
        with self.driver.session() as s:
            if since:
                s.run(
                    f"MERGE (a:{from_label} {{name: $from_name}}) "
                    f"MERGE (b:{to_label}   {{name: $to_name}}) "
                    f"MERGE (a)-[r:{rel_type} {{since: $since}}]->(b) "
                    f"ON CREATE SET r += $props "
                    f"ON MATCH  SET r += $props",
                    from_name=from_name.strip(),
                    to_name=to_name.strip(),
                    since=since,
                    props=p,
                )
            else:
                s.run(
                    f"MERGE (a:{from_label} {{name: $from_name}}) "
                    f"MERGE (b:{to_label}   {{name: $to_name}}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    f"ON CREATE SET r += $props "
                    f"ON MATCH  SET r += $props",
                    from_name=from_name.strip(),
                    to_name=to_name.strip(),
                    props=p,
                )

    def batch_merge_relations(self, rows: list[dict]) -> None:
        """
        Bulk MERGE relationships.
        Each row: {from_label, from_name, rel_type, to_label, to_name, props}
        """
        with self.driver.session() as s:
            for row in rows:
                p = row.get("props", {})
                since = p.get("since", "")
                if since:
                    s.run(
                        f"MERGE (a:{row['from_label']} {{name: $fn}}) "
                        f"MERGE (b:{row['to_label']}   {{name: $tn}}) "
                        f"MERGE (a)-[r:{row['rel_type']} {{since: $since}}]->(b) "
                        f"ON CREATE SET r += $props "
                        f"ON MATCH  SET r += $props",
                        fn=row["from_name"].strip(),
                        tn=row["to_name"].strip(),
                        since=since,
                        props=p,
                    )
                else:
                    s.run(
                        f"MERGE (a:{row['from_label']} {{name: $fn}}) "
                        f"MERGE (b:{row['to_label']}   {{name: $tn}}) "
                        f"MERGE (a)-[r:{row['rel_type']}]->(b) "
                        f"ON CREATE SET r += $props "
                        f"ON MATCH  SET r += $props",
                        fn=row["from_name"].strip(),
                        tn=row["to_name"].strip(),
                        props=p,
                    )

    # ── Read helpers ──────────────────────────────────────────────────────────

    def graph_stats(self) -> dict:
        """Returns {label: count} for all entity labels, avoiding unrecognized label warnings."""
        stats = {}
        with self.driver.session() as s:
            # Get existing labels and types. 
            # In Neo4j 5.x, CALL db.labels() returns a column called 'label'
            try:
                existing_labels = {r["label"] for r in s.run("CALL db.labels()")}
            except: 
                # Fallback for different driver/DB versions if 'label' key missing
                existing_labels = set()
            
            try:
                existing_rels = {r["relationshipType"] for r in s.run("CALL db.relationshipTypes()")}
            except:
                existing_rels = set()

            for label in ENTITY_LABELS:
                if label in existing_labels:
                    try:
                        result = s.run(f"MATCH (n:{label}) RETURN count(n) AS c")
                        stats[label] = result.single()["c"]
                    except:
                        stats[label] = 0
                else:
                    stats[label] = 0

            for rel in REL_TYPES:
                if rel in existing_rels:
                    try:
                        result = s.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c")
                        stats[f"→{rel}"] = result.single()["c"]
                    except:
                        stats[f"→{rel}"] = 0
                else:
                    stats[f"→{rel}"] = 0
        return stats

    def neighbours(self, label: str, name: str, limit: int = 50) -> list[dict]:
        """
        Return all (rel_type, neighbour_label, neighbour_name) for a given node.
        """
        with self.driver.session() as s:
            result = s.run(
                "MATCH (n {name: $name})-[r]-(m) "
                "RETURN type(r) AS rel, labels(m)[0] AS label, m.name AS name "
                "LIMIT $limit",
                name=name, limit=limit,
            )
            return [dict(record) for record in result]

    def search_nodes(self, label: str, query: str, limit: int = 20) -> list[str]:
        """Full-text prefix search on node names."""
        with self.driver.session() as s:
            result = s.run(
                f"MATCH (n:{label}) "
                f"WHERE toLower(n.name) CONTAINS toLower($q) "
                f"RETURN n.name AS name ORDER BY n.name LIMIT $limit",
                q=query, limit=limit,
            )
            return [r["name"] for r in result]

    def top_nodes_by_degree(self, label: str, limit: int = 10) -> list[dict]:
        """
        Returns the top `limit` nodes of `label` sorted by total relationship count.
        Each result: {name, degree}
        """
        with self.driver.session() as s:
            result = s.run(
                f"MATCH (n:{label}) "
                f"OPTIONAL MATCH (n)-[r]-() "
                f"RETURN n.name AS name, count(r) AS degree "
                f"ORDER BY degree DESC LIMIT $limit",
                limit=limit,
            )
            return [{"name": r["name"], "degree": r["degree"]} for r in result]


def get_client(uri: str | None = None, user: str | None = None,
               password: str | None = None) -> Neo4jClient:
    """
    Returns a Neo4jClient, falling back to config.py defaults.
    Callers are responsible for closing (use as context manager).
    """
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    return Neo4jClient(
        uri      or NEO4J_URI,
        user     or NEO4J_USER,
        password or NEO4J_PASSWORD,
    )
