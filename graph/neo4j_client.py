"""
graph/neo4j_client.py — Neo4j driver, schema constraints, and MERGE helpers.

Designed to be imported by extractor scripts and the Streamlit UI alike.
All writes use MERGE so re-running extractors is safe (idempotent).
"""
import sys
from contextlib import contextmanager

from neo4j import GraphDatabase, exceptions as neo4j_exc

from graph.constants import ENTITY_LABELS, REL_TYPES  # noqa: F401 — re-exported


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

            # Labels whose stat card should show the sum of a relationship
            # property instead of the raw node count.
            # Artist/Song cards show unique node count; only Activity uses
            # the aggregated activity_count.
            _AGG_LABELS = {
                "Activity": ("INTERESTED_IN", "activity_count"),
            }

            # Interest card: count unique target nodes of INTERESTED_IN
            # relationships (they may be labeled Interest or Activity).
            _REL_COUNT_LABELS = {
                "Interest": "INTERESTED_IN",
            }

            for label in ENTITY_LABELS:
                rel_count_rel = _REL_COUNT_LABELS.get(label)
                if rel_count_rel and rel_count_rel in existing_rels:
                    # Count distinct target nodes of the relationship
                    try:
                        result = s.run(
                            f"MATCH ()-[:{rel_count_rel}]->(n) "
                            f"RETURN count(DISTINCT n) AS c"
                        )
                        stats[label] = result.single()["c"]
                    except:
                        stats[label] = 0
                elif label in existing_labels:
                    try:
                        agg = _AGG_LABELS.get(label)
                        if agg and agg[0] in existing_rels:
                            rel_type, prop = agg
                            result = s.run(
                                f"MATCH ()-[r:{rel_type}]->(n:{label}) "
                                f"RETURN coalesce(sum(r.{prop}), count(n)) AS c"
                            )
                        else:
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

    def top_nodes_by_degree(self, label: str, limit: int = 10,
                            exclude_names: list[str] | None = None) -> list[dict]:
        """
        Returns the top `limit` nodes of `label` sorted by total relationship count.
        Each result: {name, degree}

        For Activity nodes, ``degree`` is the sum of ``activity_count`` stored on
        incoming INTERESTED_IN relationships (i.e. total activities, not just 1
        per type).  For all other labels the plain relationship count is used.

        Args:
            exclude_names: node names to filter out (e.g. the self-identity node).
        """
        excluded = [n.upper() for n in (exclude_names or [])]
        # Labels where "degree" should be a relationship property sum
        _AGG_DEGREE = {
            "Activity": ("INTERESTED_IN", "activity_count"),
            "Artist":   ("LISTENED_TO",   "play_count"),
            "Song":     ("LISTENED_TO",   "play_count"),
        }

        with self.driver.session() as s:
            agg = _AGG_DEGREE.get(label)
            if agg:
                rel_type, prop = agg
                result = s.run(
                    f"MATCH (n:{label}) "
                    f"WHERE NOT toUpper(n.name) IN $excluded "
                    f"OPTIONAL MATCH ()-[r:{rel_type}]->(n) "
                    f"RETURN n.name AS name, "
                    f"coalesce(sum(r.{prop}), count(r)) AS degree "
                    f"ORDER BY degree DESC LIMIT $limit",
                    limit=limit, excluded=excluded,
                )
            else:
                result = s.run(
                    f"MATCH (n:{label}) "
                    f"WHERE NOT toUpper(n.name) IN $excluded "
                    f"OPTIONAL MATCH (n)-[r]-() "
                    f"RETURN n.name AS name, count(r) AS degree "
                    f"ORDER BY degree DESC LIMIT $limit",
                    limit=limit, excluded=excluded,
                )
            return [{"name": r["name"], "degree": r["degree"]} for r in result]

    def interest_profile(self, self_name: str = "ME") -> dict[str, float]:
        """
        Returns {interest_name: percentage} for the self-identity node.
        Queries INTERESTED_IN relationships from the Person node and
        computes relative percentages.
        """
        with self.driver.session() as s:
            result = s.run(
                "MATCH (p:Person {name: $name})-[r:INTERESTED_IN]->(i:Interest) "
                "RETURN i.name AS interest, "
                "       CASE WHEN r.weight IS NOT NULL THEN r.weight ELSE 1.0 END AS weight "
                "ORDER BY weight DESC",
                name=self_name,
            )
            rows = [(r["interest"], r["weight"]) for r in result]

        if not rows:
            return {}

        total = sum(w for _, w in rows) or 1
        return {name: round(w / total * 100, 1) for name, w in rows}


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
