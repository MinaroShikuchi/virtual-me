#!/usr/bin/env python3
"""
tools/extractors/facebook_messages.py
--------------------------------------
Extract entities and relationships from facebook_messages.json
and write them to the Neo4j knowledge graph.

Entities extracted:
  Person     — every conversation partner + third parties from spaCy PER
  Place      — GPE/LOC entities from spaCy
  Company    — ORG entities from spaCy (filtered for work-context words)
  Interest   — keyword-matched topics (sport, gaming, travel, music…)

Relationships:
  Person → MET          → Person      (if ≥10 msgs exchanged)
  Person → VISITED      → Place       (mentioned in messages)
  Person → LIVES_IN     → Place       (inferred from "habite", "live in", frequent GPE)
  Person → WORKS_AT     → Company     (inferred from work-context words near ORG)
  Person → INTERESTED_IN→ Interest    (keyword matched)

Usage:
  python3 tools/extractors/facebook_messages.py [options]

Options:
  --json-file FILE   Path to facebook_messages.json  [default: facebook_messages.json]
  --self-name NAME   Your name in the graph (auto-detected if omitted)
  --limit N          Process only the first N conversation chunks
  --dry-run          Print extracted triples, do not write to Neo4j
  --neo4j-uri URI    Neo4j bolt URI       [env: NEO4J_URI]
  --neo4j-user USER  Neo4j user           [env: NEO4J_USER]
  --neo4j-pass PASS  Neo4j password       [env: NEO4J_PASSWORD]
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ensure project root is on sys.path so `graph.*` imports work when run standalone
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from graph.constants import INTEREST_KEYWORDS  # noqa: E402
from graph.neo4j_client import Neo4jClient     # noqa: E402

# ── spaCy models (load lazily) ────────────────────────────────────────────────
_nlp_fr = None
_nlp_en  = None


def _load_nlp():
    global _nlp_fr, _nlp_en
    if _nlp_fr is not None:
        return
    import spacy
    try:
        _nlp_fr = spacy.load("fr_core_news_md")
        print("✅ Loaded fr_core_news_md", flush=True)
    except OSError:
        print("⚠️  fr_core_news_md not found — run: python3 -m spacy download fr_core_news_md", flush=True)
        _nlp_fr = None
    try:
        _nlp_en = spacy.load("en_core_web_sm")
        print("✅ Loaded en_core_web_sm", flush=True)
    except OSError:
        print("⚠️  en_core_web_sm not found — run: python3 -m spacy download en_core_web_sm", flush=True)
        _nlp_en = None


# INTEREST_KEYWORDS imported from graph.constants above


# LIVES_IN triggers
LIVES_IN_TRIGGERS = {
    "fr": ["habite", "maison", "chez moi", "mon appart", "mon studio", "adresse", "emménagé"],
    "en": ["live in", "my place", "my house", "my apartment", "my address", "moved to", "living in"],
}

# Relationship classification keywords
REL_CLASSIFIER_KEYWORDS = {
    "PARTNER": {
        "fr": ["mon amour", "chéri", "ma puce", "bébé", "ma femme", "mon mari", "je t'aime", "bisous partout"],
        "en": ["my love", "honey", "baby", "my wife", "my husband", "i love you", "kisses"],
    },
    "FAMILY": {
        "fr": ["maman", "papa", "frère", "soeur", "cousin", "tante", "oncle", "mamie", "papi", "famille"],
        "en": ["mom", "dad", "brother", "sister", "cousin", "aunt", "uncle", "grandma", "grandpa", "family"],
    },
    "COLLEAGUE": {
        "fr": ["bureau", "collègue", "réunion", "zoom", "slack", "cafétéria", "projet", "client", "rdv", "boulot"],
        "en": ["office", "colleague", "meeting", "zoom", "slack", "cafeteria", "project", "client", "appointment", "work"],
    }
}


# ── NER helpers ───────────────────────────────────────────────────────────────

def _ner_entities(text: str) -> dict[str, list[str]]:
    """Run NER on text and return {ent_type: [names]}."""
    result: dict[str, list[str]] = defaultdict(list)
    for nlp in (_nlp_fr, _nlp_en):
        if nlp is None:
            continue
        doc = nlp(text[:5000])  # cap per chunk
        for ent in doc.ents:
            label = ent.label_
            name  = ent.text.strip()
            if len(name) < 2 or name.isdigit():
                continue
            result[label].append(name)
    return result


def _detect_interests(text: str) -> list[str]:
    t = text.lower()
    found = []
    for topic, kws in INTEREST_KEYWORDS.items():
        if any(kw in t for kw in kws):
            found.append(topic)
    return found


def _has_lives_in_context(text: str) -> bool:
    t = text.lower()
    all_triggers = LIVES_IN_TRIGGERS["fr"] + LIVES_IN_TRIGGERS["en"]
    return any(w in t for w in all_triggers)


def classify_relationship(all_text: str, msg_count: int) -> tuple[str, float]:
    """
    Classifies the relationship based on conversation text.
    Returns (label, confidence).
    """
    text = all_text.lower()
    scores = {"PARTNER": 0.0, "FAMILY": 0.0, "COLLEAGUE": 0.0}

    for label, langs in REL_CLASSIFIER_KEYWORDS.items():
        for lang, kws in langs.items():
            for kw in kws:
                if kw in text:
                    scores[label] += text.count(kw)

    # Heuristic for FRIEND
    # If high frequency but no specific marker, it's likely a friend
    if max(scores.values()) < 2:
        if msg_count > 50:
            return "FRIEND_OF", min(0.4 + (msg_count / 1000), 0.9)
        return "MET", 0.5

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    
    # Confidence calculation
    conf = min(0.3 + (best_score * 0.1) + (msg_count / 5000), 0.95)
    
    mapping = {
        "PARTNER": "PARTNER_OF",
        "FAMILY": "FAMILY_OF",
        "COLLEAGUE": "COLLEAGUE_OF"
    }
    return mapping.get(best_label, "FRIEND_OF"), conf


# ── Main extraction logic ──────────────────────────────────────────────────────

def extract(chunks: list[dict], self_name: str, dry_run: bool = False,
            client=None) -> dict[str, int]:
    _load_nlp()

    # 1. Aggregate by conversation ID
    # In the raw JSON: "conversation" is the ID, "sender_name" is the sender, "text" is content
    conv_id_to_msgs = defaultdict(list)
    conv_id_to_friends = defaultdict(set)
    
    # Canonicalise 'ME' as the user identity
    ME_NODE = "ME"
    
    # Track the real name used by 'self' in this dataset
    self_aliases = Counter()
    
    for c in chunks:
        cid = c.get("conversation", "unknown")
        snd = c.get("sender_name", "")
        txt = c.get("text", c.get("content", ""))
        
        if txt:
            c["content"] = txt
            conv_id_to_msgs[cid].append(c)
            # Find the name that corresponds to the 'self' (the one who isn't the friend)
            # but wait, we need to know who the friend is first.
            # In Facebook export, 'friend' is often implicit or in metadata.
            # Let's collect all senders and the metadata sender.
    
    # Refined sender mapping
    for cid, msgs in conv_id_to_msgs.items():
        # The friend is usually the one who appears in the conversation title or metadata
        # In our ingest format, if we don't have 'friend' explicitly, we'll have to infer.
        # But wait, self_name was passed in. If self_name is "ME", it's generic.
        # If the user passed a real name, we use that to find which sender is 'ME'.
        pass

    # Actually, simpler: Any sender who matches self_name OR doesn't match the known friends
    # is considered 'ME'.
    
    # First, collect all unique senders per conversation to find 'the other part'
    all_senders = Counter()
    for c in chunks:
        if c.get("sender_name"): all_senders[c.get("sender_name")] += 1
    
    # If self_name is "ME" or empty, we try to detect the 'owner' (most frequent sender)
    detected_self = self_name if self_name and self_name != "ME" else ""
    if not detected_self:
        # Heuristic: The owner is usually the one present in almost all conversations
        # or the most frequent sender overall.
        if all_senders:
            detected_self = all_senders.most_common(1)[0][0]
            print(f"🕵️ Auto-detected self-name: {detected_self!r}", flush=True)

    for cid, msgs in conv_id_to_msgs.items():
        for m in msgs:
            snd = m.get("sender_name", "")
            if snd and snd.lower() == detected_self.lower():
                # This is ME
                pass
            elif snd:
                if snd.lower() != "facebook user":
                    conv_id_to_friends[cid].add(snd)

    triples: list[dict] = []
    counters: Counter   = Counter()

    # Anchor 'ME' in the graph with their real name as an alias
    if detected_self:
        triples.append({
            "from_label": "Person", "from_name": ME_NODE,
            "rel_type":   "ALIAS_OF",
            "to_label":   "Alias", "to_name": detected_self,
            "props":      {"source": "facebook"}
        })
        counters["ALIAS_OF"] += 1

    print(f"🧵 Analyzing {len(conv_id_to_msgs)} conversation threads...", flush=True)

    # 2. Global trackers for temporal logic
    residence_evidence = defaultdict(list) # place -> [dates]
    
    for i, (cid, msgs) in enumerate(conv_id_to_msgs.items()):
        friends = list(conv_id_to_friends[cid])
        if not friends:
            continue
        
        # Sort messages by date
        sorted_msgs = sorted(msgs, key=lambda x: x.get("date", "9999"))
        full_text   = " ".join(m.get("content", "") for m in sorted_msgs)
        msg_count   = len(msgs)
        since_date  = sorted_msgs[0].get("date", "")[:10]

        # Process each friend
        for friend in friends:
            rel_label, confidence = classify_relationship(full_text, msg_count)
            triples.append({
                "from_label": "Person", "from_name": ME_NODE,
                "rel_type":   rel_label,
                "to_label":   "Person", "to_name": friend,
                "props":      {"confidence": confidence, "since": since_date, "evidence": f"{msg_count} msgs"},
            })
            counters[rel_label] += 1
            print(f"[REL] {ME_NODE!r} --{rel_label}--> {friend!r} (conf: {confidence:.2f})", flush=True)

        # Interest detection
        interests = _detect_interests(full_text)
        for topic in set(interests):
            count = interests.count(topic)
            triples.append({
                "from_label": "Person", "from_name": ME_NODE,
                "rel_type":   "INTERESTED_IN",
                "to_label":   "Interest", "to_name": topic,
                "props":      {"confidence": 0.8, "weight": min(count/5.0, 1.0)},
            })
            counters["INTERESTED_IN"] += 1
            print(f"[ENT] Detected Interest: {topic!r}", flush=True)

        # NER for Companies & Mentions
        all_ents = defaultdict(Counter)
        for m in sorted_msgs:
            m_text = m.get("content", "")
            if len(m_text) < 10: continue
            
            ents = _ner_entities(m_text)
            for etype, names in ents.items():
                for name in names:
                    all_ents[etype][name] += 1
            
            # Evidence for LIVES_IN (collect globally)
            if _has_lives_in_context(m_text):
                for place in ents.get("GPE", []) + ents.get("LOC", []):
                    residence_evidence[place].append(m.get("date", "")[:10])
            
            # Record visitation evidence (if not a residence, we'll handle below)
            for place in set(ents.get("GPE", []) + ents.get("LOC", [])):
                triples.append({
                    "from_label": "Person", "from_name": ME_NODE,
                    "rel_type":   "VISITED",
                    "to_label":   "Place",  "to_name": place,
                    "props":      {"since": m.get("date", "")[:10], "confidence": 0.6},
                })
                counters["VISITED"] += 1


        # Mentions
        for p_mention, count in all_ents["PER"].items():
            if p_mention.lower() in (self_name.lower(), detected_self.lower()) or any(f.lower() == p_mention.lower() for f in friends):
                continue
            if count >= 2:
                triples.append({
                    "from_label": "Person", "from_name": ME_NODE,
                    "rel_type":   "MET",
                    "to_label":   "Person", "to_name": p_mention,
                    "props":      {"confidence": min(0.2 + (count/10), 0.8), "source": "mention"},
                })
                counters["MET"] += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(conv_id_to_msgs):
            pct = int((i + 1) / len(conv_id_to_msgs) * 100)
            print(f"PROGRESS: {pct}% | {i+1}/{len(conv_id_to_msgs)} conversations", flush=True)

    # 3. Global Post-processing for LIVES_IN
    print("🏠 Chaining residence history...", flush=True)
    res_candidates = []
    for place, dates in residence_evidence.items():
        if len(dates) >= 2 or len(set(d[:4] for d in dates)) >= 2: # multi-mention or multi-year
            dates.sort()
            res_candidates.append({
                "name":  place,
                "since": dates[0],
                "last":  dates[-1],
                "count": len(dates)
            })
    
    res_candidates.sort(key=lambda x: x["since"])
    for i, res in enumerate(res_candidates):
        props = {
            "since": res["since"],
            "confidence": min(0.5 + (res["count"]/20), 0.98),
            "evidence_count": res["count"]
        }
        # If there's a subsequent residence, set the 'until' of this one
        if i < len(res_candidates) - 1:
            props["until"] = res_candidates[i+1]["since"]
        
        triples.append({
            "from_label": "Person", "from_name": ME_NODE,
            "rel_type":   "LIVES_IN",
            "to_label":   "Place",  "to_name": res["name"],
            "props":      props,
        })
        counters["LIVES_IN"] += 1
        print(f"[REL] {ME_NODE!r} --LIVES_IN--> {res['name']!r} (since {props['since']}" + 
              (f", until {props['until']}" if "until" in props else "") + ")", flush=True)

    # Deduplicate triples by (from, rel, to, since) if since is in props
    seen    = set()
    unique  = []
    for t in triples:
        p = t.get("props", {})
        # User requested including time in key
        key = (t["from_name"], t["rel_type"], t["to_name"], p.get("since", ""))
        if key not in seen:
            seen.add(key)
            unique.append(t)

    print(f"\n📊 Extracted {len(unique)} unique triples:", flush=True)
    for rel, cnt in sorted(counters.items()):
        print(f"   {rel:20s} {cnt:>6,}", flush=True)

    # ── Interest summary for spider chart ──────────────────────────────────────
    import json as _json
    interest_totals: Counter = Counter()
    for t in unique:
        if t["rel_type"] == "INTERESTED_IN":
            interest_totals[t["to_name"]] += 1

    if interest_totals:
        total = sum(interest_totals.values()) or 1
        scores = {k: round(v / total * 100, 1) for k, v in interest_totals.most_common()}
        print(f"INTERESTS_CHART: {_json.dumps(scores)}", flush=True)

    if dry_run:
        print("\n🔍 DRY RUN — completion", flush=True)
    elif client is not None:
        client.ensure_constraints()
        print(f"\n✍️  Writing {len(unique)} triples to Neo4j…", flush=True)
        BATCH = 200
        for i in range(0, len(unique), BATCH):
            client.batch_merge_relations(unique[i:i+BATCH])
            pct = int((i + BATCH) / len(unique) * 100)
            if pct > 100: pct = 100
            print(f"PROGRESS: {pct}% | Writing {min(i+BATCH, len(unique))}/{len(unique)}", flush=True)
        print("✅ Done.", flush=True)

    return dict(counters)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Extract KG triples from Facebook messages")
    p.add_argument("--json-file",   default="facebook_messages.json",
                   help="Path to facebook_messages.json")
    p.add_argument("--self-name",   default="ME",
                   help="Your name in the graph (auto-detected if omitted or 'ME')")
    p.add_argument("--limit",       type=int, default=0,
                   help="Process only the first N chunks (0 = all)")
    p.add_argument("--dry-run",     action="store_true",
                   help="Print triples without writing to Neo4j")
    p.add_argument("--neo4j-uri",   default=os.environ.get("NEO4J_URI",      "bolt://localhost:7687"))
    p.add_argument("--neo4j-user",  default=os.environ.get("NEO4J_USER",     "neo4j"))
    p.add_argument("--neo4j-pass",  default=os.environ.get("NEO4J_PASSWORD", "password"))
    return p.parse_args()


def _auto_detect_self(chunks: list[dict]) -> str:
    """
    The most frequent 'friend' value is actually other people.
    The owner is the sender who appears in conversation metadata most often.
    We use the chunk metadata field if present, else fall back to the most
    frequent distinct value listed under a special sentinel.
    """
    # facebook_messages.json from ingest_facebook_messages stores the
    # conversation partner as 'friend'. The self-name is NOT stored there.
    # Return empty string — caller will use fallback or prompt.
    return ""


def main():
    args = parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"❌ File not found: {json_path}", flush=True)
        sys.exit(1)

    print(f"📂 Loading {json_path} …", flush=True)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept both a bare list of chunk dicts and a wrapped {"chunks": [...]} format
    if isinstance(data, list):
        chunks = data
    elif isinstance(data, dict):
        chunks = data.get("chunks", data.get("messages", []))
    else:
        print("❌ Unexpected JSON format.", flush=True)
        sys.exit(1)

    if args.limit:
        chunks = chunks[:args.limit]

    print(f"📦 {len(chunks):,} chunks loaded.", flush=True)

    # Self-name
    self_name = args.self_name.strip()
    if not self_name:
        self_name = os.environ.get("SELF_NAME", "").strip()
    if not self_name:
        print("⚠️  --self-name not set. Using 'Me' as placeholder. "
              "Set --self-name 'Your Full Name' for a better graph.", flush=True)
        self_name = "Me"

    print(f"👤 Self: {self_name!r}", flush=True)

    if args.dry_run:
        print("🔍 DRY RUN — no writes to Neo4j\n", flush=True)
        extract(chunks, self_name, dry_run=True)
    else:
        with Neo4jClient(args.neo4j_uri, args.neo4j_user, args.neo4j_pass) as client:
            if not client.verify():
                print(f"❌ Cannot connect to Neo4j at {args.neo4j_uri}", flush=True)
                sys.exit(1)
            print(f"✅ Connected to Neo4j at {args.neo4j_uri}", flush=True)
            extract(chunks, self_name, dry_run=False, client=client)


if __name__ == "__main__":
    main()
