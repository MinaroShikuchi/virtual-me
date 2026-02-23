"""
services/mapping_service.py — Conversation name mapping loader.
"""
from __future__ import annotations

import json
from pathlib import Path
from config import NAME_MAPPING_FILE

_id_to_name: dict | None = None
_name_to_id: dict | None = None


def get_mappings() -> tuple[dict, dict]:
    """Return (id_to_name, name_to_id) dicts, loading from file if needed."""
    global _id_to_name, _name_to_id
    if _id_to_name is not None:
        return _id_to_name, _name_to_id

    _id_to_name, _name_to_id = {}, {}
    try:
        if Path(NAME_MAPPING_FILE).exists():
            with open(NAME_MAPPING_FILE, "r", encoding="utf-8") as f:
                _id_to_name = json.load(f)
            for cid, name in _id_to_name.items():
                if name:
                    _name_to_id[name.lower()] = cid
    except Exception as e:
        print(f"Name mapping warning: {e}")

    return _id_to_name, _name_to_id


def invalidate():
    """Clear cached mappings."""
    global _id_to_name, _name_to_id
    _id_to_name = None
    _name_to_id = None
