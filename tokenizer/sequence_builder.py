from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _group_id(primitive: Dict[str, Any]) -> str:
    meta = primitive.get("meta") or {}
    return str(meta.get("parent_group_id") or "NO_GROUP")


def _pos_value(primitive: Dict[str, Any], key: str) -> float:
    try:
        return float((primitive.get("position") or {}).get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def build_sequence(primitives: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        primitives,
        key=lambda p: (
            _group_id(p),
            _pos_value(p, "cy"),
            _pos_value(p, "cx"),
            str(p.get("primitive_id") or ""),
        ),
    )

