from __future__ import annotations

import math
from typing import Any, Dict, List


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _quantize01(value: float, bins: int) -> int:
    value = max(0.0, min(1.0, value))
    return round(value * (bins - 1))


def _sincos(value: object) -> float:
    return (_safe_float(value) + 1.0) / 2.0


def _flag(value: object) -> float:
    return 1.0 if _safe_float(value) >= 0.5 else 0.0


def _normalize_coords(typ: str, coords: List[Any], viewbox: List[float]) -> List[float]:
    vx, vy, vw, vh = viewbox
    vw = vw or 1.0
    vh = vh or 1.0
    diag = math.hypot(vw, vh) or 1.0
    values = [_safe_float(x) for x in coords[:8]]
    values += [0.0] * (8 - len(values))

    if typ == "LINE":
        return [
            (values[0] - vx) / vw,
            (values[1] - vy) / vh,
            (values[2] - vx) / vw,
            (values[3] - vy) / vh,
            values[4] / diag,
            _sincos(values[5]),
            _sincos(values[6]),
            0.0,
        ]
    if typ == "POLYLINE":
        return [
            (values[0] - vx) / vw,
            (values[1] - vy) / vh,
            (values[2] - vx) / vw,
            (values[3] - vy) / vh,
            values[4] / diag,
            min(values[5], 255.0) / 255.0,
            _flag(values[6]),
            0.0,
        ]
    if typ == "ARC":
        return [
            (values[0] - vx) / vw,
            (values[1] - vy) / vh,
            values[2] / vw,
            values[3] / vh,
            _sincos(values[4]),
            _sincos(values[5]),
            _sincos(values[6]),
            _sincos(values[7]),
        ]
    if typ in {"TEXT", "ANNOTATION"}:
        return [
            (values[0] - vx) / vw,
            (values[1] - vy) / vh,
            values[2] / vw,
            values[3] / vh,
            _sincos(values[4]),
            _sincos(values[5]),
            values[6] / vh,
            0.0,
        ]
    return [0.0] * 8


def tokenize_geometry(primitive: Dict[str, Any], viewbox: List[float], bins: int) -> List[str]:
    typ = str(primitive.get("type") or "UNK").upper()
    geometry = primitive.get("geometry") or {}
    coords = geometry.get("coords") or []
    mask = geometry.get("geom_mask") or []
    norm = _normalize_coords(typ, coords, viewbox)

    tokens: List[str] = []
    for idx, value in enumerate(norm):
        valid = idx < len(mask) and int(mask[idx]) == 1
        tokens.append(f"G_{_quantize01(value, bins)}" if valid else "PAD")
    return tokens
