from __future__ import annotations

from typing import Any, Dict, List


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _quantize01(value: float, bins: int) -> int:
    value = max(0.0, min(1.0, value))
    return round(value * (bins - 1))


def tokenize_position(position: Dict[str, Any], viewbox: List[float], bins: int) -> List[str]:
    vx, vy, vw, vh = viewbox
    vw = vw or 1.0
    vh = vh or 1.0
    position = position or {}
    values = {
        "CX": (_safe_float(position.get("cx")) - vx) / vw,
        "CY": (_safe_float(position.get("cy")) - vy) / vh,
        "W": _safe_float(position.get("w")) / vw,
        "H": _safe_float(position.get("h")) / vh,
    }
    return [f"{name}_{_quantize01(value, bins)}" for name, value in values.items()]
