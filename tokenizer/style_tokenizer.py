from __future__ import annotations

import re
from typing import Any, Dict, List


def _clean(value: object, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    text = re.sub(r"\s+", "_", text)
    return re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff(),#]+", "_", text)


def _quantize(value: object, max_value: float, bins: int) -> int:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = 0.0
    if max_value <= 0:
        return 0
    x = max(0.0, min(max_value, x))
    return round((x / max_value) * (bins - 1))


def tokenize_style(style: Dict[str, Any], width_bins: int, opacity_bins: int) -> List[str]:
    style = style or {}
    opacity = _quantize(style.get("opacity", 1.0), 1.0, opacity_bins)
    width = _quantize(style.get("stroke_width"), 10.0, width_bins)
    return [
        f"LAYER_{_clean(style.get('layer'), 'NO_LAYER')}",
        f"STROKE_{_clean(style.get('stroke'), 'NO_STROKE')}",
        f"WIDTH_{width}",
        f"FILL_{_clean(style.get('fill'), 'NO_FILL')}",
        f"OP_{opacity}",
    ]

