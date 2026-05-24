from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List


def _clean(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", "_", value)
    return re.sub(r"[^0-9a-z_\-\u4e00-\u9fff]+", "_", value).strip("_") or "EMPTY"


def build_text_vocab(primitives: Iterable[Dict[str, Any]], min_freq: int) -> set[str]:
    counts = Counter()
    for primitive in primitives:
        text = (primitive.get("text") or {}).get("normalized_text")
        if text:
            counts[str(text)] += 1
    return {text for text, count in counts.items() if count >= min_freq}


def tokenize_text(text_record: Dict[str, Any], vocab: set[str]) -> List[str]:
    text = (text_record or {}).get("normalized_text")
    if not text:
        return []
    text = str(text)
    return [f"TEXT_{_clean(text)}" if text in vocab else "UNK_TEXT"]

