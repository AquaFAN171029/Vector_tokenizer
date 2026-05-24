from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

try:
    from .geom_tokenizer import tokenize_geometry
    from .pos_tokenizer import tokenize_position
    from .style_tokenizer import tokenize_style
    from .text_tokenizer import tokenize_text
    from .type_tokenizer import tokenize_type
except ImportError:
    from geom_tokenizer import tokenize_geometry
    from pos_tokenizer import tokenize_position
    from style_tokenizer import tokenize_style
    from text_tokenizer import tokenize_text
    from type_tokenizer import tokenize_type


def _clean_group(value: object) -> str:
    text = str(value or "NO_GROUP").strip()
    text = re.sub(r"\s+", "_", text)
    return re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", text) or "NO_GROUP"


@dataclass(frozen=True)
class TokenizerConfig:
    geom_bins: int = 256
    pos_bins: int = 256
    width_bins: int = 32
    opacity_bins: int = 11
    min_text_freq: int = 1


class PrimitiveTokenizer:
    def __init__(self, viewbox: List[float], text_vocab: set[str], config: TokenizerConfig):
        self.viewbox = viewbox
        self.text_vocab = text_vocab
        self.config = config

    def tokenize(self, primitive: Dict[str, Any]) -> Dict[str, Any]:
        meta = primitive.get("meta") or {}
        group = meta.get("parent_group_id")
        return {
            "primitive_id": primitive.get("primitive_id"),
            "type_token": tokenize_type(primitive.get("type")),
            "geom_tokens": tokenize_geometry(primitive, self.viewbox, self.config.geom_bins),
            "geom_mask": list((primitive.get("geometry") or {}).get("geom_mask") or []),
            "style_tokens": tokenize_style(
                primitive.get("style") or {},
                self.config.width_bins,
                self.config.opacity_bins,
            ),
            "pos_tokens": tokenize_position(
                primitive.get("position") or {},
                self.viewbox,
                self.config.pos_bins,
            ),
            "text_tokens": tokenize_text(primitive.get("text") or {}, self.text_vocab),
            "group_token": f"GROUP_{_clean_group(group)}",
        }
