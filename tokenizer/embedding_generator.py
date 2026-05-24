from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class EmbeddingConfig:
    output_dim: int = 256
    type_dim: int = 32
    geom_dim: int = 128
    style_dim: int = 32
    pos_dim: int = 64
    text_dim: int = 64
    group_dim: int = 16
    seed: int = 13
    decimals: int = 6

    @property
    def concat_dim(self) -> int:
        return (
            self.type_dim
            + self.geom_dim
            + self.style_dim
            + self.pos_dim
            + self.text_dim
            + self.group_dim
        )

    def validate(self) -> None:
        allowed = {16, 32, 64, 128}
        dims = {
            "type_dim": self.type_dim,
            "geom_dim": self.geom_dim,
            "style_dim": self.style_dim,
            "pos_dim": self.pos_dim,
            "text_dim": self.text_dim,
            "group_dim": self.group_dim,
        }
        bad = {name: dim for name, dim in dims.items() if dim not in allowed}
        if bad:
            raise ValueError(f"Channel dimensions must be one of {sorted(allowed)}: {bad}")
        if self.output_dim != 256:
            raise ValueError("This requirement-matched embedder currently fuses to 256 dimensions.")


class EmbeddingGenerator:
    def __init__(self, config: EmbeddingConfig):
        config.validate()
        self.config = config
        self._token_cache: Dict[tuple[str, int], List[float]] = {}
        self._projection = self._make_projection(config.concat_dim, config.output_dim)

    def embed_primitive(self, primitive: Dict[str, Any]) -> Dict[str, Any]:
        channel_embeddings = {
            "type_emb": self._embed_tokens([primitive.get("type_token")], self.config.type_dim),
            "geom_emb": self._embed_tokens(primitive.get("geom_tokens") or [], self.config.geom_dim),
            "style_emb": self._embed_tokens(primitive.get("style_tokens") or [], self.config.style_dim),
            "pos_emb": self._embed_tokens(primitive.get("pos_tokens") or [], self.config.pos_dim),
            "text_emb": self._embed_tokens(primitive.get("text_tokens") or [], self.config.text_dim),
            "group_emb": self._embed_tokens([primitive.get("group_token")], self.config.group_dim),
        }
        concat = [
            x
            for name in ("type_emb", "geom_emb", "style_emb", "pos_emb", "text_emb", "group_emb")
            for x in channel_embeddings[name]
        ]
        fused = self._project(concat)
        return {
            "primitive_id": primitive.get("primitive_id"),
            "group_token": primitive.get("group_token"),
            "embedding": [round(x, self.config.decimals) for x in fused],
        }

    def embed_sequence(self, tokenized: Dict[str, Any]) -> Dict[str, Any]:
        primitives = tokenized.get("tokenized_primitives")
        if not isinstance(primitives, list):
            raise ValueError("Tokenized JSON must contain a 'tokenized_primitives' list.")
        embeddings = [self.embed_primitive(primitive) for primitive in primitives]
        return {
            "source_file": tokenized.get("source_file"),
            "viewBox": tokenized.get("viewBox"),
            "num_primitives": len(embeddings),
            "embedding_config": asdict(self.config),
            "fusion_rule": "z_i = L2_normalize(tanh(W * concat(type_emb, geom_emb, style_emb, pos_emb, text_emb, group_emb))))",
            "embeddings": embeddings,
        }

    def _embed_tokens(self, tokens: Iterable[Any], dim: int) -> List[float]:
        vectors = [self._token_vector(str(token), dim) for token in tokens if token and token != "PAD"]
        if not vectors:
            return [0.0] * dim
        scale = 1.0 / len(vectors)
        return [sum(vec[i] for vec in vectors) * scale for i in range(dim)]

    def _token_vector(self, token: str, dim: int) -> List[float]:
        key = (token, dim)
        if key in self._token_cache:
            return self._token_cache[key]

        digest = hashlib.sha256(f"{self.config.seed}:{token}:{dim}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        vec = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        vec = [x / norm for x in vec]
        self._token_cache[key] = vec
        return vec

    def _make_projection(self, in_dim: int, out_dim: int) -> List[List[float]]:
        rng = random.Random(f"{self.config.seed}:projection:{in_dim}:{out_dim}")
        scale = 1.0 / math.sqrt(in_dim)
        return [[rng.uniform(-scale, scale) for _ in range(in_dim)] for _ in range(out_dim)]

    def _project(self, vec: List[float]) -> List[float]:
        out = []
        for row in self._projection:
            out.append(math.tanh(sum(w * x for w, x in zip(row, vec))))
        norm = math.sqrt(sum(x * x for x in out)) or 1.0
        return [x / norm for x in out]


def load_tokenized_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_embedding_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
