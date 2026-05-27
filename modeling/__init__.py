"""Reusable model heads for vector primitive embeddings."""

from .prediction_heads import (
    DEFAULT_RELATION_LABELS,
    TYPE_PAIR_RELATION_LABELS,
    PairwiseRelationHead,
    TypePairRelationHead,
)

__all__ = [
    "DEFAULT_RELATION_LABELS",
    "TYPE_PAIR_RELATION_LABELS",
    "PairwiseRelationHead",
    "TypePairRelationHead",
]
