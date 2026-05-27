from __future__ import annotations

from typing import Mapping, Sequence

try:
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "The modeling heads require PyTorch. Install torch before training prediction heads."
    ) from exc


DEFAULT_RELATION_LABELS = ("parallel", "perpendicular", "joined", "none")

TYPE_PAIR_RELATION_LABELS = {
    "LINE-LINE": ("connected", "intersect", "collinear_overlap", "perpendicular", "parallel", "near"),
    "LINE-ARC": ("connected", "intersect", "tangent", "near"),
    "LINE-POLYLINE": (
        "connected",
        "intersect",
        "overlap_segment",
        "perpendicular_to_segment",
        "parallel_to_segment",
        "near",
    ),
    "ARC-ARC": ("connected", "intersect", "overlap", "concentric", "tangent", "near"),
    "ARC-POLYLINE": ("connected", "intersect", "tangent_to_segment", "near"),
    "POLYLINE-POLYLINE": (
        "connected",
        "intersect",
        "overlap_segment",
        "perpendicular_segment",
        "parallel_segment",
        "containment",
        "near",
    ),
}


def pairwise_features(e_i: Tensor, e_j: Tensor) -> Tensor:

    if e_i.shape != e_j.shape:
        raise ValueError(f"e_i and e_j must have the same shape")
    return torch.cat([e_i, e_j, torch.abs(e_i - e_j), e_i * e_j], dim=-1)


class PairwiseRelationHead(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 256,
        num_relations: int | None = None,
        relation_labels: Sequence[str] = DEFAULT_RELATION_LABELS,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.embedding_dim = embedding_dim
        self.relation_labels = tuple(relation_labels)
        self.num_relations = num_relations or len(self.relation_labels)
        if self.num_relations <= 1:
            raise ValueError("num_relations must be greater than 1.")

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_relations),
        )
# predict relation
    def forward(self, e_i: Tensor, e_j: Tensor) -> Tensor:

        if e_i.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {e_i.shape[-1]}.")
        features = pairwise_features(e_i, e_j)
        return self.mlp(features)

    def forward_from_sequence(self, embeddings: Tensor, pair_indices: Tensor) -> Tensor:

        if embeddings.ndim != 3:
            raise ValueError(f"embeddings must have shape [B, N, D], got {embeddings.shape}.")
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {embeddings.shape[-1]}.")
        if pair_indices.ndim == 2:
            pair_indices = pair_indices.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
        if pair_indices.ndim != 3 or pair_indices.shape[-1] != 2:
            raise ValueError(f"pair_indices must have shape [B, P, 2] or [P, 2], got {pair_indices.shape}.")
        if pair_indices.shape[0] != embeddings.shape[0]:
            raise ValueError("pair_indices batch size must match embeddings batch size.")

        pair_indices = pair_indices.to(device=embeddings.device, dtype=torch.long)
        e_i = self._gather_nodes(embeddings, pair_indices[..., 0])
        e_j = self._gather_nodes(embeddings, pair_indices[..., 1])
        return self.forward(e_i, e_j)

    @staticmethod
    def _gather_nodes(embeddings: Tensor, indices: Tensor) -> Tensor:
        expanded = indices.unsqueeze(-1).expand(-1, -1, embeddings.shape[-1])
        return torch.gather(embeddings, dim=1, index=expanded)


class TypePairRelationHead(nn.Module):
    """Multi-label relation head with a separate classifier for each geometric type pair."""

    def __init__(
        self,
        embedding_dim: int = 256,
        relation_labels_by_type_pair: Mapping[str, Sequence[str]] = TYPE_PAIR_RELATION_LABELS,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.embedding_dim = embedding_dim
        self.relation_labels_by_type_pair = {
            type_pair: tuple(labels)
            for type_pair, labels in relation_labels_by_type_pair.items()
        }
        self.relation_to_id_by_type_pair = {
            type_pair: {label: idx for idx, label in enumerate(labels)}
            for type_pair, labels in self.relation_labels_by_type_pair.items()
        }

        self.classifiers = nn.ModuleDict(
            {
                self._module_key(type_pair): self._build_classifier(
                    input_dim=embedding_dim * 4,
                    hidden_dim=hidden_dim,
                    output_dim=len(labels),
                    dropout=dropout,
                )
                for type_pair, labels in self.relation_labels_by_type_pair.items()
            }
        )

    def forward(self, e_i: Tensor, e_j: Tensor, type_pair: str) -> Tensor:
        if type_pair not in self.relation_labels_by_type_pair:
            raise KeyError(f"Unknown type_pair: {type_pair}")
        if e_i.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {e_i.shape[-1]}.")
        features = pairwise_features(e_i, e_j)
        return self.classifiers[self._module_key(type_pair)](features)

    def forward_from_sequence(self, embeddings: Tensor, pair_indices: Tensor, type_pair: str) -> Tensor:
        if embeddings.ndim != 3:
            raise ValueError(f"embeddings must have shape [B, N, D], got {embeddings.shape}.")
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {embeddings.shape[-1]}.")
        if pair_indices.ndim == 2:
            pair_indices = pair_indices.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
        if pair_indices.ndim != 3 or pair_indices.shape[-1] != 2:
            raise ValueError(f"pair_indices must have shape [B, P, 2] or [P, 2], got {pair_indices.shape}.")
        if pair_indices.shape[0] != embeddings.shape[0]:
            raise ValueError("pair_indices batch size must match embeddings batch size.")

        pair_indices = pair_indices.to(device=embeddings.device, dtype=torch.long)
        e_i = PairwiseRelationHead._gather_nodes(embeddings, pair_indices[..., 0])
        e_j = PairwiseRelationHead._gather_nodes(embeddings, pair_indices[..., 1])
        return self.forward(e_i, e_j, type_pair)

    def labels_for(self, type_pair: str) -> tuple[str, ...]:
        if type_pair not in self.relation_labels_by_type_pair:
            raise KeyError(f"Unknown type_pair: {type_pair}")
        return self.relation_labels_by_type_pair[type_pair]

    @staticmethod
    def _module_key(type_pair: str) -> str:
        return type_pair.replace("-", "__")

    @staticmethod
    def _build_classifier(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
