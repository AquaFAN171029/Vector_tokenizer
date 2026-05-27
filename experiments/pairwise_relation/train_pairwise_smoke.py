#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PIPELINE_ROOT.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("Install PyTorch before running pairwise training smoke tests.") from exc

from modeling.prediction_heads import TypePairRelationHead


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    candidate = PIPELINE_ROOT / p
    if candidate.exists():
        return candidate
    candidate = WORKSPACE_ROOT / p
    if candidate.exists():
        return candidate
    return p


def embedding_path_for_source(source_file: str, embeddings_dir: Path) -> Path:
    name = Path(source_file).name
    if name.endswith(".parsed.json"):
        name = name.replace(".parsed.json", ".embeddings.json")
    elif name.endswith(".json"):
        name = name.replace(".json", ".embeddings.json")
    else:
        name = f"{Path(name).stem}.embeddings.json"
    return embeddings_dir / name


def load_rows(labels_jsonl: Path, max_samples: int | None, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    total = 0
    with labels_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            if max_samples is None:
                rows.append(row)
            elif len(rows) < max_samples:
                rows.append(row)
            else:
                idx = rng.randrange(total)
                if idx < max_samples:
                    rows[idx] = row
    rng.shuffle(rows)
    print(f"Scanned label rows: {total}", flush=True)
    return rows


def load_embedding_cache(embeddings_dir: Path, source_files: List[str]) -> Dict[str, torch.Tensor]:
    cache: Dict[str, torch.Tensor] = {}
    unique_sources = sorted(set(source_files))
    print(f"Loading embedding files: {len(unique_sources)}", flush=True)
    for idx, source_file in enumerate(unique_sources, 1):
        path = embedding_path_for_source(source_file, embeddings_dir)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        vectors = [row["embedding"] for row in data.get("embeddings", [])]
        if vectors:
            cache[source_file] = torch.tensor(vectors, dtype=torch.float32)
        if idx % 100 == 0:
            print(f"Loaded embedding files: {idx}/{len(unique_sources)}", flush=True)
    return cache


class PairwiseSmokeDataset(Dataset):

    def __init__(self, rows: List[Dict[str, Any]], embedding_cache: Dict[str, torch.Tensor]) -> None:
        self.samples = []
        self.embedding_cache = embedding_cache
        for row in rows:
            embeddings = embedding_cache.get(row["source_file"])
            if embeddings is None:
                continue
            i = int(row["i"])
            j = int(row["j"])
            if i < len(embeddings) and j < len(embeddings):
                self.samples.append(row)
        if not self.samples:
            raise ValueError("No usable samples. Check that labels and embeddings come from the same pipeline output.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.samples[idx]
        embeddings = self.embedding_cache[row["source_file"]]
        return {
            "e_i": embeddings[int(row["i"])],
            "e_j": embeddings[int(row["j"])],
            "target": torch.tensor(row["label_vector"], dtype=torch.float32),
            "type_pair": row["type_pair"],
        }


def collate_same_type_pair(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    type_pairs = {item["type_pair"] for item in batch}
    if len(type_pairs) != 1:
        raise ValueError("Batch contains mixed type_pair values.")
    return {
        "e_i": torch.stack([item["e_i"] for item in batch]),
        "e_j": torch.stack([item["e_j"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "type_pair": batch[0]["type_pair"],
    }


def train_type_pair(
    model: TypePairRelationHead,
    rows: List[Dict[str, Any]],
    embedding_cache: Dict[str, torch.Tensor],
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Dict[str, float]:
    dataset = PairwiseSmokeDataset(rows, embedding_cache)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_same_type_pair)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    last_loss = 0.0
    steps = 0
    model.train()
    for _ in range(epochs):
        for batch in loader:
            e_i = batch["e_i"].to(device)
            e_j = batch["e_j"].to(device)
            target = batch["target"].to(device)
            logits = model(e_i, e_j, batch["type_pair"])
            loss = criterion(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            steps += 1
    return {"samples": len(dataset), "steps": steps, "last_loss": last_loss}


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke train the typed pairwise relation head.")
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    embeddings_dir = resolve_path(args.embeddings_dir)
    labels_jsonl = resolve_path(args.labels_jsonl)
    rows = load_rows(labels_jsonl, args.max_samples, args.seed)
    embedding_cache = load_embedding_cache(embeddings_dir, [row["source_file"] for row in rows])
    if not embedding_cache:
        raise ValueError(f"No embedding files loaded from {embeddings_dir}")

    embedding_dim = int(next(iter(embedding_cache.values())).shape[-1])
    device = torch.device(args.device)
    model = TypePairRelationHead(embedding_dim=embedding_dim, hidden_dim=args.hidden_dim).to(device)

    rows_by_type_pair: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        rows_by_type_pair.setdefault(row["type_pair"], []).append(row)

    print(f"Loaded rows: {len(rows)}")
    print(f"Loaded embedding files: {len(embedding_cache)}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Type-pair counts: {dict(Counter(row['type_pair'] for row in rows))}")

    for type_pair, type_rows in sorted(rows_by_type_pair.items()):
        if type_pair not in model.relation_labels_by_type_pair:
            print(f"Skipping unsupported type_pair: {type_pair}")
            continue
        result = train_type_pair(
            model=model,
            rows=type_rows,
            embedding_cache=embedding_cache,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )
        print(
            f"{type_pair}: samples={result['samples']} "
            f"steps={result['steps']} last_loss={result['last_loss']:.6f}"
        )

    print("Smoke training finished.")


if __name__ == "__main__":
    main()
