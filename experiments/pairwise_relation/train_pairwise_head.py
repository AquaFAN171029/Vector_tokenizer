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
    raise ModuleNotFoundError("Install PyTorch before training prediction heads.") from exc

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


def output_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == PIPELINE_ROOT.name:
        return WORKSPACE_ROOT / p
    return PIPELINE_ROOT / p


def embedding_path_for_source(source_file: str, embeddings_dir: Path) -> Path:
    name = Path(source_file).name
    if name.endswith(".parsed.json"):
        name = name.replace(".parsed.json", ".embeddings.json")
    elif name.endswith(".json"):
        name = name.replace(".json", ".embeddings.json")
    else:
        name = f"{Path(name).stem}.embeddings.json"
    return embeddings_dir / name


def load_rows(labels_jsonl: Path, max_samples: int | None, seed: int, name: str) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
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
    print(f"{name}: scanned={total} selected={len(rows)}", flush=True)
    return rows


def load_embedding_cache(embeddings_dir: Path, rows_by_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
    source_files = sorted(
        {
            row["source_file"]
            for rows in rows_by_split.values()
            for row in rows
        }
    )
    cache: Dict[str, torch.Tensor] = {}
    print(f"Loading embedding files: {len(source_files)}", flush=True)
    for idx, source_file in enumerate(source_files, 1):
        path = embedding_path_for_source(source_file, embeddings_dir)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        vectors = [row["embedding"] for row in data.get("embeddings", [])]
        if vectors:
            cache[source_file] = torch.tensor(vectors, dtype=torch.float32)
        if idx % 100 == 0:
            print(f"Loaded embedding files: {idx}/{len(source_files)}", flush=True)
    if not cache:
        raise ValueError(f"No embedding files loaded from {embeddings_dir}")
    return cache


class PairwiseDataset(Dataset):

    def __init__(self, rows: List[Dict[str, Any]], embedding_cache: Dict[str, torch.Tensor]) -> None:
        self.rows = []
        self.embedding_cache = embedding_cache
        for row in rows:
            embeddings = embedding_cache.get(row["source_file"])
            if embeddings is None:
                continue
            i = int(row["i"])
            j = int(row["j"])
            if i < len(embeddings) and j < len(embeddings):
                self.rows.append(row)
        if not self.rows:
            raise ValueError("No usable rows after matching labels to embeddings.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
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


def group_rows_by_type_pair(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["type_pair"], []).append(row)
    return grouped


def make_loaders(
    grouped_rows: Dict[str, List[Dict[str, Any]]],
    embedding_cache: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool,
) -> Dict[str, DataLoader]:
    loaders = {}
    for type_pair, rows in grouped_rows.items():
        dataset = PairwiseDataset(rows, embedding_cache)
        loaders[type_pair] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_same_type_pair,
        )
    return loaders


def train_epoch(
    model: TypePairRelationHead,
    loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_steps = 0
    for type_pair, loader in sorted(loaders.items()):
        if type_pair not in model.relation_labels_by_type_pair:
            continue
        for batch in loader:
            e_i = batch["e_i"].to(device)
            e_j = batch["e_j"].to(device)
            target = batch["target"].to(device)
            logits = model(e_i, e_j, type_pair)
            loss = criterion(logits, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = int(target.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size
            total_steps += 1
    return {
        "loss": total_loss / max(total_samples, 1),
        "samples": total_samples,
        "steps": total_steps,
    }


@torch.no_grad()
def evaluate(
    model: TypePairRelationHead,
    loaders: Dict[str, DataLoader],
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0
    by_type_pair: Dict[str, Dict[str, float]] = {}

    for type_pair, loader in sorted(loaders.items()):
        if type_pair not in model.relation_labels_by_type_pair:
            continue
        split_loss = 0.0
        split_samples = 0
        split_tp = split_fp = split_fn = 0.0
        for batch in loader:
            e_i = batch["e_i"].to(device)
            e_j = batch["e_j"].to(device)
            target = batch["target"].to(device)
            logits = model(e_i, e_j, type_pair)
            loss = criterion(logits, target)
            pred = (torch.sigmoid(logits) >= threshold).float()

            tp = float((pred * target).sum().detach().cpu())
            fp = float((pred * (1.0 - target)).sum().detach().cpu())
            fn = float(((1.0 - pred) * target).sum().detach().cpu())
            batch_size = int(target.shape[0])

            split_loss += float(loss.detach().cpu()) * batch_size
            split_samples += batch_size
            split_tp += tp
            split_fp += fp
            split_fn += fn

        by_type_pair[type_pair] = metrics_from_counts(
            loss=split_loss / max(split_samples, 1),
            samples=split_samples,
            true_positive=split_tp,
            false_positive=split_fp,
            false_negative=split_fn,
        )
        total_loss += split_loss
        total_samples += split_samples
        true_positive += split_tp
        false_positive += split_fp
        false_negative += split_fn

    out = metrics_from_counts(
        loss=total_loss / max(total_samples, 1),
        samples=total_samples,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )
    out["by_type_pair"] = by_type_pair
    return out


def metrics_from_counts(
    loss: float,
    samples: int,
    true_positive: float,
    false_positive: float,
    false_negative: float,
) -> Dict[str, float]:
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "loss": loss,
        "samples": samples,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
    }


def save_checkpoint(
    path: Path,
    model: TypePairRelationHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "history": history,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: TypePairRelationHead,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, List[Dict[str, Any]]]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    last_epoch = int(checkpoint.get("epoch") or 0)
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    history = list(checkpoint.get("history") or [])
    return last_epoch, best_val_loss, history


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the typed pairwise relation prediction head.")
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--train-labels-jsonl", required=True)
    parser.add_argument("--val-labels-jsonl", required=True)
    parser.add_argument("--output-dir", default="outputs/checkpoints/pairwise_relation_head")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Resume from a checkpoint. --epochs is interpreted as the target total epoch count.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    embeddings_dir = resolve_path(args.embeddings_dir)
    train_labels = resolve_path(args.train_labels_jsonl)
    val_labels = resolve_path(args.val_labels_jsonl)
    out_dir = output_path(args.output_dir)

    train_rows = load_rows(train_labels, args.max_train_samples, args.seed, "train")
    val_rows = load_rows(val_labels, args.max_val_samples, args.seed + 1, "val")
    embedding_cache = load_embedding_cache(
        embeddings_dir,
        {"train": train_rows, "val": val_rows},
    )

    embedding_dim = int(next(iter(embedding_cache.values())).shape[-1])
    device = torch.device(args.device)
    model = TypePairRelationHead(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_grouped = group_rows_by_type_pair(train_rows)
    val_grouped = group_rows_by_type_pair(val_rows)
    train_loaders = make_loaders(train_grouped, embedding_cache, args.batch_size, shuffle=True)
    val_loaders = make_loaders(val_grouped, embedding_cache, args.batch_size, shuffle=False)

    config = {
        "embeddings_dir": str(embeddings_dir),
        "train_labels_jsonl": str(train_labels),
        "val_labels_jsonl": str(val_labels),
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "embedding_dim": embedding_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "threshold": args.threshold,
        "seed": args.seed,
        "device": str(device),
        "resume_checkpoint": args.resume_checkpoint,
        "train_type_pair_counts": dict(Counter(row["type_pair"] for row in train_rows)),
        "val_type_pair_counts": dict(Counter(row["type_pair"] for row in val_rows)),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", config)

    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows: {len(val_rows)}")
    print(f"Embedding files loaded: {len(embedding_cache)}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Output dir: {out_dir}", flush=True)

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    start_epoch = 1
    if args.resume_checkpoint:
        resume_path = resolve_path(args.resume_checkpoint)
        last_epoch, best_val_loss, history = load_checkpoint(resume_path, model, optimizer, device)
        start_epoch = last_epoch + 1
        print(
            f"Resumed checkpoint: {resume_path} "
            f"last_epoch={last_epoch} best_val_loss={best_val_loss:.6f}",
            flush=True,
        )
        if start_epoch > args.epochs:
            print(f"Nothing to train: checkpoint epoch {last_epoch} >= target epochs {args.epochs}.")
            return

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(model, train_loaders, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loaders, criterion, device, args.threshold)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        write_json(out_dir / "history.json", {"history": history})

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_micro_f1={val_metrics['micro_f1']:.6f}",
            flush=True,
        )

        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, best_val_loss, config, history)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, best_val_loss, config, history)
            print(f"saved best checkpoint: epoch={epoch} val_loss={best_val_loss:.6f}", flush=True)

    print(f"Training finished. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
