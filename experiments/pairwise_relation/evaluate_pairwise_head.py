#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PIPELINE_ROOT.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("Install PyTorch before evaluating prediction heads.") from exc

from experiments.pairwise_relation.train_pairwise_head import (
    evaluate,
    group_rows_by_type_pair,
    load_embedding_cache,
    load_rows,
    make_loaders,
    output_path,
    resolve_path,
    write_json,
)
from modeling.prediction_heads import TypePairRelationHead


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[TypePairRelationHead, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config") or {}
    embedding_dim = int(config.get("embedding_dim") or 256)
    hidden_dim = int(config.get("hidden_dim") or 256)
    dropout = float(config.get("dropout") or 0.0)
    model = TypePairRelationHead(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, {
        "checkpoint_epoch": int(checkpoint.get("epoch") or 0),
        "best_val_loss": float(checkpoint.get("best_val_loss", 0.0)),
        "checkpoint_config": config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained pairwise relation head.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--output-json", default="outputs/evaluations/pairwise_relation_head_test.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    checkpoint_path = resolve_path(args.checkpoint)
    embeddings_dir = resolve_path(args.embeddings_dir)
    labels_jsonl = resolve_path(args.labels_jsonl)
    result_path = output_path(args.output_json)

    rows = load_rows(labels_jsonl, args.max_samples, args.seed, "eval")
    embedding_cache = load_embedding_cache(embeddings_dir, {"eval": rows})
    loaders = make_loaders(
        group_rows_by_type_pair(rows),
        embedding_cache,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model, checkpoint_info = load_model_from_checkpoint(checkpoint_path, device)
    criterion = nn.BCEWithLogitsLoss()
    metrics = evaluate(model, loaders, criterion, device, args.threshold)

    result = {
        "checkpoint": str(checkpoint_path),
        "embeddings_dir": str(embeddings_dir),
        "labels_jsonl": str(labels_jsonl),
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "seed": args.seed,
        "device": str(device),
        "num_rows_selected": len(rows),
        "num_embedding_files_loaded": len(embedding_cache),
        "type_pair_counts": dict(Counter(row["type_pair"] for row in rows)),
        **checkpoint_info,
        "metrics": metrics,
    }
    write_json(result_path, result)

    print(f"checkpoint_epoch={checkpoint_info['checkpoint_epoch']}")
    print(f"best_val_loss={checkpoint_info['best_val_loss']:.6f}")
    print(f"eval_loss={metrics['loss']:.6f}")
    print(f"eval_micro_precision={metrics['micro_precision']:.6f}")
    print(f"eval_micro_recall={metrics['micro_recall']:.6f}")
    print(f"eval_micro_f1={metrics['micro_f1']:.6f}")
    print(f"results={result_path}")


if __name__ == "__main__":
    main()
