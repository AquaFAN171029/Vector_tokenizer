#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PIPELINE_ROOT.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    candidate = PIPELINE_ROOT / p
    if candidate.exists():
        return candidate
    return WORKSPACE_ROOT / p


def output_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PIPELINE_ROOT / p


def load_records(manifest_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = data.get("files")
    if not isinstance(records, list):
        raise ValueError(f"Expected manifest with a 'files' list: {manifest_path}")
    out = []
    seen = set()
    for record in records:
        source_file = record.get("source_file")
        if not source_file:
            continue
        key = str(source_file)
        if key in seen:
            raise ValueError(f"Duplicate source_file in manifest: {key}")
        seen.add(key)
        out.append(record)
    if not out:
        raise ValueError("No files found in manifest.")
    return out


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Split ratios must be non-negative.")
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}.")


def split_counts(n: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    train_n = int(round(n * train_ratio))
    val_n = int(round(n * val_ratio))
    if train_n + val_n > n:
        overflow = train_n + val_n - n
        val_n = max(0, val_n - overflow)
    test_n = n - train_n - val_n
    if n >= 3:
        if train_n == 0:
            train_n, test_n = 1, max(0, test_n - 1)
        if val_n == 0:
            val_n, train_n = 1, max(0, train_n - 1)
        if test_n == 0:
            test_n, train_n = 1, max(0, train_n - 1)
    return train_n, val_n, test_n


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    type_counts = Counter()
    primitive_counts = []
    text_primitives = 0
    for record in records:
        primitive_counts.append(int(record.get("num_primitives") or 0))
        type_counts.update(record.get("type_counts") or {})
        text_primitives += int(record.get("text_primitives") or 0)
    total_primitives = sum(primitive_counts)
    return {
        "num_files": len(records),
        "total_primitives": total_primitives,
        "avg_primitives_per_file": round(total_primitives / len(records), 3) if records else 0,
        "min_primitives_per_file": min(primitive_counts) if primitive_counts else 0,
        "max_primitives_per_file": max(primitive_counts) if primitive_counts else 0,
        "type_counts": dict(type_counts),
        "text_primitives": text_primitives,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a file-level train/val/test split manifest.")
    parser.add_argument("--manifest", required=True, help="Pipeline manifest.json.")
    parser.add_argument("--output-json", default="outputs/splits/file_split_seed13.json")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    manifest_path = resolve_path(args.manifest)
    records = load_records(manifest_path)

    rng = random.Random(args.seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    train_n, val_n, test_n = split_counts(len(shuffled), args.train_ratio, args.val_ratio)
    splits = {
        "train": shuffled[:train_n],
        "val": shuffled[train_n:train_n + val_n],
        "test": shuffled[train_n + val_n:train_n + val_n + test_n],
    }

    assigned = [
        record["source_file"]
        for split_records in splits.values()
        for record in split_records
    ]
    if len(assigned) != len(set(assigned)):
        raise ValueError("Split leakage detected: at least one source_file appears in multiple splits.")

    out = {
        "split_unit": "source_file",
        "leakage_rule": "A source SVG file appears in exactly one of train, val, or test.",
        "source_manifest": str(manifest_path),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "summary": {name: summarize(split_records) for name, split_records in splits.items()},
        "splits": {
            name: [record["source_file"] for record in split_records]
            for name, split_records in splits.items()
        },
    }

    output_json = output_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Files: {len(records)}")
    for name in ("train", "val", "test"):
        info = out["summary"][name]
        print(f"{name}: files={info['num_files']} primitives={info['total_primitives']}")
    print(f"Split manifest: {output_json}")


if __name__ == "__main__":
    main()
