#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PIPELINE_ROOT.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from experiments.pairwise_relation.relation_labels import (
    PRIMITIVE_TYPES,
    RelationThresholds,
    TYPE_PAIR_RELATION_LABELS,
    TYPE_PAIR_RELATION_TO_ID,
    classify_pair,
    label_vector,
)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_dir(path: str) -> Path:
    p = Path(path)
    if p.exists() or p.is_absolute():
        return p
    candidate = PIPELINE_ROOT / p
    if candidate.exists():
        return candidate
    candidate = WORKSPACE_ROOT / p
    if candidate.exists():
        return candidate
    return p


def resolve_output_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == PIPELINE_ROOT.name:
        return WORKSPACE_ROOT / p
    return PIPELINE_ROOT / p


def iter_parsed_files(parsed_dir: Path) -> List[Path]:
    return sorted(parsed_dir.rglob("*.parsed.json"))


def split_key_from_source(source_file: str) -> str:
    return Path(source_file).with_suffix("").name


def split_key_from_parsed_path(parsed_path: Path, parsed_dir: Path) -> str:
    rel = parsed_path.relative_to(parsed_dir)
    return rel.name.replace(".parsed.json", "")


def load_split_keys(split_json: str | None, split: str | None) -> Set[str] | None:
    if not split_json:
        return None
    if not split:
        raise ValueError("--split is required when --split-json is provided.")

    path = resolve_dir(split_json)
    data = load_json(path)
    splits = data.get("splits")
    if not isinstance(splits, dict) or split not in splits:
        available = sorted(splits.keys()) if isinstance(splits, dict) else []
        raise ValueError(f"Split '{split}' not found in {path}. Available splits: {available}")

    return {split_key_from_source(source_file) for source_file in splits[split]}


def tokenized_path_for(parsed_path: Path, parsed_dir: Path, tokenized_dir: Path) -> Path:
    rel = parsed_path.relative_to(parsed_dir)
    name = rel.name.replace(".parsed.json", ".tokenized.json")
    return tokenized_dir / rel.with_name(name)


def ordered_primitive_ids(tokenized: Dict[str, Any]) -> List[str]:
    return [p["primitive_id"] for p in tokenized.get("tokenized_primitives", []) if "primitive_id" in p]


def sampled_pairs(indices: List[int], max_pairs: int, rng: random.Random) -> Iterable[Tuple[int, int]]:
    total = len(indices) * (len(indices) - 1) // 2
    if total <= max_pairs:
        yield from combinations(indices, 2)
        return

    seen = set()
    while len(seen) < max_pairs:
        i, j = rng.sample(indices, 2)
        if i > j:
            i, j = j, i
        if (i, j) not in seen:
            seen.add((i, j))
            yield i, j


def build_labels_for_file(
    parsed: Dict[str, Any],
    tokenized: Dict[str, Any],
    source_key: str,
    thresholds: RelationThresholds,
    negatives_per_positive: int,
    max_pairs: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], Counter]:
    primitives = parsed.get("primitives") or []
    primitive_by_id = {p.get("primitive_id"): p for p in primitives}
    ordered_ids = ordered_primitive_ids(tokenized)
    ordered_supported_indices = [
        idx for idx, pid in enumerate(ordered_ids)
        if (primitive_by_id.get(pid) or {}).get("type") in PRIMITIVE_TYPES
    ]

    positives: List[Dict[str, Any]] = []
    none_candidates: List[Dict[str, Any]] = []
    counts = Counter()

    for i, j in sampled_pairs(ordered_supported_indices, max_pairs, rng):
        pid_i = ordered_ids[i]
        pid_j = ordered_ids[j]
        result = classify_pair(
            primitive_by_id[pid_i],
            primitive_by_id[pid_j],
            parsed.get("viewBox") or [0, 0, 1, 1],
            thresholds,
        )
        if result is None:
            continue
        type_pair, labels = result
        labels = list(labels)
        labels_vec = label_vector(type_pair, labels)
        sample = {
            "source_file": source_key,
            "i": i,
            "j": j,
            "primitive_id_i": pid_i,
            "primitive_id_j": pid_j,
            "type_i": primitive_by_id[pid_i].get("type"),
            "type_j": primitive_by_id[pid_j].get("type"),
            "type_pair": type_pair,
            "labels": labels,
            "label_ids": [TYPE_PAIR_RELATION_TO_ID[type_pair][label] for label in labels],
            "label_vector": labels_vec,
        }
        if not labels:
            none_candidates.append(sample)
        else:
            positives.append(sample)
            for label in labels:
                counts[f"{type_pair}:{label}"] += 1

    max_negatives = max(len(positives) * negatives_per_positive, negatives_per_positive)
    negatives = rng.sample(none_candidates, min(len(none_candidates), max_negatives))
    for sample in negatives:
        counts[f"{sample['type_pair']}:none"] += 1

    samples = positives + negatives
    rng.shuffle(samples)
    return samples, counts


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pseudo-labels for typed primitive pairwise relation prediction.")
    parser.add_argument("--parsed-dir", default="outputs/processed_dataset/parsed")
    parser.add_argument("--tokenized-dir", default="outputs/processed_dataset/tokenized")
    parser.add_argument("--output-jsonl", default="outputs/pairwise_relation/pairwise_labels.jsonl")
    parser.add_argument("--summary-json", default="outputs/pairwise_relation/summary.json")
    parser.add_argument("--split-json", default=None, help="File-level split manifest from build_file_split.py.")
    parser.add_argument("--split", choices=("train", "val", "test"), default=None, help="Split name to process.")
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--max-pairs-per-file", type=int, default=50000)
    parser.add_argument(
        "--max-line-pairs-per-file",
        type=int,
        default=None,
        help="Deprecated alias for --max-pairs-per-file.",
    )
    parser.add_argument("--parallel-dot", type=float, default=0.95)
    parser.add_argument("--perpendicular-dot", type=float, default=0.15)
    parser.add_argument("--joined-diag-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    parsed_dir = resolve_dir(args.parsed_dir)
    tokenized_dir = resolve_dir(args.tokenized_dir)
    output_jsonl = resolve_output_path(args.output_jsonl)
    summary_json = resolve_output_path(args.summary_json)
    split_keys = load_split_keys(args.split_json, args.split)

    thresholds = RelationThresholds(
        parallel_dot=args.parallel_dot,
        perpendicular_dot=args.perpendicular_dot,
        joined_diag_ratio=args.joined_diag_ratio,
    )
    max_pairs_per_file = args.max_line_pairs_per_file or args.max_pairs_per_file
    rng = random.Random(args.seed)
    all_samples: List[Dict[str, Any]] = []
    label_counts = Counter()
    failures = []

    parsed_files_all = iter_parsed_files(parsed_dir)
    parsed_files = [
        parsed_path for parsed_path in parsed_files_all
        if split_keys is None or split_key_from_parsed_path(parsed_path, parsed_dir) in split_keys
    ]
    for idx, parsed_path in enumerate(parsed_files, 1):
        tokenized_path = tokenized_path_for(parsed_path, parsed_dir, tokenized_dir)
        print(f"[{idx}/{len(parsed_files)}] {parsed_path}")
        try:
            parsed = load_json(parsed_path)
            tokenized = load_json(tokenized_path)
            samples, counts = build_labels_for_file(
                parsed=parsed,
                tokenized=tokenized,
                source_key=str(parsed_path),
                thresholds=thresholds,
                negatives_per_positive=args.negatives_per_positive,
                max_pairs=max_pairs_per_file,
                rng=rng,
            )
            all_samples.extend(samples)
            label_counts.update(counts)
        except Exception as exc:
            failures.append({"parsed_file": str(parsed_path), "error": str(exc)})
            print(f"FAILED: {parsed_path}: {exc}")

    write_jsonl(all_samples, output_jsonl)
    summary = {
        "num_parsed_files": len(parsed_files),
        "num_parsed_files_available": len(parsed_files_all),
        "split_json": args.split_json,
        "split": args.split,
        "num_samples": len(all_samples),
        "label_counts": dict(label_counts),
        "type_pair_relation_labels": TYPE_PAIR_RELATION_LABELS,
        "type_pair_relation_to_id": TYPE_PAIR_RELATION_TO_ID,
        "target_format": "multi_label",
        "none_encoding": "empty labels with an all-zero label_vector",
        "thresholds": {
            "parallel_dot": thresholds.parallel_dot,
            "perpendicular_dot": thresholds.perpendicular_dot,
            "joined_diag_ratio": thresholds.joined_diag_ratio,
            "near_diag_ratio": thresholds.near_diag_ratio,
            "overlap_diag_ratio": thresholds.overlap_diag_ratio,
            "tangent_diag_ratio": thresholds.tangent_diag_ratio,
            "concentric_diag_ratio": thresholds.concentric_diag_ratio,
        },
        "negatives_per_positive": args.negatives_per_positive,
        "max_pairs_per_file": max_pairs_per_file,
        "failures": failures,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDone")
    print(f"Samples: {len(all_samples)}")
    print(f"Label counts: {dict(label_counts)}")
    print(f"Labels: {output_jsonl}")
    print(f"Summary: {summary_json}")


if __name__ == "__main__":
    main()
