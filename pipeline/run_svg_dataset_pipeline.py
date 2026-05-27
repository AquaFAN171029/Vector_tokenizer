#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PIPELINE_ROOT.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from svg_parser.svg_parser_v2 import SvgParser
from tokenizer.dataset_reader import save_tokenized_json
from tokenizer.embedding_generator import EmbeddingConfig, EmbeddingGenerator, save_embedding_json
from tokenizer.flow_embedding_generator import FlowEmbeddingConfig, FlowEmbeddingGenerator, save_flow_embedding_json
from tokenizer.primitive_tokenizer import TokenizerConfig
from tokenizer.run_tokenizer import tokenize_parser_json


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_svg_files(input_dir: Path, limit: int | None) -> List[Path]:
    files = sorted(p for p in input_dir.rglob("*.svg") if p.is_file())
    return files[:limit] if limit is not None else files


def relative_stem(svg_path: Path, input_dir: Path) -> Path:
    rel = svg_path.relative_to(input_dir).with_suffix("")
    return rel


def output_paths(output_dir: Path, rel_stem: Path) -> Dict[str, Path]:
    return {
        "parsed": output_dir / "parsed" / rel_stem.with_suffix(".parsed.json"),
        "tokenized": output_dir / "tokenized" / rel_stem.with_suffix(".tokenized.json"),
        "embedding": output_dir / "embeddings" / rel_stem.with_suffix(".embeddings.json"),
    }


def summarize_records(records: List[Dict[str, Any]], failures: List[Dict[str, str]]) -> Dict[str, Any]:
    type_counts = Counter()
    group_counts = Counter()
    primitive_counts = []
    text_count = 0

    for record in records:
        primitive_counts.append(record["num_primitives"])
        type_counts.update(record.get("type_counts", {}))
        group_counts.update(record.get("group_counts", {}))
        text_count += record.get("text_primitives", 0)

    total = sum(primitive_counts)
    return {
        "num_files_total": len(records) + len(failures),
        "num_files_success": len(records),
        "num_files_failed": len(failures),
        "total_primitives": total,
        "avg_primitives_per_file": round(total / len(records), 3) if records else 0,
        "min_primitives_per_file": min(primitive_counts) if primitive_counts else 0,
        "max_primitives_per_file": max(primitive_counts) if primitive_counts else 0,
        "type_counts": dict(type_counts),
        "num_text_or_annotation_primitives": text_count,
        "top_groups": group_counts.most_common(20),
        "failures": failures,
    }


def compact_embedding_record(embedding_json: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    return {
        "source_file": source_file,
        "num_primitives": embedding_json["num_primitives"],
        "primitive_ids": [item["primitive_id"] for item in embedding_json["embeddings"]],
        "group_tokens": [item["group_token"] for item in embedding_json["embeddings"]],
        "embeddings": [item["embedding"] for item in embedding_json["embeddings"]],
    }


def process_one(
    svg_path: Path,
    input_dir: Path,
    output_dir: Path,
    tokenizer_config: TokenizerConfig,
    embedding_config: EmbeddingConfig | FlowEmbeddingConfig,
    embedding_strategy: str,
    overwrite: bool,
    write_embeddings: bool,
) -> Dict[str, Any]:
    rel_stem = relative_stem(svg_path, input_dir)
    paths = output_paths(output_dir, rel_stem)

    if not overwrite and all(path.exists() for path in paths.values() if write_embeddings or path != paths["embedding"]):
        parsed = json.loads(paths["parsed"].read_text(encoding="utf-8"))
        return summarize_file(svg_path, parsed, skipped=True)

    parsed = SvgParser().parse(svg_path)
    write_json(parsed, paths["parsed"])

    tokenized = tokenize_parser_json(parsed, tokenizer_config)
    save_tokenized_json(tokenized, paths["tokenized"])

    if write_embeddings:
        if embedding_strategy == "flow":
            embedded = FlowEmbeddingGenerator(embedding_config).embed_parser_json(parsed)
            save_flow_embedding_json(embedded, paths["embedding"])
        else:
            embedded = EmbeddingGenerator(embedding_config).embed_sequence(tokenized)
            save_embedding_json(embedded, paths["embedding"])

    return summarize_file(svg_path, parsed, skipped=False)


def summarize_file(svg_path: Path, parsed: Dict[str, Any], skipped: bool) -> Dict[str, Any]:
    primitives = parsed.get("primitives") or []
    type_counts = Counter(p.get("type") or "UNK" for p in primitives)
    group_counts = Counter((p.get("meta") or {}).get("parent_group_id") or "NO_GROUP" for p in primitives)
    text_primitives = sum(1 for p in primitives if (p.get("text") or {}).get("normalized_text"))
    return {
        "source_file": str(svg_path),
        "num_primitives": len(primitives),
        "type_counts": dict(type_counts),
        "group_counts": dict(group_counts),
        "text_primitives": text_primitives,
        "skipped_existing": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch parser -> tokenizer -> embedding pipeline for an SVG dataset.")
    parser.add_argument("--input-dir", default="SVG Dataset", help="Directory containing SVG files.")
    parser.add_argument("--output-dir", default="outputs/processed_dataset", help="Directory for parsed/tokenized/embedding outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N SVG files.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs even if files already exist.")
    parser.add_argument("--skip-embeddings", action="store_true", help="Only write parsed and tokenized JSON.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately if one SVG fails.")
    parser.add_argument("--geom-bins", type=int, default=256)
    parser.add_argument("--pos-bins", type=int, default=256)
    parser.add_argument("--width-bins", type=int, default=32)
    parser.add_argument("--opacity-bins", type=int, default=11)
    parser.add_argument("--min-text-freq", type=int, default=1)
    parser.add_argument(
        "--embedding-strategy",
        choices=("token-hash", "flow"),
        default="token-hash",
        help=(
            "token-hash uses tokenized discrete geometry/position tokens. "
            "flow uses raw continuous geometry/position values with hash for discrete fields."
        ),
    )
    parser.add_argument("--text-backend", choices=("hash", "qwen"), default="hash")
    parser.add_argument("--qwen-model", default="text-embedding-v4")
    parser.add_argument("--qwen-base-url", default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--qwen-cache-path", default="outputs/cache/qwen_text_embeddings_64.json")
    parser.add_argument("--qwen-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() and (WORKSPACE_ROOT / input_dir).exists():
        input_dir = WORKSPACE_ROOT / input_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PIPELINE_ROOT / output_dir
    qwen_cache_path = Path(args.qwen_cache_path)
    if not qwen_cache_path.is_absolute():
        qwen_cache_path = PIPELINE_ROOT / qwen_cache_path
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    tokenizer_config = TokenizerConfig(
        geom_bins=args.geom_bins,
        pos_bins=args.pos_bins,
        width_bins=args.width_bins,
        opacity_bins=args.opacity_bins,
        min_text_freq=args.min_text_freq,
    )
    embedding_config: EmbeddingConfig | FlowEmbeddingConfig
    if args.embedding_strategy == "flow":
        embedding_config = FlowEmbeddingConfig(
            text_backend=args.text_backend,
            qwen_model=args.qwen_model,
            qwen_base_url=args.qwen_base_url,
            qwen_cache_path=str(qwen_cache_path),
            qwen_api_key_env=args.qwen_api_key_env,
            seed=args.seed,
        )
    else:
        embedding_config = EmbeddingConfig(
            text_backend=args.text_backend,
            qwen_model=args.qwen_model,
            qwen_base_url=args.qwen_base_url,
            qwen_cache_path=str(qwen_cache_path),
            qwen_api_key_env=args.qwen_api_key_env,
            seed=args.seed,
        )
    svg_files = find_svg_files(input_dir, args.limit)

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    dataset_rows: List[Dict[str, Any]] = []

    for idx, svg_path in enumerate(svg_files, 1):
        print(f"[{idx}/{len(svg_files)}] {svg_path}")
        try:
            record = process_one(
                svg_path=svg_path,
                input_dir=input_dir,
                output_dir=output_dir,
                tokenizer_config=tokenizer_config,
                embedding_config=embedding_config,
                embedding_strategy=args.embedding_strategy,
                overwrite=args.overwrite,
                write_embeddings=not args.skip_embeddings,
            )
            records.append(record)

            if not args.skip_embeddings:
                rel_stem = relative_stem(svg_path, input_dir)
                embedded = json.loads(output_paths(output_dir, rel_stem)["embedding"].read_text(encoding="utf-8"))
                dataset_rows.append(compact_embedding_record(embedded, str(svg_path)))
        except Exception as exc:
            failure = {
                "source_file": str(svg_path),
                "error": str(exc),
                "traceback": traceback.format_exc(limit=3),
            }
            failures.append(failure)
            print(f"FAILED: {svg_path}: {exc}")
            if args.stop_on_error:
                raise

    summary = summarize_records(records, failures)
    summary["tokenizer_config"] = asdict(tokenizer_config)
    summary["embedding_strategy"] = args.embedding_strategy
    summary["embedding_config"] = None if args.skip_embeddings else asdict(embedding_config)
    write_json(summary, output_dir / "summary.json")
    write_json({"files": records}, output_dir / "manifest.json")
    if dataset_rows:
        write_jsonl(dataset_rows, output_dir / "embedding_dataset.jsonl")

    print("\nDone")
    print(f"Success: {summary['num_files_success']}")
    print(f"Failed: {summary['num_files_failed']}")
    print(f"Total primitives: {summary['total_primitives']}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
