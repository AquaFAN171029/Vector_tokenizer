#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .embedding_generator import (
        EmbeddingConfig,
        EmbeddingGenerator,
        load_tokenized_json,
        save_embedding_json,
    )
except ImportError:
    from embedding_generator import (
        EmbeddingConfig,
        EmbeddingGenerator,
        load_tokenized_json,
        save_embedding_json,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate primitive embeddings from tokenized JSON.")
    parser.add_argument("input_json", type=str, help="Tokenized JSON from run_tokenizer.py.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Embedding JSON path.")
    parser.add_argument("--dim", type=int, choices=(256,), default=256)
    parser.add_argument("--type-dim", type=int, choices=(16, 32, 64, 128), default=32)
    parser.add_argument("--geom-dim", type=int, choices=(16, 32, 64, 128), default=128)
    parser.add_argument("--style-dim", type=int, choices=(16, 32, 64, 128), default=32)
    parser.add_argument("--pos-dim", type=int, choices=(16, 32, 64, 128), default=64)
    parser.add_argument("--text-dim", type=int, choices=(16, 32, 64, 128), default=64)
    parser.add_argument("--group-dim", type=int, choices=(16, 32, 64, 128), default=16)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    config = EmbeddingConfig(
        output_dim=args.dim,
        type_dim=args.type_dim,
        geom_dim=args.geom_dim,
        style_dim=args.style_dim,
        pos_dim=args.pos_dim,
        text_dim=args.text_dim,
        group_dim=args.group_dim,
        seed=args.seed,
    )
    tokenized = load_tokenized_json(args.input_json)
    result = EmbeddingGenerator(config).embed_sequence(tokenized)
    output = Path(args.output) if args.output else Path(args.input_json).with_suffix(".embeddings.json")
    save_embedding_json(result, output)
    print(f"Embedded {result['num_primitives']} primitives")
    print(f"Embedding dim: {config.output_dim}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
