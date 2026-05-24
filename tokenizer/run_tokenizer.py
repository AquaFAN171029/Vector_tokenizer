#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

try:
    from .dataset_reader import load_parser_json, save_tokenized_json
    from .primitive_tokenizer import PrimitiveTokenizer, TokenizerConfig
    from .sequence_builder import build_sequence
    from .text_tokenizer import build_text_vocab
except ImportError:
    from dataset_reader import load_parser_json, save_tokenized_json
    from primitive_tokenizer import PrimitiveTokenizer, TokenizerConfig
    from sequence_builder import build_sequence
    from text_tokenizer import build_text_vocab


def tokenize_parser_json(data: Dict[str, Any], config: TokenizerConfig) -> Dict[str, Any]:
    primitives = build_sequence(data["primitives"])
    text_vocab = build_text_vocab(primitives, config.min_text_freq)
    tokenizer = PrimitiveTokenizer(data["viewBox"], text_vocab, config)
    tokenized = [tokenizer.tokenize(primitive) for primitive in primitives]

    return {
        "source_file": data.get("source_file"),
        "viewBox": data["viewBox"],
        "num_primitives": len(tokenized),
        "tokenizer_config": asdict(config),
        "tokenized_primitives": tokenized,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize parser JSON primitives")
    parser.add_argument("input_json", type=str, help="Parser output JSON")
    parser.add_argument("-o", "--output", type=str, default=None, help="Tokenized JSON path.")
    parser.add_argument("--geom-bins", type=int, default=256)
    parser.add_argument("--pos-bins", type=int, default=256)
    parser.add_argument("--width-bins", type=int, default=32)
    parser.add_argument("--opacity-bins", type=int, default=11)
    parser.add_argument("--min-text-freq", type=int, default=1)
    args = parser.parse_args()

    config = TokenizerConfig(
        geom_bins=args.geom_bins,
        pos_bins=args.pos_bins,
        width_bins=args.width_bins,
        opacity_bins=args.opacity_bins,
        min_text_freq=args.min_text_freq,
    )
    data = load_parser_json(args.input_json)
    result = tokenize_parser_json(data, config)
    output = Path(args.output) if args.output else Path(args.input_json).with_suffix(".tokenized.json")
    save_tokenized_json(result, output)
    print(f"Tokenized {result['num_primitives']} primitives")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
