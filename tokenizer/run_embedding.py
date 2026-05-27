#!/usr/bin/env python3
from __future__ import annotations

try:
    from .cli.run_embedding import main
except ImportError:
    from pathlib import Path
    import sys

    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from tokenizer.cli.run_embedding import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
