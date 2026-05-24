from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class ParserJsonError(ValueError):
    pass


def load_parser_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ParserJsonError("Parser JSON root must be an object.")
    if "viewBox" not in data or "primitives" not in data:
        raise ParserJsonError("Parser JSON must contain 'viewBox' and 'primitives'.")
    if not isinstance(data["viewBox"], list) or len(data["viewBox"]) != 4:
        raise ParserJsonError("'viewBox' must be a list of four numbers.")
    if not isinstance(data["primitives"], list):
        raise ParserJsonError("'primitives' must be a list.")

    return data


def save_tokenized_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

