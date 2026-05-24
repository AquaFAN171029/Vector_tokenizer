from __future__ import annotations

SUPPORTED_TYPES = {"LINE", "ARC", "POLYLINE", "TEXT", "ANNOTATION"}


def tokenize_type(value: object) -> str:
    typ = str(value or "UNK").upper()
    return typ if typ in SUPPORTED_TYPES else "UNK_TYPE"

