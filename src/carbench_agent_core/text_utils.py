"""Shared text normalization helpers."""

from __future__ import annotations

import re
import unicodedata


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_marks = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return " ".join(
        re.sub(r"[^a-zA-Z0-9]+", " ", without_marks).casefold().split()
    )
