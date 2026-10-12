"""Console output helpers.

Every message printed through `say()` is kept within LINE_WIDTH
characters: long messages are wrapped at word boundaries with the
original indentation preserved, and anything that still exceeds the
width (URLs, JSON payloads, ...) is truncated with an ellipsis.
"""

from __future__ import annotations

import sys
import textwrap
from typing import TextIO

LINE_WIDTH = 80


def _fit_line(indent: str, body: str, width: int) -> list[str]:
    avail = width - len(indent)
    if avail < 8:
        indent = ""
        avail = width
    wrapped = textwrap.wrap(
        body,
        width=avail,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]
    out = []
    for chunk in wrapped:
        if len(chunk) > avail:
            chunk = chunk[: max(avail - 1, 0)] + "\u2026"
        out.append(indent + chunk)
    return out


def say(
    message: str,
    *,
    width: int = LINE_WIDTH,
    file: TextIO | None = None,
) -> None:
    """Print `message`, never exceeding `width` characters per line."""
    lines: list[str] = []
    for line in str(message).split("\n"):
        if not line.strip():
            lines.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        lines.extend(_fit_line(indent, line[len(indent):], width))
    print("\n".join(lines), file=file or sys.stdout)
