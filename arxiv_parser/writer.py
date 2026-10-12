"""Output: render and write the daily digest as a markdown file."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from .models import Entry, wrap_title


def render_digest(
    day: date,
    categories: list[str],
    entries: list[Entry],
    permutation: list[int],
) -> str:
    """Render the full digest markdown for one day."""
    cats = ", ".join(sorted({c for c in categories if c})) or "n/a"
    lines = [
        f"# arXiv digest - {day:%Y-%m-%d}",
        "",
        "---",
        f"- **Categories:** {cats}",
        f"- **Papers:** {len(entries)}",
        "---",
        "",
    ]
    if not entries:
        lines.append("No papers found for this date.")
        return "\n".join(lines).rstrip() + "\n"

    for rank_idx, entry_idx in enumerate(permutation, start=1):
        e = entries[entry_idx]
        lines.append(wrap_title(e.title, rank_idx))
        lines.append("")
        lines.append(e.to_markdown_body())
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_digest(
    output_dir: Path,
    day: date,
    categories: list[str],
    entries: list[Entry],
    permutation: list[int],
) -> Path:
    """Write the digest to <output_dir>/yyyy-mm-dd.md, overwriting if present."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{day:%Y-%m-%d}.md"
    path.write_text(
        render_digest(day, categories, entries, permutation),
        encoding="utf-8",
    )
    return path
