"""Core data model: a single arXiv paper entry."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field

# Output text is hard-wrapped at this width (titles and abstracts).
WRAP_WIDTH = 80


@dataclass
class Entry:
    """One arXiv paper, exactly as reported by the API.

    The fields are kept verbatim so that the final output loses no
    information present in the source (modulo whitespace normalization).
    """

    arxiv_id: str            # e.g. "2610.02207" (no version suffix)
    version: int             # e.g. 1
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    categories: list[str] = field(default_factory=list)
    primary_category: str = ""
    published: str = ""      # announcement datetime, ISO 8601 (from API)
    abs_url: str = ""

    def to_markdown_body(self) -> str:
        """Render the entry's details (no heading) as markdown."""
        authors = ", ".join(self.authors) if self.authors else "n/a"
        cats = ", ".join(self.categories) if self.categories else "n/a"
        lines = [
            f"- **Authors:** {authors}",
            f"- **arXiv:** [{self.arxiv_id}v{self.version}]({self.abs_url})",
            f"- **Categories:** {cats} (primary: {self.primary_category})",
            f"- **Published (announced):** {self.published or 'n/a'}",
            "",
            wrap_text(self.abstract),
        ]
        return "\n".join(lines)


def wrap_text(text: str, width: int = WRAP_WIDTH) -> str:
    """Hard-wrap text at `width`, breaking only at whitespace (never inside
    long math expressions or URLs)."""
    if not text:
        return ""
    return "\n".join(
        textwrap.wrap(
            text,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def wrap_title(title: str, number: int, width: int = WRAP_WIDTH) -> str:
    """Render a numbered heading, wrapping long titles onto continuation
    lines prefixed with '##   ' (matching the digest house style)."""
    prefix = f"## {number}. "
    wrapped = textwrap.wrap(
        title,
        width=width - len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]
    return "\n".join([prefix + wrapped[0]] + ["##   " + w for w in wrapped[1:]])
