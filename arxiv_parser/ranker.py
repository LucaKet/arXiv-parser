"""Ranking: ask the LLM for a permutation of the entries, then apply it.

The LLM never rewrites content - it only returns indices. This makes it
structurally impossible to lose, alter, or invent entries.
"""

from __future__ import annotations

import json
import time

from .console import say
from .llm import LLMError, LLMProvider
from .models import Entry

SYSTEM_PROMPT = """\
You rank a list of arXiv papers by how interesting they are to a specific user.
You will be given the user's interests and a numbered list of papers.

Rules:
1. Respond with ONLY a JSON array of integers: the paper indices in their new
   order, from most to least interesting for this user.
2. Every index from 0 to N-1 must appear exactly once.
3. Do not include any explanation, markdown fences, or other text.
4. Decide quickly. Speed matters more than perfect precision — do not
   overthink individual decisions. Papers matching the stated interests
   should come first; the rest ordered by likely relevance.
"""


def build_papers_text(entries: list[Entry], with_abstracts: bool = True) -> str:
    parts = []
    for i, e in enumerate(entries):
        authors = "; ".join(e.authors) if e.authors else "n/a"
        if with_abstracts:
            parts.append(f"[{i}] {e.arxiv_id} | {e.title} | {authors} | {e.abstract}")
        else:
            cats = ", ".join(e.categories) if e.categories else "n/a"
            parts.append(f"[{i}] {e.arxiv_id} | {e.title} | {authors} | {cats}")
    return "\n".join(parts)


def build_user_prompt(
    entries: list[Entry], interests: str, with_abstracts: bool = True
) -> str:
    interests = interests.strip()
    if interests:
        head = f"USER INTERESTS:\n{interests}\n"
    else:
        head = "USER INTERESTS:\n(none provided - rank by general scientific significance)\n"
    return f"{head}\nPAPERS ({len(entries)}):\n{build_papers_text(entries, with_abstracts)}"


def validate_permutation(perm: object, n: int) -> bool:
    """True iff `perm` is a list containing each of 0..n-1 exactly once."""
    if not isinstance(perm, list) or len(perm) != n:
        return False
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in perm):
        return False
    return sorted(perm) == list(range(n))


def parse_permutation(text: str, n: int) -> list[int] | None:
    """Extract a valid permutation of 0..n-1 from an LLM reply, or None.

    Tolerates markdown fences and stray surrounding text by extracting the
    first [...] substring that parses as JSON.
    """
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if validate_permutation(data, n) else None


def rank(
    provider: LLMProvider | None,
    entries: list[Entry],
    interests: str,
    max_retries: int = 1,
    temperature: float = 0.2,
    with_abstracts: bool = True,
) -> tuple[list[int], str]:
    """Return (permutation, note).

    `permutation` is a list of indices into `entries` in the desired output
    order. On any failure the identity permutation is returned and `note`
    explains why, so the pipeline always produces a valid file.
    """
    n = len(entries)
    identity = list(range(n))
    if n <= 1:
        return identity, ""
    if provider is None:
        return identity, "no LLM provider configured - original order kept"

    user = build_user_prompt(entries, interests, with_abstracts)
    for attempt in range(max_retries + 1):
        try:
            reply = provider.complete(SYSTEM_PROMPT, user, temperature)
        except LLMError as e:
            if attempt < max_retries:
                time.sleep(3 * (attempt + 1))
                continue
            return identity, f"LLM call failed: {e}"
        perm = parse_permutation(reply, n)
        if perm is not None:
            return perm, ""
        if attempt < max_retries:
            time.sleep(3 * (attempt + 1))
            user += (
                f"\n\nYour previous response was not a valid permutation of "
                f"0..{n - 1}. Reply with only the JSON array."
            )
    return identity, "LLM returned no valid permutation - original order kept"


def rank_digest(
    provider: LLMProvider | None,
    entries: list[Entry],
    interests: str,
    max_retries: int = 1,
    temperature: float = 0.2,
    two_pass_threshold: int = 60,
    refine_top: int = 30,
) -> tuple[list[int], str]:
    """Rank all entries for the digest output.

    Small digests get a single pass with full abstracts. Large digests
    (more than `two_pass_threshold` papers) get a cheap first pass over
    titles/authors only, then a full-abstract refinement of the top
    `refine_top` papers - keeping the LLM call fast and reliable on free
    tiers without sacrificing ranking quality where it matters most.
    """
    n = len(entries)
    if provider is None or n <= 1 or n <= two_pass_threshold:
        return rank(
            provider, entries, interests,
            max_retries=max_retries,
            temperature=temperature,
            with_abstracts=True,
        )

    say(f"  first pass: ranking all {n} papers by title/authors ...")
    perm1, note1 = rank(
        provider, entries, interests,
        max_retries=max_retries,
        temperature=temperature,
        with_abstracts=False,
    )
    if note1:
        return perm1, note1

    k = min(refine_top, n)
    say(f"  second pass: refining top {k} with full abstracts ...")
    top = [entries[i] for i in perm1[:k]]
    perm2, note2 = rank(
        provider, top, interests,
        max_retries=max_retries,
        temperature=temperature,
        with_abstracts=True,
    )
    if note2:
        return perm1, f"second pass failed ({note2}); first-pass order kept"
    refined = [perm1[i] for i in perm2]
    return refined + perm1[k:], ""
