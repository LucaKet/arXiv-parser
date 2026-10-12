"""Exact announcement-batch discovery via the arXiv listing page.

The arXiv API can only search by submission date; it cannot tell which
announcement batch a paper belonged to. The listing page (/list/<cat>/new)
knows the current batch exactly - including cross-lists and replacement
submissions, which the API date search cannot recover.

We use the listing page as the batch index and enrich each paper through
the API (id_list), keeping the listing (email) order.
"""

from __future__ import annotations

import html
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date

from .arxiv_api import parse_entries
from .console import say
from .models import Entry

LISTING_URL = "https://arxiv.org/list/{cat}/new"
API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "Mozilla/5.0 (compatible; arxiv-parser/0.1; personal use)"
_IDLIST_CHUNK = 100

_ID_RE = re.compile(r"arXiv:(\d{4}\.\d{4,5})")
_TITLE_RE = re.compile(r"Title:</span>\s*([^<]+)")
_DATE_RE = re.compile(
    r"Showing new listings for[^<]*?(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})"
)

_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}


class ListingError(RuntimeError):
    """Raised when the listing page cannot be fetched or parsed."""


@dataclass
class Batch:
    """One announcement batch: (arxiv_id, title) pairs per section."""

    label: str = ""
    batch_date: date | None = None
    new: list[tuple[str, str]] = field(default_factory=list)
    cross: list[tuple[str, str]] = field(default_factory=list)
    replacements: list[tuple[str, str]] = field(default_factory=list)

    def all_ids(self) -> list[str]:
        """All ids in email order: new, then cross, then replacements."""
        ids: list[str] = []
        for section in (self.new, self.cross, self.replacements):
            for aid, _ in section:
                if aid not in ids:
                    ids.append(aid)
        return ids

    def titles(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for section in (self.new, self.cross, self.replacements):
            for aid, title in section:
                out.setdefault(aid, title)
        return out


def _http_get(url: str, timeout: float = 30.0, retries: int = 4) -> bytes:
    """GET with retries/backoff (arXiv rate-limits bursts)."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (400, 404, 406):
                raise ListingError(f"request rejected (HTTP {e.code}): {url}") from e
            last = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        if attempt < retries:
            time.sleep(5 * (attempt + 1))
    raise ListingError(f"unreachable after {retries} retries ({last}): {url}")


def parse_listing(raw: str) -> Batch:
    """Parse a /list/<cat>/new page into a Batch.

    The page has one heading ('Showing new listings for <date>') followed by
    up to three sections: New submissions, Cross submissions, Replacement
    submissions.
    """
    batch = Batch()
    m = _DATE_RE.search(raw)
    if m:
        day, month_name, year = m.groups()
        mon = _MONTHS.get(month_name)
        if mon is not None:
            batch.batch_date = date(int(year), mon, int(day))
    heading = re.search(r"Showing new listings for[^<]*", raw)
    batch.label = html.unescape(re.sub(r"<[^>]+>", "", heading.group(0))).strip() if heading else ""

    sections = re.split(r"<h3>(.*?)</h3>", raw)
    for i in range(1, len(sections), 2):
        head = html.unescape(re.sub(r"<[^>]+>", "", sections[i])).strip()
        body = sections[i + 1]
        ids = _ID_RE.findall(body)
        titles = [html.unescape(t).strip() for t in _TITLE_RE.findall(body)]
        pairs = [
            (aid, titles[j] if j < len(titles) else "")
            for j, aid in enumerate(ids)
        ]
        target = None
        if "new submissions" in head.lower():
            target = batch.new
        elif "cross submissions" in head.lower():
            target = batch.cross
        elif "replacement submissions" in head.lower():
            target = batch.replacements
        if target is not None:
            target.extend(pairs)

    if not batch.new and not batch.cross and not batch.replacements:
        raise ListingError("listing page contained no paper sections")
    return batch


def enrich(ids: list[str], titles: dict[str, str]) -> list[Entry]:
    """Fetch full metadata for `ids` via the API, in the given order.

    Papers the API does not return (rare) become stub entries carrying the
    title from the listing page, so nothing visible in the email is lost.
    """
    by_id: dict[str, Entry] = {}
    for start in range(0, len(ids), _IDLIST_CHUNK):
        chunk = ids[start : start + _IDLIST_CHUNK]
        url = API_URL + "?" + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk) * 2}
        )
        for e in parse_entries(_http_get(url)):
            by_id[e.arxiv_id] = e

    entries: list[Entry] = []
    missing = 0
    for aid in ids:
        e = by_id.get(aid)
        if e is None:
            missing += 1
            e = Entry(
                arxiv_id=aid,
                version=1,
                title=titles.get(aid, "(title unavailable)"),
                abs_url=f"https://arxiv.org/abs/{aid}",
            )
        entries.append(e)
    if missing:
        say(f"  note: {missing} paper(s) missing from API metadata; kept listing titles only")
    return entries


def resolve_batch(
    categories: list[str], day: date, within_days: int = 1
) -> tuple[list[Entry], str] | None:
    """Return (entries_in_email_order, source_note) for the batch matching
    `day`, or None if the batch source cannot satisfy the request.

    Fetches one listing page per category, checks the batch date is within
    `within_days` of `day` (the digest you received on the morning of D is
    dated D-1), merges the sections in email order (new, cross,
    replacements) deduplicating by arXiv id, and enriches via the API.
    """
    batches: list[Batch] = []
    labels: list[str] = []
    for cat in categories:
        raw = _http_get(LISTING_URL.format(cat=cat)).decode("utf-8", errors="replace")
        b = parse_listing(raw)
        if b.batch_date is not None and abs((day - b.batch_date).days) > within_days:
            return None
        batches.append(b)
        labels.append(b.label)

    ordered: list[str] = []
    for section in ("new", "cross", "replacements"):
        for b in batches:
            for aid, _ in getattr(b, section):
                if aid not in ordered:
                    ordered.append(aid)

    titles: dict[str, str] = {}
    for b in batches:
        titles.update(b.titles())

    entries = enrich(ordered, titles)
    # dedupe identical batch labels (one per category) so the note does not
    # repeat the same announcement heading for every category
    labels = list(dict.fromkeys(b.label for b in batches))
    label = "; ".join(labels) if labels else "listing"
    note = (
        f"exact announcement batch [{', '.join(categories)}] ({label}), "
        f"{len(entries)} papers"
    )
    return entries, note
