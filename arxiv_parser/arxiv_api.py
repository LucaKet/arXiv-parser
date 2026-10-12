"""arXiv API client: query papers by category and submission-date window.

Uses only the standard library (urllib + ElementTree).
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import Iterable

from .models import Entry

API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "arxiv-parser/0.1 (personal use)"

NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "os": "http://a9.com/-/spec/opensearch/1.1/",
}

_ID_RE = re.compile(r"^(.*?)v(\d+)$")


class ArxivAPIError(RuntimeError):
    """Raised when the arXiv API cannot be queried successfully."""


def build_search_query(categories: Iterable[str], day: date, window_days_back: int) -> str:
    """Build a search_query covering all categories in the date window.

    The window covers submittedDate from (day - window_days_back) 00:00 to
    day 23:59 UTC.
    """
    cats = sorted({c for c in categories if c})
    cat_clause = " OR ".join(f"cat:{c}" for c in cats)
    start = day - timedelta(days=window_days_back)
    window = f"submittedDate:[{start:%Y%m%d}0000 TO {day:%Y%m%d}2359]"
    return f"({cat_clause}) AND {window}"


def parse_arxiv_id(id_url: str) -> tuple[str, int]:
    """Extract (arxiv_id, version) from an Atom <id> URL.

    Handles new-style ids ("2610.02207v2") and old-style ids
    ("cond-mat/0204058v1"), with or without a version suffix.
    """
    base = id_url.rsplit("/abs/", 1)[-1].strip()
    m = _ID_RE.match(base)
    if m:
        return m.group(1), int(m.group(2))
    return base, 1


def _clean(text: str | None) -> str:
    """Collapse all whitespace runs into single spaces."""
    return " ".join((text or "").split())


def parse_entries(xml_bytes: bytes) -> list[Entry]:
    """Parse an Atom feed payload into Entry objects."""
    root = ET.fromstring(xml_bytes)
    entries: list[Entry] = []
    for el in root.findall("atom:entry", NAMESPACES):
        arxiv_id, version = parse_arxiv_id(el.findtext("atom:id", "", NAMESPACES))
        primary = el.find("arxiv:primary_category", NAMESPACES)
        primary_cat = primary.attrib.get("term", "") if primary is not None else ""
        authors = [
            _clean(a.text)
            for a in el.findall("atom:author/atom:name", NAMESPACES)
            if a.text
        ]
        entries.append(
            Entry(
                arxiv_id=arxiv_id,
                version=version,
                title=_clean(el.findtext("atom:title", "", NAMESPACES)),
                authors=authors,
                abstract=_clean(el.findtext("atom:summary", "", NAMESPACES)),
                categories=[
                    c.attrib.get("term", "")
                    for c in el.findall("atom:category", NAMESPACES)
                ],
                primary_category=primary_cat,
                published=_clean(el.findtext("atom:published", "", NAMESPACES)),
                abs_url=f"https://arxiv.org/abs/{arxiv_id}",
            )
        )
    return entries


def parse_feed(xml_bytes: bytes) -> tuple[list[Entry], int]:
    """Parse a full Atom feed: (entries, total_results)."""
    root = ET.fromstring(xml_bytes)
    total = int(root.findtext("os:totalResults", "0", NAMESPACES) or "0")
    return parse_entries(xml_bytes), total


def dedupe(entries: Iterable[Entry]) -> list[Entry]:
    """Keep the first occurrence of each arXiv id (a paper may appear in
    several queried categories)."""
    seen: dict[str, Entry] = {}
    for e in entries:
        seen.setdefault(e.arxiv_id, e)
    return list(seen.values())


def _http_get(url: str, timeout: float, retries: int) -> bytes:
    """GET with retries/backoff for transient errors; 400s are raised at once."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 400:
                raise ArxivAPIError(f"arXiv API rejected the query (HTTP 400): {url}") from e
            last = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        if attempt < retries:
            time.sleep(2**attempt)
    raise ArxivAPIError(f"arXiv API unreachable after {retries} retries ({last})")


def fetch_entries(
    categories: Iterable[str],
    day: date,
    window_days_back: int = 2,
    per_page: int = 1000,
    timeout: float = 30.0,
    retries: int = 3,
    page_delay: float = 1.0,
) -> list[Entry]:
    """Fetch all papers in `categories` with submittedDate inside the window.

    Paginates through results, dedupes by arXiv id, and returns entries
    sorted by announcement datetime (ascending).

    Known limitation of the arXiv API: replacement versions and cross-lists
    of papers submitted before the window cannot be found by date search
    (only submittedDate is searchable; there is no announcement/update-date
    search field).
    """
    query = build_search_query(categories, day, window_days_back)
    params: dict[str, str | int] = {
        "search_query": query,
        "start": 0,
        "max_results": per_page,
        "sortBy": "submittedDate",
        "sortOrder": "ascending",
    }
    collected: list[Entry] = []
    total = 0
    while True:
        params["start"] = len(collected)  # counts collected, not just this page
        url = API_URL + "?" + urllib.parse.urlencode(params)
        body = _http_get(url, timeout, retries)
        page_entries, total = parse_feed(body)
        collected.extend(page_entries)
        if not page_entries or len(collected) >= total:
            break
        time.sleep(page_delay)
    entries = dedupe(collected)
    entries.sort(key=lambda e: (e.published, e.arxiv_id))
    return entries
