"""CLI entry point.

Usage:
    python -m arxiv_parser [--date YYYY-MM-DD] [--config PATH]
                           [--no-rank] [--dry-run]

No date means today's local date.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tomllib
from datetime import date
from pathlib import Path

from . import __version__
from .arxiv_api import ArxivAPIError, fetch_entries
from .batch import ListingError, resolve_batch
from .console import say
from .llm import FallbackProvider, LLMProvider, OpenAICompatProvider
from .ranker import rank_digest
from .writer import write_digest

DEFAULT_CONFIG = "config.toml"
DEFAULT_INTERESTS = "interests.md"
DEFAULT_OUTPUT_DIR = "digests"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="arxiv-parser",
        description="Fetch arXiv submissions for your categories and re-rank "
        "them by personal interest.",
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help="submission date to query, YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"path to config.toml (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--no-rank",
        action="store_true",
        help="skip LLM ranking, keep original submission order",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and rank but do not write the output file",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser.parse_args(argv)


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        say(f"error: config file not found: {p}", file=sys.stderr)
        raise SystemExit(2)
    try:
        with p.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        say(f"error: invalid config file {p}: {e}", file=sys.stderr)
        raise SystemExit(2)


def read_interests(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        say(f"note: interests file not found ({p}); ranking without interests")
        return ""
    return p.read_text(encoding="utf-8")


def resolve_provider(cfg: dict, no_rank: bool) -> LLMProvider | None:
    llm_cfg = cfg.get("llm", {})
    provider_name = llm_cfg.get("provider", "none").strip().lower()
    if no_rank or provider_name in ("", "none"):
        return None
    if provider_name != "openai-compatible":
        say(
            f"error: unknown llm.provider {provider_name!r} "
            '(supported: "none", "openai-compatible")',
            file=sys.stderr,
        )
        raise SystemExit(2)

    model = str(llm_cfg.get("model", "")).strip()
    if not model:
        say("error: llm.model is empty but llm.provider is set", file=sys.stderr)
        raise SystemExit(2)
    key_env = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(key_env, "").strip()
    if not api_key:
        say(
            f"warning: {key_env} is not set; skipping ranking "
            "(original order will be kept)"
        )
        return None
    models = [model] + [
        str(m).strip() for m in llm_cfg.get("fallback_models", []) if str(m).strip()
    ]
    providers = [
        OpenAICompatProvider(
            base_url=llm_cfg.get("base_url", "https://api.openai.com/v1"),
            model=m,
            api_key=api_key,
            timeout=float(llm_cfg.get("timeout", 300)),
        )
        for m in models
    ]
    if len(providers) == 1:
        return providers[0]
    return FallbackProvider(providers)


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    categories = list(cfg.get("arxiv", {}).get("categories", []))
    if not categories:
        say("error: no categories configured (arxiv.categories in config)", file=sys.stderr)
        return 2

    window_days_back = int(cfg.get("arxiv", {}).get("window_days_back", 2))
    output_dir = Path(cfg.get("output", {}).get("dir", DEFAULT_OUTPUT_DIR))
    interests_path = cfg.get("output", {}).get("interests_file", DEFAULT_INTERESTS)

    day = args.date or date.today()
    start_day = day.fromordinal(day.toordinal() - window_days_back)

    mode = cfg.get("arxiv", {}).get("mode", "auto").strip().lower()
    entries: list | None = None
    source_note = ""
    batch_order = False
    if mode in ("auto", "batch"):
        try:
            resolved = resolve_batch(categories, day)
        except ListingError as e:
            resolved = None
            say(f"  batch source unavailable ({e})")
        if resolved is not None:
            entries, source_note = resolved
            batch_order = True
        else:
            say(
                "  exact batch not available for this date; "
                "falling back to submission-date window"
            )
    if entries is None:
        if mode not in ("auto", "batch", "api"):
            say(f"error: unknown arxiv.mode {mode!r}", file=sys.stderr)
            return 2
        say(f"Fetching {', '.join(categories)} for {day:%Y-%m-%d} "
              f"(window: {start_day:%Y-%m-%d} .. {day:%Y-%m-%d}) ...")
        try:
            entries = fetch_entries(categories, day, window_days_back=window_days_back)
        except ArxivAPIError as e:
            say(f"error: {e}", file=sys.stderr)
            return 1
        source_note = (
            "submission-date window "
            f"[{start_day:%Y-%m-%d} .. {day:%Y-%m-%d}]"
        )
        say(f"  {len(entries)} papers found")
        say(
            "  note: the arXiv API can only search by submission date; "
            "replacement versions and cross-lists of papers submitted earlier "
            "than the window are not included"
        )
    else:
        say(f"  {source_note}")

    interests = read_interests(interests_path)
    provider = resolve_provider(cfg, args.no_rank)

    if provider is None and not args.no_rank and len(entries) > 1:
        say("  ranking: skipped (no LLM provider configured)")

    permutation: list[int] = list(range(len(entries)))
    if provider is not None:
        say(f"  ranking with {provider.name} ...")
        t0 = time.monotonic()
        permutation, note = rank_digest(
            provider,
            entries,
            interests,
            max_retries=int(cfg.get("ranker", {}).get("max_retries", 1)),
            temperature=float(cfg.get("ranker", {}).get("temperature", 0.2)),
            two_pass_threshold=int(cfg.get("ranker", {}).get("two_pass_threshold", 60)),
            refine_top=int(cfg.get("ranker", {}).get("refine_top", 30)),
        )
        if note:
            say(f"  warning: {note}")
        else:
            say(f"  done in {time.monotonic() - t0:.1f}s")

    if args.dry_run:
        out_path = output_dir / f"{day:%Y-%m-%d}.md"
        say(f"dry-run: would write {out_path}")
        return 0

    out_path = write_digest(output_dir, day, categories, entries, permutation)
    say(f"Wrote {out_path} ({len(entries)} papers)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
