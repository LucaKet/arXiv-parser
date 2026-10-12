"""End-to-end tests of the CLI run() with mocked network and LLM."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from arxiv_parser.main import parse_args, run
from arxiv_parser.models import Entry

ENTRIES = [
    Entry(
        arxiv_id="2610.00001",
        version=1,
        title="Paper one",
        authors=["A"],
        abstract="First.",
        categories=["cond-mat.mes-hall"],
        primary_category="cond-mat.mes-hall",
        published="2026-10-01T17:59:00Z",
        abs_url="https://arxiv.org/abs/2610.00001",
    ),
    Entry(
        arxiv_id="2610.00002",
        version=1,
        title="Paper two",
        authors=["B"],
        abstract="Second.",
        categories=["cond-mat.mes-hall"],
        primary_category="cond-mat.mes-hall",
        published="2026-10-01T18:00:00Z",
        abs_url="https://arxiv.org/abs/2610.00002",
    ),
]


def make_config(tmp: str) -> str:
    cfg = Path(tmp) / "config.toml"
    cfg.write_text(
        f"""
[arxiv]
categories = ["cond-mat.mes-hall"]
window_days_back = 2

[output]
dir = "{tmp}/digests"
interests_file = "{tmp}/interests.md"

[llm]
provider = "none"

[ranker]
temperature = 0.2
max_retries = 1
""",
        encoding="utf-8",
    )
    Path(tmp, "interests.md").write_text("spin qubits\n", encoding="utf-8")
    return str(cfg)


class TestRun(unittest.TestCase):
    def test_end_to_end_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            args = argparse.Namespace(
                date=date(2026, 10, 3),
                config=cfg,
                no_rank=False,
                dry_run=False,
            )
            with mock.patch("arxiv_parser.main.resolve_batch", return_value=None), \
                 mock.patch("arxiv_parser.main.fetch_entries", return_value=ENTRIES):
                code = run(args)
            self.assertEqual(code, 0)
            out = Path(tmp) / "digests" / "2026-10-03.md"
            self.assertTrue(out.is_file())
            md = out.read_text(encoding="utf-8")
            self.assertIn("## 1. Paper one", md)
            self.assertIn("## 2. Paper two", md)
            self.assertIn("- **Papers:** 2", md)

    def test_batch_mode_writes_email_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            args = argparse.Namespace(
                date=date(2026, 10, 3),
                config=cfg,
                no_rank=False,
                dry_run=False,
            )
            with mock.patch(
                "arxiv_parser.main.resolve_batch",
                return_value=(ENTRIES, "exact announcement batch (Friday, 2 October 2026), 2 papers"),
            ), mock.patch("arxiv_parser.main.fetch_entries") as f:
                code = run(args)
            self.assertEqual(code, 0)
            f.assert_not_called()
            out = Path(tmp) / "digests" / "2026-10-03.md"
            md = out.read_text(encoding="utf-8")
            self.assertIn("## 1. Paper one", md)
            self.assertIn("## 2. Paper two", md)

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            args = argparse.Namespace(
                date=date(2026, 10, 3),
                config=cfg,
                no_rank=False,
                dry_run=True,
            )
            with mock.patch("arxiv_parser.main.resolve_batch", return_value=None), \
                 mock.patch("arxiv_parser.main.fetch_entries", return_value=ENTRIES):
                code = run(args)
            self.assertEqual(code, 0)
            self.assertFalse((Path(tmp) / "digests").exists())

    def test_missing_config_fails(self):
        with mock.patch("arxiv_parser.main.resolve_batch"), \
             mock.patch("arxiv_parser.main.fetch_entries") as f:
            args = argparse.Namespace(
                date=date(2026, 10, 3),
                config="/nonexistent/config.toml",
                no_rank=False,
                dry_run=False,
            )
            with self.assertRaises(SystemExit) as cm:
                run(args)
            self.assertEqual(cm.exception.code, 2)
            f.assert_not_called()

    def test_no_categories_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.toml"
            cfg.write_text("[arxiv]\ncategories = []\n[llm]\nprovider = \"none\"\n", encoding="utf-8")
            args = argparse.Namespace(
                date=date(2026, 10, 3),
                config=str(cfg),
                no_rank=False,
                dry_run=False,
            )
            with mock.patch("arxiv_parser.main.resolve_batch") as b, \
                 mock.patch("arxiv_parser.main.fetch_entries") as f:
                self.assertEqual(run(args), 2)
                b.assert_not_called()
                f.assert_not_called()


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        args = parse_args([])
        self.assertIsNone(args.date)
        self.assertFalse(args.no_rank)
        self.assertFalse(args.dry_run)

    def test_date_parsing(self):
        args = parse_args(["--date", "2026-10-03"])
        self.assertEqual(args.date, date(2026, 10, 3))

    def test_bad_date_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(["--date", "03.10.2026"])


if __name__ == "__main__":
    unittest.main()
