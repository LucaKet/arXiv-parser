"""Tests for digest rendering and file writing."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from arxiv_parser.models import Entry
from arxiv_parser.writer import render_digest, write_digest


def make_entries() -> list[Entry]:
    return [
        Entry(
            arxiv_id="2610.00001",
            version=1,
            title="Mesoscopic transport",
            authors=["Alice A.", "Bob B."],
            abstract="We study quantum transport.",
            categories=["cond-mat.mes-hall"],
            primary_category="cond-mat.mes-hall",
            published="2026-10-01T17:59:00Z",
            abs_url="https://arxiv.org/abs/2610.00001",
        ),
        Entry(
            arxiv_id="2610.00002",
            version=2,
            title="Topological phases",
            authors=["Carol C."],
            abstract="Moire systems are fun.",
            categories=["cond-mat.mes-hall"],
            primary_category="cond-mat.mes-hall",
            published="2026-10-01T18:00:00Z",
            abs_url="https://arxiv.org/abs/2610.00002",
        ),
    ]


class TestRenderDigest(unittest.TestCase):
    def test_header_preamble(self):
        md = render_digest(date(2026, 10, 3), ["cond-mat.mes-hall"], make_entries(), [0, 1])
        # preamble: title, hr, metadata, hr
        self.assertTrue(md.startswith("# arXiv digest - 2026-10-03\n\n---\n"))
        self.assertIn(
            "---\n- **Categories:** cond-mat.mes-hall\n- **Papers:** 2\n---\n", md
        )

    def test_entries_are_separated_by_horizontal_rules(self):
        md = render_digest(date(2026, 10, 3), ["x"], make_entries(), [0, 1])
        # 2 rules in the preamble + 1 after each of the 2 entries
        self.assertEqual(md.count("\n---\n"), 4)
        self.assertTrue(md.rstrip().endswith("---"))

    def test_permutation_order_applied(self):
        md = render_digest(date(2026, 10, 3), ["x"], make_entries(), [1, 0])
        self.assertLess(md.index("## 1. Topological phases"), md.index("## 2. Mesoscopic transport"))

    def test_entry_details_present(self):
        md = render_digest(date(2026, 10, 3), ["x"], make_entries(), [0, 1])
        self.assertIn("**Authors:** Alice A., Bob B.", md)
        self.assertIn("[2610.00001v1](https://arxiv.org/abs/2610.00001)", md)
        self.assertIn("We study quantum transport.", md)

    def test_no_entries(self):
        md = render_digest(date(2026, 10, 3), ["x"], [], [])
        self.assertIn("No papers found for this date.", md)
        self.assertIn("- **Papers:** 0", md)


class TestWriteDigest(unittest.TestCase):
    def test_filename_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            path = write_digest(out, date(2026, 10, 3), ["x"], make_entries(), [0, 1])
            self.assertEqual(path.name, "2026-10-03.md")
            first = path.read_text(encoding="utf-8")
            # overwriting is allowed (re-rank same day)
            path2 = write_digest(out, date(2026, 10, 3), ["x"], make_entries(), [1, 0])
            self.assertEqual(path, path2)
            second = path.read_text(encoding="utf-8")
            self.assertNotEqual(first, second)
            self.assertLess(second.index("## 1. Topological phases"), second.index("## 2. Mesoscopic transport"))


class TestWrap(unittest.TestCase):
    def test_short_title_unchanged(self):
        from arxiv_parser.models import wrap_title
        self.assertEqual(wrap_title("Short title", 3), "## 3. Short title")

    def test_long_title_wraps_with_continuation_prefix(self):
        from arxiv_parser.models import wrap_title
        title = "word " * 20
        out = wrap_title(title, 1, width=30)
        lines = out.split("\n")
        self.assertGreater(len(lines), 1)
        self.assertTrue(lines[0].startswith("## 1. "))
        for line in lines[1:]:
            self.assertTrue(line.startswith("##   "))
            self.assertLessEqual(len(line), 30)

    def test_wrap_never_splits_long_tokens(self):
        from arxiv_parser.models import wrap_text
        text = "a $\mathrm{veryveryveryverylongmathexpression}$ b"
        out = wrap_text(text, width=20)
        self.assertIn("$\mathrm{veryveryveryverylongmathexpression}$", out)

    def test_wrap_splits_at_whitespace(self):
        from arxiv_parser.models import wrap_text
        out = wrap_text("one two three four five", width=10)
        for line in out.split("\n"):
            self.assertLessEqual(len(line), 10)


if __name__ == "__main__":
    unittest.main()
