"""Tests for the arXiv API client (no network; recorded fixture + synthetic XML)."""

from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path

from arxiv_parser.arxiv_api import (
    build_search_query,
    dedupe,
    parse_arxiv_id,
    parse_entries,
    parse_feed,
)
from arxiv_parser.models import Entry

FIXTURE = Path(__file__).parent / "fixtures" / "mes-hall_sample.xml"

SYNTHETIC = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns="http://www.w3.org/2005/Atom">
  <title>test</title>
  <opensearch:totalResults>3</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2610.00001v1</id>
    <title>  Paper   One </title>
    <published>2026-10-01T17:59:00Z</published>
    <arxiv:primary_category term="cond-mat.mes-hall"/>
    <category term="cond-mat.mes-hall"/>
    <category term="cond-mat.str-el"/>
    <summary>First  abstract\n   with   weird spacing.</summary>
    <author><name>Alice A.</name></author>
    <author><name>Bob B.</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/cond-mat/0204058v2</id>
    <title>Old style id</title>
    <published>2002-04-05T00:00:00Z</published>
    <summary>Old paper.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2610.00002</id>
    <title>No version suffix</title>
    <published>2026-10-01T18:00:00Z</published>
    <summary>Odd id.</summary>
  </entry>
</feed>
"""


class TestBuildSearchQuery(unittest.TestCase):
    def test_single_category(self):
        q = build_search_query(["cond-mat.mes-hall"], date(2026, 10, 3), 2)
        self.assertIn("cat:cond-mat.mes-hall", q)
        self.assertIn("submittedDate:[202610010000 TO 202610032359]", q)

    def test_multiple_categories_sorted_deduped(self):
        q = build_search_query(["cs.LG", "cs.LG", "stat.ML"], date(2026, 10, 3), 2)
        self.assertIn("(cat:cs.LG OR cat:stat.ML)", q)

    def test_window_end_date_is_query_day(self):
        q = build_search_query(["cs.LG"], date(2026, 1, 31), 1)
        self.assertIn("submittedDate:[202601300000 TO 202601312359]", q)


class TestParseArxivId(unittest.TestCase):
    def test_new_style_with_version(self):
        self.assertEqual(parse_arxiv_id("http://arxiv.org/abs/2610.02207v1"), ("2610.02207", 1))

    def test_old_style_with_version(self):
        self.assertEqual(
            parse_arxiv_id("http://arxiv.org/abs/cond-mat/0204058v2"),
            ("cond-mat/0204058", 2),
        )

    def test_no_version_suffix(self):
        self.assertEqual(parse_arxiv_id("http://arxiv.org/abs/2610.00002"), ("2610.00002", 1))


class TestParseEntries(unittest.TestCase):
    def test_recorded_fixture(self):
        entries, total = parse_feed(FIXTURE.read_bytes())
        self.assertEqual(total, 21)
        self.assertEqual(len(entries), 5)
        for e in entries:
            self.assertTrue(e.arxiv_id)
            self.assertTrue(e.title)
            self.assertTrue(e.abstract)
            self.assertTrue(e.published)
            self.assertTrue(e.abs_url.startswith("https://arxiv.org/abs/"))

    def test_synthetic_whitespace_and_fields(self):
        entries = parse_entries(SYNTHETIC.encode())
        self.assertEqual(len(entries), 3)
        e0 = entries[0]
        self.assertEqual(e0.title, "Paper One")
        self.assertEqual(e0.abstract, "First abstract with weird spacing.")
        self.assertEqual(e0.authors, ["Alice A.", "Bob B."])
        self.assertEqual(e0.categories, ["cond-mat.mes-hall", "cond-mat.str-el"])
        self.assertEqual(e0.primary_category, "cond-mat.mes-hall")
        self.assertEqual(e0.abs_url, "https://arxiv.org/abs/2610.00001")

    def test_old_style_id_version(self):
        entries = parse_entries(SYNTHETIC.encode())
        self.assertEqual((entries[1].arxiv_id, entries[1].version), ("cond-mat/0204058", 2))

    def test_no_version_suffix_defaults_to_1(self):
        entries = parse_entries(SYNTHETIC.encode())
        self.assertEqual((entries[2].arxiv_id, entries[2].version), ("2610.00002", 1))


class TestDedupe(unittest.TestCase):
    def test_keeps_first_occurrence(self):
        e1 = Entry(arxiv_id="1", version=1, title="A", published="2026-10-01T00:00:00Z")
        e2 = Entry(arxiv_id="1", version=1, title="A copy", published="2026-10-01T00:00:00Z")
        e3 = Entry(arxiv_id="2", version=1, title="B", published="2026-10-01T00:00:00Z")
        out = dedupe([e1, e2, e3])
        self.assertEqual([e.arxiv_id for e in out], ["1", "2"])
        self.assertEqual(out[0].title, "A")


if __name__ == "__main__":
    unittest.main()
