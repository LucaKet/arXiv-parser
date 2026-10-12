"""Tests for the listing-page batch discovery (recorded fixture, no network)."""

from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from arxiv_parser.batch import Batch, ListingError, enrich, parse_listing, resolve_batch
from arxiv_parser.models import Entry

FIXTURE = Path(__file__).parent / "fixtures" / "mh-new.html"


class TestParseListing(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = parse_listing(FIXTURE.read_text(encoding="utf-8"))

    def test_sections_have_expected_counts(self):
        self.assertEqual(len(self.batch.new), 21)
        self.assertEqual(len(self.batch.cross), 10)
        self.assertEqual(len(self.batch.replacements), 12)

    def test_label_and_batch_date(self):
        self.assertIn("Friday, 2 October 2026", self.batch.label)
        self.assertEqual(self.batch.batch_date, date(2026, 10, 2))

    def test_first_new_submission_is_the_email_first_entry(self):
        self.assertEqual(self.batch.new[0][0], "2610.00273")
        self.assertIn("Tunable flat bands", self.batch.new[0][1])

    def test_cross_and_replacement_samples(self):
        cross_ids = [aid for aid, _ in self.batch.cross]
        repl_ids = [aid for aid, _ in self.batch.replacements]
        self.assertIn("2610.00324", cross_ids)   # Causality violations (str-el)
        self.assertIn("2509.05114", repl_ids)    # Non-Abelian Quantum Metric

    def test_all_ids_email_order_no_dupes(self):
        ids = self.batch.all_ids()
        self.assertEqual(len(ids), 43)
        self.assertEqual(len(set(ids)), 43)
        # email order: new section before cross before replacements
        self.assertEqual(ids[0], "2610.00273")
        self.assertLess(ids.index("2610.00324"), ids.index("2509.05114"))

    def test_unknown_section_ignored(self):
        batch = Batch()
        # parse_listing raises on empty, so test via a minimal synthetic page
        raw = (
            "<html><h3>Showing new listings for Monday, 1 October 2026</h3>"
            "<h3>New submissions (showing 1 of 1 entries)</h3>"
            "<div>arXiv:2610.00001 Title:</span> A Paper</div>"
            "<h3>Something else</h3><div>arXiv:2610.00002</div></html>"
        )
        b = parse_listing(raw)
        self.assertEqual([aid for aid, _ in b.new], ["2610.00001"])
        self.assertEqual(b.cross, [])
        self.assertEqual(b.replacements, [])

    def test_empty_page_raises(self):
        with self.assertRaises(ListingError):
            parse_listing("<html><body>nothing here</body></html>")


class TestEnrich(unittest.TestCase):
    def test_missing_ids_become_stubs_in_order(self):
        # Simulate an API response that returns only one of the two papers.
        import arxiv_parser.batch as batch_mod

        xml = (
            '<?xml version="1.0"?>'
            '<feed xmlns:arxiv="http://arxiv.org/schemas/atom" '
            'xmlns="http://www.w3.org/2005/Atom">'
            "<entry>"
            "<id>http://arxiv.org/abs/2610.00002v1</id>"
            "<title>Paper Two</title><published>2026-10-01T00:00:00Z</published>"
            "<summary>abs2</summary>"
            "</entry></feed>"
        ).encode()
        with mock.patch.object(batch_mod, "_http_get", return_value=xml):
            entries = enrich(["2610.00001", "2610.00002"], {"2610.00001": "Paper One (listing)"})
        self.assertEqual([e.arxiv_id for e in entries], ["2610.00001", "2610.00002"])
        self.assertEqual(entries[0].title, "Paper One (listing)")  # stub from listing
        self.assertEqual(entries[1].title, "Paper Two")            # API metadata


class TestResolveBatch(unittest.TestCase):
    def test_matching_date_returns_entries_in_email_order(self):
        import arxiv_parser.batch as batch_mod

        xml = (
            '<?xml version="1.0"?>'
            '<feed xmlns:arxiv="http://arxiv.org/schemas/atom" '
            'xmlns="http://www.w3.org/2005/Atom">'
        )
        for aid, title in [("2610.00273", "First"), ("2610.00324", "Cross"), ("2509.05114", "Repl")]:
            xml += (
                f"<entry><id>http://arxiv.org/abs/{aid}v1</id><title>{title}</title>"
                f"<published>2026-09-01T00:00:00Z</published><summary>abs</summary></entry>"
            )
        xml += "</feed>"
        with mock.patch.object(
            batch_mod, "_http_get", side_effect=[FIXTURE.read_bytes(), xml.encode()]
        ):
            entries, note = resolve_batch(["cond-mat.mes-hall"], date(2026, 10, 3))
        # email order: all new submissions first (the API only had 3 of 43;
        # the rest become stubs, order preserved)
        self.assertEqual([e.arxiv_id for e in entries][:3], ["2610.00273", "2610.00326", "2610.00336"])
        self.assertEqual(len(entries), 43)
        self.assertLess(entries.index(next(e for e in entries if e.arxiv_id == "2610.00324")), entries.index(next(e for e in entries if e.arxiv_id == "2509.05114")))
        self.assertIn("Friday, 2 October 2026", note)

    def test_date_too_old_returns_none(self):
        import arxiv_parser.batch as batch_mod

        with mock.patch.object(batch_mod, "_http_get", return_value=FIXTURE.read_bytes()):
            self.assertIsNone(resolve_batch(["cond-mat.mes-hall"], date(2026, 9, 20)))

    def test_merges_multiple_categories_and_dedupes(self):
        import arxiv_parser.batch as batch_mod

        other = (
            "<html><h3>Showing new listings for Friday, 2 October 2026</h3>"
            "<h3>New submissions (showing 2 of 2 entries)</h3>"
            "<div>arXiv:2610.00273 Title:</span> Same paper as mes-hall</div>"
            "<div>arXiv:2610.99999 Title:</span> Only in other cat</div>"
            "</html>"
        ).encode()
        xml = (
            '<?xml version="1.0"?>'
            '<feed xmlns:arxiv="http://arxiv.org/schemas/atom" '
            'xmlns="http://www.w3.org/2005/Atom">'
            "<entry><id>http://arxiv.org/abs/2610.99999v1</id><title>Only</title>"
            "<published>2026-09-01T00:00:00Z</published><summary>abs</summary></entry>"
            "</feed>"
        ).encode()
        with mock.patch.object(
            batch_mod,
            "_http_get",
            side_effect=[FIXTURE.read_bytes(), other, xml],
        ):
            entries, note = resolve_batch(["cond-mat.mes-hall", "cond-mat.str-el"], date(2026, 10, 3))
        ids = [e.arxiv_id for e in entries]
        self.assertEqual(len(ids), len(set(ids)), "duplicates must be removed")
        self.assertIn("2610.00273", ids)      # shared paper kept once
        self.assertIn("2610.99999", ids)      # other-category paper included
        self.assertIn("str-el", note)
        # the announcement heading must appear only once, even with two categories
        self.assertEqual(note.count("Showing new listings"), 1)


if __name__ == "__main__":
    unittest.main()
