"""Tests for permutation validation, LLM reply parsing, and the rank() flow."""

from __future__ import annotations

import unittest

from arxiv_parser.llm import LLMError, LLMProvider
from arxiv_parser.models import Entry
from arxiv_parser.ranker import (
    build_papers_text,
    build_user_prompt,
    parse_permutation,
    rank,
    rank_digest,
    validate_permutation,
)


def make_entries(n: int) -> list[Entry]:
    return [
        Entry(
            arxiv_id=f"2610.000{i:02d}",
            version=1,
            title=f"Paper {i}",
            authors=[f"Author {i}"],
            abstract=f"Abstract {i}",
            published="2026-10-01T00:00:00Z",
        )
        for i in range(n)
    ]


class FakeProvider(LLMProvider):
    def __init__(self, replies: list[str] | Exception) -> None:
        self.replies = replies if isinstance(replies, list) else [replies]
        self.calls = 0
        self.prompts: list[tuple[str, str]] = []

    @property
    def name(self) -> str:  # type: ignore[override]
        return "fake"

    def complete(self, system: str, user: str, temperature: float) -> str:
        self.calls += 1
        self.prompts.append((system, user))
        reply = self.replies[min(self.calls - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply


class TestValidatePermutation(unittest.TestCase):
    def test_valid(self):
        self.assertTrue(validate_permutation([2, 0, 1], 3))

    def test_wrong_length(self):
        self.assertFalse(validate_permutation([0, 1], 3))

    def test_duplicate(self):
        self.assertFalse(validate_permutation([0, 0, 2], 3))

    def test_out_of_range(self):
        self.assertFalse(validate_permutation([0, 1, 3], 3))

    def test_not_list(self):
        self.assertFalse(validate_permutation("012", 3))
        self.assertFalse(validate_permutation({"x": 1}, 1))

    def test_floats_and_bools_rejected(self):
        self.assertFalse(validate_permutation([0.0, 1.0], 2))
        self.assertFalse(validate_permutation([True, False], 2))


class TestParsePermutation(unittest.TestCase):
    def test_plain_array(self):
        self.assertEqual(parse_permutation("[2, 0, 1]", 3), [2, 0, 1])

    def test_markdown_fence(self):
        self.assertEqual(parse_permutation("```json\n[1, 0, 2]\n```", 3), [1, 0, 2])

    def test_surrounding_chatter(self):
        self.assertEqual(
            parse_permutation("Here you go: [0, 3, 1, 2] hope that helps", 4),
            [0, 3, 1, 2],
        )

    def test_invalid_permutation_returns_none(self):
        self.assertIsNone(parse_permutation("[0, 0, 2]", 3))
        self.assertIsNone(parse_permutation("[0, 1]", 3))

    def test_malformed_json_returns_none(self):
        self.assertIsNone(parse_permutation("[0, 1,", 3))
        self.assertIsNone(parse_permutation("no brackets here", 3))


class TestPromptBuilding(unittest.TestCase):
    def test_papers_text_indexes_and_content(self):
        entries = make_entries(2)
        text = build_papers_text(entries)
        self.assertIn("[0] 2610.00000 | Paper 0 | Author 0 | Abstract 0", text)
        self.assertIn("[1] 2610.00001 | Paper 1 | Author 1 | Abstract 1", text)

    def test_user_prompt_includes_interests(self):
        entries = make_entries(2)
        prompt = build_user_prompt(entries, "  spin qubits\nmoires  ")
        self.assertIn("USER INTERESTS:\nspin qubits\nmoires", prompt)
        self.assertIn("PAPERS (2):", prompt)

    def test_user_prompt_without_interests(self):
        prompt = build_user_prompt(make_entries(1), "   ")
        self.assertIn("none provided", prompt)

    def test_papers_text_without_abstracts(self):
        entries = make_entries(2)
        entries[0].categories = ["cond-mat.mes-hall"]
        text = build_papers_text(entries, with_abstracts=False)
        self.assertIn("[0] 2610.00000 | Paper 0 | Author 0 | cond-mat.mes-hall", text)
        self.assertNotIn("Abstract 0", text)

    def test_papers_text_with_abstracts(self):
        text = build_papers_text(make_entries(2), with_abstracts=True)
        self.assertIn("| Abstract 0", text)


class TestRank(unittest.TestCase):
    def test_success_returns_permutation(self):
        perm, note = rank(FakeProvider("[2, 0, 1]"), make_entries(3), "x")
        self.assertEqual(perm, [2, 0, 1])
        self.assertEqual(note, "")

    def test_invalid_then_valid_retry(self):
        provider = FakeProvider(["garbage", "[1, 0, 2]"])
        perm, note = rank(provider, make_entries(3), "x", max_retries=1)
        self.assertEqual(perm, [1, 0, 2])
        self.assertEqual(note, "")
        self.assertEqual(provider.calls, 2)

    def test_all_invalid_falls_back_to_identity(self):
        provider = FakeProvider(["garbage", "also garbage"])
        perm, note = rank(provider, make_entries(3), "x", max_retries=1)
        self.assertEqual(perm, [0, 1, 2])
        self.assertIn("no valid permutation", note)

    def test_llm_error_falls_back_to_identity(self):
        provider = FakeProvider(LLMError("boom"))
        perm, note = rank(provider, make_entries(3), "x", max_retries=1)
        self.assertEqual(perm, [0, 1, 2])
        self.assertIn("LLM call failed", note)

    def test_no_provider_identity(self):
        perm, note = rank(None, make_entries(3), "x")
        self.assertEqual(perm, [0, 1, 2])
        self.assertIn("no LLM provider", note)

    def test_trivial_sizes(self):
        perm, note = rank(FakeProvider("anything"), make_entries(1), "x")
        self.assertEqual(perm, [0])
        self.assertEqual(note, "")
        perm, note = rank(FakeProvider("anything"), [], "x")
        self.assertEqual(perm, [])


class TestRankDigest(unittest.TestCase):
    def test_small_digest_single_pass_with_abstracts(self):
        provider = FakeProvider("[2, 0, 1]")
        perm, note = rank_digest(provider, make_entries(3), "x", two_pass_threshold=60)
        self.assertEqual(perm, [2, 0, 1])
        self.assertEqual(note, "")
        self.assertEqual(provider.calls, 1)
        self.assertIn("Abstract 0", provider.prompts[0][1])

    def test_large_digest_two_passes(self):
        # 5 papers, threshold 3, refine top 3
        provider = FakeProvider(["[4, 0, 1, 2, 3]", "[2, 0, 1]"])
        perm, note = rank_digest(provider, make_entries(5), "x", two_pass_threshold=3, refine_top=3)
        self.assertEqual(provider.calls, 2)
        # pass 1: titles only (no abstracts); pass 2: abstracts for the top 3
        self.assertNotIn("Abstract 0", provider.prompts[0][1])
        self.assertIn("Abstract 0", provider.prompts[1][1])
        self.assertIn("PAPERS (3):", provider.prompts[1][1])
        # pass1 order: [4,0,1,2,3]; top 3 = [4,0,1]; pass2 reorders them to [1,4,0]
        self.assertEqual(perm, [1, 4, 0, 2, 3])
        self.assertEqual(note, "")

    def test_second_pass_failure_keeps_first_pass_order(self):
        provider = FakeProvider(["[4, 0, 1, 2, 3]", "garbage"])
        perm, note = rank_digest(
            provider, make_entries(5), "x",
            two_pass_threshold=3, refine_top=3, max_retries=0,
        )
        self.assertEqual(perm, [4, 0, 1, 2, 3])
        self.assertIn("second pass failed", note)

    def test_first_pass_failure_propagates(self):
        provider = FakeProvider("garbage")
        perm, note = rank_digest(
            provider, make_entries(5), "x",
            two_pass_threshold=3, refine_top=3, max_retries=0,
        )
        self.assertEqual(perm, [0, 1, 2, 3, 4])
        self.assertIn("no valid permutation", note)

    def test_no_provider_identity(self):
        perm, note = rank_digest(None, make_entries(100), "x")
        self.assertEqual(perm, list(range(100)))
        self.assertIn("no LLM provider", note)


if __name__ == "__main__":
    unittest.main()
