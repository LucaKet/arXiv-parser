"""Tests for the console width-limiting helper."""

from __future__ import annotations

import contextlib
import io
import unittest

from arxiv_parser.console import say


def capture(message: str, width: int = 40) -> list[str]:
    buf = io.StringIO()
    say(message, width=width, file=buf)
    return buf.getvalue().rstrip("\n").split("\n")


class TestSay(unittest.TestCase):
    def test_short_message_unchanged(self):
        lines = capture("hello", width=40)
        self.assertEqual(lines, ["hello"])

    def test_long_message_wraps_within_width(self):
        lines = capture("one two three four five six seven", width=15)
        for line in lines:
            self.assertLessEqual(len(line), 15)

    def test_wraps_at_word_boundaries(self):
        lines = capture("alpha bravo charlie delta", width=13)
        self.assertEqual(lines, ["alpha bravo", "charlie delta"])

    def test_indent_preserved_on_continuation_lines(self):
        lines = capture("  warning: alpha bravo charlie delta", width=20)
        self.assertEqual(lines, ["  warning: alpha", "  bravo charlie", "  delta"])

    def test_overlong_token_truncated_with_ellipsis(self):
        lines = capture("prefix http://very.long.url/that/never/ends/aaaaaaaaaaaa", width=20)
        for line in lines:
            self.assertLessEqual(len(line), 20)
        self.assertIn("\u2026", "\n".join(lines))

    def test_multiline_input_handled(self):
        lines = capture("first line\nsecond line", width=40)
        self.assertEqual(lines, ["first line", "second line"])


if __name__ == "__main__":
    unittest.main()
