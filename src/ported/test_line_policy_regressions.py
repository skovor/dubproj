#!/usr/bin/env python3
"""Small regression suite for policy mistakes found in the 2026-07-28 audit."""
from __future__ import annotations

import unittest

import line_policy as lp


class LinePolicyRegressionTests(unittest.TestCase):
    def assert_policy(self, source: str, target: str, action: str) -> None:
        self.assertEqual(lp.classify_line(source, target).action, action)

    def test_elongated_lexical_words_are_dubbed(self) -> None:
        for source, target in (
            ("Pleeeease!", "Biiiitte!"),
            ("Chaaarge!", "Angriiiff!"),
            ("Helloooo!", "Haaallo!"),
            ("CHIDORI!", "CHIDORI!"),
        ):
            with self.subTest(target=target):
                self.assert_policy(source, target, lp.SHORT_TTS_QA)

    def test_laughter_and_animals_remain_original(self) -> None:
        for source, target in (
            ("Heheheh.", "Hehehe."),
            ("Hahahahaha!", "Hahahahaha!"),
            ("Ruff!", "Ruff!"),
            ("Hisssss!", "Zischhhh!"),
        ):
            with self.subTest(target=target):
                self.assert_policy(source, target, lp.KEEP_ORIGINAL)

    def test_mild_fillers_are_short_tts(self) -> None:
        for source, target in (
            ("Mm...", "Mm ..."),
            ("Ah...", "Ah ..."),
            ("Uhhh...", "Ähhh ..."),
        ):
            with self.subTest(target=target):
                self.assert_policy(source, target, lp.SHORT_TTS_QA)

    def test_plain_and_extreme_negations_differ(self) -> None:
        self.assert_policy("No.", "Nein.", lp.SHORT_TTS_QA)
        self.assert_policy("NOOOOOO!", "NEEEEEIN!", lp.KEEP_ORIGINAL)

    def test_lexical_sentence_is_not_hidden_by_elongation(self) -> None:
        self.assert_policy(
            "M-Mooommy! Where aaaaaare you!?",
            "M-Mamaaaa! Wo biiiiist du?!",
            lp.TTS,
        )

    def test_inline_stage_direction_does_not_swallow_speech(self) -> None:
        self.assert_policy(
            "Sorry... *pant* *pant*",
            "Sorry ... *keuch* *keuch*",
            lp.SHORT_TTS_QA,
        )
        self.assert_policy(
            "*pant* *pant*",
            "*keuch* *keuch*",
            lp.KEEP_ORIGINAL,
        )

    def test_untranslated_and_developer_rows_do_not_reach_tts(self) -> None:
        self.assertEqual(
            lp.classify_line("おお…", "おお…").reason,
            "foreign_or_untranslated_text",
        )
        self.assertEqual(
            lp.classify_line("UNUSED", "UNUSED").reason,
            "developer_or_unused_text",
        )


if __name__ == "__main__":
    unittest.main()
