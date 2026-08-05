from __future__ import annotations

import unittest

from dubbing_pipeline.alignment import _character_evidence, normalize_alignment_text
from dubbing_pipeline.qa_v2 import fold


class GermanNormalizationTests(unittest.TestCase):
    def test_umlauts_and_sharp_s_are_not_collapsed(self):
        for accented, plain in (("schön", "schon"), ("würde", "wurde"), ("müsste", "musste"), ("für", "fur"), ("Maße", "Masse")):
            self.assertNotEqual(fold(accented), fold(plain), (accented, plain))
            self.assertNotEqual(normalize_alignment_text(accented), normalize_alignment_text(plain), (accented, plain))

    def test_case_and_apostrophe_variants_remain_equivalent(self):
        self.assertEqual(fold("Für"), fold("FÜR"))
        self.assertEqual(fold("nächste"), fold("NÄCHSTE"))
        self.assertEqual(normalize_alignment_text("geht’s"), normalize_alignment_text("geht's"))

    def test_character_alignment_reports_real_contrast(self):
        def chars(value: str):
            return [{"char": char, "start": index * .02, "end": (index + 1) * .02, "score": .9} for index, char in enumerate(value)]

        result = _character_evidence("Maße", {"char_segments": chars("Masse")}, [])
        expected = [row for row in result["char_segments"] if row.get("expected_index") is not None]
        self.assertTrue(any(row.get("operation") in {"SUBSTITUTE", "DELETE"} for row in expected))
        self.assertTrue(any(row.get("operation") == "INSERT" for row in result["char_segments"]))
        self.assertLess(result["native_char_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
