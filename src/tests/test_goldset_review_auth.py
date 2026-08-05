from __future__ import annotations

import unittest

from scripts.serve_goldset_review import authorized_identity, parse_token_specs


class GoldsetReviewAuthTests(unittest.TestCase):
    def test_tokens_bind_mutations_to_preconfigured_identity(self):
        tokens = parse_token_specs(["reviewer-a:secret"])
        self.assertTrue(authorized_identity({"X-Goldset-Token": "secret"}, tokens, "reviewer-a"))
        self.assertFalse(authorized_identity({"X-Goldset-Token": "secret"}, tokens, "reviewer-b"))
        self.assertFalse(authorized_identity({}, tokens, "reviewer-a"))

    def test_token_specs_are_not_ambiguous(self):
        with self.assertRaises(ValueError):
            parse_token_specs(["reviewer-a"])


if __name__ == "__main__":
    unittest.main()
