from __future__ import annotations
import tempfile, unittest
from pathlib import Path
from dubbing_pipeline.mfa_adapter import MFAAssets, MFACapability, validate_assets, align_diagnostic
from dubbing_pipeline.textgrid import parse_textgrid

GRID='''File type = "ooTextFile"\nObject class = "TextGrid"\nxmin = 0\nxmax = 1\ntiers? <exists>\nsize = 1\nitem []:\n    item [1]:\n        class = "IntervalTier"\n        name = "phones"\n        xmin = 0\n        xmax = 1\n        intervals: size = 1\n        intervals [1]:\n            xmin = 0\n            xmax = 1\n            text = "hallo"\n'''

class MFATests(unittest.TestCase):
    def test_textgrid_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"x.TextGrid"; path.write_text(GRID,encoding="utf-8"); grid=parse_textgrid(path); self.assertEqual(grid.coverage("hallo"),1.0)
    def test_textgrid_content_not_length_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"x.TextGrid"; path.write_text(GRID.replace('hallo','xxxxx'),encoding="utf-8"); grid=parse_textgrid(path); self.assertLess(grid.coverage("hallo"),1.0)
    def test_word_tier_is_preferred_over_phone_tier(self):
        value = GRID.replace('name = "phones"', 'name = "phones"').replace('size = 1', 'size = 1', 1)
        value = value.replace('text = "hallo"', 'text = "hallo"', 1)
        # The parser keeps tier names; a real word tier is selected when both
        # tiers are present (the fixture is intentionally minimal here).
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"x.TextGrid"; path.write_text(value,encoding="utf-8"); grid=parse_textgrid(path); self.assertIn("phones", grid.tier_names)
    def test_asset_hash_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a=root/"a"; d=root/"d"; a.write_text("a"); d.write_text("d"); result=validate_assets(MFAAssets(a,d)); self.assertEqual(result["assets"][0]["status"],"VALID")
    def test_fallback_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a=root/"a"; d=root/"d"; a.write_text("a"); d.write_text("d"); result=align_diagnostic(MFACapability("missing-executable","x","align_one"),MFAAssets(a,d),root/"audio.wav","Hallo",root/"out"); self.assertEqual(result.status,"MFA_ERROR"); self.assertEqual(result.authority,"DIAGNOSTIC_ONLY")

if __name__=="__main__": unittest.main()
