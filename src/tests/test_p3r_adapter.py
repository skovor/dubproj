from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from adapters.p3r.adapter import P3RAdapter, P3RAdapterConfig


class P3RAdapterTests(unittest.TestCase):
    def test_runtime_destination_is_confined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config=P3RAdapterConfig(root, root/"manifest.json", root/"runtime")
            adapter=P3RAdapter(config)
            self.assertEqual(adapter.runtime_destinations("audio/file.wav")[0], (root/"runtime"/"audio/file.wav").resolve())
            with self.assertRaises(ValueError): adapter.runtime_destinations("../escape.wav")

    def test_runtime_smoke_is_explicitly_blocked_without_game_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); runtime=root/"runtime"; runtime.mkdir()
            result=P3RAdapter(P3RAdapterConfig(root, root/"manifest.json", runtime)).runtime_smoke()
            self.assertEqual(result["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
