#!/usr/bin/env python3
"""Run the dependency-light generic tests without pytest."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("generic_tests", Path(__file__).with_name("test_generic_core.py"))
module = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(module)
for name in sorted(item for item in dir(module) if item.startswith("test_")):
    getattr(module, name)()
print("generic tests: PASS")
