"""Create a gold-set queue from a JSONL clip manifest; never creates labels."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing_pipeline.goldset import ClipRecord, GoldsetStore

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("manifest"); parser.add_argument("database"); parser.add_argument("--export")
    args = parser.parse_args(); rows = json.loads(Path(args.manifest).read_text(encoding="utf-8")) if Path(args.manifest).suffix == ".json" else [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    store = GoldsetStore(args.database)
    try:
        for row in rows: store.add_clip(ClipRecord(**row))
        if args.export: print(json.dumps(store.export(args.export), indent=2))
    finally: store.close()
    return 0
if __name__ == "__main__": raise SystemExit(main())
