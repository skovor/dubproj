"""Require benchmark evidence and second-game validation before promotion."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("benchmark"); p.add_argument("second_game"); args=p.parse_args(); benchmark=json.loads(Path(args.benchmark).read_text(encoding="utf-8")); second=json.loads(Path(args.second_game).read_text(encoding="utf-8")); errors=[]
    if not benchmark.get("manifest_digest"): errors.append("benchmark manifest digest missing")
    if int(benchmark.get("blocked",1)) or int(benchmark.get("failed",1)): errors.append("benchmark has blocked/failed lines")
    if not benchmark.get("real_audio"): errors.append("benchmark is not marked real_audio")
    if not second.get("valid") or not second.get("independent_adapter"): errors.append("second-game adapter validation missing")
    result={"promotable":not errors,"errors":errors,"benchmark":benchmark.get("manifest_digest"),"second_game":second.get("game_id")}; print(json.dumps(result,indent=2)); return 0 if not errors else 2
if __name__=="__main__": raise SystemExit(main())
