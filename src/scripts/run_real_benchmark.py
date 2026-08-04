"""Run a benchmark from a manifest JSON; no fake real-audio claim is inferred."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dubbing_pipeline.benchmark import BenchmarkManifest, run_benchmark

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("manifest"); p.add_argument("output"); args=p.parse_args(); value=json.loads(Path(args.manifest).read_text(encoding="utf-8")); manifest=BenchmarkManifest(**{**value,"line_ids":tuple(value["line_ids"]),"audio_paths":tuple(value["audio_paths"]),"reference_paths":tuple(value["reference_paths"])})
    result=run_benchmark(manifest,lambda line,audio,reference:{"status":"BLOCKED","reason":"runner_not_configured"},require_files=True); Path(args.output).write_text(json.dumps(result.to_dict(),ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(result.to_dict(),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
