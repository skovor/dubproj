from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dubbing_pipeline.mfa_adapter import MFAAssets, validate_assets, probe_mfa

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("acoustic_model"); p.add_argument("dictionary"); p.add_argument("--g2p"); p.add_argument("--executable",default="mfa"); args=p.parse_args(); assets=MFAAssets(Path(args.acoustic_model),Path(args.dictionary),Path(args.g2p) if args.g2p else None); print(json.dumps({"capability":probe_mfa(args.executable).__dict__,"assets":validate_assets(assets)},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
