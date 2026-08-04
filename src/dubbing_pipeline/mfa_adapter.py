"""MFA subprocess adapter restricted to diagnostics."""
from __future__ import annotations
import hashlib, os, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .textgrid import TextGrid, parse_textgrid

@dataclass(frozen=True)
class MFAAssets:
    acoustic_model: Path
    dictionary: Path
    g2p: Path | None = None
    language: str = "de"
    acoustic_sha256: str = ""
    dictionary_sha256: str = ""
    g2p_sha256: str | None = None

@dataclass(frozen=True)
class MFACapability:
    executable: str
    version: str
    command_variant: str

@dataclass(frozen=True)
class MFAResult:
    status: str
    textgrid_path: str | None
    coverage: float | None
    authority: str = "DIAGNOSTIC_ONLY"
    reason: str = ""

def probe_mfa(executable: str = "mfa", *, timeout_seconds: float = 10.0) -> MFACapability:
    try: result=subprocess.run([executable,"--version"],capture_output=True,text=True,timeout=timeout_seconds,check=False)
    except (OSError, subprocess.TimeoutExpired) as exc: raise RuntimeError(f"MFA probe failed: {exc}") from exc
    if result.returncode != 0: raise RuntimeError(result.stderr.strip() or "MFA --version failed")
    version=(result.stdout or result.stderr).strip().splitlines()[0] if (result.stdout or result.stderr).strip() else "unknown"
    variant="align_one_hf" if "align_one_hf" in (result.stdout+result.stderr) else "align_one"
    return MFACapability(executable,version,variant)

def validate_assets(assets: MFAAssets) -> dict[str, Any]:
    rows=[]
    for key, path, expected in (("acoustic_model",assets.acoustic_model,assets.acoustic_sha256),("dictionary",assets.dictionary,assets.dictionary_sha256),("g2p",assets.g2p,assets.g2p_sha256)):
        if path is None: rows.append({"asset":key,"status":"NOT_CONFIGURED"}); continue
        if not Path(path).is_file(): raise FileNotFoundError(f"MFA asset missing: {path}")
        digest=hashlib.sha256(Path(path).read_bytes()).hexdigest();
        if expected and digest.casefold()!=expected.casefold(): raise ValueError(f"MFA asset hash mismatch: {key}")
        rows.append({"asset":key,"path":str(path),"sha256":digest,"status":"VALID"})
    return {"language":assets.language,"assets":rows}

def align_diagnostic(capability: MFACapability, assets: MFAAssets, audio_path: str | Path, transcript: str, output_dir: str | Path, *, timeout_seconds: float = 120.0) -> MFAResult:
    if not transcript.strip(): return MFAResult("MFA_NOT_APPLICABLE",None,None,reason="empty_transcript")
    validate_assets(assets); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    # The command is explicit; no trial-and-error fallback between MFA APIs.
    cmd=[capability.executable, capability.command_variant, "--clean", "--single_speaker", str(audio_path), str(out)]
    try: completed=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout_seconds,check=False)
    except subprocess.TimeoutExpired: return MFAResult("MFA_TIMEOUT",None,None,reason="timeout")
    except OSError as exc: return MFAResult("MFA_ERROR",None,None,reason=str(exc))
    if completed.returncode!=0: return MFAResult("MFA_ERROR",None,None,reason=(completed.stderr or completed.stdout).strip()[-1000:])
    grids=sorted(out.rglob("*.TextGrid"))
    if not grids: return MFAResult("MFA_NO_TEXTGRID",None,None,reason="aligner_returned_no_textgrid")
    grid=parse_textgrid(grids[0]); return MFAResult("MFA_DIAGNOSTIC",str(grids[0]),grid.coverage(transcript),reason="not_authoritative")

__all__=["MFAAssets","MFACapability","MFAResult","probe_mfa","validate_assets","align_diagnostic"]
