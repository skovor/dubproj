"""MFA subprocess adapter restricted to diagnostics."""
from __future__ import annotations
import hashlib, json, os, shutil, subprocess, tempfile
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
    supports_single_speaker: bool = False

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
    # ``--version`` does not advertise subcommands consistently across MFA
    # releases. Probe help for the exact command instead of guessing from the
    # version string.
    variant = None
    for candidate in ("align_one", "align_one_hf"):
        try:
            help_result = subprocess.run([executable, candidate, "--help"], capture_output=True, text=True, timeout=timeout_seconds, check=False)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if help_result.returncode == 0:
            variant = candidate
            help_text = (help_result.stdout or "") + (help_result.stderr or "")
            return MFACapability(executable, version, candidate, "single_speaker" in help_text)
    raise RuntimeError("MFA does not expose align_one or align_one_hf")

def validate_assets(assets: MFAAssets) -> dict[str, Any]:
    def digest(path: Path) -> str:
        if path.is_file():
            return hashlib.sha256(path.read_bytes()).hexdigest()
        if path.is_dir():
            rows = []
            for item in sorted(item for item in path.rglob("*") if item.is_file()):
                rows.append((item.relative_to(path).as_posix(), item.stat().st_size, hashlib.sha256(item.read_bytes()).hexdigest()))
            return hashlib.sha256(repr(rows).encode("utf-8")).hexdigest()
        raise FileNotFoundError(path)
    rows=[]
    for key, path, expected in (("acoustic_model",assets.acoustic_model,assets.acoustic_sha256),("dictionary",assets.dictionary,assets.dictionary_sha256),("g2p",assets.g2p,assets.g2p_sha256)):
        if path is None: rows.append({"asset":key,"status":"NOT_CONFIGURED"}); continue
        path = Path(path)
        if not path.is_file() and not path.is_dir(): raise FileNotFoundError(f"MFA asset missing: {path}")
        file_digest=digest(path)
        if expected and file_digest.casefold()!=expected.casefold(): raise ValueError(f"MFA asset hash mismatch: {key}")
        rows.append({"asset":key,"path":str(path),"sha256":file_digest,"status":"VALID"})
    return {"language":assets.language,"assets":rows}

def align_diagnostic(capability: MFACapability, assets: MFAAssets, audio_path: str | Path, transcript: str, output_dir: str | Path, *, timeout_seconds: float = 120.0) -> MFAResult:
    if not transcript.strip(): return MFAResult("MFA_NOT_APPLICABLE",None,None,reason="empty_transcript")
    validate_assets(assets); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    audio = Path(audio_path)
    if not audio.is_file(): return MFAResult("MFA_ERROR",None,None,reason=f"missing_audio:{audio}")
    # MFA's align_one family requires the transcript as a positional input;
    # the previous adapter omitted it and therefore never aligned the intended
    # words. Keep the temporary corpus files isolated and deterministic.
    with tempfile.TemporaryDirectory(prefix="mfa-diagnostic-") as temp:
        corpus = Path(temp) / "corpus"; corpus.mkdir()
        corpus_audio = corpus / audio.name; shutil.copy2(audio, corpus_audio)
        transcript_path = corpus_audio.with_suffix(".txt"); transcript_path.write_text(transcript.strip() + "\n", encoding="utf-8")
        cmd=[capability.executable, capability.command_variant, "--clean", "--output_format", "json"]
        if capability.supports_single_speaker:
            cmd.append("--single_speaker")
        cmd.extend([str(corpus_audio), str(transcript_path), str(assets.dictionary), str(assets.acoustic_model), str(out)])
        completed = None
        try:
            completed=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout_seconds,check=False)
        except subprocess.TimeoutExpired: return MFAResult("MFA_TIMEOUT",None,None,reason="timeout")
        except OSError as exc: return MFAResult("MFA_ERROR",None,None,reason=str(exc))
    if completed.returncode!=0: return MFAResult("MFA_ERROR",None,None,reason=(completed.stderr or completed.stdout).strip()[-1000:])
    grids=sorted(out.rglob("*.TextGrid")); jsons=sorted(out.rglob("*.json"))
    if grids:
        grid=parse_textgrid(grids[0]); return MFAResult("MFA_DIAGNOSTIC",str(grids[0]),grid.coverage(transcript),reason="not_authoritative")
    if jsons:
        value = json.loads(jsons[0].read_text(encoding="utf-8"))
        rows = value.get("words", value.get("word_segments", [])) if isinstance(value, dict) else []
        heard = " ".join(str(row.get("word", row.get("text", ""))) for row in rows if isinstance(row, dict))
        from difflib import SequenceMatcher
        coverage = SequenceMatcher(a=transcript.casefold().split(), b=heard.casefold().split(), autojunk=False).ratio()
        return MFAResult("MFA_DIAGNOSTIC",str(jsons[0]),coverage,reason="not_authoritative")
    return MFAResult("MFA_NO_ALIGNMENT",None,None,reason="aligner_returned_no_alignment")

__all__=["MFAAssets","MFACapability","MFAResult","probe_mfa","validate_assets","align_diagnostic"]
