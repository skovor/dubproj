from pathlib import Path
from tempfile import TemporaryDirectory
import json

from dubbing_pipeline.config import PipelineConfig
from dubbing_pipeline.asr import ASRCache
from dubbing_pipeline.runtime_lock import collect_runtime_lock, lock_digest, validate_models_lock, validate_runtime_lock
from dubbing_pipeline.models import Line
from dubbing_pipeline.package import PackageFile
from dubbing_pipeline.policy import KEEP_ORIGINAL, SHORT_TTS_QA, TTS, append_ellipsis, classify_line
from dubbing_pipeline.splice import energy_end, split_lead
from dubbing_pipeline.timing import atempo_chain


def test_config_has_no_project_specific_default():
    with TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        config = tmp_path / "config.json"
        config.write_text('{"project_root":"project","target_language":"de"}', encoding="utf-8")
        loaded = PipelineConfig.load(config)
        assert loaded.project_root == (tmp_path / "project").resolve()
        assert "P3R" not in str(loaded.project_root)


def test_subtitle_authority_and_ellipsis():
    line = Line("x", "Speaker", "Hello there", "Hallo dort", 0, 1, subtitle_authorized=False)
    assert classify_line(line).policy == KEEP_ORIGINAL
    line.subtitle_authorized = True
    decision = classify_line(line, append_ellipsis_experiment=True)
    assert decision.policy in {TTS, SHORT_TTS_QA}
    assert decision.tts_text == "Hallo dort..."
    assert append_ellipsis("Hallo dort...") == "Hallo dort..."


def test_isolated_name_call_is_preserved():
    line = Line("x", "Speaker", "Palladion!", "Palladion!", 0, 1, subtitle_authorized=True)
    assert classify_line(line).policy == KEEP_ORIGINAL


def test_fmv_requires_visual_evidence():
    line = Line("x", "Speaker", "Hello", "Hallo", 0, 1, topology="EMBEDDED_FMV", subtitle_authorized=True)
    assert classify_line(line).policy == "BLOCKED"
    line.movie_identity_verified = line.card_identity_verified = line.card_timebase_verified = True
    assert classify_line(line).policy in {TTS, SHORT_TTS_QA}


def test_splice_token_and_atempo_chain():
    assert split_lead("Ugh! Nein, das geht nicht!") == ("Ugh", "Nein, das geht nicht!")
    assert atempo_chain(4.0).startswith("atempo=2.000000")


def test_unprovisioned_lock_is_visible_in_lab_but_blocks_production():
    config = PipelineConfig()
    lab = config.reproducibility_report(strict=False)
    assert lab["status"] == "LAB_UNPINNED"
    strict = config.reproducibility_report(strict=True)
    assert strict["status"] == "BLOCKED"
    assert any("runtime_lock" in item for item in strict["errors"])


def test_runtime_and_model_lock_contracts_reject_unknowns():
    runtime = collect_runtime_lock()
    assert validate_runtime_lock(runtime, strict=False) == []
    assert validate_runtime_lock({"schema": "generic-dubbing-runtime-lock-v1", "status": "UNPROVISIONED", "environment": {}, "dependencies": {}}, strict=True)
    assert validate_models_lock({"schema": "generic-dubbing-model-lock-v1", "status": "UNPROVISIONED", "models": [{"model_id": "m", "revision": None, "sha256": None, "language": "de", "sample_rate": 24000, "backend": "b", "backend_version": None, "files": []}]}, strict=True)


def test_lock_digest_is_unicode_and_order_stable():
    first = {"text": "Für nächste", "sample_rate": 24000}
    second = {"sample_rate": 24000, "text": "Für nächste"}
    assert lock_digest(first) == lock_digest(second)


def test_checked_in_lock_and_policy_schemas_are_valid_json():
    root = Path(__file__).resolve().parents[1]
    for name in ("runtime.lock.json", "models.lock.json", "qa-policy.schema.json", "calibration.schema.json"):
        value = json.loads((root / "config" / name).read_text(encoding="utf-8"))
        assert isinstance(value, dict)


def test_model_revision_changes_asr_cache_identity():
    first = ASRCache(backend_id="whisper", model_id="large-v3", model_revision="rev-a")
    second = ASRCache(backend_id="whisper", model_id="large-v3", model_revision="rev-b")
    assert first._key("a" * 64, "forced_target", "de") != second._key("a" * 64, "forced_target", "de")
