from pathlib import Path
from tempfile import TemporaryDirectory
import json

from dubbing_pipeline.config import PipelineConfig
from dubbing_pipeline.asr import ASRCache
from dubbing_pipeline.runtime_lock import (
    MODELS_LOCK_SCHEMA,
    RUNTIME_LOCK_SCHEMA,
    aggregate_model_sha256,
    assert_backend_matches_lock,
    collect_runtime_lock,
    compare_runtime_snapshot,
    lock_digest,
    model_file_entry,
    validate_models_lock,
    validate_runtime_lock,
    verify_model_files,
)
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
    assert validate_runtime_lock({"schema": RUNTIME_LOCK_SCHEMA, "status": "UNPROVISIONED", "environment": {}, "dependencies": {}}, strict=True)
    assert validate_models_lock({"schema": MODELS_LOCK_SCHEMA, "status": "UNPROVISIONED", "models": [{"model_id": "m", "revision": None, "sha256": None, "language": "de", "sample_rate": 24000, "backend": "b", "backend_id": "b", "backend_version": None, "files": []}]}, strict=True)


def test_not_installed_component_cannot_be_complete():
    runtime = collect_runtime_lock(device="cpu", capabilities={"mfa_fallback": {"enabled": False, "requires": ["mfa"]}})
    runtime["status"] = "COMPLETE"
    runtime["components"]["mfa"] = {"status": "INSTALLED", "version": "not-installed"}
    assert any("mfa" in item for item in validate_runtime_lock(runtime, strict=True, required_capabilities=runtime["capabilities"]))


def test_disabled_optional_mfa_is_valid_but_enabled_missing_ctc_is_blocked():
    disabled = collect_runtime_lock(device="cpu", capabilities={"mfa_fallback": {"enabled": False, "requires": ["mfa"]}})
    assert disabled["status"] == "COMPLETE"
    assert validate_runtime_lock(disabled, strict=True, required_capabilities=disabled["capabilities"]) == []
    enabled = collect_runtime_lock(device="cpu", capabilities={"ctc_alignment": {"enabled": True, "requires": ["whisperx"]}})
    assert enabled["status"] == "UNPROVISIONED"
    assert any("whisperx" in item for item in validate_runtime_lock(enabled, strict=True, required_capabilities=enabled["capabilities"]))


def test_model_bytes_and_aggregate_are_verified_after_freeze():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        original = root / "old" / "model.bin"
        original.parent.mkdir()
        original.write_bytes(b"stable weights")
        row = model_file_entry(original, logical_path="model.bin")
        model = {"model_id": "model", "revision": "rev-1", "sha256": aggregate_model_sha256([row]), "language": "de", "sample_rate": 24000, "backend": "test", "backend_id": "test", "backend_version": "backend-1", "files": [row]}
        lock = {"schema": MODELS_LOCK_SCHEMA, "status": "COMPLETE", "models_root": str(root), "models": [model]}
        assert validate_models_lock(lock, strict=True) == []
        assert verify_model_files(lock, base_dir=root, strict=True) == []
        relocated = root / "model.bin"
        original.replace(relocated)
        assert verify_model_files(lock, base_dir=root, strict=True) == []
        relocated.write_bytes(b"tampered weights")
        assert any("SHA-256 changed" in item for item in verify_model_files(lock, base_dir=root, strict=True))


def test_live_environment_mismatch_is_blocked():
    runtime = collect_runtime_lock(device="cpu", capabilities={"mfa_fallback": {"enabled": False, "requires": ["mfa"]}})
    runtime["status"] = "COMPLETE"
    snapshot = json.loads(json.dumps(runtime))
    snapshot["environment"]["python"] = "0.0.0"
    errors, _warnings = compare_runtime_snapshot(runtime, snapshot)
    assert any("python" in item for item in errors)


def test_loaded_backend_revision_mismatch_is_blocked():
    class Backend:
        model_id = "model"
        backend_id = "test-backend"
        model_revision = "rev-b"
        backend_version = "backend-1"
    lock = {"schema": MODELS_LOCK_SCHEMA, "models": [{"model_id": "model", "revision": "rev-a", "backend_id": "test-backend", "backend_version": "backend-1"}]}
    try:
        assert_backend_matches_lock(Backend(), lock, role="test")
    except ValueError as exc:
        assert "revision" in str(exc)
    else:
        raise AssertionError("a loaded backend with a different revision must be blocked")


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
