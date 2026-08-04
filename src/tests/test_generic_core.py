from pathlib import Path
from tempfile import TemporaryDirectory

from dubbing_pipeline.config import PipelineConfig
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
