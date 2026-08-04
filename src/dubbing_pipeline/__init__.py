"""Portable, contract-first audio dubbing pipeline."""

from .config import PipelineConfig
from .models import Candidate, Line, Scene
from .policy import Decision, classify_line
from .orchestration import run_scene
from .contracts import AudioArtifact, CandidateArtifact, DeliveryWindow, GateEvidence, GateStatus, ReferenceEvidence
from .contracts import EvidenceFamily, EvidenceRecord
from .qa_v2 import LanguageProfile, LinguisticDecision, LinguisticStatus, QAResultV2, apply_independent_evidence, decide_linguistic_evidence, evaluate_candidate_v2, select_passed_v2
from .asr import ASRCache, ASRReading, DualASREvidence, FasterWhisperBackend, WhisperXEscalationRequest, prepare_whisperx_escalation, transcribe_dual
from .alignment import AlignmentCache, AlignmentUnavailable, ContrastiveAlignment, MFAAlignerAdapter, SpeechBrainVoxLingua107, WhisperXCTCAligner, contrastive_align, language_id_evidence
from .generation_v2 import GenerationRequest, GenerationRuntimeV2, generate_candidates_v2, generate_cohort_v2
from .montage import mount_surgical
from .orchestration_v2 import run_scene_v2
from .post_qa import POST_TRANSFORM_STAGES, StageAudit, audit_candidate_stage, audit_scene_stage, persist_audio_atomic
from .runtime_lock import RuntimeLockError, assert_backend_matches_lock, assert_reproducible, reproducibility_report, verify_model_files

__all__ = ["ASRCache", "ASRReading", "AlignmentCache", "AlignmentUnavailable", "AudioArtifact", "Candidate", "CandidateArtifact", "ContrastiveAlignment", "Decision", "DeliveryWindow", "DualASREvidence", "EvidenceFamily", "EvidenceRecord", "FasterWhisperBackend", "GenerationRequest", "GenerationRuntimeV2", "GateEvidence", "GateStatus", "LanguageProfile", "Line", "LinguisticDecision", "LinguisticStatus", "MFAAlignerAdapter", "PipelineConfig", "POST_TRANSFORM_STAGES", "QAResultV2", "ReferenceEvidence", "RuntimeLockError", "Scene", "SpeechBrainVoxLingua107", "StageAudit", "WhisperXCTCAligner", "WhisperXEscalationRequest", "apply_independent_evidence", "assert_backend_matches_lock", "assert_reproducible", "audit_candidate_stage", "audit_scene_stage", "classify_line", "contrastive_align", "decide_linguistic_evidence", "evaluate_candidate_v2", "generate_candidates_v2", "generate_cohort_v2", "language_id_evidence", "mount_surgical", "persist_audio_atomic", "prepare_whisperx_escalation", "reproducibility_report", "run_scene", "run_scene_v2", "select_passed_v2", "transcribe_dual", "verify_model_files"]
