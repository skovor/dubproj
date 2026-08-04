"""Portable, contract-first audio dubbing pipeline."""

from .config import PipelineConfig
from .models import Candidate, Line, Scene
from .policy import Decision, classify_line
from .orchestration import run_scene
from .contracts import AudioArtifact, CandidateArtifact, DeliveryWindow, GateEvidence, GateStatus, ReferenceEvidence
from .qa_v2 import LanguageProfile, QAResultV2, evaluate_candidate_v2, select_passed_v2
from .generation_v2 import GenerationRequest, GenerationRuntimeV2, generate_candidates_v2, generate_cohort_v2
from .montage import mount_surgical
from .orchestration_v2 import run_scene_v2
from .post_qa import POST_TRANSFORM_STAGES, StageAudit, audit_candidate_stage, audit_scene_stage, persist_audio_atomic

__all__ = ["AudioArtifact", "Candidate", "CandidateArtifact", "Decision", "DeliveryWindow", "GenerationRequest", "GenerationRuntimeV2", "GateEvidence", "GateStatus", "LanguageProfile", "Line", "PipelineConfig", "POST_TRANSFORM_STAGES", "QAResultV2", "ReferenceEvidence", "Scene", "StageAudit", "audit_candidate_stage", "audit_scene_stage", "classify_line", "evaluate_candidate_v2", "generate_candidates_v2", "generate_cohort_v2", "mount_surgical", "persist_audio_atomic", "run_scene", "run_scene_v2", "select_passed_v2"]
