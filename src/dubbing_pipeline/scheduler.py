"""Global cohort scheduler: generation is sealed before heavy QA/retries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .contracts import FailureClass
from .telemetry import TelemetryCollector


PHASES = ("PREFLIGHT", "PLAN", "GENERATE_INITIAL_COHORT", "SEAL_GENERATION_MANIFEST", "RELEASE_OMNIVOICE", "QA_INITIAL_COHORT", "RETRY_PLAN", "GENERATE_RETRY_COHORT", "QA_RETRY_COHORT", "SELECT_WINNERS", "MOUNT_SCENES", "SERIALIZATION_AUDIT", "CONTINUOUS_AUDIT", "PACKAGE", "PACKAGE_ROUNDTRIP", "DEPLOY_TRANSACTION", "RUNTIME_SMOKE")


@dataclass
class CohortReport:
    run_id: str
    phases: list[str]
    candidates: dict[str, list[Any]]
    evaluations: dict[str, list[Any]]
    retry_ids: list[str]
    blockers: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "phases": self.phases, "candidates": self.candidates, "evaluations": self.evaluations, "retry_ids": self.retry_ids, "blockers": self.blockers}


def run_cohorts(items: Iterable[Any], *, item_id: Callable[[Any], str], generate: Callable[[list[Any], int], Mapping[str, list[Any]]], evaluate: Callable[[Any], Any], max_retry_rounds: int = 1, telemetry: TelemetryCollector | None = None) -> CohortReport:
    collector = telemetry or TelemetryCollector()
    values = list(items); phases: list[str] = ["PREFLIGHT", "PLAN"]
    candidates: dict[str, list[Any]] = {}; evaluations: dict[str, list[Any]] = {}; blockers: list[dict[str, Any]] = []
    with collector.stage("GENERATE_INITIAL_COHORT", count=len(values)):
        candidates.update({key: list(value) for key, value in generate(values, 1).items()})
    phases.extend(["GENERATE_INITIAL_COHORT", "SEAL_GENERATION_MANIFEST", "RELEASE_OMNIVOICE", "QA_INITIAL_COHORT"])
    with collector.stage("QA_INITIAL_COHORT", count=sum(len(value) for value in candidates.values())):
        for key, rows in candidates.items():
            evaluations[key] = [evaluate(candidate) for candidate in rows]
    for round_index in range(2, max_retry_rounds + 2):
        retry_ids = []
        for key, results in evaluations.items():
            if any((getattr(result, "failure_class", None) == FailureClass.STOCHASTIC_TTS or getattr(result, "failure_class", None) == FailureClass.STOCHASTIC_TTS.value) for result in results) and not any(getattr(result, "passed", False) for result in results):
                retry_ids.append(key)
        phases.append("RETRY_PLAN")
        if not retry_ids:
            break
        retry_items = [item for item in values if item_id(item) in retry_ids]
        with collector.stage("GENERATE_RETRY_COHORT", round_index=round_index, count=len(retry_items)):
            extra = generate(retry_items, round_index)
        phases.extend(["GENERATE_RETRY_COHORT", "RELEASE_OMNIVOICE", "QA_RETRY_COHORT"])
        for key, rows in extra.items():
            candidates.setdefault(key, []).extend(rows)
            evaluations.setdefault(key, []).extend(evaluate(candidate) for candidate in rows)
    unresolved = [key for key, results in evaluations.items() if not any(getattr(result, "passed", False) for result in results)]
    # ASR_UNCERTAIN is deliberately a hold for selective alignment, not a
    # stochastic TTS retry.  Keep it visible as a blocker while ensuring the
    # scheduler cannot spend another OmniVoice round on the same evidence.
    retry_ids = []
    for key in unresolved:
        results = evaluations.get(key, [])
        failures = {getattr(result, "failure_class", None) for result in results}
        if FailureClass.STOCHASTIC_TTS in failures or FailureClass.STOCHASTIC_TTS.value in failures:
            retry_ids.append(key)
        elif FailureClass.ASR_UNCERTAIN in failures or FailureClass.ASR_UNCERTAIN.value in failures:
            blockers.append({"line_id": key, "reason": "ASR_UNCERTAIN_HOLD"})
        else:
            blockers.append({"line_id": key, "reason": "NO_PASSING_CANDIDATE"})
    blockers.extend({"line_id": key, "reason": "NO_PASSING_CANDIDATE"} for key in retry_ids)
    # Selection, processing, mounting, serialization and deployment are owned
    # by the caller.  Do not advertise those phases merely because they exist
    # in the global phase vocabulary; a report must contain executed evidence.
    return CohortReport(collector.run_id, phases, candidates, evaluations, retry_ids, blockers)


__all__ = ["CohortReport", "PHASES", "run_cohorts"]
