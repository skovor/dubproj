"""Serializable scene/line/candidate contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReferenceSegment:
    path: str
    start: float = 0.0
    end: float | None = None
    text: str = ""
    channel: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReferenceSegment":
        return cls(
            path=str(value["path"]), start=float(value.get("start", 0.0)),
            end=float(value["end"]) if value.get("end") is not None else None,
            text=str(value.get("text", "")),
            channel=int(value["channel"]) if value.get("channel") is not None else None,
        )


@dataclass
class Line:
    id: str
    speaker: str
    source_text: str
    target_text: str
    start: float = 0.0
    end: float = 0.0
    topology: str = "LINE_SEPARATED"
    source_audio: str | None = None
    reference_audio: str | None = None
    reference_segments: list[ReferenceSegment] = field(default_factory=list)
    subtitle_authorized: bool = False
    movie_identity_verified: bool = False
    card_identity_verified: bool = False
    card_timebase_verified: bool = False
    force_keep_original: bool = False
    preserve_reason: str | None = None
    synthesis_text_override: str | None = None
    delivery_text: str | None = None
    speech_start: float | None = None
    speech_end: float | None = None
    preserved_source_intervals: list[dict[str, Any]] = field(default_factory=list)
    source_resume: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any], topology: str | None = None) -> "Line":
        known = {
            "id", "speaker", "source_text", "target_text", "start", "end",
            "topology", "source_audio", "reference_audio", "reference_segments",
            "subtitle_authorized", "movie_identity_verified", "card_identity_verified",
            "card_timebase_verified", "force_keep_original", "preserve_reason",
            "synthesis_text_override", "delivery_text", "speech_start", "speech_end",
            "preserved_source_intervals", "source_resume", "metadata",
        }
        data = {key: value[key] for key in known if key in value}
        data["id"] = str(data["id"])
        data["speaker"] = str(data.get("speaker", ""))
        data["source_text"] = str(data.get("source_text", ""))
        data["target_text"] = str(data.get("target_text", ""))
        data["reference_segments"] = [ReferenceSegment.from_dict(item) for item in data.get("reference_segments", [])]
        if topology:
            data["topology"] = topology
        data["metadata"] = {key: item for key, item in value.items() if key not in known}
        metadata = dict(value.get("metadata") or {})
        metadata.update({key: item for key, item in value.items() if key not in known})
        data["metadata"] = metadata
        return cls(**data)

    @property
    def window(self) -> tuple[float, float]:
        return float(self.start), float(self.end)

    @property
    def effective_target_text(self) -> str:
        return self.delivery_text or self.target_text

    @property
    def reference_text(self) -> str:
        """The transcript that describes ref_audio: always source language."""
        # A reference segment is the physical source of truth.  Falling back
        # to source_text is safe only for a full-file reference with no
        # segment-level transcript.  This prevents ref_audio/ref_text drift
        # when a line is cut from a longer English stem.
        texts = [segment.text.strip() for segment in self.reference_segments if segment.text.strip()]
        return " ".join(texts) if texts else self.source_text

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reference_segments"] = [asdict(item) for item in self.reference_segments]
        return value


@dataclass
class Scene:
    id: str
    topology: str
    lines: list[Line]
    source_stem: str | None = None
    dialogue_channel: int = 0
    movie_identity_verified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Scene":
        topology = str(value.get("topology", "LINE_SEPARATED"))
        lines = [Line.from_dict(item, topology=topology) for item in value.get("lines", [])]
        if topology == "EMBEDDED_FMV" and value.get("movie_identity_verified"):
            for line in lines:
                line.movie_identity_verified = True
        known = {"id", "topology", "lines", "source_stem", "dialogue_channel", "movie_identity_verified", "metadata"}
        metadata = dict(value.get("metadata") or {})
        metadata.update({key: item for key, item in value.items() if key not in known})
        return cls(
            id=str(value["id"]), topology=topology, lines=lines,
            source_stem=value.get("source_stem"),
            dialogue_channel=int(value.get("dialogue_channel", 0)),
            movie_identity_verified=bool(value.get("movie_identity_verified", False)),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "topology": self.topology,
            "source_stem": self.source_stem,
            "dialogue_channel": self.dialogue_channel,
            "movie_identity_verified": self.movie_identity_verified,
            "lines": [line.to_dict() for line in self.lines],
            **self.metadata,
        }


@dataclass
class Candidate:
    line_id: str
    path: str
    round_index: int
    take_index: int
    synthesis_text: str
    generation_hash: str
    processing_hash: str | None = None
    qa_hash: str | None = None
    passed: bool = False
    hard_gates: dict[str, bool] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    failure_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
