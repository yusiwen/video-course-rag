"""Where a run's files live.

Every artifact of a run belongs to exactly one directory: ``<runs>/<run-id>/``.
That is a deliberate choice, not a convenience: it means the recorded material —
transcripts, frames, embeddings, delivered package — can be removed, backed up or
kept out of version control by one path, and that a public checkout never carries
someone else's lecture.

    runs/<id>/
      source.json        video path, fingerprint, duration
      audio.wav          mono 16 kHz
      asr.json           raw recogniser output
      asr_sent.json      sentences with absolute milliseconds
      subs.srt           subtitles
      states.json        visual units (onset, mode, title, ocr_lines)
      activity.json      per-unit screen activity
      calib_ocr.json     OCR cache used by calibration
      thumbs/            one PNG per unit
      profile.json       measured title geometry (calibrate.py)
      chapters.json      unit-derived chapter table
      chapters_llm.json  transcript-derived chapter table (needs an LLM)
      chapters_fused.json  the delivered table
      chunks.jsonl       retrieval units
      glossary.json      per-run term corrections
      overrides.json     hand-written title overrides
      index/             meta.json, vectors.f32, chapters.json, subs.vtt
      eval/questions.jsonl  this run's own evaluation questions
      report/            calibration.md, title_audit.md, run.json
      page/index.html    the deliverable for this run alone
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ENV_RUNS = "VIDEO_COURSE_RAG_RUNS"


def runs_root() -> Path:
    """Root holding every run; ``--runs-root`` or ``$VIDEO_COURSE_RAG_RUNS`` overrides."""
    return Path(os.environ.get(ENV_RUNS, "runs")).expanduser()


@dataclass
class Run:
    id: str
    root: Path = field(init=False)

    def __post_init__(self):
        self.root = runs_root() / self.id

    # --- construction -------------------------------------------------------
    @classmethod
    def get(cls, run_id: str) -> "Run":
        return cls(run_id)

    @classmethod
    def existing(cls, path_or_id: str | os.PathLike) -> "Run":
        """Accept either a run id or a directory (the CLI allows both)."""
        p = Path(path_or_id).expanduser()
        if p.is_dir():
            return cls(p.name, )
        return cls(str(path_or_id))

    def ensure(self) -> "Run":
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    # --- files --------------------------------------------------------------
    def p(self, name: str) -> Path:
        return self.root / name

    @property
    def source(self) -> Path: return self.p("source.json")
    @property
    def audio(self) -> Path: return self.p("audio.wav")
    @property
    def asr(self) -> Path: return self.p("asr.json")
    @property
    def asr_raw(self) -> Path: return self.p("asr_raw.json")
    @property
    def asr_sent(self) -> Path: return self.p("asr_sent.json")
    @property
    def subs(self) -> Path: return self.p("subs.srt")
    @property
    def states(self) -> Path: return self.p("states.json")
    @property
    def activity(self) -> Path: return self.p("activity.json")
    @property
    def calib_ocr(self) -> Path: return self.p("calib_ocr.json")
    @property
    def thumbs(self) -> Path: return self.p("thumbs")
    @property
    def profile(self) -> Path: return self.p("profile.json")
    @property
    def chapters(self) -> Path: return self.p("chapters.json")
    @property
    def chapters_llm(self) -> Path: return self.p("chapters_llm.json")
    @property
    def chapters_fused(self) -> Path: return self.p("chapters_fused.json")
    @property
    def chunks(self) -> Path: return self.p("chunks.jsonl")
    @property
    def glossary(self) -> Path: return self.p("glossary.json")
    @property
    def overrides(self) -> Path: return self.p("overrides.json")
    @property
    def index_dir(self) -> Path: return self.p("index")
    @property
    def questions(self) -> Path: return self.p("eval") / "questions.jsonl"
    @property
    def report_dir(self) -> Path: return self.p("report")
    @property
    def page(self) -> Path: return self.p("page") / "index.html"

    # --- json helpers -------------------------------------------------------
    def read_json(self, path: Path, default=None):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default

    def write_json(self, path: Path, obj) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def all_runs() -> list[Run]:
    """Runs that exist, newest name last (the CLI's default when --run is absent)."""
    root = runs_root()
    if not root.is_dir():
        return []
    ids = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    return [Run(i) for i in ids]


def run(x) -> Run:
    """Coerce a run id, Run or path into a Run."""
    return x if isinstance(x, Run) else Run.get(str(x))


def mkdirs(path: Path) -> Path:
    """Make the parent directory of *path* and return the path (for open())."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
