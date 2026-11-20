"""Shared configuration, JSONL logging, URL validation and pydantic models."""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Paths & configuration (all overridable with environment variables)
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", ROOT_DIR / "outputs")).resolve()
LOGS_DIR = Path(os.environ.get("LOGS_DIR", ROOT_DIR / "logs")).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT_DIR / "data")).resolve()
FRONTEND_DIR = ROOT_DIR / "frontend"
COOKIES_PATH = DATA_DIR / "cookies.txt"

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "").strip()  # empty -> auto-detect from /v1/models
LLM_DISABLED = os.environ.get("DISABLE_LLM", "0") == "1"
LLM_CHUNK_CHARS = int(os.environ.get("LLM_CHUNK_CHARS", "8000"))
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "240"))

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "").strip()  # empty -> small (GPU) / base (CPU)
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "").strip() or None
SUBTITLE_FONT = os.environ.get("SUBTITLE_FONT", "DejaVu Sans")
KEEP_SOURCE = os.environ.get("KEEP_SOURCE", "0") == "1"
MAX_VIDEO_SECONDS = int(os.environ.get("MAX_VIDEO_SECONDS", str(4 * 3600)))

for _d in (OUTPUTS_DIR, LOGS_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# JSONL logging -> logs/app.jsonl (+ human readable console)
# --------------------------------------------------------------------------- #
_STD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {"message", "asctime"}


class JsonlFormatter(logging.Formatter):
    """Format every log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STD_ATTRS and not key.startswith("_"):
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


_LOGGING_READY = False


def _setup_logging() -> None:
    global _LOGGING_READY
    if _LOGGING_READY:
        return
    root = logging.getLogger("studio")
    root.setLevel(logging.INFO)
    root.propagate = False
    try:
        fh = RotatingFileHandler(LOGS_DIR / "app.jsonl", maxBytes=10_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(JsonlFormatter())
        root.addHandler(fh)
    except OSError as exc:  # never fail because of logging
        print(f"[warn] cannot open log file: {exc}", file=sys.stderr)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"))
    root.addHandler(sh)
    _LOGGING_READY = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger that writes JSONL to logs/app.jsonl."""
    _setup_logging()
    return logging.getLogger(f"studio.{name}")


log = get_logger("models")

# --------------------------------------------------------------------------- #
# YouTube URL validation (YouTube ONLY)
# --------------------------------------------------------------------------- #
_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be"}
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def normalize_youtube_url(url: str) -> str:
    """Validate a YouTube video URL and return its canonical form.

    Raises ValueError for anything that is not a single YouTube video.
    """
    raw = (url or "").strip()
    if not raw:
        raise ValueError("Please paste a YouTube URL.")
    if "://" not in raw:
        raw = "https://" + raw
    try:
        parsed = urlparse(raw)
    except Exception as exc:  # urlparse is lenient, but be safe
        raise ValueError(f"Invalid URL: {exc}") from exc
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or host not in _YT_HOSTS:
        raise ValueError("Only YouTube links are supported (youtube.com / youtu.be).")

    video_id: Optional[str] = None
    parts = [p for p in parsed.path.split("/") if p]
    if host.endswith("youtu.be"):
        video_id = parts[0] if parts else None
    elif parsed.path == "/watch":
        video_id = (parse_qs(parsed.query).get("v") or [None])[0]
    elif len(parts) >= 2 and parts[0] in ("shorts", "live", "embed", "v"):
        video_id = parts[1]

    if not video_id or not _VIDEO_ID.match(video_id):
        raise ValueError("That does not look like a single YouTube video link (playlists/channels are not supported).")
    return f"https://www.youtube.com/watch?v={video_id}"


# --------------------------------------------------------------------------- #
# API models
# --------------------------------------------------------------------------- #
SubtitleStyle = Literal["hormozi", "clean", "neon"]


class JobCreate(BaseModel):
    youtube_url: str
    num_clips: int = Field(default=5, ge=1, le=15)
    min_duration: int = Field(default=30, ge=10, le=120)
    max_duration: int = Field(default=90, ge=15, le=180)
    subtitle_style: SubtitleStyle = "hormozi"

    @field_validator("youtube_url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        return normalize_youtube_url(value)

    @model_validator(mode="after")
    def _check_durations(self) -> "JobCreate":
        if self.max_duration < self.min_duration + 5:
            raise ValueError("max_duration must be at least 5 seconds larger than min_duration.")
        return self


class Word(BaseModel):
    start: float
    end: float
    word: str


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: list[Word] = []


class ClipInfo(BaseModel):
    id: str
    index: int
    title: str
    description: str = ""
    hashtags: list[str] = []
    score: float
    reason: str = ""
    start: float
    end: float
    duration: float
    source: str = "heuristic"  # "llm" or "heuristic"
    preview_url: str = ""
    download_url: str = ""
    thumbnail_url: str = ""


class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "running", "completed", "failed"]
    stage: str
    progress: float
    message: str = ""
    error: Optional[str] = None
    youtube_url: str
    video_title: str = ""
    created_at: float
    updated_at: float
    transcript_source: str = ""
    analysis_method: str = ""
    clips_count: int = 0
    params: dict[str, Any] = {}
