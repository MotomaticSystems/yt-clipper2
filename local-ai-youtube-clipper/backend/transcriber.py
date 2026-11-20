"""Local transcription with faster-whisper + transcript building/saving helpers."""
from __future__ import annotations

import gc
import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from backend.models import WHISPER_LANGUAGE, WHISPER_MODEL, get_logger

log = get_logger("transcriber")

ProgressCb = Callable[[float, Optional[str]], None]
_TERMINAL = (".", "!", "?", "…")

_model_lock = threading.Lock()
_model: Any = None
_model_device: str = ""


# --------------------------------------------------------------------------- #
# Sentence / segment building (shared by caption + whisper transcripts)
# --------------------------------------------------------------------------- #
def build_sentences(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group words into sentence-like units used for clip boundary snapping."""
    if not words:
        return []
    punct_ratio = sum(1 for w in words if w["word"].endswith(_TERMINAL)) / len(words)
    punctuated = punct_ratio > 0.02
    gap_break = 1.2 if punctuated else 0.45
    max_words = 45 if punctuated else 28

    sentences: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] = []

    def flush() -> None:
        if cur:
            sentences.append({
                "start": cur[0]["start"],
                "end": cur[-1]["end"],
                "text": " ".join(w["word"] for w in cur),
                "words": list(cur),
            })
            cur.clear()

    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["start"] - w["end"]) if nxt else 99
        too_long = len(cur) >= max_words or (w["end"] - cur[0]["start"]) > 20
        if w["word"].endswith(_TERMINAL) or gap >= gap_break or too_long or nxt is None:
            flush()
    return sentences


def segments_from_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build subtitle-cue sized segments (<= 10 words / 6 s) from words."""
    segs: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] = []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["start"] - w["end"]) if nxt else 99
        if (len(cur) >= 10 or w["word"].endswith(_TERMINAL) or gap >= 0.8
                or (w["end"] - cur[0]["start"]) >= 6 or nxt is None):
            segs.append({"start": cur[0]["start"], "end": cur[-1]["end"],
                         "text": " ".join(x["word"] for x in cur), "words": list(cur)})
            cur = []
    return segs


def transcript_from_words(words: list[dict[str, Any]], source: str, language: str = "") -> dict[str, Any]:
    """Create the canonical transcript dict from a flat word list."""
    segments = segments_from_words(words)
    return {
        "source": source,
        "language": language,
        "duration": words[-1]["end"] if words else 0.0,
        "text": " ".join(w["word"] for w in words),
        "segments": segments,
        "sentences": build_sentences(words),
        "words": words,
    }


# --------------------------------------------------------------------------- #
# Saving: .txt / .srt / .vtt / .json
# --------------------------------------------------------------------------- #
def _ts(seconds: float, sep: str) -> str:
    seconds = max(0.0, seconds)
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def save_transcript(transcript: dict[str, Any], base: Path) -> dict[str, str]:
    """Write transcript.txt/.srt/.vtt/.json next to `base` (without extension)."""
    segs = transcript.get("segments", [])
    paths = {ext: base.with_suffix(f".{ext}") for ext in ("txt", "srt", "vtt", "json")}
    try:
        with open(paths["txt"], "w", encoding="utf-8") as f:
            for s in segs:
                f.write(f"[{_ts(s['start'], '.')[:8]}] {s['text'].strip()}\n")
        with open(paths["srt"], "w", encoding="utf-8") as f:
            for i, s in enumerate(segs, 1):
                f.write(f"{i}\n{_ts(s['start'], ',')} --> {_ts(s['end'], ',')}\n{s['text'].strip()}\n\n")
        with open(paths["vtt"], "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for s in segs:
                f.write(f"{_ts(s['start'], '.')} --> {_ts(s['end'], '.')}\n{s['text'].strip()}\n\n")
        with open(paths["json"], "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in transcript.items() if k != "sentences"}, f, ensure_ascii=False)
        log.info("transcript saved", extra={"base": str(base)})
    except OSError as exc:
        log.error(f"could not save transcript: {exc}", exc_info=True)
    return {k: str(v) for k, v in paths.items()}


# --------------------------------------------------------------------------- #
# faster-whisper
# --------------------------------------------------------------------------- #
def _cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _load_model(force_cpu: bool = False) -> tuple[Any, str]:
    """Load (and cache) the Whisper model. GPU float16 -> CPU int8 fallback."""
    global _model, _model_device
    with _model_lock:
        if _model is not None and not (force_cpu and _model_device != "cpu"):
            return _model, _model_device
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("faster-whisper is not installed. Run: pip install -r requirements.txt") from exc

        _model = None
        gc.collect()
        if not force_cpu and _cuda_available():
            size = WHISPER_MODEL or "small"
            try:
                log.info(f"loading faster-whisper '{size}' on GPU (float16)")
                _model = WhisperModel(size, device="cuda", compute_type="float16")
                _model_device = "cuda"
                return _model, _model_device
            except Exception as exc:
                log.warning(f"GPU model load failed ({exc}); falling back to CPU int8")
        size = WHISPER_MODEL or "base"
        log.info(f"loading faster-whisper '{size}' on CPU (int8)")
        _model = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
        _model_device = "cpu"
        return _model, _model_device


def unload_model() -> None:
    """Free the Whisper model (and its GPU memory)."""
    global _model, _model_device
    with _model_lock:
        _model = None
        _model_device = ""
    gc.collect()
    log.info("whisper model unloaded")


def _extract_audio(video_path: Path, wav_path: Path) -> float:
    """Extract 16 kHz mono WAV with ffmpeg. Returns duration in seconds."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed or not on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Audio extraction timed out.") from exc
    if res.returncode != 0 or not wav_path.exists():
        raise RuntimeError(f"Audio extraction failed: {res.stderr[-300:]}")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nw=1:nk=1", str(wav_path)], capture_output=True, text=True)
    try:
        return float(probe.stdout.strip())
    except ValueError:
        return 0.0


def _run_whisper(model: Any, device: str, wav: Path, duration: float, progress: ProgressCb) -> tuple[list[dict], str]:
    seg_iter, info = model.transcribe(
        str(wav),
        language=WHISPER_LANGUAGE,
        beam_size=5 if device == "cuda" else 1,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    language = getattr(info, "language", "") or ""
    segments: list[dict[str, Any]] = []
    for seg in seg_iter:
        words = [{"start": float(w.start), "end": float(w.end), "word": w.word.strip()}
                 for w in (seg.words or []) if w.word and w.word.strip()]
        segments.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip(), "words": words})
        if duration > 0:
            progress(min(seg.end / duration, 0.99), f"Transcribing with Whisper ({device})… {seg.end / duration * 100:.0f}%")
    return segments, language


def transcribe(video_path: Path, job_dir: Path, progress: ProgressCb) -> dict[str, Any]:
    """Transcribe a video file locally. Returns the canonical transcript dict."""
    wav = job_dir / "audio.wav"
    try:
        progress(0.0, "Extracting audio…")
        duration = _extract_audio(video_path, wav)
        progress(0.02, "Loading Whisper model…")
        model, device = _load_model()
        try:
            segments, language = _run_whisper(model, device, wav, duration, progress)
        except Exception as exc:
            if device != "cuda":
                raise
            log.warning(f"GPU transcription failed ({exc}); retrying on CPU", exc_info=True)
            progress(0.02, "GPU failed - retrying on CPU…")
            model, device = _load_model(force_cpu=True)
            segments, language = _run_whisper(model, device, wav, duration, progress)

        words = [w for s in segments for w in s["words"]]
        if len(words) < 10:
            raise RuntimeError("No speech detected in this video (transcript is empty).")
        sentences = build_sentences(words)
        transcript = {
            "source": f"faster-whisper ({device})",
            "language": language,
            "duration": duration or words[-1]["end"],
            "text": " ".join(s["text"] for s in segments),
            "segments": segments,
            "sentences": sentences,
            "words": words,
        }
        progress(1.0, "Transcription complete")
        log.info("transcription done", extra={"words": len(words), "device": device, "language": language})
        return transcript
    except Exception as exc:
        log.error(f"transcription failed: {exc}", exc_info=True)
        raise
    finally:
        try:
            if wav.exists():
                wav.unlink()
        except OSError:
            pass
