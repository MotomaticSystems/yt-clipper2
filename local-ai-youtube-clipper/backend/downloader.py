"""YouTube download (yt-dlp only) + YouTube caption extraction with word timings."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

from backend.models import MAX_VIDEO_SECONDS, get_logger

log = get_logger("downloader")

ProgressCb = Callable[[float, Optional[str]], None]

# Best quality up to 1080p; prefer H.264/AAC (fast to decode), fall back gracefully.
VIDEO_FORMAT = "bv*[height<=1080]+ba/b[height<=1080]/best[height<=1080]/best"
FORMAT_SORT = ["res:1080", "vcodec:h264", "acodec:aac"]
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}


class DownloadError(RuntimeError):
    """Raised with a user-friendly message when the download cannot continue."""


def _friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "sign in to confirm" in low or "not a bot" in low or "cookies" in low and "sign in" in low:
        return ("YouTube is blocking this server (bot check). Upload a fresh cookies.txt in Settings "
                "(export it from a logged-in browser while on youtube.com) and try again.")
    if "private video" in low:
        return "This video is private. Upload cookies.txt from an account that can watch it."
    if "video unavailable" in low or "this video is not available" in low:
        return "This video is unavailable (removed, region-locked or private)."
    if "age" in low and "confirm" in low:
        return "This video is age-restricted. Upload cookies.txt from a logged-in account."
    if "live event" in low or "is live" in low:
        return "Live streams cannot be clipped until they have finished."
    if "http error 429" in low:
        return "YouTube rate-limited this IP (HTTP 429). Upload cookies.txt or wait a few minutes."
    if "requested format is not available" in low:
        return "No downloadable format found. Update yt-dlp (pip install -U yt-dlp) and/or upload cookies.txt."
    return f"Download failed: {msg[-400:]}"


def _find_video_file(work: Path) -> Path:
    candidates = [p for p in work.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS and p.stem.startswith("video")]
    if not candidates:
        raise DownloadError("Download finished but no video file was produced.")
    return max(candidates, key=lambda p: p.stat().st_size)


def pick_caption_lang(info: dict[str, Any]) -> Optional[tuple[str, bool]]:
    """Choose the best caption track in the video's *spoken* language.

    Returns (language_code, is_auto_generated) or None when nothing suitable exists.
    """
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    spoken = (info.get("language") or "").split("-")[0].lower()

    def has_json3_or_any(tracks: Any) -> bool:
        return bool(tracks)

    langs = [spoken] if spoken else ["en"]
    for lang in langs:
        for key in manual:
            if key.split("-")[0].lower() == lang and has_json3_or_any(manual[key]):
                return key, False
    for lang in langs:
        if f"{lang}-orig" in auto:
            return f"{lang}-orig", True
    # automatic track flagged as the original spoken language
    for key in auto:
        if key.endswith("-orig"):
            return key, True
    for lang in langs:
        if lang in auto:
            return lang, True
    return None


def parse_json3(path: Path) -> list[dict[str, Any]]:
    """Parse a YouTube json3 caption file into word dicts {start,end,word} (seconds)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    words: list[dict[str, Any]] = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        t0 = ev.get("tStartMs")
        if not segs or t0 is None:
            continue
        ev_end = t0 + int(ev.get("dDurationMs", 0) or 0)
        starts = [t0 + int(s.get("tOffsetMs", 0) or 0) for s in segs]
        for i, seg in enumerate(segs):
            text = (seg.get("utf8") or "").replace("\n", " ").strip()
            if not text or (text.startswith("[") and text.endswith("]")):
                continue  # skip [Music], [Applause] ...
            s_start = starts[i]
            s_end = starts[i + 1] if i + 1 < len(segs) else ev_end
            if s_end <= s_start:
                s_end = s_start + 300
            tokens = text.split()
            step = (s_end - s_start) / len(tokens)
            for j, tok in enumerate(tokens):
                words.append({
                    "start": (s_start + j * step) / 1000.0,
                    "end": (s_start + (j + 1) * step) / 1000.0,
                    "word": tok,
                })
    words.sort(key=lambda w: w["start"])

    cleaned: list[dict[str, Any]] = []
    for w in words:
        if cleaned and w["word"] == cleaned[-1]["word"] and abs(w["start"] - cleaned[-1]["start"]) < 0.05:
            continue  # duplicate caused by overlapping caption events
        cleaned.append(w)
    for i, w in enumerate(cleaned):
        nxt = cleaned[i + 1]["start"] if i + 1 < len(cleaned) else w["end"]
        end = min(w["end"], nxt if nxt > w["start"] else w["end"], w["start"] + 1.5)
        w["end"] = max(end, w["start"] + 0.05)
    return cleaned


def _fetch_captions(ydl_cls: Any, base_opts: dict[str, Any], url: str, info: dict[str, Any], work: Path) -> Optional[list[dict[str, Any]]]:
    """Try to download YouTube captions. Never raises - returns None on any problem."""
    try:
        pick = pick_caption_lang(info)
        if not pick:
            log.info("no usable YouTube captions for this video")
            return None
        lang, is_auto = pick
        log.info("downloading captions", extra={"lang": lang, "auto": is_auto})
        opts = {
            **base_opts,
            "skip_download": True,
            "writesubtitles": not is_auto,
            "writeautomaticsub": is_auto,
            "subtitleslangs": [lang],
            "subtitlesformat": "json3",
            "outtmpl": str(work / "subs.%(ext)s"),
        }
        with ydl_cls(opts) as ydl:
            ydl.download([url])
        files = sorted(work.glob("subs.*.json3"))
        if not files:
            log.warning("caption download produced no file")
            return None
        words = parse_json3(files[0])
        log.info("parsed captions", extra={"words": len(words)})
        return words if len(words) >= 30 else None
    except Exception as exc:  # captions are an optimisation, Whisper is the fallback
        log.warning(f"caption fetch failed, will use Whisper: {exc}")
        return None


def download_video(url: str, job_dir: Path, cookies_path: Path, progress: ProgressCb) -> dict[str, Any]:
    """Download the video (<=1080p) and, when available, YouTube captions.

    Returns a dict: video_path, title, duration, channel, video_id, caption_words (or None).
    """
    try:
        import yt_dlp  # imported lazily so the API can start even if it is missing
    except ImportError as exc:
        raise DownloadError("yt-dlp is not installed. Run: pip install -r requirements.txt") from exc

    work = job_dir / "source"
    work.mkdir(parents=True, exist_ok=True)

    cookie_copy: Optional[Path] = None
    if cookies_path.exists() and cookies_path.stat().st_size > 0:
        cookie_copy = job_dir / ".cookies.txt"  # yt-dlp rewrites the file, so never hand it the original
        shutil.copyfile(cookies_path, cookie_copy)
        log.info("using uploaded cookies.txt")

    base_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 30,
        "ignoreerrors": False,
    }
    if cookie_copy:
        base_opts["cookiefile"] = str(cookie_copy)

    state = {"files": [], "last": None}

    def hook(d: dict[str, Any]) -> None:
        try:
            if d.get("status") == "downloading":
                name = d.get("filename")
                if name != state["last"]:
                    state["last"] = name
                    state["files"].append(name)
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                frac = (d.get("downloaded_bytes", 0) / total) if total else 0.0
                idx = min(len(state["files"]), 2)
                overall = ((idx - 1) + frac) / 2 if idx else frac
                speed = d.get("speed") or 0
                progress(min(overall, 0.97), f"Downloading video… {overall * 100:.0f}% ({speed / 1e6:.1f} MB/s)")
        except Exception:  # progress reporting must never break the download
            pass

    try:
        progress(0.0, "Reading video metadata…")
        with yt_dlp.YoutubeDL(base_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        if not info:
            raise DownloadError("Could not read video metadata.")
        if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
            raise DownloadError("Live streams cannot be clipped until they have finished.")
        duration = float(info.get("duration") or 0)
        if duration and duration > MAX_VIDEO_SECONDS:
            raise DownloadError(f"Video is too long ({duration / 3600:.1f} h). Limit is {MAX_VIDEO_SECONDS / 3600:.1f} h.")
        title = info.get("title") or "video"
        log.info("metadata ok", extra={"title": title, "duration": duration})

        dl_opts = {
            **base_opts,
            "format": VIDEO_FORMAT,
            "format_sort": FORMAT_SORT,
            "merge_output_format": "mp4",
            "outtmpl": str(work / "video.%(ext)s"),
            "progress_hooks": [hook],
            "concurrent_fragment_downloads": 4,
        }
        progress(0.01, "Downloading video…")
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            ydl.download([url])
        video_path = _find_video_file(work)
        log.info("video downloaded", extra={"path": str(video_path), "mb": round(video_path.stat().st_size / 1e6, 1)})

        progress(0.97, "Checking YouTube captions…")
        caption_words = _fetch_captions(yt_dlp.YoutubeDL, base_opts, url, info, work)
        progress(1.0, "Download complete")
        return {
            "video_path": video_path,
            "title": title,
            "duration": duration,
            "channel": info.get("channel") or info.get("uploader") or "",
            "video_id": info.get("id") or "",
            "caption_words": caption_words,
        }
    except DownloadError:
        raise
    except Exception as exc:
        log.error(f"download failed: {exc}", exc_info=True)
        raise DownloadError(_friendly_error(exc)) from exc
    finally:
        if cookie_copy and cookie_copy.exists():
            try:
                cookie_copy.unlink()
            except OSError:
                pass
