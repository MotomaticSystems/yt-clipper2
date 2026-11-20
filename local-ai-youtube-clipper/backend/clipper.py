"""FFmpeg-only vertical clip rendering: 1080x1920, blurred background, karaoke subtitles, loudnorm."""
from __future__ import annotations

import json
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from backend.models import SUBTITLE_FONT, get_logger

log = get_logger("clipper")
ProgressCb = Callable[[float, Optional[str]], None]

OUT_W, OUT_H, FPS = 1080, 1920, 30

# name -> (font size, outline, margin_v, words per caption, uppercase, highlight colour (BGR) or None)
STYLES: dict[str, dict[str, Any]] = {
    "hormozi": {"size": 86, "outline": 8, "margin_v": 540, "max_words": 3, "upper": True, "hl": "00FFFF"},   # yellow
    "neon": {"size": 86, "outline": 8, "margin_v": 540, "max_words": 3, "upper": True, "hl": "00FF66"},      # green
    "clean": {"size": 70, "outline": 5, "margin_v": 460, "max_words": 5, "upper": False, "hl": None},
}


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #
def probe_video(path: Path) -> dict[str, Any]:
    """Return width/height/duration of the first video stream."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height:format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60)
        data = json.loads(res.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        return {"width": int(stream.get("width", 1920)), "height": int(stream.get("height", 1080)),
                "duration": float((data.get("format") or {}).get("duration", 0) or 0)}
    except Exception as exc:
        log.warning(f"ffprobe failed, assuming 1920x1080: {exc}")
        return {"width": 1920, "height": 1080, "duration": 0.0}


# --------------------------------------------------------------------------- #
# ASS subtitles (word-by-word karaoke highlight)
# --------------------------------------------------------------------------- #
def _ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _clean_token(tok: str, upper: bool) -> str:
    tok = re.sub(r"[{}\\]", "", tok).strip()
    tok = tok.replace("\n", " ")
    return tok.upper() if upper else tok


def _chunk_words(words: list[dict[str, Any]], max_words: int) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for w in words:
        if cur and (len(cur) >= max_words or w["start"] - cur[-1]["end"] > 0.6
                    or cur[-1]["word"].endswith((".", "!", "?", "…"))):
            chunks.append(cur)
            cur = []
        cur.append(w)
        if w["word"].endswith((",", ";", ":")) and len(cur) >= 2:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def build_ass(words: list[dict[str, Any]], clip_start: float, clip_end: float,
              style_name: str, path: Path) -> bool:
    """Write an .ass file for the clip. Returns False if there are no words to show."""
    style = STYLES.get(style_name, STYLES["hormozi"])
    rel = []
    for w in words:
        if w["end"] <= clip_start or w["start"] >= clip_end:
            continue
        txt = _clean_token(w["word"], style["upper"])
        if not txt:
            continue
        rel.append({"start": max(0.0, w["start"] - clip_start),
                    "end": min(clip_end - clip_start, w["end"] - clip_start), "word": txt})
    if not rel:
        return False

    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {OUT_W}", f"PlayResY: {OUT_H}", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Main,{SUBTITLE_FONT},{style['size']},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,"
        f"{style['outline']},3,2,70,70,{style['margin_v']},1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    chunks = _chunk_words(rel, style["max_words"])
    hl = style["hl"]
    for ci, chunk in enumerate(chunks):
        next_chunk_start = chunks[ci + 1][0]["start"] if ci + 1 < len(chunks) else None
        for k, w in enumerate(chunk):
            start = w["start"]
            if k + 1 < len(chunk):
                end = chunk[k + 1]["start"]
            else:
                end = max(w["end"], start + 0.3)
                if next_chunk_start is not None:
                    end = min(end, next_chunk_start)
            if end - start < 0.04:
                end = start + 0.04
            if hl:
                parts = []
                for m, x in enumerate(chunk):
                    if m == k:
                        parts.append(f"{{\\1c&H{hl}&\\fscx108\\fscy108}}{x['word']}{{\\1c&HFFFFFF&\\fscx100\\fscy100}}")
                    else:
                        parts.append(x["word"])
                text = " ".join(parts)
            else:
                text = " ".join(x["word"] for x in chunk)
                if k > 0:
                    continue  # plain style shows one event per chunk
                start, end = chunk[0]["start"], max(chunk[-1]["end"], chunk[0]["start"] + 0.3)
                if next_chunk_start is not None:
                    end = min(end, next_chunk_start)
            lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Main,,0,0,0,,{text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


# --------------------------------------------------------------------------- #
# Filter graph
# --------------------------------------------------------------------------- #
def build_filter_graph(width: int, height: int, ass_name: Optional[str]) -> str:
    """Blurred 9:16 background + smart-cropped foreground (+ optional subtitles)."""
    aspect = width / max(height, 1)
    if aspect <= 0.58:  # already (almost) 9:16: just cover the frame
        graph = (f"[0:v]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
                 f"crop={OUT_W}:{OUT_H},setsar=1,fps={FPS},format=yuv420p")
    else:
        # landscape/square: centre-crop to 4:5 (keeps the speaker), scale to full width, float over blurred copy
        fg_crop = "crop=trunc(ih*0.4)*2:ih," if aspect >= 1.3 else ""
        graph = (
            "[0:v]split=2[a][b];"
            f"[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=12:3,"
            f"scale={OUT_W}:{OUT_H}:flags=bilinear,eq=brightness=-0.08:saturation=1.15[bg];"
            f"[b]{fg_crop}scale={OUT_W}:-2:flags=lanczos,setsar=1[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2:shortest=1,fps={FPS},format=yuv420p"
        )
    if ass_name:
        graph += f",ass={ass_name}"
    return graph + "[v]"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_clip(video_path: Path, clip: dict[str, Any], words: list[dict[str, Any]],
                out_dir: Path, style: str, src_info: dict[str, Any],
                progress: ProgressCb) -> dict[str, Path]:
    """Render one clip to out_dir/<clip_id>.mp4 (+ .jpg thumbnail). Raises RuntimeError on failure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_id = clip["id"]
    start, end = float(clip["start"]), float(clip["end"])
    duration = max(end - start, 1.0)
    ass_path = out_dir / f"{clip_id}.ass"
    mp4 = out_dir / f"{clip_id}.mp4"
    err_log = out_dir / f"{clip_id}.ffmpeg.log"

    try:
        has_subs = build_ass(words, start, end, style, ass_path)
    except Exception as exc:
        log.warning(f"subtitle generation failed for {clip_id}: {exc}", exc_info=True)
        has_subs = False

    graph = build_filter_graph(src_info["width"], src_info["height"], ass_path.name if has_subs else None)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-progress", "pipe:1", "-nostats",
        "-ss", f"{start:.3f}", "-i", str(video_path.resolve()), "-t", f"{duration:.3f}",
        "-filter_complex", graph, "-map", "[v]", "-map", "0:a:0?",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-r", str(FPS), "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", mp4.name,
    ]
    log.info(f"rendering {clip_id}", extra={"start": start, "end": end, "style": style, "subs": has_subs})

    try:
        with open(err_log, "w", encoding="utf-8") as errf:
            proc = subprocess.Popen(cmd, cwd=str(out_dir), stdout=subprocess.PIPE, stderr=errf,
                                    text=True, bufsize=1)
            watchdog = threading.Timer(3600, proc.kill)
            watchdog.start()
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    if line.startswith("out_time_ms="):
                        try:
                            done = int(line.split("=", 1)[1]) / 1_000_000
                            progress(min(done / duration, 0.99), None)
                        except ValueError:
                            pass
                proc.wait()
            finally:
                watchdog.cancel()
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed or not on PATH.") from exc

    if proc.returncode != 0 or not mp4.exists() or mp4.stat().st_size < 1000:
        tail = err_log.read_text(encoding="utf-8", errors="ignore")[-500:] if err_log.exists() else ""
        raise RuntimeError(f"ffmpeg failed for {clip_id} (code {proc.returncode}): {tail}")

    thumb = out_dir / f"{clip_id}.jpg"
    try:
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{min(duration * 0.3, 8):.2f}",
                        "-i", str(mp4), "-frames:v", "1", "-vf", "scale=360:-2", str(thumb)],
                       capture_output=True, timeout=60)
    except Exception as exc:
        log.warning(f"thumbnail failed for {clip_id}: {exc}")

    try:
        err_log.unlink()
    except OSError:
        pass
    progress(1.0, None)
    log.info(f"rendered {clip_id}", extra={"mb": round(mp4.stat().st_size / 1e6, 2)})
    return {"mp4": mp4, "thumb": thumb if thumb.exists() else mp4}
