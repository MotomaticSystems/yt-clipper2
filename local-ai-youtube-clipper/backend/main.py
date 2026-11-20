"""FastAPI application: REST API + static single-page frontend."""
from __future__ import annotations

import re
import shutil
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from backend import analyzer
from backend.jobs import get_manager
from backend.models import (COOKIES_PATH, FRONTEND_DIR, ClipInfo, JobCreate, JobStatus,
                            get_logger)

log = get_logger("api")

JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")
CLIP_ID_RE = re.compile(r"^clip_\d{2}$")
MAX_COOKIE_BYTES = 5 * 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_manager()
    log.info("ffmpeg=%s ffprobe=%s" % (bool(shutil.which("ffmpeg")), bool(shutil.which("ffprobe"))))
    if not shutil.which("ffmpeg"):
        log.error("ffmpeg NOT FOUND on PATH - clipping will fail. Install it (apt install ffmpeg).")
    yield


app = FastAPI(title="Personal YouTube Clipping Studio", version="1.0.0", lifespan=lifespan)


def _valid_job(job_id: str) -> dict[str, Any]:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    job = get_manager().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _valid_clip(clip_id: str) -> str:
    if not CLIP_ID_RE.match(clip_id):
        raise HTTPException(status_code=404, detail="Clip not found")
    return clip_id


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):  # never leak stack traces, always log them
    log.error(f"unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ------------------------------------------------------------------ frontend
@app.get("/", include_in_schema=False)
def index():
    page = FRONTEND_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail="frontend/index.html is missing")
    return FileResponse(page, media_type="text/html", headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------------ health
@app.get("/api/health")
def health():
    ffmpeg_ok = bool(shutil.which("ffmpeg"))
    return {
        "status": "ok",
        "ffmpeg": ffmpeg_ok,
        "llm": analyzer.llm_status(),
        "cookies": COOKIES_PATH.exists() and COOKIES_PATH.stat().st_size > 0,
    }


# ------------------------------------------------------------------ jobs
@app.post("/api/jobs", response_model=JobStatus, status_code=202)
def create_job(req: JobCreate):
    if not shutil.which("ffmpeg"):
        raise HTTPException(status_code=503, detail="ffmpeg is not installed on the server.")
    job = get_manager().create(req)
    return job


@app.get("/api/jobs", response_model=list[JobStatus])
def list_jobs():
    return get_manager().list()


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    return _valid_job(job_id)


@app.get("/api/jobs/{job_id}/clips", response_model=list[ClipInfo])
def get_clips(job_id: str):
    _valid_job(job_id)
    return get_manager().clips(job_id)


def _clip_path(job_id: str, clip_id: str, ext: str) -> Path:
    _valid_job(job_id)
    _valid_clip(clip_id)
    path = get_manager().clip_file(job_id, clip_id, ext)
    if not path:
        raise HTTPException(status_code=404, detail="File not found")
    return path


@app.get("/api/jobs/{job_id}/preview/{clip_id}")
def preview_clip(job_id: str, clip_id: str):
    return FileResponse(_clip_path(job_id, clip_id, "mp4"), media_type="video/mp4")


@app.get("/api/jobs/{job_id}/thumb/{clip_id}")
def thumb_clip(job_id: str, clip_id: str):
    return FileResponse(_clip_path(job_id, clip_id, "jpg"), media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/download/{clip_id}")
def download_clip(job_id: str, clip_id: str):
    path = _clip_path(job_id, clip_id, "mp4")
    title = clip_id
    for c in get_manager().clips(job_id):
        if c.get("id") == clip_id:
            title = re.sub(r"[^A-Za-z0-9 _-]", "", c.get("title", clip_id)).strip().replace(" ", "_")[:60] or clip_id
            break
    return FileResponse(path, media_type="video/mp4", filename=f"{clip_id}_{title}.mp4")


@app.get("/api/jobs/{job_id}/transcript/{fmt}")
def download_transcript(job_id: str, fmt: str):
    _valid_job(job_id)
    if fmt not in ("txt", "srt", "vtt", "json"):
        raise HTTPException(status_code=404, detail="Unknown format")
    path = get_manager().job_dir(job_id) / f"transcript.{fmt}"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript not available yet")
    return FileResponse(path, filename=f"transcript_{job_id}.{fmt}")


# ------------------------------------------------------------------ cookies
@app.post("/api/upload-cookies")
async def upload_cookies(file: UploadFile = File(...)):
    try:
        data = await file.read(MAX_COOKIE_BYTES + 1)
    except Exception as exc:
        log.error(f"cookie upload read failed: {exc}")
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.")
    if not data or len(data) > MAX_COOKIE_BYTES:
        raise HTTPException(status_code=400, detail="cookies.txt must be between 1 byte and 5 MB.")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="cookies.txt must be a UTF-8 text file (Netscape format).")
    if "\t" not in text or ("youtube.com" not in text and "google.com" not in text):
        raise HTTPException(status_code=400,
                            detail="This does not look like a Netscape cookies.txt that contains youtube.com cookies.")
    if not text.lstrip().startswith(("# Netscape", "# HTTP Cookie File")):
        text = "# Netscape HTTP Cookie File\n" + text
    try:
        COOKIES_PATH.parent.mkdir(parents=True, exist_ok=True)
        COOKIES_PATH.write_text(text.replace("\r\n", "\n"), encoding="utf-8")
        try:
            COOKIES_PATH.chmod(0o600)
        except OSError:
            pass
    except OSError as exc:
        log.error(f"cannot save cookies: {exc}")
        raise HTTPException(status_code=500, detail="Could not save cookies on the server.")
    n = sum(1 for ln in text.splitlines() if ln.strip() and not ln.startswith("#"))
    log.info("cookies.txt uploaded", extra={"entries": n})  # never log cookie contents
    return {"status": "saved", "entries": n}


@app.get("/api/cookies")
def cookies_status():
    ok = COOKIES_PATH.exists() and COOKIES_PATH.stat().st_size > 0
    return {"present": ok}


@app.delete("/api/cookies")
def delete_cookies():
    try:
        COOKIES_PATH.unlink(missing_ok=True)
    except OSError as exc:
        log.error(f"cannot delete cookies: {exc}")
        raise HTTPException(status_code=500, detail="Could not delete cookies.")
    return {"present": False}
