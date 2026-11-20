"""Background job manager: download -> transcribe -> analyse -> clip. State lives in outputs/{job_id}/."""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

from backend import analyzer, clipper, downloader, transcriber
from backend.models import COOKIES_PATH, KEEP_SOURCE, OUTPUTS_DIR, JobCreate, get_logger

log = get_logger("jobs")

# (stage name, start %, end %)
STAGES = {
    "downloading": (0.0, 25.0),
    "transcribing": (25.0, 50.0),
    "analyzing": (50.0, 65.0),
    "clipping": (65.0, 100.0),
}


class JobManager:
    """Single-worker queue (the GPU is shared, so jobs run one at a time)."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="job")
        self._load_existing()

    # ------------------------------------------------------------------ persistence
    def job_dir(self, job_id: str) -> Path:
        return OUTPUTS_DIR / job_id

    def _persist(self, job: dict[str, Any]) -> None:
        try:
            d = self.job_dir(job["id"])
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / "job.json.tmp"
            tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(d / "job.json")
        except OSError as exc:
            log.error(f"could not persist job {job.get('id')}: {exc}")

    def _load_existing(self) -> None:
        try:
            for f in OUTPUTS_DIR.glob("*/job.json"):
                try:
                    job = json.loads(f.read_text(encoding="utf-8"))
                    if job.get("status") in ("queued", "running"):
                        job.update(status="failed", error="Interrupted by a server restart.", message="Interrupted")
                        self._persist(job)
                    self._jobs[job["id"]] = job
                except (OSError, ValueError, KeyError) as exc:
                    log.warning(f"skipping unreadable job file {f}: {exc}")
            log.info(f"loaded {len(self._jobs)} previous jobs")
        except Exception as exc:
            log.error(f"failed to load previous jobs: {exc}", exc_info=True)

    # ------------------------------------------------------------------ public API
    def create(self, req: JobCreate) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        now = time.time()
        job = {
            "id": job_id, "status": "queued", "stage": "queued", "progress": 0.0,
            "message": "Waiting in queue…", "error": None, "youtube_url": req.youtube_url,
            "video_title": "", "created_at": now, "updated_at": now, "transcript_source": "",
            "analysis_method": "", "clips_count": 0, "params": req.model_dump(),
        }
        with self._lock:
            self._jobs[job_id] = job
        self.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        self._persist(job)
        self._executor.submit(self._run_safe, job_id)
        log.info("job created", extra={"job_id": job_id, "url": req.youtube_url})
        return dict(job)

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def list(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j["created_at"], reverse=True)
            return [dict(j) for j in jobs[:limit]]

    def clips(self, job_id: str) -> list[dict[str, Any]]:
        f = self.job_dir(job_id) / "clips.json"
        if not f.exists():
            return []
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.error(f"cannot read clips.json for {job_id}: {exc}")
            return []

    def clip_file(self, job_id: str, clip_id: str, ext: str = "mp4") -> Optional[Path]:
        p = self.job_dir(job_id) / "clips" / f"{clip_id}.{ext}"
        return p if p.exists() else None

    # ------------------------------------------------------------------ internals
    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.update(fields)
            job["updated_at"] = time.time()
            snapshot = dict(job)
        self._persist(snapshot)

    def _stage_cb(self, job_id: str, stage: str) -> Callable[[float, Optional[str]], None]:
        lo, hi = STAGES[stage]
        last = {"t": 0.0, "p": -1.0}

        def cb(frac: float, message: Optional[str] = None) -> None:
            pct = lo + (hi - lo) * max(0.0, min(frac, 1.0))
            now = time.time()
            if message is None and now - last["t"] < 0.5 and frac < 1.0:
                return  # throttle disk writes for noisy progress streams
            last["t"] = now
            fields: dict[str, Any] = {"stage": stage, "progress": round(max(pct, last["p"]), 1)}
            last["p"] = fields["progress"]
            if message:
                fields["message"] = message
            self._update(job_id, **fields)

        return cb

    def _run_safe(self, job_id: str) -> None:
        try:
            self._run(job_id)
        except Exception as exc:  # last line of defence - the worker thread must survive
            log.error(f"job crashed: {exc}", extra={"job_id": job_id}, exc_info=True)
            self._update(job_id, status="failed", stage="failed", error=str(exc)[:600] or "Unexpected error",
                         message="Failed")
        finally:
            try:
                transcriber.unload_model()
            except Exception:
                pass

    def _run(self, job_id: str) -> None:
        job = self.get(job_id)
        if not job:
            return
        params = job["params"]
        jdir = self.job_dir(job_id)
        ex = {"job_id": job_id}
        self._update(job_id, status="running", stage="downloading", message="Starting…")

        # ---------------- 1. download ----------------
        try:
            info = downloader.download_video(job["youtube_url"], jdir, COOKIES_PATH,
                                             self._stage_cb(job_id, "downloading"))
        except Exception as exc:
            log.error(f"download stage failed: {exc}", extra=ex)
            self._update(job_id, status="failed", stage="failed", error=str(exc), message="Download failed")
            return
        video_path: Path = info["video_path"]
        self._update(job_id, video_title=info["title"])

        # ---------------- 2. transcript ----------------
        self._update(job_id, stage="transcribing", message="Preparing transcript…")
        transcript: Optional[dict[str, Any]] = None
        if info.get("caption_words"):
            try:
                self._update(job_id, message="Using YouTube captions (fast path)")
                transcript = transcriber.transcript_from_words(info["caption_words"], "youtube-captions")
                if len(transcript["sentences"]) < 3:
                    transcript = None
            except Exception as exc:
                log.warning(f"caption transcript failed, using Whisper: {exc}", extra=ex)
                transcript = None
        if transcript is None:
            try:
                transcript = transcriber.transcribe(video_path, jdir, self._stage_cb(job_id, "transcribing"))
            except Exception as exc:
                log.error(f"transcription failed: {exc}", extra=ex, exc_info=True)
                self._update(job_id, status="failed", stage="failed", error=f"Transcription failed: {exc}"[:600],
                             message="Transcription failed")
                return
            finally:
                transcriber.unload_model()  # free VRAM for the LLM / ffmpeg
        transcriber.save_transcript(transcript, jdir / "transcript")
        self._update(job_id, transcript_source=transcript["source"], progress=50.0)

        # ---------------- 3. analyse ----------------
        self._update(job_id, stage="analyzing", message="Finding viral moments…")
        try:
            clips, method = analyzer.analyze(transcript, params["num_clips"], params["min_duration"],
                                             params["max_duration"], info["title"],
                                             self._stage_cb(job_id, "analyzing"))
        except Exception as exc:
            log.error(f"analysis failed: {exc}", extra=ex, exc_info=True)
            self._update(job_id, status="failed", stage="failed", error=f"Analysis failed: {exc}"[:600],
                         message="Analysis failed")
            return
        self._update(job_id, analysis_method=method, progress=65.0)

        # ---------------- 4. clip ----------------
        self._update(job_id, stage="clipping", message=f"Rendering {len(clips)} clips…")
        src_info = clipper.probe_video(video_path)
        clips_dir = jdir / "clips"
        words = transcript["words"]
        done: list[dict[str, Any]] = []
        lo, hi = STAGES["clipping"]
        for n, clip in enumerate(clips):
            base = lo + (hi - lo) * n / len(clips)
            span = (hi - lo) / len(clips)

            def cb(frac: float, msg: Optional[str] = None, _b: float = base, _s: float = span, _n: int = n) -> None:
                fields: dict[str, Any] = {"stage": "clipping", "progress": round(_b + _s * min(frac, 1.0), 1)}
                if msg:
                    fields["message"] = msg
                self._update(job_id, **fields)

            self._update(job_id, message=f"Rendering clip {n + 1}/{len(clips)}: {clip['title'][:50]}")
            try:
                clipper.render_clip(video_path, clip, words, clips_dir, params["subtitle_style"], src_info, cb)
            except Exception as exc:
                log.error(f"clip {clip['id']} failed: {exc}", extra=ex, exc_info=True)
                continue
            clip["preview_url"] = f"/api/jobs/{job_id}/preview/{clip['id']}"
            clip["download_url"] = f"/api/jobs/{job_id}/download/{clip['id']}"
            clip["thumbnail_url"] = f"/api/jobs/{job_id}/thumb/{clip['id']}"
            done.append(clip)
            (jdir / "clips.json").write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
            self._update(job_id, clips_count=len(done))

        if not done:
            self._update(job_id, status="failed", stage="failed", error="All clips failed to render. See logs/app.jsonl.",
                         message="Rendering failed")
            return

        if not KEEP_SOURCE:
            try:
                shutil.rmtree(jdir / "source", ignore_errors=True)
            except Exception as exc:
                log.warning(f"could not remove source files: {exc}", extra=ex)
        self._update(job_id, status="completed", stage="completed", progress=100.0, clips_count=len(done),
                     message=f"Done - {len(done)} clips ready")
        log.info("job completed", extra={**ex, "clips": len(done), "method": method})


_manager: Optional[JobManager] = None
_manager_lock = threading.Lock()


def get_manager() -> JobManager:
    """Lazy singleton so importing the module has no side effects."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
        return _manager
