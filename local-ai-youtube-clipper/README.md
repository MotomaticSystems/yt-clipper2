# 🎬 Personal YouTube Clipping Studio — No API Version

Paste a YouTube URL → get ready-to-post **1080×1920 vertical clips** with **karaoke subtitles**.
100 % local AI, single user, no login, no database, no paid API.

| Step | Tool |
|---|---|
| Download (≤1080p) | `yt-dlp` (cookies.txt supported) |
| Transcript | YouTube captions first (word timings) → fallback `faster-whisper` (GPU `float16` / CPU `int8`) |
| Clip selection | Local **vLLM** (`Qwen/Qwen3-4B-AWQ`, OpenAI-compatible on `127.0.0.1:8001`) → **heuristic engine** fallback |
| Rendering | FFmpeg only: blurred 9:16 background, centre-cropped foreground, Hormozi-style word highlight, `loudnorm`, H.264 + AAC, 30 fps |
| Backend / UI | FastAPI + one static `frontend/index.html` |

The only secret ever used is `NGROK_AUTHTOKEN` (free) — and only to reach Kaggle from your browser.
**If vLLM is down or runs out of memory the tool keeps working** with the heuristic scoring engine.

## Project structure

```
/
├── backend/
│   ├── main.py          # FastAPI app + routes
│   ├── downloader.py    # yt-dlp download + YouTube captions (json3 → word timings)
│   ├── transcriber.py   # faster-whisper, sentence building, .txt/.srt/.vtt export
│   ├── analyzer.py      # local LLM clip finder + heuristic fallback + title/description/hashtags
│   ├── clipper.py       # FFmpeg filter graph, ASS karaoke subtitles, loudnorm
│   ├── jobs.py          # background job queue + pipeline
│   └── models.py        # config, JSONL logging, URL validation, pydantic models
├── frontend/index.html  # single page, dark, mobile friendly, no build step
├── scripts/
│   ├── kaggle_studio.py # backend + ngrok, prints URL after health check
│   └── kaggle_vllm.py   # local vLLM server on :8001 (T4-friendly)
├── requirements.txt         # studio venv
├── requirements-vllm.txt    # vLLM venv (kept separate)
├── kaggle.ipynb             # 5-cell Kaggle notebook
└── README.md
```

Runtime output: `outputs/{job_id}/` (`job.json`, `clips.json`, `transcript.*`, `clips/clip_01.mp4 …`), logs in `logs/app.jsonl`.

---

## A) Kaggle (free T4) — recommended

1. Push this project to your own GitHub repo (or upload it as a Kaggle Dataset named `clipping-studio`).
2. Kaggle → **New Notebook** → *File → Import Notebook* → upload `kaggle.ipynb`.
3. Settings (right panel): **Accelerator: GPU T4 x1** (use one T4), **Internet: On**.
4. *Add-ons → Secrets* → add **`NGROK_AUTHTOKEN`** (free token from <https://dashboard.ngrok.com/get-started/your-authtoken>). Attach it to the notebook.
5. In **Cell 1** set `REPO_URL` to your repo (skip if you use the Dataset route).
6. Run the cells in order:
   | Cell | What it does | Time |
   |---|---|---|
   | 1 | clone repo | seconds |
   | 2 | Python 3.12 venvs via `uv` (studio + vLLM, separate from the kernel) | ~1 min |
   | 3 | `ffmpeg`, fonts, studio deps, vLLM | ~5–8 min |
   | 4 | local vLLM server on `127.0.0.1:8001` (model download + load) | ~3–6 min |
   | 5 | FastAPI on `:8000` + ngrok → prints `PUBLIC URL: https://….ngrok-free.app` | ~20 s |
7. Open the public URL (click *Visit Site* on the ngrok notice). **Keep Cell 5 running** while you work.

### YouTube blocks Kaggle? → cookies.txt
YouTube frequently shows *“Sign in to confirm you’re not a bot”* on datacenter IPs.
1. In a desktop browser, log in to YouTube (use a throw-away account if you prefer).
2. Export cookies in **Netscape format** with a "Get cookies.txt (locally)" style extension while on `youtube.com`.
3. In the UI: *Settings → Upload cookies.txt*. Cookies are stored only in `data/cookies.txt` (mode 600), copied per-job and never logged.

### Using Gemma instead of Qwen
In Cell 1 set `LLM_PRESET = "gemma4-e2b"` (`google/gemma-4-E2B-it-qat-w4a16-ct`). It needs vLLM ≥ 0.29 (Cell 3 upgrades automatically) and a driver supporting its CUDA build. If it fails to start, the heuristic engine takes over — nothing else changes.

---

## B) Run locally

Prerequisites: Python 3.12, **FFmpeg** (`apt install ffmpeg fonts-dejavu-core` / `brew install ffmpeg` / `choco install ffmpeg`).

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

Works immediately with the heuristic engine. For the local LLM (NVIDIA GPU ≥ 8 GB):

```bash
python3.12 -m venv .venv-vllm && .venv-vllm/bin/pip install -r requirements-vllm.txt
python scripts/kaggle_vllm.py --python .venv-vllm/bin/python --preset qwen3-4b-awq
```

CPU only? Whisper automatically uses `base` + `int8`. GPU? `small` + `float16`.

---

## API

| Method | Path | Description |
|---|---|---|
| POST | `/api/jobs` | `{youtube_url, num_clips, min_duration, max_duration, subtitle_style}` → job (202) |
| GET | `/api/jobs/{id}` | `status`, `progress` (0-100), `stage` (`downloading/transcribing/analyzing/clipping/completed/failed`), `message`, `error` |
| GET | `/api/jobs/{id}/clips` | clips: `title, description, hashtags, score, duration, preview_url, download_url` |
| GET | `/api/jobs/{id}/preview/{clip_id}` | MP4 stream (range requests) |
| GET | `/api/jobs/{id}/download/{clip_id}` | MP4 download |
| GET | `/api/jobs/{id}/transcript/{txt\|srt\|vtt\|json}` | transcript files |
| POST | `/api/upload-cookies` | multipart `file` = cookies.txt |
| GET/DELETE | `/api/cookies` | status / remove cookies |
| GET | `/api/health` | ffmpeg, LLM, cookies status |

`subtitle_style`: `hormozi` (white + yellow word highlight, default), `neon` (green highlight), `clean` (white).

```bash
curl -X POST localhost:8000/api/jobs -H 'content-type: application/json' \
  -d '{"youtube_url":"https://youtu.be/VIDEO_ID","num_clips":5,"min_duration":30,"max_duration":90,"subtitle_style":"hormozi"}'
```

## Configuration (environment variables, all optional)

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:8001/v1` | local OpenAI-compatible server |
| `LLM_MODEL` | auto | model id (auto-detected from `/v1/models`) |
| `DISABLE_LLM` | `0` | `1` = always use the heuristic engine |
| `WHISPER_MODEL` | `small` GPU / `base` CPU | any faster-whisper model name |
| `WHISPER_LANGUAGE` | auto | e.g. `en` |
| `SUBTITLE_FONT` | `DejaVu Sans` | any installed font |
| `KEEP_SOURCE` | `0` | `1` keeps the downloaded source video |
| `MAX_VIDEO_SECONDS` | `14400` | reject longer videos |
| `OUTPUTS_DIR`, `LOGS_DIR`, `DATA_DIR` | `./outputs`, `./logs`, `./data` | storage paths |

## How clip selection works

1. **Transcript** → sentence units (punctuation + pauses).
2. **LLM** (if reachable): transcript is split into chunks with `[seconds] text` lines; the model must return a JSON array
   `[{"start":120,"end":175,"title":"…","score":9,"reason":"…"}]`. Output is parsed defensively (think-tags, code fences, truncated JSON).
3. Every clip is **snapped to sentence boundaries** and forced into `min_duration…max_duration`; overlaps are removed.
4. **Heuristic engine** fills in whatever the LLM didn’t deliver (or everything, if it is offline): hook keywords in the first 5 s
   (*secret, mistake, how to, why, stop, never, numbers, $, %…*), questions, energy words, speaking pace, complete endings.
5. Title, description and 5 hashtags come from the LLM when present, otherwise they are generated locally from the transcript.

## Troubleshooting

* **Bot-check / HTTP 429 / age-restricted** → upload a fresh `cookies.txt`; `pip install -U yt-dlp`.
* **`Could not load library libcudnn…`** → `kaggle_studio.py` exports the pip CUDA libs; if you run uvicorn yourself, add `site-packages/nvidia/*/lib` to `LD_LIBRARY_PATH`. On failure Whisper automatically retries on CPU.
* **vLLM out of memory** → lower `--gpu-memory-utilization` / `--max-model-len` in `scripts/kaggle_vllm.py`, or ignore it: the heuristic engine is used.
* **Subtitles look wrong / missing glyphs** → install a font and set `SUBTITLE_FONT`.
* Everything is logged to `logs/app.jsonl` (one JSON object per line); vLLM → `logs/vllm.log`; backend stdout → `logs/uvicorn.log`.

Use only with content you have the rights to repurpose.
