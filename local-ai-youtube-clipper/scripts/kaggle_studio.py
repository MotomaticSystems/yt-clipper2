#!/usr/bin/env python3
"""Start the FastAPI backend (port 8000) + ngrok tunnel and print the public URL.

Run with the *studio venv* python. The public URL is printed only after the backend health
check passes (locally AND through the tunnel). The script keeps running - keep the notebook
cell alive while you use the tool. Ctrl+C / interrupting the cell stops everything.

    NGROK_AUTHTOKEN=xxxx python scripts/kaggle_studio.py
"""
from __future__ import annotations

import argparse
import glob
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "logs" / "uvicorn.log"


def nvidia_lib_dirs() -> list[str]:
    """Locate pip-installed CUDA libs (cublas/cudnn) so ctranslate2 can use the GPU."""
    dirs: list[str] = []
    for sp in {p for p in sys.path if p.endswith("site-packages")}:
        dirs += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
    return dirs


def tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="ignore").splitlines()[-n:])
    except OSError:
        return "(no log available)"


def wait_health(url: str, proc: subprocess.Popen | None, timeout: int, headers: dict | None = None) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            r = requests.get(url, timeout=5, headers=headers)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-tunnel", action="store_true", help="do not start ngrok (local use)")
    args = ap.parse_args()

    ld = os.pathsep.join(nvidia_lib_dirs() + [os.environ.get("LD_LIBRARY_PATH", "")])
    extra_path = os.pathsep.join([str(Path.home() / ".deno" / "bin"), os.path.dirname(sys.executable)])
    env = {**os.environ, "LD_LIBRARY_PATH": ld, "PYTHONUNBUFFERED": "1",
           "PATH": extra_path + os.pathsep + os.environ.get("PATH", "")}
    env.setdefault("PYTHONPATH", str(ROOT))

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", str(args.port),
           "--log-level", "info"]
    print(f"[studio] starting backend on port {args.port} …")
    logf = open(LOG_FILE, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT)

    def shutdown(*_: object) -> None:
        print("\n[studio] stopping…")
        try:
            ngrok_stop()
        except Exception:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(0)

    def ngrok_stop() -> None:
        from pyngrok import ngrok
        ngrok.kill()

    signal.signal(signal.SIGTERM, shutdown)

    local = f"http://127.0.0.1:{args.port}"
    if not wait_health(f"{local}/api/health", proc, timeout=90):
        print(f"[studio] BACKEND FAILED health check. Last log lines:\n{tail(LOG_FILE)}")
        if proc.poll() is None:
            proc.terminate()
        return 1
    health = requests.get(f"{local}/api/health", timeout=10).json()
    print(f"[studio] backend healthy | ffmpeg={health['ffmpeg']} | LLM available={health['llm']['available']}"
          f" ({health['llm'].get('model') or health['llm'].get('reason', '')})")
    if not health["llm"]["available"]:
        print("[studio] note: local LLM is offline - clips will be picked by the heuristic engine.")

    public_url = None
    token = os.environ.get("NGROK_AUTHTOKEN", "").strip()
    if args.no_tunnel:
        print(f"[studio] tunnel disabled. Local URL: {local}")
    elif not token:
        print("[studio] NGROK_AUTHTOKEN is not set - no public URL. (Add it as a Kaggle secret.)")
        print(f"[studio] Local URL: {local}")
    else:
        try:
            from pyngrok import conf, ngrok
            conf.get_default().auth_token = token
            ngrok.set_auth_token(token)
            tunnel = ngrok.connect(args.port, "http")
            candidate = tunnel.public_url.replace("http://", "https://")
            if wait_health(f"{candidate}/api/health", proc, timeout=45, headers={"ngrok-skip-browser-warning": "1"}):
                public_url = candidate
            else:
                print("[studio] tunnel opened but the health check through it failed.")
        except Exception as exc:
            print(f"[studio] ngrok failed: {exc}")

    if public_url:
        print("\n" + "=" * 64)
        print(f"  PUBLIC URL: {public_url}")
        print("=" * 64)
        print("Open it in your browser (click 'Visit Site' on the ngrok notice). Keep this cell running.\n")

    try:
        while True:
            if proc.poll() is not None:
                print(f"[studio] backend exited with code {proc.returncode}. Last log lines:\n{tail(LOG_FILE)}")
                return 1
            time.sleep(10)
    except KeyboardInterrupt:
        shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
