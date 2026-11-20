#!/usr/bin/env python3
"""Start a LOCAL vLLM OpenAI-compatible server on 127.0.0.1:8001 (single T4, GPU 0).

Uses only the standard library, so it can be launched from any Python. The server itself runs
with the interpreter given by --python (the dedicated vLLM venv). The server keeps running in the
background after this script exits; this script waits until /health is OK (or fails loudly).

    python scripts/kaggle_vllm.py --python /kaggle/working/venv-vllm/bin/python --preset qwen3-4b-awq
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "logs" / "vllm.log"
PID_FILE = ROOT / "logs" / "vllm.pid"

# Presets that fit a single 16 GB T4 (weights are 4-bit; T4 has no bf16 -> --dtype half).
PRESETS = {
    "qwen3-4b-awq": {
        "model": "Qwen/Qwen3-4B-AWQ",
        "args": ["--max-model-len", "8192", "--gpu-memory-utilization", "0.70", "--max-num-seqs", "4"],
    },
    "gemma4-e2b": {  # needs vLLM >= 0.29 (cell 3 upgrades it when this preset is selected)
        "model": "google/gemma-4-E2B-it-qat-w4a16-ct",
        "args": ["--max-model-len", "8192", "--gpu-memory-utilization", "0.75", "--max-num-seqs", "4"],
    },
}


def http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="ignore").splitlines()[-n:])
    except OSError:
        return "(no log available)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", default=sys.executable, help="python of the vLLM venv")
    ap.add_argument("--preset", default=os.environ.get("LLM_PRESET", "qwen3-4b-awq"), choices=sorted(PRESETS))
    ap.add_argument("--model", default="", help="override the HF model id")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--timeout", type=int, default=900, help="seconds to wait for the server")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    model = args.model or preset["model"]
    base = f"http://127.0.0.1:{args.port}"

    if http_ok(f"{base}/health"):
        print(f"[vllm] already running at {base}")
        return 0

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--served-model-name", model,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--dtype", "half",            # T4 (compute 7.5) has no bfloat16
        "--enforce-eager",            # less VRAM + faster start; throughput is irrelevant for this tool
        "--trust-remote-code",
        *preset["args"],
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "VLLM_LOGGING_LEVEL": "INFO",
           "TOKENIZERS_PARALLELISM": "false", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    print(f"[vllm] starting {model} on {base} (log: {LOG_FILE})")
    try:
        logf = open(LOG_FILE, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT),
                                start_new_session=True)
        PID_FILE.write_text(str(proc.pid))
    except (OSError, FileNotFoundError) as exc:
        print(f"[vllm] cannot launch vLLM: {exc}")
        return 1

    deadline = time.time() + args.timeout
    last_print = 0.0
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"[vllm] server exited with code {proc.returncode}. Last log lines:\n{tail(LOG_FILE)}")
            print("[vllm] The studio will still work using the heuristic engine.")
            return 1
        if http_ok(f"{base}/health"):
            print(f"[vllm] READY after {int(time.time() - (deadline - args.timeout))}s -> {base}/v1 ({model})")
            return 0
        if time.time() - last_print > 30:
            last_print = time.time()
            lines = tail(LOG_FILE, 1)
            print(f"[vllm] waiting… {lines[-110:]}")
        time.sleep(3)

    print(f"[vllm] timed out waiting for the server. Last log lines:\n{tail(LOG_FILE)}")
    print("[vllm] The studio will still work using the heuristic engine.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
