const stages = [
  { icon: "⬇", name: "Download", text: "yt-dlp, up to 1080p, cookies.txt support" },
  { icon: "📝", name: "Transcribe", text: "YouTube captions first, faster-whisper fallback" },
  { icon: "🧠", name: "Analyze", text: "Local vLLM (Qwen3-4B-AWQ) + heuristic fallback" },
  { icon: "✂", name: "Clip", text: "FFmpeg 1080×1920, karaoke subtitles, loudnorm" },
];

const tree = `backend/      main · downloader · transcriber · analyzer · clipper · jobs · models
frontend/     index.html  (served by FastAPI at "/")
scripts/      kaggle_studio.py · kaggle_vllm.py
kaggle.ipynb  5-cell Kaggle T4 notebook
requirements.txt · requirements-vllm.txt · README.md`;

const run = `pip install -r requirements.txt
uvicorn backend.main:app --port 8000
# → http://localhost:8000`;

export default function App() {
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="mx-auto max-w-4xl px-5 py-14">
        <p className="text-sm font-semibold uppercase tracking-widest text-yellow-400">No API version</p>
        <h1 className="mt-2 text-4xl font-bold tracking-tight sm:text-5xl">
          Personal YouTube <span className="text-yellow-400">Clipping Studio</span>
        </h1>
        <p className="mt-4 max-w-2xl text-zinc-400">
          A single-user Python tool that turns a YouTube URL into vertical, subtitled clips using 100% local AI. The
          full project (FastAPI backend, single-file frontend, Kaggle notebook) is in this repository. The studio UI
          itself is served by the Python backend from <code className="text-zinc-200">frontend/index.html</code>.
        </p>

        <div className="mt-10 grid gap-3 sm:grid-cols-2">
          {stages.map((s, i) => (
            <div key={s.name} className="rounded-xl border border-zinc-800 bg-zinc-900 p-4">
              <div className="flex items-center gap-3">
                <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-yellow-400 text-lg text-black">
                  {s.icon}
                </span>
                <h3 className="font-semibold">
                  {i + 1}. {s.name}
                </h3>
              </div>
              <p className="mt-2 text-sm text-zinc-400">{s.text}</p>
            </div>
          ))}
        </div>

        <div className="mt-10 grid gap-4 md:grid-cols-2">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-5">
            <h2 className="mb-3 font-semibold">Project layout</h2>
            <pre className="overflow-x-auto whitespace-pre-wrap text-xs leading-relaxed text-zinc-300">{tree}</pre>
          </div>
          <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-5">
            <h2 className="mb-3 font-semibold">Run locally</h2>
            <pre className="overflow-x-auto text-xs leading-relaxed text-zinc-300">{run}</pre>
            <p className="mt-3 text-sm text-zinc-400">
              On Kaggle: import <code className="text-zinc-200">kaggle.ipynb</code>, add the{" "}
              <code className="text-zinc-200">NGROK_AUTHTOKEN</code> secret and run the 5 cells.
            </p>
          </div>
        </div>

        <p className="mt-10 text-sm text-zinc-500">
          No Gemini / OpenAI / Anthropic / NIM / Pexels. Only secret: a free ngrok token for the Kaggle tunnel. If the
          local LLM fails, the heuristic engine keeps the pipeline running.
        </p>
      </div>
    </div>
  );
}
