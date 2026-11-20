"""Clip detection: local LLM (vLLM, OpenAI-compatible on localhost) + heuristic fallback.

The pipeline can never fail because of the LLM: any LLM problem silently degrades to the
heuristic scoring engine, and the heuristic engine also tops up when the LLM returns too few clips.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Callable, Optional

import requests

from backend.models import (LLM_BASE_URL, LLM_CHUNK_CHARS, LLM_DISABLED, LLM_MODEL,
                            LLM_TIMEOUT, get_logger)

log = get_logger("analyzer")
ProgressCb = Callable[[float, Optional[str]], None]

_llm_model_cache: dict[str, str] = {}


# =========================================================================== #
# Sentence-boundary snapping
# =========================================================================== #
def snap_to_sentences(sentences: list[dict], start: float, end: float,
                      min_dur: float, max_dur: float) -> tuple[float, float]:
    """Expand/shrink [start,end] to whole sentences while honouring min/max duration."""
    n = len(sentences)
    i = min(range(n), key=lambda k: abs(sentences[k]["start"] - start))
    # if the requested start is clearly inside a sentence, begin at that sentence
    if sentences[i]["start"] > start + 0.5 and i > 0 and sentences[i - 1]["end"] > start:
        i -= 1
    j = min(range(i, n), key=lambda k: abs(sentences[k]["end"] - end))

    def dur() -> float:
        return sentences[j]["end"] - sentences[i]["start"]

    while dur() < min_dur:
        if j + 1 < n and sentences[j + 1]["end"] - sentences[i]["start"] <= max_dur:
            j += 1
        elif i > 0 and sentences[j]["end"] - sentences[i - 1]["start"] <= max_dur:
            i -= 1
        else:
            break
    while dur() > max_dur and j > i:
        j -= 1

    s, e = sentences[i]["start"], sentences[j]["end"]
    if e - s > max_dur:  # single huge sentence: cut at the last word that fits
        fit = [w["end"] for k in range(i, j + 1) for w in sentences[k]["words"] if w["end"] <= s + max_dur]
        e = fit[-1] if fit else s + max_dur
    return s, e


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


# =========================================================================== #
# Text helpers (titles / descriptions / hashtags - fully local)
# =========================================================================== #
STOPWORDS = set("""a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how i if in into is it
its just like me more most my no nor not now of off on once only or other our out over own same she should so some such than that the
their them then there these they this those through to too under until up very was we were what when where which while who whom why will
with would you your yeah okay really going gonna got get thing things know right think mean actually kind lot want say said let one
yes well cause don dont doesn didn isn wasn aren won ll re ve""".split())

FILLERS = ("so", "and", "but", "um", "uh", "like", "well", "yeah", "okay", "ok", "because", "then", "also", "or")
DEFAULT_TAGS = ["shorts", "viral", "fyp", "mustwatch", "trending"]


def _clean_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z']{2,}", text.lower()) if w not in STOPWORDS]


def make_hashtags(text: str, given: Optional[list[Any]] = None, count: int = 5) -> list[str]:
    """Return exactly `count` hashtags (LLM-provided first, then keywords, then defaults)."""
    tags: list[str] = []

    def add(raw: str) -> None:
        tag = re.sub(r"[^A-Za-z0-9]", "", str(raw).lstrip("#"))
        if 2 <= len(tag) <= 28 and tag.lower() not in {t.lower() for t in tags}:
            tags.append(tag)

    for g in given or []:
        add(g)
    for word, _ in Counter(_clean_words(text)).most_common(12):
        add(word)
    for d in DEFAULT_TAGS:
        add(d)
    return [f"#{t}" for t in tags[:count]]


def make_title(text: str, max_len: int = 62) -> str:
    """Create a short, punchy title from the clip's opening hook."""
    sentence = re.split(r"(?<=[.!?])\s+", text.strip())[0] if text.strip() else "Must-watch moment"
    words = sentence.split()
    while words and words[0].lower().strip(",.") in FILLERS:
        words.pop(0)
    title = " ".join(words) or sentence
    if len(title) > max_len:
        title = title[:max_len].rsplit(" ", 1)[0].rstrip(",;:- ") + "…"
    title = title.strip(" .")
    return (title[:1].upper() + title[1:]) if title else "Must-watch moment"


def make_description(text: str, max_len: int = 220) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    desc = " ".join(sentences[:2]).strip()
    if len(desc) > max_len:
        desc = desc[:max_len].rsplit(" ", 1)[0].rstrip(",;:- ") + "…"
    return desc


# =========================================================================== #
# Heuristic scoring engine
# =========================================================================== #
KEYWORDS: dict[str, float] = {
    # secrets / mistakes / how-to
    r"\bsecret(s)?\b": 3, r"\bmistake(s)?\b": 3, r"\bhow to\b": 2.5, r"\bwhy\b": 2, r"\bstop\b": 2.5,
    r"\bnever\b": 2.5, r"\bnobody\b": 2.5, r"\bno one\b": 2, r"\bwarning\b": 2, r"\btruth\b": 2.5,
    r"\bhack(s)?\b": 2, r"\bhidden\b": 2, r"\bwrong\b": 2, r"\bbiggest\b": 2, r"\bworst\b": 2,
    r"\bbest\b": 1.5, r"\bproblem\b": 1.5, r"\bsolution\b": 1.5, r"\bstep\b": 1.5, r"\bfree\b": 1.5,
    r"\bmoney\b": 2, r"\brich\b": 1.5, r"\bmillion(s)?\b": 2, r"\bbillion\b": 2, r"\bproof\b": 1.5,
    r"\bregret\b": 2, r"\bfail(ed|ure)?\b": 2, r"\bquit\b": 1.5, r"\blesson(s)?\b": 2,
    r"\bnumber (one|1|two|2|three|3)\b": 2, r"\bthe (first|only|real) (thing|reason|way)\b": 2,
    r"\bnobody tells\b": 3, r"\bnobody talks\b": 3, r"\bdon'?t\b": 1,
    r"\bsecrets? (to|of)\b": 2,
}
ENERGY = {"amazing", "insane", "crazy", "incredible", "shocking", "unbelievable", "literally", "huge",
          "massive", "terrifying", "powerful", "ridiculous", "obsessed", "game-changer", "unreal", "wow",
          "dangerous", "genius", "brilliant", "life-changing", "mind-blowing", "always", "everything"}
HOOK_OPENERS = ("here's", "here is", "the truth", "nobody", "stop", "let me tell", "what if", "imagine",
                "listen", "most people", "the biggest", "the secret", "i'll show", "you need", "you should never",
                "the reason", "this is why", "did you know", "if you")
_NUM_RE = re.compile(r"(\$\s?\d|\d+\s?%|\b\d{2,}\b|\b\d+(?:\.\d+)?\s?(k|m|x)\b)", re.I)
_KW_RES = [(re.compile(p, re.I), w) for p, w in KEYWORDS.items()]


def score_text(text: str) -> tuple[float, list[str]]:
    """Return (score, matched-feature labels) for a piece of text."""
    low = text.lower()
    score = 0.0
    hits: list[str] = []
    for rx, weight in _KW_RES:
        found = rx.findall(low)
        if found:
            score += weight * min(len(found), 2)
            hits.append(rx.pattern.replace(r"\b", "").replace("(s)?", "").replace("\\", "")[:18])
    nums = len(_NUM_RE.findall(text))
    if nums:
        score += min(nums, 3) * 1.2
        hits.append("numbers")
    q = text.count("?")
    if q:
        score += min(q, 3) * 1.5
        hits.append("questions")
    ex = text.count("!")
    if ex:
        score += min(ex, 3) * 0.8
    energy = sum(1 for w in re.findall(r"[a-z'-]+", low) if w in ENERGY)
    if energy:
        score += min(energy, 4) * 1.0
        hits.append("energy words")
    return score, hits


def heuristic_candidates(sentences: list[dict], min_dur: float, max_dur: float,
                         exclude: Optional[list[tuple[float, float]]] = None,
                         limit: int = 30) -> list[dict[str, Any]]:
    """Score sliding sentence windows and return the best non-overlapping ones."""
    n = len(sentences)
    if n == 0:
        return []
    exclude = exclude or []
    per_sentence = [score_text(s["text"]) for s in sentences]
    prefix = [0.0]
    for sc, _ in per_sentence:
        prefix.append(prefix[-1] + sc)
    target = min(max_dur, max(min_dur, (min_dur + max_dur) / 2))

    windows: dict[tuple[int, int], None] = {}
    for i in range(n):
        for goal in (min_dur, target):
            j = i
            while j < n - 1 and sentences[j]["end"] - sentences[i]["start"] < goal:
                j += 1
            d = sentences[j]["end"] - sentences[i]["start"]
            if min_dur * 0.9 <= d <= max_dur:
                windows[(i, j)] = None

    scored: list[dict[str, Any]] = []
    for (i, j) in windows:
        s, e = sentences[i]["start"], sentences[j]["end"]
        dur = e - s
        if any(_overlap((s, e), ex) > 0 for ex in exclude):
            continue
        # hook: text spoken in the first 5 seconds
        hook_text = ""
        for k in range(i, j + 1):
            if sentences[k]["start"] - s > 5:
                break
            hook_text += " " + sentences[k]["text"]
        hook_score, hook_hits = score_text(hook_text)
        first = sentences[i]["text"].lower().strip()
        opener = any(first.startswith(h) for h in HOOK_OPENERS)
        body = (prefix[j + 1] - prefix[i]) / max(dur / 30.0, 1.0)
        n_words = sum(len(sentences[k]["words"]) for k in range(i, j + 1))
        rate = n_words / max(dur, 1.0)
        pace = 1.5 if 2.2 <= rate <= 4.0 else (-2.0 if rate < 1.4 else 0.0)
        raw = hook_score * 2.5 + body + pace + (3.0 if opener else 0.0)
        if first.split()[:1] and first.split()[0].strip(",.") in ("and", "but", "so", "because", "or", "then"):
            raw -= 2.0  # starts mid-thought
        if not sentences[j]["text"].rstrip().endswith((".", "!", "?", "…")):
            raw -= 0.5
        reasons = []
        if hook_hits:
            reasons.append("hook with " + ", ".join(dict.fromkeys(hook_hits[:3])))
        if opener:
            reasons.append("strong opening line")
        if prefix[j + 1] - prefix[i] > 6:
            reasons.append("dense with power keywords")
        if pace > 0:
            reasons.append("energetic pace")
        scored.append({"i": i, "j": j, "start": s, "end": e, "raw": raw,
                       "reason": "; ".join(reasons) or "steady, self-contained segment"})

    if not scored:
        return []
    raws = sorted(c["raw"] for c in scored)
    lo, hi = raws[len(raws) // 2], raws[-1]
    for c in scored:
        frac = 0.0 if hi <= lo else max(0.0, (c["raw"] - lo) / (hi - lo))
        c["score"] = round(max(1.0, min(10.0, 3.0 + 7.0 * frac)), 1)

    scored.sort(key=lambda c: c["raw"], reverse=True)
    picked: list[dict[str, Any]] = []
    for c in scored:
        if all(_overlap((c["start"], c["end"]), (p["start"], p["end"])) == 0 for p in picked):
            picked.append(c)
        if len(picked) >= limit:
            break
    for c in picked:
        c["source"] = "heuristic"
    return picked


# =========================================================================== #
# Local LLM (vLLM OpenAI-compatible server on localhost)
# =========================================================================== #
def llm_status() -> dict[str, Any]:
    """Check whether the local vLLM server is reachable. Never raises."""
    if LLM_DISABLED:
        return {"available": False, "model": "", "reason": "disabled via DISABLE_LLM=1"}
    try:
        r = requests.get(f"{LLM_BASE_URL}/models", timeout=3)
        r.raise_for_status()
        models = r.json().get("data", [])
        name = LLM_MODEL or (models[0]["id"] if models else "")
        if not name:
            return {"available": False, "model": "", "reason": "server has no models loaded"}
        _llm_model_cache["name"] = name
        return {"available": True, "model": name, "reason": ""}
    except Exception as exc:
        return {"available": False, "model": "", "reason": str(exc)[:160]}


SYSTEM_PROMPT = (
    "You are an expert short-form video editor who finds the most viral moments in long videos "
    "for TikTok, YouTube Shorts and Reels. You reply with ONLY valid JSON, no prose, no markdown."
)


def _user_prompt(chunk_text: str, video_title: str, k: int, min_dur: int, max_dur: int) -> str:
    return (
        f"Video title: {video_title}\n"
        f"Below is part of the transcript. Each line starts with its start time in seconds: [123] text.\n\n"
        f"{chunk_text}\n\n"
        f"Pick up to {k} of the most viral, self-contained moments (strong hook in the first 5 seconds, a clear "
        f"story/insight/emotion, a satisfying ending). Each clip must last between {min_dur} and {max_dur} seconds, "
        f"start at the beginning of a sentence and end at the end of a sentence. Use ONLY timestamps that appear above "
        f"(end = the start time of the line AFTER the last line you want, or a bit later). Clips must not overlap.\n"
        f"Return ONLY a JSON array in exactly this format:\n"
        f'[{{"start": 120, "end": 175, "title": "short viral title (max 60 chars)", "score": 9, '
        f'"reason": "why it is viral", "description": "1-2 sentence social caption", '
        f'"hashtags": ["tag1","tag2","tag3","tag4","tag5"]}}]\n'
        f"score is an integer 1-10. JSON array only."
    )


def _extract_json_array(raw: str) -> list[dict[str, Any]]:
    """Robustly pull a JSON array of objects out of an LLM reply."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    a, b = text.find("["), text.rfind("]")
    if a != -1 and b > a:
        try:
            data = json.loads(text[a:b + 1])
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass
    out = []
    for m in re.finditer(r"\{[^{}]*\}", text, flags=re.S):  # salvage individual objects (truncated output)
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            continue
    if not out:
        raise ValueError("LLM reply contained no parsable JSON")
    return out


def _chat(model: str, messages: list[dict[str, str]]) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1800,
        "chat_template_kwargs": {"enable_thinking": False},  # Qwen3: no <think> block
    }
    r = requests.post(f"{LLM_BASE_URL}/chat/completions", json=payload, timeout=LLM_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"] or ""


def _chunk_sentences(sentences: list[dict], chunk_chars: int) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    size = 0
    for s in sentences:
        line_len = len(s["text"][:400]) + 8
        if cur and size + line_len > chunk_chars:
            chunks.append(cur)
            cur = cur[-6:]  # small overlap so moments on the boundary are not lost
            size = sum(len(x["text"][:400]) + 8 for x in cur)
        cur.append(s)
        size += line_len
    if cur:
        chunks.append(cur)
    max_chunks = 16
    if len(chunks) > max_chunks:  # very long video: sample evenly
        step = len(chunks) / max_chunks
        chunks = [chunks[int(i * step)] for i in range(max_chunks)]
    return chunks


def llm_candidates(sentences: list[dict], num_clips: int, min_dur: int, max_dur: int,
                   video_title: str, progress: ProgressCb) -> list[dict[str, Any]]:
    """Ask the local LLM for clip candidates, chunk by chunk. Raises if nothing usable came back."""
    status = llm_status()
    if not status["available"]:
        raise RuntimeError(f"LLM unavailable: {status['reason']}")
    model = status["model"]
    chunks = _chunk_sentences(sentences, LLM_CHUNK_CHARS)
    k = min(num_clips + 2, max(2, math.ceil(num_clips * 1.5 / len(chunks)) + 1))
    log.info("LLM analysis", extra={"model": model, "chunks": len(chunks), "per_chunk": k})
    results: list[dict[str, Any]] = []
    failures = 0
    for idx, chunk in enumerate(chunks):
        progress(idx / len(chunks), f"AI analysing transcript (part {idx + 1}/{len(chunks)})…")
        text = "\n".join(f"[{int(s['start'])}] {s['text'][:400]}" for s in chunk)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _user_prompt(text, video_title, k, min_dur, max_dur)}]
        parsed: list[dict[str, Any]] = []
        for attempt in (1, 2):
            try:
                parsed = _extract_json_array(_chat(model, messages))
                break
            except Exception as exc:
                log.warning(f"LLM chunk {idx + 1} attempt {attempt} failed: {exc}")
        if not parsed:
            failures += 1
            if failures >= 2 and not results:
                raise RuntimeError("LLM failed repeatedly; switching to heuristic engine")
            continue
        for item in parsed:
            try:
                s, e = float(item["start"]), float(item["end"])
                if e <= s:
                    continue
                score = float(item.get("score", 6))
                results.append({
                    "start": s, "end": e,
                    "score": max(1.0, min(10.0, score)),
                    "title": str(item.get("title") or "").strip(),
                    "reason": str(item.get("reason") or "").strip(),
                    "description": str(item.get("description") or "").strip(),
                    "hashtags": item.get("hashtags") if isinstance(item.get("hashtags"), list) else [],
                    "source": "llm",
                })
            except (KeyError, TypeError, ValueError):
                continue
    if not results:
        raise RuntimeError("LLM returned no usable clips")
    progress(1.0, "AI analysis complete")
    return results


# =========================================================================== #
# Public entry point
# =========================================================================== #
def _text_between(sentences: list[dict], start: float, end: float) -> str:
    return " ".join(s["text"] for s in sentences if s["end"] > start + 0.01 and s["start"] < end - 0.01)


def analyze(transcript: dict[str, Any], num_clips: int, min_dur: int, max_dur: int,
            video_title: str, progress: ProgressCb) -> tuple[list[dict[str, Any]], str]:
    """Find the best clips. Returns (clips, method) where method is llm / heuristic / llm+heuristic."""
    sentences = transcript.get("sentences") or []
    if not sentences:
        raise ValueError("Transcript is empty - nothing to analyse.")
    video_end = float(sentences[-1]["end"])
    cap = max(min_dur + 1.0, max_dur - 0.6)  # leaves room for the small audio padding added later

    chosen: list[dict[str, Any]] = []

    def accept(cand: dict[str, Any]) -> bool:
        span = (cand["start"], cand["end"])
        if any(_overlap(span, (c["start"], c["end"])) > 0.25 * (span[1] - span[0]) for c in chosen):
            return False
        chosen.append(cand)
        return True

    # ---- 1. LLM ---------------------------------------------------------------------------
    try:
        raw = llm_candidates(sentences, num_clips, min_dur, max_dur, video_title, progress)
        raw.sort(key=lambda c: c["score"], reverse=True)
        for c in raw:
            try:
                s, e = snap_to_sentences(sentences, c["start"], c["end"], float(min_dur), cap)
            except Exception as exc:
                log.warning(f"snap failed for LLM clip: {exc}")
                continue
            if e - s < min_dur * 0.8:
                continue
            c["start"], c["end"] = s, e
            accept(c)
            if len(chosen) >= num_clips:
                break
        log.info("LLM clips accepted", extra={"count": len(chosen)})
    except Exception as exc:
        log.warning(f"LLM stage skipped, using heuristic engine: {exc}")
        progress(0.5, "Local LLM unavailable - using heuristic engine…")

    # ---- 2. heuristic fallback / top-up ------------------------------------------------------
    n_llm = len(chosen)
    if len(chosen) < num_clips:
        try:
            exclude = [(c["start"], c["end"]) for c in chosen]
            for c in heuristic_candidates(sentences, min_dur, cap, exclude=exclude, limit=num_clips * 3):
                accept(c)
                if len(chosen) >= num_clips:
                    break
        except Exception as exc:
            log.error(f"heuristic engine failed: {exc}", exc_info=True)

    if not chosen:
        # absolute last resort: evenly spaced windows, so the tool still produces output
        log.warning("no candidates - using evenly spaced windows")
        step = max(video_end / max(num_clips, 1), min_dur)
        t = 0.0
        while t < video_end and len(chosen) < num_clips:
            s, e = snap_to_sentences(sentences, t, t + min_dur + 10, float(min_dur), cap)
            accept({"start": s, "end": e, "score": 3.0, "reason": "evenly spaced fallback", "source": "heuristic"})
            t += step

    n_heur = len(chosen) - n_llm
    method = "llm" if n_heur == 0 and n_llm else ("heuristic" if n_llm == 0 else "llm+heuristic")

    # ---- 3. finalise metadata (local, no API) -----------------------------------------------
    chosen.sort(key=lambda c: c["score"], reverse=True)
    clips: list[dict[str, Any]] = []
    for idx, c in enumerate(chosen[:num_clips], start=1):
        start = max(0.0, c["start"] - 0.15)
        end = min(video_end + 0.5, c["end"] + 0.35)
        text = _text_between(sentences, c["start"], c["end"])
        title = (c.get("title") or "").strip() or make_title(text)
        if len(title) > 80:
            title = title[:80].rsplit(" ", 1)[0] + "…"
        clips.append({
            "id": f"clip_{idx:02d}",
            "index": idx,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "title": title,
            "description": (c.get("description") or "").strip() or make_description(text),
            "hashtags": make_hashtags(f"{title} {text}", c.get("hashtags")),
            "score": round(float(c["score"]), 1),
            "reason": c.get("reason") or "Selected for hook strength and completeness",
            "source": c.get("source", "heuristic"),
        })
    log.info("analysis complete", extra={"method": method, "clips": len(clips)})
    progress(1.0, f"Selected {len(clips)} clips ({method})")
    return clips, method
