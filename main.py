"""Tone autocorrect: a cascade router over NVIDIA Nemotron models on Nebius Token Factory.

Flow per message:
  Nano   -> fast tone check (every call)
  Super  -> writes rewrites (only when Nano flags the text as risky)
  Ultra  -> deep analysis (only when Nano is unsure, or the user asks for it)
"""
import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
API_KEY = os.getenv("NEBIUS_API_KEY", "")
MODEL_NANO = os.getenv("MODEL_NANO", "")
MODEL_SUPER = os.getenv("MODEL_SUPER", "")
MODEL_ULTRA = os.getenv("MODEL_ULTRA", "")

RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.45"))  # below this: no rewrite
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))  # below this: escalate to Ultra
MIN_WORDS = 3
CACHE_SIZE = 256

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY or "missing")
app = FastAPI(title="Tone Autocorrect")

TONES = {
    "friendly": "warm, relaxed and approachable",
    "professional": "polite, clear and workplace-appropriate",
    "firm": "direct and confident, but respectful, with no hostility",
    "empathetic": "caring and considerate of the reader's feelings",
    "casual": "chill, conversational, and texting-friendly with natural everyday phrasing or light internet shorthand, zero corporate stiffness, without forcing cringey slang",
}

CLASSIFY_SYSTEM = """You are a tone checker for written messages. Judge how the message will LAND with the reader, not its spelling.
Reply with ONLY a JSON object, no prose, in this exact shape:
{"tone": "fine|blunt|passive_aggressive|vague|overly_apologetic|harsh|unclear_ask",
 "risk": 0.0-1.0,
 "confidence": 0.0-1.0,
 "issues": [{"phrase": "exact substring copied from the message", "problem": "max 8 words"}]}
risk = chance the message is misread or upsets the reader. confidence = how sure you are of this judgement.
Use at most 3 issues. Each phrase must be copied exactly from the message. If the message is fine, use tone "fine", low risk and no issues."""

REWRITE_SYSTEM = """You rewrite messages so they land well while keeping the sender's intent and every fact (names, dates, numbers).
Reply with ONLY a JSON object, no prose:
{"options": [{"label": "2-3 word label", "text": "the full rewritten message"}]}
Give exactly 3 options that differ in approach (for example: softer, more direct, more concise). Keep each close in length to the original. Never add information the sender did not give. Never use em dashes (—), en dashes (–), or semicolons (;) in your rewrites; use periods or commas instead."""

DEEP_SYSTEM = """You are an expert in workplace and personal communication. Analyse how the message will land, including sarcasm, implied meaning, vagueness and context.
Reply with ONLY a JSON object, no prose:
{"tone": "fine|blunt|passive_aggressive|vague|overly_apologetic|harsh|unclear_ask",
 "risk": 0.0-1.0,
 "confidence": 0.0-1.0,
 "explanation": "1-2 sentences on how the reader is likely to feel and why",
 "issues": [{"phrase": "exact substring copied from the message", "problem": "max 8 words"}],
 "options": [{"label": "2-3 word label", "text": "full rewritten message"}]}
Give exactly 3 options that keep the sender's intent and every fact. If the message is already fine, return an empty options list. Never use em dashes (—), en dashes (–), or semicolons (;) in your rewrites; use periods or commas instead."""


class AnalyzeRequest(BaseModel):
    text: str
    target_tone: str = "professional"
    context: str = ""  # optional earlier messages in the thread
    deep: bool = False


_cache: "OrderedDict[tuple, dict]" = OrderedDict()


def _strip_and_parse(raw: str) -> dict:
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        raise ValueError("model did not return JSON")
    return json.loads(match.group(0))


async def call_model(model: str, system: str, user: str, max_tokens: int, temperature: float):
    if not API_KEY or not model:
        raise HTTPException(500, "Set NEBIUS_API_KEY and MODEL_NANO / MODEL_SUPER / MODEL_ULTRA in .env")
    started = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        data = _strip_and_parse(resp.choices[0].message.content)
    except HTTPException:
        raise
    except Exception as exc:  # network, auth, or bad JSON
        raise HTTPException(502, f"Model call failed: {exc}") from exc
    tokens = resp.usage.total_tokens if resp.usage else 0
    return data, tokens, round(time.perf_counter() - started, 2)


def _clean_issues(issues, text: str) -> list:
    """Keep only issues whose phrase really appears in the text, so highlights never break."""
    out = []
    for item in issues or []:
        phrase = str(item.get("phrase", "")).strip()
        if phrase and phrase in text:
            out.append({"phrase": phrase, "problem": str(item.get("problem", ""))[:80]})
    return out[:3]


def _sanitize_punctuation(text: str) -> str:
    # Replace em/en dashes and spaced double-hyphens with a comma
    text = re.sub(r"\s*(?:—|–|--)\s*", ", ", text)
    # Replace semicolons with a period and capitalize the next letter
    text = re.sub(
        r";\s*([a-zA-Z])",
        lambda m: f". {m.group(1).upper()}",
        text,
    )
    text = text.replace(";", ".")
    # Collapse any accidental double commas or comma-periods
    return re.sub(r",\s*([,.])", r"\1", text)


def _clean_options(options) -> list:
    out = []
    for opt in options or []:
        label = str(opt.get("label", "")).strip()
        body = _sanitize_punctuation(str(opt.get("text", "")).strip())
        if body:
            out.append({"label": label or "Rewrite", "text": body})
    return out[:3]



def _num(value, default):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


async def run_deep(req: AnalyzeRequest, text: str, tone_desc: str, steps: list, tokens_total: int):
    user = f"Target tone: {tone_desc}\n"
    if req.context.strip():
        user += f"Earlier in the thread:\n{req.context.strip()[:3000]}\n\n"
    user += f"Message to analyse:\n{text}"
    data, tokens, secs = await call_model(MODEL_ULTRA, DEEP_SYSTEM, user, 1500, 0.4)
    steps.append({"model": "ultra", "seconds": secs, "tokens": tokens})
    return {
        "tone": str(data.get("tone", "fine")),
        "risk": _num(data.get("risk"), 0.5),
        "confidence": _num(data.get("confidence"), 0.8),
        "explanation": str(data.get("explanation", ""))[:400],
        "issues": _clean_issues(data.get("issues"), text),
        "options": _clean_options(data.get("options")),
    }, tokens_total + tokens


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    tone_desc = TONES.get(req.target_tone, TONES["professional"])
    empty = {"tone": "fine", "risk": 0.0, "confidence": 1.0, "explanation": "", "issues": [], "options": [],
             "steps": [], "tokens": 0, "cached": False}

    if len(text.split()) < MIN_WORDS:
        return empty

    key = (text, req.target_tone, req.context.strip(), req.deep)
    if key in _cache:
        _cache.move_to_end(key)
        return {**_cache[key], "cached": True}

    steps, tokens_total = [], 0

    if req.deep:
        result, tokens_total = await run_deep(req, text, tone_desc, steps, tokens_total)
    else:
        # 1) Nano: fast check
        data, tokens, secs = await call_model(
            MODEL_NANO,
            CLASSIFY_SYSTEM,
            f"Target tone: {tone_desc}\nMessage:\n{text}",
            500,
            0.1,
        )
        steps.append({"model": "nano", "seconds": secs, "tokens": tokens})
        tokens_total += tokens
        result = {
            "tone": str(data.get("tone", "fine")),
            "risk": _num(data.get("risk"), 0.0),
            "confidence": _num(data.get("confidence"), 1.0),
            "explanation": "",
            "issues": _clean_issues(data.get("issues"), text),
            "options": [],
        }
        # 2) Not sure? escalate to Ultra. Risky? ask Super for rewrites. Otherwise stop here.
        if result["confidence"] < CONFIDENCE_THRESHOLD and result["risk"] >= 0.25:
            result, tokens_total = await run_deep(req, text, tone_desc, steps, tokens_total)
        elif result["risk"] >= RISK_THRESHOLD:
            problems = "; ".join(f'"{i["phrase"]}" ({i["problem"]})' for i in result["issues"]) or result["tone"]
            user = f"Target tone: {tone_desc}\nProblems found: {problems}\n\nMessage:\n{text}"
            data, tokens, secs = await call_model(MODEL_SUPER, REWRITE_SYSTEM, user, 700, 0.6)
            steps.append({"model": "super", "seconds": secs, "tokens": tokens})
            tokens_total += tokens
            result["options"] = _clean_options(data.get("options"))

    final = {**result, "steps": steps, "tokens": tokens_total, "cached": False}
    _cache[key] = final
    if len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return final


@app.get("/api/health")
async def health():
    return {
        "ok": bool(API_KEY and MODEL_NANO and MODEL_SUPER and MODEL_ULTRA),
        "base_url": BASE_URL,
        "models": {"nano": MODEL_NANO, "super": MODEL_SUPER, "ultra": MODEL_ULTRA},
    }


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")
