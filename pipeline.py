#!/usr/bin/env python3
"""
pipeline.py — YouTube Auto-Upload Pipeline

Full automation:
  1. Fetch today's trending videos (YouTube Data API)
  2. Claude or OpenAI picks the best trend and writes a structured video script
  3. video_maker.py renders the script into an MP4
  4. youtube_uploader.py uploads it to your channel

Usage:
  python3 pipeline.py              # run once right now
  python3 pipeline.py --schedule   # run daily at SCHEDULE_HOUR (set in .env)
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# ── Load .env before any other import ────────────────────────────────────────
def _load_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key   = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value

_load_dotenv()

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from video_maker      import make_video_from_script
from youtube_uploader import get_authenticated_service, upload_video
from quality_gate     import run_quality_gate, run_script_quality_gate
from ai_budget        import format_budget_check, run_ai_budget_checks

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_FILE = Path(__file__).parent / "pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(_LOG_FILE)),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Config from .env ──────────────────────────────────────────────────────────
YT_API_KEY    = os.environ.get("YOUTUBE_API_KEY", "")
CHANNEL_ID    = os.environ.get("YOUTUBE_CHANNEL_ID", "")
CLAUDE_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_KEY    = os.environ.get("OPENAI_API_KEY", "")
AI_PROVIDER   = os.environ.get("AI_PROVIDER", "auto").strip().lower()
OPENAI_MODEL  = os.environ.get("OPENAI_MODEL", "gpt-5.5")
OPENAI_API_MODE = os.environ.get("OPENAI_API_MODE", "responses").strip().lower()
OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "high").strip().lower()
OPENAI_VERBOSITY = os.environ.get("OPENAI_VERBOSITY", "high").strip().lower()
OPENAI_MAX_OUTPUT_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "50000"))
CLAUDE_MODEL  = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
REGION        = os.environ.get("DEFAULT_REGION", "US")
MAX_RESULTS   = int(os.environ.get("MAX_RESULTS", "25"))
PRIVACY       = os.environ.get("YOUTUBE_PRIVACY", "public")
SCHEDULE_HOUR = int(os.environ.get("SCHEDULE_HOUR", "9"))
VIDEO_RENDERER = os.environ.get("VIDEO_RENDERER", "moviepy").strip().lower()
QUALITY_GATE_MODE = os.environ.get("QUALITY_GATE_MODE", "block").strip().lower()
QUALITY_GATE_THRESHOLD = int(os.environ.get("QUALITY_GATE_THRESHOLD", "78"))
SCRIPT_REWRITE_PASSES = int(os.environ.get("SCRIPT_REWRITE_PASSES", "1"))
AI_CREDIT_INFO = os.environ.get("AI_CREDIT_INFO", "on").strip().lower()
SHORTS_MODE   = os.environ.get("SHORTS_MODE", "also").strip().lower()  # off | also | only

# ── Free AI provider keys (no credits required) ──────────────────────────────
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "").strip()
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "").strip()
CEREBRAS_API_KEY  = os.environ.get("CEREBRAS_API_KEY", "").strip()
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY", "").strip()
HF_TOKEN          = os.environ.get("HF_TOKEN", "").strip()
GROQ_MODEL        = os.environ.get("GROQ_MODEL",      "llama-3.3-70b-versatile").strip()
GEMINI_MODEL      = os.environ.get("GEMINI_MODEL",    "gemini-2.5-flash-lite").strip()
CEREBRAS_MODEL    = os.environ.get("CEREBRAS_MODEL",  "gpt-oss-120b").strip()   # check available: GET api.cerebras.ai/v1/models
SAMBANOVA_MODEL   = os.environ.get("SAMBANOVA_MODEL", "Meta-Llama-3.3-70B-Instruct").strip()


# ── Step 1: Fetch trending videos ─────────────────────────────────────────────

def fetch_trending_youtube(n: int = 25) -> list[dict]:
    """Primary source: YouTube Data API v3."""
    yt = build("youtube", "v3", developerKey=YT_API_KEY, cache_discovery=False)
    resp = yt.videos().list(
        part="snippet,statistics",
        chart="mostPopular",
        regionCode=REGION,
        maxResults=n,
    ).execute()
    videos = []
    for v in resp.get("items", []):
        sn = v["snippet"]
        st = v.get("statistics", {})
        videos.append({
            "title":    sn.get("title", ""),
            "channel":  sn.get("channelTitle", ""),
            "tags":     sn.get("tags", [])[:8],
            "views":    int(st.get("viewCount",   0)),
            "likes":    int(st.get("likeCount",   0)),
            "comments": int(st.get("commentCount", 0)),
        })
    return videos


def fetch_trending_google() -> list[dict]:
    """Fallback source: Google Trends (no API key required)."""
    from pytrends.request import TrendReq
    pt = TrendReq(hl="en-US", tz=0, timeout=(10, 30))
    country_map = {
        "US": "united_states", "GB": "united_kingdom", "CA": "canada",
        "AU": "australia",     "IN": "india",           "PH": "philippines",
        "SG": "singapore",     "MY": "malaysia",
    }
    pn  = country_map.get(REGION, "united_states")
    df  = pt.trending_searches(pn=pn)
    terms = df[0].tolist()[:25]
    return [{"title": t, "channel": "Google Trends", "tags": [],
             "views": 0, "likes": 0, "comments": 0} for t in terms]


def fetch_trending(n: int = 25) -> list[dict]:
    """Try YouTube API first; fall back to Google Trends automatically."""
    if YT_API_KEY:
        try:
            videos = fetch_trending_youtube(n)
            log.info("  Source: YouTube Data API v3")
            return videos
        except HttpError as e:
            log.warning(f"YouTube API failed ({e.reason}) — switching to Google Trends")
        except Exception as e:
            log.warning(f"YouTube API error ({e}) — switching to Google Trends")

    log.info("  Source: Google Trends (fallback)")
    return fetch_trending_google()


# ── Step 2: Generate structured script via Claude or OpenAI ───────────────────

def _strip_json(raw: str) -> str:
    """Accept perfect JSON, fenced JSON, or extra model chatter around JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.strip().startswith("json"):
            raw = raw.strip()[4:]
        raw = raw.strip()
    if not raw.startswith("{"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start:end + 1]
    return raw


def _repair_json(raw: str) -> str:
    """
    Repair common LLM truncation artefacts so json.loads can parse the result.

    Handles:
    - Unterminated strings (LLM hit token limit mid-value)
    - Missing closing brackets / braces (truncated response)
    - Trailing commas before ] or } (some models add them)
    - Incomplete last key-value pair (drop it so the object stays valid)
    """
    import re as _re

    # 1. Remove trailing commas before ] or }
    raw = _re.sub(r',\s*([\]}])', r'\1', raw)

    # 2. Walk character-by-character to detect open state
    stack: list[str] = []
    in_string   = False
    escape_next = False
    last_complete_obj_end = -1   # position of last '}' that closed the root object

    for i, ch in enumerate(raw):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}':
            if stack and stack[-1] == '{':
                stack.pop()
                if not stack:          # root object just closed
                    last_complete_obj_end = i
        elif ch == ']':
            if stack and stack[-1] == '[':
                stack.pop()

    # 3. If the string is truncated inside a string literal, close it
    if in_string:
        raw = raw.rstrip() + '"'

    # 4. Drop a dangling key without a value at the very end (common truncation)
    #    e.g.  ..."last_key": <truncated>  → remove the whole dangling entry
    #    Only do this if we still have unclosed brackets to add.
    if stack:
        raw = raw.rstrip().rstrip(',')

    # 5. Close any unclosed brackets/braces in reverse order
    closers = {'{': '}', '[': ']'}
    for opener in reversed(stack):
        raw += closers.get(opener, '')

    return raw


def _parse_json_safe(raw: str) -> dict:
    """
    Parse LLM output as JSON with two automatic fallbacks:
      1. Standard json.loads (perfect response)
      2. _repair_json + json.loads (truncated / malformed response)
      3. Extract largest valid sub-object (partially valid response)
    Raises json.JSONDecodeError only after all three attempts fail.
    """
    stripped = _strip_json(raw)

    # Attempt 1: clean parse
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Attempt 2: repair then parse
    try:
        repaired = _repair_json(stripped)
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Attempt 3: find the largest { … } block that parses cleanly
    start = stripped.find("{")
    while start >= 0:
        end = stripped.rfind("}")
        while end > start:
            try:
                return json.loads(stripped[start:end + 1])
            except json.JSONDecodeError:
                end = stripped.rfind("}", start, end)
        start = stripped.find("{", start + 1)

    # All attempts failed — raise so the caller can try the next provider
    raise json.JSONDecodeError(
        "Could not parse or repair JSON from LLM response", stripped, 0
    )


# Free-provider availability helpers
def _has_groq()      -> bool: return bool(GROQ_API_KEY)
def _has_gemini()    -> bool: return bool(GEMINI_API_KEY)
def _has_cerebras()  -> bool: return bool(CEREBRAS_API_KEY)
def _has_sambanova() -> bool: return bool(SAMBANOVA_API_KEY)
def _has_hf()        -> bool: return bool(HF_TOKEN)

def _choose_ai_provider() -> str:
    """
    Return the active AI provider string.

    Priority when AI_PROVIDER=auto:
      Free tier (no credits): cerebras → groq → sambanova → gemini → hf
      Paid:                   openai   → anthropic
    This lets the pipeline run at zero cost when paid keys are absent.
    """
    provider = (AI_PROVIDER or "auto").lower()

    # Explicit paid provider
    if provider in ("openai", "chatgpt", "gpt"):
        if not OPENAI_KEY:
            raise RuntimeError("AI_PROVIDER=openai but OPENAI_API_KEY is missing")
        return "openai"
    if provider in ("anthropic", "claude"):
        if not CLAUDE_KEY:
            raise RuntimeError("AI_PROVIDER=anthropic but ANTHROPIC_API_KEY is missing")
        return "anthropic"

    # Explicit free providers
    if provider == "groq"      and _has_groq():      return "groq"
    if provider == "gemini"    and _has_gemini():    return "gemini"
    if provider == "cerebras"  and _has_cerebras():  return "cerebras"
    if provider == "sambanova" and _has_sambanova(): return "sambanova"
    if provider == "hf"        and _has_hf():        return "hf"

    # auto — free first, then paid
    if _has_cerebras():  return "cerebras"
    if _has_groq():      return "groq"
    if _has_sambanova(): return "sambanova"
    if _has_gemini():    return "gemini"
    if _has_hf():        return "hf"
    if OPENAI_KEY:       return "openai"
    if CLAUDE_KEY:       return "anthropic"
    raise RuntimeError(
        "No AI key found. Set at least one of: GROQ_API_KEY, GEMINI_API_KEY, "
        "CEREBRAS_API_KEY, SAMBANOVA_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env"
    )


# ── System-prompt constants ───────────────────────────────────────────────────
# These prime the model's creative persona before the main user prompt.
# A rich system prompt is the single highest-leverage improvement for output
# quality because the Responses API burns it into persistent context.

_SYSTEM_PROMPT_WRITER = """\
You are the showrunner and creative director of a YouTube channel on track for 10 million subscribers.

Your mandatory creative standards — non-negotiable on every output:
• MrBeast-level promise clarity, escalation discipline, and mini-payoff density. No section ends without a concrete reward the viewer did not have before.
• Zach King visual logic: every scene earns one impossible-feeling or counterintuitive reveal. Visual surprise is information, never decoration.
• Veritasium / Johnny Harris narrative tension: the viewer never knows what to believe until the proof lands. Evidence beats claim every time.
• 2026 creator discipline: human stakes beat effects. Gravity beats speed. Specificity beats abstraction.

PSYCHOLOGICAL ARCHITECTURE — mandatory on every Shorts output:

ZEIGARNIK EFFECT (non-negotiable):
• The hook MUST open an information loop that the brain physically cannot close on its own.
• That loop must remain OPEN through 80% of the narration — it is ONLY resolved in the final 20%.
• Any hook that gives away the answer before the 80% mark is a failed hook — rewrite it.
• Use nested loops: Open Loop A in the hook → Open Loop B midway → Close B → Open Loop C → Close A and C simultaneously at the end.

LOSS-AVERSION FRAMING (mandatory):
• Frame the hook with what the viewer is LOSING or RISKING — NOT what they could gain.
• Loss language is 2.5× more powerful than gain language. "You're bleeding $X/month" beats "Make $X/month."
• The specific number, risk, or consequence must appear in the first 10 words.

EMOTIONAL ESCALATION LADDER (mandatory):
• Narration must escalate through this exact ladder: Curiosity → Surprise → Tension → Relief → Curiosity (repeat).
• Never stay on one emotional note for more than 3 sentences.
• The escalation must feel earned — each rung builds on the previous.

SOCIAL IDENTITY MIRROR (mandatory):
• Include one sentence where the viewer says "that's exactly me" — hyper-specific to the community.
• Use the community's own vocabulary: in-group terms, specific situations, exact dollar amounts, real scenarios.
• Generic community references ("if you hold crypto") are forbidden — must be razor-specific.

COMMENT BAIT (mandatory):
• Include one slightly controversial but low-stakes claim that forces a response from every viewer who disagrees.
• Do NOT use blatant rage-bait. The claim must be defensible but contestable.
• Frame it as a confident assertion, not a question. Questions get ignored; assertions get corrections.

RE-WATCH MECHANIC (mandatory):
• Drop one detail in the first 10 seconds that only makes full sense after hearing the payoff.
• It should be subtle enough that 80% of viewers miss it on first watch.
• This is the "hidden clue" — it makes re-watching feel rewarding, not redundant.

END-SCREEN (mandatory):
• The last 2 sentences must NOT be generic "like and subscribe."
• Instead, open a NEW curiosity gap about a DIFFERENT video — make not watching the next video feel irrational.
• Format: "[Resolve current hook]. [One sentence that makes the NEXT video impossible to skip]."

Before writing a single word, run this internal filter:
• Does every narration sentence sound like a confident human speaking from genuine knowledge? If it sounds AI-written, rewrite it.
• Does every visual_keyword and asset_prompt describe a frame a real person would stop scrolling for? If it is abstract, vague, or generic, replace it with a specific cinematic scene.
• Is the Zeigarnik loop open for the full first 80% of narration — NOT prematurely resolved?
• Does the hook lead with what the viewer is LOSING — not what they could gain?
• Does the emotional ladder escalate Curiosity → Surprise → Tension → Relief within the narration?
• Is there a "that's exactly me" identity mirror moment with hyper-specific community language?
• Does the end screen open a curiosity gap for the NEXT video, not just ask for a subscribe?
• Are all numbers, names, examples, and outcomes specific and concrete? Vague claims are deleted.

Return ONLY valid, complete JSON. Never emit markdown fences, partial JSON, or any explanation text before or after the JSON object.\
"""

_SYSTEM_PROMPT_DIRECTOR = """\
You are a ruthless YouTube creative director, retention specialist, and monetization editor with 10M+ subscriber credits.

You are fixing a script that failed a quality gate. Your job is surgical precision:
• Preserve what works. Upgrade only what the quality report flags.
• Narration that sounds human stays. Narration that sounds AI-written gets rewritten from scratch.
• Visual prompts that are abstract, generic, or diagram-like get replaced with specific cinematic frames a human would stop for.
• Every section must earn its runtime. If a section cannot add new information, a new human scenario, or a new mechanism, it is cut and merged.
• Advertiser safety is non-negotiable: informative, non-graphic, non-hateful, non-misleading framing throughout.

Return ONLY the complete, valid JSON object with the same schema. No markdown. No explanation.\
"""

# ─────────────────────────────────────────────────────────────────────────────

def _openai_text_from_response(response) -> str:
    text = getattr(response, "output_text", "") or ""
    if text:
        return text
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(value)
    return "".join(chunks)


def _call_openai_json(prompt: str, system_prompt: str, temperature: float) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Run: pip install openai")

    client = OpenAI(api_key=OPENAI_KEY)
    reasoning_effort = OPENAI_REASONING_EFFORT if OPENAI_REASONING_EFFORT in {
        "none", "minimal", "low", "medium", "high", "xhigh"
    } else "high"
    # gpt-5.x models do not accept temperature — only classic GPT-4 series does
    supports_temperature = not OPENAI_MODEL.startswith("gpt-5")

    # ── Attempt 1: Responses API (preferred — supports reasoning + instructions) ──
    if OPENAI_API_MODE in {"responses", "auto", "best"}:
        base_kwargs: dict = {
            "model":             OPENAI_MODEL,
            "instructions":      system_prompt,
            "input":             prompt,
            "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
            "text":              {"format": {"type": "json_object"}},
            "reasoning":         {"effort": reasoning_effort},
        }
        # Some model variants reject the reasoning block — strip it as fallback
        attempts = [base_kwargs, {k: v for k, v in base_kwargs.items() if k != "reasoning"}]
        if supports_temperature:
            attempts.insert(0, {**base_kwargs, "temperature": temperature})
        for kwargs in attempts:
            try:
                response = client.responses.create(**kwargs)
                raw = _openai_text_from_response(response)
                if raw:
                    return raw
            except Exception as e:
                log.warning(f"OpenAI Responses API attempt failed ({e}); trying next")

    # ── Attempt 2: Chat Completions fallback ──────────────────────────────────
    # NOTE: verbosity is NOT a valid Chat Completions parameter — omit it here.
    # reasoning_effort is valid only for o-series models; strip it on failure.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt},
    ]
    base_kwargs = {
        "model":                  OPENAI_MODEL,
        "response_format":        {"type": "json_object"},
        "max_completion_tokens":  OPENAI_MAX_OUTPUT_TOKENS,
        "reasoning_effort":       reasoning_effort,
        "messages":               messages,
    }
    attempts = [
        base_kwargs,
        {k: v for k, v in base_kwargs.items() if k != "reasoning_effort"},
        {"model": OPENAI_MODEL, "messages": messages},
    ]
    if supports_temperature:
        attempts.insert(0, {**base_kwargs, "temperature": temperature})
    for kwargs in attempts:
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            log.warning(f"OpenAI Chat Completions attempt failed ({e}); trying next")
    raise RuntimeError("OpenAI script generation failed after all Responses and Chat fallback attempts")


def _call_free_provider_json(prompt: str, system_prompt: str,
                              temperature: float = 0.70) -> str:
    """
    Call free-tier OpenAI-compatible providers in waterfall order.
    Tries: cerebras → groq → sambanova → gemini → hf
    Automatically skips any provider whose key is missing or whose call fails
    (model-not-found, rate-limit, network error, etc.).
    Returns raw JSON string from the first successful provider.
    """
    if OpenAI is None:
        raise RuntimeError("openai package is not installed — run: pip install openai")

    # Ordered list of (name, api_key, base_url, model, max_tokens)
    # Only entries with a non-empty api_key are tried.
    _FREE_PROVIDERS = [
        ("cerebras",  CEREBRAS_API_KEY,  "https://api.cerebras.ai/v1",
         CEREBRAS_MODEL,  8192),
        ("groq",      GROQ_API_KEY,      "https://api.groq.com/openai/v1",
         GROQ_MODEL,      32768),
        ("sambanova", SAMBANOVA_API_KEY, "https://api.sambanova.ai/v1",
         SAMBANOVA_MODEL, 8192),
        ("gemini",    GEMINI_API_KEY,
         "https://generativelanguage.googleapis.com/v1beta/openai/",
         GEMINI_MODEL,    32768),
        ("hf",        HF_TOKEN,          "https://router.huggingface.co/v1",
         "openai/gpt-oss-120b:fastest",  8192),
    ]

    # Respect explicit AI_PROVIDER setting — put it first if specified
    chosen = _choose_ai_provider()
    if chosen in ("cerebras", "groq", "sambanova", "gemini", "hf"):
        _FREE_PROVIDERS.sort(key=lambda x: 0 if x[0] == chosen else 1)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt},
    ]

    last_err: Exception | None = None
    for name, api_key, base_url, model, max_tok in _FREE_PROVIDERS:
        if not api_key:
            continue  # key not configured — skip silently
        log.info(f"  LLM → {name} ({model})")
        client = OpenAI(api_key=api_key, base_url=base_url)
        # Try with json_object response format first; some providers don't support it
        for use_json_fmt in (True, False):
            kwargs: dict = {
                "model":      model,
                "messages":   messages,
                "max_tokens": min(max_tok, 8192),
                "temperature": temperature,
            }
            if use_json_fmt:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                response = client.chat.completions.create(**kwargs)
                content  = response.choices[0].message.content or ""
                if content.strip():
                    # Validate JSON before returning — truncated responses cause
                    # an "Unterminated string" crash in the caller. If we can
                    # repair it here, great; otherwise fall through to next provider.
                    try:
                        _parse_json_safe(content)
                        return content   # valid (or repairable) — use it
                    except json.JSONDecodeError:
                        log.warning(
                            f"  {name} returned truncated/malformed JSON "
                            f"({len(content)} chars) — trying next provider"
                        )
                        last_err = json.JSONDecodeError(
                            f"{name} response is not valid JSON", content, 0
                        )
                        break   # skip to next provider
            except Exception as e:
                err_str = str(e).lower()
                if "temperature" in err_str:
                    # Provider rejects temperature — retry without it
                    kwargs.pop("temperature", None)
                    try:
                        response = client.chat.completions.create(**kwargs)
                        content  = response.choices[0].message.content or ""
                        if content.strip():
                            return content
                    except Exception as e2:
                        last_err = e2
                        break   # try next provider
                elif use_json_fmt and ("response_format" in err_str or "json" in err_str):
                    last_err = e
                    continue    # retry this provider without response_format
                else:
                    log.warning(f"  {name} failed: {e} — trying next provider")
                    last_err = e
                    break       # try next provider

    raise RuntimeError(
        f"All free AI providers failed or have no key configured. "
        f"Last error: {last_err}. "
        f"Set at least one of: CEREBRAS_API_KEY, GROQ_API_KEY, SAMBANOVA_API_KEY, "
        f"GEMINI_API_KEY, HF_TOKEN in .env"
    )


def _validate_script(script: dict) -> dict:
    required = ["video_title", "hook", "thumbnail_concept", "description", "tags", "sections"]
    missing = [key for key in required if key not in script]
    if missing:
        raise ValueError(f"Generated script missing keys: {', '.join(missing)}")
    if not isinstance(script.get("sections"), list) or not script["sections"]:
        raise ValueError("Generated script has no sections")
    script.setdefault("creator_standard", {
        "title_promise": script.get("video_title", ""),
        "first_8_seconds_visual": script.get("thumbnail_concept", ""),
        "payoff_ladder": "Each section resolves one mini-question and opens the next.",
        "freshness_rule": "Use a distinct visual treatment from recent uploads.",
    })
    # Default new psychological fields if LLM omitted them
    script.setdefault("comment_bait", "")
    script.setdefault("zeigarnik_map", "")
    script.setdefault("emotional_arc", "curiosity → surprise → tension → relief")
    script.setdefault("identity_mirror", "")
    script.setdefault("rewatch_clue", "")
    script.setdefault("end_screen_hook", "")
    _SCENE_STYLES = [
        "cold_open", "human_action", "documentary", "evidence",
        "split_screen", "timeline", "blueprint", "action_steps", "finale", "cinematic",
    ]
    _COLOR_MOODS = ["curiosity", "danger", "evidence", "mechanism", "action", "payoff"]
    _RETENTION_DEVICES = [
        "open loop", "fakeout", "countdown", "proof drop", "reversal",
        "mystery object", "challenge", "bet", "experiment", "comparison",
    ]
    _VIEWER_QUESTIONS = [
        "What would I do in this situation?",
        "Is this actually true?",
        "How does this actually work?",
        "What happens next?",
        "Could this happen to me?",
        "What's the catch?",
        "Why haven't I heard this before?",
        "Who figured this out?",
    ]
    _CONTRAST_PAIRS = [
        "expected reality vs actual reality",
        "what you think vs what you get",
        "before vs after the reveal",
        "public knowledge vs hidden truth",
        "expert claim vs real-world result",
        "popular belief vs proven fact",
        "simple surface vs complex truth",
        "slow start vs explosive payoff",
    ]
    _MOTION_CUES = [
        "camera push-in with documentary energy and human-driven action",
        "slow zoom-out revealing full scale of the situation",
        "whip-pan cut to the key evidence or reaction",
        "static locked-off framing broken by sudden movement",
        "handheld close-up that snaps to wide for context",
        "aerial pull-back transitioning to eye-level human moment",
    ]
    _VISUAL_BEATS_POOL = [
        ["establish the human moment", "reveal the surprising detail", "pay off with a clear visual change"],
        ["open on tension", "cut to the proof", "close on the consequence"],
        ["show the before state", "introduce the disruption", "demonstrate the new reality"],
        ["tight on the face/hands", "pull back to full context", "freeze on the key frame"],
        ["montage setup", "single evidence shot", "reaction confirms the reveal"],
        ["tease the mystery", "drop the first clue", "land the implication"],
        ["contrast the two worlds", "highlight the pivot point", "commit to the conclusion"],
        ["fast cuts establishing pace", "slow-motion emphasis beat", "sharp cut to next idea"],
    ]
    _PATTERN_INTERRUPTS = [
        "visual reset tied to the strongest new information",
        "sudden cut to black before re-entering with new energy",
        "text card flash that reframes the scene",
        "scale shift — macro detail snapping to wide establishing shot",
        "colour grade change signalling a new phase",
        "audio drop to silence before impact sound",
    ]
    _SOUND_CUES = [
        "subtle impact hit, riser, or silence timed to the reveal",
        "whoosh transition into the next beat",
        "low bass drop building under the key claim",
        "sharp percussive hit on the evidence reveal",
        "musical swell resolving on the payoff beat with chime",
        "ambient room-tone silence before the key word lands",
    ]

    for idx, sec in enumerate(script["sections"], 1):
        for key in ("name", "narration", "callout_text", "visual_keyword", "lower_third"):
            if key not in sec:
                raise ValueError(f"Section {idx} missing {key}")
        i = idx - 1  # 0-based offset for rotation
        sec.setdefault("retention_bridge", "")
        if "scene_style" not in sec:
            sec["scene_style"] = _SCENE_STYLES[i % len(_SCENE_STYLES)]
        if "color_mood" not in sec:
            sec["color_mood"] = _COLOR_MOODS[i % len(_COLOR_MOODS)]
        if "retention_device" not in sec:
            sec["retention_device"] = _RETENTION_DEVICES[i % len(_RETENTION_DEVICES)]

        # Ensure visual_beats has at least 3 items
        beats = sec.get("visual_beats") or []
        if len(beats) < 3:
            fallback = _VISUAL_BEATS_POOL[i % len(_VISUAL_BEATS_POOL)]
            while len(beats) < 3:
                beats.append(fallback[len(beats) % len(fallback)])
            sec["visual_beats"] = beats

        sec.setdefault("human_behavior", "viewer reacts, decides, avoids risk, or takes action")
        sec.setdefault("viewer_question", _VIEWER_QUESTIONS[i % len(_VIEWER_QUESTIONS)])
        sec.setdefault("color_mood", _COLOR_MOODS[i % len(_COLOR_MOODS)])
        sec.setdefault("contrast_pair", _CONTRAST_PAIRS[i % len(_CONTRAST_PAIRS)])
        sec.setdefault("motion_cue", _MOTION_CUES[i % len(_MOTION_CUES)])
        sec.setdefault("visual_metaphor", sec.get("visual_keyword", "cinematic scene"))
        sec.setdefault("first_frame", sec.get("callout_text", sec.get("name", "")))
        sec.setdefault("mini_payoff", sec.get("retention_bridge") or sec.get("callout_text") or sec.get("name", ""))
        sec.setdefault("pattern_interrupt", _PATTERN_INTERRUPTS[i % len(_PATTERN_INTERRUPTS)])
        sec.setdefault("sound_cue", _SOUND_CUES[i % len(_SOUND_CUES)])

        # ── Psychological architecture fields (Session 5) ─────────────────────
        _DEFAULT_EMOTIONS = [
            "shock", "dread", "recognition", "cognitive_dissonance",
            "confusion", "analytical_trust", "intellectual_awe",
            "anxiety", "relief", "belonging",
        ]
        _DEFAULT_LOADS = [
            "light", "heavy", "light", "heavy", "light",
            "heavy", "heavy", "light", "heavy", "light",
        ]
        _DEFAULT_LOOPS = [
            "opens Loop A",
            "sustains Loop A, opens social proof thread",
            "opens Loop B",
            "closes Loop B, sustains Loop A",
            "partially closes Loop A with wrong answer, opens Loop C",
            "sustains Loop C, opens Loop D",
            "sustains Loop C, escalates Loop D",
            "Loop D peaks — consequence becomes viscerally real",
            "closes Loop A and Loop D simultaneously",
            "closes Loop C — social proof validates the solution",
        ]
        sec.setdefault("emotion_target",  _DEFAULT_EMOTIONS[i % len(_DEFAULT_EMOTIONS)])
        sec.setdefault("peak_sentence",   sec.get("callout_text", sec.get("name", "")))
        sec.setdefault("cognitive_load",  _DEFAULT_LOADS[i % len(_DEFAULT_LOADS)])
        sec.setdefault("loop_action",     _DEFAULT_LOOPS[i % len(_DEFAULT_LOOPS)])
        # Section-specific optional fields
        sec.setdefault("fake_out_content",    "")
        sec.setdefault("mid_video_tease",     "")
        sec.setdefault("social_proof_anchor", "")
        sec.setdefault("rewatch_clue",        "")

        # Strengthen asset_prompt: ensure it's specific enough for real image sourcing
        ap = sec.get("asset_prompt", "")
        kw = sec.get("visual_keyword", "cinematic scene")
        cm = sec.get("color_mood", "curiosity")
        _MOOD_LIGHTING = {
            "curiosity":  "warm amber side-lighting",
            "danger":     "harsh red-orange rim lighting",
            "evidence":   "cold blue-white monitor glow",
            "mechanism":  "technical overhead fluorescent, stark shadows",
            "action":     "dynamic golden-hour backlighting",
            "payoff":     "warm sunlit resolution, soft fill light",
        }
        lighting = _MOOD_LIGHTING.get(cm, "dramatic cinematic split-tone lighting")
        if not ap or len(ap.split()) < 12:
            sec["asset_prompt"] = (
                f"Real person reacting to {kw}, genuine emotion, "
                f"{lighting}, "
                "documentary-style handheld camera feel, "
                "shallow depth of field, cinematic color grade, "
                "no text, no watermarks, photorealistic."
            )

    # Enforce tags minimum of 10
    tags = script.get("tags", [])
    if len(tags) < 10:
        fallback_tags = ["viral", "trending", "mustwatch", "facts", "explained",
                         "documentary", "real", "truth", "shocking", "mindblowing",
                         "interesting", "analysis", "breakdown", "revealed", "exposed"]
        tags_set = set(t.lower() for t in tags)
        for candidate in fallback_tags:
            if len(tags) >= 10:
                break
            if candidate.lower() not in tags_set:
                tags.append(candidate)
                tags_set.add(candidate.lower())
        script["tags"] = tags

    return script


def _recent_script_context(limit: int = 8) -> str:
    """Summarize recent generated titles so the next script avoids repetition."""
    out_dir = Path(__file__).parent / "output"
    if not out_dir.exists():
        return "No recent scripts found."
    rows = []
    for path in sorted(out_dir.glob("script_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            data = json.loads(path.read_text())
            title = data.get("video_title", "")
            hook = data.get("hook", "")
            if title:
                rows.append(f'- "{title}" — hook: {hook[:130]}')
        except Exception:
            continue
    return "\n".join(rows) if rows else "No recent scripts found."


def _build_chapter_timestamps(script: dict) -> str:
    """
    Compute chapter timestamps from script narration word counts.
    Uses 145 wpm TTS pace + overhead per section for intro/callout cards.
    Returns a newline-separated string ready to append to the description.
    """
    WPM           = 145
    CARD_OVERHEAD = 3.5   # seconds for intro card + callout card per section

    t = 0.0
    lines = []
    for sec in script.get("sections", []):
        mins = int(t // 60)
        secs = int(t % 60)
        name = sec.get("name", "")
        if name:
            lines.append(f"{mins}:{secs:02d} {name}")
        narration = sec.get("narration", "")
        word_count = len(narration.split())
        t += (word_count / WPM) * 60 + CARD_OVERHEAD
    return "\n".join(lines)


def _inject_chapter_timestamps(script: dict) -> dict:
    """
    Append chapter timestamps to the script description if not already present.
    Mutates and returns script in-place.
    """
    desc = script.get("description", "")
    if "0:00" in desc:
        return script   # already has chapters
    chapters = _build_chapter_timestamps(script)
    if chapters:
        desc = desc.rstrip() + "\n\n" + chapters

    # Pad description to at least 380 chars with SEO suffix
    if len(desc) < 380:
        title = script.get("video_title", "")
        seo_suffix = (
            f"\n\nLearn the full story behind {title}. "
            "We research, verify, and break down what the mainstream media misses. "
            "Hit subscribe and the bell so you never miss a deep-dive like this."
        )
        desc = desc.rstrip() + seo_suffix

    script["description"] = desc
    return script


def _generate_producer_brief(trending: list[dict], recent_scripts: str) -> dict:
    """
    Stage 1 of a two-stage generation process.

    Gives the model a small, dedicated reasoning pass to:
      • Pick the single strongest topic from trending data
      • Commit to a creative angle, payoff ladder, and scroll-stopping visual
      • Output a compact brief (~500-800 tokens) that grounds Stage 2

    Why two stages beats one: in a single 5,000-word generation pass the model
    must simultaneously do strategy (what to make, why) AND execution (write
    10 sections of tight narration). Separating them lets reasoning concentrate
    fully on creative decision-making before any narration is written, which
    produces stronger topic selection and more coherent payoff ladders.
    """
    trend_lines = "\n".join(
        f'- "{v["title"]}"  ({v["views"]:,} views, by {v["channel"]})'
        for v in trending[:15]
    )
    prompt = f"""## Today's Trending Topics
{trend_lines}

## Recently Published — DO NOT repeat these topics or angles
{recent_scripts}

## Your Task
You are picking ONE topic and designing the creative brief for a 10-12 minute YouTube video.

Choose the topic that simultaneously has:
1. Universal appeal OR an intensely passionate niche that over-indexes on comments
2. A counterintuitive or surprising angle that competitors have clearly missed
3. Immediately actionable value the viewer can use TODAY
4. A genuinely debate-worthy element — people have strong, opposing opinions
5. Evergreen search potential: still searched 12 months from now

Then design:
• The most scroll-stopping first sentence (the hook)
• The 15-word payoff tease shown at 0:00 before the hook (the MrBeast flash-forward)
• A 5-step payoff ladder: each step is the concrete mini-reward delivered by that section
• The single cinematic frame that would stop a human mid-scroll and make them press play
• The comment question that will generate 200+ genuine replies from strong opinions

Avoid boredom at the brief stage:
• Do not choose a topic if the best angle is just "here are facts about X".
• The payoff ladder must escalate: each reward changes the viewer's mental model,
  not just adds another example.
• The freshness_angle must specify a different structure or rhythm from recent videos,
  not just a different topic.
• The comment_question must come after satisfaction, not as a substitute for payoff.

Return ONLY valid JSON:
{{
  "topic": "Exact topic chosen. One sentence on why it beats every alternative in this trend list.",
  "title": "Primary keyword first. Under 70 chars. Power word or specific number included.",
  "hook_sentence": "First spoken sentence. 10-16 words. Most shocking fact. No greeting.",
  "payoff_tease": "15-word flash-forward shown at 0:00 — describe the finale payoff the viewer will see.",
  "payoff_ladder": [
    "Section 2 reward: what the viewer knows after this section that they did not before",
    "Section 4 reward: the counterintuitive insight that reframes everything",
    "Section 6 reward: the strongest piece of evidence delivered",
    "Section 8 reward: the costly mistake revealed before the viewer makes it",
    "Section 10 reward: full resolution of the hook promise + reason to comment"
  ],
  "scroll_stopper_image": "One sentence. Describe the exact cinematic frame — person, location, action, lighting — that stops a human mid-scroll.",
  "freshness_angle": "One sentence. Exactly how this video avoids repeating recent output in structure, framing, or visual rhythm.",
  "comment_question": "The exact closing question. Strong opinions required. Phrased to make even lurkers want to answer."
}}"""

    provider = _choose_ai_provider()
    log.info("  Stage 1: Generating producer brief…")
    if provider in ("cerebras", "groq", "sambanova", "gemini", "hf"):
        raw = _call_free_provider_json(prompt, _SYSTEM_PROMPT_WRITER, temperature=0.70)
    elif provider == "openai":
        raw = _call_openai_json(prompt, _SYSTEM_PROMPT_WRITER, temperature=0.70)
    else:
        if anthropic is None:
            raise RuntimeError("anthropic package not installed")
        client = anthropic.Anthropic(api_key=CLAUDE_KEY)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=5000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

    brief = _parse_json_safe(raw)
    log.info(f"  Brief locked: \"{brief.get('title', '?')}\"")
    log.info(f"  Angle: {brief.get('topic', '')[:100]}")
    return brief


# ─────────────────────────────────────────────────────────────────────────────
# Zeigarnik field synthesis + targeted rewrite
# ─────────────────────────────────────────────────────────────────────────────

def _synthesize_zeigarnik_fields(script: dict) -> dict:
    """
    Derive missing psychological fields from the narration itself.
    Called before the Zeigarnik audit so the audit has the best possible
    input — rather than failing on an empty field the LLM forgot to fill.

    Fields synthesized if empty:
      comment_bait   — most assertive sentence (contains a comparison or superlative)
      end_screen_hook — last full sentence of narration
      identity_mirror — sentence containing "you", a number, or "if you"
      emotional_arc   — default "curiosity → surprise → tension → relief"
    """
    sections  = script.get("sections") or [{}]
    narration = " ".join(s.get("narration", "") for s in sections).strip()

    # Split narration into sentences
    import re as _re
    sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", narration) if len(s.split()) >= 4]

    # ── comment_bait ─────────────────────────────────────────────────────────
    if not (script.get("comment_bait") or "").strip():
        # Pick the most assertive sentence: contains "is", "are", "will", "always",
        # "never", "every", "most", "worst", "best", or a comparison word
        _ASSERT_WORDS = {"is the", "are the", "will never", "will always", "worst",
                         "best", "most", "every single", "never works", "always fails",
                         "only way", "only works", "cannot", "you can't", "impossible"}
        best = next(
            (s for s in sentences
             if any(w in s.lower() for w in _ASSERT_WORDS) and len(s.split()) >= 6),
            sentences[-3] if len(sentences) >= 3 else "",
        )
        if best:
            script["comment_bait"] = best
            log.debug(f"  [Zeig] Synthesized comment_bait: {best[:60]}")

    # ── end_screen_hook ───────────────────────────────────────────────────────
    if not (script.get("end_screen_hook") or "").strip():
        # Use the last 1-2 sentences — most likely to point forward
        hook_candidate = " ".join(sentences[-2:]) if len(sentences) >= 2 else sentences[-1] if sentences else ""
        if hook_candidate:
            script["end_screen_hook"] = hook_candidate
            log.debug(f"  [Zeig] Synthesized end_screen_hook: {hook_candidate[:60]}")

    # ── identity_mirror ───────────────────────────────────────────────────────
    if not (script.get("identity_mirror") or "").strip():
        # Find sentence with "you" + specific detail (number, "if you", "when you")
        _MIRROR_SIGNALS = {"if you", "when you", "you've", "you have", "you're", "you are",
                           "you bought", "you sold", "you held", "you lost", "your portfolio"}
        mirror = next(
            (s for s in sentences
             if any(w in s.lower() for w in _MIRROR_SIGNALS) and any(c.isdigit() for c in s)),
            next((s for s in sentences if any(w in s.lower() for w in _MIRROR_SIGNALS)), ""),
        )
        if mirror:
            script["identity_mirror"] = mirror
            log.debug(f"  [Zeig] Synthesized identity_mirror: {mirror[:60]}")

    # ── emotional_arc ─────────────────────────────────────────────────────────
    if not (script.get("emotional_arc") or "").strip():
        script["emotional_arc"] = "curiosity → surprise → tension → relief"

    return script


def _zeigarnik_rewrite(script: dict, issues: list, provider: str) -> dict:
    """
    Targeted single-pass rewrite that fixes specific Zeigarnik audit failures.

    Rather than regenerating the full script (expensive), this prompts the LLM
    to fix ONLY the failing fields — keeping the narration, title, and visual
    fields intact.
    """
    narration = " ".join(
        s.get("narration", "") for s in (script.get("sections") or [{}])
    ).strip()
    hook = script.get("hook", "")

    # Build a compact failure list for the prompt
    failure_lines = "\n".join(f"  - {i}" for i in issues)

    rewrite_prompt = f"""You wrote this YouTube Shorts script, but it failed the Zeigarnik psychological audit. Fix ONLY the failing elements. Keep everything else identical.

## Current Hook
{hook}

## Current Narration (keep the substance — only adjust framing)
{narration}

## Audit Failures to Fix
{failure_lines}

## How to fix each failure:

**Hook lacks open-loop signal language**: Rewrite the hook so it leads with loss/risk AND opens an unanswerable question. Use words like: "losing", "hidden", "nobody", "invisible", "bleeding", "wrong", "mistake", "secret". Do NOT give any hint of the answer.

**Hard loop closure before 80% mark**: Find where the narration gives away the answer too early. Move that resolution to the LAST 20% of the narration. Replace early "the answer is / the solution is" with more tension-building instead.

**comment_bait is missing or generic**: Write one confident, contestable claim (not a question). It must be specific, defensible, and polarizing enough that someone who disagrees MUST reply. At least 5 words.

**end_screen_hook is missing or generic**: Write the final 1-2 sentences as a curiosity gap pointing to a DIFFERENT next video. Do NOT say "subscribe". Make not watching the next video feel irrational.

**identity_mirror is empty**: Find or write ONE sentence where a specific viewer says "that's exactly me" — include a real dollar amount, specific date, or precise situation.

Return ONLY valid JSON with these fields updated. Keep ALL other fields from the original script unchanged:
{{
  "hook": "Rewritten hook — loss frame + open loop, NO answer given",
  "comment_bait": "Specific contestable claim (≥5 words, not a question)",
  "end_screen_hook": "Curiosity gap to next video — NOT a subscribe ask",
  "identity_mirror": "Hyper-specific 'that's exactly me' sentence from the narration",
  "emotional_arc": "curiosity → surprise → tension → relief",
  "sections": [
    {{
      "name": "{(script.get('sections') or [{}])[0].get('name', 'Short')}",
      "narration": "Revised narration (same length ~95-110 words). Keep the topic and payoff. Only adjust: hook framing, loop closure timing, identity mirror sentence."
    }}
  ]
}}"""

    try:
        if provider in ("cerebras", "groq", "sambanova", "gemini", "hf"):
            raw = _call_free_provider_json(rewrite_prompt, _SYSTEM_PROMPT_WRITER, temperature=0.60)
        elif provider == "openai":
            raw = _call_openai_json(rewrite_prompt, _SYSTEM_PROMPT_WRITER, temperature=0.60)
        else:
            if anthropic is None:
                raise RuntimeError("anthropic package not installed")
            client = anthropic.Anthropic(api_key=CLAUDE_KEY)
            response = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=3000,
                messages=[{"role": "user", "content": rewrite_prompt}],
            )
            raw = response.content[0].text.strip()

        patch = _parse_json_safe(raw)

        # Merge patch into original script — only update fields that were fixed
        if patch.get("hook"):
            script["hook"] = patch["hook"]
        if patch.get("comment_bait"):
            script["comment_bait"] = patch["comment_bait"]
        if patch.get("end_screen_hook"):
            script["end_screen_hook"] = patch["end_screen_hook"]
        if patch.get("identity_mirror"):
            script["identity_mirror"] = patch["identity_mirror"]
        if patch.get("emotional_arc"):
            script["emotional_arc"] = patch["emotional_arc"]

        # Update narration if the rewrite changed it
        patch_secs = patch.get("sections") or []
        orig_secs  = script.get("sections") or []
        for i, psec in enumerate(patch_secs):
            if i < len(orig_secs) and psec.get("narration"):
                orig_secs[i]["narration"] = psec["narration"]

        log.info("  [Shorts] Targeted Zeigarnik rewrite applied")
    except Exception as _e:
        log.warning(f"  [Shorts] Zeigarnik rewrite failed: {_e} — keeping original script")

    return script


def generate_shorts_script(trending: list[dict]) -> dict:
    """
    Generate a YouTube Shorts script: one punchy section, 45-55 seconds total.

    Target: 95-110 words (at ~145 wpm real TTS pace = 39-45 seconds of narration,
    leaving 15 seconds for the callout + CTA tail clips → total ≤59 seconds).

    WHY 95-110 and NOT 115-140:
      Real TTS engines (edge-tts, OpenAI nova) speak at ~145 wpm, not 160 wpm.
      At 145 wpm: 140 words = 57.9s narration + 6s tail = 63.9s → YouTube removes it.
      At 145 wpm: 110 words = 45.5s narration + 6s tail = 51.5s → safe ✅

    Structure:
      HOOK   (0-3s,   ~10 words) — shocking first sentence, no greeting
      BODY   (3-38s,  ~75 words) — proof + counterintuitive insight + mechanism
      PAYOFF (38-45s, ~20 words) — resolution + end-screen curiosity gap
    """
    trend_lines = "\n".join(
        f'- "{v["title"]}"  ({v["views"]:,} views, by {v["channel"]})'
        for v in trending[:12]
    )
    recent_scripts = _recent_script_context(limit=5)

    prompt = f"""You are an elite YouTube Shorts creator. Your Shorts routinely hit 2M+ views in the first 48 hours because you understand how the human brain works — not just what topics are trending.

## Today's Trending Videos
{trend_lines}

## Recent Titles Already Published — do NOT repeat these topics or angles
{recent_scripts}

## Psychological Architecture You MUST Follow

### HOOK (first 10 words)
- Frame with LOSS or RISK — what the viewer is already losing RIGHT NOW.
- Open Loop A: ask an implicit question the viewer's brain cannot answer yet.
- DO NOT hint at the answer. The hook's job is to create unbearable curiosity, not satisfy it.
- Use the community's exact in-group language (crypto: "diamond hands", "DCA", "rekt" etc.).

### BODY (words 11-80)
- Sentence 5-7: Open Loop B (a second unresolved question layered inside the first).
- Escalation ladder: Curiosity → Surprise → Tension → Relief, in that order.
- "Identity mirror": one sentence that makes the viewer say "that's EXACTLY me" (hyper-specific scenario, real dollar amounts, specific situation).
- Re-watch clue: plant one detail in the first 10 words that only makes full sense after the payoff — subtle enough that 80% miss it on first watch.
- Midpoint: Close Loop B → immediately open Loop C.
- Comment bait: one confident, slightly contestable claim (not a question). Defensible but polarizing.
- DO NOT close Loop A before word 80.

### PAYOFF + END SCREEN (last 20-30 words)
- Close Loop A and Loop C SIMULTANEOUSLY in the same sentence — the brain gets a double-dopamine hit.
- End-screen curiosity gap: one sentence that makes NOT watching the next video feel irrational. This replaces "like and subscribe."
- The comment bait naturally surfaces here — state the polarizing claim with confidence.

## Rules
- Pick the SINGLE strongest trending topic with a counterintuitive angle.
- Write ONE continuous narration. No section breaks. No greetings.
- STRICT WORD LIMIT: 95-110 words ONLY (39-45 seconds at real 145 wpm TTS pace). Count carefully. Going over 110 words will push the video past 60 seconds and YouTube will reclassify it as long-form — it will NOT appear in Shorts.
- First sentence MUST lead with loss/risk framing + the most shocking fact. No warm-up.
- Include exactly one "#Shorts" in the tags array.
- Title: 5-9 words, primary keyword first, loss-frame or curiosity-gap. 35-60 chars.
- Description: 2 keyword-rich SEO sentences + "#Shorts". 150-300 chars.
- Tags: 10-14 tags, include "#Shorts" and "Shorts" and topic-specific tags.
- category_id: "22" (People & Blogs) or appropriate category.

## Emotional Ladder Color Mapping
Your `color_mood` field MUST follow the emotional arc:
- Hook section opens → "danger" (loss-frame energy, red urgency)
- Surprise reveal → "curiosity" (electric yellow mystery)
- Tension build → "evidence" (cold blue pressure)
- Relief payoff → "payoff" (warm gold resolution)
- End screen → "action" (green momentum)

Return ONLY valid JSON in this exact shape — no markdown fences, no explanation:

{{
  "video_title": "Loss-frame or curiosity-gap title 35-60 chars",
  "hook": "First 10-word sentence. Loss/risk frame. In-group language. NO answer given.",
  "thumbnail_concept": "Bold 3-4 word loss-frame text + shocked expression + dark urgent background",
  "thumbnail_concept_b": "Alternative thumbnail: different angle, contrast — relief/payoff energy instead of danger",
  "description": "Loss-frame keyword-rich sentence revealing the risk. Second sentence with related keywords. #Shorts",
  "tags": ["Shorts", "#Shorts", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8", "tag9"],
  "category_id": "22",
  "comment_bait": "One confident, slightly polarizing claim from the narration — the sentence designed to force a reply from people who disagree. Should be in the narration too.",
  "zeigarnik_map": "One sentence: when does Loop A open? when does it close? when do Loops B and C open/close?",
  "emotional_arc": "curiosity → surprise → tension → relief",
  "identity_mirror": "The exact sentence from narration where viewer says 'that is exactly me'",
  "rewatch_clue": "The subtle detail planted in first 10 words that only makes sense after the payoff",
  "end_screen_hook": "The final sentence — curiosity gap pointing to the next video. NOT a generic subscribe ask.",
  "sections": [
    {{
      "name": "Short",
      "narration": "95-110 word narration. Structure: [Hook=loss frame + open Loop A ~10 words] [Surprise/Open Loop B ~25 words] [Tension=identity mirror + comment bait ~25 words] [Close Loop B, open Loop C ~15 words] [Relief = close Loop A + C simultaneously + end screen curiosity gap ~15 words]. Spoken word only. COUNT YOUR WORDS — exceeding 110 words makes the video too long for Shorts.",
      "callout_text": "MOST SHOCKING 4-6 WORDS — the statistic or fact that hits hardest",
      "visual_keyword": "cinematic dramatic human scene matching emotional arc peak",
      "lower_third": "3-4 word loss-frame or tension phrase",
      "scene_style": "cold_open",
      "color_mood": "danger",
      "human_behavior": "specific physical action the on-screen human is doing — must match emotional arc",
      "retention_device": "zeigarnik loop",
      "viewer_question": "the EXACT unresolved question burning in the viewer's mind at the 50% mark",
      "contrast_pair": "what the viewer THINKS is true vs what is ACTUALLY true",
      "visual_beats": ["loss-frame hook frame — face realizing something is wrong", "tension reveal — the evidence drops", "relief payoff — the solution or consequence lands"],
      "first_frame": "scroll-stopping image: person in the moment of realizing a loss — genuine distress not staged",
      "mini_payoff": "the double-loop closure — the single sentence that resolves Loop A and C simultaneously",
      "pattern_interrupt": "colour grade shift from danger-red to payoff-gold at the 80% mark",
      "sound_cue": "low bass riser building under tension, sharp impact hit on the payoff beat",
      "asset_prompt": "premium cinematic 9:16 portrait, real human face showing genuine surprise or distress, natural light, photorealistic, shallow DOF, no text"
    }}
  ]
}}"""

    provider = _choose_ai_provider()
    log.info(f"  [Shorts] AI provider: {provider}")

    if provider in ("cerebras", "groq", "sambanova", "gemini", "hf"):
        raw = _call_free_provider_json(prompt, _SYSTEM_PROMPT_WRITER, temperature=0.70)
    elif provider == "openai":
        raw = _call_openai_json(
            prompt,
            _SYSTEM_PROMPT_WRITER,
            temperature=0.8,
        )
    else:
        if anthropic is None:
            raise RuntimeError("anthropic package not installed")
        client = anthropic.Anthropic(api_key=CLAUDE_KEY)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

    script = _validate_script(_parse_json_safe(raw))

    # ── Zeigarnik field synthesis (before audit) ──────────────────────────────
    # If the LLM omitted any critical psychological field, extract it from
    # the narration rather than accepting an empty default.
    script = _synthesize_zeigarnik_fields(script)

    # ── Zeigarnik audit — rewrite once if score < 96 ─────────────────────────
    try:
        from quality_gate import _audit_zeigarnik
        zeig = _audit_zeigarnik(script)
        log.info(f"  [Shorts] Zeigarnik score (initial): {zeig['zeigarnik_score']}/100")
        if zeig["zeigarnik_score"] < 96 and zeig["zeigarnik_issues"]:
            log.info("  [Shorts] Score below 96 — running targeted Zeigarnik rewrite pass…")
            script = _zeigarnik_rewrite(script, zeig["zeigarnik_issues"], provider)
            # Re-synthesize in case rewrite omitted fields again
            script = _synthesize_zeigarnik_fields(script)
            zeig2 = _audit_zeigarnik(script)
            log.info(f"  [Shorts] Zeigarnik score (after rewrite): {zeig2['zeigarnik_score']}/100")
    except Exception as _ze:
        log.warning(f"  [Shorts] Zeigarnik audit/rewrite skipped: {_ze}")

    # Mark as Shorts so upload logic can add correct tags/metadata
    script["_is_shorts"] = True
    # Ensure #Shorts is in tags
    tags = script.get("tags", [])
    if "#Shorts" not in tags:
        tags.insert(0, "#Shorts")
    if "Shorts" not in tags:
        tags.insert(1, "Shorts")
    script["tags"] = tags

    # Hard-cap Shorts narration at 115 words.
    # Real TTS pace: ~145 wpm.  Budget: 60s total − 4s CTA − 2s callout − 1s margin = 53s.
    # At 145 wpm: 53s × (145/60) = 128 words max.  Cap at 115 for a safe 6s buffer.
    # Previous cap of 150 words → ~62s narration + 6s tail = 68s → YouTube reclassifies.
    _SHORTS_WORD_CAP = 115
    for sec in script.get("sections", []):
        narration = sec.get("narration", "")
        words = narration.split()
        if len(words) > _SHORTS_WORD_CAP:
            # Truncate at the last sentence boundary before the cap
            truncated = " ".join(words[:_SHORTS_WORD_CAP])
            last_dot = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
            if last_dot > 40:
                sec["narration"] = truncated[: last_dot + 1].strip()
            else:
                sec["narration"] = truncated.strip()

    return script


def generate_script(trending: list[dict]) -> dict:
    """
    Target runtime: 10-12 minutes — safe under YouTube's 15-min hard cap
    for unverified accounts while still hitting the algorithm sweet spot.

    Science behind the length:
      • 8 min threshold  → mid-roll ads unlock, RPM jumps 40-60 %
      • 10-12 min        → mid-roll ad slots + strong watch-time signal,
                           matches Kurzgesagt (8-12 min), MKBHD (10-14 min)
      • >15 min          → YouTube REMOVES video for unverified channels ❌
      • >20 min          → drop-off risk without an established audience

    Word count maths (conservative, always-safe):
      TTS speaks at ~145 wpm.
      1,500-1,700 words ÷ 145 = 10.3-11.7 min narration.
      Add section-title cards (~7s), callout cards (~20s), fade transitions:
      final video lands at 10-12 minutes — safely under the 15-min cap. ✓

    Structure: 10 sections (same chapter density used by top educational creators).
    """
    trend_lines = "\n".join(
        f'- "{v["title"]}"  ({v["views"]:,} views, by {v["channel"]})'
        for v in trending[:15]
    )
    recent_scripts = _recent_script_context()
    out_dir = Path(__file__).parent / "output"
    ctr_context = _summarize_recent_ctr(out_dir)

    # ── Stage 1: Producer brief (dedicated reasoning pass for topic + strategy) ──
    brief: dict = {}
    try:
        brief = _generate_producer_brief(trending, recent_scripts)
    except Exception as _e:
        log.warning(f"  Producer brief failed ({_e}); continuing with single-stage generation")

    brief_block = (
        f"""## APPROVED PRODUCER BRIEF — Your script MUST honour every decision below.
The brief was approved after dedicated creative reasoning. Do not change the topic,
title, hook_sentence, payoff_tease, or payoff_ladder without a compelling reason.

{json.dumps(brief, indent=2, ensure_ascii=False)}

"""
        if brief else ""
    )

    prompt = f"""{brief_block}## Creative Standards (non-negotiable even when brief is provided)
Benchmark against elite creator craft without copying anyone's identity:
  • MrBeast-level stakes, escalation, clarity, curiosity, and payoff density.
  • Zach King-style visual surprise, clean reveals, and impossible-feeling transitions.
  • Veritasium / Johnny Harris-level explanation, evidence, and narrative tension.
  • 2026 top-creator discipline: slower when the story needs gravity, faster only
    when a new visual idea lands. No generic overstimulation.
  • Creator-grade kinetic motion graphics with realistic VFX-style compositing.
  • Hyper-realistic action-movie shots anchored on human behavior: fear,
    curiosity, confusion, decision-making, urgency, relief, trust, social proof,
    status, loss aversion, and the need to feel in control.
  • Retain general audiences and expert audiences at the same time: simple surface,
    smart substance underneath.

Do NOT write a boring lecture. Every section must contain:
  • A reason to keep watching now.
  • One concrete image the renderer can visualize.
  • A specific beat where the visual changes, not just background + centered text.
  • A human behavior anchor: what the viewer/person in the scene feels, notices,
    misunderstands, fears, chooses, or changes.
  • A color role that tells the viewer how to feel before they process the words.
  • A retention device: mystery object, countdown, challenge, fakeout, reversal,
    bet, proof drop, experiment, or high-stakes comparison.
  • A viewer question: the exact question the audience should be asking internally.
  • A human sentence that sounds spoken, not essay-written.

## Top Creator Producer Standard
Write for the edit before writing the words:
  • First 8 seconds must visually prove the title/thumbnail promise.
  • Every 30-60 seconds must create a mini-story: setup → tension → payoff.
  • Every section needs a concrete mini_payoff that rewards the viewer for staying.
  • Every section needs a pattern_interrupt that is information-driven, not random.
  • Every section needs a sound_cue: impact hit, riser, silence, whoosh, bass drop,
    snap zoom, or soft room-tone pause. Sound must serve the story.
  • Every asset_prompt must describe a high-budget cinematic frame a real viewer
    would want to look at. Never request abstract boxes, circles, generic arrows,
    dashboards, empty diagrams, or wallpaper gradients.
  • Avoid repeating the previous video's structure. Change framing, rhythm,
    visual motif, and scene order when the topic changes.
  • Think: title promise, human stakes, visual surprise, mini-payoffs, and a final
    comment question that people genuinely want to answer.

## PSYCHOLOGICAL ARCHITECTURE — Non-Negotiable Emotional Engineering

Human attention is driven by ANTICIPATION, not information. The brain releases dopamine
at the PREDICTION of reward, not the reward itself. Every section must be engineered to
create and sustain that prediction loop.

### 10-Section Emotional Arc Map

Each section has a prescribed psychological role. You MUST honor this architecture:

| # | Section | Psychological Role | Viewer Emotion | Key Mechanism |
|---|---|---|---|---|
| 1 | Hook | Macro-loop opens + loss frame | Shock + dread | Plant Loop A (unanswerable). Loss-aversion opening 2.5× stronger than gain. |
| 2 | Why Now | Stakes amplification | Fear + urgency | FOMO cascade. "People like you are already affected." Social proof anchor. |
| 3 | Real Problem | Identity mirror | Recognition + pain | The "that's EXACTLY me" sentence. Viewer must feel seen and understood. |
| 4 | Gets Wrong | Worldview disruption | Cognitive dissonance | State the popular belief confidently. Then detonate it with one fact. |
| 5 | Hidden Truth | **FAKE-OUT** (deliberate false resolution) | Confusion → shock | Plant a CONFIDENT wrong answer. Let viewer feel they finally get it. Then shatter it. |
| 6 | Evidence | Authority cascade + mid-video lock | Analytical trust | Proof drops. Then the mid-video tease that chains viewers past the 50% cliff. |
| 7 | How It Works | Mechanism "aha" | Intellectual pleasure | The viewer can explain it in one sentence. That pride keeps them watching. |
| 8 | Biggest Mistake | **ANXIETY PEAK** — most intense moment | Dread + loss aversion | The most visceral, financially/personally threatening moment. Sits at 65-70% — the sharpest YouTube drop-off zone. |
| 9 | Solution | Double-loop resolution | Relief + dopamine hit | Close Loop A AND Loop D SIMULTANEOUSLY. The brain gets a double-reward. |
| 10 | CTA | Tribal identity reward | Belonging + satisfaction | "You now know what 97% never figure out." In-group confirmation. |

### FAKE-OUT DIRECTIVE (Section 5 — "The Hidden Truth") — MANDATORY

At the 40% mark of this section, deliver a CONFIDENT, PLAUSIBLE-SOUNDING wrong answer.
Write it as if it IS the truth — let the viewer feel satisfied. Then 2-3 sentences later,
shatter it with: "But here's what no one tells you about that explanation..."

This creates a double-dopamine hit: satisfaction of knowing → surprise of being wrong →
STRONGER curiosity than if the answer was never offered. It is the single most powerful
retention device in long-form educational content.

The `fake_out_content` field MUST contain the specific false answer you planted.

### PEAK ENGINEERING — Section 8 Must Be the Highest-Stakes Moment

Section 8 sits at ~65-70% of the video's total duration — exactly where YouTube
analytics show the sharpest viewer drop-off when content goes flat.
This section MUST contain the most shocking, personally threatening consequence of
the entire video. Use a real cautionary example with specific names/numbers/outcomes.
Make the viewer feel: "This could happen to me right now."

### MID-VIDEO TEASE (Section 6 — after the like trigger) — MANDATORY

After the mid-video like trigger, add one sentence explicitly naming what's coming:
"And in 3 minutes, you'll see the ONE mistake that destroys even people who know all of this."
This is the single most effective completion driver at the 50% drop-off point.
The `mid_video_tease` field in Section 6 MUST contain this exact forward reference.

### COGNITIVE LOAD ALTERNATION — Prevent Fatigue, Sustain Engagement

Alternate HEAVY (analytical) and LIGHT (emotional/story) sections like interval training:
- Section 1 Hook → LIGHT (visceral, fast, emotional)
- Section 2 Why Now → HEAVY (data, urgency, specific numbers)
- Section 3 Problem → LIGHT (relatable story, identity mirror)
- Section 4 Gets Wrong → HEAVY (evidence, logic, myth demolition)
- Section 5 Hidden Truth → LIGHT (counterintuitive narrative, fake-out)
- Section 6 Evidence → HEAVY (data, proof cascade, research)
- Section 7 How It Works → HEAVY (mechanism, explanation)
- Section 8 Biggest Mistake → LIGHT (story, warning, personal threat)
- Section 9 Solution → HEAVY (actionable steps, specifics)
- Section 10 CTA → LIGHT (warm, conversational, belonging)

The `cognitive_load` field on every section MUST match this pattern exactly.

### OPEN LOOP DISCIPLINE — Zeigarnik Architecture

- Section 1: **Opens Loop A** (the macro-promise — the question only Section 9 answers)
- Section 3: **Opens Loop B** (a micro-question resolved at Section 4's end)
- Section 5: **Partially closes Loop A with the WRONG ANSWER** → immediately opens Loop C
- Section 6: **Opens Loop D** (the danger of the mistake in Section 8)
- Section 8: **Loop D peaks** (the consequence becomes viscerally real)
- Section 9: **Closes Loop A + Loop D simultaneously** (double resolution = dopamine peak)
- Section 10: **Closes Loop C** (social proof that the solution worked for others)

The `loop_action` field on every section MUST describe which loop opens or closes.

### REWATCH SEED — Plant in Section 1, Pays Off in Section 9

In Section 1's narration, hide ONE detail — a specific word, number, or object — that
only makes complete sense after Section 9's resolution. Viewers who rewatch will have
the "I should have seen that" moment. This drives 2nd and 3rd views.
The `rewatch_clue` field MUST contain this exact detail.

### SOCIAL PROOF CASCADE

- Section 2: "Most people in your situation already..." (external proof)
- Section 6: "[Study/person/event] confirmed what millions discovered..." (authority)
- Section 10: "You now know what 97% in this space never figure out." (exclusive in-group)

### PULL-QUOTE SENTENCE — One Per Section

Every section has ONE sentence that is so powerful it deserves to be rendered full-screen
for 3 seconds in large centered text — the "screenshot moment" viewers share.
The `peak_sentence` field MUST contain this exact sentence verbatim from the narration.

---

## Anti-Boredom + Algorithm Satisfaction Rules
YouTube recommendations follow viewer satisfaction: people must click, keep watching,
feel the video paid off, and avoid "not interested" reactions. Boredom is usually
caused by predictability, not by slow pacing.

Every section must pass this boredom test:
  • Novelty: introduce one new fact, contradiction, object, scene, or human decision.
    If a section only restates earlier information, merge it or replace it.
  • Curiosity pressure: keep exactly one unresolved question alive until the finale.
    Do not open five loops and forget them; unresolved confusion feels like clickbait.
  • Cognitive ease: use short spoken sentences and one idea per paragraph. No dense
    essay voice. If a sentence has two clauses that compete, split it.
  • Variable rhythm: never use the same scene_style, color_mood, retention_device,
    sentence opening, or mini_payoff pattern in adjacent sections.
  • Reward rate: deliver a concrete mini-payoff every 30-60 seconds. A payoff is a
    new usable insight, proof, reversal, named example, or visual transformation.
  • Visual novelty: each section must change what the viewer is looking at, not just
    change the words over the same kind of background.
  • CTA restraint: ask for subscribe/like only after value has landed. Too many asks
    early create friction and reduce satisfaction.
  • Ending satisfaction: resolve the title promise before the subscribe ask. The
    final comment question must extend the topic, not replace the payoff.

## Today's Trending Videos (context only — topic already chosen in the brief above)
{trend_lines}

## Recent Videos Already Generated — avoid repeating these angles
{recent_scripts}

## Recent Thumbnail CTR Performance (use this to inform thumbnail_concept choices)
{ctr_context}

## Topic Selection
{"Use the topic, title, hook_sentence, payoff_tease, and payoff_ladder EXACTLY as specified in the APPROVED PRODUCER BRIEF above. Do not change the creative direction." if brief else """Pick the SINGLE BEST topic that has ALL of:
1. Universal appeal OR intensely passionate niche
2. A surprising / counterintuitive angle competitors have missed
3. Immediate actionable value viewers can use TODAY
4. A debate-worthy element (drives comment section)
5. Evergreen search potential (still searched 12 months from now)

Do not produce another airport-security/TSA/security-checkpoint topic unless it is
clearly the strongest trend AND you can take a genuinely new angle. If recent
titles cluster around one topic, deliberately diversify into a fresher trend."""}

═══════════════════════════════════════════════════
TARGET: 10-12 MINUTE VIDEO  (1,500-1,700 spoken words TOTAL)
YouTube REMOVES videos over 15 minutes on unverified channels.
Stay safely under: 1,500-1,700 words ÷ 145 wpm = 10.3-11.7 min narration.
Add cards & transitions → 10-12 min final. Mid-roll ads unlock at 8 min. ✓
═══════════════════════════════════════════════════

## Absolute Script Rules

COLD OPEN TEASE (0:00-0:08 · ~15 words — MANDATORY):
  • Before anything else, flash the PAYOFF: tease the finale moment in 1-2 sentences.
    Example: "In 9 minutes, you'll see exactly why [surprising outcome]. Here's how."
  • This must be the first 15 words of the Hook narration, then pivot into the hook.
  • Works sound-off: the image shown is the finale frame — a before/after, a reveal,
    a transformation, or the most dramatic visual in the video.
  • Top creators (MrBeast, Zach King) always open with the payoff to lock viewers
    past the 30-second cliff. Mirror that discipline here.

HOOK (0:00-0:25 · 65-80 words total, including tease):
  • First 15 words: flash the payoff tease (see above).
  • Next sentences: most shocking / counterintuitive fact. No greetings ever.
  • Immediately plant an open loop: "By the end I'll reveal the one thing…"
  • The first 8 seconds must be understandable with sound off through visuals.

COLD OPEN (0:25-1:15 · 130-155 words):
  • Why does this topic matter RIGHT NOW? Urgency. Stakes. Real numbers.
  • Value-anchored subscribe ask: name the specific content the channel covers every week and why missing the next video is a real loss. Generic "subscribe for more videos" lines are forbidden.

PATTERN INTERRUPT every ~90 seconds:
  • Shock stat, rhetorical question, or tonal shift to reset attention.
  • Include a visual reset: split-screen, evidence card, timeline, map-style reveal,
    fakeout, zoom into a detail, or before/after transformation.

RETENTION BRIDGE at every section end:
  • Tease next section to prevent drop-off: "But what I found next changed everything…"

TONE: Conversational, direct, energetic. Write EXACTLY as you speak.
  Contractions. Short sentences. Genuine enthusiasm. Zero corporate language.

SPECIFICITY: Real numbers, named examples, concrete outcomes. Vague = boring.
FACTUAL SAFETY: Do not invent sources, studies, quotes, or named data. If a precise
source is unavailable, phrase it as a clearly labeled estimate or remove it.
NO CLAIM RECYCLING: Do not repeat the same shock statistic in more than two sections.
If a section cannot add a new fact, it must add a new human scenario, mechanism,
or counterintuitive decision.

EMOTIONAL ARC: Curiosity → Tension → Revelation → Satisfaction → Action
RETENTION ARC: Mystery object → stakes → wrong belief → proof drop → mechanism →
costly mistake → concrete win → unresolved comment question.

## Color + Retention System
Use color like top creators use sound design: as an attention reset.
The visual system is high-contrast and cinematic:
  • Base: near-black / deep navy for premium contrast and focus.
  • Curiosity: electric yellow (#FFD400) for hooks, open loops, and key objects.
  • Danger / mistakes: red (#FF2D2D) for threat, wrong choices, and stakes.
  • Evidence / trust: cyan (#00E5FF) for data, timelines, mechanisms, and proof.
  • Action / payoff: green (#00FF85) for solutions, wins, and relief.
  • White is reserved for must-read text.

Do not make the video one-color. Rotate color_mood by narrative purpose:
curiosity → danger → evidence → mechanism → action → payoff.
Never use beige, low-contrast grey, or soft pastel palettes for key retention beats.

## Visual Direction Rules
The renderer can use scene_style to pick the animation language:
  • "cold_open"     = fast-cut hook, huge visual stakes, shock reveal
  • "human_action"  = hyper-realistic action-movie shot anchored on behavior,
                      with handheld motion, subject tracking, urgency, and payoff
  • "documentary"   = realistic image-first scene, side captions, lower-third details
  • "evidence"      = cinematic proof scene, real visual receipts, data only if meaningful
  • "split_screen"  = compare myth vs truth, before vs after, bad vs good
  • "timeline"      = chronological escalation with moving markers
  • "blueprint"     = mechanism reveal through light paths, human tools, or real-world analogy
  • "action_steps"  = practical transformation shown through a human doing the step
  • "finale"        = payoff recap, comment question, subscribe reason

Use scene_style intentionally. NEVER make every section the same scene_style —
you MUST use at least 6 DIFFERENT scene_style values across all sections.
You MUST use at least 4 DIFFERENT color_mood values across all sections.
You MUST use at least 6 DIFFERENT retention_device values across all sections.
When using "human_action", visual_keyword MUST describe a real person doing a
clear physical action in a cinematic environment, not an abstract idea.
Every section must include 3 visual_beats:
  1. Establish the human moment.
  2. Escalate with a motion/VFX reveal.
  3. Pay off the idea visually.
Every section must also include:
  • retention_device: one named retention tactic.
  • viewer_question: a specific question the viewer should be asking.
  • contrast_pair: a sharp before/after, myth/truth, expected/actual, or bad/good comparison.
  • first_frame: the exact image the viewer sees at the start of this section.
  • mini_payoff: the concrete reward/reveal delivered by this section.
  • pattern_interrupt: the exact visual reset or story twist in this section.
  • sound_cue: the exact sound-design beat that supports the visual change.
  • asset_prompt: a DETAILED cinematic image prompt (see formula below).

## Asset Prompt Formula (CRITICAL — used to generate or source real images)
Every asset_prompt MUST follow this exact structure:
  "[specific human action/emotion] + [precise environment/location] + [lighting style] + [camera angle/lens] + [color grade/mood]"

Examples of GOOD asset_prompts:
  "Man in dark suit staring at wall of screens showing financial data, corporate war room, cold blue-white monitor glow, tight over-shoulder shot, desaturated with cyan highlights"
  "Young woman running through rain-soaked city street at night, motion blur headlights behind her, 35mm film look, high-contrast neon reflections"
  "Crypto trader hands trembling over keyboard, extreme close-up on screen showing red chart, under-lit with green and red monitor glow, shallow depth of field"
  "Empty boardroom with single chair, overhead fluorescent light, wide establishing shot from doorway, cold grey-blue tone"
  "Two people arguing over a document across desk, documentary-style handheld, harsh window sidelight, desaturated urban office"

BAD asset_prompts (too vague — never write these):
  "cinematic scene", "dramatic background", "stock footage style", "generic video background"

asset_prompt must be at least 20 words. Always include:
  1. A human subject OR a specific real-world object (not abstract shapes)
  2. A concrete location or environment
  3. The lighting source (monitor glow, sunlight, neon, overhead, etc.)
  4. The camera angle or focal length feel
  5. The color grade that matches the color_mood

The JSON examples below are templates. Add retention_device, viewer_question,
contrast_pair, first_frame, mini_payoff, pattern_interrupt, sound_cue, and
asset_prompt to EVERY section object exactly as normal JSON string fields.

## Return ONLY valid JSON — no markdown fences, no explanation:

{{
  "video_title": "Primary keyword in first 3 words + power word or number, under 70 chars",
  "hook": "Literal first spoken sentence — shocking/bold/curious, max 20 words, no greeting",
  "thumbnail_concept": "Precise thumbnail: 3-5 word text overlay + expression + background color/scene + graphic",
  "thumbnail_concept_b": "Alternative A/B test thumbnail: different framing, text position, or emotional angle than thumbnail_concept",
  "description": "First 2 sentences keyword-dense (above the fold). Chapter timestamps 0:00-11:xx. End: Subscribe for weekly videos. 470-500 chars total.",
  "tags": ["tag1","tag2","tag3","tag4","tag5","tag6","tag7","tag8","tag9","tag10","tag11","tag12","tag13","tag14","tag15"],
  "category_id": "22",
  "creator_standard": {{
    "title_promise": "The viewer promise made by the title and thumbnail",
    "first_8_seconds_visual": "The exact first image that proves the promise visually",
    "payoff_ladder": "One sentence describing how each section escalates mini-payoffs",
    "freshness_rule": "How this video avoids repeating recent output structure"
  }},
  "sections": [

NOTE: The following 4 fields are REQUIRED on EVERY section object:
  "emotion_target"  — the specific emotion the viewer must feel during this section
                      (e.g. "shock", "dread", "recognition", "cognitive_dissonance",
                       "confusion", "analytical_trust", "intellectual_awe",
                       "anxiety", "relief", "belonging")
  "peak_sentence"   — the single most powerful sentence in this section's narration,
                      verbatim, used for full-screen pull-quote rendering
  "cognitive_load"  — "heavy" (analytical/data-dense) or "light" (emotional/story)
  "loop_action"     — which Zeigarnik loop opens or closes (e.g. "opens Loop A",
                      "closes Loop B", "opens Loop D + closes Loop A simultaneously")

    {{
      "name": "Hook",
      "narration": "COLD OPEN TEASE first (15 words): flash the finale payoff, then immediately pivot. Then: loss-aversion opening (what the viewer is already losing RIGHT NOW). Plant Loop A — an implicit question only Section 9 answers. Embed the REWATCH SEED: one specific detail that only makes sense after the full resolution. End with urgency bridge to Section 2. Spoken words only. (65-80 words)",
      "callout_text": "Most shocking loss-frame claim — 4-7 words ALL CAPS",
      "visual_keyword": "cinematic dramatic human realizing financial or personal loss",
      "lower_third": "2-4 bold words",
      "scene_style": "cold_open",
      "color_mood": "danger",
      "motion_cue": "fast zoom into the loss-frame detail, freeze on the open loop question",
      "visual_metaphor": "one concrete object that embodies the loss the viewer hasn't noticed yet",
      "human_behavior": "the viewer feels danger, dread, and a desperate need to know more",
      "visual_beats": ["scroll-stopping loss-frame image", "violent punch-in on the shocking detail", "freeze on the unanswered loop"],
      "first_frame": "one scroll-stopping cinematic image — person at the exact moment of realizing a hidden loss",
      "mini_payoff": "the viewer understands the shocking claim is real and personal",
      "pattern_interrupt": "snap from calm image to high-stakes loss-frame reveal",
      "sound_cue": "bass impact then half-second silence on the reveal",
      "asset_prompt": "Person frozen mid-action realizing something is wrong, tense facial expression, dramatic split-tone lighting (deep shadow one side, harsh highlight other), tight 85mm portrait framing, desaturated with one accent color matching the hook emotion",
      "emotion_target": "shock",
      "peak_sentence": "The single most shocking sentence from this section's narration — verbatim",
      "cognitive_load": "light",
      "loop_action": "opens Loop A",
      "rewatch_clue": "The subtle detail in the first 10 words that only resolves at Section 9",
      "retention_bridge": ""
    }},

    {{
      "name": "Why This Matters Now",
      "narration": "Stakes amplification: why RIGHT NOW, not 6 months from now. Specific current data point or cultural flashpoint. Social proof anchor: 'Most people in your situation are already...' Then a value-anchored subscribe ask — name exactly what this channel covers weekly and why missing the next video is a measurable loss. Generic subscribe lines are forbidden. (130-155 words)",
      "callout_text": "Urgency stat or current impact — 4-7 words",
      "visual_keyword": "real person reacting to urgent breaking news cinematic action shot",
      "lower_third": "2-4 words",
      "scene_style": "human_action",
      "color_mood": "danger",
      "motion_cue": "handheld push-in on a person reacting, quick punch-ins on the detail they notice",
      "visual_metaphor": "what the viewer would physically see if this happened to them today",
      "human_behavior": "urgency, attention snap, FOMO, need to act immediately",
      "visual_beats": ["person notices the urgent signal", "environment compresses with red warning overlays", "cut to the consequence if ignored"],
      "first_frame": "a real person caught at the instant the stakes become visible and personal",
      "mini_payoff": "the viewer understands why this matters to THEM right now",
      "pattern_interrupt": "fast cut from human reaction to the personal consequence",
      "sound_cue": "riser into a sharp whoosh cut",
      "asset_prompt": "photoreal cinematic human urgency reaction scene, real stakes visible, no generic graphics",
      "emotion_target": "dread",
      "peak_sentence": "The single most urgency-creating sentence from this section — verbatim",
      "cognitive_load": "heavy",
      "loop_action": "sustains Loop A, opens social proof thread",
      "social_proof_anchor": "The specific 'most people in your situation' reference for this section",
      "cta_type": "subscribe",
      "retention_bridge": "One-sentence tease of the painful problem coming next — must make NOT watching feel irrational"
    }},

    {{
      "name": "The Real Problem",
      "narration": "IDENTITY MIRROR MANDATE: Open with a scene the viewer has personally lived — not a hypothetical, not 'imagine this', but a specific real moment they have already experienced. Write: 'You've been in this exact situation...' or 'You know this feeling.' The viewer must think: 'That's EXACTLY me.' Then reveal why the problem is worse than they think — drop a stat that makes the familiar pain feel newly alarming. Open Loop B immediately: name the specific question that Section 4 will answer. End with the viewer feeling understood AND newly curious. (150-170 words)",
      "callout_text": "Core problem stat — 4-7 words",
      "visual_keyword": "frustrated person making wrong decision under pressure cinematic action shot",
      "lower_third": "2-4 words",
      "scene_style": "human_action",
      "color_mood": "danger",
      "motion_cue": "track the person from confusion to mistake, then freeze-frame the moment everything goes wrong",
      "visual_metaphor": "relatable everyday moment where the problem becomes visible",
      "human_behavior": "recognition and solidarity — viewer sees themselves, feels seen, trusts narrator",
      "visual_beats": ["viewer-recognizable pressure moment", "wrong assumption gets visually locked on", "impact frame reveals the cost"],
      "first_frame": "a relatable mistake forming in a real-world scene — NOT a diagram or concept graphic",
      "mini_payoff": "viewer thinks: 'someone finally understands this and is about to give me the real answer'",
      "pattern_interrupt": "freeze-frame the exact wrong assumption — add a super-imposed word label",
      "sound_cue": "hard stop followed by low bass hit",
      "asset_prompt": "cinematic real-person mistake under pressure, visible consequence, no empty diagrams",
      "emotion_target": "recognition",
      "peak_sentence": "The 'that's EXACTLY me' identity mirror sentence — the most viscerally relatable line — verbatim",
      "cognitive_load": "light",
      "loop_action": "opens Loop B — names the micro-question that Section 4 closes",
      "retention_bridge": "Name the exact wrong belief that Section 4 is about to demolish — one sentence, no hedging"
    }},

    {{
      "name": "What Everyone Gets Wrong",
      "narration": "WORLDVIEW DETONATION MANDATE: Start by naming the exact belief the viewer held five seconds ago — validate it briefly so they feel heard ('And that makes sense — because...') — then systematically detonate it with two or three pieces of evidence that cannot be argued with. Be opinionated and specific: 'Most people think X. That logic is exactly why they get Y outcome.' Add ONE authoritative external source by name. Close Loop B: deliver the answer to the micro-question opened in Section 3 — make the resolution feel satisfying but immediately raise the stakes for Section 5's deeper puzzle. (160-185 words)",
      "callout_text": "The myth being busted — 4-7 words",
      "visual_keyword": "myth exploding wrong belief dramatic cinematic documentary",
      "lower_third": "2-4 words",
      "scene_style": "evidence",
      "color_mood": "evidence",
      "motion_cue": "myth label gets crossed out frame-by-frame, counter-evidence slams in with impact",
      "visual_metaphor": "an object or structure the viewer trusted that collapses under closer inspection",
      "human_behavior": "belief reversal — viewer's existing mental model shatters, replaced by sharper truth",
      "visual_beats": ["common belief shown as plausible scene", "evidence contradicts it hard", "viewer's worldview replaced with sharper truth"],
      "first_frame": "the common belief playing out as a real moment — NOT a myth-bubble graphic",
      "mini_payoff": "viewer's old belief collapses — they feel smarter for knowing this now",
      "pattern_interrupt": "misdirection reveal: what looked true flips into actual cause with a hard visual cut",
      "sound_cue": "glitch snap into clean clinical tone — signalling new accuracy",
      "asset_prompt": "cinematic myth-versus-truth scene — belief collapsing under evidence, real objects, human context, no floating labels",
      "emotion_target": "cognitive_dissonance",
      "peak_sentence": "The worldview-detonation sentence — the line that makes the old belief feel obviously wrong — verbatim",
      "cognitive_load": "heavy",
      "loop_action": "closes Loop B, sustains Loop A — keeps the bigger original question open",
      "retention_bridge": "Tease the even deeper hidden truth in Section 5 — one line that makes watching optional feel irrational"
    }},

    {{
      "name": "The Hidden Truth",
      "narration": "FAKE-OUT STRUCTURE (mandatory): First, deliver a CONFIDENT, PLAUSIBLE wrong answer — write it as if it IS the truth. Let the viewer feel they finally understand (~40-50 words). Then shatter it: 'But here's what no one tells you about that explanation...' Then deliver the REAL counterintuitive insight with a specific data point or story. Plant Loop C immediately after the rug pull. (165-190 words)",
      "callout_text": "The real revelation — 4-7 impactful words (NOT the fake answer)",
      "visual_keyword": "revelation hidden truth discovery cinematic dramatic reveal",
      "lower_third": "2-4 words",
      "scene_style": "documentary",
      "color_mood": "curiosity",
      "motion_cue": "slow push-in on the false answer → sudden visual shatter → real truth emerges with spotlight",
      "visual_metaphor": "an object that looks like one thing but reveals a completely different mechanism when examined",
      "human_behavior": "false satisfaction → shock → stronger curiosity than before",
      "visual_beats": ["false answer settles as truth", "visual shatter — rug pull", "real hidden detail emerges unmissably"],
      "first_frame": "a realistic scene where the false answer looks completely plausible",
      "mini_payoff": "the fake-out + real truth creates stronger impact than either alone",
      "pattern_interrupt": "visual shattering moment as the false answer collapses",
      "sound_cue": "brief resolution tone → hard glitch snap → new riser for the real truth",
      "asset_prompt": "premium documentary reveal frame — object or scene that transforms meaning entirely on second look, cinematic light, real texture",
      "emotion_target": "confusion",
      "peak_sentence": "The rug-pull sentence — the one that shatters the false answer — verbatim",
      "cognitive_load": "light",
      "loop_action": "partially closes Loop A with wrong answer, opens Loop C",
      "fake_out_content": "The exact WRONG ANSWER planted as truth in the first 50 words of this section",
      "retention_bridge": "Tease the evidence that proves it — make NOT watching feel like staying blind"
    }},

    {{
      "name": "Deep Dive — The Evidence",
      "narration": "Proof cascade: studies, historical examples, named experts, specific case studies with dates/numbers. Pattern interrupt mid-section: 'Here's the part most videos skip.' After the single most surprising proof — the fact that changes how the viewer sees everything — insert the mid-video like trigger: 'If that just changed how you see [specific fact], hit like — it helps this reach people who need it.' Then IMMEDIATELY add the mid-video tease: 'And in 3 minutes, you'll see the ONE mistake that destroys even people who know all of this.' (195-225 words)",
      "callout_text": "Key evidence number or finding — 4-7 words",
      "visual_keyword": "research data evidence investigative professional cinematic",
      "lower_third": "2-4 words",
      "scene_style": "timeline",
      "color_mood": "evidence",
      "motion_cue": "evidence appears as chronological markers with escalating stakes",
      "visual_metaphor": "paper trail, receipts, timeline, or data wall — tangible proof that exists in the real world",
      "human_behavior": "skepticism dissolves into trust as receipts line up",
      "visual_beats": ["receipt one appears", "timeline accelerates through proof", "final marker lands as undeniable"],
      "first_frame": "a cinematic proof scene with tangible evidence already in frame — not an empty graphic",
      "mini_payoff": "the viewer gets the strongest evidence, not just a claim",
      "pattern_interrupt": "new proof arrives from an unexpected visual angle — the receipt no one expected",
      "sound_cue": "paper/metal impact synced to the proof reveal",
      "asset_prompt": "high-end investigative documentary evidence scene, monitors, paper trail, real texture, no empty boxes",
      "emotion_target": "analytical_trust",
      "peak_sentence": "The single most surprising evidence sentence — the fact that changes everything — verbatim",
      "cognitive_load": "heavy",
      "loop_action": "sustains Loop C, opens Loop D (the coming danger in Section 8)",
      "mid_video_tease": "The exact forward-reference sentence naming Section 8's danger — verbatim",
      "social_proof_anchor": "The specific study, person, or event confirming what millions discovered",
      "cta_type": "like",
      "retention_bridge": "Tease the mechanism that makes this possible — one sentence, visceral"
    }},

    {{
      "name": "Deep Dive — How It Actually Works",
      "narration": "INTELLECTUAL AWE MANDATE: The goal of this section is to make the viewer feel genuinely clever for understanding the mechanism. Do NOT just explain — use a single, instantly graspable analogy that reframes how they see the system (e.g., 'This works exactly like X — and once you see that, you can't unsee it.'). Then walk through the mechanism in three concrete steps — not vague concepts, but specific cause-and-effect actions. End with one real-world story or case study (names, outcomes, numbers) that proves the mechanism is predictable when you know it. Sustain Loop C (the hidden truth from Section 5) and escalate Loop D: close with a forward reference — 'Which makes the mistake I'm about to show you even more devastating.' (190-215 words)",
      "callout_text": "Core mechanism — 4-7 words",
      "visual_keyword": "mechanism system working blueprint cinematic analytical",
      "lower_third": "2-4 words",
      "scene_style": "blueprint",
      "color_mood": "mechanism",
      "motion_cue": "chaos resolves into blueprint — arrows reveal the mechanism step by step with satisfying precision",
      "visual_metaphor": "a machine, circuit, or system that makes the cause-and-effect path unmissable",
      "human_behavior": "intellectual clarity — viewer feels competent, can now predict the system's behavior",
      "visual_beats": ["show confusing system as visible chaos", "analogy makes it click — chaos reorganizes", "mechanism steps confirm predictability"],
      "first_frame": "a real-world system at the exact moment before the mechanism reveals itself",
      "mini_payoff": "viewer can describe the mechanism in one sentence and feels rewarded for understanding it",
      "pattern_interrupt": "the single analogy lands — chaos visually reorganizes into clean cause-and-effect",
      "sound_cue": "riser that resolves into a precise satisfying click — then soft low tension tone for Loop D escalation",
      "asset_prompt": "cinematic mechanism reveal — real tools, screens, light path following logical steps, no empty arrows or blank boxes",
      "emotion_target": "intellectual_awe",
      "peak_sentence": "The analogy sentence — the one line that makes the whole mechanism instantly click — verbatim",
      "cognitive_load": "heavy",
      "loop_action": "sustains Loop C, escalates Loop D — the biggest mistake is now a ticking clock",
      "retention_bridge": "Name the EXACT mistake being teased — 'And the worst part? Most people with this knowledge still do this one thing wrong.' Make Section 8 feel mandatory."
    }},

    {{
      "name": "The Biggest Mistake",
      "narration": "ANXIETY PEAK — this is the most viscerally threatening moment of the entire video (sits at 65-70%). The single error that destroys results even for people who understand everything in Sections 1-7. Be direct and opinionated. Use a REAL cautionary example with specific names, numbers, or publicly known outcomes. Make the viewer feel: 'This could be happening to me RIGHT NOW.' This is Loop D's peak moment — the consequence becomes personally real. (155-180 words)",
      "callout_text": "The fatal consequence — 4-7 words ALL CAPS",
      "visual_keyword": "person at exact moment of catastrophic wrong decision cinematic close up",
      "lower_third": "2-4 words",
      "scene_style": "human_action",
      "color_mood": "danger",
      "motion_cue": "action-movie freeze frame before the mistake, red tracking box locks onto the wrong choice, time-stop before impact",
      "visual_metaphor": "a fork in the road where one path leads to visible destruction",
      "human_behavior": "loss aversion at maximum — the viewer physically flinches at the consequence",
      "visual_beats": ["person reaches for the wrong choice", "red lock-on freezes the action at peak danger", "cut to the real-world consequence"],
      "first_frame": "a human at the precise moment before the catastrophic wrong choice — genuine fear on face",
      "mini_payoff": "the viewer learns the fatal mistake before making it themselves",
      "pattern_interrupt": "time-freeze before impact — then hard cut to consequence",
      "sound_cue": "time-stop suction hit then complete silence — then single bass note",
      "asset_prompt": "cinematic decision-crisis moment, real human face showing fear at peak of wrong choice, dark environment, red accent lighting, no HUD boxes or labels",
      "emotion_target": "anxiety",
      "peak_sentence": "The single most viscerally threatening sentence — the one that makes the viewer feel personally at risk — verbatim",
      "cognitive_load": "light",
      "loop_action": "Loop D peaks — consequence becomes viscerally real",
      "retention_bridge": "One sentence making the solution feel like the only way out — make NOT watching feel dangerous"
    }},

    {{
      "name": "The Solution",
      "narration": "DOUBLE-LOOP RESOLUTION: the narration must explicitly close Loop A (the macro-question from Section 1) AND Loop D (the danger from Section 8) in the SAME PARAGRAPH. 'Here is the answer to what I showed you at the very start — and here's how it also protects you from the mistake that destroys most people.' Then numbered, specific, actionable steps. Each step = one concrete action the viewer can start TODAY. No vague advice. Also include the rewatch callback: briefly reference the seed detail planted in Section 1 — it now makes complete sense. (175-205 words)",
      "callout_text": "Action callout — 4-7 words",
      "visual_keyword": "person executing solution successfully cinematic transformation",
      "lower_third": "2-4 words",
      "scene_style": "action_steps",
      "color_mood": "payoff",
      "motion_cue": "each action appears as a crisp step with motion and proof — warm amber lighting replaces red",
      "visual_metaphor": "practical toolkit or transformation — the exact opposite of the Section 8 danger scene",
      "human_behavior": "relief + control — the viewer sees exactly what to do and feels capable of doing it",
      "visual_beats": ["the problem that opened Section 1 now has a clear answer", "each step clicks into place with visual proof", "viewer sees the outcome transformation complete"],
      "first_frame": "a human beginning the practical fix — the mirror of Section 8's wrong choice, now done right",
      "mini_payoff": "Loop A + Loop D close simultaneously — viewer gets the double-dopamine hit",
      "pattern_interrupt": "visual transformation — the danger image from Section 8 becomes the solution image",
      "sound_cue": "three warm clicks ending with a payoff resonance — relief, not triumph",
      "asset_prompt": "cinematic practical transformation scene, warm amber light replacing red, hands doing the right action, real outcome visible, no checklist templates",
      "emotion_target": "relief",
      "peak_sentence": "The double-loop closure sentence — the one that answers both Loop A and Loop D simultaneously — verbatim",
      "cognitive_load": "heavy",
      "loop_action": "closes Loop A and Loop D simultaneously",
      "retention_bridge": "Tease the social proof that confirms this works — make the viewer want to see it validated"
    }},

    {{
      "name": "Proof + Call To Action",
      "narration": "TRIBAL IDENTITY REWARD: Open by closing Loop C (social proof that the solution works for others). Then deliver the EXCLUSIVE IN-GROUP framing: 'You now understand what 97% of people in this space never figure out.' Make the viewer feel they've been elevated into the group that knows. Then three asks in exact order: (1) LIKE — name the specific thing they just learned and why sharing it helps others like them. (2) COMMENT — one genuinely debate-worthy question that even lurkers feel compelled to answer. Make it personal and specific to the topic. (3) SUBSCRIBE — name exactly what the channel delivers weekly and why the next video is specifically connected to what they just watched. Callback to Section 1's opening image/promise — the full circle. Warm, confident, genuine — no robotic filler. (145-165 words)",
      "callout_text": "LIKE · COMMENT · SUBSCRIBE",
      "visual_keyword": "success transformation in-group belonging community cinematic",
      "lower_third": "Like · Comment · Subscribe",
      "scene_style": "finale",
      "color_mood": "payoff",
      "motion_cue": "callback to Section 1's first frame — now transformed — then warm pull-back to invite the comment",
      "visual_metaphor": "the before/after — Section 1's danger scene, now resolved and transformed",
      "human_behavior": "satisfaction + belonging — the viewer feels elevated into the group that knows",
      "visual_beats": ["Section 1's opening promise returns — now fully paid off", "before/after transformation visible", "specific debate-worthy comment question lands"],
      "first_frame": "the opening promise image from Section 1 — now showing the full resolution",
      "mini_payoff": "Loop C closes — viewer gets social proof AND in-group elevation",
      "pattern_interrupt": "full visual callback to Section 1 — the circle closes",
      "sound_cue": "final warm riser → payoff impact → clean resolution beat",
      "asset_prompt": "premium cinematic payoff and belonging scene — warm amber light, human face showing earned satisfaction, not forced happiness",
      "emotion_target": "belonging",
      "peak_sentence": "The tribal elevation sentence — 'You now know what 97%...' — verbatim from narration",
      "cognitive_load": "light",
      "loop_action": "closes Loop C — social proof validates the solution",
      "social_proof_anchor": "The 'you now know what 97%...' exclusive in-group framing sentence",
      "cta_type": "all",
      "retention_bridge": ""
    }}

  ]
}}"""

    provider = _choose_ai_provider()
    log.info(f"  AI provider: {provider}")

    if provider in ("cerebras", "groq", "sambanova", "gemini", "hf"):
        raw = _call_free_provider_json(prompt, _SYSTEM_PROMPT_WRITER, temperature=0.70)
    elif provider == "openai":
        raw = _call_openai_json(
            prompt,
            _SYSTEM_PROMPT_WRITER,
            temperature=0.75,
        )
    else:
        if anthropic is None:
            raise RuntimeError("anthropic package is not installed. Run: pip install anthropic")
        client = anthropic.Anthropic(api_key=CLAUDE_KEY)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=8000,
            system=_SYSTEM_PROMPT_WRITER,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

    return _validate_script(_parse_json_safe(raw))


def _call_script_model(prompt: str, temperature: float = 0.55) -> dict:
    active = _choose_ai_provider()
    if active in ("cerebras", "groq", "sambanova", "gemini", "hf"):
        raw = _call_free_provider_json(prompt, _SYSTEM_PROMPT_DIRECTOR, temperature)
    elif active == "openai":
        raw = _call_openai_json(
            prompt,
            _SYSTEM_PROMPT_DIRECTOR,
            temperature=temperature,
        )
    else:
        # anthropic
        import anthropic as _ant
        _aclient = _ant.Anthropic(api_key=CLAUDE_KEY)
        response = _aclient.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=8000,
            system=_SYSTEM_PROMPT_DIRECTOR,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
    return _validate_script(_parse_json_safe(raw))


def improve_script_with_quality_gate(script: dict, report, pass_num: int) -> dict:
    prompt = f"""Rewrite this YouTube script JSON to pass a monetization-focused quality gate.
Return ONLY the complete revised JSON object. Preserve the same schema and 10-section structure.

Quality score: {report.score}/100, threshold: {report.threshold}
Critical issues: {report.critical}
Warnings: {report.warnings}
Metrics: {json.dumps(report.metrics, ensure_ascii=False)}

Upgrade priorities:
1. Raise total narration to 1,500-1,750 words without filler.
2. Make hook 8-20 words but still shocking and specific.
3. Use at least 5 distinct retention_device values.
4. Add missing metadata to every section.
5. Avoid repeated claims, repeated callouts, and stale clickbait language.
6. Keep advertiser-friendly framing: informative, non-graphic, non-hateful, non-misleading.
7. Add/repair creator_standard with title_promise, first_8_seconds_visual, payoff_ladder, and freshness_rule.
8. Add/repair first_frame, mini_payoff, pattern_interrupt, sound_cue, and asset_prompt in every section.
9. Asset prompts must be cinematic and human or concrete: no abstract boxes, circles, generic arrows, dashboards, or gradient backgrounds.
10. Preserve premium creator motion direction, visual_beats, viewer_question, and contrast_pair.
11. Reduce boredom_risk metrics: no adjacent repeated rhythm, no generic viewer questions, no vague mini-payoffs, no repeated sentence openings, no early CTA overload.
12. Every section must add new information or a new human decision. If two sections feel similar, rewrite one with a different scene, emotion, proof type, and payoff.

Original JSON:
{json.dumps(script, ensure_ascii=False, indent=2)}
"""
    log.info(f"  Script rewrite pass {pass_num}: improving quality score {report.score}/{report.threshold}")
    return _call_script_model(prompt, temperature=0.45)


def generate_script_with_quality_loop(trending: list[dict], out_dir: Path, ts: str) -> dict:
    script = generate_script(trending)
    for pass_idx in range(max(0, SCRIPT_REWRITE_PASSES) + 1):
        report_path = out_dir / f"script_quality_{ts}_pass{pass_idx}.json"
        report = run_script_quality_gate(script, str(report_path), threshold=QUALITY_GATE_THRESHOLD, is_shorts=False)
        log.info(f"  Script quality pass {pass_idx}: {report.score}/100")
        if report.passed or pass_idx >= SCRIPT_REWRITE_PASSES:
            return script
        script = improve_script_with_quality_gate(script, report, pass_idx + 1)
    return script


def render_video(script: dict, video_path: str) -> None:
    """Render via MoviePy, Remotion, or Remotion-with-fallback."""
    renderer = VIDEO_RENDERER or "moviepy"
    if renderer in ("remotion", "remotion_next", "nextgen"):
        from remotion_renderer import make_video_from_script_remotion
        make_video_from_script_remotion(script, video_path)
        return

    if renderer in ("auto", "best"):
        try:
            from remotion_renderer import make_video_from_script_remotion
            make_video_from_script_remotion(script, video_path)
            log.info("  Renderer: Remotion")
            return
        except Exception as e:
            log.warning(f"  Remotion renderer unavailable ({e}); falling back to MoviePy")

    make_video_from_script(script, video_path)
    log.info("  Renderer: MoviePy")


def render_shorts(script: dict, video_path: str) -> None:
    """Render a Shorts video (1080×1920, 50-58 seconds) via MoviePy."""
    from video_maker import make_shorts_from_script
    make_shorts_from_script(script, video_path)
    log.info("  Renderer: MoviePy Shorts")


# ── Thumbnail generation ──────────────────────────────────────────────────────

def _generate_thumbnail(script: dict, thumb_path: str) -> bool:
    """
    Generate a YouTube thumbnail (1280×720 JPEG) from thumbnail_concept + title.

    Strategy:
    1. Parse thumbnail_concept for Pexels search keywords
    2. Download a high-quality Pexels photo (landscape orientation)
    3. Crop/resize to 1280×720
    4. Add bold title text overlay (bottom 30%) with dark gradient backing
    5. Save as JPEG (quality 95)

    Returns True on success, False on failure (caller logs the result).
    """
    import re
    import io

    try:
        import requests
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        log.warning("  Thumbnail generation skipped — requests/Pillow not installed")
        return False

    PEXELS_KEY = os.environ.get("PEXELS_API_KEY", "").strip()
    W, H = 1280, 720

    concept = script.get("thumbnail_concept", "")
    title   = script.get("video_title", "")

    # --- Build Pexels search query from concept ---
    # Strip common layout instructions; keep concrete nouns/adjectives
    stop_words = {
        "bold", "text", "background", "color", "dramatic", "expression",
        "overlay", "thumbnail", "close-up", "closeup", "split", "screen",
        "left", "right", "top", "bottom", "center", "frame", "image",
        "photo", "picture", "graphic", "alternative", "a/b", "test",
        "variant", "different", "framing", "position", "angle", "style",
    }
    words = re.findall(r"[a-zA-Z]{3,}", concept.lower())
    kw_words = [w for w in words if w not in stop_words]
    # Fall back to title words if concept yields too little
    if len(kw_words) < 2:
        kw_words = re.findall(r"[a-zA-Z]{3,}", title.lower())
    search_query = " ".join(kw_words[:5]) or title[:50] or "dramatic scene"

    # --- Pexels photo fetch ---
    bg_img = None
    if PEXELS_KEY:
        try:
            resp = requests.get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": PEXELS_KEY},
                params={"query": search_query, "per_page": 5,
                        "orientation": "landscape", "size": "large"},
                timeout=10,
            )
            photos = resp.json().get("photos", [])
            if photos:
                img_url = photos[0]["src"].get("large2x") or photos[0]["src"]["original"]
                img_resp = requests.get(img_url, timeout=20)
                bg_img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
        except Exception as _e:
            log.debug(f"  Pexels thumbnail fetch failed: {_e}")

    # --- Fallback: gradient ---
    if bg_img is None:
        bg_img = Image.new("RGB", (W, H), (18, 18, 28))
        draw_fb = ImageDraw.Draw(bg_img)
        for y in range(H):
            v = int(40 + (y / H) * 60)
            draw_fb.line([(0, y), (W, y)], fill=(v, v // 2, int(v * 0.3)))

    # --- Crop/resize to exactly 1280×720 ---
    img_w, img_h = bg_img.size
    scale = max(W / img_w, H / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    bg_img = bg_img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - W) // 2
    top  = (new_h - H) // 2
    bg_img = bg_img.crop((left, top, left + W, top + H))

    # --- Dark gradient overlay (bottom 35%) ---
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    grad_start = int(H * 0.55)
    for y in range(grad_start, H):
        alpha = int(210 * ((y - grad_start) / (H - grad_start)) ** 1.4)
        draw_ov.line([(0, y), (W, y)], fill=(0, 0, 0, alpha))
    bg_img = Image.alpha_composite(bg_img.convert("RGBA"), overlay).convert("RGB")

    # --- Text: use video title trimmed to ~5 words for impact ---
    draw = ImageDraw.Draw(bg_img)
    title_words = title.split()
    # Show up to 6 words; if longer, add "..."
    display_text = " ".join(title_words[:6])
    if len(title_words) > 6:
        display_text += "…"

    # Try to load a bold system font; fall back to default
    font_size = 72
    font = None
    font_candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for fc in font_candidates:
        if Path(fc).exists():
            try:
                font = ImageFont.truetype(fc, font_size)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    # Center text in lower zone
    bbox  = draw.textbbox((0, 0), display_text, font=font)
    tw    = bbox[2] - bbox[0]
    text_x = (W - tw) // 2
    text_y = H - 140

    # Drop shadow
    draw.text((text_x + 3, text_y + 3), display_text, font=font, fill=(0, 0, 0, 200))
    # White text
    draw.text((text_x, text_y), display_text, font=font, fill=(255, 255, 255))

    # --- Save ---
    bg_img.save(thumb_path, "JPEG", quality=95)
    log.info(f"  Thumbnail generated → {thumb_path}  ({W}×{H})")
    return True


# ── CTR readback helpers ──────────────────────────────────────────────────────

def _schedule_ctr_check(video_id: str, thumbnail_concept: str, output_dir: Path) -> None:
    """
    Schedule a one-time CTR readback 24 hours after upload via APScheduler.
    Fetches impressionClickThroughRate from YouTube Analytics API and saves
    a JSON report to output_dir/ctr_VIDEOID.json.
    Silently no-ops if YOUTUBE_API_KEY is missing or APScheduler fails.
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        import datetime as _dt

        def _fetch_ctr():
            try:
                from googleapiclient.discovery import build as _build
                from google.oauth2.credentials import Credentials as _Creds
                token_path = Path(__file__).parent / "token.json"
                if not token_path.exists():
                    return
                try:
                    token_data = json.loads(token_path.read_text(encoding="utf-8"))
                    token_scopes = set(token_data.get("scopes") or [])
                except Exception:
                    token_scopes = set()
                analytics_scope = "https://www.googleapis.com/auth/yt-analytics.readonly"
                if analytics_scope not in token_scopes:
                    log.info("CTR readback skipped: token.json does not include YouTube Analytics readonly scope.")
                    return
                creds = _Creds.from_authorized_user_file(str(token_path))
                analytics = _build("youtubeAnalytics", "v2", credentials=creds)
                end_date   = _dt.date.today().isoformat()
                start_date = (_dt.date.today() - _dt.timedelta(days=2)).isoformat()
                resp = analytics.reports().query(
                    ids=f"channel==MINE",
                    startDate=start_date,
                    endDate=end_date,
                    metrics="impressionClickThroughRate,impressions,views",
                    dimensions="video",
                    filters=f"video=={video_id}",
                ).execute()
                rows = resp.get("rows", [])
                ctr_data = {
                    "video_id": video_id,
                    "thumbnail_concept": thumbnail_concept,
                    "fetched_at": _dt.datetime.utcnow().isoformat(),
                    "rows": rows,
                    "ctr": rows[0][1] if rows else None,
                    "impressions": rows[0][2] if rows else None,
                    "views": rows[0][3] if rows else None,
                }
                report_path = output_dir / f"ctr_{video_id}.json"
                report_path.write_text(json.dumps(ctr_data, indent=2), encoding="utf-8")
                log.info(f"CTR report saved → {report_path} | CTR={ctr_data['ctr']}")
            except Exception as e:
                log.warning(f"CTR readback failed: {e}")

        sched = BackgroundScheduler()
        run_time = _dt.datetime.now() + _dt.timedelta(hours=24)
        sched.add_job(_fetch_ctr, "date", run_date=run_time)
        sched.start()
        log.info(f"CTR readback scheduled for {run_time.strftime('%Y-%m-%d %H:%M UTC')} (video_id={video_id})")
    except Exception as e:
        log.warning(f"Could not schedule CTR check: {e}")


def _summarize_recent_ctr(output_dir: Path, limit: int = 5) -> str:
    """
    Return a formatted summary of recent CTR reports for use in script generation context.
    Format: '- "{thumbnail_concept}": CTR={ctr:.1%} ({impressions} impressions)'
    """
    rows = []
    for path in sorted(output_dir.glob("ctr_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ctr = data.get("ctr")
            concept = data.get("thumbnail_concept", "")
            impressions = data.get("impressions", 0)
            if concept and ctr is not None:
                rows.append(f'- "{concept}": CTR={float(ctr):.1%} ({int(impressions or 0):,} impressions)')
        except Exception:
            continue
    return "\n".join(rows) if rows else "No CTR data available yet."


# ── Main pipeline ─────────────────────────────────────────────────────────────

_is_shorts_run: bool = True  # set by run_pipeline() before use


def run_pipeline(shorts: bool = False) -> None:
    global _is_shorts_run
    _is_shorts_run = shorts
    log.info("=" * 60)
    log.info(f"Pipeline started {'[SHORTS MODE]' if shorts else '[LONG-FORM MODE]'}")
    log.info("=" * 60)

    # Fast-fail: check client_secrets.json before doing any expensive work
    _secrets = Path(__file__).parent / "client_secrets.json"
    if not _secrets.exists():
        log.error(
            "\n\n  ❌  client_secrets.json not found — upload will fail.\n\n"
            "  Steps to fix (one-time only):\n"
            "  1. https://console.cloud.google.com/\n"
            "  2. APIs & Services → Credentials\n"
            "  3. + Create Credentials → OAuth 2.0 Client ID → Desktop app\n"
            "  4. Download JSON → rename to client_secrets.json\n"
            f"  5. Place it in: {Path(__file__).parent}\n"
        )
        return

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 0 — Paid AI credit information
    if AI_CREDIT_INFO not in {"off", "0", "false", "no"}:
        log.info("AI credit info — checking paid provider balances…")
        budget_checks = run_ai_budget_checks(shorts=shorts)
        for item in budget_checks:
            message = format_budget_check(item)
            if item.active and item.status in {"unknown", "missing-key", "error"}:
                log.warning(f"  {message} — {item.detail}")
            else:
                log.info(f"  {message}")

    # 1 — Trending
    log.info("Step 1/4 — Fetching trending videos…")
    try:
        trending = fetch_trending(MAX_RESULTS)
        log.info(f"  Got {len(trending)} trending videos")
    except Exception as e:
        log.error(f"Fetch failed: {e}")
        return

    is_shorts = _is_shorts_run

    # 2 — Script
    if is_shorts:
        log.info("Step 2/4 — Generating Shorts script (45-55s)…")
        try:
            script      = generate_shorts_script(trending)
            script_path = out_dir / f"script_shorts_{ts}.json"
            script_path.write_text(json.dumps(script, indent=2, ensure_ascii=False))
            log.info(f"  Shorts title: {script['video_title']}")
            log.info(f"  Script saved → {script_path}")
        except Exception as e:
            log.error(f"Shorts script generation failed: {e}")
            return
    else:
        log.info("Step 2/4 — Generating creator-grade script…")
        try:
            script      = generate_script_with_quality_loop(trending, out_dir, ts)
            # Inject chapter timestamps into description before saving/upload
            script      = _inject_chapter_timestamps(script)
            script_path = out_dir / f"script_{ts}.json"
            script_path.write_text(json.dumps(script, indent=2, ensure_ascii=False))
            log.info(f"  Title: {script['video_title']}")
            log.info(f"  Script saved → {script_path}")
        except Exception as e:
            log.error(f"Script generation failed: {e}")
            return

    # 2b — Preview quality gate (fast static frames — runs in seconds, no full render)
    log.info("Step 2b/4 — Preview quality gate (static frame check)…")
    try:
        from preview_quality import run_preview
        preview_report = run_preview(script, out_dir, is_shorts=is_shorts)
        critical_frames = preview_report.get("critical_frames", [])
        if critical_frames:
            for cf in critical_frames:
                log.warning(f"  ⚠   Preview issue [{cf['frame']}]: {cf['issue']}")
            log.warning("  Preview flagged issues — proceeding to render (fix script to resolve)")
        else:
            log.info("  ✅  Preview frames look clean — proceeding to render")
    except Exception as _pq_err:
        log.warning(f"  Preview gate skipped: {_pq_err}")

    # 3 — Video
    if is_shorts:
        log.info("Step 3/4 — Rendering Shorts video (portrait 1080×1920)…")
        video_path = str(out_dir / f"shorts_{ts}.mp4")
        try:
            render_shorts(script, video_path)
            log.info(f"  Shorts video saved → {video_path}")
        except Exception as e:
            log.error(f"Shorts render failed: {e}")
            return
    else:
        log.info("Step 3/4 — Rendering video…")
        video_path = str(out_dir / f"video_{ts}.mp4")
        try:
            render_video(script, video_path)
            log.info(f"  Video saved → {video_path}")
        except Exception as e:
            log.error(f"Video render failed: {e}")
            return

    # ── Runtime guard: YouTube removes videos > 15 min on unverified channels ──
    YT_MAX_SECONDS  = 15 * 60   # 900 s hard cap (unverified long-form)
    SAFE_CEILING    = 14 * 60   # 840 s — warn if we creep above here
    SHORTS_MAX_SEC  = 60        # Shorts hard cap
    try:
        from moviepy import VideoFileClip as _VFC
        with _VFC(video_path, audio=False) as _vc:
            _dur = _vc.duration
        mins, secs = divmod(int(_dur), 60)
        log.info(f"  Video duration: {mins}m {secs:02d}s")
        if is_shorts:
            if _dur > SHORTS_MAX_SEC:
                log.warning(
                    f"  ⚠   Shorts video is {_dur:.0f}s — over the 60s Shorts cap. "
                    f"YouTube may treat it as long-form. Reduce narration word count."
                )
            else:
                log.info(f"  ✅  Shorts duration OK ({_dur:.0f}s / 60s max)")
        else:
            if _dur > YT_MAX_SECONDS:
                log.error(
                    f"  ❌  Video is {mins}m {secs:02d}s — exceeds YouTube's 15-min "
                    f"limit for unverified channels. Reduce word counts in generate_script()."
                )
                return
            elif _dur > SAFE_CEILING:
                log.warning(
                    f"  ⚠   Video is {mins}m {secs:02d}s — approaching the 15-min cap. "
                    f"Consider verifying your channel for unlimited length."
                )
    except Exception as _e:
        log.warning(f"  Duration check skipped: {_e}")

    # 3b — Thumbnail
    thumb_path = video_path.replace(".mp4", "_thumb.jpg")
    if not Path(thumb_path).exists():
        log.info("Step 3b/4 — Generating thumbnail (1280×720)…")
        try:
            ok = _generate_thumbnail(script, thumb_path)
            if not ok:
                log.warning("  Thumbnail generation failed — quality gate will warn")
        except Exception as _te:
            log.warning(f"  Thumbnail generation error: {_te}")
    else:
        log.info(f"  Thumbnail already exists → {thumb_path}")

    # 4 — Upload
    log.info("Quality gate — analyzing monetization readiness…")
    quality_report_path = str(out_dir / f"quality_{ts}.json")
    srt_path   = video_path.replace(".mp4", ".srt")
    if QUALITY_GATE_MODE != "off":
        try:
            shorts_threshold = max(55, QUALITY_GATE_THRESHOLD - 20)
            report = run_quality_gate(
                script=script,
                video_path=video_path,
                thumbnail_path=thumb_path,
                srt_path=srt_path,
                report_path=quality_report_path,
                threshold=shorts_threshold if is_shorts else QUALITY_GATE_THRESHOLD,
                is_shorts=is_shorts,
            )
            log.info(f"  Quality score: {report.score}/100 (threshold {report.threshold})")
            log.info(f"  Quality report → {quality_report_path}")
            for item in report.critical:
                log.error(f"  Quality critical: {item}")
            for item in report.warnings[:8]:
                log.warning(f"  Quality warning: {item}")
            if not report.passed and QUALITY_GATE_MODE == "block":
                log.error("  ❌  Quality gate blocked upload. Set QUALITY_GATE_MODE=warn to override.")
                return
        except Exception as e:
            log.error(f"Quality gate failed: {e}")
            if QUALITY_GATE_MODE == "block":
                return
    else:
        log.info("  Quality gate disabled by QUALITY_GATE_MODE=off")

    log.info("Step 4/4 — Uploading to YouTube…")
    try:
        youtube  = get_authenticated_service()
        upload_result = upload_video(
            youtube        = youtube,
            video_path     = video_path,
            title          = script["video_title"],
            description    = script["description"],
            tags           = script["tags"],
            category_id    = script.get("category_id", "22"),
            privacy        = PRIVACY,
            thumbnail_path = thumb_path,
            srt_path       = srt_path,
        )
        if hasattr(upload_result, "video_id"):
            video_id = upload_result.video_id
            url = upload_result.url
            thumbnail_status = "uploaded" if upload_result.thumbnail_uploaded else (
                f"not uploaded — {upload_result.thumbnail_error}" if upload_result.thumbnail_error else "not uploaded"
            )
            captions_status = "uploaded" if upload_result.captions_uploaded else (
                f"not uploaded — {upload_result.captions_error}" if upload_result.captions_error else "auto-generated by YouTube"
            )
            post_upload_warnings = upload_result.warnings
        else:
            video_id = str(upload_result)
            url = f"https://www.youtube.com/watch?v={video_id}"
            thumbnail_status = thumb_path if Path(thumb_path).exists() else "not generated"
            captions_status = srt_path if Path(srt_path).exists() else "auto-generated by YouTube"
            post_upload_warnings = []
        log.info(f"  Uploaded → {url}")

        # Schedule CTR readback 24 h after upload
        if video_id:
            thumb_concept = script.get("thumbnail_concept", "")
            out_dir_ctr = Path(video_path).parent
            _schedule_ctr_check(video_id, thumb_concept, out_dir_ctr)

        print(f"\n{'='*60}")
        print(f"  ✅  Video uploaded!")
        print(f"  Title    : {script['video_title']}")
        print(f"  URL      : {url}")
        print(f"  Privacy  : {PRIVACY}")
        print(f"  Hook     : {script.get('hook', '')}")
        print(f"  Thumbnail: {thumbnail_status}")
        print(f"  Captions : {captions_status}")
        for warning in post_upload_warnings:
            print(f"  Warning  : {warning}")
        print(f"{'='*60}\n")
    except FileNotFoundError as e:
        log.error(str(e))
        return
    except Exception as e:
        log.error(f"Upload failed: {e}")
        return

    log.info("Pipeline complete ✅")


# ── Scheduler ─────────────────────────────────────────────────────────────────

def run_scheduled() -> None:
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        print("APScheduler not installed. Run:  pip install APScheduler")
        return

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        run_pipeline,
        trigger="cron",
        hour=SCHEDULE_HOUR,
        minute=0,
        id="yt_pipeline",
        replace_existing=True,
    )

    print(f"\n  Scheduler active — pipeline runs every day at {SCHEDULE_HOUR:02d}:00 UTC")
    print(f"  Privacy setting : {PRIVACY}")
    print("  Press Ctrl+C to stop\n")
    log.info(f"Scheduler started — daily at {SCHEDULE_HOUR:02d}:00 UTC")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("Scheduler stopped by user")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YouTube Auto-Upload Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 pipeline.py              Run the pipeline once (long-form, 10-12 min)
  python3 pipeline.py --shorts     Run once as a Shorts video (50-58 seconds, 9:16)
  python3 pipeline.py --schedule   Run daily at SCHEDULE_HOUR (from .env)
  python3 pipeline.py --schedule --shorts  Schedule daily Shorts uploads
        """,
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run on a daily schedule instead of once",
    )
    parser.add_argument(
        "--shorts", action="store_true",
        help="Generate and upload a YouTube Short (50-58 seconds, portrait 1080×1920)",
    )
    args = parser.parse_args()

    # SHORTS_MODE env var can also force shorts: 'only' or 'also'
    force_shorts = args.shorts or SHORTS_MODE in ("only", "yes", "true", "1")

    if args.schedule:
        run_scheduled()
    else:
        run_pipeline(shorts=force_shorts)
