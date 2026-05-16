#!/usr/bin/env python3
"""
pipeline.py — YouTube Auto-Upload Pipeline (Session 7 — Monetization Maximisation)

Psychological retention architecture (10 sections), quality gate (target 88+/100),
script rewrite loop, Shorts generation, and automated upload for both formats.

Usage:
  python pipeline.py              # run once right now
  python pipeline.py --schedule   # run daily at SCHEDULE_HOUR (set in .env)
"""

import os
import re
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

import anthropic
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from video_maker      import make_video_from_script, make_shorts_from_script
from youtube_uploader import get_authenticated_service, upload_video, upload_thumbnail

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
YT_API_KEY           = os.environ.get("YOUTUBE_API_KEY", "")
CHANNEL_ID           = os.environ.get("YOUTUBE_CHANNEL_ID", "")
CLAUDE_KEY           = os.environ.get("ANTHROPIC_API_KEY", "")
REGION               = os.environ.get("DEFAULT_REGION", "US")
MAX_RESULTS          = int(os.environ.get("MAX_RESULTS", "25"))
PRIVACY              = os.environ.get("YOUTUBE_PRIVACY", "private")
SCHEDULE_HOUR        = int(os.environ.get("SCHEDULE_HOUR", "9"))
QUALITY_THRESHOLD    = int(os.environ.get("QUALITY_GATE_THRESHOLD", "88"))
SCRIPT_REWRITE_PASSES = int(os.environ.get("SCRIPT_REWRITE_PASSES", "2"))
SHORTS_MODE          = os.environ.get("SHORTS_MODE", "also")   # off | also | only
QUALITY_GATE_MODE    = os.environ.get("QUALITY_GATE_MODE", "block")  # block | warn


# ── JSON resilience ───────────────────────────────────────────────────────────

def _repair_json(raw: str) -> str:
    """Walk char-by-char, close unterminated strings, append missing brackets."""
    in_string = False
    escape    = False
    stack     = []
    result    = []
    for ch in raw:
        if escape:
            result.append(ch)
            escape = False
            continue
        if ch == "\\" and in_string:
            result.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if stack and stack[-1] == ch:
                    stack.pop()
        result.append(ch)
    if in_string:
        result.append('"')
    while stack:
        result.append(stack.pop())
    return "".join(result)


def _parse_json_safe(raw: str) -> dict:
    """3 attempts: clean → repair → scan largest sub-object."""
    text = raw.strip()
    # Strip markdown fences
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: repair then parse
    try:
        return json.loads(_repair_json(text))
    except json.JSONDecodeError:
        pass

    # Attempt 3: scan for the largest embedded JSON object
    start = text.find("{")
    if start >= 0:
        for end in range(len(text), start, -1):
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from LLM response (first 300 chars): {text[:300]}")


# ── Step 1: Fetch trending videos ─────────────────────────────────────────────

def fetch_trending_youtube(n: int = 25) -> list[dict]:
    yt = build("youtube", "v3", developerKey=YT_API_KEY)
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


# ── Chapter timestamp builder ─────────────────────────────────────────────────

def _build_chapter_timestamps(sections: list[dict]) -> str:
    """Estimate chapter start times from narration word counts at 150 wpm."""
    WPM = 150
    elapsed = 0.0
    lines = []
    for sec in sections:
        words   = len((sec.get("narration") or "").split())
        mins, s = divmod(int(elapsed), 60)
        lines.append(f"{mins}:{s:02d} {sec.get('name', '')}")
        elapsed += (words / WPM) * 60
    return "\n".join(lines)


def _inject_timestamps(script: dict) -> dict:
    """Replace the LLM's placeholder timestamps with accurate word-count estimates."""
    sections = script.get("sections", [])
    if not sections:
        return script
    ts_block = _build_chapter_timestamps(sections)
    desc = script.get("description", "")
    # Remove any existing timestamp block (lines matching M:SS pattern)
    cleaned = re.sub(r"(\d+:\d{2}[^\n]*\n?)+", "", desc).strip()
    script["description"] = cleaned + "\n\n" + ts_block
    return script


# ── Step 2: Generate 10-section psychological script ─────────────────────────

_SCRIPT_SYSTEM = """\
You are a world-class YouTube scriptwriter specialising in psychological retention architecture.
Your scripts achieve 85%+ audience retention and accelerate monetisation eligibility.
You ALWAYS return ONLY valid JSON — no markdown fences, no explanation, no preamble.
"""

_SCRIPT_PROMPT = """\
TODAY'S TRENDING TOPICS:
{trend_lines}

TASK: Choose the single BEST topic for maximum watch-time and viral potential.
Write a 10-section psychological script following the EXACT architecture below.

━━━ MANDATORY 10-SECTION ARCHITECTURE ━━━

S1  HOOK                      | emotion: shock            | load: light
    Opens Loop A. First 30-45 s. Most arresting opening possible — shocking statistic,
    impossible-seeming claim, or visceral scene. No "Hey guys". Straight to the hook.
    rewatch_clue: hide one subtle detail that ONLY rewatchers notice and feel rewarded for.

S2  WHY THIS MATTERS NOW      | emotion: dread            | load: heavy
    Sustains Loop A, opens social proof thread. Establish stakes — why THIS WEEK, not last year.
    social_proof_anchor: cite a real named authority, study, or statistic.

S3  THE REAL PROBLEM          | emotion: recognition      | load: light
    Opens Loop B. MUST open with the exact words: "You've been in this exact situation…"
    then paint a scene the viewer has LIVED — not a hypothetical.

S4  WHAT EVERYONE GETS WRONG  | emotion: cognitive_dissonance | load: heavy
    Closes Loop B, sustains Loop A. First: validate the common belief ("That makes sense, because…")
    Then: demolish it with 2-3 unarguable evidence points + one named authority.

S5  THE HIDDEN TRUTH          | emotion: confusion        | load: light
    Fake-out + opens Loop C. Deliver a confident WRONG answer first (40-50 words, written as truth).
    Then: "But here's what no one tells you…" and shatter it. Double-dopamine hit.
    fake_out_content = the wrong-answer paragraph ONLY.

S6  DEEP DIVE — THE EVIDENCE  | emotion: analytical_trust | load: heavy
    Sustains Loop C, opens Loop D. After the most surprising proof, insert:
    "In [N] minutes, you'll see the ONE mistake that destroys even people who know all of this."
    mid_video_tease = that exact sentence. social_proof_anchor = data / research / expert.

S7  DEEP DIVE — HOW IT WORKS  | emotion: intellectual_awe | load: heavy
    Sustains Loop C, escalates Loop D. One instantly graspable analogy ("This works exactly like X…").
    3 cause-and-effect steps. Real case study with names AND numbers.

S8  THE BIGGEST MISTAKE       | emotion: anxiety          | load: light
    Loop D peaks at 65-70% mark — most visceral moment. Real cautionary example: names + outcomes.
    Viewer must think "This could be happening to me RIGHT NOW."

S9  THE SOLUTION              | emotion: relief           | load: heavy
    Closes Loop A AND Loop D simultaneously. Concrete, actionable payoff.
    Must feel EARNED after 8 sections of buildup.

S10 PROOF + CTA               | emotion: belonging        | load: light
    Closes Loop C — LAST loop to close for maximum wait-tension.
    "You now understand what 97% of people in this space never figure out."
    CTA order: like → comment → subscribe (never change this order).
    social_proof_anchor = success example that validates the viewer's new identity.

━━━ OPEN LOOP DISCIPLINE ━━━
A: opened S1 → sustained S2,S4 → closed S9
B: opened S3 → closed S4
C: opened S5 → sustained S6,S7 → closed S10  ← last to close
D: opened S6 → escalated S7 → peaked S8 → closed S9

━━━ COGNITIVE LOAD PATTERN (mandatory) ━━━
S1:light S2:heavy S3:light S4:heavy S5:light S6:heavy S7:heavy S8:light S9:heavy S10:light

━━━ SEO REQUIREMENTS ━━━
title: 40-70 chars, curiosity-gap + high-value keyword, no clickbait
description: 400-500 chars, chapter timestamps 0:00 onward, ends with subscribe CTA
tags: exactly 10 tags — broad + niche + long-tail mix
category_id: correct category for the topic (22 = People & Blogs is fine for general topics)

━━━ RETURN ONLY VALID JSON ━━━

{{
  "video_title": "40-70 char SEO title with curiosity gap",
  "description": "400-500 char description.\\n\\n0:00 Hook\\n...\\nSubscribe for weekly insights.",
  "tags": ["tag1","tag2","tag3","tag4","tag5","tag6","tag7","tag8","tag9","tag10"],
  "category_id": "22",
  "sections": [
    {{
      "name": "Hook",
      "narration": "130-180 words. Shocking opening. No greeting — straight to the arresting claim.",
      "visual_keyword": "2-3 word Pexels search term",
      "lower_third": "2-4 word on-screen overlay",
      "emotion_target": "shock",
      "cognitive_load": "light",
      "loop_action": "Opens Loop A",
      "peak_sentence": "The single most powerful sentence in this section — verbatim, at least 10 words",
      "rewatch_clue": "The subtle detail hidden here that rewatchers notice",
      "social_proof_anchor": "",
      "fake_out_content": "",
      "mid_video_tease": "",
      "retention_bridge": "One-line tease of S2 to display as 'NEXT:' overlay in final 28% of this section"
    }},
    {{
      "name": "Why This Matters Now",
      "narration": "200-250 words. Establish stakes. Why THIS WEEK. Recency signals.",
      "visual_keyword": "2-3 word Pexels search term",
      "lower_third": "2-4 word on-screen overlay",
      "emotion_target": "dread",
      "cognitive_load": "heavy",
      "loop_action": "Sustains Loop A, opens social proof thread",
      "peak_sentence": "Most powerful sentence verbatim ≥10 words",
      "rewatch_clue": "",
      "social_proof_anchor": "Named authority, study, or statistic cited in narration",
      "fake_out_content": "",
      "mid_video_tease": "",
      "retention_bridge": "One-line tease of S3"
    }},
    {{
      "name": "The Real Problem",
      "narration": "180-220 words. MUST open: 'You\\'ve been in this exact situation…' then paint a lived scene.",
      "visual_keyword": "2-3 word Pexels search term",
      "lower_third": "2-4 word on-screen overlay",
      "emotion_target": "recognition",
      "cognitive_load": "light",
      "loop_action": "Opens Loop B",
      "peak_sentence": "Most powerful sentence verbatim ≥10 words",
      "rewatch_clue": "",
      "social_proof_anchor": "",
      "fake_out_content": "",
      "mid_video_tease": "",
      "retention_bridge": "One-line tease of S4"
    }},
    {{
      "name": "What Everyone Gets Wrong",
      "narration": "200-250 words. Validate belief first ('That makes sense, because…'). Then demolish with evidence + authority.",
      "visual_keyword": "2-3 word Pexels search term",
      "lower_third": "2-4 word on-screen overlay",
      "emotion_target": "cognitive_dissonance",
      "cognitive_load": "heavy",
      "loop_action": "Closes Loop B, sustains Loop A",
      "peak_sentence": "Most powerful sentence verbatim ≥10 words",
      "rewatch_clue": "",
      "social_proof_anchor": "",
      "fake_out_content": "",
      "mid_video_tease": "",
      "retention_bridge": "One-line tease of S5"
    }},
    {{
      "name": "The Hidden Truth",
      "narration": "180-220 words. Fake answer first, then shatter: 'But here\\'s what no one tells you…'",
      "visual_keyword": "2-3 word Pexels search term",
      "lower_third": "2-4 word on-screen overlay",
      "emotion_target": "confusion",
      "cognitive_load": "light",
      "loop_action": "Closes fake-out, opens Loop C",
      "peak_sentence": "Most powerful sentence verbatim ≥10 words",
      "rewatch_clue": "",
      "social_proof_anchor": "",
      "fake_out_content": "The wrong-answer paragraph only (~40-50 words), written as truth before being shattered",
      "mid_video_tease": "",
      "retention_bridge": "One-line tease of S6"
    }},
    {{
      "name": "Deep Dive — The Evidence",
      "narration": "250-300 words. Hard evidence. Insert mid_video_tease after the most surprising proof.",
      "visual_keyword": "2-3 word Pexels search term",
      "lower_third": "2-4 word on-screen overlay",
      "emotion_target": "analytical_trust",
      "cognitive_load": "heavy",
      "loop_action": "Sustains Loop C, opens Loop D",
      "peak_sentence": "Most powerful sentence verbatim ≥10 words",
      "rewatch_clue": "",
      "social_proof_anchor": "Data, research, or expert cited in narration",
      "fake_out_content": "",
      "mid_video_tease": "Exact sentence: 'In N minutes, you\\'ll see the ONE mistake that destroys even people who know all of this.'",
      "retention_bridge": "One-line tease of S7"
    }},
    {{
      "name": "Deep Dive — How It Works",
      "narration": "250-300 words. One analogy ('This works exactly like X…'). 3 cause-effect steps. Real case study names+numbers.",
      "visual_keyword": "2-3 word Pexels search term",
      "lower_third": "2-4 word on-screen overlay",
      "emotion_target": "intellectual_awe",
      "cognitive_load": "heavy",
      "loop_action": "Sustains Loop C, escalates Loop D",
      "peak_sentence": "Most powerful sentence verbatim ≥10 words",
      "rewatch_clue": "",
      "social_proof_anchor": "",
      "fake_out_content": "",
      "mid_video_tease": "",
      "retention_bridge": "One-line tease of S8"
    }},
    {{
      "name": "The Biggest Mistake",
      "narration": "180-220 words. MOST VISCERAL SECTION. Real cautionary example with names and outcomes. 'This could be happening to me RIGHT NOW.'",
      "visual_keyword": "2-3 word Pexels search term",
      "lower_third": "2-4 word on-screen overlay",
      "emotion_target": "anxiety",
      "cognitive_load": "light",
      "loop_action": "Loop D peaks — 65-70% mark",
      "peak_sentence": "Most powerful sentence verbatim ≥10 words",
      "rewatch_clue": "",
      "social_proof_anchor": "",
      "fake_out_content": "",
      "mid_video_tease": "",
      "retention_bridge": "One-line tease of S9"
    }},
    {{
      "name": "The Solution",
      "narration": "200-250 words. Concrete actionable payoff. Closes Loop A + Loop D simultaneously. Must feel EARNED.",
      "visual_keyword": "2-3 word Pexels search term",
      "lower_third": "2-4 word on-screen overlay",
      "emotion_target": "relief",
      "cognitive_load": "heavy",
      "loop_action": "Closes Loop A and Loop D simultaneously",
      "peak_sentence": "Most powerful sentence verbatim ≥10 words",
      "rewatch_clue": "",
      "social_proof_anchor": "",
      "fake_out_content": "",
      "mid_video_tease": "",
      "retention_bridge": "One-line tease of S10"
    }},
    {{
      "name": "Proof and CTA",
      "narration": "120-160 words. 'You now understand what 97% of people never figure out.' CTA: like → comment → subscribe.",
      "visual_keyword": "youtube subscribe channel community",
      "lower_third": "Like & Subscribe",
      "emotion_target": "belonging",
      "cognitive_load": "light",
      "loop_action": "Closes Loop C — tribal identity reward",
      "peak_sentence": "Most powerful sentence verbatim ≥10 words",
      "rewatch_clue": "",
      "social_proof_anchor": "Success example that validates the viewer's new identity",
      "fake_out_content": "",
      "mid_video_tease": "",
      "retention_bridge": ""
    }}
  ]
}}"""


def generate_script(trending: list[dict]) -> dict:
    client = anthropic.Anthropic(api_key=CLAUDE_KEY)

    trend_lines = "\n".join(
        f'- "{v["title"]}"  ({v["views"]:,} views, by {v["channel"]})'
        for v in trending[:12]
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=_SCRIPT_SYSTEM,
        messages=[{
            "role": "user",
            "content": _SCRIPT_PROMPT.format(trend_lines=trend_lines),
        }],
    )
    return _parse_json_safe(response.content[0].text)


# ── Quality gate ──────────────────────────────────────────────────────────────

def score_script(script: dict) -> tuple[int, list[str]]:
    """
    Score the script 0-100 against all psychological + SEO mandates.
    Returns (score, list_of_failure_reasons).
    Target: 88+.
    """
    score    = 100
    failures = []
    sections = script.get("sections", [])

    # Section count (fatal — 20 pts)
    if len(sections) != 10:
        failures.append(f"Expected 10 sections, got {len(sections)}")
        score -= 20
        return max(0, score), failures

    # peak_sentence presence in every section (20 pts — 2 per section)
    for i, sec in enumerate(sections):
        ps = (sec.get("peak_sentence") or "").strip()
        if len(ps.split()) < 4:
            failures.append(f"S{i+1} peak_sentence too short or missing: '{ps}'")
            score -= 2

    # Loop discipline (14 pts — 2 per check)
    loop_checks = [
        (0, "loop_action", "loop a",  "S1 must open Loop A"),
        (2, "loop_action", "loop b",  "S3 must open Loop B"),
        (4, "loop_action", "loop c",  "S5 must open Loop C"),
        (5, "loop_action", "loop d",  "S6 must open Loop D"),
        (3, "loop_action", "close",   "S4 must close a loop"),
        (8, "loop_action", "close",   "S9 must close loops"),
        (9, "loop_action", "close",   "S10 must close Loop C"),
    ]
    for idx, field, keyword, msg in loop_checks:
        val = (sections[idx].get(field) or "").lower()
        if keyword not in val:
            failures.append(msg)
            score -= 2

    # Section-specific mandates (20 pts)
    # S1: rewatch_clue (4 pts)
    if not (sections[0].get("rewatch_clue") or "").strip():
        failures.append("S1 missing rewatch_clue")
        score -= 4

    # S3: identity mirror (4 pts)
    s3n = (sections[2].get("narration") or "").lower()
    if "you've been in this exact situation" not in s3n and "you have been in this exact" not in s3n:
        failures.append("S3 must open with \"You've been in this exact situation…\"")
        score -= 4

    # S5: fake_out_content (4 pts)
    if not (sections[4].get("fake_out_content") or "").strip():
        failures.append("S5 missing fake_out_content (the wrong-answer paragraph)")
        score -= 4

    # S6: mid_video_tease (4 pts)
    if not (sections[5].get("mid_video_tease") or "").strip():
        failures.append("S6 missing mid_video_tease (forward reference to S8 danger)")
        score -= 4

    # S10: tribal identity (4 pts)
    s10n = (sections[9].get("narration") or "").lower()
    if "97%" not in s10n and "most people" not in s10n and "never figure" not in s10n:
        failures.append("S10 missing tribal identity reward (must reference '97%' or 'most people')")
        score -= 4

    # Social proof anchors in S2, S6, S10 (9 pts — 3 each)
    for idx, label in [(1, "S2"), (5, "S6"), (9, "S10")]:
        if not (sections[idx].get("social_proof_anchor") or "").strip():
            failures.append(f"{label} missing social_proof_anchor")
            score -= 3

    # SEO checks (17 pts)
    title = (script.get("video_title") or "")
    if len(title) > 70:
        failures.append(f"Title too long ({len(title)} chars, max 70)")
        score -= 3
    if len(title) < 30:
        failures.append("Title too short (< 30 chars)")
        score -= 2
    tags = script.get("tags", [])
    if len(tags) < 8:
        failures.append(f"Too few tags ({len(tags)}, need 8+)")
        score -= 4
    desc = (script.get("description") or "")
    if "subscribe" not in desc.lower():
        failures.append("Description missing subscribe CTA")
        score -= 3
    if len(desc) < 300:
        failures.append("Description too short (< 300 chars)")
        score -= 3
    if "0:00" not in desc:
        failures.append("Description missing chapter timestamps")
        score -= 2

    return max(0, score), failures


# ── Script rewriter ───────────────────────────────────────────────────────────

def _rewrite_script(script: dict, failures: list[str]) -> dict:
    """Send the failing script + failure list back to Claude for targeted fixes."""
    client = anthropic.Anthropic(api_key=CLAUDE_KEY)

    failure_block = "\n".join(f"- {f}" for f in failures)
    prompt = f"""\
The script below failed the quality gate. Fix EVERY listed failure without changing
the topic, title, or overall structure. Return ONLY the corrected JSON.

FAILURES TO FIX:
{failure_block}

CURRENT SCRIPT JSON:
{json.dumps(script, indent=2, ensure_ascii=False)}
"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=_SCRIPT_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_json_safe(response.content[0].text)


# ── Step 3: Generate Shorts script ───────────────────────────────────────────

_SHORTS_PROMPT = """\
You are a YouTube Shorts specialist. Topic: {topic}

Write a 60-second Short script with Zeigarnik retention mechanics.

RULES:
- narration: 95-110 words MAXIMUM — no overrun
- First line: instant hook, no greeting
- comment_bait: one burning question to drive comments
- identity_mirror: the "that's exactly me" moment to trigger sharing
- end_screen_hook: 2-second tease driving to the long-form video
- chunks: 4-5 narration segments, each ~20-25 words

Return ONLY valid JSON:
{{
  "short_title": "Title under 60 chars — curiosity-gap hook",
  "description": "1-2 sentence description. #shorts #[topic] #[keyword]",
  "tags": ["shorts","tag2","tag3","tag4","tag5"],
  "narration": "Full 95-110 word script — the exact words spoken",
  "hook_sentence": "First 12 words shown as opening text overlay",
  "comment_bait": "Single question pinned to drive comments",
  "identity_mirror": "The 'that's exactly me' moment shown as mid-video text overlay",
  "end_screen_hook": "2-second text at end: 'Full breakdown →' style tease",
  "chunks": [
    {{"text": "~20-25 word narration chunk", "visual_keyword": "Pexels keyword", "overlay_text": "2-4 word on-screen text"}},
    {{"text": "chunk 2", "visual_keyword": "keyword", "overlay_text": "overlay"}},
    {{"text": "chunk 3", "visual_keyword": "keyword", "overlay_text": "overlay"}},
    {{"text": "chunk 4", "visual_keyword": "keyword", "overlay_text": "overlay"}}
  ]
}}"""


def generate_shorts_script(full_script: dict) -> dict:
    client = anthropic.Anthropic(api_key=CLAUDE_KEY)
    topic  = full_script.get("video_title", "the topic")
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=_SCRIPT_SYSTEM,
        messages=[{"role": "user", "content": _SHORTS_PROMPT.format(topic=topic)}],
    )
    return _parse_json_safe(response.content[0].text)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    log.info("=" * 60)
    log.info("Pipeline started — Session 7 Monetisation Maximisation")
    log.info("=" * 60)

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

    # ── 1: Trending ──────────────────────────────────────────────────────────
    log.info("Step 1/5 — Fetching trending videos…")
    try:
        trending = fetch_trending(MAX_RESULTS)
        log.info(f"  Got {len(trending)} trending topics")
    except Exception as e:
        log.error(f"Fetch failed: {e}")
        return

    # ── 2: Script + quality gate loop ───────────────────────────────────────
    log.info("Step 2/5 — Generating 10-section psychological script…")
    script = None
    for attempt in range(1 + SCRIPT_REWRITE_PASSES):
        try:
            if attempt == 0:
                script = generate_script(trending)
            else:
                log.info(f"  Rewrite pass {attempt}/{SCRIPT_REWRITE_PASSES}…")
                script = _rewrite_script(script, failures)
        except Exception as e:
            log.error(f"  Script generation error (attempt {attempt+1}): {e}")
            if attempt >= SCRIPT_REWRITE_PASSES:
                return
            continue

        script = _inject_timestamps(script)
        score, failures = score_script(script)
        log.info(f"  Quality score: {score}/100  (threshold: {QUALITY_THRESHOLD})")

        if failures:
            for f in failures:
                log.info(f"    ✗ {f}")

        if score >= QUALITY_THRESHOLD:
            log.info(f"  ✅ Quality gate passed ({score}/100)")
            break

        if QUALITY_GATE_MODE == "warn":
            log.warning(f"  ⚠ Quality gate NOT passed ({score}/100) — continuing (warn mode)")
            break

        if attempt < SCRIPT_REWRITE_PASSES:
            log.info(f"  Quality gate NOT passed — requesting rewrite…")
        else:
            log.warning(f"  Quality gate NOT passed after all rewrite passes. Proceeding with best script ({score}/100).")

    script_path = out_dir / f"script_{ts}.json"
    script_path.write_text(json.dumps(script, indent=2, ensure_ascii=False))
    log.info(f"  Title: {script['video_title']}")
    log.info(f"  Script saved → {script_path}")

    # ── 3: Long-form video ───────────────────────────────────────────────────
    if SHORTS_MODE != "only":
        log.info("Step 3/5 — Rendering long-form video…")
        video_path = str(out_dir / f"video_{ts}.mp4")
        try:
            make_video_from_script(script, video_path)
            log.info(f"  Video saved → {video_path}")
        except Exception as e:
            log.error(f"Long-form render failed: {e}")
            video_path = None
    else:
        video_path = None

    # ── 4: Shorts video ──────────────────────────────────────────────────────
    shorts_path = None
    if SHORTS_MODE in ("also", "only"):
        log.info("Step 4/5 — Generating & rendering Short…")
        try:
            shorts_script = generate_shorts_script(script)
            shorts_path   = str(out_dir / f"short_{ts}.mp4")
            make_shorts_from_script(shorts_script, shorts_path)
            short_script_path = out_dir / f"short_script_{ts}.json"
            short_script_path.write_text(json.dumps(shorts_script, indent=2, ensure_ascii=False))
            log.info(f"  Short saved → {shorts_path}")
        except Exception as e:
            log.error(f"Shorts render failed: {e}")
            shorts_path = None
    else:
        log.info("Step 4/5 — Shorts disabled (SHORTS_MODE=off)")

    # ── 5: Upload ────────────────────────────────────────────────────────────
    log.info("Step 5/5 — Uploading to YouTube…")
    try:
        youtube = get_authenticated_service()

        if video_path and Path(video_path).exists():
            video_id = upload_video(
                youtube     = youtube,
                video_path  = video_path,
                title       = script["video_title"],
                description = script["description"],
                tags        = script["tags"],
                category_id = script.get("category_id", "22"),
                privacy     = PRIVACY,
            )
            url = f"https://www.youtube.com/watch?v={video_id}"
            log.info(f"  Long-form uploaded → {url}")
            print(f"\n{'='*60}")
            print(f"  ✅  Long-form video uploaded!")
            print(f"  Title  : {script['video_title']}")
            print(f"  URL    : {url}")
            print(f"  Privacy: {PRIVACY}")
            print(f"{'='*60}\n")

        if shorts_path and Path(shorts_path).exists():
            short_id = upload_video(
                youtube     = youtube,
                video_path  = shorts_path,
                title       = shorts_script.get("short_title", script["video_title"])[:60] + " #shorts",
                description = shorts_script.get("description", "") + "\n#shorts",
                tags        = shorts_script.get("tags", script["tags"][:5]),
                category_id = script.get("category_id", "22"),
                privacy     = PRIVACY,
            )
            short_url = f"https://www.youtube.com/watch?v={short_id}"
            log.info(f"  Short uploaded → {short_url}")
            print(f"  ✅  Short uploaded!")
            print(f"  URL    : {short_url}\n")

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
    print(f"  Privacy: {PRIVACY}  |  Shorts: {SHORTS_MODE}  |  Quality threshold: {QUALITY_THRESHOLD}")
    print("  Press Ctrl+C to stop\n")
    log.info(f"Scheduler started — daily at {SCHEDULE_HOUR:02d}:00 UTC")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("Scheduler stopped by user")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YouTube Auto-Upload Pipeline — Monetisation Maximisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py              Run once right now
  python pipeline.py --schedule   Run daily at SCHEDULE_HOUR (from .env)
        """,
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run on a daily schedule instead of once",
    )
    args = parser.parse_args()

    if args.schedule:
        run_scheduled()
    else:
        run_pipeline()
