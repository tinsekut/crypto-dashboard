#!/usr/bin/env python3
"""
pipeline.py — YouTube Auto-Upload Pipeline

Full automation:
  1. Fetch today's trending videos (YouTube Data API)
  2. Claude picks the best trend and writes a structured video script
  3. video_maker.py renders the script into an MP4
  4. youtube_uploader.py uploads it to your channel

Usage:
  python pipeline.py              # run once right now
  python pipeline.py --schedule   # run daily at SCHEDULE_HOUR (set in .env)
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

import anthropic
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from video_maker      import make_video_from_script
from youtube_uploader import get_authenticated_service, upload_video

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
REGION        = os.environ.get("DEFAULT_REGION", "US")
MAX_RESULTS   = int(os.environ.get("MAX_RESULTS", "25"))
PRIVACY       = os.environ.get("YOUTUBE_PRIVACY", "private")
SCHEDULE_HOUR = int(os.environ.get("SCHEDULE_HOUR", "9"))


# ── Step 1: Fetch trending videos ─────────────────────────────────────────────

def fetch_trending_youtube(n: int = 25) -> list[dict]:
    """Primary source: YouTube Data API v3."""
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


# ── Step 2: Generate structured script via Claude ─────────────────────────────

def generate_script(trending: list[dict]) -> dict:
    client = anthropic.Anthropic(api_key=CLAUDE_KEY)

    trend_lines = "\n".join(
        f'- "{v["title"]}"  ({v["views"]:,} views, by {v["channel"]})'
        for v in trending[:12]
    )

    prompt = f"""You are a professional YouTube scriptwriter and SEO specialist.

Today's top trending videos:
{trend_lines}

Task:
1. Choose the single BEST topic to make an ORIGINAL video about (inspired by the trend, not a copy).
2. Write a complete, conversational, engaging script for a 6-8 minute video.
3. Return ONLY valid JSON — no markdown fences, no explanation, just the JSON object below.

Required JSON format:
{{
  "video_title": "Catchy, keyword-rich title under 70 characters",
  "description": "YouTube description 400-500 chars. Include timestamps:\\n0:00 Intro\\n1:00 ...\\nEnd with a subscribe call-to-action.",
  "tags": ["tag1","tag2","tag3","tag4","tag5","tag6","tag7","tag8","tag9","tag10"],
  "category_id": "22",
  "sections": [
    {{
      "name": "Hook",
      "narration": "The first 30 seconds — an attention-grabbing statement or question. Write the exact spoken words (50-80 words).",
      "visual_keyword": "short phrase for stock image search",
      "lower_third": "2-4 word overlay text"
    }},
    {{
      "name": "Introduction",
      "narration": "Introduce the topic and what viewers will learn. Natural, conversational tone. (120-150 words)",
      "visual_keyword": "short phrase for stock image search",
      "lower_third": "2-4 word overlay text"
    }},
    {{
      "name": "Point 1",
      "narration": "First key insight or step. Detailed, clear, with an example. (180-220 words)",
      "visual_keyword": "short phrase for stock image search",
      "lower_third": "2-4 word overlay text"
    }},
    {{
      "name": "Point 2",
      "narration": "Second key insight or step. Build on point 1. (180-220 words)",
      "visual_keyword": "short phrase for stock image search",
      "lower_third": "2-4 word overlay text"
    }},
    {{
      "name": "Point 3",
      "narration": "Third key insight or step. Most impactful point saved for here. (180-220 words)",
      "visual_keyword": "short phrase for stock image search",
      "lower_third": "2-4 word overlay text"
    }},
    {{
      "name": "Call To Action",
      "narration": "Ask viewers to like, comment their thoughts, and subscribe. Tease next video. (60-80 words)",
      "visual_keyword": "youtube subscribe button notification",
      "lower_third": "Like & Subscribe"
    }}
  ]
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if Claude adds them
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    log.info("=" * 60)
    log.info("Pipeline started")
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

    # 1 — Trending
    log.info("Step 1/4 — Fetching trending videos…")
    try:
        trending = fetch_trending(MAX_RESULTS)
        log.info(f"  Got {len(trending)} trending videos")
    except Exception as e:
        log.error(f"Fetch failed: {e}")
        return

    # 2 — Script
    log.info("Step 2/4 — Generating script with Claude…")
    try:
        script      = generate_script(trending)
        script_path = out_dir / f"script_{ts}.json"
        script_path.write_text(json.dumps(script, indent=2, ensure_ascii=False))
        log.info(f"  Title: {script['video_title']}")
        log.info(f"  Script saved → {script_path}")
    except Exception as e:
        log.error(f"Script generation failed: {e}")
        return

    # 3 — Video
    log.info("Step 3/4 — Rendering video…")
    video_path = str(out_dir / f"video_{ts}.mp4")
    try:
        make_video_from_script(script, video_path)
        log.info(f"  Video saved → {video_path}")
    except Exception as e:
        log.error(f"Video render failed: {e}")
        return

    # 4 — Upload
    log.info("Step 4/4 — Uploading to YouTube…")
    try:
        youtube  = get_authenticated_service()
        video_id = upload_video(
            youtube      = youtube,
            video_path   = video_path,
            title        = script["video_title"],
            description  = script["description"],
            tags         = script["tags"],
            category_id  = script.get("category_id", "22"),
            privacy      = PRIVACY,
        )
        url = f"https://www.youtube.com/watch?v={video_id}"
        log.info(f"  Uploaded → {url}")
        print(f"\n{'='*60}")
        print(f"  ✅  Video uploaded!")
        print(f"  Title  : {script['video_title']}")
        print(f"  URL    : {url}")
        print(f"  Privacy: {PRIVACY}")
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
  python pipeline.py              Run the pipeline once right now
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
