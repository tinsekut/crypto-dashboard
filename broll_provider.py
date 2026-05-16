"""
broll_provider.py — AI-generated B-roll via Kling AI + Pexels video fallback.

Priority waterfall for section background video:
  1. Kling AI  (KLING_API_KEY set) — AI-generated 5s clip matching the narration
  2. Pexels Video (PEXELS_API_KEY set) — stock footage search
  3. Pexels Image → still frame (existing behaviour)
  4. Gradient fallback

Set BROLL_MODE=ai in .env to activate Kling AI B-roll.
Leave unset or BROLL_MODE=image for the original image-only behaviour.

Kling AI API docs: https://klingai.com/api
"""

import os
import time
import logging
import tempfile
from pathlib import Path
from io import BytesIO

import requests

log = logging.getLogger(__name__)

KLING_API_KEY  = os.environ.get("KLING_API_KEY", "")
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
BROLL_MODE     = os.environ.get("BROLL_MODE", "image").lower()  # image | ai | video

_KLING_BASE = "https://api.klingai.com/v1"


# ── Kling AI video generation ─────────────────────────────────────────────────

def _kling_generate(prompt: str, duration: int = 5) -> str | None:
    """
    Submit a Kling AI text-to-video job, poll until complete, return video URL.
    duration: 5 or 10 seconds.
    Returns download URL or None on failure.
    """
    if not KLING_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {KLING_API_KEY}",
        "Content-Type":  "application/json",
    }

    # Submit generation job
    try:
        resp = requests.post(
            f"{_KLING_BASE}/videos/text2video",
            headers=headers,
            json={
                "model":        "kling-v1",
                "prompt":       prompt[:800],
                "duration":     duration,
                "aspect_ratio": "16:9",
                "cfg_scale":    0.5,
            },
            timeout=30,
        )
        resp.raise_for_status()
        task_id = resp.json().get("data", {}).get("task_id")
        if not task_id:
            log.warning("Kling: no task_id in response")
            return None
    except Exception as e:
        log.warning(f"Kling submit failed: {e}")
        return None

    # Poll for completion (max 3 min)
    log.info(f"  Kling generating clip (task {task_id})…")
    for attempt in range(36):  # 36 × 5s = 3 min
        time.sleep(5)
        try:
            poll = requests.get(
                f"{_KLING_BASE}/videos/text2video/{task_id}",
                headers=headers,
                timeout=15,
            )
            poll.raise_for_status()
            data   = poll.json().get("data", {})
            status = data.get("task_status", "")
            if status == "succeed":
                videos = data.get("task_result", {}).get("videos", [])
                if videos:
                    url = videos[0].get("url")
                    log.info(f"  Kling clip ready: {url[:60]}…")
                    return url
                log.warning("Kling: task succeeded but no video URL")
                return None
            if status == "failed":
                log.warning(f"Kling task failed: {data.get('task_status_msg', '')}")
                return None
        except Exception as e:
            log.warning(f"Kling poll attempt {attempt+1} error: {e}")

    log.warning("Kling: timed out after 3 minutes")
    return None


def _download_video(url: str, dest_path: str) -> bool:
    """Download a video URL to a local file. Returns True on success."""
    try:
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        return True
    except Exception as e:
        log.warning(f"Video download failed: {e}")
        return False


# ── Pexels video search ───────────────────────────────────────────────────────

def _pexels_video(keyword: str) -> str | None:
    """Search Pexels Videos API. Returns a download URL or None."""
    if not PEXELS_API_KEY:
        return None
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": keyword, "per_page": 3, "orientation": "landscape"},
            timeout=10,
        )
        videos = r.json().get("videos", [])
        if not videos:
            return None
        # Pick the SD file (smallest, fastest to download)
        files = videos[0].get("video_files", [])
        sd    = next((f for f in files if f.get("quality") == "sd"), None)
        hd    = next((f for f in files if f.get("quality") == "hd"), None)
        chosen = sd or hd
        if chosen:
            return chosen.get("link")
    except Exception as e:
        log.warning(f"Pexels video '{keyword}': {e}")
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def get_broll_clip(
    keyword: str,
    narration: str,
    duration_s: float,
    tmp_dir: str,
    section_index: int,
) -> str | None:
    """
    Return the path to a local video file to use as B-roll for a section.
    Returns None if only a still image is available (caller falls back to Pexels image).

    Priority:
      BROLL_MODE=ai   → Kling AI → Pexels video → None
      BROLL_MODE=video → Pexels video → None
      BROLL_MODE=image → None (use existing still-image path from video_maker.py)
    """
    tmp = Path(tmp_dir)

    if BROLL_MODE == "ai" and KLING_API_KEY:
        prompt = f"Cinematic 16:9 video clip: {keyword}. {narration[:200]}. No text, no people talking, dramatic lighting."
        clip_url = _kling_generate(prompt, duration=min(10, max(5, int(duration_s))))
        if clip_url:
            dest = str(tmp / f"kling_{section_index:02d}.mp4")
            if _download_video(clip_url, dest):
                log.info(f"  Kling B-roll ready → {dest}")
                return dest
        log.info("  Kling B-roll failed — trying Pexels video…")

    if BROLL_MODE in ("ai", "video") and PEXELS_API_KEY:
        video_url = _pexels_video(keyword)
        if video_url:
            dest = str(tmp / f"pexels_video_{section_index:02d}.mp4")
            if _download_video(video_url, dest):
                log.info(f"  Pexels video B-roll ready → {dest}")
                return dest
        log.info("  Pexels video unavailable — falling back to still image")

    return None  # caller uses still image
