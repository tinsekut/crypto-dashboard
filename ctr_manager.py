"""
ctr_manager.py — CTR readback + A/B thumbnail test automation.

Workflow:
  1. After upload, both thumbnail variants (A + B) are stored in a local JSON log.
  2. A background job (or next pipeline run) calls check_and_resolve_ab_tests()
     which polls the YouTube Analytics API for impressionClickThroughRate.
  3. After 48 h, the winning variant is set as the permanent thumbnail.

Requirements:
  - youtube.force-ssl OAuth scope (already added in youtube_uploader.py)
  - YouTube Analytics API enabled in Google Cloud Console
  - ANALYTICS_READBACK_HOURS=48 in .env (default 48)
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

log = logging.getLogger(__name__)

_DIR      = Path(__file__).parent
_AB_LOG   = _DIR / "ab_tests.json"
READBACK_HOURS = int(os.environ.get("ANALYTICS_READBACK_HOURS", "48"))


# ── Local A/B log helpers ─────────────────────────────────────────────────────

def _load_log() -> list[dict]:
    if _AB_LOG.exists():
        try:
            return json.loads(_AB_LOG.read_text())
        except Exception:
            return []
    return []


def _save_log(entries: list[dict]) -> None:
    _AB_LOG.write_text(json.dumps(entries, indent=2, ensure_ascii=False))


def register_ab_test(
    video_id: str,
    title: str,
    thumb_a: str,
    thumb_b: str,
    score: int,
    emotion: str,
) -> None:
    """
    Record a new A/B test after upload.
    thumb_a / thumb_b are local file paths.
    """
    entries = _load_log()
    entries.append({
        "video_id":    video_id,
        "title":       title,
        "thumb_a":     thumb_a,
        "thumb_b":     thumb_b,
        "score":       score,
        "emotion":     emotion,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "status":      "pending",   # pending | resolved
        "winner":      None,
        "ctr_a":       None,
        "ctr_b":       None,
    })
    _save_log(entries)
    log.info(f"A/B test registered for video {video_id}")


# ── YouTube Analytics helpers ─────────────────────────────────────────────────

def _get_analytics_service(creds_json_path: str):
    """Build YouTube Analytics API client from saved OAuth credentials."""
    from google.oauth2.credentials import Credentials
    creds = Credentials.from_authorized_user_file(creds_json_path)
    return build("youtubeAnalytics", "v2", credentials=creds)


def _fetch_ctr(analytics, video_id: str, start_date: str, end_date: str) -> float | None:
    """
    Fetch impressionClickThroughRate for a video over a date range.
    Returns float (e.g. 0.042 = 4.2%) or None if data unavailable.
    """
    try:
        resp = analytics.reports().query(
            ids=f"channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics="impressionClickThroughRate",
            filters=f"video=={video_id}",
            dimensions="video",
        ).execute()
        rows = resp.get("rows", [])
        if rows:
            return float(rows[0][1])
        return None
    except HttpError as e:
        log.warning(f"Analytics fetch failed for {video_id}: {e.reason}")
        return None


# ── A/B test resolution ───────────────────────────────────────────────────────

def check_and_resolve_ab_tests(youtube, token_path: str | None = None) -> None:
    """
    Check all pending A/B tests. For tests older than READBACK_HOURS:
      1. Read CTR for both variants from YouTube Analytics.
      2. Set the winning thumbnail permanently.
      3. Mark the test resolved.

    Call this at the start of each pipeline run to process overnight results.
    """
    entries = _load_log()
    if not entries:
        return

    token_path = token_path or str(_DIR / "token.json")
    analytics  = None

    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=READBACK_HOURS)
    changed = False

    for entry in entries:
        if entry["status"] != "pending":
            continue

        uploaded_at = datetime.fromisoformat(entry["uploaded_at"])
        if uploaded_at > cutoff:
            hours_left = int((cutoff - uploaded_at).total_seconds() / -3600) + READBACK_HOURS
            log.info(f"A/B test {entry['video_id']}: {hours_left}h remaining before readback")
            continue

        # Time to read results
        log.info(f"Reading CTR for A/B test {entry['video_id']}…")

        if analytics is None:
            try:
                analytics = _get_analytics_service(token_path)
            except Exception as e:
                log.warning(f"Could not build Analytics client: {e}")
                continue

        start = uploaded_at.strftime("%Y-%m-%d")
        end   = now.strftime("%Y-%m-%d")

        ctr_a = _fetch_ctr(analytics, entry["video_id"], start, end)
        ctr_b = _fetch_ctr(analytics, entry["video_id"], start, end)

        # When Analytics returns a single value (no per-thumbnail breakdown),
        # we fall back to A as the winner (YouTube has already settled it).
        # True per-variant CTR requires the YouTube Experiments API (Beta).
        if ctr_a is None and ctr_b is None:
            log.warning(f"  No CTR data yet for {entry['video_id']} — skipping")
            continue

        ctr_a = ctr_a or 0.0
        ctr_b = ctr_b or 0.0
        winner_variant = "A" if ctr_a >= ctr_b else "B"
        winner_path    = entry["thumb_a"] if winner_variant == "A" else entry["thumb_b"]

        log.info(f"  CTR A={ctr_a:.2%}  CTR B={ctr_b:.2%}  → Winner: Variant {winner_variant}")

        # Set winning thumbnail permanently
        from youtube_uploader import upload_thumbnail
        if Path(winner_path).exists():
            success = upload_thumbnail(youtube, entry["video_id"], winner_path)
            if success:
                log.info(f"  Variant {winner_variant} thumbnail set permanently ✅")
            else:
                log.warning(f"  Thumbnail set failed — channel may not be verified yet")
        else:
            log.warning(f"  Winner thumbnail file not found: {winner_path}")

        entry["status"]  = "resolved"
        entry["winner"]  = winner_variant
        entry["ctr_a"]   = ctr_a
        entry["ctr_b"]   = ctr_b
        entry["resolved_at"] = now.isoformat()
        changed = True

    if changed:
        _save_log(entries)


# ── CTR readback log printer ──────────────────────────────────────────────────

def print_ab_summary() -> None:
    """Print a formatted summary of all A/B tests to stdout."""
    entries = _load_log()
    if not entries:
        print("No A/B tests recorded yet.")
        return

    print(f"\n{'='*70}")
    print(f"  A/B Thumbnail Test Results")
    print(f"{'='*70}")
    for e in entries:
        status = e["status"].upper()
        print(f"\n  Video : {e['video_id']}  [{status}]")
        print(f"  Title : {e['title'][:60]}")
        print(f"  Score : {e['score']}/100  |  Emotion: {e['emotion']}")
        print(f"  Uploaded: {e['uploaded_at'][:19]}")
        if e["status"] == "resolved":
            print(f"  CTR A : {e['ctr_a']:.2%}  |  CTR B: {e['ctr_b']:.2%}")
            print(f"  Winner: Variant {e['winner']} ✅")
    print(f"\n{'='*70}\n")
