#!/usr/bin/env python3
"""
check.py — Pre-flight diagnostic for the YouTube Auto-Upload Pipeline.
Run this before pipeline.py to confirm everything is ready.

  python check.py
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

# ── Load .env ────────────────────────────────────────────────────────────────
def _load_dotenv():
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

G = "\033[32m"; Y = "\033[33m"; R = "\033[31m"; C = "\033[36m"; B = "\033[1m"; N = "\033[0m"

ok   = lambda msg: print(f"  {G}✅  {msg}{N}")
warn = lambda msg: print(f"  {Y}⚠   {msg}{N}")
fail = lambda msg: print(f"  {R}❌  {msg}{N}")
info = lambda msg: print(f"  {C}ℹ   {msg}{N}")
hr   = lambda:     print(f"  {C}{'─'*54}{N}")

issues = []


def check(label, passed, fix_msg=""):
    if passed:
        ok(label)
    else:
        fail(label)
        if fix_msg:
            issues.append(fix_msg)


def section(title):
    print(f"\n{B}{C}  {title}{N}")
    hr()


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{B}  YouTube Auto-Upload Pipeline — Pre-flight Check{N}")
hr()

# 1. Python
section("1. Python")
v = sys.version_info
check(f"Python {v.major}.{v.minor} found", v >= (3, 8),
      "Install Python 3.8+ from https://python.org")

# 2. System tools
section("2. System Tools")
ffmpeg_ok = subprocess.run(
    ["ffmpeg", "-version"], capture_output=True
).returncode == 0
check("ffmpeg installed (required by MoviePy)", ffmpeg_ok,
      "Install ffmpeg:\n"
      "      macOS  → brew install ffmpeg\n"
      "      Linux  → sudo apt install ffmpeg")

# 3. Python packages
section("3. Python Packages")
REQUIRED_PACKAGES = [
    ("streamlit",               "streamlit"),
    ("googleapiclient",         "google-api-python-client"),
    ("google.auth",             "google-auth"),
    ("google_auth_oauthlib",    "google-auth-oauthlib"),
    ("anthropic",               "anthropic"),
    ("moviepy",                 "moviepy"),
    ("PIL",                     "Pillow"),
    ("gtts",                    "gTTS"),
    ("edge_tts",                "edge-tts"),
    ("requests",                "requests"),
    ("apscheduler",             "APScheduler"),
    ("pytrends",                "pytrends"),
]

missing_pkgs = []
for mod, pkg in REQUIRED_PACKAGES:
    installed = importlib.util.find_spec(mod) is not None
    check(f"{pkg}", installed)
    if not installed:
        missing_pkgs.append(pkg)

if missing_pkgs:
    issues.append(
        "Install missing packages:\n"
        f"      pip install --no-cache-dir {' '.join(missing_pkgs)}"
    )

# 4. .env file
section("4. Environment (.env)")
env_file = Path(".env")
check(".env file exists", env_file.exists(),
      "Run:  bash setup.sh  or  cp .env.example .env")

YT_KEY      = os.environ.get("YOUTUBE_API_KEY",   "")
CH_ID       = os.environ.get("YOUTUBE_CHANNEL_ID","")
CLAUDE_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
PEXELS_KEY  = os.environ.get("PEXELS_API_KEY",    "")
TTS_VOICE   = os.environ.get("TTS_VOICE",         "en-US-JennyNeural")
PRIVACY     = os.environ.get("YOUTUBE_PRIVACY",   "private")
SCHED_HOUR  = os.environ.get("SCHEDULE_HOUR",     "9")

check("YOUTUBE_API_KEY set",    bool(YT_KEY),
      "Add YOUTUBE_API_KEY to .env\n"
      "      Get it at: console.cloud.google.com → APIs & Services → Credentials")
check("YOUTUBE_CHANNEL_ID set", bool(CH_ID),
      "Add YOUTUBE_CHANNEL_ID to .env  (starts with UC...)")
check("ANTHROPIC_API_KEY set",  bool(CLAUDE_KEY),
      "Add ANTHROPIC_API_KEY to .env\n"
      "      Get it at: console.anthropic.com")

if PEXELS_KEY:
    ok("PEXELS_API_KEY set (stock image backgrounds enabled)")
else:
    warn("PEXELS_API_KEY not set — gradient backgrounds will be used instead\n"
         "      Get a free key at: pexels.com/api")

info(f"TTS_VOICE     = {TTS_VOICE}")
info(f"YOUTUBE_PRIVACY = {PRIVACY}")
info(f"SCHEDULE_HOUR = {SCHED_HOUR}:00 UTC")

# 5. YouTube API key validity
section("5. YouTube Data API v3")
if YT_KEY:
    try:
        import requests as req
        r = req.get(
            "https://www.googleapis.com/youtube/v3/videoCategories",
            params={"part": "snippet", "regionCode": "US", "key": YT_KEY},
            timeout=8,
        )
        if r.status_code == 200:
            ok("YouTube API key is valid ✓")
        elif r.status_code == 400:
            fail(f"YouTube API key rejected (HTTP 400)")
            issues.append(
                "Fix YouTube API key:\n"
                "      1. console.cloud.google.com → APIs & Services → Library\n"
                "      2. Enable 'YouTube Data API v3'\n"
                "      3. Credentials → your API key → remove all restrictions\n"
                "      4. Wait 60 seconds then retry"
            )
        elif r.status_code == 403:
            detail = r.json().get("error", {}).get("errors", [{}])[0].get("reason", "")
            if "accessNotConfigured" in detail:
                fail("YouTube Data API v3 not enabled on this project")
                issues.append(
                    "Enable YouTube Data API v3:\n"
                    "      console.cloud.google.com → APIs & Services → Library\n"
                    "      → search 'YouTube Data API v3' → Enable"
                )
            else:
                fail(f"YouTube API key forbidden (HTTP 403) — quota or restriction issue")
                issues.append(
                    "Fix YouTube API key restrictions:\n"
                    "      APIs & Services → Credentials → your key\n"
                    "      → API restrictions: set to 'YouTube Data API v3'\n"
                    "      → Application restrictions: set to 'None'"
                )
        else:
            warn(f"YouTube API returned HTTP {r.status_code} — check key")
    except Exception as e:
        warn(f"Could not reach YouTube API: {e}")
else:
    warn("Skipped — YOUTUBE_API_KEY not set")
    info("Pipeline will use Google Trends as fallback")

# 6. Anthropic API key
section("6. Claude (Anthropic) API")
if CLAUDE_KEY:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=CLAUDE_KEY)
        r = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "ping"}],
        )
        ok("Anthropic API key is valid ✓")
    except Exception as e:
        fail(f"Anthropic API error: {e}")
        issues.append("Check ANTHROPIC_API_KEY in .env — get one at console.anthropic.com")
else:
    warn("Skipped — ANTHROPIC_API_KEY not set")

# 7. OAuth / Upload credentials
section("7. YouTube Upload (OAuth)")
secrets_ok = Path("client_secrets.json").exists()
token_ok   = Path("token.json").exists()

check("client_secrets.json present", secrets_ok,
      "Create OAuth credentials:\n"
      "      1. console.cloud.google.com → APIs & Services → Credentials\n"
      "      2. + Create Credentials → OAuth 2.0 Client ID → Desktop app\n"
      "      3. Download JSON → rename to client_secrets.json\n"
      "      4. Place it in:  " + str(Path(".").resolve()))

if secrets_ok and not token_ok:
    warn("token.json not found — run  python pipeline.py  once to log in via browser")
    issues.append(
        "Complete OAuth login:\n"
        "      python pipeline.py\n"
        "      A browser tab will open → log in → grant permission\n"
        "      token.json saved → all future runs are automatic"
    )
elif token_ok:
    ok("token.json present (OAuth already authorised)")

# 8. output folder
section("8. Output Directory")
out = Path("output")
out.mkdir(exist_ok=True)
ok(f"output/ folder ready ({out.resolve()})")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{B}  Summary{N}")
hr()

if not issues:
    print(f"\n  {G}{B}All checks passed — ready to run the pipeline!{N}\n")
    print(f"  Run once:        {C}python pipeline.py{N}")
    print(f"  Run on schedule: {C}python pipeline.py --schedule{N}\n")
else:
    print(f"\n  {R}{B}{len(issues)} issue(s) to fix before the pipeline will work:{N}\n")
    for i, issue in enumerate(issues, 1):
        print(f"  {B}{i}.{N} {issue}\n")
    print(f"  After fixing, run  {C}python check.py{N}  again to verify.\n")
