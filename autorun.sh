#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# autorun.sh — YouTube Auto-Upload Pipeline launcher (Session 7)
#
# What it does automatically:
#   Step 1  Fetch 25 trending videos (YouTube API or Google Trends fallback)
#   Step 2  Claude writes a 10-section psychological script (JSON)
#             • Emotion-keyed architecture (shock→dread→recognition→…→belonging)
#             • Open-loop discipline (A/B/C/D), peak_sentence, fake-out, rewatch clue
#             • Quality gate (target 88+/100) with up to SCRIPT_REWRITE_PASSES rewrites
#   Step 3  Render long-form 1920×1080 MP4
#             • Emotion-keyed gradient palettes + vignette
#             • Silent 2.5s pull-quote cards after each section
#             • Retention-bridge overlays (last 28% of every section)
#             • Pattern-interrupt visual resets every 35s
#             • Rewatch-clue easter egg on Section 1
#   Step 4  Render Shorts 1080×1920 MP4 (≤60s) — SHORTS_MODE=also
#             • comment_bait, identity_mirror, end_screen_hook visual layers
#   Step 5  Upload both to your YouTube channel
#
# First run:
#   bash autorun.sh            ← runs the pipeline once now
#
# Daily automatic:
#   bash autorun.sh --schedule ← runs every day at SCHEDULE_HOUR (from .env)
#
# Key .env settings:
#   SHORTS_MODE=also            # off | also | only
#   QUALITY_GATE_THRESHOLD=88   # rescore target
#   SCRIPT_REWRITE_PASSES=2     # how many LLM rewrites to attempt
#   YOUTUBE_PRIVACY=private     # flip to public when ready
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"
GREEN="\033[32m"; RED="\033[31m"; CYAN="\033[36m"; YELLOW="\033[33m"; NC="\033[0m"
ok()   { echo -e "${GREEN}✅  $*${NC}"; }
fail() { echo -e "${RED}❌  $*${NC}"; }
info() { echo -e "${CYAN}ℹ   $*${NC}"; }
warn() { echo -e "${YELLOW}⚠   $*${NC}"; }
hr()   { echo -e "${CYAN}────────────────────────────────────────────────────${NC}"; }

hr
echo -e "${CYAN}  YouTube Auto-Upload Pipeline${NC}"
hr

# ── 1. Virtual environment ─────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    info "Virtual environment not found — running setup first…"
    bash "$DIR/setup.sh"
fi
source "$VENV/bin/activate"
ok "Virtual environment active"

# ── 2. Load .env ───────────────────────────────────────────────────────────
if [[ ! -f "$DIR/.env" ]]; then
    fail ".env not found — run bash setup.sh first"
    exit 1
fi
set -a; source "$DIR/.env"; set +a
ok ".env loaded"

# ── 3. Install / update dependencies silently ─────────────────────────────
info "Checking dependencies…"
pip install --quiet --no-cache-dir -r "$DIR/requirements.txt"
ok "Dependencies ready"

# ── 4. ffmpeg check ────────────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
    warn "ffmpeg not found — attempting install…"
    if [[ "$OSTYPE" == "darwin"* ]] && command -v brew &>/dev/null; then
        brew install ffmpeg --quiet && ok "ffmpeg installed"
    elif command -v apt-get &>/dev/null; then
        sudo apt-get install -y -qq ffmpeg && ok "ffmpeg installed"
    else
        fail "ffmpeg required but could not be installed automatically."
        echo "  Install manually: brew install ffmpeg"
        exit 1
    fi
else
    ok "ffmpeg found"
fi

# ── 5. Required credentials check ─────────────────────────────────────────
ABORT=0

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    fail "ANTHROPIC_API_KEY missing in .env"
    echo "      Get one at: https://console.anthropic.com"
    ABORT=1
fi

if [[ -z "${YOUTUBE_CHANNEL_ID:-}" ]]; then
    fail "YOUTUBE_CHANNEL_ID missing in .env"
    echo "      Find it: YouTube Studio → Customization → Basic info"
    ABORT=1
fi

if [[ ! -f "$DIR/client_secrets.json" ]]; then
    fail "client_secrets.json not found in $DIR"
    echo ""
    echo "  How to create it (one-time):"
    echo "    1. Go to https://console.cloud.google.com/"
    echo "    2. APIs & Services → Credentials"
    echo "    3. + Create Credentials → OAuth 2.0 Client ID → Desktop app"
    echo "    4. Download JSON → rename to client_secrets.json"
    echo "    5. Move it to: $DIR"
    ABORT=1
fi

[[ "$ABORT" -eq 1 ]] && { echo ""; fail "Fix the issues above then run again."; exit 1; }

# Warn if YouTube API key is missing (non-fatal — falls back to Google Trends)
if [[ -z "${YOUTUBE_API_KEY:-}" ]]; then
    warn "YOUTUBE_API_KEY not set — will use Google Trends as trend source"
fi

hr
ok "All checks passed — starting pipeline"
hr

# ── 6. Run the pipeline ────────────────────────────────────────────────────
cd "$DIR"

if [[ "${1:-}" == "--schedule" ]]; then
    HOUR="${SCHEDULE_HOUR:-9}"
    echo ""
    info "Scheduler mode: pipeline runs daily at ${HOUR}:00 UTC"
    info "Privacy    : ${YOUTUBE_PRIVACY:-private}"
    info "Shorts     : ${SHORTS_MODE:-also}"
    info "Thumbnails : A/B=${AB_THUMBNAILS:-1}  Readback=${ANALYTICS_READBACK_HOURS:-48}h"
    info "B-roll     : ${BROLL_MODE:-image}"
    info "3D Scenes  : ${USE_3D_SCENES:-0}"
    echo ""
    echo "  Leave this terminal open. Press Ctrl+C to stop."
    echo ""
    python pipeline.py --schedule
elif [[ "${1:-}" == "--ab-report" ]]; then
    python pipeline.py --ab-report
else
    echo ""
    info "Privacy    : ${YOUTUBE_PRIVACY:-private}"
    info "Shorts     : ${SHORTS_MODE:-also}"
    info "Thumbnails : A/B=${AB_THUMBNAILS:-1}  Readback=${ANALYTICS_READBACK_HOURS:-48}h"
    info "B-roll     : ${BROLL_MODE:-image}"
    info "3D Scenes  : ${USE_3D_SCENES:-0}"
    echo "  (Set YOUTUBE_PRIVACY=public in .env to publish immediately)"
    echo ""
    python pipeline.py
fi
