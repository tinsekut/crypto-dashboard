#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# YouTube Trend Generator — One-Command Setup
# Usage:  bash setup.sh          (interactive)
#         bash setup.sh --run    (setup + auto-launch app)
#         bash setup.sh --check  (verify credentials only)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────
GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; CYAN="\033[36m"; NC="\033[0m"
ok()   { echo -e "${GREEN}✅ $*${NC}"; }
info() { echo -e "${CYAN}ℹ  $*${NC}"; }
warn() { echo -e "${YELLOW}⚠  $*${NC}"; }
err()  { echo -e "${RED}✖  $*${NC}"; exit 1; }
hr()   { echo -e "${CYAN}────────────────────────────────────────────${NC}"; }

hr
echo -e "${CYAN}  YouTube Trend Generator — Setup${NC}"
hr

# ── 1. Python version check ────────────────────────────────────────────────
PY=$(command -v python3 || command -v python || true)
[ -z "$PY" ] && err "Python 3 not found. Install it from https://python.org"
PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python: $PY_VER  ($PY)"
[[ "${PY_VER%%.*}" -lt 3 ]] && err "Python 3.8+ required (found $PY_VER)"

# ── 2. pip check ───────────────────────────────────────────────────────────
PIP=$(command -v pip3 || command -v pip || true)
[ -z "$PIP" ] && err "pip not found. Run: python3 -m ensurepip --upgrade"
info "pip: $($PIP --version | head -1)"

# ── 3. Install Python dependencies ────────────────────────────────────────
info "Installing Python packages from requirements.txt …"
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet -r requirements.txt
ok "All packages installed"

# ── 4. .env file setup ────────────────────────────────────────────────────
if [ ! -f .env ]; then
    cp .env.example .env
    warn ".env created from .env.example — please fill in your API keys."
    echo ""
    info "Opening .env for editing …"
    echo ""

    # Interactive prompts
    read -rp "  YouTube Data API v3 Key (AIza...): " YT_KEY
    read -rp "  Your YouTube Channel ID  (UC...):  " CH_ID
    read -rp "  Anthropic Claude API Key (optional, press Enter to skip): " CLAUDE_KEY
    read -rp "  Default region code [US]: " REGION
    REGION="${REGION:-US}"
    read -rp "  Default category [All]: " CATEGORY
    CATEGORY="${CATEGORY:-All}"
    read -rp "  Max results per fetch [25]: " MAX_R
    MAX_R="${MAX_R:-25}"

    # Write values into .env
    sed -i "s|^YOUTUBE_API_KEY=.*|YOUTUBE_API_KEY=${YT_KEY}|"      .env
    sed -i "s|^YOUTUBE_CHANNEL_ID=.*|YOUTUBE_CHANNEL_ID=${CH_ID}|" .env
    sed -i "s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=${CLAUDE_KEY}|" .env
    sed -i "s|^DEFAULT_REGION=.*|DEFAULT_REGION=${REGION}|"        .env
    sed -i "s|^DEFAULT_CATEGORY=.*|DEFAULT_CATEGORY=${CATEGORY}|"   .env
    sed -i "s|^MAX_RESULTS=.*|MAX_RESULTS=${MAX_R}|"               .env
    ok ".env saved"
else
    ok ".env already exists — skipping interactive setup"
fi

# ── 5. Load .env and validate ─────────────────────────────────────────────
set -a; source .env; set +a

hr
info "Credential check:"

YT_KEY="${YOUTUBE_API_KEY:-}"
CH_ID="${YOUTUBE_CHANNEL_ID:-}"
CLAUDE_KEY="${ANTHROPIC_API_KEY:-}"

if [ -z "$YT_KEY" ]; then
    warn "YOUTUBE_API_KEY is not set — app will prompt you in the sidebar"
else
    # Quick API ping (list video categories — 1 unit cost)
    info "Pinging YouTube Data API v3 …"
    HTTP=$(curl -s -o /dev/null -w "%{http_code}" \
        "https://www.googleapis.com/youtube/v3/videoCategories?part=snippet&regionCode=US&key=${YT_KEY}" \
        2>/dev/null || echo "000")
    if [ "$HTTP" = "200" ]; then
        ok "YouTube API key is valid (HTTP 200)"
    elif [ "$HTTP" = "400" ]; then
        warn "YouTube API key returned 400 — check if it's correct"
    elif [ "$HTTP" = "403" ]; then
        warn "YouTube API key returned 403 — quota exceeded or API not enabled"
    elif [ "$HTTP" = "000" ]; then
        warn "No internet / curl unavailable — skipping API ping"
    else
        warn "YouTube API returned HTTP $HTTP — verify the key"
    fi
fi

if [ -z "$CH_ID" ]; then
    warn "YOUTUBE_CHANNEL_ID is not set — channel tab will be empty"
else
    ok "Channel ID set: $CH_ID"
fi

if [ -z "$CLAUDE_KEY" ]; then
    info "ANTHROPIC_API_KEY not set — rule-based script templates will be used"
else
    ok "Anthropic Claude API key set — AI script generation enabled"
fi

hr

# ── 6. Done / launch ──────────────────────────────────────────────────────
ok "Setup complete!"
echo ""

if [[ "${1:-}" == "--check" ]]; then
    info "Credential check done. Run the app with:  bash setup.sh --run"
    exit 0
fi

if [[ "${1:-}" == "--run" ]] || { read -rp "  Launch the app now? [Y/n]: " LAUNCH; [[ "${LAUNCH,,}" != "n" ]]; }; then
    info "Starting YouTube Trend Generator on http://localhost:8501 …"
    exec streamlit run youtube_trend_generator.py \
        --server.enableCORS false \
        --server.enableXsrfProtection false
fi
