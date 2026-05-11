#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# YouTube Trend Generator — First-Time Local Setup
# Run once after cloning:  bash setup.sh
# After setup, use:        bash run.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; CYAN="\033[36m"; NC="\033[0m"
ok()   { echo -e "${GREEN}✅  $*${NC}"; }
info() { echo -e "${CYAN}ℹ   $*${NC}"; }
warn() { echo -e "${YELLOW}⚠   $*${NC}"; }
err()  { echo -e "${RED}✖   $*${NC}"; exit 1; }
hr()   { echo -e "${CYAN}────────────────────────────────────────────${NC}"; }

hr
echo -e "${CYAN}  YouTube Trend Generator — First-Time Setup${NC}"
hr

# ── 1. Python check ────────────────────────────────────────────────────────
PY=$(command -v python3 || command -v python || true)
[[ -z "$PY" ]] && err "Python 3 not found. Install from https://python.org"
PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python $PY_VER found at $PY"
[[ "${PY_VER%%.*}" -lt 3 ]] && err "Python 3.8+ required (found $PY_VER)"

# ── 2. Create virtual environment ─────────────────────────────────────────
VENV_DIR="$(dirname "$0")/.venv"
if [[ -d "$VENV_DIR" ]]; then
    info "Virtual environment already exists at .venv — skipping creation"
else
    info "Creating virtual environment at .venv …"
    "$PY" -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
ok "Virtual environment activated"

# ── 3. Install dependencies ────────────────────────────────────────────────
info "Upgrading pip …"
pip install --quiet --upgrade pip

info "Installing packages from requirements.txt …"
pip install --quiet -r "$(dirname "$0")/requirements.txt"
ok "All packages installed into .venv"

# ── 4. .env setup ─────────────────────────────────────────────────────────
ENV_FILE="$(dirname "$0")/.env"
EXAMPLE_FILE="$(dirname "$0")/.env.example"

if [[ -f "$ENV_FILE" ]]; then
    ok ".env already exists — skipping key entry"
else
    cp "$EXAMPLE_FILE" "$ENV_FILE"
    hr
    echo -e "${CYAN}  Enter your API credentials (stored in .env — never committed to git)${NC}"
    hr

    read -rp "  YouTube Data API v3 Key  (AIza...): " YT_KEY
    read -rp "  Your YouTube Channel ID  (UC...):   " CH_ID
    read -rp "  Anthropic Claude API Key (optional, Enter to skip): " CLAUDE_KEY
    read -rp "  Default region code      [US]:       " REGION;   REGION="${REGION:-US}"
    read -rp "  Default category         [All]:      " CATEGORY; CATEGORY="${CATEGORY:-All}"
    read -rp "  Max results per fetch    [25]:       " MAX_R;    MAX_R="${MAX_R:-25}"

    sed -i "s|^YOUTUBE_API_KEY=.*|YOUTUBE_API_KEY=${YT_KEY}|"         "$ENV_FILE"
    sed -i "s|^YOUTUBE_CHANNEL_ID=.*|YOUTUBE_CHANNEL_ID=${CH_ID}|"   "$ENV_FILE"
    sed -i "s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=${CLAUDE_KEY}|" "$ENV_FILE"
    sed -i "s|^DEFAULT_REGION=.*|DEFAULT_REGION=${REGION}|"           "$ENV_FILE"
    sed -i "s|^DEFAULT_CATEGORY=.*|DEFAULT_CATEGORY=${CATEGORY}|"     "$ENV_FILE"
    sed -i "s|^MAX_RESULTS=.*|MAX_RESULTS=${MAX_R}|"                  "$ENV_FILE"
    ok ".env saved"

    # Validate YouTube key
    if [[ -n "$YT_KEY" ]]; then
        info "Validating YouTube API key …"
        HTTP=$(curl -s -o /dev/null -w "%{http_code}" \
            "https://www.googleapis.com/youtube/v3/videoCategories?part=snippet&regionCode=US&key=${YT_KEY}" \
            2>/dev/null || echo "000")
        case "$HTTP" in
            200) ok "YouTube API key is valid" ;;
            400) warn "YouTube API returned 400 — double-check the key" ;;
            403) warn "YouTube API returned 403 — quota exceeded or API not enabled in Google Cloud" ;;
            000) warn "Could not reach YouTube API — check your internet connection" ;;
            *)   warn "YouTube API returned HTTP $HTTP — verify the key" ;;
        esac
    fi
fi

hr
ok "Setup complete!"
echo ""
echo -e "  To start the app any time, run:  ${GREEN}bash run.sh${NC}"
echo ""
