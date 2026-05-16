#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# YouTube Trend Generator — Daily launcher
# Usage:  bash run.sh
# Run setup.sh first if this is a fresh clone.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"
APP="$DIR/youtube_trend_generator.py"

GREEN="\033[32m"; CYAN="\033[36m"; RED="\033[31m"; NC="\033[0m"
ok()  { echo -e "${GREEN}✅  $*${NC}"; }
err() { echo -e "${RED}✖   $*${NC}"; exit 1; }
info(){ echo -e "${CYAN}ℹ   $*${NC}"; }

# ── Guard: run setup.sh first if .venv is missing ─────────────────────────
if [[ ! -d "$VENV" ]]; then
    echo -e "${RED}Virtual environment not found.${NC}"
    echo "Run first-time setup with:  bash setup.sh"
    exit 1
fi

# ── Activate virtualenv ────────────────────────────────────────────────────
source "$VENV/bin/activate"
ok "Virtual environment activated"

# ── Load .env if present ───────────────────────────────────────────────────
if [[ -f "$DIR/.env" ]]; then
    set -a; source "$DIR/.env"; set +a
    ok ".env loaded"
else
    echo -e "${RED}⚠  .env not found — run bash setup.sh first${NC}"
fi

# ── Launch ─────────────────────────────────────────────────────────────────
info "Starting app at http://localhost:8501 …"
echo -e "${CYAN}  Press Ctrl+C to stop${NC}"
echo ""

exec python3 -m streamlit run "$APP" \
    --server.enableCORS false \
    --server.enableXsrfProtection false
