"""
video_maker.py — Hyper-animated 1080p video engine.

Architecture fuses kinetic typography (MrBeast/Kurzgesagt) with
content-explaining animations (3Blue1Brown/Veritasium):

  • Kinetic typography   — every word slides up & fades in sequentially
  • Number counters      — statistics count up from 0 to their final value
  • Particle field       — 28 drifting gold/orange/blue particles per cut
  • Spring intro cards   — section name zooms in with ease-out-back (overshoot)
  • Pulsing callout cards — gradient breathes + typewriter text reveal
  • 4-direction Ken Burns — zoom-in / zoom-out / pan-left / pan-right
  • Pop-in on every clip  — 1.07× → 1.0× ease-out in the first 0.2 s
  • Animated progress bar — gold fill advances as the video plays
  • Vignette              — cinematic edge darkening on all backgrounds
  • Thumbnail             — 1280×720 high-CTR thumbnail saved alongside video
  • Background music      — optional, BACKGROUND_MUSIC_PATH in .env
"""

import os
import sys
import math
import re
import asyncio
import textwrap
import tempfile
import logging
from io import BytesIO
from pathlib import Path

import warnings
# Suppress MoviePy/ffmpeg "N bytes wanted but 0 bytes read at frame M" warning.
# This fires when a downloaded video clip is shorter than its container metadata
# claims. MoviePy recovers by using the last valid frame — no action needed.
warnings.filterwarnings(
    "ignore",
    message=r".*bytes wanted but 0 bytes read.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Using the last valid frame.*",
    category=UserWarning,
)

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import requests

log = logging.getLogger(__name__)

# ── Optional cinematic engine (CINEMATIC_ENGINE=1 in .env) ───────────────────
_CINEMATIC_ENGINE = os.environ.get("CINEMATIC_ENGINE", "0").strip() == "1"
_USE_3D_SCENES    = os.environ.get("USE_3D_SCENES",    "0").strip() == "1"

# ── Free image generation providers ──────────────────────────────────────────
_POLLINATIONS_KEY  = os.environ.get("POLLINATIONS_KEY", "").strip()
_HF_TOKEN          = os.environ.get("HF_TOKEN", "").strip()
_UNSPLASH_KEY      = os.environ.get("UNSPLASH_ACCESS_KEY", "").strip()

# ── Video background providers (hyper-realistic motion) ───────────────────────
# PEXELS_VIDEO_BG=1  — use real Pexels video clips as moving backgrounds (FREE)
#                      Requires PEXELS_API_KEY. Real human motion, no cost.
# LTX_VIDEO_BG=1     — use Replicate LTX-Video AI generation (~$0.07/clip)
#                      Requires REPLICATE_API_KEY. AI-generated cinematic motion.
# KLING_VIDEO_BG=1   — use Kling AI via fal.ai (~$0.38/clip, highest quality)
#                      Requires KLING_API_KEY (from fal.ai). Zach King-level realism.
_PEXELS_VIDEO_BG = os.environ.get("PEXELS_VIDEO_BG", "1").strip()  # on by default if PEXELS key set
_LTX_VIDEO_BG    = os.environ.get("LTX_VIDEO_BG",   "0").strip()   # opt-in (costs ~$0.07/clip)
_KLING_VIDEO_BG  = os.environ.get("KLING_VIDEO_BG",  "0").strip()   # opt-in (costs ~$0.38/clip)
_KLING_API_KEY   = os.environ.get("KLING_API_KEY",   "").strip()    # fal.ai API key for Kling
try:
    if _CINEMATIC_ENGINE:
        import cinematic_engine as _ce
        if _USE_3D_SCENES:
            log.info("Cinematic engine + procedural 3D scenes enabled (MoviePy path)")
        else:
            log.info("Cinematic engine loaded — hyper-realistic 3D motion enabled")
except ImportError:
    _CINEMATIC_ENGINE = False
    log.warning("cinematic_engine not found — falling back to Ken Burns")

WIDTH, HEIGHT   = 1920, 1080
FPS             = 24
WORDS_PER_CUT   = 7       # words shown per visual cut (caption style)
MIN_CUT_DUR     = 2.8     # minimum seconds per cut
INTRO_DUR       = 0.50    # section intro card duration
CALLOUT_DUR     = 2.0     # stat-reveal card duration
POP_DUR         = 0.14    # pop-in animation duration (ease from 1.11× to 1.0×)

_SCENE_STYLES = {
    "cold_open",
    "human_action",
    "documentary",
    "evidence",
    "split_screen",
    "timeline",
    "blueprint",
    "action_steps",
    "finale",
    "cinematic",
}

_RETENTION_COLORS = {
    # Emotional ladder — aligned to psychological arc
    "danger":    (255,  45,  45),   # hook/loss-frame: scroll-stopping red
    "curiosity": (255, 212,   0),   # surprise: electric yellow mystery
    "evidence":  (  0, 229, 255),   # tension: cold blue pressure / data trust
    "mechanism": (  0, 229, 255),   # tension variant
    "payoff":    (255, 180,   0),   # relief: warm amber gold resolution
    "action":    (  0, 255, 133),   # end-screen: momentum green
}

# Emotional ladder background tint — overlaid on raw video/photo to mood-sync
# Applied at 8% alpha so it colours the scene without washing it out
_MOOD_TINTS = {
    "danger":    np.array([80,  8,   8],  dtype=np.float32),   # deep red
    "curiosity": np.array([40, 30,   4],  dtype=np.float32),   # amber dark
    "evidence":  np.array([ 4, 28,  72],  dtype=np.float32),   # cold blue
    "mechanism": np.array([ 4, 24,  72],  dtype=np.float32),
    "payoff":    np.array([60, 36,   4],  dtype=np.float32),   # warm gold
    "action":    np.array([ 4, 56,  18],  dtype=np.float32),   # deep green
}


def _accent_for(color_mood: str) -> tuple[int, int, int]:
    mood = (color_mood or "curiosity").strip().lower().replace(" ", "_")
    return _RETENTION_COLORS.get(mood, _RETENTION_COLORS["curiosity"])


def _apply_mood_tint(frame: np.ndarray, color_mood: str, strength: float = 0.08) -> np.ndarray:
    """
    Overlay a subtle mood-coloured tint on a frame (RGB uint8).
    Aligns background color temperature to the emotional arc.
    strength=0.08 tints gently without washing out the image.
    """
    tint = _MOOD_TINTS.get(
        (color_mood or "curiosity").strip().lower().replace(" ", "_"),
        _MOOD_TINTS["curiosity"],
    )
    f = frame.astype(np.float32)
    f = f * (1.0 - strength) + tint[np.newaxis, np.newaxis, :] * strength * (255.0 / 80.0)
    return np.clip(f, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# MoviePy v1 / v2 detection  (lazy + cached)
# ─────────────────────────────────────────────────────────────────────────────

_V1: "bool | None" = None

def _is_v1() -> bool:
    global _V1
    if _V1 is None:
        try:
            from moviepy.editor import ImageClip   # noqa: F401
            _V1 = True
        except ImportError:
            _V1 = False
    return _V1


def _make_videoclip(make_frame_fn, duration: float):
    """Create a VideoClip and set fps — works on both MoviePy versions."""
    try:
        from moviepy.editor import VideoClip
    except ImportError:
        from moviepy import VideoClip
    clip = VideoClip(make_frame_fn, duration=duration)
    try:
        return clip.with_fps(FPS)   # v2
    except AttributeError:
        return clip.set_fps(FPS)    # v1


def _set_audio(clip, audio):
    return clip.set_audio(audio) if _is_v1() else clip.with_audio(audio)


def _subclip(clip, start, end):
    """
    Trim clip to [start, end] seconds.
    Tries every API name across MoviePy versions (newest first):
      subclipped  — MoviePy 2.1.x  (CompositeVideoClip + plain clips)
      with_subclip — MoviePy 2.0 transition API
      subclip      — MoviePy 1.x
      with_end     — last-resort: just cut the tail
    """
    if hasattr(clip, "subclipped"):          # MoviePy 2.1.x
        return clip.subclipped(start, end)
    if hasattr(clip, "with_subclip"):        # MoviePy 2.0 transition
        return clip.with_subclip(start, end)
    if hasattr(clip, "subclip"):             # MoviePy 1.x
        return clip.subclip(start, end)
    if hasattr(clip, "with_end"):            # absolute last resort
        return clip.with_end(end)
    if hasattr(clip, "set_end"):
        return clip.set_end(end)
    log.warning("_subclip: no trim method found — returning full clip")
    return clip


# ─────────────────────────────────────────────────────────────────────────────
# Easing functions
# ─────────────────────────────────────────────────────────────────────────────

def _ease_out_quad(x: float) -> float:
    """Fast start, decelerates smoothly to a stop."""
    x = max(0.0, min(1.0, x))
    return 1.0 - (1.0 - x) ** 2


def _ease_out_back(x: float, s: float = 1.70158) -> float:
    """Ease-out with a slight overshoot / spring pop."""
    x = max(0.0, min(1.0, x))
    c3 = s + 1
    return 1 + c3 * (x - 1) ** 3 + s * (x - 1) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# Numpy compositing helper
# ─────────────────────────────────────────────────────────────────────────────

def _paste_np(base: np.ndarray, overlay: np.ndarray,
              x: int, y: int, alpha_factor: float = 1.0) -> None:
    """
    Alpha-composite an RGBA overlay onto an RGB base array **in-place**.
    Clips to canvas bounds automatically.

    base    : (H, W, 3) uint8
    overlay : (h, w, 4) uint8
    """
    bh, bw = base.shape[:2]
    oh, ow = overlay.shape[:2]
    y1, y2 = max(0, y), min(bh, y + oh)
    x1, x2 = max(0, x), min(bw, x + ow)
    if y1 >= y2 or x1 >= x2:
        return
    oy1, ox1 = y1 - y, x1 - x
    oy2, ox2 = oy1 + (y2 - y1), ox1 + (x2 - x1)

    a  = overlay[oy1:oy2, ox1:ox2, 3:4].astype(np.float32) * alpha_factor / 255.0
    ov = overlay[oy1:oy2, ox1:ox2, :3].astype(np.float32)
    bg = base[y1:y2, x1:x2].astype(np.float32)
    base[y1:y2, x1:x2] = np.clip(bg * (1.0 - a) + ov * a, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Particle system
# ─────────────────────────────────────────────────────────────────────────────

def _particles_layer(w: int, h: int, t: float,
                     count: int = 28, seed: int = 0) -> np.ndarray:
    """
    Returns a (H, W, 4) RGBA numpy array with a field of gently drifting
    particles.  Particles drift upward and pulse in alpha — gives every
    scene a sense of energy without distracting from the text.
    """
    img  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    rng  = np.random.RandomState(seed)

    ox = rng.uniform(0,  w, count)
    oy = rng.uniform(0,  h, count)
    vx = rng.uniform(-6, 6,  count)
    vy = rng.uniform(-22, -8, count)          # upward drift
    rs = rng.randint(2, 7,   count)
    cs = rng.randint(0, 3,   count)

    COLORS = [(255, 214, 0), (255, 130, 50), (140, 200, 255)]   # gold/amber/ice

    for i in range(count):
        cx = int((ox[i] + vx[i] * t) % w)
        cy = int(((oy[i] + vy[i] * t) % h + h) % h)
        ri = int(rs[i])
        pulse = max(18, min(95, int(48 + 36 * math.sin(t * 1.4 + i * 0.85))))
        r, g, b = COLORS[cs[i]]
        draw.ellipse([cx - ri, cy - ri, cx + ri, cy + ri],
                     fill=(r, g, b, pulse))

    return np.array(img)


# ─────────────────────────────────────────────────────────────────────────────
# Background pre-computation cache  (avoids repeated LANCZOS at 110% size)
# ─────────────────────────────────────────────────────────────────────────────

_BG_ENLARGE_CACHE: dict = {}


def _get_enlarged_bg(bg_arr: np.ndarray) -> np.ndarray:
    """
    Return a vignette-applied, 110%-enlarged version of bg_arr, cached by
    array identity so each background image is only processed once per run.
    """
    key = id(bg_arr)
    if key not in _BG_ENLARGE_CACHE:
        margin  = int(HEIGHT * 0.10)
        big_h   = HEIGHT + margin * 2
        big_w   = int(big_h * WIDTH / HEIGHT)
        _BG_ENLARGE_CACHE[key] = _vignette(
            np.array(
                Image.fromarray(bg_arr).resize((big_w, big_h), Image.LANCZOS)
            ),
            0.62,
        )
    return _BG_ENLARGE_CACHE[key]


# ─────────────────────────────────────────────────────────────────────────────
# Fonts + text helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    mac = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    linux = [
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans-{'Bold' if bold else 'Regular'}.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in (mac if sys.platform == "darwin" else []) + linux:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _text_w(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    try:
        return int(draw.textlength(text, font=font))
    except AttributeError:
        return draw.textsize(text, font=font)[0]


def _rounded_rect(draw, xy, radius: int, fill) -> None:
    try:
        draw.rounded_rectangle(xy, radius=radius, fill=fill)
    except AttributeError:
        draw.rectangle(xy, fill=fill)


def _scene_style(value: str, fallback_idx: int = 0) -> str:
    style = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if style in _SCENE_STYLES:
        return style
    cycle = [
        "human_action", "split_screen", "evidence", "documentary",
        "timeline", "blueprint", "action_steps", "finale",
    ]
    return cycle[fallback_idx % len(cycle)]


def _fit_text_lines(text: str, font, max_width: int, max_lines: int = 3) -> list[str]:
    """Wrap text by measured width so side panels do not overflow."""
    words = text.split()
    if not words:
        return []
    dummy = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    lines = []
    cur = ""
    for word in words:
        test = word if not cur else f"{cur} {word}"
        if _text_w(dummy, test, font) <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
            if len(lines) >= max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Background helpers
# ─────────────────────────────────────────────────────────────────────────────

import json as _json

# ─────────────────────────────────────────────────────────────────────────────
# Persistent cross-video background deduplication
# ─────────────────────────────────────────────────────────────────────────────
# TWO layers of dedup:
#   1. URL-based  — exact URL match (fast, zero-cost)
#   2. Phash-based — perceptual image hash (catches same photo at different URLs,
#                    CDN variants, resized crops)
#
# Both stores live in output/seen_bg_*.json and survive across all runs.
# No image ever repeats in any video — long-form or Shorts.
# ─────────────────────────────────────────────────────────────────────────────

_BG_URL_STORE   = Path(__file__).resolve().parent / "output" / "seen_bg_urls.json"
_BG_HASH_STORE  = Path(__file__).resolve().parent / "output" / "seen_bg_hashes.json"

# Synonym sets — tried automatically when a keyword's Pexels pool is exhausted
_KW_SYNONYMS: dict[str, list[str]] = {
    "crypto":        ["blockchain finance dark", "digital currency technology", "defi fintech dark room"],
    "bitcoin":       ["cryptocurrency market charts", "digital asset investment", "crypto exchange screens"],
    "stock":         ["wall street finance", "trading floor dramatic", "investment banker office"],
    "finance":       ["banking corporate dramatic", "money economy business", "financial advisor dramatic"],
    "economy":       ["recession unemployment line", "economic growth chart person", "inflation market crash"],
    "money":         ["wealth management office", "cash transaction dramatic", "luxury penthouse executive"],
    "tech":          ["silicon valley startup", "computer science laboratory", "innovation engineering dramatic"],
    "ai":            ["machine learning visualization", "neural network server room", "autonomous robot dramatic"],
    "robot":         ["automation factory worker", "mechanical arm precision", "drone aerial surveillance"],
    "cybersecurity": ["network breach alert screens", "password encryption dramatic", "firewall server glow"],
    "social media":  ["content creator studio ring light", "smartphone viral reaction", "live streaming dramatic"],
    "health":        ["hospital emergency dramatic", "fitness workout intense", "nutrition science lab"],
    "science":       ["physics experiment dramatic", "biology lab close-up", "chemistry reaction intense"],
    "space":         ["nasa control room dramatic", "telescope dark sky", "satellite orbit dramatic"],
    "nature":        ["wilderness survival dramatic", "storm approaching landscape", "glacier climate dramatic"],
    "gaming":        ["game developer office screens", "virtual reality headset dramatic", "gaming tournament crowd"],
    "music":         ["recording studio vocal booth", "concert crowd energy", "DJ mixing dramatic light"],
    "sports":        ["athlete training sunrise", "championship trophy dramatic", "stadium crowd energy"],
    "movies":        ["cinematic director chair", "film festival red carpet", "special effects set dramatic"],
    "viral":         ["breaking news anchor dramatic", "scandal press conference", "trending moment crowd"],
    "climate":       ["solar farm vast landscape", "flood disaster dramatic", "wildfire dramatic sky"],
    "politics":      ["senate chamber dramatic", "election night dramatic", "diplomat meeting tension"],
    "crime":         ["police evidence board", "surveillance camera dramatic", "criminal trial testimony"],
    "food":          ["street food market dramatic", "Michelin restaurant plating", "farm harvest dramatic"],
    "travel":        ["airport departure dramatic", "backpacker mountain summit", "exotic destination dramatic"],
    "business":      ["merger negotiation boardroom", "startup pitch investor", "global headquarters dramatic"],
    "relationships": ["conversation emotional cafe", "reunion embrace dramatic", "negotiation trust handshake"],
    "education":     ["library study dramatic", "graduation ceremony emotional", "mentor student dramatic"],
}


def _load_persistent_bg_urls() -> set[str]:
    try:
        if _BG_URL_STORE.exists():
            data = _json.loads(_BG_URL_STORE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return set(data)
    except Exception:
        pass
    return set()


def _load_persistent_bg_hashes() -> set[int]:
    try:
        if _BG_HASH_STORE.exists():
            data = _json.loads(_BG_HASH_STORE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return set(int(h) for h in data)
    except Exception:
        pass
    return set()


# In-memory mirrors — loaded at import time.
# _reset_bg_cache() reloads from disk so all previously-used URLs/hashes stay blocked.
_SEEN_BG_URLS:   set[str] = _load_persistent_bg_urls()
_SEEN_BG_HASHES: set[int] = _load_persistent_bg_hashes()

# Pending writes accumulated during a render — flushed once at end via _flush_bg_stores()
_PENDING_BG_URLS:   set[str] = set()
_PENDING_BG_HASHES: set[int] = set()


def _dhash(img: Image.Image, hash_size: int = 8) -> int:
    """
    Difference hash (dhash) — compares horizontally adjacent pixel pairs.
    More robust than average-hash: works on real photos with any luminance range.
    hash_size=8 produces a 64-bit hash.
    Returns 0 for degenerate (perfectly uniform) images — treated as unchecked.
    """
    small = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    arr   = np.array(small, dtype=np.int16)
    diff  = arr[:, 1:] > arr[:, :-1]   # True where right pixel is brighter
    bits  = diff.flatten()
    return int(sum(int(b) << i for i, b in enumerate(bits)))


def _phash_distance(h1: int, h2: int) -> int:
    """Hamming distance between two 64-bit hashes."""
    return bin(h1 ^ h2).count("1")


def _is_visually_duplicate(img: Image.Image) -> bool:
    """
    Return True if img is perceptually too similar to any previously used image.
    Uses dhash with a Hamming-distance threshold of 8/64 bits (~12.5% tolerance).
    Skips check for degenerate images (hash == 0) to avoid false positives on
    solid-color fallback gradients.
    """
    h = _dhash(img)
    if h == 0:
        return False    # degenerate/uniform image — cannot meaningfully hash, let it through
    for seen_h in _SEEN_BG_HASHES:
        if seen_h == 0:
            continue    # skip degenerate stored hashes
        if _phash_distance(h, seen_h) < 8:   # threshold: <8/64 bits differ → near-duplicate
            return True
    return False


def _register_bg(url: str, img: Image.Image) -> None:
    """Mark a URL + its dhash as used (in-memory + pending batch write)."""
    global _PENDING_BG_URLS, _PENDING_BG_HASHES
    h = _dhash(img)
    _SEEN_BG_URLS.add(url)
    _SEEN_BG_HASHES.add(h)
    _PENDING_BG_URLS.add(url)
    _PENDING_BG_HASHES.add(h)


def _flush_bg_stores() -> None:
    """Batch-write all pending URLs and hashes to disk.
    Called once at the end of each render — avoids one disk write per image."""
    global _PENDING_BG_URLS, _PENDING_BG_HASHES
    if not _PENDING_BG_URLS and not _PENDING_BG_HASHES:
        return
    try:
        _BG_URL_STORE.parent.mkdir(parents=True, exist_ok=True)
        # URLs
        existing_urls = _load_persistent_bg_urls()
        existing_urls |= _PENDING_BG_URLS
        _BG_URL_STORE.write_text(
            _json.dumps(sorted(existing_urls), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Hashes
        existing_hashes = _load_persistent_bg_hashes()
        existing_hashes |= _PENDING_BG_HASHES
        _BG_HASH_STORE.write_text(
            _json.dumps(sorted(existing_hashes), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info(f"  BG store: +{len(_PENDING_BG_URLS)} URLs, +{len(_PENDING_BG_HASHES)} hashes persisted")
    except Exception as _e:
        log.warning(f"Could not flush bg stores: {_e}")
    finally:
        _PENDING_BG_URLS   = set()
        _PENDING_BG_HASHES = set()


def _reset_bg_cache() -> None:
    """Reload persistent stores at render start.
    Does NOT clear the seen sets — that would re-allow previously used images."""
    global _SEEN_BG_URLS, _SEEN_BG_HASHES, _BG_ENLARGE_CACHE
    _SEEN_BG_URLS   = _load_persistent_bg_urls()
    _SEEN_BG_HASHES = _load_persistent_bg_hashes()
    _BG_ENLARGE_CACHE = {}


def _fetch_pexels(keyword: str, size: tuple[int, int] = (WIDTH, HEIGHT)) -> "Image.Image | None":
    """
    Fetch a unique Pexels photo — NEVER returns an image already used in any video.

    Strategy:
      • Samples up to 15 random pages (per_page=80 → up to 1,200 candidates)
      • Stops as soon as a URL+phash-clear image is found
      • If keyword pool is exhausted, auto-retries with synonym keywords
      • NEVER falls back to a seen image — returns None to force cascade to next source
    """
    key = os.environ.get("PEXELS_API_KEY", "")
    if not key:
        return None

    import random
    orient = "landscape" if size[0] >= size[1] else "portrait"

    def _try_keyword(kw: str) -> "Image.Image | None":
        # Sample up to 15 pages randomly from a pool of 20 pages (1,200 candidates max)
        page_pool = list(range(1, 21))
        random.shuffle(page_pool)
        for page in page_pool[:15]:
            try:
                r = requests.get(
                    "https://api.pexels.com/v1/search",
                    headers={"Authorization": key},
                    params={"query": kw, "per_page": 80, "orientation": orient, "page": page},
                    timeout=10,
                )
                if r.status_code != 200:
                    continue
                photos = r.json().get("photos", [])
                if not photos:
                    break  # no more results for this keyword at this depth
                random.shuffle(photos)
                for photo in photos:
                    url = photo["src"]["large2x"]
                    if url in _SEEN_BG_URLS:
                        continue  # URL already used — skip
                    try:
                        img = Image.open(BytesIO(requests.get(url, timeout=15).content)).convert("RGB")
                    except Exception:
                        continue
                    if _is_visually_duplicate(img):
                        # Same-looking photo at a different URL — skip
                        _SEEN_BG_URLS.add(url)  # poison the URL so we never re-fetch
                        continue
                    img = img.resize(size, Image.LANCZOS)
                    _register_bg(url, img)
                    return img
            except Exception as e:
                log.debug(f"Pexels page {page} for '{kw}': {e}")
                continue
        return None  # exhausted — do NOT fall back to seen image

    # 1. Try the requested keyword
    img = _try_keyword(keyword)
    if img:
        return img

    # 2. Try synonym keywords automatically before giving up
    synonyms = _KW_SYNONYMS.get(keyword.split()[0].lower(), [])
    for syn in synonyms:
        img = _try_keyword(syn)
        if img:
            log.debug(f"Pexels: keyword '{keyword}' exhausted → synonym '{syn}' succeeded")
            return img

    log.debug(f"Pexels: all pools exhausted for '{keyword}' — cascading to next source")
    return None


def _fetch_pexels_video_frame(keyword: str, size: tuple[int, int] = (WIDTH, HEIGHT)) -> "Image.Image | None":
    """
    Extract a unique cinematic still from a Pexels video.
    Tries ALL returned videos (not just the first) before giving up.
    Uses HTTP Range streaming — downloads only the first 8 MB to extract a frame.
    """
    key = os.environ.get("PEXELS_API_KEY", "")
    if not key:
        return None

    import random
    orient = "landscape" if size[0] >= size[1] else "portrait"
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": key},
            params={"query": keyword, "per_page": 15, "orientation": orient},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        videos = r.json().get("videos", [])
        if not videos:
            return None
        random.shuffle(videos)
    except Exception as e:
        log.warning(f"Pexels video search '{keyword}': {e}")
        return None

    # Try EVERY video until we find one whose link is unseen
    for video in videos:
        files = sorted(video.get("video_files", []), key=lambda f: f.get("width", 0), reverse=True)
        hd    = [f for f in files if f.get("width", 0) >= 1280]
        if not (hd or files):
            continue
        link = (hd or files)[0]["link"]
        if link in _SEEN_BG_URLS:
            continue  # try next video

        tmp_path: str | None = None
        try:
            _MAX_VIDEO_BYTES = 8 * 1024 * 1024  # 8 MB — moov atom safe margin
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                tmp_path = tf.name
            with requests.get(link, timeout=45, stream=True) as resp:
                resp.raise_for_status()
                downloaded = 0
                with open(tmp_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=65536):
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= _MAX_VIDEO_BYTES:
                            break
            # Remux the partial file: fixes container header so vc.duration
            # matches the frames actually downloaded (not the original video length)
            _remux_video(tmp_path)
            try:
                from moviepy import VideoFileClip
            except ImportError:
                from moviepy.editor import VideoFileClip
            with VideoFileClip(tmp_path, audio=False) as vc:
                # Read from 15-70% of the REAL (post-remux) duration
                t     = vc.duration * random.uniform(0.15, 0.70)
                frame = vc.get_frame(t)
            os.unlink(tmp_path)
            tmp_path = None
            img = Image.fromarray(frame).resize(size, Image.LANCZOS)
            if _is_visually_duplicate(img):
                _SEEN_BG_URLS.add(link)
                continue
            _register_bg(link, img)
            return img
        except Exception as e:
            log.debug(f"Pexels video frame '{link[:60]}': {e}")
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            continue  # try next video

    log.debug(f"Pexels video: all {len(videos)} clips seen/failed for '{keyword}'")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO BACKGROUND PROVIDERS — hyper-realistic moving backgrounds
# ─────────────────────────────────────────────────────────────────────────────

def _remux_video(path: str) -> str:
    """
    Fix corrupted MP4 container metadata by remuxing with ffmpeg (stream copy, no re-encode).

    Problem: Downloaded video files (Pexels, Kling AI) sometimes have container
    headers that claim a longer duration than the actual encoded data — e.g.
    container says 8s but only 2.5s of frames exist. MoviePy reads the wrong
    duration, tries to seek past the real data, and emits:
      "N bytes wanted but 0 bytes read at frame M — using last valid frame"

    Fix: `ffmpeg -c copy` remuxes the stream into a new container. ffmpeg reads
    only the actual encoded packets, so the output container metadata correctly
    reflects the real duration. No quality loss (codec copy, not re-encode).

    Returns the same path on success (file is replaced in-place).
    Returns the original path unchanged if ffmpeg fails or isn't available.
    """
    import subprocess
    fixed = path + ".remux.mp4"
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",            # overwrite output
                "-v", "error",             # suppress all output except errors
                "-i", path,               # input
                "-c", "copy",             # stream copy — no re-encode
                "-movflags", "+faststart", # fix moov atom position
                fixed,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and os.path.exists(fixed):
            fixed_size = os.path.getsize(fixed)
            orig_size  = os.path.getsize(path)
            if fixed_size > 1000:
                os.replace(fixed, path)  # atomic rename
                log.debug(f"  Remux: {orig_size//1024}KB → {fixed_size//1024}KB  ({path})")
            else:
                os.unlink(fixed)
                log.debug(f"  Remux produced empty file — keeping original")
        elif result.returncode != 0:
            log.debug(f"  Remux ffmpeg error: {result.stderr[:200]}")
            if os.path.exists(fixed):
                os.unlink(fixed)
    except FileNotFoundError:
        pass  # ffmpeg not in PATH — skip remux
    except Exception as _e:
        log.debug(f"  Remux failed: {_e}")
        if os.path.exists(fixed):
            try: os.unlink(fixed)
            except OSError: pass
    return path

def _fetch_pexels_video_clip(
    keyword: str,
    target_duration: float = 6.0,
    size: tuple[int, int] = (_SHORTS_W if False else (1920, 1080)),  # resolved at call time
) -> "str | None":
    """
    Download a real Pexels video clip and return the temp file path.
    Returns path to an MP4 file (caller must delete after use).
    This is FREE and gives REAL human motion, real camera movement, real cinematics.

    Set PEXELS_VIDEO_BG=1 (default if PEXELS_API_KEY is set) to enable.
    """
    if _PEXELS_VIDEO_BG != "1":
        return None
    key = os.environ.get("PEXELS_API_KEY", "")
    if not key:
        return None

    import random
    orient = "portrait" if size[0] < size[1] else "landscape"
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": key},
            params={"query": keyword, "per_page": 15, "orientation": orient,
                    "size": "medium"},  # medium = 1280x720 min, manageable file sizes
            timeout=12,
        )
        if r.status_code != 200:
            return None
        videos = r.json().get("videos", [])
        if not videos:
            return None
        random.shuffle(videos)
    except Exception as e:
        log.warning(f"Pexels video clip search '{keyword}': {e}")
        return None

    for video in videos:
        files = video.get("video_files", [])
        # Pick the best resolution that fits our target — prefer HD, avoid 4K (too large)
        if size[0] < size[1]:  # portrait
            portrait_files = [f for f in files if f.get("height", 0) > f.get("width", 0)]
            candidates = portrait_files or files
        else:
            candidates = files
        # Sort by width descending, cap at 1920
        candidates = sorted(candidates, key=lambda f: f.get("width", 0), reverse=True)
        candidates = [f for f in candidates if 720 <= f.get("width", 0) <= 1920] or candidates[:1]
        if not candidates:
            continue
        link = candidates[0]["link"]
        if link in _SEEN_BG_URLS:
            continue

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False,
                                             prefix="yt_pexvid_") as tf:
                tmp_path = tf.name
            # Download full clip (videos are 5-30s, typically 10-40MB)
            with requests.get(link, timeout=60, stream=True) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=131072):
                        fh.write(chunk)
            # Fix container metadata (corrupted duration headers cause MoviePy read errors)
            _remux_video(tmp_path)
            # Validate
            try:
                from moviepy import VideoFileClip as _VFC
            except ImportError:
                from moviepy.editor import VideoFileClip as _VFC
            with _VFC(tmp_path, audio=False) as _vc:
                actual_dur = _vc.duration
            if actual_dur < 1.0:
                raise ValueError(f"clip too short: {actual_dur:.1f}s")
            _SEEN_BG_URLS.add(link)
            log.info(f"  Pexels video clip: {actual_dur:.1f}s → {tmp_path}")
            return tmp_path
        except Exception as e:
            log.debug(f"Pexels video clip download '{link[:60]}': {e}")
            if tmp_path:
                try: os.unlink(tmp_path)
                except OSError: pass
            continue

    log.debug(f"Pexels video clip: all candidates seen/failed for '{keyword}'")
    return None


def _generate_video_replicate_ltx(
    prompt: str,
    duration_seconds: float = 5.0,
    size: tuple[int, int] = (544, 960),  # portrait default for Shorts
) -> "str | None":
    """
    Generate hyper-realistic AI video via Replicate LTX-Video (~$0.07/clip).
    Returns path to temp MP4. Set LTX_VIDEO_BG=1 to enable.

    LTX-Video (Lightricks): fastest open-source T2V model, licensed Apache 2.0.
    Generates smooth, cinematic 5-second clips on Replicate's GPU cloud.
    """
    if _LTX_VIDEO_BG != "1":
        return None
    api_key = os.environ.get("REPLICATE_API_KEY", "")
    if not api_key:
        return None

    w, h = size
    # LTX-Video requires dimensions divisible by 32
    w = (w // 32) * 32
    h = (h // 32) * 32
    num_frames = max(25, min(121, int(duration_seconds * 24)))

    safe_prompt = (
        "Hyper-realistic cinematic video, no text, no watermarks, no logos, "
        "smooth camera motion, professional cinematography. "
        f"{prompt[:600]}"
    )
    try:
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
            "Prefer": "wait",
        }
        payload = {
            "input": {
                "prompt": safe_prompt,
                "negative_prompt": "text, watermark, logo, cartoon, animation, blurry, static, still",
                "num_frames": num_frames,
                "frame_rate": 24,
                "width": w,
                "height": h,
                "guidance_scale": 3.5,
                "num_inference_steps": 30,
                "seed": int(abs(hash(prompt)) % 999999),
            }
        }
        log.info(f"  LTX-Video generating: {prompt[:60]}… ({w}×{h}, {num_frames}f)")
        resp = requests.post(
            "https://api.replicate.com/v1/models/lightricks/ltx-video/predictions",
            headers=headers,
            json=payload,
            timeout=300,
        )
        if resp.status_code not in (200, 201):
            log.warning(f"LTX-Video returned {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        # Poll if not synchronous
        pred_id = data.get("id")
        if data.get("status") not in ("succeeded", "processing") and pred_id:
            for _ in range(60):
                import time as _t; _t.sleep(3)
                poll = requests.get(
                    f"https://api.replicate.com/v1/predictions/{pred_id}",
                    headers=headers, timeout=15
                )
                data = poll.json()
                if data.get("status") == "succeeded":
                    break
                if data.get("status") in ("failed", "canceled"):
                    log.warning(f"LTX-Video prediction failed: {data.get('error')}")
                    return None

        output = data.get("output")
        if isinstance(output, list):
            output = output[0]
        if not (isinstance(output, str) and output.startswith("http")):
            log.warning(f"LTX-Video: unexpected output: {str(output)[:100]}")
            return None

        # Download the generated video
        vid_resp = requests.get(output, timeout=60, stream=True)
        vid_resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False,
                                         prefix="yt_ltxvid_") as tf:
            tmp_path = tf.name
        with open(tmp_path, "wb") as fh:
            for chunk in vid_resp.iter_content(chunk_size=131072):
                fh.write(chunk)
        _remux_video(tmp_path)
        log.info(f"  LTX-Video clip saved → {tmp_path}")
        return tmp_path
    except Exception as e:
        log.warning(f"LTX-Video generation failed: {e}")
        return None


def _generate_video_kling(
    prompt: str,
    duration_seconds: float = 5.0,
    size: tuple[int, int] = (544, 960),
) -> "str | None":
    """
    Generate hyper-realistic Zach King-level AI video via Kling AI on fal.ai (~$0.38/clip).
    Returns path to temp MP4. Set KLING_VIDEO_BG=1 and KLING_API_KEY to enable.

    Kling 1.6 Pro: best motion quality, physics-accurate, supports image-to-video.
    Get API key at https://fal.ai (key format: key_xxxxxxxx)
    """
    if _KLING_VIDEO_BG != "1":
        return None
    if not _KLING_API_KEY:
        return None

    w, h = size
    aspect = "9:16" if h > w else "16:9"
    safe_prompt = (
        "Hyper-realistic cinematic video, photorealistic, professional camera work, "
        "smooth motion, no text, no watermarks. "
        f"{prompt[:500]}"
    )
    try:
        headers = {
            "Authorization": f"Key {_KLING_API_KEY}",
            "Content-Type": "application/json",
        }
        # Kling API only accepts duration as the string "5" or "10" — nothing else
        kling_duration = "10" if duration_seconds > 7 else "5"
        payload = {
            "prompt": safe_prompt,
            "negative_prompt": "text, watermark, cartoon, low quality, static, blurry",
            "duration": kling_duration,   # MUST be string "5" or "10"
            "aspect_ratio": aspect,
            "cfg_scale": 0.5,
        }
        log.info(f"  Kling AI generating: {prompt[:60]}… ({aspect}, {int(duration_seconds)}s)")
        # Submit request
        resp = requests.post(
            "https://queue.fal.run/fal-ai/kling-video/v1.6/pro/text-to-video",
            headers=headers,
            json=payload,
            timeout=30,
        )
        # Always log status + first 300 chars of body for diagnostics
        log.debug(f"  Kling submit: HTTP {resp.status_code}  body={resp.text[:300]!r}")
        if resp.status_code not in (200, 201, 202):
            log.warning(f"Kling returned HTTP {resp.status_code}: {resp.text[:300]}")
            return None
        raw = resp.text.strip()
        if not raw:
            log.warning(f"Kling returned HTTP {resp.status_code} with empty body — "
                        "check fal.ai API key format (should be key_xxxxxxxx or uuid:secret)")
            return None
        try:
            data = resp.json()
        except Exception as _je:
            log.warning(f"Kling JSON parse failed ({_je}): body={raw[:300]!r}")
            return None

        # fal.ai queue: poll for result using URLs from the submit response.
        # IMPORTANT: the status/result endpoints do NOT include the model version path
        # (e.g. /v1.6/pro/text-to-video). fal.ai provides the correct URLs directly.
        request_id   = data.get("request_id")
        # Use fal.ai-provided URLs; fall back to base path (no version segment) if absent
        _base        = "https://queue.fal.run/fal-ai/kling-video"
        status_url   = data.get("status_url")   or f"{_base}/requests/{request_id}/status"
        response_url = data.get("response_url") or f"{_base}/requests/{request_id}"
        log.info(f"  Kling queued — polling status_url={status_url}")

        if request_id:
            import time as _t
            for attempt in range(180):  # up to 9 min (180 × 3s)
                _t.sleep(3)
                try:
                    poll = requests.get(status_url, headers=headers, timeout=20)
                except Exception as _pe:
                    log.debug(f"  Kling poll error ({_pe}), retrying…")
                    continue
                raw_poll = poll.text.strip()
                if not raw_poll:
                    log.debug(f"  Kling poll: empty body (attempt {attempt}), retrying…")
                    continue
                try:
                    pdata = poll.json()
                except Exception:
                    log.debug(f"  Kling poll JSON error (attempt {attempt}), retrying…")
                    continue

                poll_status = pdata.get("status", "")
                log.debug(f"  Kling poll [{attempt}] status={poll_status}")

                if poll_status == "COMPLETED":
                    try:
                        result_resp = requests.get(response_url, headers=headers, timeout=30)
                        data = result_resp.json()
                        log.info(f"  Kling completed — fetching video URL from {response_url}")
                    except Exception as _re:
                        log.warning(f"Kling result fetch failed: {_re}")
                        return None
                    break
                if poll_status in ("FAILED", "CANCELLED", "ERROR"):
                    log.warning(f"Kling job {poll_status}: {pdata}")
                    return None
                # IN_QUEUE / IN_PROGRESS — keep polling
                if attempt % 10 == 0:
                    log.info(f"  Kling still {poll_status} (attempt {attempt}/180)…")
            else:
                log.warning("Kling timed out after 9 minutes — falling back to Pexels video")
                return None

        # Extract video URL from response
        video_url = None
        video_field = data.get("video") or {}
        if isinstance(video_field, dict):
            video_url = video_field.get("url")
        elif isinstance(video_field, str):
            video_url = video_field
        if not video_url:
            videos_list = data.get("videos", [])
            if videos_list:
                video_url = (videos_list[0] or {}).get("url")
        if not video_url:
            log.warning(f"Kling: no video URL in response: {str(data)[:200]}")
            return None

        # Download
        vid_resp = requests.get(video_url, timeout=90, stream=True)
        vid_resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False,
                                         prefix="yt_kling_") as tf:
            tmp_path = tf.name
        with open(tmp_path, "wb") as fh:
            for chunk in vid_resp.iter_content(chunk_size=131072):
                fh.write(chunk)
        # Fix container metadata — Kling often returns MP4s where the moov atom
        # claims a longer duration than the actual encoded frames
        _remux_video(tmp_path)
        log.info(f"  Kling AI clip saved → {tmp_path}")
        return tmp_path
    except Exception as e:
        log.warning(f"Kling video generation failed: {e}")
        return None


def _get_video_bg_clip(
    keyword: str,
    scene_prompt: str,
    duration_needed: float,
    size: tuple[int, int],
    slot_idx: int = 0,
) -> "str | None":
    """
    Unified video background fetcher — tries all video providers in priority order:
    1. Kling AI     (best quality, opt-in, ~$0.38/clip)
    2. LTX-Video    (AI generation, opt-in, ~$0.07/clip)
    3. Pexels Video (real footage, FREE, on by default)

    Returns path to temp MP4, or None (fall back to static image).
    Caller is responsible for deleting the temp file.
    """
    # 1. Kling (highest quality AI video)
    if _KLING_VIDEO_BG == "1" and _KLING_API_KEY:
        path = _generate_video_kling(scene_prompt, duration_seconds=duration_needed, size=size)
        if path:
            return path

    # 2. LTX-Video via Replicate (AI generated, cinematic motion)
    if _LTX_VIDEO_BG == "1":
        path = _generate_video_replicate_ltx(scene_prompt, duration_seconds=duration_needed, size=size)
        if path:
            return path

    # 3. Pexels real video (free, real human motion)
    if _PEXELS_VIDEO_BG == "1" and os.environ.get("PEXELS_API_KEY"):
        path = _fetch_pexels_video_clip(keyword, target_duration=duration_needed, size=size)
        if path:
            return path

    return None  # fall through to static image


def _make_video_frame_at_t(
    vid_path: str,
    t: float,
    size: tuple[int, int],
    scale: float = 1.0,
    xd: int = 0,
    yd: int = 0,
) -> np.ndarray:
    """
    Extract one frame from a video file at time t, with Ken Burns scale+pan applied.
    Used for video-background clips — wraps around video duration automatically.
    """
    try:
        from moviepy import VideoFileClip as _VFC
    except ImportError:
        from moviepy.editor import VideoFileClip as _VFC
    W, H = size
    # Cache open VideoFileClip per path to avoid re-opening every frame
    cache = _make_video_frame_at_t._cache
    if vid_path not in cache:
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")  # suppress MoviePy truncated-frame warnings
            cache[vid_path] = _VFC(vid_path, audio=False)
    vc = cache[vid_path]
    # Clamp t to 95% of duration to avoid reading past valid frames
    t_safe = (t % vc.duration) if vc.duration > 0 else 0.0
    t_safe = min(t_safe, vc.duration * 0.95)
    import warnings as _warnings
    with _warnings.catch_warnings():
        # Suppress MoviePy/ffmpeg "N bytes wanted but 0 bytes read at frame M" UserWarning
        _warnings.filterwarnings("ignore", message=".*bytes wanted but 0 bytes read.*")
        _warnings.filterwarnings("ignore", message=".*Using the last valid frame.*")
        _warnings.filterwarnings("ignore", category=UserWarning, module="moviepy")
        frame = vc.get_frame(t_safe)  # numpy RGB array at native resolution
    frame_img = Image.fromarray(frame)
    # Scale + crop to target size
    nw = max(W, int(W * scale))
    nh = max(H, int(H * scale))
    frame_img = frame_img.resize((nw, nh), Image.BILINEAR)
    x0 = max(0, min((nw - W) // 2 + xd, nw - W))
    y0 = max(0, min((nh - H) // 2 + yd, nh - H))
    return np.array(frame_img.crop((x0, y0, x0 + W, y0 + H)))

_make_video_frame_at_t._cache: dict = {}


def _cleanup_video_cache() -> None:
    """Close all cached VideoFileClip handles and clear the cache."""
    for vc in _make_video_frame_at_t._cache.values():
        try: vc.close()
        except Exception: pass
    _make_video_frame_at_t._cache.clear()


def _generate_image_dalle(prompt: str, size: tuple[int, int] = (WIDTH, HEIGHT)) -> "Image.Image | None":
    """
    Generate a hyper-realistic cinematic background with DALL-E 3.
    Requires OPENAI_API_KEY. Set DALLE_BG=1 to enable (disabled by default
    to avoid unexpected spend).
    """
    if not os.environ.get("DALLE_BG"):
        return None
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        dalle_size = "1792x1024" if size[0] >= size[1] else "1024x1792"
        safe_prompt = (
            "Hyper-realistic cinematic photograph. "
            "No text, no watermarks, no logos, no UI elements. "
            f"{prompt[:800]}"
        )
        resp = client.images.generate(
            model="dall-e-3",
            prompt=safe_prompt,
            n=1,
            size=dalle_size,
            quality="hd",
            style="natural",
        )
        url = resp.data[0].url
        img = Image.open(BytesIO(requests.get(url, timeout=30).content)).convert("RGB")
        log.info(f"DALL-E 3 image generated for: {prompt[:60]}…")
        return img.resize(size, Image.LANCZOS)
    except Exception as e:
        log.warning(f"DALL-E 3 generation failed: {e}")
        return None


def _generate_image_pollinations(prompt: str, size: tuple = (WIDTH, HEIGHT)) -> "np.ndarray | None":
    """
    Pollinations.ai — completely free FLUX-based image generation.
    No rate limits documented; uses Authorization header with POLLINATIONS_KEY.
    Falls back to keyless GET URL if no key set (public endpoint).
    """
    import urllib.parse, io as _io, random as _rand
    w, h = size
    clean = (prompt or "cinematic scene")[:500].replace("\n", " ").strip()
    encoded = urllib.parse.quote(clean)
    seed    = _rand.randint(1, 999999)
    # Keyless public endpoint (no rate limits published, works without a key)
    url = (f"https://image.pollinations.ai/prompt/{encoded}"
           f"?width={w}&height={h}&seed={seed}&nologo=true&enhance=true&model=flux")
    headers = {}
    if _POLLINATIONS_KEY:
        headers["Authorization"] = f"Bearer {_POLLINATIONS_KEY}"
    try:
        resp = requests.get(url, headers=headers, timeout=60)
        if resp.status_code == 200 and len(resp.content) > 10_000:
            img = Image.open(_io.BytesIO(resp.content)).convert("RGB")
            img = img.resize(size, Image.LANCZOS)
            arr = np.array(img)
            log.info(f"  Pollinations.ai image generated ({w}x{h})")
            return arr
    except Exception as e:
        log.warning(f"Pollinations.ai failed: {e}")
    return None


def _generate_image_hf_flux(prompt: str, size: tuple = (WIDTH, HEIGHT)) -> "np.ndarray | None":
    """
    Hugging Face FLUX.1-schnell — free tier via HF Inference API.
    Requires HF_TOKEN env var. Apache 2.0 license — safe for commercial use.
    """
    if not _HF_TOKEN:
        return None
    import io as _io
    w, h = size
    clean = (prompt or "cinematic scene")[:500].replace("\n", " ").strip()
    try:
        resp = requests.post(
            "https://router.huggingface.co/hf-inference/models/"
            "black-forest-labs/FLUX.1-schnell",
            headers={"Authorization": f"Bearer {_HF_TOKEN}",
                     "Content-Type": "application/json"},
            json={"inputs": f"Hyper-realistic cinematic photograph. {clean}",
                  "parameters": {"width": min(w, 1344), "height": min(h, 768),
                                  "num_inference_steps": 4}},
            timeout=90,
        )
        if resp.status_code == 200 and len(resp.content) > 5_000:
            img = Image.open(_io.BytesIO(resp.content)).convert("RGB")
            img = img.resize(size, Image.LANCZOS)
            arr = np.array(img)
            log.info(f"  HF FLUX.1-schnell image generated ({w}x{h})")
            return arr
    except Exception as e:
        log.warning(f"HF FLUX failed: {e}")
    return None


def _fetch_unsplash(keyword: str, size: tuple = (WIDTH, HEIGHT)) -> "np.ndarray | None":
    """
    Unsplash API — free tier: 50 req/hr; real photography (not AI).
    Requires UNSPLASH_ACCESS_KEY env var. Must credit photographer per Unsplash TOS.
    """
    if not _UNSPLASH_KEY:
        return None
    import io as _io
    w, h = size
    try:
        resp = requests.get(
            "https://api.unsplash.com/search/photos",
            params={"query": keyword[:100], "per_page": 15, "orientation":
                    "landscape" if w > h else "portrait",
                    "content_filter": "high"},
            headers={"Authorization": f"Client-ID {_UNSPLASH_KEY}"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        results = resp.json().get("results", [])
        if not results:
            return None
        # Pick a result not already in seen URLs
        for photo in results:
            dl_url = photo.get("urls", {}).get("raw", "")
            if not dl_url or dl_url in _SEEN_BG_URLS:
                continue
            # Append size params to raw URL
            img_url = f"{dl_url}&w={w}&h={h}&fit=crop&auto=format&q=80"
            img_resp = requests.get(img_url, timeout=20)
            if img_resp.status_code == 200 and len(img_resp.content) > 20_000:
                arr = np.array(
                    Image.open(_io.BytesIO(img_resp.content)).convert("RGB").resize(size, Image.LANCZOS)
                )
                _register_bg(dl_url, Image.fromarray(arr))
                log.info(f"  Unsplash photo: {keyword[:40]}")
                return arr
    except Exception as e:
        log.warning(f"Unsplash failed: {e}")
    return None


def _generate_image_stability(prompt: str, size: tuple[int, int] = (WIDTH, HEIGHT)) -> "Image.Image | None":
    """
    Generate a hyper-realistic cinematic background with Stability AI (Stable Image Core).
    Requires STABILITY_API_KEY. Set STABILITY_BG=1 to enable (disabled by default
    to avoid unexpected spend).
    """
    api_key = os.environ.get("STABILITY_API_KEY", "")
    if not api_key:
        return None
    if os.environ.get("STABILITY_BG") != "1":
        return None
    try:
        aspect_ratio = "16:9" if size[0] >= size[1] else "9:16"
        full_prompt = (
            "Hyper-realistic cinematic photograph. No text, no watermarks, no logos. "
            f"{prompt[:800]}"
        )
        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/generate/core",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "image/*",
            },
            data={
                "prompt": full_prompt,
                "negative_prompt": "text, watermark, logo, ui, interface, cartoon, illustration, painting, blurry",
                "output_format": "jpeg",
                "aspect_ratio": aspect_ratio,
            },
            timeout=60,
        )
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = img.resize(size, Image.LANCZOS)
            log.info(f"Stability AI image generated for: {prompt[:60]}…")
            return img
        else:
            log.warning(f"Stability AI returned status {response.status_code}: {response.text[:200]}")
            return None
    except Exception as e:
        log.warning(f"Stability AI generation failed: {e}")
        return None


def _generate_image_replicate(prompt: str, size: tuple[int, int] = (WIDTH, HEIGHT)) -> "Image.Image | None":
    """
    Generate a hyper-realistic cinematic background via Replicate (Flux or SDXL).
    Requires REPLICATE_API_KEY. Set REPLICATE_BG=1 to enable (disabled by default).
    Cost: ~$0.003–0.008/image depending on model. Sits between Stability AI and DALL-E 3
    in the fallback chain.
    """
    if os.environ.get("REPLICATE_BG") != "1":
        return None
    api_key = os.environ.get("REPLICATE_API_KEY", "")
    if not api_key:
        return None
    try:
        aspect_ratio = "16:9" if size[0] >= size[1] else "9:16"
        safe_prompt = (
            "Hyper-realistic cinematic photograph, no text, no watermarks, no logos. "
            f"{prompt[:700]}"
        )
        # Use Flux Dev (high quality) — falls back to SDXL if unavailable
        model = "black-forest-labs/flux-dev"
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
            "Prefer": "wait",  # synchronous — block until the prediction is ready
        }
        payload = {
            "input": {
                "prompt": safe_prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": "jpg",
                "output_quality": 90,
                "negative_prompt": "text, watermark, logo, ui, cartoon, illustration, blurry, low quality",
            }
        }
        resp = requests.post(
            f"https://api.replicate.com/v1/models/{model}/predictions",
            headers=headers,
            json=payload,
            timeout=90,
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            # Synchronous "Prefer: wait" response returns output directly
            output = data.get("output")
            if isinstance(output, list) and output:
                output = output[0]
            if isinstance(output, str) and output.startswith("http"):
                img = Image.open(BytesIO(requests.get(output, timeout=30).content)).convert("RGB")
                log.info(f"Replicate Flux image generated for: {prompt[:60]}…")
                return img.resize(size, Image.LANCZOS)
        log.warning(f"Replicate returned status {resp.status_code}: {resp.text[:200]}")
        return None
    except Exception as e:
        log.warning(f"Replicate generation failed: {e}")
        return None


_TOPIC_PEXELS_MAP = {
    # ── Finance / Markets ────────────────────────────────────────────────────
    "crypto":       ["crypto trader dark office monitors", "bitcoin investor phone screen night", "blockchain technology person laptop"],
    "bitcoin":      ["bitcoin trader nervous screen glow", "cryptocurrency investor dramatic", "digital currency person laptop dark"],
    "stock":        ["stock market trader wall street", "investor watching charts dramatic", "financial analyst office screens"],
    "finance":      ["businessman counting money dramatic", "financial district person walking", "bank vault serious person"],
    "economy":      ["economist serious office window city", "recession empty street dramatic", "inflation grocery store worried shopper"],
    "money":        ["cash money dramatic lighting hands", "wealthy person luxury office window", "entrepreneur laptop coffee shop focused"],
    # ── Technology ──────────────────────────────────────────────────────────
    "tech":         ["software engineer laptop nighttime", "developer coding dark room neon", "technology startup person focused"],
    "ai":           ["artificial intelligence researcher lab", "tech person futuristic dark room", "data scientist multiple screens"],
    "robot":        ["robot industrial factory dramatic", "humanoid robot closeup dramatic", "automation manufacturing dramatic lighting"],
    "cybersecurity":["hacker dark room screens dramatic", "cybersecurity analyst monitoring screens", "digital lock security dramatic blue"],
    "social media": ["person scrolling phone intense", "influencer filming content dramatic", "viral content creator studio lighting"],
    # ── Health / Science ────────────────────────────────────────────────────
    "health":       ["athlete running dawn city street", "person exercising dramatic lighting", "doctor examining urgent focus"],
    "science":      ["scientist laboratory dramatic close-up", "researcher microscope focused dramatic", "lab experiment results dramatic reveal"],
    "space":        ["astronaut dramatic suit backlit", "observatory night sky person", "rocket launch control room"],
    "nature":       ["dramatic storm approaching landscape", "wildlife dramatic wildlife close-up", "environmental scientist field dramatic"],
    # ── Entertainment / Culture ─────────────────────────────────────────────
    "gaming":       ["gamer intense focus neon lights", "esports player competition arena", "streamer dramatic rgb lighting"],
    "music":        ["musician performing stage spotlight", "singer emotional performance dramatic", "producer studio headphones focus"],
    "sports":       ["athlete winning dramatic celebration", "sports competition intense close-up", "coach sideline dramatic tension"],
    "movies":       ["film director dramatic set lights", "movie production clapperboard dramatic", "cinema audience dramatic reaction"],
    # ── Society / Events ────────────────────────────────────────────────────
    "viral":        ["person shocked looking at phone", "crowd reaction surprise moment", "social media influencer dramatic"],
    "climate":      ["environmental activist dramatic sky", "nature destruction dramatic wide shot", "scientist field research urgent"],
    "politics":     ["politician speaking crowd dramatic", "protest march dramatic lighting", "government building ominous sky"],
    "crime":        ["detective dark office investigation", "urban street night dramatic", "courtroom serious dramatic lighting"],
    # ── Lifestyle ───────────────────────────────────────────────────────────
    "food":         ["chef cooking intense kitchen fire", "food market dramatic overhead", "restaurant dramatic plating moment"],
    "travel":       ["traveler dramatic landscape backlit", "adventure person cliff dramatic", "explorer remote location dramatic lighting"],
    "business":     ["ceo boardroom dramatic presentation", "startup team working late night", "entrepreneur pitch dramatic lighting"],
    "relationships":["couple emotional dramatic conversation", "friends dramatic outdoor adventure", "family moment emotional dramatic light"],
    "education":    ["student studying intense dramatic", "teacher classroom dramatic moment", "university campus dramatic architecture"],
}


def _topic_keyword(keyword: str, asset_prompt: str) -> "str | None":
    """
    Return a topic-optimized Pexels query string based on detected keywords,
    or None if no topic is matched.
    """
    import random
    combined = (keyword + " " + asset_prompt).lower()
    for key in _TOPIC_PEXELS_MAP:
        if key in combined:
            return random.choice(_TOPIC_PEXELS_MAP[key])
    return None


def _fetch_bg(
    keyword: str,
    asset_prompt: str = "",
    size: tuple[int, int] = (WIDTH, HEIGHT),
    use_video_frames: bool = True,
) -> "Image.Image | None":
    """
    Multi-source background fetcher with automatic fallback chain:
      1. Pexels photo  — topic-optimized query (_TOPIC_PEXELS_MAP match)
      2. Unsplash      — real photography, free (50 req/hr), requires UNSPLASH_ACCESS_KEY
      3. Pexels photo  — original keyword (3 random pages, 15 photos/page, deduped)
      4. Pexels video  — cinematic frame extracted from a video clip (5 MB range-cap)
      5. Stability AI  — hyper-realistic generation (opt-in: STABILITY_BG=1)
      6. Replicate     — Flux Dev generation (opt-in: REPLICATE_BG=1)
      7. Pollinations  — free FLUX-based AI generation (no key required)
      8. HF FLUX       — Hugging Face FLUX.1-schnell free tier (requires HF_TOKEN)
      9. DALL-E 3      — HD generation (opt-in: DALLE_BG=1)
    All URLs are persisted to output/seen_bg_urls.json — never re-used across ANY video.
    Returns None if all sources fail (caller should use _gradient_bg).
    """
    # 1. Try topic-optimized keyword first for higher visual relevance
    topic_kw = _topic_keyword(keyword, asset_prompt)
    if topic_kw:
        img = _fetch_pexels(topic_kw, size=size)
        if img:
            return img

    # 2. Unsplash — real photography, free tier
    arr = _fetch_unsplash(keyword, size=size)
    if arr is not None:
        return Image.fromarray(arr)

    # 3. Pexels photo — original keyword
    img = _fetch_pexels(keyword, size=size)
    if img:
        return img

    # 4. Pexels video frame
    if use_video_frames:
        img = _fetch_pexels_video_frame(keyword, size=size)
        if img:
            return img

    if asset_prompt:
        # 5. Stability AI
        img = _generate_image_stability(asset_prompt, size=size)
        if img:
            return img
        # 6. Replicate
        img = _generate_image_replicate(asset_prompt, size=size)
        if img:
            return img

    # 7. Pollinations.ai — completely free, no key required
    arr = _generate_image_pollinations(asset_prompt or keyword, size=size)
    if arr is not None:
        return Image.fromarray(arr)

    # 8. HF FLUX.1-schnell — free tier (requires HF_TOKEN)
    arr = _generate_image_hf_flux(asset_prompt or keyword, size=size)
    if arr is not None:
        return Image.fromarray(arr)

    if asset_prompt:
        # 9. DALL-E 3
        img = _generate_image_dalle(asset_prompt, size=size)
        if img:
            return img

    return None


_PALETTES = [
    # danger/hook: deep crimson dark
    ((28, 4,   4),  (72,  8,   8)),
    # curiosity/surprise: deep indigo-amber
    ((18, 14,  4),  (42, 28,   4)),
    # evidence/tension: deep navy-blue
    (( 4,  8,  28), (4,  22,  64)),
    # mechanism: deeper navy
    (( 4,  6,  24), (4,  16,  52)),
    # payoff/relief: deep warm amber
    ((28, 16,  4),  (64, 36,   4)),
    # action/end: deep forest green
    (( 4, 18,  6),  (4,  48,  12)),
    # neutral fallback deep
    ((12, 12, 18),  (28, 22,  44)),
    # secondary neutral
    ((16, 10, 24),  (36, 18,  52)),
]

def _gradient_bg(index: int = 0) -> Image.Image:
    c1, c2 = _PALETTES[index % len(_PALETTES)]
    arr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for ch in range(3):
        arr[:, :, ch] = np.linspace(c1[ch], c2[ch], HEIGHT, dtype=np.uint8)[:, None]
    return Image.fromarray(arr)


def _vignette(arr: np.ndarray, strength: float = 0.60) -> np.ndarray:
    """Darken edges for a cinematic look (fast numpy, no loop)."""
    h, w = arr.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X - w / 2) / (w / 2)) ** 2 + ((Y - h / 2) / (h / 2)) ** 2)
    mask = np.clip(1.0 - strength * dist ** 1.6, 0.0, 1.0)[:, :, np.newaxis]
    return (arr * mask).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Frame creators  (return numpy arrays — cheaper than PIL round-trips later)
# ─────────────────────────────────────────────────────────────────────────────

def _frame_intro_card(name: str, num: int, total: int) -> np.ndarray:
    """
    Section intro card — pure black BG, section name pops in.
    Mimics the punchy section-title cards used by MrBeast / educational creators.
    """
    img  = Image.new("RGB", (WIDTH, HEIGHT), (4, 4, 14))
    draw = ImageDraw.Draw(img)

    # Tiny section counter
    sm = _get_font(40)
    sm_txt = f"SECTION {num} OF {total}"
    sw = _text_w(draw, sm_txt, sm)
    draw.text(((WIDTH - sw) // 2, HEIGHT // 2 - 140), sm_txt, font=sm, fill=(90, 90, 110))

    # Big section name
    big = _get_font(100, bold=True)
    lines = textwrap.wrap(name.upper(), width=22)
    for i, line in enumerate(lines):
        lw = _text_w(draw, line, big)
        lx = (WIDTH - lw) // 2
        ly = HEIGHT // 2 - 40 + i * 116
        # Glow: faint wider version
        draw.text((lx - 2, ly - 2), line, font=big, fill=(255, 180, 0, 60))
        draw.text((lx + 2, ly + 2), line, font=big, fill=(0, 0, 0, 120))
        draw.text((lx, ly),         line, font=big, fill=(255, 214, 0))

    # Gold underline
    bar_y = HEIGHT // 2 + (len(lines) * 116) - 20
    draw.rectangle([(WIDTH // 2 - 220, bar_y), (WIDTH // 2 + 220, bar_y + 5)],
                   fill=(255, 214, 0))

    return np.array(img)


def _frame_callout_card(callout: str) -> np.ndarray:
    """
    Stat-reveal card — bold white text on deep-red gradient.
    Used to punch in key statistics / claims (MrBeast-style).
    """
    c1, c2 = (85, 12, 4), (30, 4, 2)
    arr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for ch in range(3):
        arr[:, :, ch] = np.linspace(c1[ch], c2[ch], HEIGHT, dtype=np.uint8)[:, None]
    img  = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    big   = _get_font(108, bold=True)
    lines = textwrap.wrap(callout.upper(), width=20)[:2]
    total_h = len(lines) * 125
    start_y = (HEIGHT - total_h) // 2 - 20

    for i, line in enumerate(lines):
        lw  = _text_w(draw, line, big)
        lx  = (WIDTH - lw) // 2
        ly  = start_y + i * 125
        draw.text((lx + 5, ly + 5), line, font=big, fill=(0, 0, 0, 160))
        draw.text((lx, ly),         line, font=big, fill=(255, 255, 255))

    # Thin gold bar above text
    draw.rectangle([(WIDTH // 2 - 300, start_y - 28),
                    (WIDTH // 2 + 300, start_y - 22)], fill=(255, 214, 0))

    return np.array(img)


def _frame_text_cut(bg: Image.Image, text_chunk: str, lower: str,
                    section_num: int, total: int) -> np.ndarray:
    """
    Caption-style text cut — the core visual unit.
    8-10 words, large centered font, translucent pill behind each line.
    Matches how Kurzgesagt / educational creators display on-screen text.
    """
    # Apply vignette to background
    bg_arr = _vignette(np.array(bg.convert("RGB")))
    slide  = Image.fromarray(bg_arr).convert("RGBA")

    # Medium dark overlay — background still visible
    ov    = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 148))
    slide = Image.alpha_composite(slide, ov)
    draw  = ImageDraw.Draw(slide)

    # Progress bar (thin, top)
    filled = int(WIDTH * section_num / max(total, 1))
    draw.rectangle([(0, 0), (WIDTH, 7)],      fill=(18, 18, 18))
    draw.rectangle([(0, 0), (filled, 7)],     fill=(255, 214, 0))

    # ── Main text (centered, large) ───────────────────────────────────────
    font  = _get_font(78, bold=True)
    lines = textwrap.wrap(text_chunk, width=30)[:3]

    total_h = len(lines) * 96
    base_y  = (HEIGHT - total_h) // 2 - 30

    for i, line in enumerate(lines):
        lw  = _text_w(draw, line, font)
        lx  = (WIDTH - lw) // 2
        ly  = base_y + i * 96

        # Translucent pill background per line
        pill_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        pd         = ImageDraw.Draw(pill_layer)
        pad_x, pad_y = 28, 10
        _rounded_rect(pd,
                      [(lx - pad_x, ly - pad_y),
                       (lx + lw + pad_x, ly + 80 + pad_y)],
                      radius=14,
                      fill=(0, 0, 0, 175))
        slide = Image.alpha_composite(slide, pill_layer)
        draw  = ImageDraw.Draw(slide)

        # Text shadow + text
        draw.text((lx + 3, ly + 3), line, font=font, fill=(0, 0, 0, 180))
        draw.text((lx, ly),         line, font=font, fill=(255, 255, 255))

    # ── Lower third ───────────────────────────────────────────────────────
    lt = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lt)
    ld.rectangle([(0, HEIGHT - 105), (WIDTH, HEIGHT)],       fill=(5, 5, 28, 238))
    ld.rectangle([(0, HEIGHT - 107), (WIDTH, HEIGHT - 103)], fill=(255, 214, 0, 255))
    slide = Image.alpha_composite(slide, lt)
    draw  = ImageDraw.Draw(slide)

    lf = _get_font(42, bold=True)
    draw.text((76, HEIGHT - 84), f"▶  {lower}", font=lf, fill=(255, 255, 255))

    return np.array(slide.convert("RGB"))


# ─────────────────────────────────────────────────────────────────────────────
# Animated clip builder
# ─────────────────────────────────────────────────────────────────────────────

def _animated_clip(frame_arr: np.ndarray, duration: float,
                   direction: int = 0, pop: bool = True):
    """
    Wraps a pre-rendered numpy frame into a VideoClip with:
      • Pop-in   — clip eases from 1.07× to 1.0× in the first POP_DUR seconds
      • Camera   — direction 0=zoom-in, 1=pan-right, 2=pan-left, 3=zoom-out
    Both effects work via crop+resize (no PIL text re-render per frame).
    """
    h, w = frame_arr.shape[:2]

    def make_frame(t: float) -> np.ndarray:
        progress = t / max(duration, 0.001)

        # Pop-in easing (ease-out quad: fast → slow)
        if pop and t < POP_DUR:
            pop_frac = t / POP_DUR
            extra    = 0.07 * (1 - pop_frac) ** 2   # 0.07 → 0
        else:
            extra = 0.0

        # Camera motion
        if direction == 0:      # zoom-in
            base_crop = 1.0 / (1.0 + 0.055 * progress)
        elif direction == 1:    # pan-right
            base_crop = 1.0 / 1.055
        elif direction == 2:    # pan-left
            base_crop = 1.0 / 1.055
        elif direction == 3:    # zoom-out
            base_crop = 1.0 / (1.055 - 0.050 * progress)
        else:
            base_crop = 1.0

        crop_frac = base_crop / (1.0 + extra)   # shrink further for pop-in
        ch = max(1, int(h * crop_frac))
        cw = max(1, int(w * crop_frac))

        # Panning offset
        if direction == 1:
            x0 = int((w - cw) * progress * 0.8)
            y0 = (h - ch) // 2
        elif direction == 2:
            x0 = int((w - cw) * (1 - progress * 0.8))
            y0 = (h - ch) // 2
        else:
            y0 = (h - ch) // 2
            x0 = (w - cw) // 2

        x0 = max(0, min(x0, w - cw))
        y0 = max(0, min(y0, h - ch))

        crop = frame_arr[y0: y0 + ch, x0: x0 + cw]
        return np.array(Image.fromarray(crop).resize((w, h), Image.BILINEAR))

    return _make_videoclip(make_frame, duration)


# ─────────────────────────────────────────────────────────────────────────────
# TTS  (with word-boundary timing for SRT captions)
# ─────────────────────────────────────────────────────────────────────────────

async def _edge_tts_timing_async(text: str, path: str, voice: str) -> list:
    """
    Stream edge-tts audio + WordBoundary events.
    Returns list of {"word", "start", "end"} in seconds.
    """
    import edge_tts
    comm  = edge_tts.Communicate(text, voice)
    words = []
    with open(path, "wb") as fh:
        async for event in comm.stream():
            if event["type"] == "audio":
                fh.write(event["data"])
            elif event["type"] == "WordBoundary":
                start = event["offset"]   / 10_000_000   # 100-ns units → seconds
                dur   = event["duration"] / 10_000_000
                words.append({"word": event["text"], "start": start, "end": start + dur})
    return words


def _openai_tts_with_timing(text: str, path: str, voice: str = "") -> tuple[bool, list]:
    """
    Use OpenAI TTS API (tts-1-hd) for studio-quality narration.
    Word timing is estimated proportionally from audio duration + character counts.
    Returns (success, word_timing_list).
    """
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        return False, []
    try:
        from openai import OpenAI as _OAIClient
        _OPENAI_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
        if not voice:
            voice = os.environ.get("OPENAI_TTS_VOICE", "nova")
        if voice not in _OPENAI_VOICES:
            voice = "nova"
        client = _OAIClient(api_key=openai_key)
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text,
            response_format="mp3",
        )
        with open(path, "wb") as f:
            f.write(response.content)
        # Get audio duration to estimate word timing
        try:
            from moviepy import AudioFileClip as _AFC
        except ImportError:
            from moviepy.editor import AudioFileClip as _AFC
        with _AFC(path) as ac:
            duration = ac.duration
        # Distribute word timing proportionally by character length + inter-word pause
        word_list = text.split()
        # +0.5 per word accounts for natural pause rhythm
        total_units = sum(len(w) + 0.5 for w in word_list)
        word_timing = []
        t = 0.0
        for word in word_list:
            frac = (len(word) + 0.5) / total_units
            dur = duration * frac
            word_timing.append({"word": word, "start": t, "end": t + dur})
            t += dur
        log.info(f"  OpenAI TTS: {duration:.1f}s audio, voice={voice}")
        return True, word_timing
    except Exception as e:
        log.warning(f"OpenAI TTS failed ({e}); falling back to edge-tts")
        return False, []


def _voice_for_section(section_index: int) -> str:
    """
    Return the edge-tts voice to use for a given section index.
    Reads SECTION_VOICES env var (comma-separated list of voices).
    Returns "" if not set (caller uses its own default).
    """
    raw = os.environ.get("SECTION_VOICES", "").strip()
    if not raw:
        return ""
    voices = [v.strip() for v in raw.split(",") if v.strip()]
    if not voices:
        return ""
    return voices[section_index % len(voices)]


def _tts_with_timing(text: str, path: str, voice: str = "") -> tuple:
    """
    Generate TTS audio and return (success, word_timing_list).
    Each timing entry: {"word": str, "start": float, "end": float}  (seconds).

    Priority: OpenAI TTS (studio quality) → edge-tts → gTTS
    """
    # ── OpenAI TTS (highest quality — requires OPENAI_API_KEY) ───────────
    ok, words = _openai_tts_with_timing(text, path, voice)
    if ok:
        return True, words

    if not voice:
        voice = os.environ.get("TTS_VOICE", "en-US-GuyNeural")

    # ── Try edge-tts (word boundaries give precise SRT timing) ───────────
    try:
        import edge_tts  # noqa: F401
        loop = asyncio.new_event_loop()
        try:
            words = loop.run_until_complete(
                _edge_tts_timing_async(text, path, voice)
            )
        finally:
            loop.close()
        return True, words
    except ImportError:
        pass
    except Exception as e:
        log.warning(f"edge-tts timing failed ({e}), trying gTTS…")

    # ── gTTS fallback — estimate timing at 2.5 words/second ──────────────
    try:
        from gtts import gTTS
        gTTS(text=text, lang="en", slow=False).save(path)
        wps   = 2.5
        items = text.split()
        t     = 0.0
        words = []
        for w in items:
            dur = 1.0 / wps
            words.append({"word": w, "start": t, "end": t + dur})
            t  += dur
        return True, words
    except Exception as e:
        log.error(f"TTS failed: {e}")
        return False, []


# ─────────────────────────────────────────────────────────────────────────────
# SRT caption helpers
# ─────────────────────────────────────────────────────────────────────────────

def _srt_timestamp(sec: float) -> str:
    """Convert seconds to SRT timestamp  HH:MM:SS,mmm"""
    h  = int(sec // 3600)
    m  = int((sec % 3600) // 60)
    s  = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    if ms >= 1000:          # guard for rounding edge
        s  += 1
        ms  = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt(all_words: list, srt_path: str, words_per_line: int = 8) -> None:
    """
    Group word-timing entries into SRT subtitle blocks and write the file.
    all_words: list of {"word", "start", "end"} (absolute seconds in final video).
    """
    if not all_words:
        return

    blocks = []
    i = 0
    while i < len(all_words):
        chunk = all_words[i: i + words_per_line]
        text  = " ".join(w["word"] for w in chunk)
        t0    = chunk[0]["start"]
        t1    = chunk[-1]["end"]
        # SRT requires end > start
        if t1 <= t0:
            t1 = t0 + 0.5
        blocks.append((t0, t1, text))
        i += words_per_line

    lines = []
    for idx, (t0, t1, text) in enumerate(blocks, 1):
        lines.append(str(idx))
        lines.append(f"{_srt_timestamp(t0)} --> {_srt_timestamp(t1)}")
        lines.append(text)
        lines.append("")

    Path(srt_path).write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  SRT captions → {srt_path}  ({len(blocks)} blocks)")


# ─────────────────────────────────────────────────────────────────────────────
# Background music
# ─────────────────────────────────────────────────────────────────────────────

def _add_background_music(video_clip, music_path: str, volume: float = 0.07):
    try:
        try:
            from moviepy.editor import AudioFileClip, CompositeAudioClip
            from moviepy.audio.AudioClip import AudioArrayClip
        except ImportError:
            from moviepy import AudioFileClip, CompositeAudioClip
            from moviepy.audio.AudioClip import AudioArrayClip

        music  = AudioFileClip(music_path)
        fps    = int(music.fps or 44100)
        arr    = music.to_soundarray(fps=fps)
        needed = int(video_clip.duration * fps)

        if arr.shape[0] < needed:
            arr = np.tile(arr, ((needed // arr.shape[0]) + 2, 1))
        arr = (arr[:needed] * volume).astype(np.float32)

        music_clip = AudioArrayClip(arr, fps=fps)
        orig       = video_clip.audio
        mixed      = CompositeAudioClip([orig, music_clip]) if orig else music_clip
        return _set_audio(video_clip, mixed)
    except Exception as e:
        log.warning(f"Background music skipped: {e}")
        return video_clip


# ─────────────────────────────────────────────────────────────────────────────
# Thumbnail
# ─────────────────────────────────────────────────────────────────────────────

def _generate_thumbnail(
    script: dict,
    bg: Image.Image,
    output_path: str,
    variant: int = 0,
) -> None:
    """
    Render a high-CTR thumbnail.

    variant=0  (A) — yellow title top, navy hook bar bottom (classic YouTube)
    variant=1  (B) — white title left-aligned, accent color left-edge bar,
                     darker overlay, concept text from thumbnail_concept_b if present
    """
    TW, TH = 1280, 720
    bg_arr  = _vignette(np.array(bg.resize((TW, TH), Image.LANCZOS).convert("RGB")), 0.5)
    thumb   = Image.fromarray(bg_arr).convert("RGBA")

    title  = script.get("video_title", "")
    hook   = script.get("hook", "")
    # Variant B uses thumbnail_concept_b as the overlay copy if available
    concept_b = script.get("thumbnail_concept_b", "")

    if variant == 0:
        # ── Variant A: yellow centered title + navy bottom bar ─────────────
        ov    = Image.new("RGBA", (TW, TH), (0, 0, 0, 145))
        thumb = Image.alpha_composite(thumb, ov)
        draw  = ImageDraw.Draw(thumb)

        tf     = _get_font(94, bold=True)
        tlines = textwrap.wrap(title, width=22)[:2]
        ty     = 48
        for line in tlines:
            lw = _text_w(draw, line, tf)
            lx = (TW - lw) // 2
            draw.text((lx + 5, ty + 5), line, font=tf, fill=(0,   0,   0))
            draw.text((lx, ty),         line, font=tf, fill=(255, 214, 0))
            ty += 108

        if hook:
            bl = Image.new("RGBA", (TW, TH), (0, 0, 0, 0))
            bd = ImageDraw.Draw(bl)
            bd.rectangle([(0, TH - 158), (TW, TH)],       fill=(5, 5, 26, 235))
            bd.rectangle([(0, TH - 160), (TW, TH - 156)], fill=(255, 214, 0, 255))
            thumb = Image.alpha_composite(thumb, bl)
            draw  = ImageDraw.Draw(thumb)
            hf     = _get_font(48)
            hshort = " ".join(hook.split()[:13]) + ("…" if len(hook.split()) > 13 else "")
            hlines = textwrap.wrap(hshort, width=38)[:2]
            for i, line in enumerate(hlines):
                lw = _text_w(draw, line, hf)
                lx = (TW - lw) // 2
                draw.text((lx, TH - 145 + i * 58), line, font=hf, fill=(255, 255, 255))

    else:
        # ── Variant B: white left-aligned title + accent left bar ──────────
        # Darker gradient overlay — heavier on left, lighter on right
        ov = Image.new("RGBA", (TW, TH), (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(ov)
        for x in range(TW):
            alpha = int(200 * (1.0 - (x / TW) * 0.55))
            ov_draw.line([(x, 0), (x, TH)], fill=(0, 0, 0, alpha))
        thumb = Image.alpha_composite(thumb, ov)

        # Bright red-orange accent bar on the left edge
        accent = Image.new("RGBA", (TW, TH), (0, 0, 0, 0))
        ad = ImageDraw.Draw(accent)
        ad.rectangle([(0, 0), (14, TH)], fill=(255, 45, 45, 255))
        thumb = Image.alpha_composite(thumb, accent)
        draw  = ImageDraw.Draw(thumb)

        # Title — white, left-aligned with padding
        tf     = _get_font(82, bold=True)
        px     = 54
        tlines = textwrap.wrap(concept_b or title, width=26)[:3]
        ty     = 60
        for line in tlines:
            draw.text((px + 4, ty + 4), line, font=tf, fill=(0, 0, 0))
            draw.text((px, ty),         line, font=tf, fill=(255, 255, 255))
            ty += 96

        # Hook text — smaller, accent yellow, lower left
        if hook:
            hf     = _get_font(44)
            hshort = " ".join(hook.split()[:10]) + ("…" if len(hook.split()) > 10 else "")
            draw.text((px + 2, TH - 90 + 2), hshort, font=hf, fill=(0, 0, 0))
            draw.text((px,     TH - 90),      hshort, font=hf, fill=(255, 214, 0))

    thumb.convert("RGB").save(output_path, "JPEG", quality=95)
    log.info(f"  Thumbnail{'B' if variant else 'A'} → {output_path}")


# ── Per-section energy system — drives pacing and animation variety ───────────

_STYLE_ENERGY: dict = {
    "cold_open":    0.92,   # Maximum energy — hook
    "human_action": 0.82,   # High energy — action
    "finale":       0.88,   # Very high — payoff rush
    "action_steps": 0.75,   # High — urgency
    "split_screen": 0.65,   # Medium-high — reveal
    "cinematic":    0.58,   # Medium — drama
    "timeline":     0.48,   # Medium — paced
    "documentary":  0.42,   # Medium-low — deliberate
    "blueprint":    0.38,   # Low — technical
    "evidence":     0.32,   # Lowest — analytical
}

# 5 distinct animation styles that cycle per (section_idx, cut_idx) hash
_ANIM_STYLES = ["slide_up", "scale_pop", "wipe_left", "typewriter", "cascade"]

def _section_energy(scene_style: str, section_idx: int, total_sections: int,
                    text_chunk: str = "") -> dict:
    """
    Returns pacing + animation params for one text cut.

    Keys: energy(0-1), words_per_cut(int), min_cut_dur(float),
          anim_style(str), dark_overlay(float), particle_count(int),
          flash_cut(bool), speed_lines(bool)
    """
    base = _STYLE_ENERGY.get(scene_style, 0.55)

    # Mid-video sections are slightly calmer (let the story breathe)
    pos  = section_idx / max(total_sections - 1, 1)  # 0..1
    pos_factor = 1.0 - 0.25 * (0.25 < pos < 0.75)
    energy = float(np.clip(base * pos_factor, 0.1, 1.0))

    # Deterministic animation style — varies per section+chunk hash
    hash_val  = abs(hash(f"{scene_style}_{section_idx}_{text_chunk[:8]}")) % len(_ANIM_STYLES)
    anim_style = _ANIM_STYLES[hash_val]

    return {
        "energy":        energy,
        "words_per_cut": max(3, int(10 - energy * 7)),       # 3–10 words
        "min_cut_dur":   max(0.85, 3.0 - energy * 2.15),     # 0.85–3.0 s
        "anim_style":    anim_style,
        "dark_overlay":  0.60 + (1.0 - energy) * 0.10,       # 0.60–0.70
        "particle_count": max(10, int(energy * 55)),          # 10–55 particles
        "flash_cut":     energy > 0.78,                       # flash on very high energy
        "speed_lines":   energy > 0.85,                       # speed lines on max energy
    }


def _speed_lines_overlay(w: int, h: int, energy: float, seed: int = 42) -> np.ndarray:
    """
    Radial speed lines from centre — manga-style impact effect for high-energy cuts.
    Returns RGBA (H, W, 4) numpy array.
    """
    from PIL import ImageDraw as _ID
    rng    = np.random.default_rng(seed)
    img    = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw   = _ID.Draw(img)
    cx, cy = w // 2, h // 2
    n_lines = int(energy * 30)
    for _ in range(n_lines):
        angle  = rng.uniform(0, math.tau)
        length = rng.uniform(0.25, 0.75) * max(w, h)
        width  = rng.integers(1, 3)
        alpha  = int(rng.uniform(0.04, 0.18) * 255)
        ex     = int(cx + length * math.cos(angle))
        ey     = int(cy + length * math.sin(angle))
        draw.line([(cx, cy), (ex, ey)], fill=(255, 255, 255, alpha), width=width)
    return np.array(img)


# ─────────────────────────────────────────────────────────────────────────────
# ① Kinetic-typography text-cut clip
# ─────────────────────────────────────────────────────────────────────────────

def _word_reveal_clip(
    bg_arr:       np.ndarray,
    text_chunk:   str,
    lower:        str,
    section_num:  int,
    total_sec:    int,
    duration:     float,
    direction:    int  = 0,
    pop:          bool = True,
    scene_style:   str  = "cinematic",
    callout:       str  = "",
    human_behavior: str = "",
    color_mood:    str = "curiosity",
    cta_type:      str  = "",
    retention_bridge: str = "",
    anim_style:    str  = "slide_up",
    energy:        float = 0.55,
):
    """
    Fully-animated caption-style text cut.

    Per frame:
      • Ken Burns motion on photo background  (pre-enlarged, pre-vignetted)
      • Pop-in ease (1.07× → 1.0× in first 0.2 s)
      • 28-particle drifting field
      • Scene-specific layout: human-action shots, documentary side cards, split-screen,
        evidence boards, timelines, blueprints, action checklists, or kinetic text
      • Animated gold progress bar
      • Lower-third bar
    """
    w, h = WIDTH, HEIGHT
    REVEAL_END  = 0.55      # all lines fully visible by this fraction of clip
    style       = _scene_style(scene_style)
    # Energy-responsive overlay: high energy = lighter (more vivid), low energy = darker (focus)
    DARK        = float(np.clip(0.60 + (1.0 - energy) * 0.10, 0.60, 0.70))
    accent      = _accent_for(color_mood)
    danger      = _RETENTION_COLORS["danger"]
    white       = (255, 255, 255)

    # ── Pre-computation (done once per clip, not per frame) ───────────────

    bg_big  = _get_enlarged_bg(bg_arr)     # (H*1.10, W*1.10, 3) uint8, cached
    big_h, big_w = bg_big.shape[:2]

    # Cinematic engine depth pre-computation (once per clip)
    _depth_arr: "np.ndarray | None" = None
    if _CINEMATIC_ENGINE:
        try:
            _depth_arr = _ce.precompute_depth(bg_arr)
        except Exception:
            _depth_arr = None

    # Pre-render lower-third RGBA
    lt = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lt)
    ld.rectangle([(0, h - 105), (w, h)],         fill=(5, 5, 28, 238))
    ld.rectangle([(0, h - 107), (w, h - 103)],   fill=(*accent, 255))
    lf = _get_font(42, bold=True)
    ld.text((76, h - 84), f"▶  {lower}", font=lf, fill=(255, 255, 255))
    lt_np = np.array(lt)

    # ── Pre-render CTA bump (subscribe / like button) — rendered once ─────
    _cta_np: "np.ndarray | None" = None
    if cta_type:
        _is_like   = cta_type == "like"
        _btn_color = (6, 95, 212) if _is_like else (255, 0, 0)
        _btn_label = "👍  LIKE" if _is_like else "🔔  SUBSCRIBE"
        _btn_font  = _get_font(32, bold=True)
        _btn_img   = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        _btn_draw  = ImageDraw.Draw(_btn_img)
        _bbox      = _btn_draw.textbbox((0, 0), _btn_label, font=_btn_font)
        _bw        = (_bbox[2] - _bbox[0]) + 56
        _bh        = 62
        _bx        = w - _bw - 80
        _by        = h - _bh - 130
        # Background pill
        _btn_draw.rounded_rectangle([(_bx - 12, _by - 10), (_bx + _bw + 4, _by + _bh + 8)], radius=36, fill=(0, 0, 0, 200))
        # Colored button body
        _btn_draw.rounded_rectangle([(_bx, _by), (_bx + _bw, _by + _bh)], radius=30, fill=(*_btn_color, 255))
        # Label text
        _tw = _bbox[2] - _bbox[0]
        _th = _bbox[3] - _bbox[1]
        _btn_draw.text((_bx + (_bw - _tw) // 2, _by + (_bh - _th) // 2), _btn_label, font=_btn_font, fill=(255, 255, 255, 255))
        _cta_np = np.array(_btn_img)

    # Pre-render retention bridge — full tease sentence, 2-line wrap, mixed case
    # Shown in the final 28% of the clip to create an open Zeigarnik loop.
    _rb_np:    "np.ndarray | None" = None
    _rb_lines: list[str]           = []
    if retention_bridge:
        # Strip template-style boilerplate markers if LLM leaked them
        rb_clean = (
            retention_bridge
            .replace("Bridge to", "")
            .replace("Tease the", "")
            .strip(" —:.")
        )
        # Sentence-case (not all-caps — tension words land harder in mixed case)
        if rb_clean:
            rb_clean = rb_clean[0].upper() + rb_clean[1:]
        # Wrap into at most 2 lines of ≤54 chars each
        rb_font = _get_font(26, bold=True)
        _rb_lines = textwrap.wrap(rb_clean, width=54)[:2]
        rb_h = 28 + len(_rb_lines) * 34   # padding + line height per line
        rb_img = Image.new("RGBA", (w, rb_h), (0, 0, 0, 0))
        rb_d   = ImageDraw.Draw(rb_img)
        rb_d.rounded_rectangle([(0, 0), (w, rb_h)], radius=0, fill=(0, 0, 0, 195))
        # Thin accent rule at top edge of panel
        rb_d.rectangle([(0, 0), (w, 3)], fill=(*accent, 255))
        # "▶  NEXT:" label in accent, then tease lines in white
        rb_d.text((20, 8), "▶  NEXT:", font=_get_font(22, bold=True), fill=(*accent, 255))
        for li, rb_line in enumerate(_rb_lines):
            rb_d.text((20, 30 + li * 34), rb_line, font=rb_font, fill=(255, 255, 255, 240))
        _rb_np = np.array(rb_img)

    # Pre-render each text line as tight RGBA numpy. Most styles intentionally
    # avoid the old centered-caption look by using side panels or visual systems.
    centered = style in ("cinematic", "cold_open", "finale")
    font_size = 86 if style == "cold_open" else (74 if centered else 58)
    font     = _get_font(font_size, bold=True)
    dummy_d  = ImageDraw.Draw(Image.new("RGB", (w, h)))
    if centered:
        raw_lines = _fit_text_lines(text_chunk.upper(), font, int(w * 0.76), 3)
    else:
        raw_lines = _fit_text_lines(text_chunk, font, 680, 4)
    PAD_X, PAD_Y = 32, 14
    LINE_H   = font_size + 24

    total_block_h = len(raw_lines) * LINE_H
    if style in ("human_action", "documentary", "timeline", "blueprint", "action_steps"):
        base_x = 88
        base_y = 260 if style == "human_action" else 225
    elif style == "split_screen":
        base_x = 1020
        base_y = 230
    elif style == "evidence":
        base_x = 760
        base_y = 310
    else:
        base_x = None
        base_y = (h - total_block_h) // 2 - 45

    line_data = []   # (rgba_np, paste_x, paste_y_final)
    for li, line_text in enumerate(raw_lines):
        lw  = _text_w(dummy_d, line_text, font)
        lx  = (w - lw) // 2 if base_x is None else base_x
        ly  = base_y + li * LINE_H

        pw, ph = lw + PAD_X * 2, LINE_H - 4
        pill   = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
        pd     = ImageDraw.Draw(pill)
        _rounded_rect(pd, [4, 4, pw - 1, ph - 1], radius=16, fill=(0, 0, 0, 115))
        _rounded_rect(pd, [0, 0, pw - 5, ph - 5], radius=16, fill=(8, 8, 30, 215))
        pd.text((PAD_X + 2, (ph - 86) // 2 + 2), line_text,
                font=font, fill=(0, 0, 0, 140))
        line_color = (*accent, 255) if li == 0 else (255, 255, 255, 255)
        pd.text((PAD_X,     (ph - 86) // 2),      line_text,
                font=font, fill=line_color)
        line_data.append((np.array(pill), lx - PAD_X, ly))

    pseed = (section_num * 31 + abs(hash(text_chunk[:12]))) % 99991
    micro_font = _get_font(32, bold=True)
    small_font = _get_font(42, bold=True)
    marker_font = _get_font(34, bold=True)
    callout_short = (callout or lower or "").upper()
    behavior_short = (human_behavior or "human behavior trigger").upper()[:42]

    def draw_scene_overlay(frame: np.ndarray, t: float, prog: float) -> None:
        img = Image.fromarray(frame).convert("RGBA")
        od = ImageDraw.Draw(img)

        if style == "human_action":
            # Action-movie language: handheld frame, subject tracking, speed lines,
            # freeze-frame reticle, and a behavior label that tells the viewer why
            # the moment matters psychologically.
            shake = int(math.sin(t * 18.0) * 8)
            od.rectangle([(0, 0), (w, 86)], fill=(0, 0, 0, 185))
            od.rectangle([(0, h - 122), (w, h)], fill=(0, 0, 0, 190))
            od.text((86 + shake, 28), "BEHAVIOR LOCK", font=micro_font, fill=(*accent, 255))
            od.text((430 + shake, 28), behavior_short, font=micro_font, fill=(255, 255, 255, 230))

            cx = int(w * (0.68 + 0.018 * math.sin(t * 1.7)))
            cy = int(h * (0.48 + 0.014 * math.cos(t * 1.3)))
            lock_p = _ease_out_quad(min(1.0, prog / 0.35))
            box_w = int(360 + 24 * math.sin(t * 4.0))
            box_h = int(430 + 18 * math.cos(t * 3.1))
            x1, y1 = cx - box_w // 2, cy - box_h // 2
            x2, y2 = cx + box_w // 2, cy + box_h // 2
            corner = int(82 * lock_p)
            col = (*accent, 235)
            for sx, sy, ex, ey in [
                (x1, y1, x1 + corner, y1), (x1, y1, x1, y1 + corner),
                (x2, y1, x2 - corner, y1), (x2, y1, x2, y1 + corner),
                (x1, y2, x1 + corner, y2), (x1, y2, x1, y2 - corner),
                (x2, y2, x2 - corner, y2), (x2, y2, x2, y2 - corner),
            ]:
                od.line([(sx, sy), (ex, ey)], fill=col, width=7)

            # Human subject silhouette/energy cue, useful even when stock photo is abstract.
            od.ellipse([(cx - 48, cy - 154), (cx + 48, cy - 58)], fill=(255, 255, 255, 42))
            od.rounded_rectangle([(cx - 80, cy - 48), (cx + 80, cy + 148)], radius=42, fill=(255, 255, 255, 34))
            for i in range(7):
                y = 170 + i * 86
                x0 = int(1220 + i * 38 - prog * 420)
                od.line([(x0, y), (x0 + 280, y - 54)], fill=(255, 255, 255, 35), width=5)

            pulse = int(28 + 18 * math.sin(t * 8))
            od.ellipse([(cx - pulse, cy - pulse), (cx + pulse, cy + pulse)], outline=(*danger, 210), width=5)
            od.rounded_rectangle([(88, 138), (650, 214)], radius=12, fill=(*danger, 220))
            label = callout_short[:30] or "ACTION MOMENT"
            od.text((118, 158), label, font=micro_font, fill=(*white, 255))

        elif style == "split_screen":
            split_x = w // 2
            od.rectangle([(0, 0), (split_x - 5, h)], fill=(0, 0, 0, 48))
            od.rectangle([(split_x + 5, 0), (w, h)], fill=(0, 0, 0, 96))
            od.rectangle([(split_x - 5, 0), (split_x + 5, h)], fill=(*accent, 210))
            od.text((90, 126), "WHAT PEOPLE THINK", font=small_font, fill=(*white, 235))
            od.text((1030, 126), "WHAT IS REALLY HAPPENING", font=small_font, fill=(*accent, 245))
            cross_p = _ease_out_quad(min(1.0, prog / 0.35))
            od.line([(125, 210), (125 + int(440 * cross_p), 345)], fill=(*danger, 230), width=13)
            od.line([(125, 345), (125 + int(440 * cross_p), 210)], fill=(*danger, 230), width=13)

        elif style == "evidence":
            card_w, card_h = 1040, 480
            x, y = (w - card_w) // 2, 206
            od.rounded_rectangle([(x, y), (x + card_w, y + card_h)], radius=18, fill=(246, 242, 226, 238))
            od.rectangle([(x, y), (x + card_w, y + 70)], fill=(18, 18, 24, 245))
            od.text((x + 36, y + 18), "EVIDENCE BOARD", font=micro_font, fill=(*accent, 255))
            for i in range(4):
                px = x + 80 + i * 235
                py = y + 132 + (i % 2) * 92
                r = 28 + int(8 * math.sin(t * 4 + i))
                od.ellipse([(px - r, py - r), (px + r, py + r)], fill=(*accent, 225))
                od.line([(px, py), (x + card_w // 2, y + card_h - 84)], fill=(50, 55, 68, 180), width=4)
            od.rounded_rectangle([(x + 330, y + card_h - 152), (x + 710, y + card_h - 44)], radius=14, fill=(150, 14, 10, 235))
            label = callout_short[:26]
            lw = _text_w(od, label, small_font)
            od.text((x + 520 - lw // 2, y + card_h - 126), label, font=small_font, fill=(255, 255, 255, 255))

        elif style == "timeline":
            y = 665
            od.line([(210, y), (w - 210, y)], fill=(*accent, 230), width=8)
            fill_x = 210 + int((w - 420) * _ease_out_quad(prog))
            od.line([(210, y), (fill_x, y)], fill=(*white, 255), width=13)
            labels = ["NOW", "CAUSE", "TURN", "PAYOFF"]
            for i, label in enumerate(labels):
                x = 210 + i * ((w - 420) // 3)
                scale = 1.0 + (0.18 if fill_x >= x else 0.0)
                r = int(23 * scale)
                od.ellipse([(x - r, y - r), (x + r, y + r)], fill=(12, 18, 34, 250), outline=(*accent, 255), width=5)
                od.text((x - 42, y + 42), label, font=marker_font, fill=(*white, 240))

        elif style == "blueprint":
            grid = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            gd = ImageDraw.Draw(grid)
            for gx in range(0, w, 80):
                gd.line([(gx, 0), (gx, h)], fill=(50, 120, 180, 35), width=1)
            for gy in range(0, h, 80):
                gd.line([(0, gy), (w, gy)], fill=(50, 120, 180, 35), width=1)
            img.alpha_composite(grid)
            od = ImageDraw.Draw(img)
            boxes = [(920, 220, "INPUT"), (1260, 410, "SYSTEM"), (920, 600, "RESULT")]
            for bx, by, label in boxes:
                od.rounded_rectangle([(bx, by), (bx + 250, by + 88)], radius=12, fill=(3, 20, 42, 230), outline=(120, 205, 255, 220), width=3)
                od.text((bx + 52, by + 25), label, font=micro_font, fill=(170, 225, 255, 255))
            arrow_p = _ease_out_quad(prog)
            od.line([(1175, 265), (1260 + int(210 * arrow_p), 454)], fill=(*accent, 230), width=7)
            od.line([(1260, 498), (1175 - int(230 * arrow_p), 640)], fill=(*accent, 230), width=7)

        elif style == "action_steps":
            panel_x, panel_y = 1010, 175
            od.rounded_rectangle([(panel_x, panel_y), (1780, 730)], radius=22, fill=(7, 12, 28, 222))
            od.text((panel_x + 42, panel_y + 32), "ACTION PLAN", font=small_font, fill=(*accent, 255))
            for i in range(4):
                p = min(1.0, max(0.0, (prog - i * 0.13) / 0.18))
                y = panel_y + 122 + i * 92
                od.rounded_rectangle([(panel_x + 44, y), (panel_x + 92, y + 48)], radius=8, fill=(*accent, int(255 * p)))
                if p > 0.6:
                    od.line([(panel_x + 56, y + 25), (panel_x + 70, y + 38), (panel_x + 88, y + 12)], fill=(5, 10, 24, 255), width=6)
                od.line([(panel_x + 116, y + 24), (panel_x + 655, y + 24)], fill=(255, 255, 255, int(110 * p)), width=9)

        elif style in ("documentary", "cold_open", "finale"):
            px = 80 if style != "cold_open" else 1150
            py = 130
            od.rounded_rectangle([(px, py), (px + 520, py + 82)], radius=12, fill=(0, 0, 0, 165))
            label = "COLD OPEN" if style == "cold_open" else ("PAYOFF" if style == "finale" else "REAL-WORLD VIEW")
            od.text((px + 28, py + 23), label, font=micro_font, fill=(*accent, 250))
            sweep_x = int((w + 260) * prog) - 260
            od.polygon([(sweep_x, 0), (sweep_x + 140, 0), (sweep_x - 120, h), (sweep_x - 260, h)], fill=(255, 255, 255, 20))

        frame[:] = np.array(img.convert("RGB"))

    def _ken_burns_frame(t: float, prog: float) -> np.ndarray:
        """Classic Ken Burns crop from the pre-enlarged, pre-vignetted bg_big."""
        if direction == 0:           # zoom-in
            scale = 1.0 + 0.06 * (1.0 - prog)
        elif direction == 3:         # zoom-out
            scale = 1.0 + 0.06 * prog
        else:
            scale = 1.055

        cw = int(w * scale);  ch = int(h * scale)

        if direction == 1:           # pan-right
            x0 = int((big_w - cw) * prog * 0.80)
            y0 = (big_h - ch) // 2
        elif direction == 2:         # pan-left
            x0 = int((big_w - cw) * (1.0 - prog * 0.80))
            y0 = (big_h - ch) // 2
        else:
            x0, y0 = (big_w - cw) // 2, (big_h - ch) // 2

        x0 = max(0, min(x0, big_w - cw))
        y0 = max(0, min(y0, big_h - ch))

        crop = bg_big[y0: y0 + ch, x0: x0 + cw]
        return np.array(Image.fromarray(crop).resize((w, h), Image.BILINEAR))

    def make_frame(t: float) -> np.ndarray:
        prog = max(0.0, min(1.0, t / max(duration, 0.001)))

        # ── Background: procedural 3D / cinematic engine / Ken Burns ─────
        if _CINEMATIC_ENGINE and _USE_3D_SCENES:
            # Fully procedural GPU-feel 3D environment (no Pexels image)
            try:
                frame = _ce.generate_procedural_frame(
                    w, h, t, duration,
                    scene_style, color_mood,
                    seed=pseed, fps=FPS,
                )
            except Exception:
                frame = _ken_burns_frame(t, prog)
        elif _CINEMATIC_ENGINE and _depth_arr is not None:
            # 2.5D parallax + cinematic effects on Pexels background
            try:
                frame = _ce.render_bg_frame(
                    bg_arr, _depth_arr, t, duration,
                    scene_style, color_mood, fps=FPS,
                )
            except Exception:
                frame = _ken_burns_frame(t, prog)
        else:
            frame = _ken_burns_frame(t, prog)

        # ── Pop-in ────────────────────────────────────────────────────────
        if pop and t < POP_DUR:
            sp = 1.11 - 0.11 * _ease_out_quad(t / POP_DUR)
            if sp > 1.001:
                sw2, sh2 = int(w * sp), int(h * sp)
                big2 = np.array(
                    Image.fromarray(frame).resize((sw2, sh2), Image.BILINEAR)
                )
                ox2 = (sw2 - w) // 2;  oy2 = (sh2 - h) // 2
                frame = big2[oy2: oy2 + h, ox2: ox2 + w]

        # ── Flash-cut: brief white flash at t=0 for high-energy cuts ─────
        if energy > 0.78 and t < 0.07:
            flash_alpha = max(0.0, 1.0 - t / 0.07) * 0.55
            frame = np.clip(
                frame.astype(np.float32) * (1.0 - flash_alpha)
                + np.array([255, 255, 255], dtype=np.float32) * flash_alpha,
                0, 255
            ).astype(np.uint8)

        # ── Auto-brightness + cinematic overlay ──────────────────────────
        _fm = frame.mean()
        if _fm < 80.0:
            frame = np.clip(frame.astype(np.float32) * min(2.5, 100.0 / max(_fm, 1.0)), 0, 255).astype(np.uint8)
        frame = (frame.astype(np.float32) * DARK).astype(np.uint8)

        draw_scene_overlay(frame, t, prog)

        # ── Particle field ────────────────────────────────────────────────
        particle_count = 42 if style in ("cold_open", "human_action") else (18 if style in ("documentary", "blueprint") else 26)
        parts = _particles_layer(w, h, t, count=particle_count, seed=pseed)
        _paste_np(frame, parts, 0, 0)

        # ── Speed lines: radial impact lines for max-energy hooks ─────────
        if energy > 0.85 and t < duration * 0.35:
            sl_alpha = max(0.0, 1.0 - t / (duration * 0.35)) * 0.6
            sl = _speed_lines_overlay(w, h, energy, seed=pseed + int(t * 100))
            _paste_np(frame, sl, 0, 0, alpha_factor=sl_alpha)

        # ── Kinetic text — style varies by anim_style param ──────────────
        n = len(line_data)
        for li, (pill_np, px, py_final) in enumerate(line_data):
            delay   = (li / max(n, 1)) * REVEAL_END * duration * 0.65
            local_p = min(1.0, max(0.0, t - delay) / max(duration * 0.22, 0.12))
            eased   = _ease_out_quad(local_p)

            if anim_style == "scale_pop":
                # Spring scale-in from 70% to 100% with overshoot bounce
                sc = _ease_out_back(local_p)
                if sc > 0.02:
                    ph, pw = pill_np.shape[:2]
                    nw = max(1, int(pw * sc)); nh = max(1, int(ph * sc))
                    scaled = np.array(Image.fromarray(pill_np).resize((nw, nh), Image.BILINEAR))
                    ox = px + (pw - nw) // 2
                    oy = py_final + (ph - nh) // 2
                    _paste_np(frame, scaled, ox, oy, alpha_factor=min(1.0, eased * 1.4))

            elif anim_style == "wipe_left":
                # Left-to-right reveal (wipe mask)
                reveal_frac = _ease_out_quad(min(1.0, local_p * 1.6))
                clip_w = max(0, min(pill_np.shape[1], int(reveal_frac * pill_np.shape[1])))
                if clip_w > 0:
                    _paste_np(frame, pill_np[:, :clip_w], px, py_final,
                              alpha_factor=min(1.0, eased * 1.2))

            elif anim_style == "typewriter":
                # Character-by-character reveal (PIL text drawn progressively)
                # Fall back to slide_up since pills are pre-rendered images
                # Simulate typewriter with horizontal clip expanding right
                reveal_frac = min(1.0, local_p * 2.0)
                clip_w = max(0, min(pill_np.shape[1], int(reveal_frac * pill_np.shape[1])))
                if clip_w > 0:
                    _paste_np(frame, pill_np[:, :clip_w], px, py_final, alpha_factor=1.0)

            elif anim_style == "cascade":
                # Each line drops from above with staggered bounce
                y_off = int((1.0 - _ease_out_back(local_p)) * 80)
                _paste_np(frame, pill_np, px, py_final + y_off, alpha_factor=eased)

            else:  # slide_up (default)
                y_off = int((1.0 - eased) * 42)
                _paste_np(frame, pill_np, px, py_final + y_off, alpha_factor=eased)

        # ── Lower third (slides up from bottom over first 0.30 s) ─────────
        lt_entry = _ease_out_quad(min(1.0, t / 0.30))
        lt_y_off = int((1.0 - lt_entry) * 110)
        _paste_np(frame, lt_np, 0, lt_y_off)

        # ── Retention bridge: fades in over the last 28% of clip ─────────
        # Positioned just above the lower-third bar (105px) + progress bar (6px)
        if _rb_np is not None and prog > 0.72:
            rb_alpha  = min(1.0, (prog - 0.72) / 0.10)
            _rb_panel_h = _rb_np.shape[0]
            _rb_y     = h - 105 - 6 - _rb_panel_h   # sit flush above lower-third
            _paste_np(frame, _rb_np, 0, max(0, _rb_y), alpha_factor=rb_alpha)

        # ── Animated progress bar (gold fill) ────────────────────────────
        bar_frac = (section_num - 1 + prog) / max(total_sec, 1)
        fill_w   = max(0, int(w * min(bar_frac, 1.0)))
        frame[h - 6:, :]        = [18, 18, 18]
        if fill_w:
            frame[h - 6:, :fill_w] = accent

        return frame

    return _make_videoclip(make_frame, duration)


# ─────────────────────────────────────────────────────────────────────────────
# ② Animated callout / stat-reveal card
# ─────────────────────────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(
    r'(\d[\d,]*\.?\d*)\s*(%|[KkMmBb](?:illion|illion)?)?'
)


def _callout_clip(callout: str, duration: float, color_mood: str = "danger"):
    """
    Animated stat-reveal card:
      • Deep-red gradient background pulses in brightness (sin wave)
      • Any numbers in the text count up from 0 to their target value
      • Text typewriters in character-by-character over the first 50%
      • 40 golden particles drift upward
      • Animated expanding gold bar above the text
    """
    w, h = WIDTH, HEIGHT
    REVEAL_END = 0.52     # text fully revealed by this fraction of clip

    # ── Parse numbers ────────────────────────────────────────────────────
    numbers = []
    for m in _NUMBER_RE.finditer(callout):
        raw = m.group(1).replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        suf = (m.group(2) or "").strip().upper()
        numbers.append({"full": m.group(0), "val": val, "suf": suf,
                        "start": m.start(), "end": m.end()})

    # ── Static gradient ───────────────────────────────────────────────────
    _CALLOUT_GRADIENTS = {
        "danger":    ((92, 14,  4), (36,  6,  2)),
        "curiosity": ((68, 52,  4), (22, 16,  2)),
        "evidence":  (( 4, 42, 82), ( 2, 14, 40)),
        "mechanism": (( 4, 34, 82), ( 2, 10, 46)),
        "action":    (( 4, 62, 22), ( 2, 22,  8)),
        "payoff":    (( 6, 52, 26), ( 2, 20, 10)),
    }
    c1, c2 = _CALLOUT_GRADIENTS.get(color_mood, _CALLOUT_GRADIENTS["danger"])
    base_arr = np.zeros((h, w, 3), dtype=np.uint8)
    for ch in range(3):
        base_arr[:, :, ch] = np.linspace(
            c1[ch], c2[ch], h, dtype=np.uint8
        )[:, None]

    font_big = _get_font(108, bold=True)
    lines_raw = textwrap.wrap(callout.upper(), width=22)[:2]
    total_chars = max(1, sum(len(l) for l in lines_raw))
    bar_accent = _accent_for(color_mood)

    def make_frame(t: float) -> np.ndarray:
        prog = t / max(duration, 0.001)

        # Pulsing brightness
        pulse  = 1.0 + 0.065 * math.sin(t * 3.6)
        frame  = np.clip(
            base_arr.astype(np.float32) * pulse, 0, 255
        ).astype(np.uint8)

        # Moving horizontal scan-line for energy
        scan_y = int((t * 72) % h)
        if 0 <= scan_y < h - 5:
            frame[scan_y: scan_y + 5] = np.clip(
                frame[scan_y: scan_y + 5].astype(np.int16) + 38, 0, 255
            ).astype(np.uint8)

        # Particles (denser, golden)
        parts = _particles_layer(w, h, t, count=40, seed=888)
        _paste_np(frame, parts, 0, 0)

        img  = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        total_h = len(lines_raw) * 130
        start_y = (h - total_h) // 2 - 28

        # Expanding gold bar above text
        bar_prog = min(1.0, prog / 0.28)
        bar_hw   = int(_ease_out_quad(bar_prog) * 320)
        if bar_hw > 0:
            draw.rectangle(
                [(w // 2 - bar_hw, start_y - 30),
                 (w // 2 + bar_hw, start_y - 23)],
                fill=(*bar_accent, 255)
            )

        # Typewriter reveal with number counter
        chars_visible = int(total_chars * min(1.0, prog / REVEAL_END))
        chars_done    = 0

        for li, line_text in enumerate(lines_raw):
            lw  = _text_w(draw, line_text, font_big)
            lx  = (w - lw) // 2
            ly  = start_y + li * 130

            visible = line_text[:max(0, chars_visible - chars_done)]
            chars_done += len(line_text)

            if not visible:
                continue

            # Animate numbers inside the visible portion
            display = visible
            for ni in numbers:
                frag = ni["full"].upper()
                if frag in display:
                    cur  = ni["val"] * min(1.0, prog / 0.65)
                    suf  = ni["suf"]
                    if suf == "%":
                        cur_s = f"{cur:.0f}%"
                    elif suf in ("K",):
                        cur_s = f"{cur:.0f}K"
                    elif suf in ("M", "MILLION"):
                        cur_s = f"{cur:.1f}M"
                    elif suf in ("B", "BILLION"):
                        cur_s = f"{cur:.1f}B"
                    else:
                        cur_s = f"{cur:,.0f}" if cur >= 1000 else f"{cur:.0f}"
                    display = display.replace(frag, cur_s)

            # Re-center if text width changed
            disp_w = _text_w(draw, display, font_big)
            dlx    = (w - disp_w) // 2
            draw.text((dlx + 5, ly + 5), display, font=font_big, fill=(0, 0, 0, 170))
            draw.text((dlx,     ly),     display, font=font_big, fill=(255, 255, 255))

        return np.array(img.convert("RGB"))

    return _make_videoclip(make_frame, duration)


# ─────────────────────────────────────────────────────────────────────────────
# ③-a Full-screen pull-quote card (peak sentence visual punctuation)
# ─────────────────────────────────────────────────────────────────────────────

# Emotion → (bg_top RGB, bg_bottom RGB, accent RGB, vignette_tint RGB)
_PULL_QUOTE_PALETTE = {
    "shock":              ((10,  8, 28), (28, 8, 8),   (255, 60,  60), (255, 0,  0)),
    "dread":              (( 6,  4, 20), (20, 6, 4),   (220, 80,  20), (180, 30, 0)),
    "recognition":        ((10, 18, 30), (18,14, 8),   (255,200,  60), (200,140, 0)),
    "cognitive_dissonance":(( 8, 6, 30), (30, 6, 8),   (120, 80, 255), (80,  0, 200)),
    "confusion":          ((12, 8, 28), (28,10, 4),    (255,160,  20), (200,100, 0)),
    "analytical_trust":   (( 4,12, 36), ( 4,22,18),    ( 60,200, 255), ( 0, 120,200)),
    "intellectual_awe":   (( 6,10, 36), ( 4,18,22),    ( 80,220, 255), ( 0, 160,220)),
    "anxiety":            ((28,  6, 6), (10,  4, 4),   (255, 30,  30), (200,  0,  0)),
    "relief":             (( 4,22, 10), ( 4,14, 6),    (120,255, 160), ( 0, 180, 60)),
    "belonging":          (( 8,14, 28), ( 6,18, 8),    (255,200,  80), (180,140,  0)),
}
_PQ_DEFAULT_PALETTE = ((8, 8, 24), (24, 8, 8), (220, 180, 60), (160, 120, 0))


def _pull_quote_clip(
    sentence:   str,
    emotion:    str   = "shock",
    bg_arr:     "np.ndarray | None" = None,
    duration:   float = 2.5,
) -> "VideoClip":
    """
    Full-screen 2.5s cinematic pull-quote card.

    Layout:
      • Deep gradient background (emotion-keyed color)
      • Blurred / darkened photo background if bg_arr provided
      • Thin accent rule above and below the quote text
      • Quote text centered, sentence-case, up to 3 lines, 68px bold
      • Subtle particle drift at low opacity
      • Left-edge vertical accent bar (emotion color, full height)
      • Hard letterbox bars top/bottom (8% each) — cinematic feel
      • Fade in first 20% / fade out last 15%
    """
    w, h = WIDTH, HEIGHT
    top_rgb, bot_rgb, accent_rgb, vignette_rgb = _PULL_QUOTE_PALETTE.get(
        emotion, _PQ_DEFAULT_PALETTE
    )

    # ── Static gradient base ─────────────────────────────────────────────────
    grad_base = np.zeros((h, w, 3), dtype=np.uint8)
    for ch in range(3):
        grad_base[:, :, ch] = np.linspace(
            top_rgb[ch], bot_rgb[ch], h, dtype=np.uint8
        )[:, None]

    # ── Optional photo bg — blurred dark overlay ─────────────────────────────
    _photo_bg: "np.ndarray | None" = None
    if bg_arr is not None:
        try:
            photo = Image.fromarray(bg_arr).resize((w, h)).convert("RGB")
            photo = photo.filter(ImageFilter.GaussianBlur(radius=18))
            ph_np = np.array(photo).astype(np.float32)
            # Blend toward gradient (80% gradient, 20% photo) then darken
            _photo_bg = np.clip(
                ph_np * 0.20 + grad_base.astype(np.float32) * 0.80, 0, 255
            ).astype(np.uint8)
        except Exception:
            _photo_bg = None

    base_layer = _photo_bg if _photo_bg is not None else grad_base

    # ── Letterbox bars ────────────────────────────────────────────────────────
    LB_H = int(h * 0.08)
    lb_bar = np.zeros((LB_H, w, 3), dtype=np.uint8)

    # ── Left accent bar (2px wide, full height, emotion color) ───────────────
    left_bar_np = np.zeros((h, 6, 3), dtype=np.uint8)
    left_bar_np[:] = accent_rgb

    # ── Text wrapping ─────────────────────────────────────────────────────────
    font = _get_font(68, bold=True)
    dummy_d = ImageDraw.Draw(Image.new("RGB", (w, h)))
    quote_text = sentence.strip()
    if not quote_text:
        quote_text = "★"
    # Sentence-case (don't shout pull-quotes)
    quote_text = quote_text[0].upper() + quote_text[1:] if len(quote_text) > 1 else quote_text.upper()
    lines = _fit_text_lines(quote_text, font, int(w * 0.78), 3)

    LINE_H     = 68 + 22
    total_h    = len(lines) * LINE_H
    text_top_y = (h - total_h) // 2
    RULE_GAP   = 18
    RULE_W     = int(w * 0.72)
    RULE_X     = (w - RULE_W) // 2

    def make_frame(t: float) -> np.ndarray:
        prog = t / max(duration, 0.001)

        # Fade envelope
        if prog < 0.20:
            alpha = prog / 0.20
        elif prog > 0.85:
            alpha = (1.0 - prog) / 0.15
        else:
            alpha = 1.0
        alpha = float(np.clip(alpha, 0.0, 1.0))

        # Very subtle pulse on accent
        pulse_scale = 1.0 + 0.035 * math.sin(t * 4.2)

        frame = base_layer.copy()

        # Letterbox
        frame[:LB_H, :]  = lb_bar
        frame[-LB_H:, :] = lb_bar

        # Sparse particles (low opacity for cinematic subtlety)
        parts = _particles_layer(w, h, t, count=16, seed=777)
        # Blend particles at 35% opacity
        mask = parts[:, :, 3:4].astype(np.float32) / 255.0 * 0.35
        frame_f = frame.astype(np.float32)
        frame_f += parts[:, :, :3].astype(np.float32) * mask
        frame = np.clip(frame_f, 0, 255).astype(np.uint8)

        img  = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        # Accent rules above and below text block
        rule_alpha = int(220 * alpha)
        rule_y_top = text_top_y - RULE_GAP - 3
        rule_y_bot = text_top_y + total_h + RULE_GAP
        ar = (*accent_rgb, rule_alpha)
        if rule_y_top > LB_H:
            draw.rectangle(
                [(RULE_X, rule_y_top), (RULE_X + RULE_W, rule_y_top + 3)],
                fill=ar,
            )
        if rule_y_bot + 3 < h - LB_H:
            draw.rectangle(
                [(RULE_X, rule_y_bot), (RULE_X + RULE_W, rule_y_bot + 3)],
                fill=ar,
            )

        # Text lines — fade in with alpha
        text_alpha = int(255 * alpha)
        for li, line in enumerate(lines):
            lw = _text_w(draw, line, font)
            lx = (w - lw) // 2
            ly = text_top_y + li * LINE_H
            # Shadow
            draw.text((lx + 4, ly + 4), line, font=font,
                      fill=(0, 0, 0, int(160 * alpha)))
            # Main text
            draw.text((lx, ly), line, font=font,
                      fill=(255, 255, 255, text_alpha))

        frame_out = np.array(img.convert("RGB"))

        # Left accent bar
        bar_h_px = int(h * min(1.0, prog / 0.30) * pulse_scale)
        bar_h_px = min(bar_h_px, h)
        if bar_h_px > 0:
            frame_out[:bar_h_px, :6] = left_bar_np[:bar_h_px]

        return frame_out

    return _make_videoclip(make_frame, duration)


# ─────────────────────────────────────────────────────────────────────────────
# ③ Spring-animated section intro card
# ─────────────────────────────────────────────────────────────────────────────

def _intro_clip(name: str, num: int, total: int, duration: float,
                bg_np: "np.ndarray | None" = None):
    """
    Section intro card with spring physics:
      • Section name zooms from 5% to 100% with ease-out-back (overshoot spring)
      • Two horizontal sweep lines expand outward from center
      • Gold underline grows left-to-right
      • Background: uses actual section image at 55% brightness (not pure black)
    """
    w, h = WIDTH, HEIGHT
    big_font = _get_font(100, bold=True)
    sm_font  = _get_font(40)
    lines    = textwrap.wrap(name.upper(), width=22)

    # Pre-render each line as RGBA image for smooth zoom
    dummy_d = ImageDraw.Draw(Image.new("RGB", (w, h)))
    line_imgs = []   # (rgba_np, center_x, center_y)
    for i, line in enumerate(lines):
        lw  = _text_w(dummy_d, line, big_font)
        cx  = w // 2
        cy  = h // 2 - 30 + i * 116 + 55

        extra = 48
        limg  = Image.new("RGBA", (lw + extra * 2, 138), (0, 0, 0, 0))
        ld    = ImageDraw.Draw(limg)
        # Glow halo
        ld.text((extra - 3, 18), line, font=big_font, fill=(255, 180, 0, 55))
        ld.text((extra + 3, 18), line, font=big_font, fill=(255, 180, 0, 55))
        # Drop shadow
        ld.text((extra + 3, 21), line, font=big_font, fill=(0, 0, 0, 130))
        # Main text
        ld.text((extra,     18), line, font=big_font, fill=(255, 214, 0, 255))
        line_imgs.append((np.array(limg), cx, cy))

    def make_frame(t: float) -> np.ndarray:
        prog = t / max(duration, 0.001)

        # White flash on entry (first 0.08 s): creates hard-cut energy
        if t < 0.08:
            flash = int(255 * (1.0 - t / 0.08))
            return np.full((h, w, 3), flash, dtype=np.uint8)

        # Background: real section image at 55% brightness (not pure black)
        pulse = 1.0 + 0.045 * math.sin(t * 4.2)
        if bg_np is not None:
            bg_res = np.array(Image.fromarray(bg_np).resize((w, h), Image.BILINEAR))
            # Auto-lift dark source images before overlay
            _ibm = bg_res.mean()
            if _ibm < 80.0:
                bg_res = np.clip(bg_res.astype(np.float32) * min(2.5, 100.0 / max(_ibm, 1.0)), 0, 255).astype(np.uint8)
            frame  = (bg_res.astype(np.float32) * 0.52 * pulse).clip(0, 255).astype(np.uint8)
        else:
            bv    = int(4 * pulse)
            frame = np.full((h, w, 3), (bv, bv, int(15 * pulse)), dtype=np.uint8)

        img  = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        # ── Section counter (fades in fast) ──────────────────────────────
        sm_txt = f"SECTION {num} OF {total}"
        sw     = _text_w(draw, sm_txt, sm_font)
        c_a    = min(255, int(255 * prog / 0.18))
        draw.text(((w - sw) // 2, h // 2 - 160), sm_txt,
                  font=sm_font, fill=(90, 90, min(115, c_a + 5)))

        # ── Sweep lines expand from center ────────────────────────────────
        sw_p  = _ease_out_quad(min(1.0, prog / 0.42))
        sw_hw = int(sw_p * 380)
        mid_y = h // 2 - 52
        bot_y = mid_y + len(lines) * 116 + 14
        if sw_hw > 0:
            for ly2 in (mid_y, bot_y):
                draw.rectangle(
                    [(w // 2 - sw_hw, ly2),
                     (w // 2 + sw_hw, ly2 + 3)],
                    fill=(42, 42, 68)
                )

        # ── Text zoom-in with spring (ease-out-back) ──────────────────────
        txt_p  = min(1.0, max(0.0, (prog - 0.04) / 0.52))
        scale  = max(0.04, _ease_out_back(txt_p, s=1.55))

        frame_np = np.array(img)
        for i, (line_np, cx, cy) in enumerate(line_imgs):
            lh2, lw2 = line_np.shape[:2]
            new_w = max(4, int(lw2 * scale))
            new_h = max(4, int(lh2 * scale))
            scaled = np.array(
                Image.fromarray(line_np).resize((new_w, new_h), Image.BILINEAR)
            )
            px = cx - new_w // 2
            py = cy - new_h // 2
            _paste_np(frame_np, scaled, px, py, alpha_factor=min(1.0, scale * 1.4))

        # ── Gold underline grows ──────────────────────────────────────────
        bar_y  = h // 2 + (len(lines) * 116) - 16
        bar_p  = _ease_out_quad(min(1.0, max(0.0, (prog - 0.38) / 0.38)))
        bar_hw = int(bar_p * 230)
        if bar_hw > 0:
            frame_np[bar_y: bar_y + 6,
                     w // 2 - bar_hw: w // 2 + bar_hw] = [255, 214, 0]

        # ── CTA bump: appears 33-80% through clip, fades in and out ──────
        if _cta_np is not None:
            show_start, show_end = 0.33, 0.80
            if show_start <= prog <= show_end + 0.12:
                if prog < show_start + 0.10:
                    cta_alpha = (prog - show_start) / 0.10
                elif prog > show_end:
                    cta_alpha = 1.0 - (prog - show_end) / 0.12
                else:
                    cta_alpha = 1.0
                cta_alpha = max(0.0, min(1.0, cta_alpha))
                _paste_np(frame_np, _cta_np, 0, 0, alpha_factor=cta_alpha)

        return frame_np

    return _make_videoclip(make_frame, duration)


# ─────────────────────────────────────────────────────────────────────────────
# Sound-effects system
# ─────────────────────────────────────────────────────────────────────────────

_SFX_DIR = Path(__file__).resolve().parent / "sfx"

_SFX_KEYWORD_MAP: dict[str, list[str]] = {
    "impact":    ["impact_hit.mp3", "bass_impact.mp3", "punch.mp3"],
    "hit":       ["impact_hit.mp3", "punch.mp3"],
    "riser":     ["riser.mp3", "tension_riser.mp3"],
    "whoosh":    ["whoosh.mp3", "swipe.mp3"],
    "snap":      ["snap.mp3", "click.mp3"],
    "bass":      ["bass_drop.mp3", "bass_impact.mp3"],
    "drop":      ["bass_drop.mp3", "impact_hit.mp3"],
    "chime":     ["chime.mp3", "bell.mp3"],
    "click":     ["click.mp3", "snap.mp3"],
    "swell":     ["swell.mp3", "riser.mp3"],
    "silence":   [],   # deliberate silence — no file needed
    "pause":     [],
    "room-tone": [],
}

def _resolve_sfx(sound_cue: str) -> "Path | None":
    """
    Find an SFX file for a sound_cue string.
    Scans _SFX_KEYWORD_MAP for keyword matches (first match wins).
    Returns Path to the file, or None if unavailable / intentional silence.
    """
    cue_lower = (sound_cue or "").lower()
    for keyword, filenames in _SFX_KEYWORD_MAP.items():
        if keyword in cue_lower:
            for fname in filenames:
                candidate = _SFX_DIR / fname
                if candidate.exists():
                    return candidate
            break   # keyword matched but no file found — skip rather than try other keywords
    return None


def _mix_sfx(video_clip, sound_cue: str, at_time: float, volume: float = 0.18):
    """
    Overlay a single SFX clip at `at_time` seconds into `video_clip`.
    Silently skips if no matching SFX file is found.
    Returns the (possibly composite-audio) clip.
    """
    sfx_path = _resolve_sfx(sound_cue)
    if not sfx_path:
        return video_clip
    try:
        if _is_v1():
            from moviepy.editor import AudioFileClip, CompositeAudioClip
        else:
            from moviepy import AudioFileClip, CompositeAudioClip
        sfx_clip = AudioFileClip(str(sfx_path)).with_volume_scaled(volume)
        sfx_at = sfx_clip.with_start(at_time)
        original_audio = video_clip.audio
        if original_audio is None:
            return video_clip.with_audio(sfx_at)
        composite = CompositeAudioClip([original_audio, sfx_at])
        return video_clip.with_audio(composite)
    except Exception as e:
        log.warning(f"SFX mix failed ({sfx_path.name}): {e}")
        return video_clip


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def make_video_from_script(script: dict, output_path: str) -> None:
    """
    Build a creator-grade MP4.

    Per section the visual sequence is:
      [intro card 0.75s] → [text-cut₁] → [callout card 2s] → [text-cut₂] → …

    All cuts play over ONE continuous TTS audio track per section.
    Each cut is a different camera direction + pop-in animation.
    """
    if _is_v1():
        from moviepy.editor import AudioFileClip, concatenate_videoclips
        from moviepy.editor import CompositeVideoClip                       # noqa: F401
    else:
        from moviepy import AudioFileClip, concatenate_videoclips
        from moviepy.video.fx import FadeIn, FadeOut

    sections = script.get("sections", [])
    if not sections:
        raise ValueError("Script has no sections")

    _reset_bg_cache()  # fresh dedup set for this render

    tmp          = Path(tempfile.mkdtemp(prefix="ytgen_"))
    all_clips    = []
    all_words    = []      # accumulated word-timing for SRT (absolute timestamps)
    audio_offset = 0.0    # running total of audio seconds placed so far
    first_bg     = None
    total        = len(sections)

    for sec_idx, sec in enumerate(sections):
        name        = sec.get("name",          f"Section {sec_idx + 1}")
        narration   = sec.get("narration",      "Welcome.")
        callout     = sec.get("callout_text",   "")
        keyword     = sec.get("visual_keyword", "cinematic dramatic scene")
        lower       = sec.get("lower_third",    name)
        style       = _scene_style(sec.get("scene_style", ""), sec_idx)
        behavior    = sec.get("human_behavior", "")
        color_mood  = sec.get("color_mood", "curiosity")
        cta_type    = sec.get("cta_type", "")
        peak_sentence  = sec.get("peak_sentence", "").strip()
        emotion_target = sec.get("emotion_target", "shock").strip()
        retention_bridge_text = sec.get("retention_bridge", "").strip()

        log.info(f"  [{sec_idx + 1}/{total}] {name}")

        # ── TTS audio (one file per section) + word timing for SRT ────────
        audio_path            = str(tmp / f"audio_{sec_idx:02d}.mp3")
        sec_voice = _voice_for_section(sec_idx)
        has_audio, sec_words  = _tts_with_timing(narration, audio_path, voice=sec_voice)

        # Shift word timestamps to absolute video time and collect.
        # Each section in the final video is [intro card (INTRO_DUR, silent)] +
        # [text cuts with audio (section_dur)].  Audio starts AFTER the intro card,
        # so SRT timestamps must be shifted forward by INTRO_DUR per section.
        for w in sec_words:
            all_words.append({
                "word":  w["word"],
                "start": w["start"] + audio_offset + INTRO_DUR,
                "end":   w["end"]   + audio_offset + INTRO_DUR,
            })

        if has_audio:
            audio_clip  = AudioFileClip(audio_path)
            section_dur = audio_clip.duration
        else:
            audio_clip  = None
            section_dur = max(8.0, len(narration.split()) / 2.5)

        # Accumulate total section duration: intro card + narration audio
        audio_offset += INTRO_DUR + section_dur

        # ── Fetch 6 unique backgrounds per section — new bg every 3 cuts ────
        asset_prompt  = sec.get("asset_prompt", "")
        contrast_pair = sec.get("contrast_pair", "")
        scene_style   = sec.get("scene_style", "")
        color_mood    = sec.get("color_mood", "")
        first_frame_d = sec.get("first_frame", "")
        visual_meta   = sec.get("visual_metaphor", keyword)
        human_beh_l   = sec.get("human_behavior", "")
        contrast_kw   = contrast_pair.replace(" vs ", " ").replace("/", " ")[:60] if contrast_pair else ""

        # Use SPECIFIC visual descriptions from script — not generic asset_prompt
        _sec_kw_pairs = [
            # [0] primary: exact visual keyword (most content-specific)
            (keyword,
             f"dramatic scene: {keyword}"),
            # [1] first-frame scene or human behavior
            (first_frame_d[:100] if first_frame_d else (human_beh_l if human_beh_l else f"{keyword} closeup"),
             f"dramatic human: {keyword}"),
            # [2] visual metaphor
            (visual_meta[:100] if visual_meta != keyword else f"{keyword} action",
             f"cinematic {scene_style}: {keyword}"),
            # [3] contrast A
            (contrast_kw if contrast_kw else f"{keyword} wide shot",
             f"wide cinematic: {keyword}"),
            # [4] mood atmosphere
            (f"{color_mood} dramatic lighting {keyword}",
             f"{color_mood} atmosphere: {keyword}"),
            # [5] payoff
            (f"{keyword} reveal dramatic closeup",
             f"reveal moment: {keyword}"),
        ]
        log.info(f"  [{sec_idx + 1}/{total}] Fetching 6 unique backgrounds…")
        _sec_bgs: list[np.ndarray] = []
        for _kw, _ap in _sec_kw_pairs:
            _img = _fetch_bg(_kw, asset_prompt=_ap) or _gradient_bg(sec_idx + len(_sec_bgs))
            _sec_bgs.append(np.array(_img.convert("RGB")))

        bg_a_np = _sec_bgs[0]
        bg_b_np = _sec_bgs[1]
        bg_c_np = _sec_bgs[2]
        if first_bg is None:
            first_bg = Image.fromarray(bg_a_np)

        # ── Build text chunks (caption style) ────────────────────────────
        words  = narration.split()
        chunks = [
            " ".join(words[i: i + WORDS_PER_CUT])
            for i in range(0, len(words), WORDS_PER_CUT)
        ]

        # ── Calculate timing ──────────────────────────────────────────────
        # Fixed slots: intro card + optional callout card
        fixed_dur   = INTRO_DUR + (CALLOUT_DUR if callout else 0.0)
        text_budget = max(section_dur - fixed_dur, len(chunks) * MIN_CUT_DUR)
        per_cut     = max(MIN_CUT_DUR, text_budget / max(len(chunks), 1))

        # ── 1. Animated section intro card — built separately, NO audio ─────
        # The intro card is prepended AFTER audio attachment so narration audio
        # starts exactly at the first text cut, not under the title card.
        intro_clip = _intro_clip(name, sec_idx + 1, total, INTRO_DUR, bg_np=_sec_bgs[0])
        sec_clips  = []   # text cuts only — intro card is joined after audio attach

        # ── 2. Kinetic-typography text-cut clips ──────────────────────────
        # Callout card is inserted after the FIRST text cut for impact
        next_sec          = sections[sec_idx + 1] if sec_idx + 1 < total else {}
        next_section_name = next_sec.get("name", "")
        is_last_section   = sec_idx == total - 1
        # Prefer the full LLM-written tease sentence over the bare section name.
        # Fall back to the next section's own retention_bridge if our field is
        # empty, then to the next section name as last resort.
        _rb_full = (
            retention_bridge_text
            or next_sec.get("retention_bridge", "")
            or next_section_name
        )
        _warp_styles = ["zoom_punch", "ripple", "push_warp", "zoom_punch"]
        _prev_bg_np: "np.ndarray | None" = None

        # Energy-based pacing for this section
        sec_ep = _section_energy(style, sec_idx, total)

        for cut_idx, chunk in enumerate(chunks):
            # Rotate through all 6 unique backgrounds — new image every 3 cuts
            bg_np = _sec_bgs[cut_idx % len(_sec_bgs)]
            direction = (sec_idx * len(chunks) + cut_idx) % 4

            is_last_cut = cut_idx == len(chunks) - 1

            # Zach King warp transition between consecutive cuts
            if _CINEMATIC_ENGINE and cut_idx > 0 and _prev_bg_np is not None:
                try:
                    warp_style = _warp_styles[cut_idx % len(_warp_styles)]
                    trans = _ce.make_warp_transition_clip(
                        _prev_bg_np, bg_np, duration=0.3, style=warp_style, fps=FPS
                    )
                    sec_clips.append(trans)
                except Exception:
                    pass

            clip = _word_reveal_clip(
                bg_np, chunk, lower, sec_idx + 1, total,
                per_cut, direction=direction, pop=True,
                scene_style=style,
                callout=callout,
                human_behavior=behavior,
                color_mood=color_mood,
                cta_type=cta_type if cut_idx == 0 else "",
                retention_bridge=_rb_full if (is_last_cut and not is_last_section) else "",
                anim_style=sec_ep["anim_style"],
                energy=sec_ep["energy"],
            )
            sec_clips.append(clip)
            _prev_bg_np = bg_np

            # Insert animated callout card right after first text cut
            if cut_idx == 0 and callout:
                sec_clips.append(_callout_clip(callout, CALLOUT_DUR, color_mood=color_mood))

        # ── 3. Concatenate text-cut visuals (intro card NOT included yet) ───
        if not sec_clips:
            # Safety: if all cuts were skipped, create a minimal black clip
            sec_clips = [_intro_clip("", sec_idx + 1, total, section_dur)]

        text_visual = concatenate_videoclips(sec_clips, method="compose")

        # Trim text visual to exactly match audio duration
        vis_dur = text_visual.duration
        if vis_dur > section_dur and has_audio:
            text_visual = _subclip(text_visual, 0, section_dur)
        # (if visual < audio, the last frame freezes — fine for long-form)

        # ── 4. Fade in / out — applied BEFORE audio attachment ────────────
        # Applying fades before audio means video and audio are handled
        # independently: video fades in/out at section boundaries while
        # narration audio plays at full level (no speech fade-out).
        # In MoviePy v2, FadeIn/FadeOut are VideoEffects only; we handle
        # audio fades separately below for a matched envelope.
        FADE = 0.35
        if _is_v1():
            text_visual = text_visual.fadein(FADE).fadeout(FADE)
        else:
            text_visual = text_visual.with_effects([FadeIn(FADE), FadeOut(FADE)])

        # ── 5. Attach audio with matching fade envelope ────────────────────
        if audio_clip:
            # Apply matching audio fades so voice envelope matches video fade
            if not _is_v1():
                try:
                    from moviepy.audio.fx import AudioFadeIn, AudioFadeOut
                    audio_clip = audio_clip.with_effects(
                        [AudioFadeIn(FADE), AudioFadeOut(FADE)]
                    )
                except Exception:
                    pass   # older moviepy sub-version — skip audio fades
            text_visual = _set_audio(text_visual, audio_clip)

        # ── SFX — mix sound cue at the very start of the text visual ──────
        sound_cue = sec.get("sound_cue", "")
        if sound_cue:
            text_visual = _mix_sfx(text_visual, sound_cue, at_time=0.0, volume=0.18)

        # ── Prepend the SILENT intro card ─────────────────────────────────
        # Layout per section in final video:
        #   [intro card — INTRO_DUR s, silent] + [text cuts — section_dur s, with audio]
        # Narration now starts exactly at the first text cut (after intro card).
        section_visual = concatenate_videoclips([intro_clip, text_visual], method="compose")

        # ── Pull-quote card — silent 2.5s cinematic punctuation ───────────
        # Appended AFTER the section audio, so sync is never affected.
        # The peak_sentence is the single most powerful line from this
        # section's narration — displayed full-screen as an emotional beat.
        _PQ_DUR = 2.5
        if peak_sentence and len(peak_sentence.split()) >= 4:
            try:
                pq_clip = _pull_quote_clip(
                    sentence=peak_sentence,
                    emotion=emotion_target,
                    bg_arr=_sec_bgs[0] if _sec_bgs else None,
                    duration=_PQ_DUR,
                )
                section_visual = concatenate_videoclips(
                    [section_visual, pq_clip], method="compose"
                )
                log.info(f"    ↳ pull-quote inserted ({emotion_target}): {peak_sentence[:60]}…")
            except Exception as _pq_err:
                log.warning(f"    pull-quote skipped: {_pq_err}")

        all_clips.append(section_visual)

    # ── Final assembly ────────────────────────────────────────────────────
    final = concatenate_videoclips(all_clips, method="compose")

    # ── Optional background music ─────────────────────────────────────────
    music_path = os.environ.get("BACKGROUND_MUSIC_PATH", "")
    if music_path and Path(music_path).exists():
        log.info("  Mixing background music…")
        final = _add_background_music(final, music_path)

    # ── Render ────────────────────────────────────────────────────────────
    log.info(f"Rendering {final.duration:.0f}s video → {output_path}")
    final.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(tmp / "tmp_audio.m4a"),
        remove_temp=True,
        logger=None,
    )
    log.info("Video render complete ✅")

    # ── SRT captions ──────────────────────────────────────────────────────
    srt_path = output_path.replace(".mp4", ".srt")
    try:
        _write_srt(all_words, srt_path)
    except Exception as e:
        log.warning(f"SRT generation failed: {e}")

    # ── Thumbnails (A/B variants) ──────────────────────────────────────────
    if first_bg:
        thumb_path = output_path.replace(".mp4", "_thumb.jpg")
        try:
            _generate_thumbnail(script, first_bg, thumb_path, variant=0)
        except Exception as e:
            log.warning(f"Thumbnail A generation failed: {e}")
        # Variant B — fetch a fresh background for visual contrast
        try:
            kw_b_concept = script.get("thumbnail_concept_b", script.get("thumbnail_concept", ""))
            bg_b_thumb = _fetch_bg(kw_b_concept or "dramatic person cinematic") or first_bg
            thumb_b_path = output_path.replace(".mp4", "_thumb_b.jpg")
            _generate_thumbnail(script, bg_b_thumb, thumb_b_path, variant=1)
        except Exception as e:
            log.warning(f"Thumbnail B generation failed: {e}")

    # ── QA frame export (4 frames at 10/35/60/85% of duration) ───────────
    try:
        from moviepy import VideoFileClip
        qa_dir = Path(output_path.replace(".mp4", "_qa_frames"))
        qa_dir.mkdir(exist_ok=True)
        with VideoFileClip(output_path) as vfc:
            dur = vfc.duration
            for pct in (0.10, 0.35, 0.60, 0.85):
                t = dur * pct
                frame = vfc.get_frame(t)
                frame_img = Image.fromarray(frame)
                frame_img.save(qa_dir / f"frame_{int(pct*100):02d}pct.jpg", "JPEG", quality=88)
        log.info(f"QA frames saved → {qa_dir}")
    except Exception as e:
        log.warning(f"QA frame export failed: {e}")

    # ── Flush bg URL/hash stores to disk (batch write — one I/O per render) ──
    _flush_bg_stores()


# ─────────────────────────────────────────────────────────────────────────────
# YouTube Shorts renderer  (1080 × 1920 portrait, 50-58 seconds)
# ─────────────────────────────────────────────────────────────────────────────

_SHORTS_W, _SHORTS_H = 1080, 1920
_SHORTS_WORDS_PER_CUT = 3   # 3 words → CapCut-pace cuts (~0.9-1.4s each)
_SHORTS_HOOK_CUTS     = 5   # first N chunks get hook treatment (bigger font, slower reveal)


def _fetch_pexels_portrait(keyword: str, asset_prompt: str = "") -> "Image.Image | None":
    """Portrait-oriented background via the unified multi-source fetcher."""
    return _fetch_bg(keyword, asset_prompt=asset_prompt, size=(_SHORTS_W, _SHORTS_H))


def _gradient_bg_portrait(index: int = 0) -> Image.Image:
    c1, c2 = _PALETTES[index % len(_PALETTES)]
    arr = np.zeros((_SHORTS_H, _SHORTS_W, 3), dtype=np.uint8)
    for ch in range(3):
        arr[:, :, ch] = np.linspace(c1[ch], c2[ch], _SHORTS_H, dtype=np.uint8)[:, None]
    return Image.fromarray(arr)


def _shorts_text_clip(
    bg_arr: np.ndarray,
    text_chunk: str,
    accent: tuple[int, int, int],
    duration: float,
    cut_index: int,
    total_cuts: int,
    is_hook: bool = False,
    vid_path: "str | None" = None,  # optional: video file path for moving background
    color_mood: str = "curiosity",   # emotional ladder mood for tint
) -> object:
    """
    One Shorts cut — CapCut/TikTok visual standard.

    Key differences from generic text overlay:
      • Text at 60% down (lower-center safe zone) not dead-center
      • Hook cuts: 132px font; body: 92px
      • 4-direction Ken Burns cycles per cut_index
      • 1.18→1.0 scale pop in first 0.10s (visible punch)
      • Accent color flash on first 0.07s of each cut
      • Particle layer (subtle energy field)
      • Thin 4px progress bar
      • vid_path: if set, uses REAL VIDEO FRAMES as background (Pexels/AI) instead of static image
    """
    W, H = _SHORTS_W, _SHORTS_H

    font_size   = 132 if is_hook else 92
    font        = _get_font(font_size, bold=True)
    # 3 text zones: lower-center, ultra-lower (TikTok), mid — rotate by cut
    _TEXT_ZONES = [0.60, 0.72, 0.50]
    TEXT_CY     = int(H * _TEXT_ZONES[cut_index % 3])
    lines       = _fit_text_lines(text_chunk.upper(), font, int(W * 0.90), 3)
    line_h      = font_size + 22
    total_text_h = len(lines) * line_h
    y_start     = TEXT_CY - total_text_h // 2

    def make_frame(t: float) -> np.ndarray:
        progress = t / max(duration, 0.001)

        # ── 4-direction Ken Burns ───────────────────────────────────────────
        direction = cut_index % 4
        if direction == 0:                          # zoom in
            scale  = 1.04 + progress * 0.07
            xd, yd = 0, 0
        elif direction == 1:                        # pan right
            scale  = 1.10
            xd     = int(W * 0.09 * progress)
            yd     = 0
        elif direction == 2:                        # zoom out
            scale  = 1.12 - progress * 0.07
            xd, yd = 0, 0
        else:                                       # pan left
            scale  = 1.10
            xd     = -int(W * 0.09 * progress)
            yd     = 0

        # ── Background: real video clip OR static image ────────────────────
        if vid_path:
            # REAL VIDEO BACKGROUND — live frames from Pexels/AI video clip
            # The video provides actual human motion, real camera movement
            frame = _make_video_frame_at_t(vid_path, t, (W, H), scale=scale, xd=xd, yd=yd)
        else:
            # Static image with Ken Burns pan/zoom
            nw, nh = max(W, int(W * scale)), max(H, int(H * scale))
            bg_img = Image.fromarray(bg_arr).resize((nw, nh), Image.LANCZOS)
            x0 = max(0, min((nw - W) // 2 + xd, nw - W))
            y0 = max(0, min((nh - H) // 2 + yd, nh - H))
            frame = np.array(bg_img.crop((x0, y0, x0 + W, y0 + H)))

        # ── Auto-brightness: lift dark source images to minimum target ─────
        _src_mean = frame.mean()
        if _src_mean < 80.0:
            _boost = min(3.0, 100.0 / max(_src_mean, 1.0))
            frame = np.clip(frame.astype(np.float32) * _boost, 0, 255).astype(np.uint8)

        # ── Emotional-ladder mood tint — syncs background colour to arc ───────
        # Applied before overlay so it blends naturally into the darkness
        frame = _apply_mood_tint(frame, color_mood, strength=0.10)

        # ── Cinematic overlay: darken to 62%/72% (still shows image clearly) ──
        frame = (frame.astype(np.float32) * (0.62 if is_hook else 0.72)).astype(np.uint8)

        # ── Vignette ───────────────────────────────────────────────────────
        Yi, Xi = np.ogrid[:H, :W]
        dist   = np.sqrt(((Xi - W / 2) / (W / 2)) ** 2 + ((Yi - H / 2) / (H / 2)) ** 2)
        vmask  = np.clip(1.0 - 0.28 * dist ** 2.5, 0.0, 1.0)[:, :, np.newaxis]
        frame  = (frame * vmask).astype(np.uint8)

        # ── Accent color flash (first 0.07 s of every cut except the first) ──
        if t < 0.07 and cut_index > 0:
            strength = (1.0 - t / 0.07) * 0.32
            acc_f    = np.array([[[*accent]]], dtype=np.float32) / 255.0
            frame    = np.clip(
                frame.astype(np.float32) * (1 - strength) + acc_f * 255 * strength,
                0, 255,
            ).astype(np.uint8)

        # ── Scale pop: 1.18 → 1.0 in first 0.10 s ─────────────────────────
        pop_s = 1.18 - 0.18 * _ease_out_quad(min(1.0, t / 0.10))

        img   = Image.fromarray(frame)
        draw  = ImageDraw.Draw(img)

        # ── Text reveal: 0 → 1 in first 12 % of clip ──────────────────────
        reveal = min(1.0, progress / 0.12)
        alpha  = int(255 * reveal)

        # Accent underline bar (slides out from centre)
        bar_half = int(W * 0.42 * reveal)
        draw.rectangle(
            [(W // 2 - bar_half, y_start - 16), (W // 2 + bar_half, y_start - 8)],
            fill=(*accent, alpha),
        )

        if pop_s > 1.01:
            # Render text on a temp surface and scale it for the pop effect
            tc_w = int(W * 1.2)
            tc_h = total_text_h + 40
            tc   = Image.new("RGBA", (tc_w, tc_h), (0, 0, 0, 0))
            td   = ImageDraw.Draw(tc)
            for i, line in enumerate(lines):
                bb  = td.textbbox((0, 0), line, font=font)
                tw  = bb[2] - bb[0]
                lxl = (tc_w - tw) // 2
                lyl = 20 + i * line_h
                col = accent if i == 0 else (255, 255, 255)
                td.text((lxl + 5, lyl + 5), line, font=font, fill=(0, 0, 0, alpha))
                td.text((lxl, lyl),         line, font=font, fill=(*col, alpha))
            sw = max(1, int(tc_w * pop_s))
            sh = max(1, int(tc_h * pop_s))
            tc_scaled = tc.resize((sw, sh), Image.LANCZOS)
            img.paste(tc_scaled, ((W - sw) // 2, y_start - 20 - (sh - tc_h) // 2), tc_scaled)
        else:
            for i, line in enumerate(lines):
                bb  = draw.textbbox((0, 0), line, font=font)
                tw  = bb[2] - bb[0]
                lx  = (W - tw) // 2
                ly  = y_start + i * line_h
                col = accent if i == 0 else (255, 255, 255)
                draw.text((lx + 6, ly + 6), line, font=font, fill=(0, 0, 0, min(alpha, 200)))
                draw.text((lx + 2, ly + 2), line, font=font, fill=(0, 0, 0, alpha))
                draw.text((lx, ly),         line, font=font, fill=(*col, alpha))

        # ── Particle layer (subtle energy, 18 particles) ───────────────────
        pt      = _particles_layer(W, H, t, count=18, seed=cut_index)
        frame_np = np.array(img)
        _paste_np(frame_np, pt, 0, 0, alpha_factor=0.50)

        # ── Thin progress bar (4 px — far less distracting than 12 px) ────
        overall = (cut_index + progress) / max(total_cuts, 1)
        bar_px  = int(W * overall)
        result  = Image.fromarray(frame_np)
        ImageDraw.Draw(result).rectangle([(0, H - 5), (bar_px, H)], fill=(*accent, 180))

        return np.array(result)

    return _make_videoclip(make_frame, duration)


def _shorts_callout_clip(
    callout_text: str,
    accent: tuple[int, int, int],
    bg_arr: np.ndarray,
    duration: float = 2.0,
) -> object:
    """Full-screen accent-banner callout — the 'reveal moment' card."""
    W, H = _SHORTS_W, _SHORTS_H
    cf    = _get_font(88, bold=True)
    lines = _fit_text_lines(callout_text.upper(), cf, int(W * 0.88), 2)
    lh    = 108
    total_h = len(lines) * lh + 80
    by    = (H - total_h) // 2

    def make_frame(t: float) -> np.ndarray:
        prog  = t / max(duration, 0.001)
        scale = 1.04
        nw, nh = int(W * scale), int(H * scale)
        bg_img = Image.fromarray(bg_arr).resize((nw, nh), Image.LANCZOS)
        frame  = np.array(bg_img.crop(((nw - W) // 2, (nh - H) // 2,
                                        (nw - W) // 2 + W, (nh - H) // 2 + H)))
        _cm = frame.mean()
        if _cm < 80.0:
            frame = np.clip(frame.astype(np.float32) * min(3.0, 100.0 / max(_cm, 1.0)), 0, 255).astype(np.uint8)
        frame  = (frame.astype(np.float32) * 0.58).astype(np.uint8)
        Yi, Xi = np.ogrid[:H, :W]
        dist   = np.sqrt(((Xi - W/2)/(W/2))**2 + ((Yi - H/2)/(H/2))**2)
        frame  = (frame * np.clip(1.0 - 0.30 * dist**2.5, 0, 1)[:, :, np.newaxis]).astype(np.uint8)

        reveal = min(1.0, prog / 0.15)
        alpha  = int(255 * reveal)

        banner = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        bd     = ImageDraw.Draw(banner)
        bd.rectangle([(0, by - 20), (W, by + total_h)], fill=(*accent, int(220 * reveal)))
        frame_np = frame.copy()
        _paste_np(frame_np, np.array(banner), 0, 0)

        img  = Image.fromarray(frame_np)
        draw = ImageDraw.Draw(img)
        for i, line in enumerate(lines):
            bb  = draw.textbbox((0, 0), line, font=cf)
            tw  = bb[2] - bb[0]
            lx  = (W - tw) // 2
            ly  = by + 30 + i * lh
            draw.text((lx + 5, ly + 5), line, font=cf, fill=(0, 0, 0, alpha))
            draw.text((lx, ly),         line, font=cf, fill=(255, 255, 255, alpha))
        return np.array(img)

    return _make_videoclip(make_frame, duration)


def _shorts_pause_beat(accent: tuple[int, int, int], duration: float = 0.28) -> object:
    """
    Psychological pause beat — a brief accent-colored flash between cuts.
    Resets the brain's habituation to constant motion. Lasts 0.28s.
    Creates the same effect as a 'breath' between phrases in a speech.
    """
    W, H = _SHORTS_W, _SHORTS_H

    def make_frame(t: float) -> np.ndarray:
        prog = t / max(duration, 0.001)
        # Flash in quickly (first 30%), hold briefly, flash out (last 30%)
        if prog < 0.30:
            alpha = prog / 0.30
        elif prog > 0.70:
            alpha = (1.0 - prog) / 0.30
        else:
            alpha = 1.0
        alpha = float(np.clip(alpha, 0.0, 1.0))
        r, g, b = accent
        brightness = int(alpha * 28)  # subtle — not a blinding white flash
        frame = np.full((H, W, 3), brightness, dtype=np.uint8)
        # Accent color glow at center
        cx, cy = W // 2, H // 2
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt(((X - cx) / (W * 0.3)) ** 2 + ((Y - cy) / (H * 0.3)) ** 2)
        glow = np.clip(1.0 - dist, 0.0, 1.0)[:, :, np.newaxis] * alpha * 0.45
        tint = np.array([r, g, b], dtype=np.float32) * glow
        frame = np.clip(frame.astype(np.float32) + tint, 0, 255).astype(np.uint8)
        return frame

    return _make_videoclip(make_frame, duration)


def _shorts_cta_clip(
    accent: tuple[int, int, int],
    bg_arr: np.ndarray,
    duration: float = 4.0,
    end_screen_hook: str = "",
    comment_bait: str = "",
) -> object:
    """
    End-screen card with psychological architecture:
    - Shows end_screen_hook (curiosity gap to next video) in large text
    - Shows comment_bait as a provocative sub-line (forces responses)
    - Replaces generic 'subscribe for more' with content that makes NOT subscribing feel irrational
    """
    W, H = _SHORTS_W, _SHORTS_H
    # Truncate for display
    hook_text = (end_screen_hook or "Watch what happened next ↓")[:80]
    bait_text = (comment_bait or "")[:60]

    def make_frame(t: float) -> np.ndarray:
        prog  = t / max(duration, 0.001)
        scale = max(1.01, 1.06 - prog * 0.04)
        nw, nh = int(W * scale), int(H * scale)
        bg_img = Image.fromarray(bg_arr).resize((nw, nh), Image.LANCZOS)
        frame  = np.array(bg_img.crop(((nw - W) // 2, (nh - H) // 2,
                                        (nw - W) // 2 + W, (nh - H) // 2 + H)))
        _ctam = frame.mean()
        if _ctam < 80.0:
            frame = np.clip(frame.astype(np.float32) * min(3.0, 100.0 / max(_ctam, 1.0)), 0, 255).astype(np.uint8)
        # Darker overlay — end card should feel like a different zone
        frame  = (frame.astype(np.float32) * 0.42).astype(np.uint8)
        # Mood tint: action/green energy for end screen
        frame  = _apply_mood_tint(frame, "action", strength=0.12)
        Yi, Xi = np.ogrid[:H, :W]
        dist   = np.sqrt(((Xi - W/2)/(W/2))**2 + ((Yi - H/2)/(H/2))**2)
        frame  = (frame * np.clip(1.0 - 0.35 * dist**2.5, 0, 1)[:, :, np.newaxis]).astype(np.uint8)

        reveal   = min(1.0, prog / 0.20)
        alpha    = int(255 * reveal)
        hook_font = _get_font(72, bold=True)
        bait_font = _get_font(48)
        sm_font   = _get_font(40)
        img       = Image.fromarray(frame)
        draw      = ImageDraw.Draw(img)

        # Accent top bar
        bar_prog = min(1.0, prog / 0.25)
        bar_w    = int(W * _ease_out_quad(bar_prog))
        if bar_w > 0:
            draw.rectangle([(0, int(H * 0.22)), (bar_w, int(H * 0.22) + 5)],
                           fill=(*accent, alpha))

        # End-screen hook text (curiosity gap to next video)
        hook_lines = _fit_text_lines(hook_text, hook_font, int(W * 0.88), 3)
        hook_start_y = int(H * 0.26)
        for i, line in enumerate(hook_lines):
            bb  = draw.textbbox((0, 0), line, font=hook_font)
            tw  = bb[2] - bb[0]
            lx  = (W - tw) // 2
            ly  = hook_start_y + i * 86
            draw.text((lx + 4, ly + 4), line, font=hook_font, fill=(0, 0, 0, alpha))
            draw.text((lx,     ly),     line, font=hook_font, fill=(255, 255, 255, alpha))

        # Comment bait line (accent color — stands out as the polarizing claim)
        if bait_text and prog > 0.3:
            bait_alpha = int(255 * min(1.0, (prog - 0.3) / 0.2))
            bait_lines = _fit_text_lines(bait_text, bait_font, int(W * 0.82), 2)
            bait_start_y = hook_start_y + len(hook_lines) * 86 + 28
            for i, line in enumerate(bait_lines):
                bb  = draw.textbbox((0, 0), line, font=bait_font)
                tw  = bb[2] - bb[0]
                lx  = (W - tw) // 2
                ly  = bait_start_y + i * 60
                draw.text((lx + 3, ly + 3), line, font=bait_font, fill=(0, 0, 0, bait_alpha))
                draw.text((lx,     ly),     line, font=bait_font, fill=(*accent, bait_alpha))

        # Small subscribe prompt — de-emphasized, below the real content
        if prog > 0.55:
            sub_alpha = int(255 * min(1.0, (prog - 0.55) / 0.25))
            sub_text  = "↑ Subscribe for what's next"
            bb3 = draw.textbbox((0, 0), sub_text, font=sm_font)
            tw3 = bb3[2] - bb3[0]
            draw.text(((W - tw3) // 2, int(H * 0.74)), sub_text,
                      font=sm_font, fill=(180, 180, 180, sub_alpha))

        return np.array(img)

    return _make_videoclip(make_frame, duration)


def make_shorts_from_script(script: dict, output_path: str) -> None:
    """
    Render a YouTube Shorts video (1080 × 1920, 45-55 seconds).

    CapCut/TikTok-level architecture:
      • 3 rotating portrait backgrounds (hook zone / body / payoff zone)
      • 3-word cuts at 0.9-1.4 s each — creates native Shorts pacing
      • First N cuts: hook treatment (132 px font, 1.18× scale pop)
      • Mid-video: full-screen accent callout banner
      • End: 4-second subscribe CTA card (appended after narration audio)
      • SRT captions at 3 words per line
    """
    if _is_v1():
        from moviepy.editor import AudioFileClip, concatenate_videoclips
    else:
        from moviepy import AudioFileClip, concatenate_videoclips

    _reset_bg_cache()  # fresh dedup set for this render

    section   = (script.get("sections") or [{}])[0]
    narration = " ".join(p.strip() for p in [
        script.get("hook", ""),
        section.get("narration", ""),
    ] if p.strip())

    if not narration:
        raise ValueError("Shorts script has no narration text")

    keyword      = section.get("visual_keyword", "cinematic human dramatic")
    asset_prompt = section.get("asset_prompt", "")
    mood         = section.get("color_mood", "curiosity")
    accent       = _accent_for(mood)
    callout      = section.get("callout_text", "")
    contrast_kw  = (section.get("contrast_pair", "") or "").replace(" vs ", " ")[:50]
    human_beh    = (section.get("human_behavior", "") or "")[:50]

    tmp        = Path(tempfile.mkdtemp(prefix="ytshorts_"))
    audio_path = str(tmp / "shorts_audio.mp3")

    # ── TTS ──────────────────────────────────────────────────────────────────
    # Duration budget for a Shorts video ≤60 s:
    #   CTA tail:      4.0 s
    #   Callout tail:  2.0 s (if callout present)
    #   Safety margin: 1.0 s
    #   → max narration audio: 60 − 4 − 2 − 1 = 53 s
    _SHORTS_TOTAL_CAP      = 59.0   # hard cap on total final video (1 s under YouTube limit)
    _SHORTS_TAIL_DUR       = (2.0 if callout else 0.0) + 4.0   # callout + CTA
    _SHORTS_MAX_NARR_DUR   = _SHORTS_TOTAL_CAP - _SHORTS_TAIL_DUR - 0.5   # 0.5 s extra margin

    log.info("  [Shorts] Generating TTS audio…")
    has_audio, word_timing = _tts_with_timing(narration, audio_path, voice=_voice_for_section(0))

    if has_audio:
        audio_clip = AudioFileClip(audio_path)
        total_dur  = audio_clip.duration
    else:
        audio_clip = None
        total_dur  = max(35.0, len(narration.split()) / 2.5)

    # ── Hard narration audio cap — enforced before any clips are built ────────
    # If TTS produced audio longer than the budget (e.g. AI wrote too many words),
    # trim the audio and word_timing here so every downstream calculation is correct.
    if total_dur > _SHORTS_MAX_NARR_DUR:
        log.warning(
            f"  [Shorts] Narration audio {total_dur:.1f}s > budget {_SHORTS_MAX_NARR_DUR:.1f}s "
            f"— trimming to fit 60s Shorts cap"
        )
        total_dur = _SHORTS_MAX_NARR_DUR
        if audio_clip:
            audio_clip = _subclip(audio_clip, 0, _SHORTS_MAX_NARR_DUR)
        # Trim word_timing to match — drop any word that starts after the cut point
        if word_timing:
            word_timing = [w for w in word_timing if w["start"] < _SHORTS_MAX_NARR_DUR]

    # ── 8 unique portrait backgrounds — video clips when available, static fallback ──
    log.info(f"  [Shorts] Fetching 8 unique backgrounds for '{keyword}'…")
    first_frame_desc = section.get("first_frame", "")
    visual_metaphor  = section.get("visual_metaphor", keyword)
    _use_video_bgs   = (_PEXELS_VIDEO_BG == "1" and os.environ.get("PEXELS_API_KEY")) \
                       or _LTX_VIDEO_BG == "1" or _KLING_VIDEO_BG == "1"

    if _use_video_bgs:
        log.info("  [Shorts] Video mode: fetching REAL VIDEO CLIPS as backgrounds…")

    # Use the SPECIFIC visual descriptions from the script — not generic asset_prompt
    _bg_queries = [
        # [0] hook: exact first-frame scene description
        (first_frame_desc[:100] if first_frame_desc else keyword,
         f"cinematic dramatic {keyword}"),
        # [1] primary: exact visual keyword from script
        (keyword,
         f"dramatic human emotion: {keyword}"),
        # [2] human action: actual behavior from script
        (human_beh if human_beh else f"{keyword} action",
         f"real human in action: {keyword}"),
        # [3] visual metaphor
        (visual_metaphor[:100] if visual_metaphor else keyword,
         f"cinematic {mood}: {keyword}"),
        # [4] contrast A
        (f"{keyword} closeup dramatic face",
         f"extreme closeup human reaction: {keyword}"),
        # [5] contrast B
        (f"{contrast_kw} cinematic portrait" if contrast_kw else f"{keyword} wide cinematic",
         f"wide cinematic shot: {keyword}"),
        # [6] mood atmosphere
        (f"{mood} dramatic lighting {keyword}",
         f"{mood} cinematic atmosphere: {keyword}"),
        # [7] payoff/CTA zone — energetic, action-forward
        (f"{keyword} reveal dramatic",
         f"dramatic reveal moment: {keyword}"),
    ]
    bg_arrays:    list[np.ndarray]  = []   # static fallback images
    bg_vid_paths: list["str | None"] = []  # video clip paths (None = use static image)
    _tmp_vid_paths: list[str] = []         # track for cleanup after render

    for i, (kw, ap) in enumerate(_bg_queries):
        vid_path = None
        if _use_video_bgs:
            vid_path = _get_video_bg_clip(
                kw, ap,
                duration_needed=max(5.0, total_dur / max(len(_bg_queries), 1)),
                size=(_SHORTS_W, _SHORTS_H),
                slot_idx=i,
            )
            if vid_path:
                _tmp_vid_paths.append(vid_path)
        bg_vid_paths.append(vid_path)
        # Always fetch a static image as fallback
        img = _fetch_pexels_portrait(kw, asset_prompt=ap) or _gradient_bg_portrait(len(bg_arrays))
        bg_arrays.append(np.array(img.convert("RGB")))

    # ── Text chunks ───────────────────────────────────────────────────────────
    words_list = narration.split()
    chunks     = [
        " ".join(words_list[i: i + _SHORTS_WORDS_PER_CUT])
        for i in range(0, len(words_list), _SHORTS_WORDS_PER_CUT)
    ]
    total_cuts  = len(chunks)
    cta_dur     = 4.0
    callout_dur = 2.0

    # ── Per-chunk durations from actual TTS word timing ───────────────────────
    # Root cause of Shorts sync drift: uniform per_cut_dur divides audio evenly
    # but TTS speaks each 3-word chunk in different amounts of time (fast short
    # words vs slow long words).  By the 20th cut the visual can be 2–4 s ahead
    # of or behind the audio.  Fix: derive each chunk's exact duration from the
    # TTS word-boundary timestamps, then add 120 ms natural-pause buffer.
    if word_timing:
        chunk_durations = []
        wt_idx = 0
        for ci in range(total_cuts):
            start_idx = wt_idx
            end_idx   = min(wt_idx + _SHORTS_WORDS_PER_CUT - 1, len(word_timing) - 1)
            if start_idx < len(word_timing):
                t0  = word_timing[start_idx]["start"]
                t1  = word_timing[end_idx]["end"]
                dur = max(0.50, t1 - t0 + 0.12)   # 120 ms natural-pause buffer
            else:
                dur = max(0.50, total_dur / max(total_cuts, 1))
            chunk_durations.append(dur)
            wt_idx += _SHORTS_WORDS_PER_CUT
    else:
        # No word timing available — fall back to uniform (best we can do)
        _uniform = max(0.90, total_dur / max(total_cuts, 1))
        chunk_durations = [_uniform] * total_cuts

    log.info(f"  [Shorts] Building {total_cuts} text cuts (per-chunk TTS timing)…")

    # ── Emotional ladder: which cuts are in which mood phase ──────────────────
    # danger(hook) → curiosity(surprise) → evidence(tension) → payoff(relief) → action(end)
    def _cut_mood(idx: int, total: int) -> str:
        """Map cut index to emotional ladder phase."""
        frac = idx / max(total - 1, 1)
        if frac < 0.15:    return "danger"
        elif frac < 0.38:  return "curiosity"
        elif frac < 0.65:  return "evidence"
        elif frac < 0.85:  return "payoff"
        else:              return "action"

    # ── Build narration clips ONLY — no pause beats, no callout here ──────────
    # Pause beats (0.28 s × ~12 cuts = ~3.4 s total) were previously inserted
    # during narration, pushing every subsequent text cut out of alignment with
    # the continuous audio track (cumulative drift ≈ 3 s by end of Short).
    # Callout was also inserted mid-narration causing a 2.0 s visual blackout
    # while audio kept speaking — both moved to the SILENT TAIL after narration.
    narration_clips = []
    for idx, chunk in enumerate(chunks):
        is_hook    = idx < _SHORTS_HOOK_CUTS
        cut_mood   = _cut_mood(idx, total_cuts)
        cut_accent = _accent_for(cut_mood)

        # Select bg slot — hook→[0], payoff→[7], body rotates [1..6] every 4 cuts
        if idx < _SHORTS_HOOK_CUTS:
            slot = 0
        elif idx >= total_cuts - 3:
            slot = 7
        else:
            slot = 1 + ((idx // 4) % 6)

        bg       = bg_arrays[slot]
        vid_path = bg_vid_paths[slot]

        clip = _shorts_text_clip(bg, chunk, cut_accent, chunk_durations[idx], idx, total_cuts,
                                 is_hook=is_hook, vid_path=vid_path, color_mood=cut_mood)
        narration_clips.append(clip)

    # ── Narration visual — trim to audio duration ─────────────────────────────
    narration_visual  = concatenate_videoclips(narration_clips, method="compose")
    narration_vis_dur = narration_visual.duration

    # Trim if the 120 ms per-chunk buffers caused the visual to overshoot audio
    if narration_vis_dur > total_dur + 0.25:
        narration_visual = _subclip(narration_visual, 0, total_dur)
    # Under-shoot: MoviePy freezes the last frame while audio finishes — acceptable

    # ── Attach audio to narration visual ONLY ────────────────────────────────
    if audio_clip:
        narration_visual = _set_audio(narration_visual, audio_clip)

    # ── Silent tail: callout banner + CTA (no narration audio under these) ────
    # Callout plays in silence AFTER narration ends, so the viewer sees the
    # reveal banner without any mid-speech visual break.
    # CTA card also plays in silence — subscribers see the hook, not a black screen.
    _end_hook = script.get("end_screen_hook", "")
    _cmt_bait = script.get("comment_bait", "")
    tail_clips = []
    if callout:
        tail_clips.append(
            _shorts_callout_clip(callout, _accent_for("action"),
                                 bg_arrays[0], duration=callout_dur)
        )
    tail_clips.append(_shorts_cta_clip(
        _accent_for("action"),
        bg_arrays[7] if bg_vid_paths[7] is None else bg_arrays[0],
        duration=cta_dur,
        end_screen_hook=_end_hook,
        comment_bait=_cmt_bait,
    ))

    # ── Final assembly: [narration+audio] + [silent callout+CTA] ─────────────
    final = concatenate_videoclips([narration_visual] + tail_clips, method="compose")

    # ── Absolute hard cap — YouTube removes Shorts > 60 s ────────────────────
    # Three-layer defense: word cap in prompt (95-110) → word cap in validate_script (115)
    # → audio trim above → this final trim is the last safety net.
    if final.duration > _SHORTS_TOTAL_CAP:
        log.warning(
            f"  [Shorts] Final video {final.duration:.1f}s > {_SHORTS_TOTAL_CAP:.0f}s cap "
            f"— trimming tail to fit"
        )
        final = _subclip(final, 0, _SHORTS_TOTAL_CAP)

    log.info(f"  [Shorts] Rendering {final.duration:.1f}s portrait video → {output_path}")
    final.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(tmp / "tmp_audio.m4a"),
        remove_temp=True,
        logger=None,
    )

    srt_path = str(Path(output_path).with_suffix(".srt"))
    if word_timing:
        try:
            _write_srt(word_timing, srt_path, words_per_line=3)
            log.info(f"  SRT captions → {srt_path}")
        except Exception as e:
            log.warning(f"  [Shorts] SRT failed: {e}")

    # ── QA frame export ───────────────────────────────────────────────────
    try:
        from moviepy import VideoFileClip
        qa_dir = Path(output_path.replace(".mp4", "_qa_frames"))
        qa_dir.mkdir(exist_ok=True)
        with VideoFileClip(output_path) as vfc:
            dur = vfc.duration
            for pct in (0.10, 0.35, 0.60, 0.85):
                t = dur * pct
                frame = vfc.get_frame(t)
                frame_img = Image.fromarray(frame)
                frame_img.save(qa_dir / f"frame_{int(pct*100):02d}pct.jpg", "JPEG", quality=88)
        log.info(f"  [Shorts] QA frames saved → {qa_dir}")
    except Exception as e:
        log.warning(f"  [Shorts] QA frame export failed: {e}")

    # ── Cleanup: close video clip handles + delete temp video files ──────────
    _cleanup_video_cache()
    for _vp in _tmp_vid_paths:
        try: os.unlink(_vp)
        except OSError: pass

    # ── Flush bg URL/hash stores to disk (batch write — one I/O per render) ──
    _flush_bg_stores()

    log.info("  [Shorts] Render complete ✅")
