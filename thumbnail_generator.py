"""
thumbnail_generator.py — Auto-render YouTube thumbnails optimised for CTR.

Produces TWO variants (A/B) per video:
  Variant A — emotion-colour gradient background + bold title overlay + stat badge
  Variant B — dark cinematic background + contrasting accent title + arrow CTA

Both are 1280×720 (YouTube recommended thumbnail resolution).
The pipeline uploads both and reads CTR after 48h to keep the winner.
"""

import os
import sys
import textwrap
import logging
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

log = logging.getLogger(__name__)

THUMB_W, THUMB_H = 1280, 720

# Emotion → thumbnail palette (bolder/higher-contrast than slide palettes)
THUMB_PALETTES = {
    "shock":               {"bg1": (180, 0, 0),    "bg2": (40, 0, 0),    "accent": (255, 220, 0),  "text": (255, 255, 255)},
    "dread":               {"bg1": (10, 10, 60),   "bg2": (5, 5, 20),    "accent": (100, 160, 255), "text": (255, 255, 255)},
    "recognition":         {"bg1": (200, 120, 0),  "bg2": (80, 40, 0),   "accent": (255, 240, 120), "text": (255, 255, 255)},
    "cognitive_dissonance":{"bg1": (100, 0, 140),  "bg2": (40, 0, 60),   "accent": (255, 160, 0),   "text": (255, 255, 255)},
    "confusion":           {"bg1": (0, 100, 140),  "bg2": (0, 40, 60),   "accent": (100, 240, 255), "text": (255, 255, 255)},
    "analytical_trust":    {"bg1": (10, 40, 140),  "bg2": (5, 15, 60),   "accent": (100, 180, 255), "text": (255, 255, 255)},
    "intellectual_awe":    {"bg1": (80, 10, 160),  "bg2": (30, 5, 70),   "accent": (220, 180, 255), "text": (255, 255, 255)},
    "anxiety":             {"bg1": (200, 10, 10),  "bg2": (60, 0, 0),    "accent": (255, 80, 0),    "text": (255, 255, 255)},
    "relief":              {"bg1": (10, 140, 60),  "bg2": (5, 50, 20),   "accent": (120, 255, 160), "text": (255, 255, 255)},
    "belonging":           {"bg1": (180, 80, 0),   "bg2": (70, 30, 0),   "accent": (255, 200, 80),  "text": (255, 255, 255)},
    "default":             {"bg1": (20, 20, 80),   "bg2": (10, 10, 30),  "accent": (255, 200, 50),  "text": (255, 255, 255)},
}


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    mac = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    linux = [
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans-{'Bold' if bold else 'Regular'}.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in (mac if sys.platform == "darwin" else []) + linux:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _gradient(c1: tuple, c2: tuple, w: int = THUMB_W, h: int = THUMB_H) -> Image.Image:
    img  = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    for y in range(h):
        t = y / h
        r = int(c1[0] + (c2[0] - c1[0]) * t)
        g = int(c1[1] + (c2[1] - c1[1]) * t)
        b = int(c1[2] + (c2[2] - c1[2]) * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    return img


def _fetch_bg_image(keyword: str) -> Image.Image | None:
    key = os.environ.get("PEXELS_API_KEY", "")
    if not key:
        return None
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": key},
            params={"query": keyword, "per_page": 3, "orientation": "landscape"},
            timeout=10,
        )
        photos = r.json().get("photos", [])
        if not photos:
            return None
        url = photos[0]["src"]["large2x"]
        img = Image.open(BytesIO(requests.get(url, timeout=15).content)).convert("RGB")
        return img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
    except Exception as e:
        log.warning(f"Pexels thumbnail bg '{keyword}': {e}")
        return None


def _draw_title(draw: ImageDraw.Draw, title: str, colour: tuple,
                shadow: bool = True, max_lines: int = 3,
                y_start: int = 180, font_size: int = 88) -> int:
    """Draw word-wrapped bold title. Returns y position after last line."""
    font  = _get_font(font_size, bold=True)
    lines = textwrap.wrap(title, width=22)[:max_lines]
    y     = y_start
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw   = bbox[2] - bbox[0]
        x    = (THUMB_W - tw) // 2
        if shadow:
            draw.text((x + 4, y + 4), line, font=font, fill=(0, 0, 0, 160))
        draw.text((x, y), line, font=font, fill=(*colour, 255))
        y += font_size + 14
    return y


def _stat_badge(draw: ImageDraw.Draw, text: str, accent: tuple,
                x: int, y: int) -> None:
    """Rounded rectangle badge with label (e.g. '97/100 QUALITY')."""
    badge_font = _get_font(36, bold=True)
    bbox = draw.textbbox((0, 0), text, font=badge_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 24, 12
    rx0, ry0 = x, y
    rx1, ry1 = x + tw + pad_x * 2, y + th + pad_y * 2
    draw.rounded_rectangle([rx0, ry0, rx1, ry1], radius=10, fill=(*accent, 230))
    draw.text((rx0 + pad_x, ry0 + pad_y), text, font=badge_font, fill=(0, 0, 0, 255))


def _arrow_cta(draw: ImageDraw.Draw, text: str, accent: tuple, y: int) -> None:
    """'▶ WATCH NOW' style call-to-action bar at bottom."""
    font  = _get_font(44, bold=True)
    label = f"▶  {text}"
    bbox  = draw.textbbox((0, 0), label, font=font)
    tw    = bbox[2] - bbox[0]
    x     = (THUMB_W - tw) // 2
    draw.text((x + 3, y + 3), label, font=font, fill=(0, 0, 0, 140))
    draw.text((x, y), label, font=font, fill=(*accent, 255))


# ── Variant A: Emotion-colour gradient + stat badge ───────────────────────────

def render_variant_a(
    title: str,
    score: int,
    emotion: str,
    visual_keyword: str = "",
) -> Image.Image:
    """
    Bold emotion-gradient background, large title centred, quality-score badge
    top-right, accent bottom bar.
    """
    pal = THUMB_PALETTES.get(emotion, THUMB_PALETTES["default"])

    # Background: Pexels image blurred + gradient overlay
    bg = _fetch_bg_image(visual_keyword or "dramatic cinematic") or _gradient(pal["bg1"], pal["bg2"])
    bg = bg.filter(ImageFilter.GaussianBlur(radius=6))

    # Dark gradient overlay so text always readable
    overlay = _gradient((0, 0, 0), (0, 0, 0))
    overlay.putalpha(160)
    bg = bg.convert("RGBA")
    bg.paste(overlay, mask=overlay.split()[3])
    bg = bg.convert("RGB")

    # Thin top accent bar
    draw = ImageDraw.Draw(bg)
    draw.rectangle([(0, 0), (THUMB_W, 12)], fill=pal["accent"])

    # Emotion label pill (top-left)
    em_font = _get_font(32, bold=True)
    em_label = emotion.replace("_", " ").upper()
    draw.rounded_rectangle([20, 24, 20 + len(em_label) * 19 + 20, 68],
                            radius=8, fill=(*pal["accent"], 200))
    draw.text((30, 28), em_label, font=em_font, fill=(0, 0, 0, 255))

    # Quality badge (top-right)
    if score > 0:
        _stat_badge(draw, f"{score}/100  QUALITY", pal["accent"],
                    x=THUMB_W - 280, y=24)

    # Title (centred, large bold)
    y_after = _draw_title(draw, title, pal["text"],
                          shadow=True, y_start=140, font_size=90)

    # Bottom accent bar
    draw.rectangle([(0, THUMB_H - 10), (THUMB_W, THUMB_H)], fill=pal["accent"])

    # Subtle vignette corners
    for corner_rect, alpha in [
        ([(0, 0), (300, THUMB_H)], 60),
        ([(THUMB_W - 300, 0), (THUMB_W, THUMB_H)], 60),
    ]:
        vign = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
        vd   = ImageDraw.Draw(vign)
        vd.rectangle(corner_rect, fill=(0, 0, 0, alpha))
        bg = Image.alpha_composite(bg.convert("RGBA"), vign).convert("RGB")
        draw = ImageDraw.Draw(bg)

    return bg


# ── Variant B: Dark cinematic + contrasting accent title + arrow CTA ──────────

def render_variant_b(
    title: str,
    score: int,
    emotion: str,
    visual_keyword: str = "",
) -> Image.Image:
    """
    Dark near-black background, large accent-coloured title, '▶ WATCH NOW' CTA,
    left-edge vertical accent bar. More dramatic / clickbait-safe contrast.
    """
    pal = THUMB_PALETTES.get(emotion, THUMB_PALETTES["default"])

    # Near-black background with faint texture from Pexels
    bg = _fetch_bg_image(visual_keyword or "dark abstract") or Image.new("RGB", (THUMB_W, THUMB_H), (8, 8, 18))
    if bg.size != (THUMB_W, THUMB_H):
        bg = bg.resize((THUMB_W, THUMB_H), Image.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=12))

    # Heavy darkening overlay
    dark = Image.new("RGB", (THUMB_W, THUMB_H), (5, 5, 15))
    bg   = Image.blend(bg, dark, alpha=0.72)

    draw = ImageDraw.Draw(bg)

    # Left vertical accent bar (full height)
    draw.rectangle([(0, 0), (14, THUMB_H)], fill=pal["accent"])

    # "MUST WATCH" label (top-left, after bar)
    lf = _get_font(30, bold=True)
    draw.text((28, 22), "MUST WATCH", font=lf, fill=(*pal["accent"], 220))

    # Horizontal rule under label
    draw.line([(28, 60), (THUMB_W - 28, 60)], fill=(*pal["accent"], 100), width=2)

    # Title in accent colour (high contrast on dark bg)
    y_after = _draw_title(draw, title, pal["accent"],
                          shadow=False, y_start=90, font_size=96)

    # Score badge
    if score > 0:
        _stat_badge(draw, f"  {score}/100  ", pal["accent"],
                    x=28, y=y_after + 20)

    # CTA arrow
    _arrow_cta(draw, "WATCH NOW", pal["accent"], y=THUMB_H - 90)

    # Bottom rule
    draw.line([(28, THUMB_H - 100), (THUMB_W - 28, THUMB_H - 100)],
              fill=(*pal["accent"], 100), width=2)

    return bg


# ── Public entry point ────────────────────────────────────────────────────────

def generate_thumbnails(
    title: str,
    score: int = 0,
    emotion: str = "default",
    visual_keyword: str = "",
    out_dir: str = ".",
    prefix: str = "thumb",
) -> tuple[str, str]:
    """
    Render and save Variant A and Variant B thumbnails.
    Returns (path_a, path_b).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    path_a = str(out / f"{prefix}_A.jpg")
    path_b = str(out / f"{prefix}_B.jpg")

    img_a = render_variant_a(title, score, emotion, visual_keyword)
    img_b = render_variant_b(title, score, emotion, visual_keyword)

    img_a.save(path_a, "JPEG", quality=95, optimize=True)
    img_b.save(path_b, "JPEG", quality=95, optimize=True)

    log.info(f"Thumbnails saved → {path_a}, {path_b}")
    return path_a, path_b
