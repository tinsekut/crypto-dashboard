"""
video_maker.py — YouTube Auto-Upload Pipeline (Session 7 — Monetisation Maximisation)

Renders long-form (1920×1080) and Shorts (1080×1920) videos with:
  • Emotion-keyed gradient palettes (10 distinct colour schemes)
  • Silent 2.5s pull-quote cards after each section
  • Retention-bridge overlay in the last 28% of every section
  • Pattern-interrupt visual resets every 35 seconds
  • Rewatch-clue easter egg on Section 1 (subtle, corner text)
  • Shorts: comment-bait, identity-mirror, end-screen-hook visual layers
"""

import os
import sys
import asyncio
import textwrap
import tempfile
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests

log = logging.getLogger(__name__)

# ── Dimensions ────────────────────────────────────────────────────────────────
LONG_W, LONG_H = 1920, 1080
SHORT_W, SHORT_H = 1080, 1920
FPS = 24

# ── Emotion-keyed colour palettes ─────────────────────────────────────────────
# Each palette: bg1/bg2 (gradient top→bottom), accent, text, vignette_colour
EMOTION_PALETTES: dict[str, dict] = {
    "shock": {
        "bg1": (20, 0, 0), "bg2": (60, 0, 10),
        "accent": (255, 50, 50), "text": (255, 240, 240),
        "vignette": (60, 0, 0),
    },
    "dread": {
        "bg1": (5, 5, 20), "bg2": (15, 10, 45),
        "accent": (100, 120, 200), "text": (200, 210, 235),
        "vignette": (5, 5, 30),
    },
    "recognition": {
        "bg1": (35, 22, 5), "bg2": (65, 42, 10),
        "accent": (255, 180, 50), "text": (255, 242, 200),
        "vignette": (40, 20, 0),
    },
    "cognitive_dissonance": {
        "bg1": (30, 0, 40), "bg2": (55, 20, 5),
        "accent": (200, 80, 255), "text": (255, 200, 100),
        "vignette": (40, 0, 40),
    },
    "confusion": {
        "bg1": (5, 28, 38), "bg2": (10, 48, 58),
        "accent": (80, 200, 220), "text": (200, 232, 242),
        "vignette": (0, 30, 45),
    },
    "analytical_trust": {
        "bg1": (5, 10, 42), "bg2": (10, 22, 72),
        "accent": (80, 140, 255), "text": (230, 240, 255),
        "vignette": (5, 10, 50),
    },
    "intellectual_awe": {
        "bg1": (18, 5, 48), "bg2": (38, 10, 80),
        "accent": (200, 150, 255), "text": (255, 240, 200),
        "vignette": (20, 5, 55),
    },
    "anxiety": {
        "bg1": (40, 5, 5), "bg2": (80, 12, 12),
        "accent": (255, 30, 30), "text": (255, 220, 220),
        "vignette": (70, 0, 0),
    },
    "relief": {
        "bg1": (5, 30, 15), "bg2": (10, 58, 25),
        "accent": (80, 220, 120), "text": (220, 255, 232),
        "vignette": (0, 35, 10),
    },
    "belonging": {
        "bg1": (38, 20, 5), "bg2": (68, 35, 10),
        "accent": (255, 160, 60), "text": (255, 242, 210),
        "vignette": (45, 20, 0),
    },
    "default": {
        "bg1": (15, 15, 35), "bg2": (45, 10, 65),
        "accent": (255, 200, 50), "text": (230, 230, 230),
        "vignette": (20, 10, 40),
    },
}


def _palette(emotion: str) -> dict:
    return EMOTION_PALETTES.get(emotion, EMOTION_PALETTES["default"])


# ── Font helper ───────────────────────────────────────────────────────────────

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


# ── Background image ──────────────────────────────────────────────────────────

def _fetch_pexels_image(keyword: str, w: int = LONG_W, h: int = LONG_H) -> Image.Image | None:
    key = os.environ.get("PEXELS_API_KEY", "")
    if not key:
        return None
    orient = "portrait" if h > w else "landscape"
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": key},
            params={"query": keyword, "per_page": 3, "orientation": orient},
            timeout=10,
        )
        photos = r.json().get("photos", [])
        if not photos:
            return None
        url = photos[0]["src"]["large2x"]
        img = Image.open(BytesIO(requests.get(url, timeout=15).content)).convert("RGB")
        return img.resize((w, h), Image.LANCZOS)
    except Exception as e:
        log.warning(f"Pexels '{keyword}': {e}")
        return None


def _gradient_bg(c1: tuple, c2: tuple, w: int = LONG_W, h: int = LONG_H) -> Image.Image:
    img  = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    for y in range(h):
        t = y / h
        r = int(c1[0] + (c2[0] - c1[0]) * t)
        g = int(c1[1] + (c2[1] - c1[1]) * t)
        b = int(c1[2] + (c2[2] - c1[2]) * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    return img


def _add_vignette(img: Image.Image, vignette_colour: tuple) -> Image.Image:
    """Dark radial vignette to increase visual drama."""
    w, h = img.size
    arr  = np.array(img).astype(np.float32)
    cx, cy = w / 2, h / 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    mask = np.clip(dist - 0.5, 0, 1) ** 1.8  # smooth fade starts at 50% radius
    vc   = np.array(vignette_colour, dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1 - mask) + vc[c] * mask
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ── Long-form slide compositor ────────────────────────────────────────────────

def _create_slide(
    bg: Image.Image,
    name: str,
    narration: str,
    lower: str,
    emotion: str = "default",
    rewatch_clue: str = "",
    w: int = LONG_W,
    h: int = LONG_H,
) -> Image.Image:
    pal   = _palette(emotion)
    slide = bg.copy().convert("RGBA")

    # Darkening overlay (heavier for heavy cognitive-load sections)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 165))
    slide   = Image.alpha_composite(slide, overlay)

    # Left-edge vertical accent bar (8px)
    accent_bar = Image.new("RGBA", (8, h), (*pal["accent"], 230))
    slide.paste(accent_bar, (0, 0))

    draw = ImageDraw.Draw(slide)

    # Section name label (top-left, above rule)
    label_font = _get_font(52, bold=True)
    draw.text((28, 52), name.upper(), font=label_font, fill=(*pal["accent"], 255))

    # Horizontal rule under label
    draw.line([(28, 128), (w - 28, 128)], fill=(*pal["accent"], 160), width=2)

    # Narration preview — first 28 words
    body_font = _get_font(42)
    preview   = " ".join(narration.split()[:28]) + ("…" if len(narration.split()) > 28 else "")
    lines     = textwrap.wrap(preview, width=58)[:4]
    y = 158
    for line in lines:
        draw.text((28, y), line, font=body_font, fill=(*pal["text"], 240))
        y += 56

    # Lower-third bar
    draw.rectangle([(0, h - 100), (w, h)], fill=(10, 10, 10, 215))
    lower_font = _get_font(38, bold=True)
    draw.text((28, h - 76), lower, font=lower_font, fill=(255, 255, 255, 255))

    # Rewatch clue — subtle bottom-right easter egg (Section 1 only)
    if rewatch_clue:
        clue_font = _get_font(18)
        clue_text = rewatch_clue[:60]
        draw.text((w - 420, h - 28), clue_text, font=clue_font, fill=(180, 180, 180, 80))

    return slide.convert("RGB")


# ── Pull-quote card ───────────────────────────────────────────────────────────

def _pull_quote_image(
    sentence: str,
    emotion: str,
    w: int = LONG_W,
    h: int = LONG_H,
) -> Image.Image:
    """Full-screen silent pull-quote card — emotion-keyed gradient + letterbox."""
    pal  = _palette(emotion)
    bg   = _gradient_bg(pal["bg1"], pal["bg2"], w, h)
    bg   = _add_vignette(bg, pal["vignette"])
    draw = ImageDraw.Draw(bg)

    # Cinematic letterbox bars (top/bottom 9%)
    bar_h = int(h * 0.09)
    draw.rectangle([(0, 0), (w, bar_h)], fill=(0, 0, 0))
    draw.rectangle([(0, h - bar_h), (w, h)], fill=(0, 0, 0))

    # Left-edge vertical accent bar
    draw.rectangle([(0, bar_h), (8, h - bar_h)], fill=(*pal["accent"], 255))

    # Quote text — centred between letterboxes
    quote_font  = _get_font(58, bold=True)
    max_w_chars = int(w * 0.85 / 30)  # rough char budget
    lines = textwrap.wrap(sentence, width=max_w_chars)[:4]
    total_h = len(lines) * 72
    y = (h - total_h) // 2

    # Thin accent rule above text
    draw.line([(int(w * 0.1), y - 24), (int(w * 0.9), y - 24)],
              fill=(*pal["accent"], 180), width=2)

    for line in lines:
        bbox  = draw.textbbox((0, 0), line, font=quote_font)
        tw    = bbox[2] - bbox[0]
        draw.text(((w - tw) // 2, y), line, font=quote_font, fill=(*pal["text"], 255))
        y += 72

    # Thin accent rule below text
    draw.line([(int(w * 0.1), y + 8), (int(w * 0.9), y + 8)],
              fill=(*pal["accent"], 180), width=2)

    return bg


# ── Retention-bridge overlay ──────────────────────────────────────────────────

def _retention_bridge_image(bridge_text: str, emotion: str, w: int, h: int) -> Image.Image:
    """Semi-transparent 'NEXT:' overlay rendered as RGBA image for compositing."""
    pal  = _palette(emotion)
    img  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    prefix_font = _get_font(30, bold=True)
    body_font   = _get_font(28)

    # Wrap bridge text to ≤54 chars per line, 2 lines max
    lines = textwrap.wrap(bridge_text, width=54)[:2]

    # Background pill behind the text
    x0   = int(w * 0.04)
    y0   = h - 155
    pill_h = 36 + len(lines) * 36
    draw.rectangle([(x0 - 8, y0 - 6), (int(w * 0.7), y0 + pill_h)],
                   fill=(0, 0, 0, 160))

    draw.text((x0, y0), "NEXT:", font=prefix_font, fill=(*pal["accent"], 220))
    y = y0 + 32
    for line in lines:
        draw.text((x0, y), line, font=body_font, fill=(230, 230, 230, 200))
        y += 34

    return img


# ── Pattern-interrupt clip ────────────────────────────────────────────────────

def _pattern_interrupt_image(emotion: str, w: int, h: int) -> Image.Image:
    """Hard 0.5s visual reset — inverted palette flash to break monotony."""
    pal = _palette(emotion)
    # Invert the accent colour for maximum visual discontinuity
    inv = tuple(255 - c for c in pal["accent"])
    bg  = _gradient_bg(pal["bg2"], pal["bg1"], w, h)  # reversed gradient
    draw = ImageDraw.Draw(bg)
    # Wide horizontal flash bar across the middle
    bar_y = int(h * 0.42)
    draw.rectangle([(0, bar_y), (w, bar_y + int(h * 0.16))], fill=(*inv, 180))
    return bg


# ── TTS ───────────────────────────────────────────────────────────────────────

async def _edge_tts_async(text: str, path: str, voice: str) -> None:
    import edge_tts
    await edge_tts.Communicate(text, voice).save(path)


def _tts(text: str, path: str) -> bool:
    voice = os.environ.get("TTS_VOICE", "en-US-JennyNeural")
    try:
        import edge_tts  # noqa: F401
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_edge_tts_async(text, path, voice))
        finally:
            loop.close()
        return True
    except ImportError:
        pass
    except Exception as e:
        log.warning(f"edge-tts failed ({e}), trying gTTS…")

    try:
        from gtts import gTTS
        gTTS(text=text, lang="en", slow=False).save(path)
        return True
    except Exception as e:
        log.error(f"TTS completely failed: {e}")
        return False


# ── Long-form video renderer ──────────────────────────────────────────────────

def make_video_from_script(script: dict, output_path: str) -> None:
    """
    Render a 1920×1080 MP4 from a 10-section psychological script.
    Includes: emotion palettes, pull-quote cards, retention bridges, pattern interrupts,
    and a rewatch-clue easter egg on Section 1.
    """
    from moviepy.editor import (
        ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip,
    )

    sections = script.get("sections", [])
    if not sections:
        raise ValueError("Script has no sections")

    tmp       = Path(tempfile.mkdtemp(prefix="ytgen_long_"))
    all_clips = []          # final sequence including pull-quotes + pattern interrupts
    cumulative_dur = 0.0    # track time for pattern-interrupt insertion
    INTERRUPT_EVERY = 35.0  # seconds between hard visual resets

    for i, sec in enumerate(sections):
        name         = sec.get("name", f"Section {i+1}")
        narration    = sec.get("narration", "Welcome.")
        keyword      = sec.get("visual_keyword", "technology abstract")
        lower        = sec.get("lower_third", name)
        emotion      = sec.get("emotion_target", "default")
        peak         = (sec.get("peak_sentence") or "").strip()
        rewatch_clue = sec.get("rewatch_clue", "") if i == 0 else ""
        bridge       = (sec.get("retention_bridge") or "").strip()
        pal          = _palette(emotion)

        log.info(f"  [{i+1}/{len(sections)}] {name} ({emotion})")

        # ── Background + slide ───────────────────────────────────────────────
        bg    = _fetch_pexels_image(keyword) or _gradient_bg(pal["bg1"], pal["bg2"])
        bg    = _add_vignette(bg, pal["vignette"])
        slide = _create_slide(bg, name, narration, lower, emotion, rewatch_clue)
        slide_path = str(tmp / f"slide_{i:02d}.png")
        slide.save(slide_path)

        # ── TTS audio ────────────────────────────────────────────────────────
        audio_path = str(tmp / f"audio_{i:02d}.mp3")
        has_audio  = _tts(narration, audio_path)

        if has_audio:
            audio_clip = AudioFileClip(audio_path)
            dur        = audio_clip.duration
        else:
            audio_clip = None
            dur = max(5.0, len(narration.split()) / 2.5)

        # ── Pattern interrupt — insert flash clip if we've hit the 35s mark ─
        if cumulative_dur > 0 and (cumulative_dur // INTERRUPT_EVERY) > (
            (cumulative_dur - dur) // INTERRUPT_EVERY
        ):
            pi_img  = _pattern_interrupt_image(emotion, LONG_W, LONG_H)
            pi_path = str(tmp / f"pattern_interrupt_{i:02d}.png")
            pi_img.save(pi_path)
            pi_clip = ImageClip(pi_path).set_duration(0.5)
            all_clips.append(pi_clip)
            log.info(f"    ↳ Pattern interrupt at {cumulative_dur:.0f}s")

        # ── Main section clip (with retention bridge overlay in last 28%) ───
        img_clip = ImageClip(slide_path).set_duration(dur)
        if audio_clip:
            img_clip = img_clip.set_audio(audio_clip)
        img_clip = img_clip.fadein(0.4).fadeout(0.4)

        # Composite retention bridge in the last 28% of the clip
        if bridge and dur > 3.0:
            bridge_img  = _retention_bridge_image(bridge, emotion, LONG_W, LONG_H)
            bridge_path = str(tmp / f"bridge_{i:02d}.png")
            bridge_img.save(bridge_path)
            bridge_start = dur * 0.72
            bridge_clip  = (
                ImageClip(bridge_path, ismask=False)
                .set_start(bridge_start)
                .set_duration(dur - bridge_start)
            )
            img_clip = CompositeVideoClip([img_clip, bridge_clip])

        all_clips.append(img_clip)
        cumulative_dur += dur

        # ── Pull-quote card (silent 2.5s) ─────────────────────────────────
        if len(peak.split()) >= 4:
            pq_img  = _pull_quote_image(peak, emotion)
            pq_path = str(tmp / f"pullquote_{i:02d}.png")
            pq_img.save(pq_path)
            pq_clip = ImageClip(pq_path).set_duration(2.5)
            all_clips.append(pq_clip)
            cumulative_dur += 2.5

    final = concatenate_videoclips(all_clips, method="compose")
    log.info(f"Rendering {final.duration:.0f}s long-form → {output_path}")
    final.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(tmp / "tmp_audio.m4a"),
        remove_temp=True,
        logger=None,
    )
    log.info("Long-form render complete")


# ── Shorts renderer (1080×1920 vertical, ≤60s) ───────────────────────────────

def _shorts_slide(
    bg: Image.Image,
    overlay_text: str,
    lower_text: str,
    emotion: str,
    special_text: str = "",
    special_colour: tuple = (255, 255, 80),
) -> Image.Image:
    """Vertical 1080×1920 slide for Shorts."""
    pal   = _palette(emotion)
    slide = bg.copy().convert("RGBA")
    overlay = Image.new("RGBA", (SHORT_W, SHORT_H), (0, 0, 0, 155))
    slide   = Image.alpha_composite(slide, overlay)

    # Top accent bar (horizontal, full width, 8px)
    bar = Image.new("RGBA", (SHORT_W, 8), (*pal["accent"], 230))
    slide.paste(bar, (0, 0))

    draw = ImageDraw.Draw(slide)

    # Main overlay text (large, centred, upper third)
    if overlay_text:
        ov_font = _get_font(72, bold=True)
        lines   = textwrap.wrap(overlay_text, width=20)[:3]
        y       = int(SHORT_H * 0.12)
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=ov_font)
            tw   = bbox[2] - bbox[0]
            draw.text(((SHORT_W - tw) // 2, y), line, font=ov_font,
                      fill=(*pal["text"], 255))
            y += 84

    # Special text (identity_mirror / comment_bait) — mid-screen
    if special_text:
        sp_font = _get_font(52, bold=True)
        lines   = textwrap.wrap(special_text, width=24)[:2]
        y       = int(SHORT_H * 0.48)
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=sp_font)
            tw   = bbox[2] - bbox[0]
            draw.text(((SHORT_W - tw) // 2, y), line, font=sp_font,
                      fill=(*special_colour, 240))
            y += 62

    # Lower-third bar
    bar_y = SHORT_H - 120
    draw.rectangle([(0, bar_y), (SHORT_W, SHORT_H)], fill=(10, 10, 10, 220))
    lo_font = _get_font(40, bold=True)
    bbox    = draw.textbbox((0, 0), lower_text, font=lo_font)
    tw      = bbox[2] - bbox[0]
    draw.text(((SHORT_W - tw) // 2, bar_y + 20), lower_text, font=lo_font,
              fill=(255, 255, 255, 255))

    return slide.convert("RGB")


def make_shorts_from_script(shorts_script: dict, output_path: str) -> None:
    """
    Render a 1080×1920 vertical MP4 Short (≤60s) with Zeigarnik visual layers:
    comment_bait, identity_mirror, and end_screen_hook overlays.
    """
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

    chunks       = shorts_script.get("chunks", [])
    narration    = shorts_script.get("narration", "")
    hook         = shorts_script.get("hook_sentence", "")
    comment_bait = shorts_script.get("comment_bait", "")
    id_mirror    = shorts_script.get("identity_mirror", "")
    end_hook     = shorts_script.get("end_screen_hook", "")

    if not chunks and narration:
        # Fallback: split narration into 4 even chunks
        words  = narration.split()
        n      = max(1, len(words) // 4)
        chunks = [
            {"text": " ".join(words[i:i+n]), "visual_keyword": "technology", "overlay_text": ""}
            for i in range(0, len(words), n)
        ][:4]

    if not chunks:
        raise ValueError("Shorts script has no chunks and no narration")

    tmp       = Path(tempfile.mkdtemp(prefix="ytgen_short_"))
    all_clips = []
    total_dur = 0.0

    for i, chunk in enumerate(chunks):
        text         = chunk.get("text", "")
        keyword      = chunk.get("visual_keyword", "technology")
        overlay_text = chunk.get("overlay_text", "")

        # Inject hook text on first chunk, identity mirror on middle chunk
        special      = ""
        special_col  = (255, 255, 80)
        if i == 0 and hook:
            special = hook
        elif i == len(chunks) // 2 and id_mirror:
            special     = id_mirror
            special_col = (255, 180, 80)

        bg    = _fetch_pexels_image(keyword, SHORT_W, SHORT_H)
        pal   = _palette("shock" if i == 0 else "analytical_trust")
        if bg is None:
            bg = _gradient_bg(pal["bg1"], pal["bg2"], SHORT_W, SHORT_H)
        bg    = _add_vignette(bg, pal["vignette"])

        slide = _shorts_slide(bg, overlay_text, hook if i == 0 else "", "analytical_trust",
                               special, special_col)
        slide_path = str(tmp / f"short_slide_{i:02d}.png")
        slide.save(slide_path)

        audio_path = str(tmp / f"short_audio_{i:02d}.mp3")
        has_audio  = _tts(text, audio_path)

        if has_audio:
            audio_clip = AudioFileClip(audio_path)
            dur        = audio_clip.duration
        else:
            audio_clip = None
            dur = max(3.0, len(text.split()) / 2.5)

        total_dur += dur
        if total_dur > 59.0:
            dur = max(1.0, dur - (total_dur - 59.0))

        img_clip = ImageClip(slide_path).set_duration(dur)
        if audio_clip:
            audio_clip = audio_clip.subclip(0, min(dur, audio_clip.duration))
            img_clip   = img_clip.set_audio(audio_clip)
        all_clips.append(img_clip)

    # End-screen hook card (2s, silent)
    if end_hook:
        pal      = _palette("belonging")
        end_bg   = _gradient_bg(pal["bg1"], pal["bg2"], SHORT_W, SHORT_H)
        end_slide = _shorts_slide(end_bg, end_hook, "Watch the full breakdown ↓",
                                   "belonging")
        end_path = str(tmp / "short_end.png")
        end_slide.save(end_path)
        all_clips.append(ImageClip(end_path).set_duration(2.0))

    # comment_bait card (1.5s, silent) — appended after end hook
    if comment_bait:
        pal    = _palette("recognition")
        cb_bg  = _gradient_bg(pal["bg1"], pal["bg2"], SHORT_W, SHORT_H)
        cb_slide = _shorts_slide(cb_bg, comment_bait, "⬇ Comment below ⬇",
                                  "recognition", special_colour=(80, 220, 255))
        cb_path = str(tmp / "short_comment_bait.png")
        cb_slide.save(cb_path)
        all_clips.append(ImageClip(cb_path).set_duration(1.5))

    final = concatenate_videoclips(all_clips, method="compose")
    # Hard-cap at 59s to stay within YouTube Shorts limit
    if final.duration > 59.0:
        final = final.subclip(0, 59.0)

    log.info(f"Rendering {final.duration:.1f}s Short → {output_path}")
    final.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(tmp / "short_tmp_audio.m4a"),
        remove_temp=True,
        logger=None,
    )
    log.info("Short render complete")
