"""
video_maker.py
Converts a structured script dict into a 1080p MP4.
Pipeline: background image (Pexels or gradient) → Pillow slide → TTS audio → MoviePy clip → concat.
"""

import os
import sys
import asyncio
import textwrap
import tempfile
import logging
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import requests

log = logging.getLogger(__name__)

WIDTH, HEIGHT = 1920, 1080
FPS           = 24


# ── Font helper ──────────────────────────────────────────────────────────────

def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    mac_fonts = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    linux_fonts = [
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans-{'Bold' if bold else 'Regular'}.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    candidates = (mac_fonts if sys.platform == "darwin" else []) + linux_fonts
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ── Background image ─────────────────────────────────────────────────────────

def _fetch_pexels_image(keyword: str) -> Image.Image | None:
    pexels_key = os.environ.get("PEXELS_API_KEY", "")
    if not pexels_key:
        return None
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": pexels_key},
            params={"query": keyword, "per_page": 3, "orientation": "landscape"},
            timeout=10,
        )
        photos = r.json().get("photos", [])
        if not photos:
            return None
        url = photos[0]["src"]["large2x"]
        img = Image.open(BytesIO(requests.get(url, timeout=15).content)).convert("RGB")
        return img.resize((WIDTH, HEIGHT), Image.LANCZOS)
    except Exception as e:
        log.warning(f"Pexels '{keyword}': {e}")
        return None


def _gradient_bg(c1=(15, 15, 35), c2=(45, 10, 65)) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)
    for y in range(HEIGHT):
        t = y / HEIGHT
        r = int(c1[0] + (c2[0] - c1[0]) * t)
        g = int(c1[1] + (c2[1] - c1[1]) * t)
        b = int(c1[2] + (c2[2] - c1[2]) * t)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))
    return img


# ── Slide compositor ─────────────────────────────────────────────────────────

def _create_slide(bg: Image.Image, name: str, narration: str, lower: str) -> Image.Image:
    slide = bg.copy().convert("RGBA")

    # Semi-transparent overlay so text is readable over any image
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 170))
    slide = Image.alpha_composite(slide, overlay)
    draw  = ImageDraw.Draw(slide)

    # Section label (top-left)
    label_font = _get_font(56, bold=True)
    draw.text((80, 60), name.upper(), font=label_font, fill=(255, 200, 50))

    # Horizontal rule under label
    draw.line([(80, 140), (WIDTH - 80, 140)], fill=(255, 200, 50, 180), width=3)

    # First ~25 words of narration as subtitle body
    body_font = _get_font(44)
    preview   = " ".join(narration.split()[:25]) + ("…" if len(narration.split()) > 25 else "")
    lines     = textwrap.wrap(preview, width=55)[:4]
    y = 180
    for line in lines:
        draw.text((80, y), line, font=body_font, fill=(230, 230, 230))
        y += 60

    # Lower-third bar
    draw.rectangle([(0, HEIGHT - 110), (WIDTH, HEIGHT)], fill=(15, 15, 15, 220))
    lower_font = _get_font(40, bold=True)
    draw.text((80, HEIGHT - 85), lower, font=lower_font, fill=(255, 255, 255))

    return slide.convert("RGB")


# ── TTS ───────────────────────────────────────────────────────────────────────

async def _edge_tts(text: str, path: str, voice: str) -> None:
    import edge_tts
    await edge_tts.Communicate(text, voice).save(path)


def _tts(text: str, path: str) -> bool:
    voice = os.environ.get("TTS_VOICE", "en-US-JennyNeural")
    # Try edge-tts (high quality Microsoft neural voices)
    try:
        import edge_tts  # noqa: F401
        # Use a dedicated event loop to avoid conflicts with any running loop
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_edge_tts(text, path, voice))
        finally:
            loop.close()
        return True
    except ImportError:
        pass
    except Exception as e:
        log.warning(f"edge-tts failed ({e}), trying gTTS…")

    # Fallback: gTTS
    try:
        from gtts import gTTS
        gTTS(text=text, lang="en", slow=False).save(path)
        return True
    except Exception as e:
        log.error(f"TTS completely failed: {e}")
        return False


# ── Main entry ───────────────────────────────────────────────────────────────

def make_video_from_script(script: dict, output_path: str) -> None:
    """
    Build an MP4 from a structured script dict.
    Expected schema:
      script["sections"] = list of {name, narration, visual_keyword, lower_third}
    """
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

    sections = script.get("sections", [])
    if not sections:
        raise ValueError("Script has no sections")

    tmp = Path(tempfile.mkdtemp(prefix="ytgen_"))
    clips = []

    for i, sec in enumerate(sections):
        name      = sec.get("name", f"Section {i+1}")
        narration = sec.get("narration", "Welcome.")
        keyword   = sec.get("visual_keyword", "technology abstract")
        lower     = sec.get("lower_third", name)

        log.info(f"  Rendering [{i+1}/{len(sections)}] {name}")

        # Background
        bg    = _fetch_pexels_image(keyword) or _gradient_bg()
        slide = _create_slide(bg, name, narration, lower)
        slide_path = str(tmp / f"slide_{i:02d}.png")
        slide.save(slide_path)

        # Audio
        audio_path = str(tmp / f"audio_{i:02d}.mp3")
        has_audio  = _tts(narration, audio_path)

        if has_audio:
            audio_clip = AudioFileClip(audio_path)
            duration   = audio_clip.duration
        else:
            audio_clip = None
            # Estimate 150 wpm reading speed as fallback duration
            duration   = max(5.0, len(narration.split()) / 2.5)

        img_clip = ImageClip(slide_path).set_duration(duration)
        if audio_clip:
            img_clip = img_clip.set_audio(audio_clip)

        # Fade in/out for polish
        img_clip = img_clip.fadein(0.5).fadeout(0.5)
        clips.append(img_clip)

    final = concatenate_videoclips(clips, method="compose")
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
    log.info("Video render complete")
