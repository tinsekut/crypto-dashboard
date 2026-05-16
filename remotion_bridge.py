"""
remotion_bridge.py — Drop-in swap for MoviePy when USE_3D_SCENES=1.

Renders video sections through the Remotion CLI (npm/Node based) for:
  • Smooth 3D camera moves and particle effects
  • React-driven typography animations
  • Lottie/SVG motion graphics

Falls back to MoviePy automatically if Remotion is not installed or fails.

Setup (one-time):
  cd remotion && npm install && npm run build

.env flag:
  USE_3D_SCENES=1          # enable Remotion renderer
  REMOTION_FPS=30          # optional (default 30)
  REMOTION_CONCURRENCY=4   # parallel section renders (default 4)
"""

import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

_DIR            = Path(__file__).parent
_REMOTION_DIR   = _DIR / "remotion"
REMOTION_FPS    = int(os.environ.get("REMOTION_FPS", "30"))
CONCURRENCY     = int(os.environ.get("REMOTION_CONCURRENCY", "4"))


def _remotion_available() -> bool:
    """Check Remotion CLI is installed inside ./remotion/node_modules."""
    cli = _REMOTION_DIR / "node_modules" / ".bin" / "remotion"
    return cli.exists()


def _render_section_remotion(
    section: dict,
    section_index: int,
    out_path: str,
    duration_s: float,
    fps: int = REMOTION_FPS,
) -> bool:
    """
    Render one section clip via Remotion CLI.
    The Remotion composition reads props via --props JSON.
    Returns True on success.
    """
    cli  = str(_REMOTION_DIR / "node_modules" / ".bin" / "remotion")
    props = {
        "sectionIndex":  section_index,
        "name":          section.get("name", ""),
        "narration":     section.get("narration", ""),
        "emotionTarget": section.get("emotion_target", "default"),
        "lowerThird":    section.get("lower_third", ""),
        "peakSentence":  section.get("peak_sentence", ""),
        "visualKeyword": section.get("visual_keyword", ""),
        "durationFrames": int(duration_s * fps),
    }
    props_json = json.dumps(props)

    cmd = [
        "node", cli, "render",
        "VideoSection",          # Remotion composition ID
        out_path,
        "--props", props_json,
        "--fps", str(fps),
        "--frames", f"0-{int(duration_s * fps) - 1}",
        "--concurrency", str(CONCURRENCY),
        "--log", "error",
    ]

    log.info(f"  Remotion rendering section {section_index}: {section.get('name')}")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_REMOTION_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            log.warning(f"  Remotion error: {result.stderr[:400]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        log.warning("  Remotion render timed out (300s)")
        return False
    except FileNotFoundError:
        log.warning("  Remotion CLI not found — falling back to MoviePy")
        return False
    except Exception as e:
        log.warning(f"  Remotion unexpected error: {e}")
        return False


def make_video_remotion(script: dict, output_path: str) -> bool:
    """
    Attempt to render the full video using Remotion.
    Renders each section to a temp MP4, then concatenates with ffmpeg.
    Returns True on success, False to signal fallback to MoviePy.
    """
    if not _remotion_available():
        log.info("Remotion not installed — using MoviePy renderer")
        return False

    sections = script.get("sections", [])
    if not sections:
        return False

    tmp      = Path(tempfile.mkdtemp(prefix="remotion_"))
    clips    = []
    WPM      = 150  # fallback duration estimate

    for i, sec in enumerate(sections):
        out = str(tmp / f"section_{i:02d}.mp4")
        dur = max(5.0, len((sec.get("narration") or "").split()) / WPM * 60)
        ok  = _render_section_remotion(sec, i, out, dur)
        if not ok:
            log.warning(f"  Section {i+1} Remotion render failed — aborting Remotion path")
            return False
        clips.append(out)

    # Concatenate with ffmpeg
    list_file = tmp / "concat.txt"
    list_file.write_text("\n".join(f"file '{c}'" for c in clips))

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            log.warning(f"ffmpeg concat failed: {result.stderr[:400]}")
            return False
        log.info(f"Remotion render complete → {output_path}")
        return True
    except Exception as e:
        log.warning(f"ffmpeg concat error: {e}")
        return False


def render_video(script: dict, output_path: str) -> None:
    """
    Smart renderer: tries Remotion first (if USE_3D_SCENES=1), then MoviePy.
    This is the function pipeline.py should call instead of make_video_from_script
    directly when USE_3D_SCENES is enabled.
    """
    use_3d = os.environ.get("USE_3D_SCENES", "0").strip() == "1"

    if use_3d:
        log.info("USE_3D_SCENES=1 — attempting Remotion renderer…")
        success = make_video_remotion(script, output_path)
        if success:
            return
        log.info("Remotion failed or not installed — falling back to MoviePy")

    from video_maker import make_video_from_script
    make_video_from_script(script, output_path)
