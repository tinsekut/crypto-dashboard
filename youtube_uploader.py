"""
youtube_uploader.py
Handles OAuth 2.0 authentication and resumable video upload to YouTube.

First run: opens browser for one-time consent → saves token.json
All subsequent runs: uses token.json (auto-refreshes silently)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

log = logging.getLogger(__name__)

UPLOAD_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",   # required for captions.insert
]
ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"

# Backward-compatible name used by older helpers. Keep this upload-only so
# optional CTR analytics can never break core upload token refresh.
SCOPES = UPLOAD_SCOPES

_DIR            = Path(__file__).parent
CLIENT_SECRETS  = _DIR / "client_secrets.json"
TOKEN_FILE      = _DIR / "token.json"


@dataclass
class UploadResult:
    video_id: str
    thumbnail_uploaded: bool = False
    captions_uploaded: bool = False
    thumbnail_error: str = ""
    captions_error: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    def __str__(self) -> str:
        return self.video_id


def _http_error_message(e: HttpError) -> str:
    try:
        import json
        payload = json.loads(e.content.decode("utf-8"))
        return payload.get("error", {}).get("message") or str(e)
    except Exception:
        return getattr(e, "reason", "") or str(e)


def _thumbnail_permission_help() -> str:
    return (
        "YouTube accepted the video but rejected the custom thumbnail. "
        "This usually means the channel is not eligible for custom thumbnails "
        "or the authenticated user is not the channel owner/manager with upload "
        "permissions. Verify the channel at https://www.youtube.com/verify, "
        "then enable/confirm custom thumbnails in YouTube Studio. If you manage "
        "multiple channels, delete token.json and re-authenticate with the correct "
        "Google account/channel."
    )


def get_authenticated_service():
    """
    Returns an authenticated YouTube Data API v3 client.

    Setup (one-time):
      1. Google Cloud Console → APIs & Services → Credentials
      2. Create OAuth 2.0 Client ID → Desktop app → Download JSON
      3. Rename the downloaded file to client_secrets.json
      4. Place it in the project root (same folder as this file)
      5. Run pipeline.py once — browser opens for consent
      6. token.json is saved; future runs are fully automatic
    """
    if not CLIENT_SECRETS.exists():
        raise FileNotFoundError(
            "\n\n  ❌  client_secrets.json not found.\n\n"
            "  Steps to fix:\n"
            "  1. Go to https://console.cloud.google.com/\n"
            "  2. APIs & Services → Credentials\n"
            "  3. Create OAuth 2.0 Client ID (type: Desktop app)\n"
            "  4. Download JSON → rename to client_secrets.json\n"
            "  5. Place it in:  " + str(_DIR) + "\n"
        )

    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), UPLOAD_SCOPES)
        # If the saved token doesn't cover the full scope set (e.g. force-ssl was
        # added after the first login), invalidate it so we re-authenticate below.
        try:
            import json
            saved_scopes = set((json.loads(TOKEN_FILE.read_text()).get("scopes") or []))
        except Exception:
            saved_scopes = set(creds.scopes or [])
        if not all(s in saved_scopes for s in UPLOAD_SCOPES):
            log.info("Token scopes outdated — re-authenticating to grant new permissions…")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            log.info("Refreshing access token…")
            try:
                creds.refresh(Request())
            except Exception as e:
                log.warning(f"Token refresh failed ({e}); re-authenticating…")
                creds = None
        if not creds or not creds.valid:
            log.info("Opening browser for YouTube OAuth consent…")
            flow  = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), UPLOAD_SCOPES)
            creds = flow.run_local_server(port=0, prompt="consent")
        TOKEN_FILE.write_text(creds.to_json())
        log.info(f"Token saved → {TOKEN_FILE}")

    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def upload_video(
    youtube,
    video_path: str,
    title: str,
    description: str,
    tags: list[str],
    category_id: str    = "22",
    privacy: str        = "private",
    thumbnail_path: str = "",
    srt_path: str       = "",
) -> UploadResult:
    """
    Upload an MP4 to YouTube using resumable upload (handles large files).
    Optionally uploads a custom thumbnail and SRT caption track.
    Returns the video ID on success.
    privacy: "private" | "unlisted" | "public"
    """
    body = {
        "snippet": {
            "title":               title[:100],
            "description":         description[:5000],
            "tags":                [t[:500] for t in tags[:500]],
            "categoryId":          category_id,
            "defaultLanguage":     "en",        # video metadata language
            "defaultAudioLanguage": "en",       # enables auto-caption suggestion
        },
        "status": {
            "privacyStatus":          privacy,
            "selfDeclaredMadeForKids": False,   # ensures comments are open
            "embeddable":             True,
            "publicStatsViewable":    True,
        },
    }

    media = MediaFileUpload(
        video_path,
        mimetype="video/mp4",
        resumable=True,
        chunksize=5 * 1024 * 1024,
    )

    insert_request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=media,
    )

    log.info(f"Starting upload: {title}")
    response = None
    while response is None:
        try:
            status, response = insert_request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                log.info(f"  Upload: {pct}%")
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504):
                log.warning(f"Transient HTTP {e.resp.status} — retrying…")
                continue
            raise

    video_id = response["id"]
    result = UploadResult(video_id=video_id)
    log.info(f"Upload complete → {result.url}")

    # ── Upload custom thumbnail ───────────────────────────────────────────
    if thumbnail_path and Path(thumbnail_path).exists():
        try:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_path, mimetype="image/jpeg"),
            ).execute()
            log.info(f"  Thumbnail uploaded → {thumbnail_path}")
            result.thumbnail_uploaded = True
        except HttpError as e:
            msg = _http_error_message(e)
            result.thumbnail_error = f"HTTP {e.resp.status}: {msg}"
            log.warning(f"  Thumbnail upload failed ({result.thumbnail_error})")
            if e.resp.status == 403:
                help_msg = _thumbnail_permission_help()
                result.warnings.append(help_msg)
                log.warning(f"  {help_msg}")
        except Exception as e:
            result.thumbnail_error = str(e)
            log.warning(f"  Thumbnail upload failed: {e}")

    # ── Upload SRT captions ───────────────────────────────────────────────
    # Requires youtube.force-ssl scope.  The track is set to non-draft so it
    # appears immediately; YouTube also runs its own auto-sync on top of this.
    if srt_path and Path(srt_path).exists():
        try:
            youtube.captions().insert(
                part="snippet",
                body={
                    "snippet": {
                        "videoId":  video_id,
                        "language": "en",
                        "name":     "English",
                        "isDraft":  False,
                    }
                },
                media_body=MediaFileUpload(srt_path, mimetype="application/x-subrip"),
            ).execute()
            log.info(f"  Captions uploaded → {srt_path}")
            result.captions_uploaded = True
        except HttpError as e:
            msg = _http_error_message(e)
            result.captions_error = f"HTTP {e.resp.status}: {msg}"
            log.warning(f"  Caption upload failed ({result.captions_error})")
        except Exception as e:
            result.captions_error = str(e)
            log.warning(f"  Caption upload failed: {e}")
    else:
        log.info("  No SRT file found — YouTube will use auto-generated captions")

    return result
