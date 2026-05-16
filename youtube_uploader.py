"""
youtube_uploader.py — OAuth 2.0 authentication + resumable YouTube upload.

Scopes include youtube.force-ssl for thumbnail upload and CTR readback.
First run: browser consent → token.json saved.
All subsequent runs: token.json auto-refreshes silently.
"""

import logging
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

log = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",  # thumbnail upload + CTR readback
]

_DIR           = Path(__file__).parent
CLIENT_SECRETS = _DIR / "client_secrets.json"
TOKEN_FILE     = _DIR / "token.json"


def get_authenticated_service():
    """
    Returns an authenticated YouTube Data API v3 client.

    Setup (one-time):
      1. Google Cloud Console → APIs & Services → Credentials
      2. Create OAuth 2.0 Client ID → Desktop app → Download JSON
      3. Rename to client_secrets.json, place in project root
      4. Run pipeline.py once — browser opens for consent
      5. token.json is saved; future runs are fully automatic

    Note: adding youtube.force-ssl scope requires re-authentication on
    first run after upgrading from the old two-scope token.
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
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            log.info("Refreshing access token…")
            try:
                creds.refresh(Request())
            except Exception:
                # Scope change or revocation — force re-auth
                log.info("Token refresh failed (scope change?) — re-authenticating…")
                creds = None

        if not creds:
            log.info("Opening browser for YouTube OAuth consent…")
            flow  = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), SCOPES)
            creds = flow.run_local_server(port=0, prompt="consent")

        TOKEN_FILE.write_text(creds.to_json())
        log.info(f"Token saved → {TOKEN_FILE}")

    return build("youtube", "v3", credentials=creds)


def upload_video(
    youtube,
    video_path: str,
    title: str,
    description: str,
    tags: list[str],
    category_id: str = "22",
    privacy: str = "private",
    thumbnail_path: str | None = None,
) -> str:
    """
    Upload an MP4 to YouTube using resumable upload.
    Returns the video ID on success.
    Optionally uploads a thumbnail if thumbnail_path is provided.
    privacy: "private" | "unlisted" | "public"
    """
    body = {
        "snippet": {
            "title":       title[:100],
            "description": description[:5000],
            "tags":        [t[:500] for t in tags[:500]],
            "categoryId":  category_id,
        },
        "status": {
            "privacyStatus":          privacy,
            "selfDeclaredMadeForKids": False,
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
    log.info(f"Upload complete → https://www.youtube.com/watch?v={video_id}")

    # Thumbnail upload (requires youtube.force-ssl scope + verified channel)
    if thumbnail_path:
        upload_thumbnail(youtube, video_id, thumbnail_path)

    return video_id


def upload_thumbnail(youtube, video_id: str, thumbnail_path: str) -> bool:
    """
    Upload a custom thumbnail for the given video.
    Requires: channel phone-verification AND youtube.force-ssl scope.
    Returns True on success, False on 403 (unverified channel — handled gracefully).
    """
    try:
        media = MediaFileUpload(thumbnail_path, mimetype="image/jpeg", resumable=False)
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=media,
        ).execute()
        log.info(f"  Thumbnail uploaded for video {video_id}")
        return True
    except HttpError as e:
        if e.resp.status == 403:
            log.warning(
                f"  Thumbnail upload 403 — channel not verified yet. "
                "Verify at https://www.youtube.com/verify then re-run."
            )
        else:
            log.warning(f"  Thumbnail upload failed (HTTP {e.resp.status}): {e.reason}")
        return False
    except Exception as e:
        log.warning(f"  Thumbnail upload error: {e}")
        return False
