"""
youtube_uploader.py
Handles OAuth 2.0 authentication and resumable video upload to YouTube.

First run: opens browser for one-time consent → saves token.json
All subsequent runs: uses token.json (auto-refreshes silently)
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
]

_DIR            = Path(__file__).parent
CLIENT_SECRETS  = _DIR / "client_secrets.json"
TOKEN_FILE      = _DIR / "token.json"


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
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            log.info("Refreshing access token…")
            creds.refresh(Request())
        else:
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
) -> str:
    """
    Upload an MP4 to YouTube using resumable upload (handles large files).
    Returns the video ID on success.
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
    url = f"https://www.youtube.com/watch?v={video_id}"
    log.info(f"Upload complete → {url}")
    return video_id
