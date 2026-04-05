"""Download images from Google Drive using service account credentials."""

import io
import json
import os
import tempfile

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

_service = None


def get_drive_service():
    global _service
    if _service is not None:
        return _service

    key_json = os.environ["GOOGLE_DRIVE_SERVICE_ACCOUNT_KEY"]
    creds_info = json.loads(key_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    _service = build("drive", "v3", credentials=creds)
    return _service


def download_image(drive_file_id: str, dest_path: str = None) -> str:
    """Download a file from Drive. Returns path to the downloaded file."""
    service = get_drive_service()

    if dest_path is None:
        fd, dest_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)

    request = service.files().get_media(
        fileId=drive_file_id, supportsAllDrives=True
    )
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    return dest_path
