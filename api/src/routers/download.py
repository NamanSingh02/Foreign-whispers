"""POST /api/download — download YouTube video + captions (issue by5)."""

import pathlib

from fastapi import APIRouter

from api.src.core.config import settings
from api.src.core.video_registry import get_video
from api.src.schemas.download import DownloadRequest, DownloadResponse
from api.src.services.download_service import DownloadService

router = APIRouter(prefix="/api")

_download_service = DownloadService(ui_dir=settings.data_dir)


@router.post("/download", response_model=DownloadResponse)
async def download_endpoint(body: DownloadRequest):
    """Download video and captions, returning video_id and caption segments.

    Behavior:
    - If local assets already exist, reuse them and skip YouTube download
    - Otherwise, fall back to the original download/caption-fetch flow
    """
    video_id, title = _download_service.get_video_info(body.url)

    # Preferred display/title stem from registry when available
    entry = get_video(video_id)
    title_stem = entry.title if entry else title.replace(":", "").strip()

    # Also support raw video-id-based naming for locally seeded files
    id_stem = video_id

    videos_dir = settings.videos_dir
    captions_dir = settings.youtube_captions_dir
    videos_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    # Candidate local file names
    title_video_path = videos_dir / f"{title_stem}.mp4"
    title_caption_path = captions_dir / f"{title_stem}.txt"

    id_video_path = videos_dir / f"{id_stem}.mp4"
    id_caption_path = captions_dir / f"{id_stem}.txt"

    # 1) Reuse title-stemmed local assets if present
    if title_video_path.exists() and title_caption_path.exists():
        segments = _download_service.read_caption_segments(title_caption_path)
        return DownloadResponse(
            video_id=video_id,
            title=title,
            caption_segments=segments,
        )

    # 2) Reuse id-stemmed local assets if present
    if id_video_path.exists() and id_caption_path.exists():
        segments = _download_service.read_caption_segments(id_caption_path)
        return DownloadResponse(
            video_id=video_id,
            title=title,
            caption_segments=segments,
        )

    # 3) Otherwise, follow the original routine using the title-based stem
    if not title_video_path.exists():
        _download_service.download_video(body.url, str(videos_dir), title_stem)

    if not title_caption_path.exists():
        _download_service.download_caption(body.url, str(captions_dir), title_stem)

    segments = _download_service.read_caption_segments(title_caption_path)

    return DownloadResponse(
        video_id=video_id,
        title=title,
        caption_segments=segments,
    )