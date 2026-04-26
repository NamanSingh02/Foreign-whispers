"""POST /api/diarize/{video_id} — speaker diarization (issue fw-lua)."""

import asyncio
import json
import subprocess

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.schemas.diarize import DiarizeResponse
from api.src.services.alignment_service import AlignmentService
from foreign_whispers.diarization import assign_speakers

router = APIRouter(prefix="/api")

_alignment_service = AlignmentService(settings=settings)


def _merge_speakers_into_transcription(title: str, diar_segments: list[dict]) -> None:
    """Update transcription JSON so each segment has a speaker label."""
    transcription_path = settings.transcriptions_dir / f"{title}.json"

    if not transcription_path.exists():
        return

    trans_data = json.loads(transcription_path.read_text())

    merged_segments = assign_speakers(
        trans_data.get("segments", []),
        diar_segments,
    )

    trans_data["segments"] = merged_segments
    transcription_path.write_text(json.dumps(trans_data, indent=2))


@router.post("/diarize/{video_id}", response_model=DiarizeResponse)
async def diarize_endpoint(video_id: str):
    """Run speaker diarization on a video's audio track.

    Steps:
    1. Extract audio from video via ffmpeg
    2. Run pyannote diarization
    3. Cache and return speaker segments
    4. Merge speaker labels into transcription JSON
    """
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    diar_dir = settings.diarization_dir
    diar_dir.mkdir(parents=True, exist_ok=True)
    diar_path = diar_dir / f"{title}.json"

    # Return cached result, but still merge speakers into transcription JSON.
    if diar_path.exists():
        data = json.loads(diar_path.read_text())
        diar_segments = data.get("segments", [])

        _merge_speakers_into_transcription(title, diar_segments)

        return DiarizeResponse(
            video_id=video_id,
            speakers=data.get("speakers", []),
            segments=diar_segments,
            skipped=True,
        )

    # Step 1: Extract audio from video
    video_path = settings.videos_dir / f"{title}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    audio_path = diar_dir / f"{title}.wav"

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-y",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Audio extraction failed: {exc.stderr}",
        )

    # Step 2: Run diarization
    loop = asyncio.get_event_loop()
    diar_segments = await loop.run_in_executor(
        None,
        _alignment_service.diarize,
        str(audio_path),
    )

    # Step 3: Extract unique speakers
    speakers = sorted(set(s["speaker"] for s in diar_segments))

    # Step 4: Cache result
    result = {
        "speakers": speakers,
        "segments": diar_segments,
    }
    diar_path.write_text(json.dumps(result, indent=2))

    # Step 5: Merge diarized speaker labels into the transcription JSON.
    _merge_speakers_into_transcription(title, diar_segments) # Task 3

    # Step 6: Return DiarizeResponse
    return DiarizeResponse(
        video_id=video_id,
        speakers=speakers,
        segments=diar_segments,
        skipped=False,
    )