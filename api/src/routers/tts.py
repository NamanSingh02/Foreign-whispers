"""POST /api/tts/{video_id} — TTS with audio-sync endpoint (issue 381)."""

import asyncio
import functools
import json
import pathlib

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.services.tts_service import TTSService

router = APIRouter(prefix="/api")


async def _run_in_threadpool(executor, fn, *args, **kwargs):
    """Run a sync function in the default thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, functools.partial(fn, *args, **kwargs))


def _copy_speakers_from_transcription_if_missing(title: str, translated_path: pathlib.Path) -> None:
    """Ensure translated segments have speaker labels when transcription has them.

    Task 3 writes speaker labels into the transcription JSON. Some translation
    caches may have been created before that, so this function copies speaker
    labels into the translated JSON by segment id/index when needed.
    """
    if not translated_path.exists():
        return

    transcription_path = settings.transcriptions_dir / f"{title}.json"
    if not transcription_path.exists():
        return

    translated_data = json.loads(translated_path.read_text())
    translated_segments = translated_data.get("segments", [])

    if not translated_segments:
        return

    # If speakers are already present in translation, do nothing.
    if any("speaker" in seg for seg in translated_segments):
        return

    transcription_data = json.loads(transcription_path.read_text())
    transcription_segments = transcription_data.get("segments", [])

    if not transcription_segments:
        return

    speaker_by_id = {
        seg.get("id"): seg.get("speaker")
        for seg in transcription_segments
        if seg.get("id") is not None and seg.get("speaker")
    }

    changed = False

    for idx, seg in enumerate(translated_segments):
        speaker = None

        # Prefer id-based matching when available.
        if seg.get("id") in speaker_by_id:
            speaker = speaker_by_id[seg.get("id")]

        # Fallback to index-based matching.
        elif idx < len(transcription_segments):
            speaker = transcription_segments[idx].get("speaker")

        if speaker:
            seg["speaker"] = speaker
            changed = True

    if changed:
        translated_data["segments"] = translated_segments
        translated_path.write_text(json.dumps(translated_data, indent=2))


def _build_speaker_voice_map(translated_path: pathlib.Path) -> dict[str, str]:
    """Map each speaker label to a reference WAV path.

    Uses round-robin assignment from pipeline_data/speakers/{language}/.
    """
    if not translated_path.exists():
        return {}

    data = json.loads(translated_path.read_text())
    language = data.get("language", "es")
    segments = data.get("segments", [])

    speakers = sorted(
        {
            seg.get("speaker")
            for seg in segments
            if seg.get("speaker")
        }
    )

    if not speakers:
        print("[tts] No speaker labels found in translated segments")
        return {}

    # settings.data_dir is usually /app/pipeline_data/api,
    # while reference voices live in /app/pipeline_data/speakers.
    speaker_base_candidates = [
        settings.data_dir / "speakers",
        settings.data_dir.parent / "speakers",
        pathlib.Path("/app/pipeline_data/speakers"),
    ]

    speakers_base = None
    for candidate in speaker_base_candidates:
        if candidate.exists():
            speakers_base = candidate
            break

    if speakers_base is None:
        print(f"[tts] No speakers directory found. Tried: {speaker_base_candidates}")
        return {}

    speaker_dir = speakers_base / language

    if not speaker_dir.exists():
        print(f"[tts] Speaker language directory not found: {speaker_dir}")
        return {}

    wav_files = sorted(speaker_dir.glob("*.wav"))

    if not wav_files:
        print(f"[tts] No reference WAV files found in: {speaker_dir}")
        return {}

    speaker_voice_map = {
        speaker: str(wav_files[i % len(wav_files)].relative_to(speakers_base))
        for i, speaker in enumerate(speakers)
    }

    print(f"[tts] speaker_voice_map={speaker_voice_map}")

    return speaker_voice_map

@router.post("/tts/{video_id}")
async def tts_endpoint(
    video_id: str,
    request: Request,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
    alignment: bool = Query(False),
):
    """Generate TTS audio for a translated transcript.

    *config* is an opaque directory name for caching.
    *alignment* enables temporal alignment (clamped stretch).
    """
    trans_dir = settings.translations_dir
    audio_dir = settings.tts_audio_dir / config
    audio_dir.mkdir(parents=True, exist_ok=True)

    svc = TTSService(
        ui_dir=settings.data_dir,
        tts_engine=None,
    )

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    wav_path = audio_dir / f"{title}.wav"

    if wav_path.exists():
        return {
            "video_id": video_id,
            "audio_path": str(wav_path),
            "config": config,
        }

    source_path = trans_dir / f"{title}.json"

    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Translated transcript not found")

    # Task 5: Ensure translated segments have speaker labels when possible.
    _copy_speakers_from_transcription_if_missing(title, source_path)

    # Task 5: Assign each speaker to a different Chatterbox reference voice.
    speaker_voice_map = _build_speaker_voice_map(source_path)

    await _run_in_threadpool(
        None,
        svc.text_file_to_speech,
        str(source_path),
        str(audio_dir),
        alignment=alignment,
        speaker_voice_map=speaker_voice_map,
    )

    return {
        "video_id": video_id,
        "audio_path": str(wav_path),
        "config": config,
    }


@router.get("/audio/{video_id}")
async def get_audio(
    video_id: str,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
):
    """Stream the TTS-synthesized WAV audio."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    audio_path = settings.tts_audio_dir / config / f"{title}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(str(audio_path), media_type="audio/wav")