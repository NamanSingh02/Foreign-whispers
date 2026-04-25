"""Speaker diarization using pyannote.audio.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M2-align).

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-3.1 licence on HuggingFace
and providing an HF token.  Returns empty list with a warning if the dep is
absent or the token is missing.
"""
import logging

logger = logging.getLogger(__name__)


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    try:
        from pyannote.audio import Pipeline
    except (ImportError, TypeError):
        logger.warning("pyannote.audio not installed — returning empty diarization.")
        return []

    try:
        pipeline    = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(audio_path)
        return [
            {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    except Exception as exc:
        logger.warning("Diarization failed for %s: %s", audio_path, exc)
        return []

def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
) -> list[dict]:
    """Assign a speaker label to each transcription segment.

    For each segment, finds the diarization interval with the greatest
    temporal overlap and copies its speaker label. If diarization is
    empty, all segments default to ``SPEAKER_00``.
    """
    if not segments:
        return []

    if not diarization:
        out = []
        for seg in segments:
            new_seg = dict(seg)
            new_seg["speaker"] = "SPEAKER_00"
            out.append(new_seg)
        return out

    out = []

    for seg in segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])

        best_speaker = "SPEAKER_00"
        best_overlap = -1.0

        for diar in diarization:
            diar_start = float(diar["start_s"])
            diar_end = float(diar["end_s"])
            speaker = diar["speaker"]

            overlap = max(0.0, min(seg_end, diar_end) - max(seg_start, diar_start))

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        new_seg = dict(seg)
        new_seg["speaker"] = best_speaker
        out.append(new_seg)

    return out
