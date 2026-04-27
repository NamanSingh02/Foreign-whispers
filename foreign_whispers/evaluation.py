"""Clip-level alignment and dubbing quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import math
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _overlap_count(aligned: list[AlignedSegment]) -> int:
    return sum(
        1 for prev, cur in zip(aligned, aligned[1:])
        if cur.scheduled_start < prev.scheduled_end - 1e-6
    )


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip."""
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
            "n_overlaps":                0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = aligned[-1].scheduled_end - aligned[-1].original_end if aligned else 0.0

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
        "n_overlaps":                _overlap_count(aligned),
    }


def _timing_score(report: dict) -> float:
    mae = report.get("mean_abs_duration_error_s", 0.0)
    severe = report.get("pct_severe_stretch", 0.0) / 100.0
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    overlaps = report.get("n_overlaps", 0)

    mae_score = _clamp01(1.0 - mae / 2.0)
    severe_score = _clamp01(1.0 - severe)
    drift_score = _clamp01(1.0 - drift / 5.0)
    overlap_score = _clamp01(1.0 - overlaps / 5.0)

    return round(0.35 * mae_score + 0.25 * severe_score + 0.25 * drift_score + 0.15 * overlap_score, 3)


def _intelligibility_score(align_report: dict | None) -> float:
    if not align_report:
        return 0.5

    wer = align_report.get("stt_roundtrip_wer")
    if wer is None:
        wer = align_report.get("word_error_rate")
    if wer is not None:
        return round(_clamp01(1.0 - float(wer)), 3)

    segments = align_report.get("segments", [])
    if not segments:
        return 0.5

    silent = sum(1 for s in segments if s.get("raw_duration_s", 0.0) == 0.0)
    bad_speed = sum(1 for s in segments if abs(s.get("speed_factor", 1.0) - 1.0) > 0.25)
    penalty = (silent + bad_speed) / max(len(segments), 1)
    return round(_clamp01(1.0 - penalty), 3)


def _semantic_score(metrics: list[SegmentMetrics], align_report: dict | None) -> float:
    if align_report:
        for key in ("semantic_similarity", "embedding_similarity", "backtranslation_similarity"):
            if key in align_report:
                return round(_clamp01(float(align_report[key])), 3)

    if not metrics:
        return 0.5

    retries = sum(1 for m in metrics if decide_action(m) in (AlignAction.REQUEST_SHORTER, AlignAction.FAIL))
    return round(_clamp01(1.0 - retries / max(len(metrics), 1)), 3)


def _naturalness_score(aligned: list[AlignedSegment], align_report: dict | None) -> float:
    speeds: list[float] = []

    if align_report:
        for seg in align_report.get("segments", []):
            sf = seg.get("speed_factor")
            if sf and sf > 0:
                speeds.append(float(sf))

    if not speeds:
        speeds = [a.stretch_factor for a in aligned if a.stretch_factor > 0]

    if not speeds:
        return 0.5

    if len(speeds) == 1:
        variance_penalty = abs(speeds[0] - 1.0)
    else:
        variance_penalty = _stats.pstdev(speeds) + abs(_stats.mean(speeds) - 1.0)

    severe = sum(1 for s in speeds if s > 1.4 or s < 0.75) / max(len(speeds), 1)
    return round(_clamp01(1.0 - variance_penalty - severe), 3)


def dubbing_scorecard(
    metrics: list[SegmentMetrics],
    aligned_segments: list[AlignedSegment],
    align_report: dict | None = None,
) -> dict:
    """Return normalized dubbing quality scores in [0, 1]."""
    report = clip_evaluation_report(metrics, aligned_segments)

    timing = _timing_score(report)
    intelligibility = _intelligibility_score(align_report)
    semantic = _semantic_score(metrics, align_report)
    naturalness = _naturalness_score(aligned_segments, align_report)

    overall = round(
        0.35 * timing
        + 0.20 * intelligibility
        + 0.25 * semantic
        + 0.20 * naturalness,
        3,
    )

    return {
        "overall": overall,
        "timing_accuracy": timing,
        "intelligibility": intelligibility,
        "semantic_fidelity": semantic,
        "naturalness": naturalness,
        "details": report,
    }
