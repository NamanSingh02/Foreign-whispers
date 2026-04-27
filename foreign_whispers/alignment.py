"""Duration-aware alignment data model and decision logic.

This module is the core of the ``foreign_whispers`` library.  It answers the
central question of the dubbing pipeline: *how do we fit a target-language
translation into the same time window as the original source-language speech?*

The module provides:

- ``SegmentMetrics`` — measures the timing mismatch for each segment.
- ``decide_action`` — per-segment policy that chooses accept / stretch / shift / retry / fail.
- ``global_align`` — greedy left-to-right pass that schedules all segments
  on a shared timeline, tracking cumulative drift from gap shifts.
- ``global_align_dp`` — beam-search/DP style optimizer that explores multiple
  timing actions and chooses the lowest-cost schedule.

No external dependencies — stdlib only.
"""
import dataclasses
import math
import re
import unicodedata
from enum import Enum


_VOWELS = "aeiouáéíóúü"


def _normalise_text(text: str) -> str:
    """Lowercase and remove accents while preserving word boundaries."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _count_syllables(text: str) -> int:
    """Count syllables with a lightweight vowel-cluster heuristic."""
    norm = _normalise_text(text)
    clusters = re.findall(r"[aeiou]+", norm)
    return max(1, len(clusters)) if norm.strip() else 0


def _estimate_duration(text: str) -> float:
    """Estimate target-language TTS duration in seconds.

    Uses syllables, words, punctuation pauses, and a small fixed startup cost.
    This is more stable than the old character-count rule for Spanish-style TTS.
    """
    text = (text or "").strip()
    if not text:
        return 0.0

    words = re.findall(r"\b[^\W\d_]+\b", text, flags=re.UNICODE)
    syllables = _count_syllables(text)
    chars = len(re.sub(r"\s+", "", text))

    comma_pauses = len(re.findall(r"[,;:]", text)) * 0.10
    sentence_pauses = len(re.findall(r"[.!?]", text)) * 0.18
    long_word_penalty = sum(1 for w in words if len(w) >= 11) * 0.035

    syllable_component = syllables / 4.8
    word_component = len(words) / 3.1
    char_component = chars / 18.5

    duration = (
        0.18
        + 0.60 * syllable_component
        + 0.25 * word_component
        + 0.15 * char_component
        + comma_pauses
        + sentence_pauses
        + long_word_penalty
    )

    return round(max(0.25, duration), 3)


@dataclasses.dataclass
class SegmentMetrics:
    """Timing measurements for one source/target transcript segment pair."""
    index:             int
    source_start:      float
    source_end:        float
    source_duration_s: float
    source_text:       str
    translated_text:   str
    src_char_count:    int
    tgt_char_count:    int
    predicted_tts_s:   float = dataclasses.field(init=False)
    predicted_stretch: float = dataclasses.field(init=False)
    overflow_s:        float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.predicted_tts_s = _estimate_duration(self.translated_text)
        self.predicted_stretch = (
            self.predicted_tts_s / self.source_duration_s
            if self.source_duration_s > 0 else 1.0
        )
        self.overflow_s = max(0.0, self.predicted_tts_s - self.source_duration_s)


class AlignAction(str, Enum):
    """Decision outcomes for the per-segment alignment policy."""
    ACCEPT          = "accept"
    MILD_STRETCH    = "mild_stretch"
    GAP_SHIFT       = "gap_shift"
    REQUEST_SHORTER = "request_shorter"
    FAIL            = "fail"


@dataclasses.dataclass
class AlignedSegment:
    """A segment with its scheduled position on the global timeline."""
    index:           int
    original_start:  float
    original_end:    float
    scheduled_start: float
    scheduled_end:   float
    text:            str
    action:          AlignAction
    gap_shift_s:     float = 0.0
    stretch_factor:  float = 1.0


def decide_action(m: SegmentMetrics, available_gap_s: float = 0.0) -> AlignAction:
    """Choose the alignment action for a single segment."""
    sf = m.predicted_stretch
    if sf <= 1.1:
        return AlignAction.ACCEPT
    if sf <= 1.4:
        return AlignAction.MILD_STRETCH
    if sf <= 1.8 and available_gap_s >= m.overflow_s:
        return AlignAction.GAP_SHIFT
    if sf <= 2.5:
        return AlignAction.REQUEST_SHORTER
    return AlignAction.FAIL


def compute_segment_metrics(
    en_transcript: dict,
    es_transcript: dict,
) -> list[SegmentMetrics]:
    """Pair source and target segments and compute per-segment timing metrics."""
    metrics = []
    for i, (en_seg, es_seg) in enumerate(
        zip(en_transcript.get("segments", []), es_transcript.get("segments", []))
    ):
        src_text = en_seg["text"].strip()
        tgt_text = es_seg["text"].strip()
        metrics.append(SegmentMetrics(
            index             = i,
            source_start      = en_seg["start"],
            source_end        = en_seg["end"],
            source_duration_s = en_seg["end"] - en_seg["start"],
            source_text       = src_text,
            translated_text   = tgt_text,
            src_char_count    = len(src_text),
            tgt_char_count    = len(tgt_text),
        ))
    return metrics


def _silence_after(end_s: float, silence_regions: list[dict]) -> float:
    for r in silence_regions:
        if r.get("label") == "silence" and r["start_s"] >= end_s - 0.1:
            return max(0.0, r["end_s"] - r["start_s"])
    return 0.0


def global_align(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
) -> list[AlignedSegment]:
    """Greedy left-to-right global alignment of dubbed segments."""
    aligned, cumulative_drift = [], 0.0

    for m in metrics:
        action    = decide_action(m, available_gap_s=_silence_after(m.source_end, silence_regions))
        gap_shift = 0.0
        stretch   = 1.0

        if action == AlignAction.GAP_SHIFT:
            gap_shift = m.overflow_s
        elif action == AlignAction.MILD_STRETCH:
            stretch = min(m.predicted_stretch, max_stretch)

        sched_start = m.source_start + cumulative_drift
        sched_end   = sched_start + m.source_duration_s + gap_shift

        aligned.append(AlignedSegment(
            index           = m.index,
            original_start  = m.source_start,
            original_end    = m.source_end,
            scheduled_start = sched_start,
            scheduled_end   = sched_end,
            text            = m.translated_text,
            action          = action,
            gap_shift_s     = gap_shift,
            stretch_factor  = stretch,
        ))

        cumulative_drift += gap_shift

    return aligned


def _candidate_actions(m: SegmentMetrics, available_gap_s: float, max_stretch: float) -> list[tuple[AlignAction, float, float, float]]:
    """Return candidate (action, gap_shift, stretch, local_cost)."""
    candidates: list[tuple[AlignAction, float, float, float]] = []
    sf = m.predicted_stretch

    if sf <= 1.1:
        candidates.append((AlignAction.ACCEPT, 0.0, 1.0, abs(1.0 - sf)))

    if sf <= max_stretch:
        stretch = max(1.0, min(sf, max_stretch))
        action = AlignAction.ACCEPT if sf <= 1.1 else AlignAction.MILD_STRETCH
        candidates.append((action, 0.0, stretch, abs(stretch - 1.0) * 1.2))

    if m.overflow_s > 0 and available_gap_s >= m.overflow_s:
        candidates.append((
            AlignAction.GAP_SHIFT,
            m.overflow_s,
            1.0,
            0.25 + 0.35 * m.overflow_s,
        ))

    if sf <= 2.5:
        candidates.append((
            AlignAction.REQUEST_SHORTER,
            0.0,
            1.0,
            1.0 + max(0.0, sf - max_stretch) * 2.0,
        ))
    else:
        candidates.append((AlignAction.FAIL, 0.0, 1.0, 12.0 + sf))

    # Remove duplicate action/gap/stretch tuples while preserving best local cost.
    best: dict[tuple[AlignAction, float, float], float] = {}
    for action, gap, stretch, cost in candidates:
        key = (action, round(gap, 3), round(stretch, 3))
        best[key] = min(best.get(key, math.inf), cost)

    return [(a, g, s, c) for (a, g, s), c in best.items()]


def _schedule_cost(aligned: list[AlignedSegment]) -> float:
    if not aligned:
        return 0.0

    drift = abs(aligned[-1].scheduled_end - aligned[-1].original_end)
    severe = sum(1 for a in aligned if a.stretch_factor > 1.4)
    retry = sum(1 for a in aligned if a.action == AlignAction.REQUEST_SHORTER)
    fail = sum(1 for a in aligned if a.action == AlignAction.FAIL)
    overlap = sum(
        1 for prev, cur in zip(aligned, aligned[1:])
        if cur.scheduled_start < prev.scheduled_end - 1e-6
    )
    gap_shift = sum(a.gap_shift_s for a in aligned)

    return (
        severe * 20.0
        + overlap * 30.0
        + fail * 25.0
        + retry * 1.0
        + drift * 4.0
        + gap_shift * 2.0
    )

def global_align_dp(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
    beam_width:      int = 32,
) -> list[AlignedSegment]:
    """Beam-search global optimizer for the dubbed timeline.

    Explores multiple actions per segment and keeps the lowest-cost schedules.
    The objective penalizes severe stretch, overlaps, translation retries,
    failures, and cumulative drift.
    """
    if not metrics:
        return []

    # state = (cost, cumulative_drift, aligned_segments)
    states: list[tuple[float, float, list[AlignedSegment]]] = [(0.0, 0.0, [])]

    for m in metrics:
        available_gap_s = _silence_after(m.source_end, silence_regions)
        choices = _candidate_actions(m, available_gap_s, max_stretch)
        next_states: list[tuple[float, float, list[AlignedSegment]]] = []

        for cost_so_far, drift, aligned_so_far in states:
            prev_end = aligned_so_far[-1].scheduled_end if aligned_so_far else -math.inf

            for action, gap_shift, stretch, local_cost in choices:
                sched_start = m.source_start + drift
                overlap_penalty = 0.0
                if sched_start < prev_end:
                    overlap_penalty = (prev_end - sched_start) * 8.0 + 10.0

                sched_end = sched_start + m.source_duration_s + gap_shift
                new_drift = drift + gap_shift

                new_seg = AlignedSegment(
                    index           = m.index,
                    original_start  = m.source_start,
                    original_end    = m.source_end,
                    scheduled_start = sched_start,
                    scheduled_end   = sched_end,
                    text            = m.translated_text,
                    action          = action,
                    gap_shift_s     = gap_shift,
                    stretch_factor  = stretch,
                )

                drift_penalty = abs(new_drift) * 3.0
                new_cost = cost_so_far + local_cost + overlap_penalty + drift_penalty
                next_states.append((new_cost, new_drift, aligned_so_far + [new_seg]))

        next_states.sort(key=lambda x: (x[0], _schedule_cost(x[2])))
        states = next_states[:max(1, beam_width)]

    best = min(states, key=lambda x: (x[0], _schedule_cost(x[2])))[2]
    return best
