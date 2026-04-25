"""Deterministic failure analysis and translation re-ranking utilities.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics. The translation re-ranking function implements a local,
rule-based strategy to generate shorter Spanish candidates that better fit
a duration budget.
"""

import dataclasses
import logging
import re

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report."""
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


_CHARS_PER_SECOND = 15.0

# Phrase-level replacements are safer than word-level ones.
_PHRASE_REPLACEMENTS: list[tuple[str, str, str]] = [
    (r"\ben este momento\b", "ahora", "shortened phrase"),
    (r"\ben este caso\b", "aquí", "shortened phrase"),
    (r"\bdebido a que\b", "porque", "shortened phrase"),
    (r"\bya que\b", "porque", "shortened phrase"),
    (r"\bcon el fin de\b", "para", "shortened phrase"),
    (r"\bpor medio de\b", "con", "shortened phrase"),
    (r"\ba causa de\b", "por", "shortened phrase"),
    (r"\bpor lo tanto\b", "así que", "shortened phrase"),
    (r"\bsin embargo\b", "pero", "shortened phrase"),
    (r"\bno obstante\b", "pero", "shortened phrase"),
    (r"\bde hecho\b", "", "removed filler"),
    (r"\ben realidad\b", "", "removed filler"),
    (r"\bes decir\b", "", "removed filler"),
    (r"\bo sea\b", "", "removed filler"),
    # Special fix for awkward "escenario del caso"
    (r"\bpeor escenario del caso\b", "peor escenario", "collapsed awkward phrase"),
    (r"\bescenario del caso\b", "escenario", "collapsed awkward phrase"),
]

# Be conservative. Avoid replacements that hurt meaning.
_WORD_REPLACEMENTS: dict[str, tuple[str, str]] = {
    "aproximadamente": ("casi", "shorter synonym"),
    "actualmente": ("hoy", "shorter synonym"),
    "finalmente": ("al fin", "shorter synonym"),
    "inmediatamente": ("ya", "shorter synonym"),
    "utilizar": ("usar", "shorter synonym"),
    "utilizando": ("usando", "shorter synonym"),
    "comenzar": ("empezar", "shorter synonym"),
    "comienza": ("empieza", "shorter synonym"),
    "comenzó": ("empezó", "shorter synonym"),
    "observar": ("ver", "shorter synonym"),
    "observó": ("vio", "shorter synonym"),
    "solamente": ("solo", "shorter synonym"),
    "únicamente": ("solo", "shorter synonym"),
    "problema": ("fallo", "shorter synonym"),
}

_FILLER_WORDS = {
    "muy", "realmente", "bastante", "simplemente", "literalmente",
    "básicamente", "prácticamente", "quizás", "tal vez",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT_RE = re.compile(r"[;:,\-—]\s+")


def _target_char_budget(target_duration_s: float) -> int:
    return max(8, int(round(target_duration_s * _CHARS_PER_SECOND)))


def _clean_text(text: str) -> str:
    """Remove subtitle junk / timestamps / noisy residue."""
    if not text:
        return ""

    text = text.replace("&gt;", " ")
    text = text.replace("&lt;", " ")
    text = text.replace("/c", " ")
    text = text.replace("\\n", " ")
    text = text.replace("\n", " ")

    # Remove timestamps like 00:00:08.240
    text = re.sub(r"\b\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\b", " ", text)
    text = re.sub(r"\b\d{1,2}:\d{2}(?:\.\d+)?\b", " ", text)

    # Remove common subtitle residue seen in your outputs
    junk_patterns = [
        r"\bcursoc\b",
        r"\bcontact\w*\b",
        r"\bingres\w*\b",
        r"\bconfianz\w*\b",
        r"\brealizad\w*\b",
        r"\bindicad\w*\b",
        r"\bescrit\w*\b",
        r"\brecomendad\w*\b",
        r"\bseleccionad\w*\b",
        r"\bautorizad\w*\b",
        r"\bcumplid\w*\b",
        r"\bimplicad\w*\b",
        r"\bint[ée]rprete\w*\b",
        r"\bintitulado\w*\b",
        r"\bfieltro\b",
        r"\bhizo referencia\b",
        r"\bpropiedad intelectual\b",
    ]
    for pattern in junk_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"[{}\[\]<>|*_]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def _dedupe_preserve_order(items: list[TranslationCandidate]) -> list[TranslationCandidate]:
    seen: set[str] = set()
    out: list[TranslationCandidate] = []
    for item in items:
        key = item.text.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _apply_phrase_replacements(text: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    out = text
    for pattern, repl, reason in _PHRASE_REPLACEMENTS:
        new_out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
        if new_out != out:
            reasons.append(reason)
            out = new_out
    out = re.sub(r"\s+", " ", out).strip()
    return out, reasons


def _apply_word_replacements(text: str) -> tuple[str, list[str]]:
    reasons: list[str] = []

    def repl(match: re.Match[str]) -> str:
        word = match.group(0)
        lower = word.lower()
        if lower in _WORD_REPLACEMENTS:
            replacement, reason = _WORD_REPLACEMENTS[lower]
            reasons.append(reason)
            return replacement
        return word

    out = re.sub(r"\b[^\W\d_]+\b", repl, text, flags=re.UNICODE)
    out = re.sub(r"\s+", " ", out).strip()
    return out, reasons


def _remove_fillers(text: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    tokens = text.split()
    kept: list[str] = []
    for tok in tokens:
        core = re.sub(r"[^\wáéíóúñüÁÉÍÓÚÑÜ]+", "", tok, flags=re.UNICODE).lower()
        if core in _FILLER_WORDS:
            reasons.append("removed filler")
            continue
        kept.append(tok)
    out = " ".join(kept)
    out = re.sub(r"\s+", " ", out).strip()
    return out, reasons


def _sentence_candidates(text: str) -> list[tuple[str, str]]:
    sents = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    out: list[tuple[str, str]] = []
    for s in sents:
        out.append((s, "selected shorter sentence"))
    if len(sents) >= 2:
        out.append((" ".join(sents[:2]), "kept main sentences"))
    return out


def _clause_candidates(text: str) -> list[tuple[str, str]]:
    parts = [p.strip(" ,;:-") for p in _CLAUSE_SPLIT_RE.split(text) if p.strip(" ,;:-")]
    out: list[tuple[str, str]] = []
    for p in parts:
        out.append((p, "selected shorter clause"))
    if len(parts) >= 2:
        out.append((" ".join(parts[:2]), "kept main clauses"))
    return out


def _score_candidate(text: str, budget: int) -> tuple[int, int]:
    """Prefer natural candidates that are close to budget, with small overflow penalty."""
    n = len(text)
    overflow = max(0, n - budget)
    distance = abs(n - budget)
    return (overflow * 5 + distance, n)


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    Strategy:
    - clean noisy subtitle/timestamp residue
    - estimate a character budget from target duration
    - generate deterministic shorter Spanish variants using phrase shortening,
      filler removal, conservative synonym replacement, and clause trimming
    - return candidates sorted best-first
    """
    budget = _target_char_budget(target_duration_s)
    baseline_es = _clean_text(baseline_es)

    logger.info(
        "get_shorter_translations called for %.1fs budget (%d-char target, %d chars baseline).",
        target_duration_s,
        budget,
        len(baseline_es),
    )

    if not baseline_es:
        return []

    candidates: list[TranslationCandidate] = []

    def add(text: str, rationale: str) -> None:
        text = _clean_text(text)
        if not text:
            return
        candidates.append(
            TranslationCandidate(
                text=text,
                char_count=len(text),
                brevity_rationale=rationale,
            )
        )

    add(baseline_es, "baseline")

    phrased, phrase_reasons = _apply_phrase_replacements(baseline_es)
    add(phrased, ", ".join(sorted(set(phrase_reasons))) or "phrase shortening")

    words_short, word_reasons = _apply_word_replacements(phrased)
    add(words_short, ", ".join(sorted(set(phrase_reasons + word_reasons))) or "word shortening")

    filler_short, filler_reasons = _remove_fillers(words_short)
    add(
        filler_short,
        ", ".join(sorted(set(phrase_reasons + word_reasons + filler_reasons))) or "removed fillers",
    )

    for text, rationale in _sentence_candidates(filler_short):
        add(text, rationale)

    for text, rationale in _clause_candidates(filler_short):
        add(text, rationale)

    # Tight-budget fallbacks
    if len(filler_short) > budget:
        words = filler_short.split()
        if len(words) >= 3:
            add(" ".join(words[:3]), "tight budget fallback")
        if len(words) >= 4:
            add(" ".join(words[:4]), "tight budget fallback")
        if len(words) >= 5:
            add(" ".join(words[:5]), "tight budget fallback")

    candidates = _dedupe_preserve_order(candidates)
    candidates.sort(key=lambda c: _score_candidate(c.text, budget))

    return candidates[:8]