"""
Text cleaning, language detection, and normalization pipeline.

Cleaning decisions and their rationale:
- **HTML entities**: YouTube comments may contain ``&amp;``, ``&lt;``, etc.
  These are decoded back to their text equivalents.
- **URLs and mentions**: Removed because they carry little lexical sentiment
  signal and introduce noise for topic modeling (unique URLs fragment topics).
- **YouTube timestamps**: Removed (e.g. ``1:23``, ``12:34``) — these are
  references to video moments, not linguistic content.
- **Emojis**: KEPT in the cleaned text because they carry strong sentiment
  signal for football reactions (e.g. ``"\U0001f1e7\U0001f1f7\U0001f525"``).
  They are also extracted into a separate ``emojis`` column for potential
  use as a feature. Earlier versions removed them; we now keep them.
- **Bullet/list markers**: Stripped from leading/trailing positions.
- **Repeating characters**: Collapsed (``"nooooo"`` \u2192 ``"nooo"``) to reduce
  vocabulary sparsity without losing the emphasis signal.
- **Deduplication**: SHA-256 hash of cleaned text catches exact duplicates
  from cross-posting or identical comments by the same user.
- **Language filtering**: Only ``es`` and ``en`` are kept; others are filtered
  out to maintain model quality (the BERT models are language-specific).
- **Spam filtering**: Comments flagged ``is_spam=True`` at collection time
  are removed in the notebook (not in this module) so the raw archive
  remains complete.
"""

import html
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd
from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

if TYPE_CHECKING:
    from spacy.language import Language

from src.config import (
    FOOTBALL_KEYWORDS,
    LANG_DETECT_CONFIDENCE_THRESHOLD,
    SPAM_PHRASES,
    SUPPORTED_LANGUAGES,
    TEAM_ALIASES,
)
from src.utils import setup_logger, text_hash

DetectorFactory.seed = 42

logger = setup_logger(__name__)

# ── Regex patterns ──────────────────────────────────────────────────────────
URL_PATTERN: re.Pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN: re.Pattern = re.compile(r"@\w+")
MULTI_SPACE_PATTERN: re.Pattern = re.compile(r"\s{2,}")
EMOJI_PATTERN: re.Pattern = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    re.UNICODE,
)
TIMESTAMP_PATTERN: re.Pattern = re.compile(r"\b\d{1,2}:\d{2}\b")
REPEATING_CHARS: re.Pattern = re.compile(r"(.)\1{3,}")
STRIP_LEADING_DASH: re.Pattern = re.compile(r"^[\s\-•*]+")
STRIP_TRAILING_DASH: re.Pattern = re.compile(r"[\s\-•*]+$")

# ── Spam / signal detection ─────────────────────────────────────────────────


def _all_team_aliases() -> List[str]:
    aliases: List[str] = []
    for names in TEAM_ALIASES.values():
        aliases.extend(names)
    return list(set(aliases))


_ALL_TEAM_ALIASES: List[str] = _all_team_aliases()


def is_likely_spam(text: str) -> bool:
    """Heuristic spam detection based on common bot/promotional phrases."""
    lower = text.lower()
    for phrase in SPAM_PHRASES:
        if phrase in lower:
            return True
    return False


def is_low_signal(text: str) -> bool:
    """Check if a comment lacks football relevance (too generic or off-topic).

    A comment is low-signal if it contains **neither**:
    - a known team alias, nor
    - a football keyword.
    """
    lower = text.lower()
    has_team = any(alias in lower for alias in _ALL_TEAM_ALIASES)
    has_football = any(kw in lower for kw in FOOTBALL_KEYWORDS)
    return not (has_team or has_football)


def deduplicate_comments(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate comments based on ``text_hash``.

    Keeps the first occurrence of each hash. Logs how many were removed.

    Args:
        df: DataFrame with a ``"text_hash"`` column.

    Returns:
        DataFrame with duplicates removed.
    """
    n_before = len(df)
    result = df.drop_duplicates(subset=["text_hash"], keep="first").copy()
    n_removed = n_before - len(result)
    if n_removed:
        logger.info("Deduplication: removed %d duplicate comments", n_removed)
    return result


def tokenize_text(
    text: str,
    lang: str,
    nlp_es: "Language",
    nlp_en: "Language",
) -> List[str]:
    """Lemmatize and remove stopwords/punctuation using spaCy.

    Uses ``nlp_es`` for Spanish (``"es"``) and ``nlp_en`` for English
    (``"en"``). For any other language code returns an empty list.

    Args:
        text: Cleaned text string.
        lang: ISO language code (``"es"``, ``"en"``, or other).
        nlp_es: Loaded ``es_core_news_sm`` pipeline.
        nlp_en: Loaded ``en_core_web_sm`` pipeline.

    Returns:
        List of lemmatized tokens (lowercase, no stopwords, no punctuation).
    """
    if lang == "es":
        doc = nlp_es(text)
    elif lang == "en":
        doc = nlp_en(text)
    else:
        return []

    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]


def detect_language(text: str) -> Tuple[str, float]:
    """Detect the language of ``text`` using langdetect.

    Returns ``("unknown", 0.0)`` if detection fails or confidence is below
    :const:`~src.config.LANG_DETECT_CONFIDENCE_THRESHOLD`.

    Args:
        text: Input string.

    Returns:
        Tuple of ``(language_code, confidence)``.
    """
    try:
        langs = detect_langs(text)
        if langs:
            lang = langs[0]
            if lang.prob >= LANG_DETECT_CONFIDENCE_THRESHOLD:
                return lang.lang, round(lang.prob, 3)
        return "unknown", 0.0
    except LangDetectException:
        return "unknown", 0.0


def extract_emojis(text: str) -> List[str]:
    """Return a list of emoji sequences found in ``text``."""
    return EMOJI_PATTERN.findall(text)


def clean_text(text: str) -> str:
    """Apply all cleaning steps to a single text string.

    Steps:
    1. Decode HTML entities (``&amp;`` → ``&``, etc.).
    2. Remove URLs.
    3. Remove @-mentions.
    4. Remove YouTube timestamps (``1:23``, ``12:34``).
    5. Strip leading/trailing dashes and bullet markers.
    6. Collapse repeating characters (4+ → 3).
    7. Collapse multiple spaces and strip.

    **Emojis are kept** in the returned text (they carry sentiment signal
    for football reactions). Use :func:`extract_emojis` separately if you
    need the emoji string as a feature.

    Args:
        text: Raw comment text.

    Returns:
        Cleaned text with emojis preserved.
    """
    text = html.unescape(text)
    text = URL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    text = TIMESTAMP_PATTERN.sub("", text)
    text = STRIP_LEADING_DASH.sub("", text)
    text = STRIP_TRAILING_DASH.sub("", text)
    text = REPEATING_CHARS.sub(r"\1\1\1", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text)
    return text.strip()


def preprocess_comment(
    text: str,
    min_length: int = 10,
) -> Optional[Dict]:
    """Run the full preprocessing pipeline on a single comment.

    Args:
        text: Raw comment text.
        min_length: Minimum character length after cleaning (shorter is dropped).

    Returns:
        Dict with preprocessed fields, or ``None`` if the comment is discarded.
    """
    raw_clean = text.strip()
    if not raw_clean or len(raw_clean) < min_length:
        return None

    lang_code, lang_conf = detect_language(raw_clean)
    cleaned = clean_text(raw_clean)

    if len(cleaned) < min_length:
        return None
    if lang_code not in SUPPORTED_LANGUAGES:
        return None

    emojis_found = extract_emojis(raw_clean)

    return {
        "text_clean": cleaned,
        "language": lang_code,
        "lang_confidence": lang_conf,
        "text_hash": text_hash(cleaned),
        "emojis": "".join(emojis_found) if emojis_found else "",
        "n_emojis": len(emojis_found),
    }


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to every row in a comments DataFrame.

    Adds columns:
    - ``text_clean``: cleaned comment body.
    - ``language``: detected language code.
    - ``lang_confidence``: detection confidence.
    - ``emojis``: extracted emoji string.
    - ``n_emojis``: count of emoji sequences.

    Processing steps:
    1. Per-row: language detection, text cleaning, emoji extraction.
    2. Drop rows that fail language filtering (not ``es``/``en``) or are
       too short (``< 10`` chars after cleaning).
    3. Deduplicate on ``text_hash`` (keeps first occurrence).

    Args:
        df: DataFrame with at least a ``"text"`` column.

    Returns:
        Preprocessed DataFrame.
    """
    if df.empty:
        return df

    results = df["text"].apply(preprocess_comment)
    valid_mask = results.notna()
    n_dropped = (~valid_mask).sum()
    if n_dropped:
        logger.info("Dropped %d rows (language filter / too short)", n_dropped)

    preprocessed = df[valid_mask].copy()
    preprocessed_df = pd.DataFrame(results[valid_mask].tolist())
    preprocessed = pd.concat(
        [preprocessed.reset_index(drop=True), preprocessed_df.reset_index(drop=True)],
        axis=1,
    )

    preprocessed = deduplicate_comments(preprocessed)

    preprocessed.reset_index(drop=True, inplace=True)
    logger.info("Preprocessing complete: %d comments kept", len(preprocessed))
    return preprocessed
