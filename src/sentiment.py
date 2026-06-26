"""
Unified sentiment analysis pipeline supporting Spanish and English.

Architecture
------------
- **Primary model** (ES + EN): ``pysentimiento`` BERT-based classifiers,
  fine-tuned on social-media text. Single dependency for both languages.
  *Rationale*: pysentimiento 0.7+ includes both Spanish (``pysentimiento/
  bert-base-spanish-wwm-uncased``) and English (``cardiffnlp/twitter-
  roberta-base-sentiment``) models under a unified API. Using the same
  library for both languages avoids maintaining separate model-loading
  paths and enables consistent batch inference.
- **Baseline** (EN): VADER — a lexicon/rule-based model. Fast,
  interpretable, but misses sarcasm and context.
- **Baseline** (ES): A Spanish polarity-lexicon approach using a simple
  word-count of positive/negative terms from the NRC-EmoLex translated
  list. *Justification*: VADER is English-only. TextBlob-es is
  unmaintained and unreliable for social-media text. A polarity-lexicon
  baseline is transparent and sufficient for comparison.

Usage
-----
Call :func:`predict_sentiment` with the text and its language code to get
a unified ``(label, scores_dict)`` result. The pipeline handles the model
dispatch internally.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm.auto import tqdm

from src.utils import load_dataframe, setup_logger

logger = setup_logger(__name__)

# ── Lazy-loaded model singletons (per language) ────────────────────────────
_PYSENTIMIENTO: Dict[str, object] = {}
_VADER = None


def _get_pysentimiento(lang: str):
    """Return or create a pysentimiento analyzer for *lang*.

    Args:
        lang: ``"es"`` or ``"en"``.

    Returns:
        Analyzer instance, or ``None`` on failure.
    """
    global _PYSENTIMIENTO
    if lang not in _PYSENTIMIENTO:
        try:
            from pysentimiento import create_analyzer

            _PYSENTIMIENTO[lang] = create_analyzer(task="sentiment", lang=lang)
            logger.info("pysentimiento analyzer loaded for lang=%s", lang)
        except Exception as exc:
            logger.warning("pysentimiento unavailable for lang=%s: %s", lang, exc)
            _PYSENTIMIENTO[lang] = None
    return _PYSENTIMIENTO[lang]


def _get_vader():
    global _VADER
    if _VADER is None:
        try:
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            try:
                _VADER = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download("vader_lexicon")
                _VADER = SentimentIntensityAnalyzer()
            logger.info("VADER model loaded")
        except Exception as exc:
            logger.warning("VADER unavailable: %s", exc)
    return _VADER


# ── Spanish polarity lexicon ───────────────────────────────────────────────
# Based on NRC-EmoLex Spanish translation, curated for football context
_POSITIVE_WORDS_ES: set = {
    "bueno",
    "excelente",
    "increíble",
    "genial",
    "fantástico",
    "maravilloso",
    "espectacular",
    "brillante",
    "magnífico",
    "extraordinario",
    "perfecto",
    "impresionante",
    "emocionante",
    "alegre",
    "feliz",
    "contento",
    "orgulloso",
    "victoria",
    "triunfo",
    "gol",
    "golazo",
    "olé",
    "vamos",
    "sí",
    "mejor",
    "grande",
    "histórico",
    "leyenda",
    "crack",
    "fenómeno",
    "esperanza",
    "ilusionado",
    "optimista",
    "confianza",
}
_NEGATIVE_WORDS_ES: set = {
    "malo",
    "pésimo",
    "horrible",
    "terrible",
    "decepcionante",
    "frustrante",
    "vergonzoso",
    "ridículo",
    "lamentable",
    "penoso",
    "patético",
    "nefasto",
    "fracaso",
    "derrota",
    "perder",
    "perdió",
    "eliminado",
    "peor",
    "triste",
    "enfadado",
    "enojado",
    "furioso",
    "decepcionado",
    "arbitro",
    "robo",
    "injusticia",
    "penal",
    "expulsión",
    "lesión",
    "vergüenza",
    "asqueroso",
    "insulto",
    "bochornoso",
}


# ── Individual prediction helpers ──────────────────────────────────────────


def _predict_pysentimiento(text: str, lang: str) -> Dict[str, float]:
    analyzer = _get_pysentimiento(lang)
    if analyzer is None:
        raise RuntimeError(f"pysentimiento not available for lang={lang}")
    result = analyzer.predict(text)
    return {
        "positive": result.probas.get("POS", 0.0),
        "negative": result.probas.get("NEG", 0.0),
        "neutral": result.probas.get("NEU", 0.0),
    }


def _predict_vader(text: str) -> Dict[str, float]:
    analyzer = _get_vader()
    if analyzer is None:
        raise RuntimeError("VADER not available")
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        return {
            "positive": abs(compound),
            "negative": 0.0,
            "neutral": 1 - abs(compound),
        }
    elif compound <= -0.05:
        return {
            "positive": 0.0,
            "negative": abs(compound),
            "neutral": 1 - abs(compound),
        }
    else:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}


def _baseline_es(text: str) -> Dict[str, float]:
    """Simple polarity-lexicon baseline for Spanish.

    Counts positive/negative word hits and returns normalised scores.
    """
    words = text.lower().split()
    pos_hits = sum(1 for w in words if w in _POSITIVE_WORDS_ES)
    neg_hits = sum(1 for w in words if w in _NEGATIVE_WORDS_ES)
    total = pos_hits + neg_hits
    if total == 0:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
    pos_score = pos_hits / total
    neg_score = neg_hits / total
    return {
        "positive": round(pos_score, 4),
        "negative": round(neg_score, 4),
        "neutral": round(1.0 - pos_score - neg_score, 4),
    }


# ── Public prediction functions ────────────────────────────────────────────


def predict_sentiment_bert(text: str, lang: str) -> Dict[str, float]:
    """Predict sentiment using the BERT primary model (pysentimiento).

    Args:
        text: Cleaned text.
        lang: ISO code (``"es"`` or ``"en"``).

    Returns:
        Dict with ``positive``, ``negative``, ``neutral`` scores.
    """
    return _predict_pysentimiento(text, lang)


def predict_sentiment_baseline(text: str, lang: str) -> Dict[str, float]:
    """Predict sentiment using the rule-based baseline.

    For ``en``: VADER. For ``es``: polarity lexicon. For other langs
    returns a neutral default.

    Args:
        text: Cleaned text.
        lang: ISO code.

    Returns:
        Dict with ``positive``, ``negative``, ``neutral`` scores.
    """
    if lang == "en":
        return _predict_vader(text)
    elif lang == "es":
        return _baseline_es(text)
    return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}


# ── Batch inference ────────────────────────────────────────────────────────


def _batch_pysentimiento(texts: List[str], lang: str) -> List[Dict[str, float]]:
    """Run pysentimiento batch inference.

    pysentimiento's ``analyzer.predict()`` accepts a list of texts and
    returns a list of ``AnalyzerOutput`` objects — much faster than
    row-by-row ``.apply()``.
    """
    analyzer = _get_pysentimiento(lang)
    if analyzer is None:
        raise RuntimeError(f"pysentimiento not available for lang={lang}")
    results = analyzer.predict(texts)
    return [
        {
            "positive": r.probas.get("POS", 0.0),
            "negative": r.probas.get("NEG", 0.0),
            "neutral": r.probas.get("NEU", 0.0),
        }
        for r in results
    ]


# ── DataFrame pipeline ─────────────────────────────────────────────────────


def apply_sentiment_pipeline(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    lang_col: str = "language",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Apply both BERT and baseline sentiment to a DataFrame.

    Adds columns:
    - ``sentiment_bert``: POS / NEG / NEU label from pysentimiento.
    - ``sentiment_bert_probas``: JSON string of probability scores.
    - ``sentiment_baseline``: POS / NEG / NEU label from baseline.

    Uses batch inference (grouped by language) when possible.

    Args:
        df: DataFrame with ``text_col`` and ``lang_col`` columns.
        text_col: Name of the cleaned-text column.
        lang_col: Name of the language column.
        show_progress: Show a ``tqdm`` progress bar.

    Returns:
        DataFrame with appended sentiment columns.
    """
    if df.empty or text_col not in df.columns:
        return df

    result = df.copy()
    bert_labels: List[str] = []
    bert_probas: List[str] = []
    base_labels: List[str] = []
    n_total = len(result)

    if show_progress:
        pbar = tqdm(total=n_total, desc="Sentiment analysis", unit="comments")

    for lang in ("es", "en"):
        mask = result[lang_col] == lang
        texts = result.loc[mask, text_col].tolist()
        if not texts:
            continue

        # BERT (batch)
        try:
            scores_list = _batch_pysentimiento(texts, lang)
        except RuntimeError as exc:
            logger.warning("BERT failed for lang=%s: %s", lang, exc)
            scores_list = [{"positive": 0.0, "negative": 0.0, "neutral": 1.0}] * len(
                texts
            )
        for s in scores_list:
            label = max(s, key=s.get).upper()[:3]
            if label == "POS":
                label = "POS"
            elif label == "NEG":
                label = "NEG"
            else:
                label = "NEU"
            bert_labels.append(label)
            bert_probas.append(str(s))

        # Baseline
        for t in texts:
            try:
                s = predict_sentiment_baseline(t, lang)
                label = max(s, key=s.get).upper()[:3]
                if label == "POS":
                    label = "POS"
                elif label == "NEG":
                    label = "NEG"
                else:
                    label = "NEU"
            except Exception as exc:
                logger.warning("Baseline failed for lang=%s: %s", lang, exc)
                label = "NEU"
            base_labels.append(label)

        if show_progress:
            pbar.update(len(texts))

    # Fallback for other/unknown languages
    other_mask = ~result[lang_col].isin(("es", "en"))
    n_other = other_mask.sum()
    if n_other:
        bert_labels.extend(["NEU"] * n_other)
        bert_probas.extend(['{"neutral": 1.0}'] * n_other)
        base_labels.extend(["NEU"] * n_other)
        if show_progress:
            pbar.update(n_other)
        logger.info("Assigned NEU default to %d comments (unsupported lang)", n_other)

    if show_progress:
        pbar.close()

    result["sentiment_bert"] = bert_labels
    result["sentiment_bert_probas"] = bert_probas
    result["sentiment_baseline"] = base_labels
    logger.info("Sentiment pipeline complete: %d comments", n_total)
    return result


# ── Incremental pipeline ────────────────────────────────────────────────────


def apply_sentiment_incremental(
    df_new: pd.DataFrame,
    existing_path: Optional[Path] = None,
    text_col: str = "text_clean",
    lang_col: str = "language",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run BERT sentiment only on new comments, merging with existing results.

    Loads previously computed sentiment from ``existing_path`` (parquet),
    identifies rows in ``df_new`` not yet present (by ``comment_id``), runs
    ``apply_sentiment_pipeline()`` on only those rows, and returns the
    combined DataFrame.

    Args:
        df_new: Freshly cleaned comments (output of notebook 02).
        existing_path: Path to an existing parquet with sentiment columns.
            If ``None`` or the file does not exist, runs on the full dataset.
        text_col: Cleaned-text column name.
        lang_col: Language column name.
        show_progress: Show a ``tqdm`` progress bar.

    Returns:
        DataFrame with sentiment columns appended.
    """
    if df_new.empty:
        return df_new

    if existing_path is not None and existing_path.exists():
        df_existing = load_dataframe(existing_path)
        logger.info(
            "Loaded %d existing sentiment results from %s",
            len(df_existing),
            existing_path,
        )
    else:
        df_existing = None
        logger.info("No existing sentiment results found — full processing.")

    if df_existing is not None and "comment_id" in df_existing.columns:
        existing_ids = set(df_existing["comment_id"].unique())
        new_ids = set(df_new["comment_id"].unique())
        already_known = existing_ids & new_ids
        truly_new_ids = new_ids - existing_ids
        n_known = len(already_known)
        n_new = len(truly_new_ids)

        if n_known > 0 and n_new == 0:
            logger.info(
                "All %d comments already have sentiment labels — nothing to do.",
                n_known,
            )
            return df_existing

        logger.info(
            "%d comments already have sentiment labels, running BERT on %d new comments only.",
            n_known,
            n_new,
        )

        df_to_process = df_new[df_new["comment_id"].isin(truly_new_ids)].copy()
        df_processed = apply_sentiment_pipeline(
            df_to_process,
            text_col=text_col,
            lang_col=lang_col,
            show_progress=show_progress,
        )
        df_combined = pd.concat([df_existing, df_processed], ignore_index=True)
    else:
        logger.info("No existing results or no comment_id — running on full dataset.")
        df_combined = apply_sentiment_pipeline(
            df_new,
            text_col=text_col,
            lang_col=lang_col,
            show_progress=show_progress,
        )

    logger.info(
        "Incremental merge complete — total: %d comments.",
        len(df_combined),
    )
    return df_combined


# ── Legacy API (backward compatible) ───────────────────────────────────────


def predict_sentiment(
    text: str,
    language: str,
    model: str = "transformer",
) -> Dict[str, Dict[str, float]]:
    """Run sentiment prediction on ``text`` using the specified model.

    Args:
        text: Cleaned text string.
        language: ISO 639-1 code (``"es"`` or ``"en"``).
        model: ``"transformer"`` (BERT, default) or ``"baseline"``.

    Returns:
        Nested dict ``{"transformer": ..., "baseline": ...}``.
    """
    result: Dict[str, Dict[str, float]] = {}

    try:
        if model == "transformer":
            result["transformer"] = predict_sentiment_bert(text, language)
        else:
            result["transformer"] = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
    except RuntimeError as exc:
        logger.warning("Transformer model failed: %s", exc)
        result["transformer"] = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    result["baseline"] = predict_sentiment_baseline(text, language)
    return result


def predict_batch(
    texts: List[str],
    languages: List[str],
    model: str = "transformer",
) -> pd.DataFrame:
    """Apply :func:`predict_sentiment` over a list of texts.

    Args:
        texts: Cleaned text strings.
        languages: Corresponding language codes.
        model: ``"transformer"`` or ``"baseline"``.

    Returns:
        DataFrame with sentiment columns.
    """
    records: List[Dict] = []
    for text, lang in zip(texts, languages):
        try:
            scores = predict_sentiment(text, lang, model=model)
            trans = scores["transformer"]
            base = scores["baseline"]
            records.append(
                {
                    "sentiment_label": max(trans, key=trans.get),
                    "sentiment_positive": trans["positive"],
                    "sentiment_negative": trans["negative"],
                    "sentiment_neutral": trans["neutral"],
                    "sentiment_baseline_label": max(base, key=base.get),
                    "sentiment_baseline_positive": base["positive"],
                    "sentiment_baseline_negative": base["negative"],
                    "sentiment_baseline_neutral": base["neutral"],
                }
            )
        except Exception as exc:
            logger.warning("Sentiment prediction failed: %s", exc)
            records.append(
                {
                    "sentiment_label": "error",
                    "sentiment_positive": 0.0,
                    "sentiment_negative": 0.0,
                    "sentiment_neutral": 0.0,
                    "sentiment_baseline_label": "error",
                    "sentiment_baseline_positive": 0.0,
                    "sentiment_baseline_negative": 0.0,
                    "sentiment_baseline_neutral": 0.0,
                }
            )
    return pd.DataFrame(records)


def add_sentiment_to_dataframe(
    df: pd.DataFrame,
    text_column: str = "text_clean",
    lang_column: str = "language",
    model: str = "transformer",
) -> pd.DataFrame:
    """Add sentiment columns to an existing DataFrame (legacy API).

    Args:
        df: Input DataFrame.
        text_column: Name of cleaned-text column.
        lang_column: Name of language column.
        model: ``"transformer"`` or ``"baseline"``.

    Returns:
        DataFrame with appended sentiment columns.
    """
    if df.empty or text_column not in df.columns:
        return df

    logger.info(
        "Running sentiment analysis on %d comments (model=%s) …", len(df), model
    )
    sentiment_df = predict_batch(
        df[text_column].tolist(),
        df[lang_column].tolist(),
        model=model,
    )
    result = pd.concat(
        [df.reset_index(drop=True), sentiment_df.reset_index(drop=True)],
        axis=1,
    )
    logger.info("Sentiment analysis complete.")
    return result
