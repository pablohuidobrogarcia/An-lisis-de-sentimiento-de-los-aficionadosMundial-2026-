"""
Unified sentiment analysis pipeline supporting Spanish and English.

Architecture
------------
- **Primary model** (ES): ``pysentimiento`` BERT-based classifier, fine-tuned
  on Spanish social-media text (``pysentimiento/bert-base-spanish-wwm-uncased``).
- **Primary model** (EN): ``cardiffnlp/twitter-roberta-base-sentiment`` via the
  HuggingFace ``transformers`` pipeline, the de-facto standard for English
  social-media sentiment.
- **Baseline** (EN): VADER — a lexicon/rule-based model. Fast, interpretable,
  but misses sarcasm and context.
- **Baseline** (ES): A Spanish polarity-lexicon approach using a simple
  word-count of positive/negative terms from the NRC-EmoLex translated list.
  *Justification*: VADER is English-only. TextBlob-es is unmaintained and
  unreliable for social-media text. A polarity-lexicon baseline is transparent
  and sufficient for comparison.

Usage
-----
Call :func:`predict_sentiment` with the text and its language code to get a
unified ``(label, scores_dict)`` result. The pipeline handles the model
dispatch internally.
"""

from typing import Dict, List

import pandas as pd

from src.utils import setup_logger

logger = setup_logger(__name__)

# ── Lazy-loaded model singletons ──────────────────────────────────────────
_PYSENTIMIENTO = None
_HF_ROBERTA = None
_VADER = None

# Spanish polarity lexicon (curated subset for social-media football context)
# Based on NRC-EmoLex Spanish translation
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


def _get_pysentimiento():
    global _PYSENTIMIENTO
    if _PYSENTIMIENTO is None:
        try:
            from pysentimiento import create_analyzer

            _PYSENTIMIENTO = create_analyzer(task="sentiment", lang="es")
            logger.info("pysentimiento analyzer loaded")
        except Exception as exc:
            logger.warning("pysentimiento unavailable: %s", exc)
    return _PYSENTIMIENTO


def _get_roberta():
    global _HF_ROBERTA
    if _HF_ROBERTA is None:
        try:
            from transformers import pipeline

            _HF_ROBERTA = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
                return_all_scores=True,
                max_length=512,
                truncation=True,
            )
            logger.info("HuggingFace RoBERTa model loaded")
        except Exception as exc:
            logger.warning("HuggingFace RoBERTa unavailable: %s", exc)
    return _HF_ROBERTA


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


# ── Spanish lexicon baseline ─────────────────────────────────────────────


def _baseline_es(text: str) -> Dict[str, float]:
    """Simple polarity-lexicon baseline for Spanish.

    Counts positive and negative word occurrences and returns normalised scores.
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


# ── Prediction functions ─────────────────────────────────────────────────


def _predict_pysentimiento(text: str) -> Dict[str, float]:
    """Predict sentiment using pysentimiento (Spanish BERT)."""
    analyzer = _get_pysentimiento()
    if analyzer is None:
        raise RuntimeError("pysentimiento not available")
    result = analyzer.predict(text)
    return {
        "positive": result.probas.get("POS", 0.0),
        "negative": result.probas.get("NEG", 0.0),
        "neutral": result.probas.get("NEU", 0.0),
    }


def _predict_roberta(text: str) -> Dict[str, float]:
    """Predict sentiment using HuggingFace RoBERTa (English)."""
    pipe = _get_roberta()
    if pipe is None:
        raise RuntimeError("HuggingFace RoBERTa not available")
    result = pipe(text)[0]
    scores = {item["label"]: item["score"] for item in result}
    # Map: LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
    return {
        "positive": scores.get("LABEL_2", 0.0),
        "negative": scores.get("LABEL_0", 0.0),
        "neutral": scores.get("LABEL_1", 0.0),
    }


def _predict_vader(text: str) -> Dict[str, float]:
    """Predict sentiment using VADER (English baseline).

    Returns positive/negative/neutral scores (compound decomposed).
    """
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


# ── Unified public API ────────────────────────────────────────────────────


def predict_sentiment(
    text: str,
    language: str,
    model: str = "transformer",
) -> Dict[str, Dict[str, float]]:
    """Run sentiment prediction on ``text`` using the specified model.

    Args:
        text: Cleaned text string.
        language: ISO 639-1 code (``"es"`` or ``"en"``).
        model: ``"transformer"`` (BERT/RoBERTa, default) or ``"baseline"``
            (VADER for EN, polarity-lexicon for ES).

    Returns:
        Nested dict::

            {
                "transformer": {"positive": ..., "negative": ..., "neutral": ...},
                "baseline": {"positive": ..., "negative": ..., "neutral": ...},
            }
    """
    result: Dict[str, Dict[str, float]] = {}

    # Primary model
    if model == "transformer":
        if language == "es":
            result["transformer"] = _predict_pysentimiento(text)
        else:
            result["transformer"] = _predict_roberta(text)
    else:
        result["transformer"] = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    # Baseline
    if language == "es":
        result["baseline"] = _baseline_es(text)
    else:
        result["baseline"] = _predict_vader(text)

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
        DataFrame with columns:
        ``sentiment_label``, ``sentiment_positive``, ``sentiment_negative``,
        ``sentiment_neutral``, ``sentiment_baseline_label``, and the
        corresponding baseline scores.
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
            logger.warning("Sentiment prediction failed for text: %s", exc)
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
    """Add sentiment columns to an existing DataFrame.

    Args:
        df: Input DataFrame with ``text_clean`` and ``language`` columns.
        text_column: Name of the column with cleaned text.
        lang_column: Name of the column with language codes.
        model: ``"transformer"`` or ``"baseline"``.

    Returns:
        DataFrame with appended sentiment columns.
    """
    if df.empty or text_column not in df.columns:
        return df

    logger.info(
        "Running sentiment analysis on %d comments (model=%s) …",
        len(df),
        model,
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
