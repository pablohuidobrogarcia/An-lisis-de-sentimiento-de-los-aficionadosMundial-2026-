"""
Topic modeling (BERTopic) and Named Entity Recognition (spaCy).

Overview
--------
- **BERTopic** extracts latent topics from the cleaned comment corpus using
  multilingual sentence embeddings, HDBSCAN clustering, and c-TF-IDF topic
  representations. Topics are labelled with human-readable names via
  key-term inspection.
- **NER** uses spaCy pipelines for Spanish and English to extract named
  entities: players, teams, brands, venues. A custom dictionary of known
  World Cup 2026 entities supplements the pre-trained NER models.
- **Topic evolution** over time is computed using BERTopic's
  ``topics_over_time``, which tracks how topic prevalence changes around
  match dates.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.config import (
    SPACY_MODELS,
    TOPIC_EMBEDDING_MODEL,
    TOPIC_LABEL_OVERRIDES,
    TOPIC_MAX_TOPICS,
    TOPIC_MIN_TOPICS,
)
from src.utils import setup_logger

logger = setup_logger(__name__)

# ── Lazy-loaded singletons ──────────────────────────────────────────────────
_BERTOPIC_MODEL = None
_NLP_PIPELINES: Dict[str, Any] = {}

# ── World Cup 2026 known entities ───────────────────────────────────────────
# Organised by team for easy extension.
KNOWN_PLAYERS: Dict[str, List[str]] = {
    "Spain": [
        "Pedri",
        "Gavi",
        "Lamine Yamal",
        "Nico Williams",
        "Rodri",
        "Unai Simón",
        "Dani Carvajal",
        "Aymeric Laporte",
        "Álvaro Morata",
        "Fermín López",
        "Mikel Merino",
        "Dani Olmo",
    ],
    "Argentina": [
        "Lionel Messi",
        "Ángel Di María",
        "Julián Álvarez",
        "Enzo Fernández",
        "Alexis Mac Allister",
        "Lautaro Martínez",
        "Emiliano Martínez",
        "Nicolás Otamendi",
        "Cristian Romero",
        "Rodrigo De Paul",
        "Leandro Paredes",
        "Nahuel Molina",
    ],
    "Brazil": [
        "Neymar",
        "Vinícius Jr",
        "Rodrygo",
        "Endrick",
        "Raphinha",
        "Casemiro",
        "Marquinhos",
        "Alisson",
        "Gabriel Martinelli",
        "Bruno Guimarães",
        "João Pedro",
        "Lucas Paquetá",
    ],
    "France": [
        "Kylian Mbappé",
        "Antoine Griezmann",
        "Eduardo Camavinga",
        "Aurélien Tchouaméni",
        "Mike Maignan",
        "Dayot Upamecano",
        "Ousmane Dembélé",
        "Randal Kolo Muani",
        "Olivier Giroud",
        "Theo Hernández",
        "Jules Koundé",
        "Adrien Rabiot",
    ],
    "England": [
        "Harry Kane",
        "Jude Bellingham",
        "Bukayo Saka",
        "Declan Rice",
        "Phil Foden",
        "Mason Mount",
        "Jack Grealish",
        "Marcus Rashford",
        "Jordan Pickford",
        "Kyle Walker",
        "John Stones",
        "Cole Palmer",
    ],
}

KNOWN_BRANDS: List[str] = [
    "Coca-Cola",
    "McDonald's",
    "Adidas",
    "Nike",
    "Puma",
    "Qatar Airways",
    "Visa",
    "Hyundai",
    "Kia",
    "Budweiser",
    "AB InBev",
    "Wanda Group",
    "Hisense",
    "Mengniu",
    "Globant",
    "Mountain Dew",
]

KNOWN_VENUES: List[str] = [
    "MetLife Stadium",
    "AT&T Stadium",
    "SoFi Stadium",
    "Arrowhead Stadium",
    "NRG Stadium",
    "Mercedes-Benz Stadium",
    "Levi's Stadium",
    "Gillette Stadium",
    "Hard Rock Stadium",
    "Allegiant Stadium",
    "Lincoln Financial Field",
    "Estadio Azteca",
    "Estadio Akron",
    "Estadio BBVA",
    "BC Place",
    "BMO Field",
]

_ALL_KNOWN_ENTITIES: List[str] = (
    KNOWN_PLAYERS["Spain"]
    + KNOWN_PLAYERS["Argentina"]
    + KNOWN_PLAYERS["Brazil"]
    + KNOWN_PLAYERS["France"]
    + KNOWN_PLAYERS["England"]
    + KNOWN_BRANDS
    + KNOWN_VENUES
)


def _get_spacy(language: str):
    """Load a spaCy pipeline for ``language``, caching the result."""
    global _NLP_PIPELINES
    if language not in _NLP_PIPELINES:
        model_name = SPACY_MODELS.get(language, "en_core_web_sm")
        try:
            import spacy

            _NLP_PIPELINES[language] = spacy.load(model_name)
            logger.info("spaCy model '%s' loaded", model_name)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found. Install with: "
                "python -m spacy download %s",
                model_name,
                model_name,
            )
            _NLP_PIPELINES[language] = None
    return _NLP_PIPELINES[language]


def _get_bertopic():
    """Lazy-load and return the BERTopic model singleton."""
    global _BERTOPIC_MODEL
    if _BERTOPIC_MODEL is None:
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer

            embedding_model = SentenceTransformer(TOPIC_EMBEDDING_MODEL)
            vectorizer = CountVectorizer(
                stop_words="english",  # vectorizer-level stop words
                ngram_range=(1, 2),
            )

            _BERTOPIC_MODEL = BERTopic(
                embedding_model=embedding_model,
                vectorizer_model=vectorizer,
                min_topic_size=TOPIC_MIN_TOPICS,
                nr_topics=TOPIC_MAX_TOPICS,
                verbose=False,
                calculate_probabilities=False,
            )
            logger.info("BERTopic model initialised")
        except Exception as exc:
            logger.error("Failed to load BERTopic: %s", exc)
            raise
    return _BERTOPIC_MODEL


# ── Topic labeling ────────────────────────────────────────────────────────────


def name_topics_interpretably(
    topic_model,
    topic_info: pd.DataFrame,
) -> Dict[int, str]:
    """Generate human-readable labels for each topic.

    Hybrid strategy:
    1. Check for an exact match in ``TOPIC_LABEL_OVERRIDES`` (concatenated keywords).
    2. Fall back to top-3 keywords joined by `` / ``.

    Args:
        topic_model: Fitted BERTopic model.
        topic_info: DataFrame from ``get_topic_info()``.

    Returns:
        Dict mapping topic ID to human-readable label.
    """
    labels: Dict[int, str] = {}

    for _, row in topic_info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            labels[tid] = "Outliers / Other"
            continue

        words = row.get("Representation", [])
        if not isinstance(words, list) or len(words) == 0:
            labels[tid] = f"Topic {tid}"
            continue

        key = " / ".join(words[:3])
        # Check manual overrides first
        if key in TOPIC_LABEL_OVERRIDES:
            labels[tid] = TOPIC_LABEL_OVERRIDES[key]
        else:
            labels[tid] = key

    return labels


def build_topic_model(
    docs: List[str],
    language: str = "multilingual",
    save_path: Optional[Path] = None,
):
    """Configure, fit, and return a BERTopic model on *docs*.

    Uses the multilingual sentence-transformer embedding model so both
    Spanish and English comments can be processed together.

    Args:
        docs: List of cleaned document strings.
        language: Ignored (the embedding model is fixed to multilingual).
        save_path: Optional path to save the fitted model.

    Returns:
        Tuple of ``(fitted_model, topics, probabilities)``.
    """
    model = _get_bertopic()
    logger.info("Fitting BERTopic on %d documents …", len(docs))
    topics, probs = model.fit_transform(docs)
    n_topics = len(set(topics)) - 1  # -1 for outlier cluster
    logger.info("BERTopic fitted: %d topics found (language=%s)", n_topics, language)
    if save_path:
        model.save(str(save_path), serialization="safetensors")
        logger.info("BERTopic model saved to %s", save_path)
    return model, topics, probs


def topics_over_time_df(
    model,
    docs: List[str],
    timestamps: pd.Series,
    nr_bins: int = 20,
    global_tuning: bool = False,
    evolution_tuning: bool = False,
) -> pd.DataFrame:
    """Wrapper around BERTopic's ``topics_over_time``.

    Performance notes
    -----------------
    - ``global_tuning=True`` makes each bin's c-TF-IDF depend on all others,
      which scales **O(n_bins x bin_size)** and is very slow for >50 bins.
      Keep ``False`` unless the temporal smoothing is essential.
    - ``evolution_tuning=True`` adds additional pairwise computation between
      consecutive bins (another multiplier). Keep ``False``.
    - *nr_bins* caps the number of equal-width time bins.  BERTopic defaults
      to one bin per unique timestamp (can be 10 000+); setting *nr_bins* to
      20-48 provides a 10-100x speedup.

    Args:
        model: Fitted BERTopic model.
        docs: Documents in the same order as used for fitting.
        timestamps: Corresponding datetime series.
        nr_bins: Number of equal-width time bins.  Default 20.  Pass 0 or a
            negative value to let BERTopic auto-detect (not recommended for
            large datasets).  Recommended: 20-48 for a multi-week span.
        global_tuning: Whether to use global c-TF-IDF tuning (slow).
            Default ``False``.
        evolution_tuning: Whether to compute evolutionary c-TF-IDF (slower).
            Default ``False``.

    Returns:
        DataFrame of topic prevalence over time, or empty if calculation fails.
    """
    try:
        kwargs: Dict[str, Any] = {
            "global_tuning": global_tuning,
            "evolution_tuning": evolution_tuning,
        }
        if nr_bins is not None and nr_bins > 0:
            kwargs["nr_bins"] = nr_bins

        result = model.topics_over_time(docs, timestamps.tolist(), **kwargs)
        return result
    except Exception as exc:
        logger.warning("topics_over_time failed: %s", exc)
        return pd.DataFrame()


# ── NER ─────────────────────────────────────────────────────────────────────


def extract_entities(
    text: str,
    language: str,
    use_custom_dict: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract named entities from ``text`` using spaCy + custom dictionary.

    Args:
        text: Input text (should be preprocessed).
        language: ``"es"`` or ``"en"``.
        use_custom_dict: Whether to merge known-entity matches on top of
            spaCy results.

    Returns:
        Dict with ``"spacy"`` and ``"custom"`` keys, each containing a list
        of ``{"text": ..., "label": ..., "start": ..., "end": ...}``.
    """
    result: Dict[str, List[Dict[str, Any]]] = {"spacy": [], "custom": []}

    nlp = _get_spacy(language)
    if nlp is not None:
        doc = nlp(text[:1_000_000])  # cap to avoid memory issues
        for ent in doc.ents:
            if ent.label_ in ("PER", "PERSON", "ORG", "GPE", "LOC", "MISC"):
                result["spacy"].append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                )

    if use_custom_dict:
        text_lower = text.lower()
        for entity in _ALL_KNOWN_ENTITIES:
            if entity.lower() in text_lower:
                idx = text_lower.index(entity.lower())
                result["custom"].append(
                    {
                        "text": entity,
                        "label": "KNOWN_ENTITY",
                        "start": idx,
                        "end": idx + len(entity),
                    }
                )

    return result


def add_entities_to_dataframe(
    df: pd.DataFrame,
    text_column: str = "text_clean",
    lang_column: str = "language",
) -> pd.DataFrame:
    """Add NER entity columns to a DataFrame.

    Adds:
    - ``entities_spacy``: list of spaCy-detected entities.
    - ``entities_custom``: list of custom-dictionary matches.
    - ``players_mentioned``: comma-separated known-player names found.
    - ``brands_mentioned``: comma-separated brand names found.

    Args:
        df: Input DataFrame.
        text_column: Column with cleaned text.
        lang_column: Column with language codes.

    Returns:
        DataFrame with entity columns appended.
    """
    if df.empty:
        return df

    players_found: List[str] = []
    brands_found: List[str] = []

    all_spacy: List[List[str]] = []
    all_custom: List[List[str]] = []

    for _, row in df.iterrows():
        text = str(row.get(text_column, ""))
        lang = str(row.get(lang_column, "en"))
        entities = extract_entities(text, lang)

        spacy_texts = [e["text"] for e in entities["spacy"]]
        custom_texts = [e["text"] for e in entities["custom"]]
        all_spacy.append(spacy_texts)
        all_custom.append(custom_texts)

        # Find known players / brands in custom matches
        pl = [
            e
            for e in custom_texts
            if any(
                e.lower() in [p.lower() for p in players_list]
                for players_list in KNOWN_PLAYERS.values()
            )
        ]
        br = [
            e for e in custom_texts if any(e.lower() == b.lower() for b in KNOWN_BRANDS)
        ]
        players_found.append(",".join(pl))
        brands_found.append(",".join(br))

    df["entities_spacy"] = all_spacy
    df["entities_custom"] = all_custom
    df["players_mentioned"] = players_found
    df["brands_mentioned"] = brands_found

    logger.info("NER complete: %d entities found", sum(len(e) for e in all_spacy))
    return df


# ── BERTopic ────────────────────────────────────────────────────────────────


def fit_topic_model(
    texts: List[str],
    save_path: Optional[Path] = None,
    **fit_kwargs: Any,
):
    """Fit BERTopic on a corpus of texts.

    Args:
        texts: List of cleaned documents.
        save_path: Optional path to save the fitted model.
        **fit_kwargs: Additional keyword arguments for ``BERTopic.fit()``.

    Returns:
        Fitted BERTopic model.
    """
    model = _get_bertopic()
    logger.info("Fitting BERTopic on %d documents …", len(texts))
    topics, probs = model.fit_transform(texts, **fit_kwargs)
    logger.info(
        "BERTopic fitted: %d topics found", len(set(topics)) - 1
    )  # -1 for outliers

    if save_path:
        model.save(str(save_path), serialization="safetensors")
        logger.info("BERTopic model saved to %s", save_path)

    return model


def get_topic_info(model) -> pd.DataFrame:
    """Return a DataFrame with topic name, size, and representative words."""
    info = model.get_topic_info()

    label_map = name_topics_interpretably(model, info)
    info["topic_label"] = info["Topic"].map(label_map).fillna("Outliers / Other")

    return info


def topics_over_time(
    model,
    texts: List[str],
    timestamps: List[pd.Timestamp],
    nr_bins: int = 20,
    global_tuning: bool = False,
    evolution_tuning: bool = False,
) -> pd.DataFrame:
    """Compute topic prevalence over time.

    Accepts the same performance-tuning parameters as
    :func:`topics_over_time_df`.  See that function for details.

    Args:
        model: Fitted BERTopic model.
        texts: Documents (same order as used for fitting).
        timestamps: Corresponding datetime stamps.
        nr_bins: Number of equal-width time bins.  Default 20.
        global_tuning: Whether to use global c-TF-IDF tuning.
        evolution_tuning: Whether to compute evolutionary c-TF-IDF.

    Returns:
        DataFrame with columns ``Topic``, ``Timestamp``, ``Frequency``,
        ``Words``, and ``topic_label``.
    """
    ts_series = pd.Series(timestamps)
    return topics_over_time_df(
        model,
        texts,
        ts_series,
        nr_bins=nr_bins,
        global_tuning=global_tuning,
        evolution_tuning=evolution_tuning,
    )


def add_topics_to_dataframe(
    df: pd.DataFrame,
    text_column: str = "text_clean",
    date_column: str = "created_utc",
    model_save_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Fit BERTopic on the DataFrame and add topic labels.

    Args:
        df: Input DataFrame with text and date columns.
        text_column: Column with cleaned text.
        date_column: Column with ISO datetime strings.
        model_save_path: Optional path to save the fitted model.

    Returns:
        Tuple of ``(df_with_topics, fitted_model)``. The DataFrame gains
        ``topic`` and ``topic_label`` columns.
    """
    texts = df[text_column].dropna().tolist()
    if not texts:
        logger.warning("No texts available for topic modeling.")
        return df, None

    model = fit_topic_model(texts, save_path=model_save_path)
    topics = model.topics_[: len(df)]  # align with original DF

    topic_info = get_topic_info(model)
    topic_map = dict(zip(topic_info["Topic"], topic_info["topic_label"]))

    df = df.copy()
    df["topic"] = topics
    df["topic_label"] = df["topic"].map(topic_map).fillna("Outliers / Other")

    logger.info("Topics assigned to %d documents", len(df))
    return df, model
