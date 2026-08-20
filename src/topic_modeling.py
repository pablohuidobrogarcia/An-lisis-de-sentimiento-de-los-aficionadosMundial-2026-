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

import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from src.config import (
    SPACY_MODELS,
    TOPIC_EMBEDDING_MODEL,
    TOPIC_EXCLUDE,
    TOPIC_FINAL_LABELS,
    TOPIC_LABEL_OVERRIDES,
    TOPIC_MAX_TOPICS,
    TOPIC_MERGE,
    TOPIC_MIN_TOPICS,
    TOPIC_REDUCE_OUTLIERS,
    TOPIC_REDUCE_STRATEGY,
    TOPIC_REDUCE_THRESHOLD,
)
from src.utils import setup_logger

logger = setup_logger(__name__)

# ── Stop words (English + Spanish) ───────────────────────────────────────────
_SPANISH_STOP_WORDS: set = {
    "a",
    "al",
    "algo",
    "algunas",
    "algunos",
    "ante",
    "antes",
    "como",
    "con",
    "contra",
    "cual",
    "cuando",
    "de",
    "del",
    "desde",
    "donde",
    "durante",
    "e",
    "el",
    "ella",
    "ellas",
    "ellos",
    "en",
    "entre",
    "era",
    "erais",
    "eran",
    "eras",
    "eres",
    "es",
    "esa",
    "esas",
    "ese",
    "eso",
    "esos",
    "esta",
    "estaba",
    "estaban",
    "estado",
    "estamos",
    "estan",
    "estar",
    "estas",
    "este",
    "esto",
    "estos",
    "etc",
    "fue",
    "han",
    "has",
    "hasta",
    "hay",
    "la",
    "las",
    "lo",
    "los",
    "más",
    "menos",
    "mi",
    "muy",
    "ni",
    "no",
    "nos",
    "o",
    "os",
    "para",
    "pero",
    "por",
    "porque",
    "que",
    "se",
    "sí",
    "sido",
    "sin",
    "sobre",
    "son",
    "su",
    "tal",
    "tan",
    "tanto",
    "te",
    "tiene",
    "todo",
    "todos",
    "un",
    "una",
    "unas",
    "unos",
    "va",
    "y",
    "ya",
    "él",
}

# ── Lazy-loaded singletons ──────────────────────────────────────────────────
_BERTOPIC_MODEL = None
_NLP_PIPELINES: dict[str, Any] = {}

# ── World Cup 2026 known entities ───────────────────────────────────────────
# Key: canonical player name.  Value: list of known variants (lowercase).
# Duplicating with/without accent is unnecessary — _ALL_KNOWN_ENTITIES
# auto-generates accent‑free variants at build time via _strip_accents().

KNOWN_PLAYERS: dict[str, list[str]] = {
    # ── ESPAÑA (Luis de la Fuente) ──────────────────────────────────────────
    "Unai Simón": ["unai simón", "unai simon", "unai", "simón", "simon"],
    "David Raya": ["david raya", "raya"],
    "Joan García": ["joan garcía", "joan garcia"],
    "Marc Cucurella": ["marc cucurella", "cucurella"],
    "Alejandro Grimaldo": [
        "alejandro grimaldo",
        "álex grimaldo",
        "alex grimaldo",
        "grimaldo",
    ],
    "Pau Cubarsí": ["pau cubarsí", "pau cubarsi", "cubarsí", "cubarsi"],
    "Aymeric Laporte": ["aymeric laporte", "laporte"],
    "Marc Pubill": ["marc pubill", "pubill"],
    "Eric García": ["eric garcía", "eric garcia"],
    "Marcos Llorente": ["marcos llorente", "llorente"],
    "Pedro Porro": ["pedro porro", "porro"],
    "Pablo Gavi": ["pablo gavi", "gavi"],
    "Pedri González": ["pedri gonzález", "pedri gonzalez", "pedri"],
    "Fabián Ruiz": ["fabián ruiz", "fabian ruiz", "fabián"],
    "Martín Zubimendi": ["martín zubimendi", "zubimendi"],
    "Rodri Hernández": [
        "rodri hernández",
        "rodrigo hernández",
        "rodri hernandez",
        "rodrigo hernandez",
        "rodri",
    ],
    "Álex Baena": ["álex baena", "alex baena", "baena"],
    "Mikel Merino": ["mikel merino", "merino"],
    "Lamine Yamal": ["lamine yamal", "lamine", "yamal"],
    "Nico Williams": ["nico williams", "williams"],
    "Yéremy Pino": ["yéremy pino", "yeremy pino", "pino"],
    "Ferran Torres": ["ferran torres", "ferran"],
    "Álvaro Morata": ["álvaro morata", "alvaro morata", "morata"],
    "Dani Olmo": ["dani olmo", "olmo"],
    "Mikel Oyarzabal": ["mikel oyarzabal", "oyarzabal"],
    "Víctor Muñoz": ["víctor muñoz", "victor munoz"],
    # ── ARGENTINA (Lionel Scaloni) ──────────────────────────────────────────
    "Emiliano Martínez": ["emiliano martínez", "emiliano martinez"],
    "Gerónimo Rulli": ["gerónimo rulli", "geronimo rulli", "rulli"],
    "Juan Musso": ["juan musso", "musso"],
    "Gonzalo Montiel": ["gonzalo montiel", "montiel"],
    "Nahuel Molina": ["nahuel molina", "molina"],
    "Lisandro Martínez": ["lisandro martínez", "lisandro martinez"],
    "Nicolás Otamendi": ["nicolás otamendi", "otamendi"],
    "Leonardo Balerdi": ["leonardo balerdi", "balerdi"],
    "Cristian Romero": ["cristian romero", "romero"],
    "Nicolás Tagliafico": ["nicolás tagliafico", "tagliafico"],
    "Facundo Medina": ["facundo medina", "medina"],
    "Giovani Lo Celso": ["giovani lo celso", "lo celso"],
    "Leandro Paredes": ["leandro paredes", "paredes"],
    "Rodrigo De Paul": ["rodrigo de paul", "de paul"],
    "Exequiel Palacios": ["exequiel palacios", "palacios"],
    "Enzo Fernández": ["enzo fernández", "enzo fernandez"],
    "Alexis Mac Allister": ["alexis mac allister", "mac allister"],
    "Valentín Barco": ["valentín barco", "valentin barco"],
    "Lionel Messi": ["lionel messi", "messi"],
    "Nicolás González": ["nicolás gonzález", "nico gonzález", "nico gonzalez"],
    "Giuliano Simeone": ["giuliano simeone", "simeone"],
    "Lautaro Martínez": ["lautaro martínez", "lautaro"],
    "José Manuel López": ["josé manuel lópez", "jose lopez"],
    "Julián Álvarez": ["julián álvarez", "julian alvarez", "álvarez"],
    "Thiago Almada": ["thiago almada", "almada"],
    "Nico Paz": ["nico paz", "paz"],
    # ── BRASIL (Carlo Ancelotti) ────────────────────────────────────────────
    "Alisson Becker": ["alisson becker", "alisson"],
    "Ederson": ["ederson"],
    "Weverton": ["weverton"],
    "Marquinhos": ["marquinhos"],
    "Gabriel Magalhães": ["gabriel magalhães", "gabriel magalhaes"],
    "Bremer": ["bremer"],
    "Danilo": ["danilo"],
    "Alex Sandro": ["alex sandro"],
    "Léo Pereira": ["léo pereira", "leo pereira"],
    "Douglas Santos": ["douglas santos"],
    "Wesley": ["wesley"],
    "Ibañez": ["ibañez", "ibanez"],
    "Casemiro": ["casemiro"],
    "Bruno Guimarães": ["bruno guimarães", "bruno guimaraes"],
    "Lucas Paquetá": ["lucas paquetá", "lucas paqueta", "paquetá", "paqueta"],
    "Fabinho": ["fabinho"],
    "Danilo Santos": ["danilo santos"],
    "Vinícius Júnior": ["vinícius júnior", "vinicius junior", "vinícius"],
    "Raphinha": ["raphinha"],
    "Matheus Cunha": ["matheus cunha"],
    "Neymar": ["neymar"],
    "Endrick": ["endrick"],
    "Gabriel Martinelli": ["gabriel martinelli", "martinelli"],
    "Igor Thiago": ["igor thiago"],
    "Luiz Henrique": ["luiz henrique"],
    "Rayan": ["rayan"],
    # ── FRANCIA (Didier Deschamps) ──────────────────────────────────────────
    "Mike Maignan": ["mike maignan", "maignan"],
    "Brice Samba": ["brice samba", "samba"],
    "Robin Risser": ["robin risser", "risser"],
    "Jules Koundé": ["jules koundé", "jules kounde", "koundé", "kounde"],
    "Malo Gusto": ["malo gusto", "gusto"],
    "William Saliba": ["william saliba", "saliba"],
    "Dayot Upamecano": ["dayot upamecano", "upamecano"],
    "Ibrahima Konaté": ["ibrahima konaté", "ibrahima konate", "konaté", "konate"],
    "Lucas Hernández": ["lucas hernández", "lucas hernandez"],
    "Théo Hernández": ["théo hernández", "theo hernandez"],
    "Lucas Digne": ["lucas digne", "digne"],
    "Maxence Lacroix": ["maxence lacroix", "lacroix"],
    "Aurélien Tchouaméni": ["aurélien tchouaméni", "tchouameni"],
    "Adrien Rabiot": ["adrien rabiot", "rabiot"],
    "N'Golo Kanté": ["n'golo kanté", "ngolo kante", "kanté", "kante"],
    "Warren Zaïre-Emery": ["warren zaïre-emery", "zaire-emery"],
    "Manu Koné": ["manu koné", "manu kone"],
    "Kylian Mbappé": ["kylian mbappé", "kylian mbappe", "mbappé", "mbappe"],
    "Michael Olise": ["michael olise", "olise"],
    "Ousmane Dembélé": ["ousmane dembélé", "ousmane dembele", "dembélé"],
    "Désiré Doué": ["désiré doué", "desire doue"],
    "Bradley Barcola": ["bradley barcola", "barcola"],
    "Marcus Thuram": ["marcus thuram", "thuram"],
    "Maghnes Akliouche": ["maghnes akliouche", "akliouche"],
    "Jean-Philippe Mateta": ["jean-philippe mateta", "mateta"],
    "Rayan Cherki": ["rayan cherki", "cherki"],
    # ── INGLATERRA (Thomas Tuchel) ───────────────────────────────────────────
    "Jordan Pickford": ["jordan pickford", "pickford"],
    "James Trafford": ["james trafford", "trafford"],
    "Dean Henderson": ["dean henderson"],
    "Dan Burn": ["dan burn", "burn"],
    "Marc Guéhi": ["marc guéhi", "marc guehi", "guéhi", "guehi"],
    "Reece James": ["reece james"],
    "Ezri Konsa": ["ezri konsa", "konsa"],
    "Tino Livramento": ["tino livramento", "livramento"],
    "Nico O'Reilly": ["nico o'reilly", "nico oreilly"],
    "Jarell Quansah": ["jarell quansah", "quansah"],
    "Djed Spence": ["djed spence", "spence"],
    "John Stones": ["john stones", "stones"],
    "Elliott Anderson": ["elliott anderson"],
    "Jude Bellingham": ["jude bellingham", "bellingham"],
    "Eberechi Eze": ["eberechi eze", "eze"],
    "Jordan Henderson": ["jordan henderson"],
    "Kobbie Mainoo": ["kobbie mainoo", "mainoo"],
    "Declan Rice": ["declan rice"],
    "Morgan Rogers": ["morgan rogers"],
    "Anthony Gordon": ["anthony gordon", "gordon"],
    "Harry Kane": ["harry kane", "kane"],
    "Noni Madueke": ["noni madueke", "madueke"],
    "Marcus Rashford": ["marcus rashford", "rashford"],
    "Bukayo Saka": ["bukayo saka", "saka"],
    "Ivan Toney": ["ivan toney", "toney"],
    "Ollie Watkins": ["ollie watkins", "watkins"],
}

KNOWN_BRANDS: list[str] = [
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

KNOWN_VENUES: list[str] = [
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


def _strip_accents(text: str) -> str:
    """Remove combining diacritics (accents, tildes, cedillas, etc.)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


# _ALL_KNOWN_ENTITIES is a list of (alias_stripped, canonical_name, entity_type)
# sorted by alias length descending so that longer (more specific) aliases
# are matched first.  Accent‑free variants are auto‑generated.
_ALL_KNOWN_ENTITIES: list[tuple[str, str, str]] = []

for canonical, aliases in KNOWN_PLAYERS.items():
    seen: set = set()
    for alias in aliases:
        flat = _strip_accents(alias.lower())
        if flat not in seen:
            seen.add(flat)
            _ALL_KNOWN_ENTITIES.append((flat, canonical, "PLAYER"))

for brand in KNOWN_BRANDS:
    flat = _strip_accents(brand.lower())
    _ALL_KNOWN_ENTITIES.append((flat, brand, "BRAND"))

for venue in KNOWN_VENUES:
    flat = _strip_accents(venue.lower())
    _ALL_KNOWN_ENTITIES.append((flat, venue, "VENUE"))

_ALL_KNOWN_ENTITIES.sort(key=lambda x: len(x[0]), reverse=True)


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
                stop_words=list(ENGLISH_STOP_WORDS) + list(_SPANISH_STOP_WORDS),
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
) -> dict[int, str]:
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
    labels: dict[int, str] = {}

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
    docs: list[str],
    language: str = "multilingual",
    save_path: Path | None = None,
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
    docs: list[str],
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
        kwargs: dict[str, Any] = {
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
) -> dict[str, list[dict[str, Any]]]:
    """Extract named entities from ``text`` using spaCy + custom dictionary.

    The custom‑dictionary pass uses ``\\b`` word‑boundary regex, accent‑free
    normalisation, and length‑descending alias priority to minimise false
    positives.  Each canonical player is recorded at most once per call.

    Args:
        text: Input text (should be preprocessed).
        language: ``"es"`` or ``"en"``.
        use_custom_dict: Whether to merge known-entity matches on top of
            spaCy results.

    Returns:
        Dict with ``"spacy"`` and ``"custom"`` keys, each containing a list
        of ``{"text": ..., "label": ..., "start": ..., "end": ...}``.

        Each **custom** entry now also includes a ``"matched_alias"`` key
        (the specific alias variant that fired the match) for downstream
        auditing.  The ``"text"`` field is always the **canonical** name.
    """
    result: dict[str, list[dict[str, Any]]] = {"spacy": [], "custom": []}

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
        text_flat = _strip_accents(text.lower())
        matched_canonicals: set = set()
        occupied_spans: list = []  # (start, end) of already-matched spans

        for alias_flat, canonical, etype in _ALL_KNOWN_ENTITIES:
            if canonical in matched_canonicals:
                continue

            pattern = r"\b" + re.escape(alias_flat) + r"\b"
            for m in re.finditer(pattern, text_flat):
                span_start = m.start()
                span_end = m.end()

                # Skip if this span is fully inside an already-occupied span
                # (avoids contaminating e.g. "Danilo" inside "Danilo Santos")
                is_inside = any(
                    s <= span_start and span_end <= e for s, e in occupied_spans
                )
                if is_inside:
                    continue

                matched_canonicals.add(canonical)
                occupied_spans.append((span_start, span_end))
                result["custom"].append(
                    {
                        "text": canonical,
                        "matched_alias": alias_flat,
                        "label": "KNOWN_ENTITY",
                        "start": span_start,
                        "end": span_end,
                    }
                )
                break  # only the first non-overlapping occurrence

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

    players_found: list[str] = []
    brands_found: list[str] = []

    all_spacy: list[list[str]] = []
    all_custom: list[list[str]] = []

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
    texts: list[str],
    save_path: Path | None = None,
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
    topics, _probs = model.fit_transform(texts, **fit_kwargs)
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
    texts: list[str],
    timestamps: list[pd.Timestamp],
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
    model_save_path: Path | None = None,
) -> tuple[pd.DataFrame, Any]:
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

    if TOPIC_REDUCE_OUTLIERS:
        logger.info(
            "Reducing outliers (strategy=%s, threshold=%s) …",
            TOPIC_REDUCE_STRATEGY,
            TOPIC_REDUCE_THRESHOLD,
        )
        new_topics = model.reduce_outliers(
            texts,
            model.topics_,
            strategy=TOPIC_REDUCE_STRATEGY,
            threshold=TOPIC_REDUCE_THRESHOLD,
        )
        model.topics_ = new_topics
        n_outliers = sum(1 for t in new_topics if t == -1)
        logger.info(
            "Outliers after reduction: %d (%.1f%%)",
            n_outliers,
            100 * n_outliers / len(new_topics),
        )

    topics = model.topics_[: len(df)]  # align with original DF

    topic_info = get_topic_info(model)
    topic_map = dict(zip(topic_info["Topic"], topic_info["topic_label"]))

    df = df.copy()
    df["topic"] = topics
    df["topic_label"] = df["topic"].map(topic_map).fillna("Outliers / Other")

    logger.info("Topics assigned to %d documents", len(df))
    return df, model


def consolidate_topics(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude spam/off-topic topics and merge semantically similar ones.

    This is a post-processing step applied after BERTopic assigns initial
    topic IDs.  It performs three operations:

    1. **Exclude** — removes rows whose ``topic`` is in ``TOPIC_EXCLUDE``
       (currently Topic 11: off-topic cricket; Topic 21: Robi spam).
    2. **Merge** — reassigns source topics into a target topic per
       ``TOPIC_MERGE`` so that semantically equivalent clusters share an ID.
    3. **Relabel** — applies ``TOPIC_FINAL_LABELS`` to produce human-readable
       names for the consolidated groups.

    Args:
        df: DataFrame with at least ``topic`` and ``topic_label`` columns
            (as returned by :func:`add_topics_to_dataframe`).

    Returns:
        Consolidated DataFrame with excluded rows removed, topic IDs
        reassigned, and labels updated.
    """
    n_before = len(df)

    # ── 1. Exclude ─────────────────────────────────────────────────────
    mask_exclude = df["topic"].isin(TOPIC_EXCLUDE)
    n_excluded_total = int(mask_exclude.sum())
    n_by_topic: dict[int, int] = {}
    for t in TOPIC_EXCLUDE:
        n_by_topic[t] = int((df["topic"] == t).sum())
    df = df[~mask_exclude].copy()

    # ── 2. Merge ───────────────────────────────────────────────────────
    n_merged_by_src: dict[int, int] = {}
    for src, dst in TOPIC_MERGE.items():
        n = int((df["topic"] == src).sum())
        n_merged_by_src[src] = n
        df.loc[df["topic"] == src, "topic"] = dst

    # ── 3. Relabel ─────────────────────────────────────────────────────
    for tid, label in TOPIC_FINAL_LABELS.items():
        df.loc[df["topic"] == tid, "topic_label"] = label

    n_after = len(df)
    logger.info(
        "Consolidation: excluded %d rows (%s), merged %d, %d topics remain",
        n_excluded_total,
        dict(sorted(n_by_topic.items())),
        sum(n_merged_by_src.values()),
        df["topic"].nunique(),
    )

    # Attach stats for downstream reporting (e.g. dry-run)
    df.attrs["consolidation_stats"] = {
        "n_before": n_before,
        "n_after": n_after,
        "n_excluded_total": n_excluded_total,
        "n_excluded_by_topic": n_by_topic,
        "n_merged_by_src": n_merged_by_src,
        "n_topics_active": int(df["topic"].nunique()),
    }

    return df
