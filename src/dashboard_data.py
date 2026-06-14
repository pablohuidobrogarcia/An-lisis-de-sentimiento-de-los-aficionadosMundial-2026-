"""Data loading layer for the Streamlit dashboard.

Separates data access from presentation logic so the dashboard only
worries about rendering.
"""


import pandas as pd
import streamlit as st

from src.config import PROCESSED_DIR
from src.results_api import build_match_results_summary
from src.utils import load_dataframe


@st.cache_data(ttl=300)
def load_dashboard_data() -> pd.DataFrame:
    """Load the full processed dataset (topics + NER + sentiment).

    Returns:
        DataFrame with all 33 columns, or empty DataFrame if the
        processed file does not exist (notebooks 01–04 not yet run).
    """
    candidates = [
        PROCESSED_DIR / "comentarios_topics_ner" / "comentarios_topics_ner.parquet",
        PROCESSED_DIR / "comentarios_sentimiento" / "comentarios_sentimiento.parquet",
        PROCESSED_DIR / "comentarios_limpios" / "comentarios_limpios.parquet",
    ]
    for path in candidates:
        if path.exists():
            df = load_dataframe(str(path))
            if not df.empty:
                return df
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_match_results() -> pd.DataFrame:
    """Fetch match results via ``build_match_results_summary``.

    The underlying function uses a 1-hour file cache internally, and
    Streamlit's ``@st.cache_data`` adds another 1-hour in-memory layer.

    Returns:
        DataFrame with columns: ``team``, ``opponent``, ``match_date``,
        ``outcome``, ``score``, ``status``, ``stage``, ``match_id``.
    """
    return build_match_results_summary(use_cache=True)
