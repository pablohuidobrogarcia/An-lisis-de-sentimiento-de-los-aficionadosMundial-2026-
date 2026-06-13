"""
Configuration constants for the World Cup 2026 Sentiment Analysis project.

Centralizes paths, API settings, team definitions, YouTube search config,
and model parameters.

API keys are resolved in priority order:
1. Direct assignment in this file (debug)
2. Environment variable (e.g. ``os.environ["YOUTUBE_API_KEY"]``) — used
   by GitHub Actions when secrets are injected.
3. ``.env`` file via ``python-dotenv`` (local development).
"""

import os
from pathlib import Path
from typing import Dict, List

# ── Load .env if available (local dev) ──────────────────────────────────────
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(str(_env_path))
    except ImportError:
        pass  # dotenv not installed; rely on env vars or defaults

# ── Project paths ───────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
CHECKPOINT_DIR: Path = RAW_DIR / ".checkpoints"

for _dir in (DATA_DIR, RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Target teams ────────────────────────────────────────────────────────────
TARGET_TEAMS: List[str] = ["Spain", "Argentina", "Brazil", "France", "England"]

# ── YouTube channel IDs (official / sports news) for video search ──────────
YOUTUBE_CHANNELS: List[str] = [
    "UCAJfDidh9NlUwBalkHjNnQg",  # FIFA
    "UC4UxZcQo8ieVY0bYClqbS1Q",  # ESPN FC
    "UCqFMq7DcDkMV0Gq2FjqP5Jw",  # beIN SPORTS
    "UCz5fM7uPzQ3WzQj5vT0Q5aA",  # TUDN USA
    "UCZFMm1mHpXqkGKljz2VqRZQ",  # FOX Soccer
]

# ── YouTube Data API v3 ────────────────────────────────────────────────────
YOUTUBE_API_KEY: str = os.environ.get("YOUTUBE_API_KEY", "")
YOUTUBE_API_SERVICE_NAME: str = "youtube"
YOUTUBE_API_VERSION: str = "v3"

YOUTUBE_DAILY_QUOTA: int = 10_000
YOUTUBE_QUOTA_SAFETY_MARGIN: int = 500
YOUTUBE_MAX_RESULTS_PER_SEARCH: int = 10
YOUTUBE_MAX_COMMENTS_PER_VIDEO: int = 500
YOUTUBE_SLEEP_BETWEEN_CALLS: float = 0.5

YOUTUBE_SEARCH_TEMPLATES: List[str] = [
    "{team} {opponent} world cup 2026 highlights",
    "{team} {opponent} resumen mundial 2026",
    "{team} world cup 2026 match recap",
    "{team} mundial 2026 partido completo resumen",
]

# ── Keywords that indicate a team mention (for cross-team detection) ───────
TEAM_KEYWORDS: Dict[str, List[str]] = {
    "Spain":    ["spain", "españa", "espan", "la roja", "selección española"],
    "Argentina":["argentina", "albiceleste", "scaloneta", "mesi", "messi"],
    "Brazil":   ["brazil", "brasil", "seleção", "canarinho", "vinicius", "neymar"],
    "France":   ["france", "francia", "les bleus", "mbappe", "griezmann"],
    "England":  ["england", "inglaterra", "three lions", "kane", "bellingham"],
}

# ── football-data.org API ───────────────────────────────────────────────────
FOOTBALL_DATA_API_KEY: str = os.environ.get("FOOTBALL_DATA_API_KEY", "")
FOOTBALL_DATA_BASE_URL: str = "https://api.football-data.org/v4/"
FOOTBALL_DATA_HEADERS: Dict[str, str] = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
FOOTBALL_DATA_RATE_LIMIT: float = 1.0

# ── NLP / Model settings ───────────────────────────────────────────────────
LANG_DETECT_CONFIDENCE_THRESHOLD: float = 0.7
SUPPORTED_LANGUAGES: List[str] = ["es", "en"]
SPACY_MODELS: Dict[str, str] = {
    "es": "es_core_news_sm",
    "en": "en_core_web_sm",
}
TOPIC_EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
TOPIC_MIN_TOPICS: int = 5
TOPIC_MAX_TOPICS: int = 25

# ── Sentiment analysis ──────────────────────────────────────────────────────
SENTIMENT_COLUMNS: Dict[str, str] = {
    "vader": "sentiment_vader",
    "pysentimiento": "sentiment_pysentimiento",
    "blob_es": "sentiment_blob_es",
}

# Time windows around matches (in hours) for causal analysis
MATCH_PRE_WINDOW_HOURS: int = 24
MATCH_POST_WINDOW_HOURS: int = 24

# ── Evaluation ──────────────────────────────────────────────────────────────
EVAL_SAMPLE_SIZE: int = 150
EVAL_OUTPUT_FILE: str = "evaluation/manual_labels.csv"

