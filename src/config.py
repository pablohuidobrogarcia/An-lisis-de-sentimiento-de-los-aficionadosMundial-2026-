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
YOUTUBE_PUBLISHED_AFTER_DAYS: int = 3
YOUTUBE_PUBLISHED_BEFORE_DAYS: int = 3

# Budget / safety limits
COLLECTION_TIME_BUDGET_SECONDS: int = 480  # 8 minutes max per run
COLLECTION_QUOTA_BUDGET: int = 9_000  # stop before hitting the 10k daily limit
MATCH_PLAYED_BUFFER_HOURS: int = 3  # skip matches until 3h after kickoff
# After a match has been played and searched, skip re-searching if it's
# older than this many days (new reaction videos are unlikely to appear).
MATCH_SEARCH_WINDOW_DAYS: int = 3

YOUTUBE_SEARCH_TEMPLATES: List[str] = [
    "{team} {opponent} world cup 2026 highlights",
    "{team} {opponent} resumen mundial 2026",
    "{team} world cup 2026 match recap",
    "{team} mundial 2026 partido completo resumen",
]

# ── Keywords that indicate a team mention (for cross-team detection) ───────
TEAM_KEYWORDS: Dict[str, List[str]] = {
    "Spain": ["spain", "españa", "espan", "la roja", "selección española"],
    "Argentina": ["argentina", "albiceleste", "scaloneta", "mesi", "messi"],
    "Brazil": ["brazil", "brasil", "seleção", "canarinho", "vinicius", "neymar"],
    "France": ["france", "francia", "les bleus", "mbappe", "griezmann"],
    "England": ["england", "inglaterra", "three lions", "kane", "bellingham"],
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
TOPIC_MIN_TOPICS: int = 15
TOPIC_MAX_TOPICS: int = 25

# Manual overrides for interpretable topic labels (populate after inspecting
# topic_model.get_topic_info() output from a real run).
# Format: {"keyword1 / keyword2 / keyword3": "Custom Label"}
TOPIC_LABEL_OVERRIDES: Dict[str, str] = {
    # Curated after inspecting 20-topic run on Brazil vs Morocco (2,636 docs)
    "morocco / brazil / marruecos": "Morocco vs Brazil Match",
    "brazil / brasil / team": "Brazil Team Performance",
    "href / gracias / comment": "General Comments / Threads",
    "paraguay / que / el": "Paraguay Match Discussion",
    "endrick / ancelotti / thiago": "Endrick / Ancelotti / Thiago Silva",
    "team / usa / teams": "USA / Team Comparisons",
    "speed / travis / tv": "Content Creators / Media",
    "goal / gol / vini": "Goals / Vinicius Jr",
    "neymar / neymar neymar / injured": "Neymar Injury / Criticism",
    "football / futbol / el": "General Football Discussion",
    "video / highlights / videos": "Video / Highlight Quality",
    "argentina / los / la": "Argentina Comparison",
    "mexico / méxico / en": "Mexico Discussion",
    "game / partido / el": "Game Analysis / Opinions",
    "world cup / cup / world": "World Cup General",
    "match / old match / old": "Match Highlights Appreciation",
    "unidos / estados unidos / estados": "United States Hosting",
    "spanish / br / speak": "Language / Commentary",
    "women / commentator / voice": "Female Commentator Discussion",
    "time / años / 2026": "Time / Future Predictions",
}

# ── Sentiment analysis ──────────────────────────────────────────────────────
SENTIMENT_COLUMNS: Dict[str, str] = {
    "vader": "sentiment_vader",
    "pysentimiento": "sentiment_pysentimiento",
    "blob_es": "sentiment_blob_es",
}

# Time windows around matches (in hours) for causal analysis
MATCH_PRE_WINDOW_HOURS: int = 24
MATCH_POST_WINDOW_HOURS: int = 24

# ── Team name aliases for video relevance checking ──────────────────────────
TEAM_ALIASES: Dict[str, List[str]] = {
    "Spain": ["spain", "españa", "espania"],
    "Argentina": ["argentina"],
    "Brazil": ["brazil", "brasil"],
    "France": ["france", "francia"],
    "England": ["england", "inglaterra"],
    "Mexico": ["mexico", "méxico"],
    "South Africa": ["south africa", "sudáfrica", "sudafrica"],
    "Cape Verde Islands": ["cape verde", "cabo verde", "cape verde islands"],
    "Morocco": ["morocco", "marruecos"],
    "Senegal": ["senegal"],
    "Algeria": ["algeria", "argelia"],
    "Croatia": ["croatia", "croacia"],
    "Ghana": ["ghana"],
    "South Korea": ["south korea", "korea republic", "corea del sur", "korea"],
    "Czechia": ["czechia", "czech republic", "república checa", "chequia"],
    "Bosnia-Herzegovina": ["bosnia", "bosnia-herzegovina"],
    "Canada": ["canada", "canadá"],
    "United States": ["united states", "usa", "eeuu", "estados unidos"],
    "Paraguay": ["paraguay"],
    "Qatar": ["qatar", "catar"],
    "Switzerland": ["switzerland", "suiza", "suiza"],
    "Haiti": ["haiti", "haití"],
    "Scotland": ["scotland", "escocia"],
    "Australia": ["australia"],
    "Turkey": ["turkey", "turquía"],
    "Germany": ["germany", "alemania"],
    "Curacao": ["curaçao", "curacao"],
    "Netherlands": ["netherlands", "holanda", "países bajos"],
    "Japan": ["japan", "japón"],
    "Ivory Coast": ["ivory coast", "côte d'ivoire", "costa de marfil"],
    "Ecuador": ["ecuador"],
    "Sweden": ["sweden", "suecia"],
    "Tunisia": ["tunisia", "túnez"],
    "Belgium": ["belgium", "bélgica"],
    "Egypt": ["egypt", "egipto"],
    "Saudi Arabia": ["saudi arabia", "arabia saudí"],
    "Uruguay": ["uruguay"],
    "Iran": ["iran", "irán"],
    "New Zealand": ["new zealand", "nueva zelanda"],
    "Iraq": ["iraq", "irak"],
    "Norway": ["norway", "noruega"],
    "Austria": ["austria"],
    "Jordan": ["jordan", "jordania"],
    "Portugal": ["portugal"],
    "Congo DR": ["congo", "dr congo"],
    "Uzbekistan": ["uzbekistan", "uzbekistán"],
    "Colombia": ["colombia"],
    "Panama": ["panamá", "panama"],
}

FOOTBALL_KEYWORDS: List[str] = [
    "world cup",
    "mundial",
    "highlights",
    "resumen",
    "goals",
    "goles",
    "match",
    "partido",
    "fifa",
    "full match",
    "full game",
    "partido completo",
    "recap",
    "debut",
    "opening",
    "inaugural",
    "winner",
    "win",
    "victory",
    "victoria",
    "goal",
    "gol",
    "score",
    "resultado",
]

# ── Spam / bot pattern detection ────────────────────────────────────────────
SPAM_PHRASES: List[str] = [
    "deserves more views",
    "learned a lot",
    "explanation is exactly what i needed",
    "editing is outstanding",
    "clear and easy to follow",
    "subscribe to my channel",
    "subscribe to my",
    "check out my channel",
    "great content",
    "amazing content",
    "underrated video",
    "underrated channel",
    "why is this not viral",
    "this should go viral",
    "keep up the good work",
    "very informative video",
    "well explained",
    "nice video",
]

SHORT_COMMENT_MIN_WORDS: int = 3

# ── Evaluation ──────────────────────────────────────────────────────────────
EVAL_SAMPLE_SIZE: int = 150
EVAL_OUTPUT_FILE: str = "evaluation/manual_labels.csv"
