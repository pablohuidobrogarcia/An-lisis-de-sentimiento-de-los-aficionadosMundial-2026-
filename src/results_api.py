"""
football-data.org API client for World Cup 2026 fixtures and results.

Provides functions to fetch match data, map comments to the nearest match
time window, and compute pre/post sentiment comparisons with statistical
testing.

Caching
-------
The API free tier (10 req/min) is respected via a file-based cache in
``data/raw/.cache/football_data_matches.json``.  Cached responses are reused
if they are less than 1 hour old, avoiding unnecessary API calls during
notebook development.
"""

import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from scipy.stats import mannwhitneyu

from src.config import (
    FOOTBALL_DATA_API_KEY,
    FOOTBALL_DATA_BASE_URL,
    FOOTBALL_DATA_HEADERS,
    FOOTBALL_DATA_RATE_LIMIT,
    MATCH_POST_WINDOW_HOURS,
    MATCH_PRE_WINDOW_HOURS,
    PROCESSED_DIR,
    TARGET_TEAMS,
)
from src.utils import load_json, save_dataframe, save_json, setup_logger

logger = setup_logger(__name__)

# Cache config — stored in processed/ (which is gitignored) so it survives
# across workflow runs but is never committed to the repository.
CACHE_DIR: Path = PROCESSED_DIR / ".cache"
CACHE_FILE: Path = CACHE_DIR / "football_data_matches.json"
CACHE_TTL_SECONDS: int = 3600  # 1 hour

# Legacy cache path for backward compatibility
FIXTURES_CACHE_FILE = PROCESSED_DIR / "fixtures_cache.json"
RESULTS_FILE = PROCESSED_DIR / "match_results"

CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── API client ──────────────────────────────────────────────────────────────


def _build_headers() -> Dict[str, str]:
    """Return request headers with API key."""
    if not FOOTBALL_DATA_API_KEY:
        logger.warning(
            "FOOTBALL_DATA_API_KEY not set. "
            "Set it in your .env file to fetch real match data."
        )
    return {
        "X-Auth-Token": FOOTBALL_DATA_API_KEY,
        **FOOTBALL_DATA_HEADERS,
    }


def _api_get(endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
    """Make a rate-limited GET request to football-data.org.

    Args:
        endpoint: API endpoint path (e.g. ``"competitions/2000/matches"``).
        params: Query parameters.

    Returns:
        JSON response as dict.
    """
    url = f"{FOOTBALL_DATA_BASE_URL}{endpoint}"
    headers = _build_headers()

    time.sleep(FOOTBALL_DATA_RATE_LIMIT)
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _read_cache_if_fresh() -> Optional[List[Dict[str, Any]]]:
    """Return cached match data if it exists and is less than 1 hour old.

    Returns:
        Cached list of match dicts, or ``None`` if cache is missing/stale.
    """
    if not CACHE_FILE.exists():
        return None
    age = time.monotonic() - os.path.getmtime(CACHE_FILE)
    if age > CACHE_TTL_SECONDS:
        logger.debug("Cache expired (%.0fs > %ds)", age, CACHE_TTL_SECONDS)
        return None
    logger.info("Loading match results from cache: %s", CACHE_FILE)
    return load_json(CACHE_FILE)


def _write_cache(data: List[Dict[str, Any]]) -> None:
    """Persist raw match data to the cache file."""
    save_json(data, CACHE_FILE)
    logger.debug("Cached %d matches to %s", len(data), CACHE_FILE)


def fetch_matches(
    competition_code: str = "WC",
    season_year: Optional[int] = None,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """Fetch matches for the World Cup 2026 competition.

    Args:
        competition_code: Football-data.org competition code (``"WC"``).
        season_year: Season year (defaults to 2026).
        use_cache: If ``True``, read from cached file on disk.

    Returns:
        List of match dicts.
    """
    if use_cache and FIXTURES_CACHE_FILE.exists():
        logger.info("Loading fixtures from cache: %s", FIXTURES_CACHE_FILE)
        return load_json(FIXTURES_CACHE_FILE)

    season = season_year or 2026
    try:
        data = _api_get(
            f"competitions/{competition_code}/matches",
            {
                "season": season,
            },
        )
        matches = data.get("matches", [])
        save_json(matches, FIXTURES_CACHE_FILE)
        logger.info("Fetched %d matches from API", len(matches))
        return matches
    except requests.HTTPError as exc:
        logger.error("API error fetching matches: %s", exc)
        return []


def matches_to_dataframe(matches: List[Dict]) -> pd.DataFrame:
    """Convert raw match API response to a clean DataFrame.

    Args:
        matches: List of match dicts from :func:`fetch_matches`.

    Returns:
        DataFrame with columns: ``match_id``, ``utc_date``, ``status``,
        ``home_team``, ``away_team``, ``home_score``, ``away_score``,
        ``winner``, ``stage``.
    """
    records = []
    for m in matches:
        records.append(
            {
                "match_id": m.get("id"),
                "utc_date": m.get("utcDate"),
                "status": m.get("status", "SCHEDULED"),
                "stage": m.get("stage", ""),
                "group": m.get("group", ""),
                "home_team": (m.get("homeTeam") or {}).get("name", ""),
                "away_team": (m.get("awayTeam") or {}).get("name", ""),
                "home_score": (m.get("score", {}).get("fullTime") or {}).get("home"),
                "away_score": (m.get("score", {}).get("fullTime") or {}).get("away"),
                "winner": m.get("score", {}).get("winner"),
            }
        )

    df = pd.DataFrame(records)
    df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True, errors="coerce")
    return df


# ── New convenience API ─────────────────────────────────────────────────────


def fetch_match_results(
    competition_code: str = "WC",
    season_year: Optional[int] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch match results (including final scores) for a competition.

    Wraps ``fetch_matches()`` + ``matches_to_dataframe()`` with a
    freshness check on the cache.

    Args:
        competition_code: Football-data.org competition code.
        season_year: Season year (defaults to 2026).
        use_cache: If ``True``, reuse cached data if less than 1 hour old.

    Returns:
        DataFrame with columns: ``match_id``, ``utc_date``, ``status``,
        ``home_team``, ``away_team``, ``home_score``, ``away_score``,
        ``winner``, ``stage``, ``group``.
    """
    if use_cache:
        cached = _read_cache_if_fresh()
        if cached is not None:
            return matches_to_dataframe(cached)

    matches = fetch_matches(competition_code, season_year, use_cache=False)
    if matches:
        _write_cache(matches)
    return matches_to_dataframe(matches)


def get_match_outcome(row: pd.Series, team: str) -> str:
    """Determine the match outcome from the perspective of *team*.

    Args:
        row: A row from the matches DataFrame (must have ``home_team``,
            ``away_team``, ``status``, ``home_score``, ``away_score``).
        team: Team name (matched case-insensitively).

    Returns:
        ``"WIN"``, ``"LOSS"``, ``"DRAW"``, or ``"NOT_PLAYED"``.
    """
    status = str(row.get("status", ""))
    if status != "FINISHED":
        return "NOT_PLAYED"

    home = str(row.get("home_team", ""))
    away = str(row.get("away_team", ""))
    home_score = row.get("home_score")
    away_score = row.get("away_score")

    if home_score is None or away_score is None:
        return "NOT_PLAYED"

    # Normalise team to home/away
    if team.lower() == home.lower():
        is_home = True
    elif team.lower() == away.lower():
        is_home = False
    else:
        # Fallback: use the existing fuzzy match from get_team_result
        result = get_team_result(row, team)
        return result.upper() if result != "unknown" else "NOT_PLAYED"

    if home_score == away_score:
        return "DRAW"
    if (is_home and home_score > away_score) or (
        not is_home and away_score > home_score
    ):
        return "WIN"
    return "LOSS"


def get_pre_post_windows(
    match_date: pd.Timestamp,
    window_hours: int = 24,
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Compute pre-match and post-match time windows.

    The pre-window spans ``[match_date - window_hours, match_date)`` and the
    post-window spans ``[match_date, match_date + window_hours]``.

    .. note::
        Comment volume tends to cluster heavily in the first few hours
        *after* a match.  The default 24-hour window is a reasonable starting
        point but should be tuned based on actual comment timestamp
        distributions — narrower windows (e.g. 6–12 h) may reduce noise for
        causal analysis.

    Args:
        match_date: The match kickoff timestamp (UTC).
        window_hours: Size of the pre/post window in hours (default 24).

    Returns:
        ``(pre_start, pre_end, post_start, post_end)`` — all as UTC
        timezone-aware timestamps.
    """
    delta = timedelta(hours=window_hours)
    pre_start = match_date - delta
    pre_end = match_date
    post_start = match_date
    post_end = match_date + delta
    return pre_start, pre_end, post_start, post_end


def build_match_results_summary(
    target_teams: Optional[List[str]] = None,
    competition_code: str = "WC",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Build a summary table of finished matches for the given teams.

    For each finished match involving a target team, adds the outcome
    from that team's perspective via ``get_match_outcome()``.

    Args:
        target_teams: List of team names (defaults to ``TARGET_TEAMS`` from
            config).
        competition_code: Football-data.org competition code.
        use_cache: Whether to use cached data.

    Returns:
        DataFrame with columns: ``team``, ``opponent``, ``match_date``,
        ``outcome``, ``score``, ``status``, ``stage``.
    """
    teams = target_teams or TARGET_TEAMS
    matches_df = fetch_match_results(competition_code, use_cache=use_cache)

    if matches_df.empty:
        return pd.DataFrame()

    records = []
    finished = matches_df[matches_df["status"] == "FINISHED"]

    for _, row in finished.iterrows():
        for team in teams:
            team_lower = team.lower()
            home_lower = str(row.get("home_team", "")).lower()
            away_lower = str(row.get("away_team", "")).lower()
            if team_lower not in (home_lower, away_lower):
                continue

            outcome = get_match_outcome(row, team)
            opponent = (
                row["away_team"] if home_lower == team_lower else row["home_team"]
            )
            score = (
                f"{int(row['home_score'])}-{int(row['away_score'])}"
                if pd.notna(row.get("home_score"))
                else "?"
            )

            records.append(
                {
                    "team": team,
                    "opponent": opponent,
                    "match_date": row["utc_date"],
                    "outcome": outcome,
                    "score": score,
                    "status": row["status"],
                    "stage": row.get("stage", ""),
                    "match_id": row.get("match_id"),
                }
            )

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("match_date").reset_index(drop=True)
    return result


# ── Match-sentiment integration ─────────────────────────────────────────────


def get_team_result(
    match_row: pd.Series,
    team: str,
) -> str:
    """Determine the result for a given team in a match.

    Args:
        match_row: A row from the matches DataFrame.
        team: Team name (must match API naming exactly or via fuzzy match).

    Returns:
        ``"win"``, ``"loss"``, ``"draw"``, or ``"unknown"``.
    """
    home = str(match_row.get("home_team", ""))
    away = str(match_row.get("away_team", ""))
    winner = match_row.get("winner")

    if team not in (home, away):
        # Try fuzzy matching
        for candidate in (home, away):
            if team.lower() in candidate.lower() or candidate.lower() in team.lower():
                team = candidate
                break
        else:
            return "unknown"

    if winner == "DRAW":
        return "draw"
    if winner == "HOME_TEAM" and team == home:
        return "win"
    if winner == "AWAY_TEAM" and team == away:
        return "win"
    if winner:
        return "loss"
    return "unknown"


def assign_matches_to_comments(
    comments_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    team_column: str = "teams",
    date_column: str = "created_utc",
) -> pd.DataFrame:
    """For each comment, find the nearest match within the time window.

    A comment is assigned to a match if:
    1. One of the comment's teams is participating in the match.
    2. The comment timestamp falls within ``[match_date - PRE, match_date + POST]``.

    Args:
        comments_df: DataFrame with at least ``teams`` and ``created_utc``.
        matches_df: DataFrame with ``utc_date``, ``home_team``, ``away_team``.
        team_column: Column in ``comments_df`` with comma-separated teams.
        date_column: Column in ``comments_df`` with ISO datetime strings.

    Returns:
        Comments DataFrame with added columns: ``match_id``, ``match_date``,
        ``team_result``, ``pre_post`` (``"pre"`` or ``"post"``).
    """
    if comments_df.empty or matches_df.empty:
        return comments_df

    comments_df = comments_df.copy()
    comments_df["_match_id"] = None
    comments_df["_match_date"] = None
    comments_df["_team_result"] = None
    comments_df["pre_post"] = None

    for _, match_row in matches_df.iterrows():
        match_date = match_row["utc_date"]
        if pd.isna(match_date):
            continue
        window_start = match_date - timedelta(hours=MATCH_PRE_WINDOW_HOURS)
        window_end = match_date + timedelta(hours=MATCH_POST_WINDOW_HOURS)

        for team in TARGET_TEAMS:
            result = get_team_result(match_row, team)
            if result == "unknown":
                continue

            # Find comments mentioning this team and within the window
            mask = (
                comments_df[team_column].str.contains(
                    team,
                    case=False,
                    na=False,
                )
                & (
                    pd.to_datetime(comments_df[date_column], utc=True, errors="coerce")
                    >= window_start
                )
                & (
                    pd.to_datetime(comments_df[date_column], utc=True, errors="coerce")
                    <= window_end
                )
                & comments_df["_match_id"].isna()
            )

            pre_post_val = (
                "pre"
                if pd.to_datetime(comments_df.loc[mask, date_column], utc=True)
                < match_date
                else "post"
            )

            comments_df.loc[mask, "_match_id"] = match_row["match_id"]
            comments_df.loc[mask, "_match_date"] = match_date
            comments_df.loc[mask, "_team_result"] = result
            comments_df.loc[mask, "pre_post"] = pre_post_val

    comments_df.rename(
        columns={
            "_match_id": "match_id",
            "_match_date": "match_date",
            "_team_result": "team_result",
        },
        inplace=True,
    )
    return comments_df


def compute_sentiment_shift(
    df: pd.DataFrame,
    result_column: str = "team_result",
    pre_post_column: str = "pre_post",
    sentiment_column: str = "sentiment_label",
) -> pd.DataFrame:
    """Compute average sentiment before vs. after matches by team and result.

    Performs a Mann-Whitney U test to compare pre/post sentiment distributions.

    Args:
        df: DataFrame with ``team_result``, ``pre_post``, ``sentiment_label``,
            and (optionally) numeric sentiment scores.
        result_column: Column with match result values.
        pre_post_column: Column with ``"pre"`` / ``"post"`` values.
        sentiment_column: Column with categorical sentiment label.

    Returns:
        DataFrame with one row per (team, result) combination and columns:
        ``n_pre``, ``n_post``, ``pre_pos_pct``, ``post_pos_pct``,
        ``pre_neg_pct``, ``post_neg_pct``, ``p_value``, ``significant``.
    """
    if df.empty:
        return pd.DataFrame()

    # Normalise POS/NEU/NEG → positive/neutral/negative if needed
    df = df.copy()
    if df[sentiment_column].isin(["POS", "NEU", "NEG"]).all():
        df[sentiment_column] = df[sentiment_column].map(
            {"POS": "positive", "NEU": "neutral", "NEG": "negative"}
        )

    # Map sentiment label to numeric score
    sentiment_map = {"positive": 2, "neutral": 1, "negative": 0}
    df["_sentiment_score"] = df[sentiment_column].map(sentiment_map).fillna(1)

    results_list = []

    for team in TARGET_TEAMS:
        team_mask = df["teams"].str.contains(team, case=False, na=False)
        team_df = df[team_mask & df["_match_id"].notna()].copy()

        if team_df.empty:
            continue

        for result in team_df[result_column].unique():
            result_df = team_df[team_df[result_column] == result]
            pre_df = result_df[result_df[pre_post_column] == "pre"]
            post_df = result_df[result_df[pre_post_column] == "post"]

            if len(pre_df) < 3 or len(post_df) < 3:
                continue

            pre_scores = pre_df["_sentiment_score"].values
            post_scores = post_df["_sentiment_score"].values

            # Mann-Whitney U test
            try:
                _, p_value = mannwhitneyu(
                    pre_scores,
                    post_scores,
                    alternative="two-sided",
                )
            except ValueError:
                p_value = 1.0

            pre_pos_pct = (pre_df[sentiment_column] == "positive").mean()
            post_pos_pct = (post_df[sentiment_column] == "positive").mean()
            pre_neg_pct = (pre_df[sentiment_column] == "negative").mean()
            post_neg_pct = (post_df[sentiment_column] == "negative").mean()

            results_list.append(
                {
                    "team": team,
                    "result": result,
                    "n_pre": len(pre_df),
                    "n_post": len(post_df),
                    "pre_pos_pct": round(pre_pos_pct, 3),
                    "post_pos_pct": round(post_pos_pct, 3),
                    "pre_neg_pct": round(pre_neg_pct, 3),
                    "post_neg_pct": round(post_neg_pct, 3),
                    "p_value": round(p_value, 4),
                    "significant": p_value < 0.05,
                }
            )

    return pd.DataFrame(results_list)


def save_and_return_results(
    shift_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    format: str = "parquet",
) -> pd.DataFrame:
    """Persist match-integrated results to disk.

    Args:
        shift_df: Result from :func:`compute_sentiment_shift`.
        comments_df: Augmented comments DataFrame.
        format: Output format.

    Returns:
        The shift DataFrame.
    """
    save_dataframe(shift_df, str(PROCESSED_DIR / "sentiment_shift"), format=format)
    save_dataframe(
        comments_df,
        str(PROCESSED_DIR / "comments_with_matches"),
        format=format,
    )
    logger.info("Match-integrated results saved.")
    return shift_df
