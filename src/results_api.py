"""
football-data.org API client for World Cup 2026 fixtures and results.

Provides functions to fetch match data, map comments to the nearest match
time window, and compute pre/post sentiment comparisons with statistical
testing.
"""

import time
from datetime import timedelta
from typing import Any, Dict, List, Optional

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

FIXTURES_CACHE_FILE = PROCESSED_DIR / "fixtures_cache.json"
RESULTS_FILE = PROCESSED_DIR / "match_results"


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

    # Map sentiment label to numeric score
    sentiment_map = {"positive": 2, "neutral": 1, "negative": 0}
    df = df.copy()
    df["_sentiment_score"] = (
        df[sentiment_column]
        .map(
            sentiment_map,
        )
        .fillna(1)
    )

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
