"""
YouTube Data API v3 comment collection with quota tracking, retry logic,
and a checkpoint/resume system to survive interruptions without duplication.

Strategy
--------
1. For each (team, opponent, match_date) triple from the football-data.org
   fixture list, search YouTube for highlight/recap videos using configured
   search templates (e.g. ``"Spain vs Germany world cup 2026 highlights"``).
2. For each new (unprocessed) video found, fetch top-level comments via
   ``commentThreads().list()`` and reply threads via ``comments().list()``.
3. Track processed video IDs in a checkpoint file to avoid re-scraping.
4. Monitor YouTube API quota (default 10 000 units/day) and stop with a
   warning when the safety margin is reached.
"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.config import (
    CHECKPOINT_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    TARGET_TEAMS,
    TEAM_KEYWORDS,
    YOUTUBE_API_KEY,
    YOUTUBE_API_SERVICE_NAME,
    YOUTUBE_API_VERSION,
    YOUTUBE_CHANNELS,
    YOUTUBE_DAILY_QUOTA,
    YOUTUBE_MAX_COMMENTS_PER_VIDEO,
    YOUTUBE_MAX_RESULTS_PER_SEARCH,
    YOUTUBE_QUOTA_SAFETY_MARGIN,
    YOUTUBE_SEARCH_TEMPLATES,
    YOUTUBE_SLEEP_BETWEEN_CALLS,
)
from src.utils import setup_logger, text_hash, save_dataframe, load_dataframe

logger = setup_logger(__name__)

# Checkpoint files live under ``data/raw/.checkpoints/`` so they are
# committed to the repository and survive across GitHub Actions runs.
CHECKPOINT_FILE: Path = CHECKPOINT_DIR / "_checkpoint_youtube.json"
VIDEOS_FILE: Path = RAW_DIR / "youtube_videos"
COMMENTS_FILE: Path = RAW_DIR / "youtube_comments"
QUOTA_LOG_FILE: Path = CHECKPOINT_DIR / "_quota_usage.json"


# ── Quota tracking ─────────────────────────────────────────────────────────


def _load_quota_usage() -> Dict[str, Any]:
    """Load today's quota usage from disk."""
    if QUOTA_LOG_FILE.exists():
        import json
        with open(QUOTA_LOG_FILE, "r") as f:
            data = json.load(f)
        # Reset if it's a new day
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if data.get("date") != today:
            data = {"date": today, "quota_used": 0}
        return data
    return {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "quota_used": 0}


def _save_quota_usage(quota_used: int) -> None:
    """Persist quota usage to disk."""
    import json
    data = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "quota_used": quota_used,
    }
    with open(QUOTA_LOG_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def _check_quota(quota_cost: int) -> bool:
    """Return ``True`` if we have enough quota remaining for *quota_cost*.

    Logs a warning and returns ``False`` when the safety margin is breached.
    """
    usage = _load_quota_usage()
    projected = usage["quota_used"] + quota_cost
    remaining = YOUTUBE_DAILY_QUOTA - projected
    if remaining <= YOUTUBE_QUOTA_SAFETY_MARGIN:
        logger.warning(
            "YouTube quota would be exceeded: %d used + %d cost = %d "
            "(safety margin %d). Stopping.",
            usage["quota_used"], quota_cost, projected, YOUTUBE_QUOTA_SAFETY_MARGIN,
        )
        return False
    return True


def _consume_quota(quota_cost: int) -> None:
    """Record *quota_cost* units as consumed."""
    usage = _load_quota_usage()
    usage["quota_used"] += quota_cost
    _save_quota_usage(usage["quota_used"])
    logger.debug("Quota consumed: +%d (total: %d)", quota_cost, usage["quota_used"])


# ── YouTube client ─────────────────────────────────────────────────────────


def _build_youtube_client():
    """Build and return an authenticated YouTube API service object."""
    if not YOUTUBE_API_KEY:
        raise ValueError(
            "YouTube API key missing. "
            "Set YOUTUBE_API_KEY in your .env file."
        )
    return build(
        YOUTUBE_API_SERVICE_NAME,
        YOUTUBE_API_VERSION,
        developerKey=YOUTUBE_API_KEY,
    )


def _api_call(
    client,
    quota_cost: int,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Make a YouTube API call with quota checking, retry, and backoff.

    Args:
        client: YouTube API service object.
        quota_cost: Units this call consumes (search=100, threads=1, etc.).
        **kwargs: Arguments for the API method (e.g. ``list()`` params).

    Returns:
        API response dict, or ``None`` if quota would be exceeded or call fails.
    """
    if not _check_quota(quota_cost):
        return None

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = kwargs.pop("execute")() if "execute" in kwargs else None
            if response is not None:
                _consume_quota(quota_cost)
                time.sleep(YOUTUBE_SLEEP_BETWEEN_CALLS)
                return response

            # Standard call pattern
            method = kwargs.pop("method")
            request = method(**kwargs)
            response = request.execute()
            _consume_quota(quota_cost)
            time.sleep(YOUTUBE_SLEEP_BETWEEN_CALLS)
            return response

        except HttpError as exc:
            if exc.resp.status == 403:
                reason = str(exc)
                if "quotaExceeded" in reason:
                    logger.error("YouTube quota exceeded. Stop collection.")
                    return None
                logger.warning("Forbidden (403): %s", exc)
                return None
            elif exc.resp.status == 429:
                wait = 2 ** attempt
                logger.warning("Rate limited (429). Waiting %ds …", wait)
                time.sleep(wait)
            else:
                logger.error("YouTube API error: %s", exc)
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)

    return None


# ── Checkpoint (processed video IDs) ──────────────────────────────────────


def _load_processed_videos() -> Set[str]:
    """Load set of already-processed video IDs."""
    if CHECKPOINT_FILE.exists():
        import json
        with open(CHECKPOINT_FILE, "r") as f:
            return set(json.load(f).get("processed_video_ids", []))
    return set()


def _save_processed_video(video_id: str) -> None:
    """Add a video ID to the checkpoint."""
    import json
    processed = _load_processed_videos()
    processed.add(video_id)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"processed_video_ids": list(processed)}, f, ensure_ascii=False)


# ── Team detection ─────────────────────────────────────────────────────────


def _resolve_teams_from_video(
    video_title: str,
    video_description: str,
    team: str,
) -> List[str]:
    """Determine which target teams a video's comments likely discuss.

    Uses the match team (from search context) plus keyword detection in
    the video title and description for cross-team mentions.

    Args:
        video_title: Title of the YouTube video.
        video_description: Description text.
        team: Primary team used in the search query.

    Returns:
        List of team names (at minimum includes *team*).
    """
    teams: List[str] = [team]
    combined = (video_title + " " + video_description).lower()
    for t in TARGET_TEAMS:
        if t == team:
            continue
        if any(kw in combined for kw in TEAM_KEYWORDS.get(t, [])):
            if t not in teams:
                teams.append(t)
    return teams


# ── Video search ───────────────────────────────────────────────────────────


def _search_videos_for_match(
    client,
    team: str,
    opponent: str,
    match_date: str,
) -> List[Dict[str, Any]]:
    """Search YouTube for match-related videos.

    Args:
        client: YouTube API service.
        team: Primary team name.
        opponent: Opponent team name.
        match_date: ISO date string of the match (used to filter recency).

    Returns:
        List of video info dicts with keys ``video_id``, ``title``,
        ``description``, ``published_at``, ``channel_id``.
    """
    videos: List[Dict[str, Any]] = []
    processed_ids = _load_processed_videos()

    for template in YOUTUBE_SEARCH_TEMPLATES:
        query = template.format(team=team, opponent=opponent, date=match_date)
        logger.debug("Searching YouTube for: %s", query)

        search_result = _api_call(
            client,
            quota_cost=100,
            method=client.search().list,
            part="snippet",
            q=query,
            type="video",
            maxResults=YOUTUBE_MAX_RESULTS_PER_SEARCH,
            order="relevance",
            publishedAfter=_date_to_rfc3339(match_date, days_before=3),
            publishedBefore=_date_to_rfc3339(match_date, days_after=3),
        )

        if search_result is None:
            continue

        items = search_result.get("items", [])
        for item in items:
            vid = item["id"]["videoId"]
            if vid in processed_ids:
                continue
            snippet = item["snippet"]
            videos.append({
                "video_id": vid,
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "published_at": snippet.get("publishedAt", ""),
                "channel_id": snippet.get("channelId", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "search_query": query,
                "team": team,
                "opponent": opponent,
                "match_date": match_date,
            })

    return videos


def _date_to_rfc3339(date_str: str, days_before: int = 0, days_after: int = 0) -> str:
    """Convert an ISO date to RFC 3339 with an offset.

    Args:
        date_str: ISO date string (e.g. ``"2026-06-15"``).
        days_before: Subtract this many days.
        days_after: Add this many days.

    Returns:
        RFC 3339 datetime string.
    """
    try:
        dt = datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        dt = datetime.now(timezone.utc)
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=timezone.utc)
    dt -= timedelta(days=days_before)
    dt += timedelta(days=days_after)
    return dt.isoformat()


# ── Comment fetching ──────────────────────────────────────────────────────


def _fetch_comments_for_video(
    client,
    video_id: str,
) -> List[Dict[str, Any]]:
    """Fetch top-level comments + replies for a single video.

    Args:
        client: YouTube API service.
        video_id: YouTube video ID.

    Returns:
        List of comment record dicts.
    """
    comments: List[Dict[str, Any]] = []
    next_page_token: Optional[str] = None

    while len(comments) < YOUTUBE_MAX_COMMENTS_PER_VIDEO:
        params: Dict[str, Any] = {
            "method": client.commentThreads().list,
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 100,
            "order": "relevance",
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        response = _api_call(client, quota_cost=1, **params)
        if response is None:
            break

        for item in response.get("items", []):
            snippet = item["snippet"]
            top_comment = snippet["topLevelComment"]["snippet"]

            record = {
                "comment_id": item["id"],
                "video_id": video_id,
                "parent_id": "",  # top-level
                "text": top_comment.get("textDisplay", ""),
                "author": top_comment.get("authorDisplayName", "[unknown]"),
                "published_at": top_comment.get("publishedAt", ""),
                "like_count": top_comment.get("likeCount", 0),
                "total_reply_count": snippet.get("totalReplyCount", 0),
                "source": "youtube",
                "collected_at": datetime.now(timezone.utc).isoformat(),
            }
            comments.append(record)

            # Fetch replies
            reply_count = snippet.get("totalReplyCount", 0)
            if reply_count > 0:
                replies = _fetch_replies(client, video_id, item["id"])
                comments.extend(replies)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments


def _fetch_replies(
    client,
    video_id: str,
    parent_id: str,
) -> List[Dict[str, Any]]:
    """Fetch replies to a specific comment thread."""
    replies: List[Dict[str, Any]] = []
    next_page_token: Optional[str] = None

    while True:
        params: Dict[str, Any] = {
            "method": client.comments().list,
            "part": "snippet",
            "parentId": parent_id,
            "maxResults": 100,
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        response = _api_call(client, quota_cost=1, **params)
        if response is None:
            break

        for item in response.get("items", []):
            snippet = item["snippet"]
            replies.append({
                "comment_id": item["id"],
                "video_id": video_id,
                "parent_id": parent_id,
                "text": snippet.get("textDisplay", ""),
                "author": snippet.get("authorDisplayName", "[unknown]"),
                "published_at": snippet.get("publishedAt", ""),
                "like_count": snippet.get("likeCount", 0),
                "total_reply_count": 0,
                "source": "youtube",
                "collected_at": datetime.now(timezone.utc).isoformat(),
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return replies


# ── Orchestration ─────────────────────────────────────────────────────────


def collect_for_matches(
    matches_df: pd.DataFrame,
    output_format: str = "parquet",
) -> pd.DataFrame:
    """Collect YouTube comments for all matches in the fixture list.

    For each match (row in *matches_df*), searches YouTube for related
    videos and fetches all comments. Skips videos whose IDs are already
    in the checkpoint.

    Args:
        matches_df: DataFrame with columns ``utc_date``, ``home_team``,
            ``away_team``.
        output_format: ``"parquet"`` (default) or ``"csv"``.

    Returns:
        DataFrame with all collected comments.
    """
    client = _build_youtube_client()
    all_videos: List[Dict[str, Any]] = []
    all_comments: List[Dict[str, Any]] = []
    processed_ids = _load_processed_videos()

    for _, match_row in matches_df.iterrows():
        home = str(match_row.get("home_team", ""))
        away = str(match_row.get("away_team", ""))
        date_str = str(match_row.get("utc_date", ""))

        # Collect for both teams (home and away)
        teams_to_search = [t for t in TARGET_TEAMS if t.lower() in home.lower() or home.lower() in t.lower()]
        teams_to_search += [t for t in TARGET_TEAMS if t not in teams_to_search and (t.lower() in away.lower() or away.lower() in t.lower())]

        for team in teams_to_search:
            opponent = away if team.lower() in home.lower() else home
            logger.info(
                "Searching videos for %s vs %s (%s)", team, opponent, date_str,
            )

            videos = _search_videos_for_match(client, team, opponent, date_str)
            new_videos = [v for v in videos if v["video_id"] not in processed_ids]

            if not new_videos:
                logger.info("  No new videos found for %s vs %s", team, opponent)
                continue

            logger.info("  Found %d new videos", len(new_videos))
            all_videos.extend(new_videos)

            for vid_info in new_videos:
                vid = vid_info["video_id"]
                logger.info("  Fetching comments for video %s …", vid)
                comment_records = _fetch_comments_for_video(client, vid)

                # Tag each comment with team info from video context
                teams_in_vid = _resolve_teams_from_video(
                    vid_info["title"],
                    vid_info["description"],
                    team,
                )

                for cr in comment_records:
                    cr["text_hash"] = text_hash(cr["text"])
                    cr["teams"] = ",".join(teams_in_vid)
                    cr["video_title"] = vid_info["title"]
                    cr["video_published_at"] = vid_info["published_at"]
                    cr["search_team"] = team
                    cr["search_opponent"] = opponent
                    cr["match_date"] = date_str

                all_comments.extend(comment_records)

                # Mark video as processed
                _save_processed_video(vid)
                logger.info(
                    "    → %d comments from video %s", len(comment_records), vid,
                )

    # Persist videos
    if all_videos:
        videos_df = pd.DataFrame(all_videos)
        save_dataframe(videos_df, str(VIDEOS_FILE), format=output_format)
        logger.info("Saved %d video records", len(videos_df))

    # Persist comments
    if all_comments:
        comments_df = pd.DataFrame(all_comments)
        save_dataframe(comments_df, str(COMMENTS_FILE), format=output_format)
        logger.info("Saved %d comments to %s", len(comments_df), COMMENTS_FILE)
    else:
        logger.warning("No comments collected.")
        comments_df = pd.DataFrame()

    # Log quota summary
    usage = _load_quota_usage()
    logger.info(
        "Quota used today: %d / %d units", usage["quota_used"], YOUTUBE_DAILY_QUOTA,
    )

    return comments_df


def load_collected(format: str = "parquet") -> pd.DataFrame:
    """Load previously collected YouTube comments from disk.

    Args:
        format: ``"parquet"`` or ``"csv"``.

    Returns:
        DataFrame with cached comments, or empty DataFrame if none exist.
    """
    path = COMMENTS_FILE.with_suffix(f".{format}")
    if not path.exists():
        logger.warning("No cached YouTube data found at %s", path)
        return pd.DataFrame()
    return load_dataframe(str(path), format=format)


def incremental_update(
    matches_df: pd.DataFrame,
    output_format: str = "parquet",
) -> pd.DataFrame:
    """Run an incremental collection that deduplicates against existing data.

    Loads existing comments (if any), collects new ones from matches that
    have not been processed yet, merges without duplicates, and saves.

    Args:
        matches_df: DataFrame with match fixtures (``home_team``, ``away_team``,
            ``utc_date``).
        output_format: ``"parquet"`` or ``"csv"``.

    Returns:
        Updated DataFrame of all comments.
    """
    existing = load_collected(format=output_format)
    existing_hashes: Set[str] = (
        set(existing["text_hash"].tolist()) if not existing.empty else set()
    )

    new_df = collect_for_matches(matches_df, output_format=output_format)

    if new_df.empty:
        logger.info("No new comments found.")
        return existing

    new_df = new_df[~new_df["text_hash"].isin(existing_hashes)]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.drop_duplicates(subset=["text_hash"], keep="last", inplace=True)

    save_dataframe(combined, str(COMMENTS_FILE), format=output_format)
    logger.info(
        "Incremental update: %d existing + %d new = %d total",
        len(existing), len(new_df), len(combined),
    )
    return combined
```

