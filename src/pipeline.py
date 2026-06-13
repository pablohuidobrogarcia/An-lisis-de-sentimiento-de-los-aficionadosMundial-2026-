"""
Full end-to-end pipeline orchestrator.

Combines all ``src/`` modules into a single configurable pipeline:
1. Data collection from YouTube via Google API (or load cached).
2. Preprocessing (cleaning, language detection, dedup).
3. Sentiment analysis (primary + baseline).
4. Topic modeling (BERTopic) and NER (spaCy).
5. Match integration (football-data.org + sentiment shift stats).
6. Persist all results to ``data/processed/``.

Run with:
    ``python -m src.pipeline``           # full pipeline
    ``python -m src.pipeline --step collect``  # collection only (for cron)
"""

from typing import Optional

import pandas as pd

from src import config
from src.utils import load_dataframe, now_iso, save_dataframe, setup_logger

logger = setup_logger(__name__, log_file=str(config.PROCESSED_DIR / "pipeline.log"))


def _check_credentials() -> None:
    """Verify that required API credentials are available.

    Raises a clear ``ValueError`` before any API call is made if a required
    environment variable is missing or empty.
    """
    import os

    required = {
        "YOUTUBE_API_KEY": "YouTube Data API v3",
        "FOOTBALL_DATA_API_KEY": "football-data.org",
    }
    missing = []
    for var, service in required.items():
        val = os.environ.get(var, "") or getattr(config, var, "")
        if not val:
            missing.append(f"{var} (needed for {service})")

    if missing:
        raise ValueError(
            "Missing required API credentials. "
            "Set them in your .env file or as environment variables:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nSee .env.example for the required format."
        )


def run_collection_only(
    output_format: str = "parquet",
    limit_matches: Optional[int] = None,
    test_team: Optional[str] = None,
) -> pd.DataFrame:
    """Run only the data collection step and save results.

    Designed for the daily GitHub Actions cron job: fetches match fixtures,
    searches YouTube for new videos, and persists comments + checkpoint data.

    Args:
        output_format: ``"parquet"`` or ``"csv"``.
        limit_matches: If set, only process the first N matches.
        test_team: If set, temporarily add this team to the target list
            for a one-off test run (does not modify ``config.py``).

    Returns:
        DataFrame with newly collected comments (may be empty if quota exhausted).
    """
    _check_credentials()

    logger.info("=" * 60)
    logger.info("Collection-only run started at %s", now_iso())
    logger.info("=" * 60)

    from src.data_collection import incremental_update
    from src.results_api import fetch_matches, matches_to_dataframe

    raw_matches = fetch_matches(use_cache=True)
    matches_df = matches_to_dataframe(raw_matches)
    logger.info("Loaded %d matches from football-data.org", len(matches_df))

    if limit_matches:
        matches_df = matches_df.head(limit_matches)

    _restore_teams = False
    if test_team and test_team not in config.TARGET_TEAMS:
        config.TARGET_TEAMS.append(test_team)
        _restore_teams = True
        logger.info(
            "TEST MODE: temporarily including '%s' in target teams for this run only",
            test_team,
        )

    try:
        df = incremental_update(
            matches_df=matches_df,
            output_format=output_format,
        )
    finally:
        if _restore_teams:
            config.TARGET_TEAMS.pop()

    if df.empty:
        logger.warning("No comments collected (quota may be exhausted).")
    else:
        logger.info("Collection complete: %d total comments", len(df))

    return df


def run_pipeline(
    collect: bool = True,
    preprocess: bool = True,
    sentiment: bool = True,
    topic_ner: bool = True,
    match_integration: bool = True,
    output_format: str = "parquet",
    limit_matches: Optional[int] = None,
    test_team: Optional[str] = None,
) -> pd.DataFrame:
    """Execute the full sentiment analysis pipeline.

    Each step can be toggled on/off. When a step is skipped, the pipeline
    attempts to load cached data from the previous run.

    Args:
        collect: Run YouTube data collection.
        preprocess: Run text preprocessing.
        sentiment: Run sentiment analysis.
        topic_ner: Run topic modeling and NER.
        match_integration: Fetch match results and compute sentiment shift.
        output_format: ``"parquet"`` or ``"csv"``.
        limit_matches: If set, only process the first N matches (for testing).

    Returns:
        Final DataFrame with all enrichment columns.
    """
    if collect or match_integration:
        _check_credentials()

    start_time = now_iso()
    logger.info("=" * 60)
    logger.info("Pipeline started at %s", start_time)
    logger.info("=" * 60)

    # ── Step 0: Fetch match data (needed for YouTube search) ─────────────
    matches_df: pd.DataFrame = pd.DataFrame()
    if match_integration or collect:
        from src.results_api import fetch_matches, matches_to_dataframe

        raw_matches = fetch_matches(use_cache=True)
        matches_df = matches_to_dataframe(raw_matches)
        logger.info("Loaded %d matches from football-data.org", len(matches_df))
        if limit_matches:
            matches_df = matches_df.head(limit_matches)

    # ── Step 1: Collection ───────────────────────────────────────────────
    if collect:
        from src.data_collection import incremental_update

        logger.info(
            "Step 1/5: YouTube comment collection for %d matches", len(matches_df)
        )

        _restore_teams = False
        if test_team and test_team not in config.TARGET_TEAMS:
            config.TARGET_TEAMS.append(test_team)
            _restore_teams = True
            logger.info(
                "TEST MODE: temporarily including '%s' in target teams for this run only",
                test_team,
            )

        try:
            df = incremental_update(matches_df=matches_df, output_format=output_format)
        finally:
            if _restore_teams:
                config.TARGET_TEAMS.pop()
    else:
        from src.data_collection import load_collected

        logger.info("Step 1/5: Loading cached YouTube data …")
        df = load_collected(format=output_format)

    if df.empty:
        logger.warning("No data available. Exiting.")
        return df

    logger.info("Data shape: %d rows, %d cols", *df.shape)

    # ── Step 2: Preprocessing ────────────────────────────────────────────
    if preprocess:
        from src.preprocessing import preprocess_dataframe

        logger.info("Step 2/5: Preprocessing …")
        df = preprocess_dataframe(df)
        save_dataframe(
            df, str(config.PROCESSED_DIR / "preprocessed"), format=output_format
        )
        logger.info("After preprocessing: %d rows", len(df))
    else:
        df = _load_cached("preprocessed", df, output_format)

    # ── Step 3: Sentiment Analysis ───────────────────────────────────────
    if sentiment and not df.empty:
        from src.sentiment import add_sentiment_to_dataframe

        logger.info("Step 3/5: Sentiment analysis …")
        df = add_sentiment_to_dataframe(df, model="transformer")
        save_dataframe(
            df, str(config.PROCESSED_DIR / "sentiment"), format=output_format
        )
        logger.info("Sentiment added. Columns: %s", list(df.columns))
    else:
        df = _load_cached("sentiment", df, output_format)

    # ── Step 4: Topic Modeling + NER ─────────────────────────────────────
    if topic_ner and not df.empty and "text_clean" in df.columns:
        from src.topic_modeling import (
            add_entities_to_dataframe,
            add_topics_to_dataframe,
        )

        logger.info("Step 4/5: Topic modeling …")
        df, _ = add_topics_to_dataframe(
            df, model_save_path=config.PROCESSED_DIR / "bertopic_model"
        )
        logger.info("Step 4/5: NER …")
        df = add_entities_to_dataframe(df)
        save_dataframe(
            df, str(config.PROCESSED_DIR / "topic_ner"), format=output_format
        )
    else:
        df = _load_cached("topic_ner", df, output_format)

    # ── Step 5: Match Integration ────────────────────────────────────────
    if match_integration and not df.empty:
        from src.results_api import (
            assign_matches_to_comments,
            compute_sentiment_shift,
            save_and_return_results,
        )

        logger.info("Step 5/5: Match integration …")
        df = assign_matches_to_comments(df, matches_df)
        shift_df = compute_sentiment_shift(df)
        if not shift_df.empty:
            save_and_return_results(shift_df, df, format=output_format)
            logger.info("Sentiment shift results:\n%s", shift_df.to_string())
        else:
            save_dataframe(
                df, str(config.PROCESSED_DIR / "final"), format=output_format
            )
            logger.warning("No match results integrated yet.")
    else:
        df = _load_cached("final", df, output_format)

    save_dataframe(df, str(config.PROCESSED_DIR / "final"), format=output_format)
    logger.info("Pipeline finished. Final shape: %s", df.shape)
    return df


def _load_cached(
    step_name: str, fallback_df: pd.DataFrame, output_format: str
) -> pd.DataFrame:
    path = config.PROCESSED_DIR / step_name
    try:
        cached = load_dataframe(str(path.with_suffix(f".{output_format}")))
        if not cached.empty:
            logger.info("Loaded cached '%s' (%d rows)", step_name, len(cached))
            return cached
    except (FileNotFoundError, ValueError):
        pass
    logger.info("No cache for '%s', using previous step data.", step_name)
    return fallback_df


def main() -> None:
    """CLI entry point.

    Supports two modes:

    - **Single-step mode**: ``--step collect`` runs only the named step.
    - **Toggle mode** (default): ``--skip-*`` flags disable specific steps.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="World Cup 2026 Sentiment Analysis Pipeline"
    )

    # Mutually exclusive: --step vs --skip-* flags
    parser.add_argument(
        "--step",
        type=str,
        choices=["collect", "preprocess", "sentiment", "topic_ner", "match", "all"],
        default=None,
        help="Run a single pipeline step and exit (overrides --skip-* flags).",
    )

    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument("--skip-topic-ner", action="store_true")
    parser.add_argument("--skip-match", action="store_true")
    parser.add_argument("--limit-matches", type=int, default=None)
    parser.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    parser.add_argument(
        "--test-team",
        type=str,
        default=None,
        help="Temporarily add a team for one-off end-to-end testing.",
    )

    args = parser.parse_args()

    # Handle --step mode
    if args.step:
        if args.step == "all":
            run_pipeline(
                output_format=args.format,
                limit_matches=args.limit_matches,
                test_team=args.test_team,
            )
        elif args.step == "collect":
            run_collection_only(
                output_format=args.format,
                limit_matches=args.limit_matches,
                test_team=args.test_team,
            )
        else:
            step_map = {
                "preprocess": "preprocess",
                "sentiment": "sentiment",
                "topic_ner": "topic_ner",
                "match": "match_integration",
            }
            kwargs = {
                "collect": False,
                "preprocess": False,
                "sentiment": False,
                "topic_ner": False,
                "match_integration": False,
                "output_format": args.format,
                "limit_matches": args.limit_matches,
            }
            kwargs[step_map[args.step]] = True
            run_pipeline(**kwargs)
    else:
        run_pipeline(
            collect=not args.skip_collect,
            preprocess=not args.skip_preprocess,
            sentiment=not args.skip_sentiment,
            topic_ner=not args.skip_topic_ner,
            match_integration=not args.skip_match,
            output_format=args.format,
            limit_matches=args.limit_matches,
        )


if __name__ == "__main__":
    main()
