"""Tests for the match results API module."""

from datetime import timedelta

import pandas as pd

from src.results_api import get_match_outcome, get_pre_post_windows


def _make_match_row(
    home_team: str = "Brazil",
    away_team: str = "Morocco",
    status: str = "FINISHED",
    home_score: int = 2,
    away_score: int = 0,
    winner: str = "HOME_TEAM",
) -> pd.Series:
    """Helper: create a synthetic match row for testing."""
    return pd.Series(
        {
            "home_team": home_team,
            "away_team": away_team,
            "status": status,
            "home_score": home_score,
            "away_score": away_score,
            "winner": winner,
            "match_id": 12345,
            "stage": "GROUP_STAGE",
        }
    )


class TestGetMatchOutcome:
    def test_home_win(self) -> None:
        row = _make_match_row(
            home_team="Brazil", away_team="Morocco", home_score=2, away_score=0
        )
        assert get_match_outcome(row, "Brazil") == "WIN"
        assert get_match_outcome(row, "Morocco") == "LOSS"

    def test_away_win(self) -> None:
        row = _make_match_row(
            home_team="Brazil",
            away_team="Morocco",
            home_score=1,
            away_score=3,
            winner="AWAY_TEAM",
        )
        assert get_match_outcome(row, "Morocco") == "WIN"
        assert get_match_outcome(row, "Brazil") == "LOSS"

    def test_draw(self) -> None:
        row = _make_match_row(
            home_team="Brazil",
            away_team="Morocco",
            home_score=1,
            away_score=1,
            winner="DRAW",
        )
        assert get_match_outcome(row, "Brazil") == "DRAW"
        assert get_match_outcome(row, "Morocco") == "DRAW"

    def test_not_played(self) -> None:
        row = _make_match_row(
            status="SCHEDULED", home_score=None, away_score=None, winner=None
        )
        assert get_match_outcome(row, "Brazil") == "NOT_PLAYED"

    def test_in_play(self) -> None:
        row = _make_match_row(status="IN_PLAY", home_score=1, away_score=0, winner=None)
        assert get_match_outcome(row, "Brazil") == "NOT_PLAYED"

    def test_missing_scores(self) -> None:
        row = _make_match_row(
            status="FINISHED", home_score=None, away_score=None, winner=None
        )
        assert get_match_outcome(row, "Brazil") == "NOT_PLAYED"

    def test_team_not_in_match(self) -> None:
        row = _make_match_row(home_team="Spain", away_team="France")
        assert get_match_outcome(row, "Brazil") == "NOT_PLAYED"

    def test_case_insensitivity(self) -> None:
        row = _make_match_row(
            home_team="Brazil", away_team="Morocco", home_score=3, away_score=1
        )
        assert get_match_outcome(row, "brazil") == "WIN"
        assert get_match_outcome(row, "BRAZIL") == "WIN"
        assert get_match_outcome(row, "morocco") == "LOSS"


class TestGetPrePostWindows:
    def test_returns_four_timestamps(self) -> None:
        match_date = pd.Timestamp("2026-06-13 22:00:00+00:00")
        pre_start, pre_end, post_start, post_end = get_pre_post_windows(match_date)
        assert pre_start < pre_end
        assert pre_end == post_start
        assert post_start < post_end

    def test_default_24h_window(self) -> None:
        match_date = pd.Timestamp("2026-06-13 22:00:00+00:00")
        pre_start, pre_end, post_start, post_end = get_pre_post_windows(match_date)
        assert pre_start == match_date - timedelta(hours=24)
        assert pre_end == match_date
        assert post_start == match_date
        assert post_end == match_date + timedelta(hours=24)

    def test_custom_window(self) -> None:
        match_date = pd.Timestamp("2026-06-13 22:00:00+00:00")
        pre_start, pre_end, post_start, post_end = get_pre_post_windows(
            match_date, window_hours=6
        )
        assert pre_start == match_date - timedelta(hours=6)
        assert post_end == match_date + timedelta(hours=6)

    def test_non_zero_window(self) -> None:
        match_date = pd.Timestamp("2026-06-13 22:00:00+00:00")
        pre_start, pre_end, _, _ = get_pre_post_windows(match_date)
        assert (pre_end - pre_start).total_seconds() == 24 * 3600

    def test_all_return_utc_aware(self) -> None:
        match_date = pd.Timestamp("2026-06-13 22:00:00+00:00")
        for ts in get_pre_post_windows(match_date):
            assert ts.tz is not None
