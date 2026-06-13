"""Tests for the preprocessing module."""

import pandas as pd
import pytest

from src.preprocessing import (
    clean_text,
    detect_language,
    extract_emojis,
    preprocess_comment,
    preprocess_dataframe,
)


class TestCleanText:
    def test_remove_urls(self) -> None:
        text = "Check this https://example.com/foo and http://bit.ly/abc"
        result = clean_text(text)
        assert "https://" not in result
        assert "http://" not in result

    def test_remove_mentions(self) -> None:
        text = "Great goal by @messi and @neymar!"
        result = clean_text(text)
        assert "@" not in result

    def test_remove_emojis(self) -> None:
        text = "I love this team! 😍🔥"
        result = clean_text(text)
        assert "😍" not in result
        assert "🔥" not in result

    def test_collapse_repeating_chars(self) -> None:
        text = "nooooooo wayyyyy"
        result = clean_text(text)
        assert "nooo" in result
        assert "nooooooo" not in result

    def test_strip_whitespace(self) -> None:
        text = "   hello   world   "
        result = clean_text(text)
        assert result == "hello world"

    def test_empty_text(self) -> None:
        assert clean_text("") == ""
        assert clean_text("   ") == ""

    def test_decode_html_entities(self) -> None:
        text = "Spain &amp; Argentina are &lt;great&gt;!"
        result = clean_text(text)
        assert "&amp;" not in result
        assert "&lt;" not in result
        assert "&" in result
        assert "<" in result


class TestDetectLanguage:
    def test_detect_spanish(self) -> None:
        lang, conf = detect_language("Me encanta el fútbol, es el mejor deporte del mundo")
        assert lang == "es"
        assert conf >= 0.7

    def test_detect_english(self) -> None:
        lang, conf = detect_language("This is an amazing goal by the best team in the world")
        assert lang == "en"
        assert conf >= 0.7

    def test_short_text_returns_unknown(self) -> None:
        lang, conf = detect_language("Hola")
        assert lang == "unknown"
        assert conf == 0.0


class TestExtractEmojis:
    def test_finds_emojis(self) -> None:
        emojis = extract_emojis("Hello 😊 world 🔥")
        assert len(emojis) == 2

    def test_no_emojis(self) -> None:
        emojis = extract_emojis("Hello world")
        assert len(emojis) == 0


class TestPreprocessComment:
    def test_returns_dict_for_valid_text(self) -> None:
        result = preprocess_comment("I love watching football matches! 😊")
        assert result is not None
        assert "text_clean" in result
        assert "language" in result
        assert "n_emojis" in result

    def test_returns_none_for_short_text(self) -> None:
        result = preprocess_comment("Hi")
        assert result is None

    def test_language_filtering(self) -> None:
        result = preprocess_comment("Detta är svenska som inte stöds")
        # Should be filtered out if not ES/EN
        if result is not None:
            assert result["language"] in ("es", "en")


class TestPreprocessDataFrame:
    def test_returns_same_columns_plus_new(self) -> None:
        df = pd.DataFrame({
            "text": ["Great game by Spain!", "Mal partido de Argentina"],
            "video_title": ["Highlights Spain vs X", "Resumen Argentina vs Y"],
            "teams": ["Spain", "Argentina"],
            "published_at": ["2026-06-15T12:00:00+00:00"] * 2,
        })
        result = preprocess_dataframe(df)
        assert "text_clean" in result.columns
        assert "language" in result.columns
        assert len(result) <= len(df)

    def test_deduplicates_by_hash(self) -> None:
        df = pd.DataFrame({
            "text": ["Same text here"] * 5,
            "video_title": ["Some video"] * 5,
            "teams": ["Spain"] * 5,
            "published_at": ["2026-06-15T12:00:00+00:00"] * 5,
        })
        result = preprocess_dataframe(df)
        assert len(result) <= 3  # some may get filtered by language
```

