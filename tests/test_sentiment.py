"""Tests for the sentiment analysis module."""

import pandas as pd

from src.sentiment import (
    _baseline_es,
    add_sentiment_to_dataframe,
    predict_batch,
    predict_sentiment,
)


class TestBaselineEs:
    def test_positive_text(self) -> None:
        scores = _baseline_es("Qué golazo increíble, vamos España!")
        assert scores["positive"] > scores["negative"]
        assert scores["positive"] > 0

    def test_negative_text(self) -> None:
        scores = _baseline_es("Qué vergüenza de arbitraje, robo total")
        assert scores["negative"] > scores["positive"]
        assert scores["negative"] > 0

    def test_neutral_text(self) -> None:
        scores = _baseline_es("El partido empieza a las 9")
        assert scores["neutral"] >= scores["positive"]
        assert scores["neutral"] >= scores["negative"]

    def test_empty_text(self) -> None:
        scores = _baseline_es("")
        assert scores["neutral"] == 1.0


class TestPredictSentiment:
    def test_returns_dict_with_two_models(self) -> None:
        result = predict_sentiment(
            "Great match today!",
            language="en",
            model="transformer",
        )
        assert "transformer" in result
        assert "baseline" in result
        for model_key in ("transformer", "baseline"):
            for dim in ("positive", "negative", "neutral"):
                assert dim in result[model_key]


class TestPredictBatch:
    def test_returns_dataframe(self) -> None:
        texts = ["Love this!", "Terrible game", "It's okay"]
        languages = ["en", "en", "en"]
        df = predict_batch(texts, languages)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "sentiment_label" in df.columns
        assert "sentiment_positive" in df.columns
        assert "sentiment_baseline_label" in df.columns


class TestAddSentimentToDataFrame:
    def test_adds_columns(self) -> None:
        df = pd.DataFrame(
            {
                "text_clean": ["Great match!", "Mal partido"],
                "language": ["en", "es"],
            }
        )
        result = add_sentiment_to_dataframe(df)
        assert "sentiment_label" in result.columns
        assert "sentiment_positive" in result.columns
        assert "sentiment_baseline_label" in result.columns
        assert len(result) == 2

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame({"text_clean": [], "language": []})
        result = add_sentiment_to_dataframe(df)
        assert result.empty
