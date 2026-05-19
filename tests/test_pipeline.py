"""
test_pipeline.py
----------------
Unit tests for the fintech review analytics pipeline.
Tests scraper, sentiment, themes, and database modules.

Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scraper import preprocess_reviews, BANK_APPS
from src.sentiment import (
    clean_text, get_vader_score, get_textblob_score,
    label_sentiment, apply_sentiment_pipeline
)
from src.themes import (
    assign_theme, apply_theme_pipeline,
    extract_tfidf_keywords, THEME_KEYWORDS
)


# ─────────────────────────────────────────
# FIXTURES — reusable test data
# ─────────────────────────────────────────

@pytest.fixture
def sample_reviews_df():
    """Sample DataFrame mimicking scraped reviews."""
    return pd.DataFrame({
        "review": [
            "The app crashes every time I try to login",
            "Great app, very fast and easy to use",
            "Cannot transfer money, always fails",
            "Best banking app I have ever used",
            "OTP never arrives, login is broken",
            "Nice interface and smooth navigation",
            "Customer support never responds to my calls",
            "Transaction keeps pending for days",
        ],
        "rating": [1, 5, 1, 5, 1, 4, 2, 2],
        "date": [
            "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04",
            "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"
        ],
        "bank": [
            "Commercial Bank of Ethiopia",
            "Commercial Bank of Ethiopia",
            "Bank of Abyssinia",
            "Bank of Abyssinia",
            "Dashen Bank",
            "Dashen Bank",
            "Commercial Bank of Ethiopia",
            "Bank of Abyssinia"
        ],
        "source": ["Google Play"] * 8
    })


@pytest.fixture
def raw_reviews_df():
    """Raw DataFrame with duplicates and missing values for preprocessing tests."""
    return pd.DataFrame({
        "review": [
            "Good app", "Good app", "Bad experience",
            None, "Fast transfers", ""
        ],
        "rating": [5, 5, 1, 4, 5, 3],
        "date": ["2026-01-01"] * 6,
        "bank": ["CBE"] * 6,
        "source": ["Google Play"] * 6
    })


# ─────────────────────────────────────────
# TASK 1 TESTS — Scraper & Preprocessing
# ─────────────────────────────────────────

class TestBankApps:
    """Tests for bank app configuration."""

    def test_bank_apps_has_three_banks(self):
        """BANK_APPS should contain exactly 3 Ethiopian banks."""
        assert len(BANK_APPS) == 3

    def test_bank_apps_has_correct_names(self):
        """BANK_APPS should contain CBE, BOA, and Dashen."""
        expected = {
            "Commercial Bank of Ethiopia",
            "Bank of Abyssinia",
            "Dashen Bank"
        }
        assert set(BANK_APPS.keys()) == expected

    def test_bank_apps_ids_are_strings(self):
        """All app IDs should be non-empty strings."""
        for name, app_id in BANK_APPS.items():
            assert isinstance(app_id, str), f"{name} app ID is not a string"
            assert len(app_id) > 0, f"{name} app ID is empty"

    def test_dashen_correct_app_id(self):
        """Dashen Bank should use the correct super app ID."""
        assert BANK_APPS["Dashen Bank"] == "com.dashen.dashensuperapp"


class TestPreprocessing:
    """Tests for review preprocessing pipeline."""

    def test_removes_duplicates(self, raw_reviews_df):
        """Preprocessing should remove duplicate reviews."""
        result = preprocess_reviews(raw_reviews_df.copy())
        assert result.duplicated(subset=["review", "bank"]).sum() == 0

    def test_drops_missing_reviews(self, raw_reviews_df):
        """Preprocessing should drop rows with missing review text."""
        result = preprocess_reviews(raw_reviews_df.copy())
        assert result["review"].isna().sum() == 0

    def test_drops_empty_reviews(self, raw_reviews_df):
        """Preprocessing should drop empty string reviews."""
        result = preprocess_reviews(raw_reviews_df.copy())
        assert (result["review"].str.strip() == "").sum() == 0

    def test_date_format_normalized(self, sample_reviews_df):
        """Dates should be in YYYY-MM-DD format after preprocessing."""
        result = preprocess_reviews(sample_reviews_df.copy())
        for date in result["date"].dropna():
            assert len(date) == 10
            assert date[4] == "-"
            assert date[7] == "-"

    def test_output_has_required_columns(self, sample_reviews_df):
        """Preprocessed DataFrame should have exactly 5 required columns."""
        result = preprocess_reviews(sample_reviews_df.copy())
        required = {"review", "rating", "date", "bank", "source"}
        assert required.issubset(set(result.columns))

    def test_output_not_empty(self, sample_reviews_df):
        """Preprocessed DataFrame should not be empty."""
        result = preprocess_reviews(sample_reviews_df.copy())
        assert len(result) > 0


# ─────────────────────────────────────────
# TASK 2 TESTS — Sentiment Analysis
# ─────────────────────────────────────────

class TestTextCleaning:
    """Tests for text cleaning pipeline."""

    def test_removes_special_characters(self):
        """clean_text should remove non-alphabetic characters."""
        result = clean_text("Hello! This is a test123.")
        assert "!" not in result
        assert "123" not in result

    def test_converts_to_lowercase(self):
        """clean_text should convert text to lowercase."""
        result = clean_text("HELLO WORLD")
        assert result == result.lower()

    def test_removes_stop_words(self):
        """clean_text should remove common stop words."""
        result = clean_text("this is the best app")
        assert "this" not in result.split()
        assert "is" not in result.split()
        assert "the" not in result.split()

    def test_handles_empty_string(self):
        """clean_text should handle empty string input."""
        result = clean_text("")
        assert isinstance(result, str)

    def test_handles_none_input(self):
        """clean_text should handle None input gracefully."""
        result = clean_text(None)
        assert isinstance(result, str)

    def test_returns_string(self):
        """clean_text should always return a string."""
        for text in ["hello", "", "123", None, "!@#$"]:
            assert isinstance(clean_text(text), str)


class TestVADERSentiment:
    """Tests for VADER sentiment scoring."""

    def test_positive_review_gets_positive_score(self):
        """A clearly positive review should get score >= 0.05."""
        score = get_vader_score("This app is excellent and amazing!")
        assert score >= 0.05

    def test_negative_review_gets_negative_score(self):
        """A clearly negative review should get score <= -0.05."""
        score = get_vader_score("This app is terrible and crashes constantly")
        assert score <= -0.05

    def test_score_in_valid_range(self):
        """VADER compound score should always be between -1 and +1."""
        texts = [
            "great app", "worst ever", "okay I guess",
            "", None, "!!!", "123"
        ]
        for text in texts:
            score = get_vader_score(text)
            assert -1.0 <= score <= 1.0, f"Score {score} out of range for: {text}"

    def test_handles_empty_string(self):
        """get_vader_score should return 0.0 for empty string."""
        assert get_vader_score("") == 0.0

    def test_handles_none(self):
        """get_vader_score should return 0.0 for None input."""
        assert get_vader_score(None) == 0.0

    def test_returns_float(self):
        """get_vader_score should always return a float."""
        assert isinstance(get_vader_score("good app"), float)


class TestTextBlobSentiment:
    """Tests for TextBlob sentiment scoring."""

    def test_score_in_valid_range(self):
        """TextBlob polarity should be between -1 and +1."""
        texts = ["great", "terrible", "okay", "", None]
        for text in texts:
            score = get_textblob_score(text)
            assert -1.0 <= score <= 1.0

    def test_returns_float(self):
        """get_textblob_score should return a float."""
        assert isinstance(get_textblob_score("good"), float)


class TestSentimentLabeling:
    """Tests for sentiment label assignment."""

    def test_positive_threshold(self):
        """Score >= 0.05 should be labeled positive."""
        assert label_sentiment(0.05) == "positive"
        assert label_sentiment(0.5) == "positive"
        assert label_sentiment(1.0) == "positive"

    def test_negative_threshold(self):
        """Score <= -0.05 should be labeled negative."""
        assert label_sentiment(-0.05) == "negative"
        assert label_sentiment(-0.5) == "negative"
        assert label_sentiment(-1.0) == "negative"

    def test_neutral_threshold(self):
        """Score between -0.05 and 0.05 should be neutral."""
        assert label_sentiment(0.0) == "neutral"
        assert label_sentiment(0.04) == "neutral"
        assert label_sentiment(-0.04) == "neutral"

    def test_returns_valid_label(self):
        """label_sentiment should only return valid labels."""
        valid_labels = {"positive", "negative", "neutral"}
        for score in [-1.0, -0.5, -0.05, 0.0, 0.05, 0.5, 1.0]:
            assert label_sentiment(score) in valid_labels


class TestSentimentPipeline:
    """Tests for the full sentiment pipeline."""

    def test_adds_required_columns(self, sample_reviews_df):
        """Pipeline should add sentiment_score, sentiment_label, tb_polarity."""
        result = apply_sentiment_pipeline(sample_reviews_df.copy())
        assert "sentiment_score" in result.columns
        assert "sentiment_label" in result.columns
        assert "tb_polarity" in result.columns
        assert "clean_text" in result.columns

    def test_no_null_sentiment_scores(self, sample_reviews_df):
        """Pipeline should not produce null sentiment scores."""
        result = apply_sentiment_pipeline(sample_reviews_df.copy())
        assert result["sentiment_score"].isna().sum() == 0

    def test_all_labels_valid(self, sample_reviews_df):
        """All sentiment labels should be positive, negative, or neutral."""
        result = apply_sentiment_pipeline(sample_reviews_df.copy())
        valid = {"positive", "negative", "neutral"}
        assert set(result["sentiment_label"].unique()).issubset(valid)

    def test_preserves_row_count(self, sample_reviews_df):
        """Pipeline should not drop any rows."""
        result = apply_sentiment_pipeline(sample_reviews_df.copy())
        assert len(result) == len(sample_reviews_df)


# ─────────────────────────────────────────
# TASK 2 TESTS — Thematic Analysis
# ─────────────────────────────────────────

class TestThemeAssignment:
    """Tests for theme keyword matching."""

    def test_crash_review_gets_app_stability(self):
        """Review mentioning crash should get App Stability theme."""
        theme = assign_theme("The app crashes every time I open it")
        assert theme == "App Stability"

    def test_login_review_gets_account_access(self):
        """Review mentioning login should get Account Access Issues theme."""
        theme = assign_theme("I cannot login, OTP never arrives")
        assert theme == "Account Access Issues"

    def test_transfer_review_gets_transaction(self):
        """Review mentioning transfer should get Transaction Performance theme."""
        theme = assign_theme("Money transfer is very slow and keeps failing")
        assert theme == "Transaction Performance"

    def test_ui_review_gets_ux_theme(self):
        """Review mentioning interface should get UI & User Experience theme."""
        theme = assign_theme("The interface is easy and navigation is smooth")
        assert theme == "UI & User Experience"

    def test_support_review_gets_customer_support(self):
        """Review mentioning support should get Customer Support theme."""
        theme = assign_theme("Customer support never responds to my calls")
        assert theme == "Customer Support"

    def test_unrelated_review_gets_general_feedback(self):
        """Review with no theme keywords should get General Feedback."""
        theme = assign_theme("incredible")
        assert theme == "General Feedback"

    def test_returns_string(self):
        """assign_theme should always return a string."""
        for text in ["crash login", "", None, "great"]:
            assert isinstance(assign_theme(text), str)

    def test_theme_keywords_has_five_themes(self):
        """THEME_KEYWORDS should define exactly 5 themes."""
        assert len(THEME_KEYWORDS) == 5


class TestThemePipeline:
    """Tests for the full theme assignment pipeline."""

    def test_adds_theme_column(self, sample_reviews_df):
        """Pipeline should add identified_theme column."""
        result = apply_theme_pipeline(sample_reviews_df.copy())
        assert "identified_theme" in result.columns

    def test_no_null_themes(self, sample_reviews_df):
        """Pipeline should not produce null themes."""
        result = apply_theme_pipeline(sample_reviews_df.copy())
        assert result["identified_theme"].isna().sum() == 0

    def test_all_themes_valid(self, sample_reviews_df):
        """All themes should be from the defined set or General Feedback."""
        result = apply_theme_pipeline(sample_reviews_df.copy())
        valid_themes = set(THEME_KEYWORDS.keys()) | {"General Feedback"}
        assert set(result["identified_theme"].unique()).issubset(valid_themes)

    def test_preserves_row_count(self, sample_reviews_df):
        """Pipeline should not drop any rows."""
        result = apply_theme_pipeline(sample_reviews_df.copy())
        assert len(result) == len(sample_reviews_df)


class TestTFIDF:
    """Tests for TF-IDF keyword extraction."""

    def test_returns_series(self, sample_reviews_df):
        """extract_tfidf_keywords should return a pandas Series."""
        result = extract_tfidf_keywords(sample_reviews_df["review"])
        assert isinstance(result, pd.Series)

    def test_respects_max_features(self, sample_reviews_df):
        """Result should not exceed max_features."""
        result = extract_tfidf_keywords(
            sample_reviews_df["review"], max_features=5)
        assert len(result) <= 5

    def test_scores_are_positive(self, sample_reviews_df):
        """All TF-IDF scores should be non-negative."""
        result = extract_tfidf_keywords(sample_reviews_df["review"])
        assert (result >= 0).all()

    def test_sorted_descending(self, sample_reviews_df):
        """Keywords should be sorted by score descending."""
        result = extract_tfidf_keywords(sample_reviews_df["review"])
        assert list(result.values) == sorted(result.values, reverse=True)