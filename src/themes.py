
"""
themes.py
---------
Modular thematic analysis functions.
Uses keyword matching to assign business-relevant themes.
Used by task_2_sentiment_analysis.ipynb
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]


# Business-relevant theme keywords
THEME_KEYWORDS = {
    "Account Access Issues": [
        "login", "password", "otp", "fingerprint", "verify",
        "verification", "sign", "access", "account", "pin", "locked"
    ],
    "Transaction Performance": [
        "transfer", "slow", "fast", "payment", "send", "money",
        "transaction", "speed", "quick", "delay", "pending", "failed"
    ],
    "App Stability": [
        "crash", "error", "bug", "update", "fix", "freeze",
        "problem", "issue", "work", "open", "load", "loading"
    ],
    "UI & User Experience": [
        "easy", "simple", "design", "interface", "navigation",
        "nice", "beautiful", "user", "friendly", "convenient", "smooth"
    ],
    "Customer Support": [
        "support", "service", "response", "call", "help",
        "agent", "staff", "customer", "contact", "solve"
    ]
}


def assign_theme(review: str) -> str:
    """
    Assign the most relevant business theme to a review
    based on keyword matching.
    
    Args:
        review: Raw review text string
    
    Returns:
        Theme name string
    """
    if not isinstance(review, str):
        return "General Feedback"

    review_lower = review.lower()
    theme_scores = {
        theme: sum(1 for kw in keywords if kw in review_lower)
        for theme, keywords in THEME_KEYWORDS.items()
    }

    best_theme = max(theme_scores, key=theme_scores.get)
    if theme_scores[best_theme] == 0:
        return "General Feedback"
    return best_theme


def apply_theme_pipeline(df: pd.DataFrame,
                         review_col: str = "review") -> pd.DataFrame:
    """
    Apply theme assignment to all reviews in a DataFrame.
    
    Adds column:
    - identified_theme: assigned business theme
    
    Args:
        df: DataFrame with review text column
        review_col: Name of the review text column
    
    Returns:
        DataFrame with identified_theme column added
    """
    df["identified_theme"] = df[review_col].apply(assign_theme)
    print("Theme distribution:")
    print(df["identified_theme"].value_counts())
    return df


def extract_tfidf_keywords(corpus: pd.Series,
                           max_features: int = 20) -> pd.Series:
    """
    Extract top TF-IDF keywords from a corpus of text.
    
    Args:
        corpus: Series of text documents
        max_features: Number of top keywords to return
    
    Returns:
        Series of keyword scores sorted descending
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    X = vectorizer.fit_transform(corpus.fillna(""))
    scores = X.toarray().sum(axis=0)
    return pd.Series(
        scores,
        index=vectorizer.get_feature_names_out()
    ).sort_values(ascending=False)
