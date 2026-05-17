
"""
sentiment.py
------------
Modular sentiment analysis functions.
Supports VADER, TextBlob, and DistilBERT transformer.
Used by task_2_sentiment_analysis.ipynb
"""

import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

# Download required NLTK data
for res in ['vader_lexicon', 'stopwords', 'punkt',
            'wordnet', 'averaged_perceptron_tagger']:
    nltk.download(res, quiet=True)

analyzer = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean and preprocess text for NLP analysis.
    - Remove non-alphabetic characters
    - Tokenize
    - Remove stop words
    - Lemmatize
    """
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())
    tokens = word_tokenize(text)
    processed = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 2
    ]
    return " ".join(processed)


def get_vader_score(text: str) -> float:
    """Return VADER compound sentiment score (-1 to +1)."""
    try:
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0.0
        return analyzer.polarity_scores(text)['compound']
    except Exception:
        return 0.0


def get_textblob_score(text: str) -> float:
    """Return TextBlob polarity score (-1 to +1)."""
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception:
        return 0.0


def label_sentiment(score: float) -> str:
    """Convert compound score to sentiment label."""
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"


def apply_sentiment_pipeline(df: pd.DataFrame,
                              review_col: str = "review") -> pd.DataFrame:
    """
    Apply full sentiment pipeline to a DataFrame.
    
    Adds columns:
    - clean_text: preprocessed review text
    - sentiment_score: VADER compound score
    - sentiment_label: positive / neutral / negative
    - tb_polarity: TextBlob polarity score
    - tb_label: TextBlob sentiment label
    
    Args:
        df: DataFrame with review text column
        review_col: Name of the review text column
    
    Returns:
        DataFrame with sentiment columns added
    """
    print("Applying text cleaning...")
    df["clean_text"] = df[review_col].apply(clean_text)

    print("Applying VADER sentiment...")
    df["sentiment_score"] = df[review_col].apply(get_vader_score)
    df["sentiment_label"] = df["sentiment_score"].apply(label_sentiment)

    print("Applying TextBlob sentiment...")
    df["tb_polarity"] = df[review_col].apply(get_textblob_score)
    df["tb_label"] = df["tb_polarity"].apply(label_sentiment)

    print(f"Sentiment pipeline complete — {len(df)} reviews processed")
    return df
