
"""
src package
-----------
Modular analytics pipeline for fintech review analysis.

Modules:
- scraper.py: Google Play Store scraping and preprocessing
- sentiment.py: VADER, TextBlob, DistilBERT sentiment analysis
- themes.py: TF-IDF keyword extraction and theme assignment
- database.py: PostgreSQL connection and data insertion
"""

from src.scraper import scrape_bank_reviews, preprocess_reviews, BANK_APPS
from src.sentiment import apply_sentiment_pipeline, get_vader_score, label_sentiment
from src.themes import apply_theme_pipeline, extract_tfidf_keywords, THEME_KEYWORDS
from src.database import get_connection, insert_reviews
