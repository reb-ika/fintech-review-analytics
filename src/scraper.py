
"""
scraper.py
----------
Modular scraping functions for Google Play Store reviews.
Used by task_1_scraping.ipynb
"""

from google_play_scraper import reviews, Sort
import pandas as pd


# Bank app IDs
BANK_APPS = {
    "Commercial Bank of Ethiopia": "com.combanketh.mobilebanking",
    "Bank of Abyssinia": "com.boa.boaMobileBanking",
    "Dashen Bank": "com.dashen.dashensuperapp"
}


def scrape_bank_reviews(app_dict: dict, count: int = 600) -> pd.DataFrame:
    """
    Scrape reviews from Google Play Store for given apps.
    
    Args:
        app_dict: Dictionary of {bank_name: app_id}
        count: Number of reviews to request per app
    
    Returns:
        DataFrame with columns: review, rating, date, bank, source
    """
    all_reviews = []

    for bank_name, app_id in app_dict.items():
        print(f"Scraping {bank_name}...")
        try:
            result, _ = reviews(
                app_id,
                lang='en',
                country='us',
                sort=Sort.NEWEST,
                count=count,
                filter_score_with=None
            )
            for r in result:
                all_reviews.append({
                    "review": r.get("content", ""),
                    "rating": r.get("score", None),
                    "date": r.get("at", None),
                    "bank": bank_name,
                    "source": "Google Play"
                })
            print(f"  Collected {len(result)} reviews")
        except Exception as e:
            print(f"  Error scraping {bank_name}: {e}")

    return pd.DataFrame(all_reviews)


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess raw review DataFrame.
    
    Steps:
    - Normalize dates to YYYY-MM-DD
    - Remove duplicates
    - Drop rows missing review text or rating
    - Remove empty reviews
    
    Args:
        df: Raw reviews DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    # Normalize dates
    df["date"] = pd.to_datetime(
        df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["review", "bank"])
    print(f"Duplicates removed: {before - len(df)}")

    # Drop missing review or rating
    before = len(df)
    df = df.dropna(subset=["review", "rating"])
    df = df[df["review"].str.strip() != ""]
    print(f"Rows dropped (missing): {before - len(df)}")

    # Ensure correct column order
    df = df[["review", "rating", "date", "bank", "source"]]
    return df.reset_index(drop=True)
