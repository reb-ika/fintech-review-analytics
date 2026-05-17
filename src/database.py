
"""
database.py
-----------
Modular database functions for PostgreSQL operations.
Used by task_3_database.ipynb
"""

import psycopg2  # type: ignore[import]
from psycopg2.extras import execute_values  # type: ignore[import]
import pandas as pd


def get_connection(host="localhost", port=5432,
                   database="bank_reviews",
                   user="postgres", password="postgres123"):
    """Create and return a PostgreSQL connection."""
    try:
        conn = psycopg2.connect(
            host=host, port=port,
            database=database,
            user=user, password=password
        )
        print(f"Connected to {database} successfully")
        return conn
    except Exception as e:
        print(f"Connection failed: {e}")
        raise


def insert_reviews(conn, df: pd.DataFrame, bank_map: dict) -> int:
    """
    Insert reviews DataFrame into PostgreSQL reviews table.
    
    Args:
        conn: Active psycopg2 connection
        df: DataFrame with review data
        bank_map: Dictionary mapping bank_name to bank_id
    
    Returns:
        Number of rows inserted
    """
    cursor = conn.cursor()
    reviews_data = []

    for _, row in df.iterrows():
        bank_id = bank_map.get(row["bank"])
        if bank_id is None:
            continue
        reviews_data.append((
            bank_id,
            str(row["review"]) if pd.notna(row["review"]) else None,
            int(row["rating"]) if pd.notna(row["rating"]) else None,
            str(row["date"]) if pd.notna(row["date"]) else None,
            str(row["sentiment_label"]) if pd.notna(
                row.get("sentiment_label", None)) else None,
            float(row["sentiment_score"]) if pd.notna(
                row.get("sentiment_score", None)) else None,
            str(row["identified_theme"]) if pd.notna(
                row.get("identified_theme", None)) else None,
            "Google Play"
        ))

    execute_values(cursor, """
        INSERT INTO reviews
        (bank_id, review_text, rating, review_date,
         sentiment_label, sentiment_score, identified_theme, source)
        VALUES %s
    """, reviews_data)

    conn.commit()
    cursor.close()
    return len(reviews_data)
