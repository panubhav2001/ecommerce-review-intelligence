"""
database.py

Handles all SQLite persistence for the project.

Tables:
    reviews          - one row per processed review
    product_summary  - one row per product, aggregated stats
"""

import os
import sqlite3
import pandas as pd

DEFAULT_DB_PATH = "database/reviews.db"

REVIEWS_COLUMNS = [
    "product_id",
    "product_name",
    "rating",
    "review_title",
    "review_text",
    "review_date",
    "cleaned_text",
    "positive_score",
    "negative_score",
    "neutral_score",
    "compound_score",
    "sentiment",
]

PRODUCT_SUMMARY_COLUMNS = [
    "product_id",
    "product_name",
    "review_count",
    "average_rating",
    "positive_percentage",
    "negative_percentage",
    "neutral_percentage",
]


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Opens (creating parent directories/file if needed) a SQLite connection."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)


def initialize_database(db_path: str = DEFAULT_DB_PATH) -> None:
    """Creates the reviews and product_summary tables if they don't already exist."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT,
                product_name TEXT,
                rating REAL,
                review_title TEXT,
                review_text TEXT,
                review_date TEXT,
                cleaned_text TEXT,
                positive_score REAL,
                negative_score REAL,
                neutral_score REAL,
                compound_score REAL,
                sentiment TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS product_summary (
                product_id TEXT PRIMARY KEY,
                product_name TEXT,
                review_count INTEGER,
                average_rating REAL,
                positive_percentage REAL,
                negative_percentage REAL,
                neutral_percentage REAL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_reviews(df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Replaces the contents of the `reviews` table with the rows in `df`.
    `df` must contain (at least) all REVIEWS_COLUMNS.
    """
    initialize_database(db_path)

    missing = [c for c in REVIEWS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot save reviews: missing columns {missing}")

    conn = get_connection(db_path)
    try:
        conn.execute("DELETE FROM reviews")
        df[REVIEWS_COLUMNS].to_sql("reviews", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()


def build_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates a processed reviews DataFrame into the product_summary shape.
    """
    grouped = df.groupby(["product_id", "product_name"])

    summary_rows = []
    for (product_id, product_name), group in grouped:
        review_count = len(group)
        average_rating = round(group["rating"].mean(), 2) if review_count else 0
        positive_pct = round((group["sentiment"] == "Positive").mean() * 100, 1) if review_count else 0
        negative_pct = round((group["sentiment"] == "Negative").mean() * 100, 1) if review_count else 0
        neutral_pct = round((group["sentiment"] == "Neutral").mean() * 100, 1) if review_count else 0

        summary_rows.append(
            {
                "product_id": product_id,
                "product_name": product_name,
                "review_count": review_count,
                "average_rating": average_rating,
                "positive_percentage": positive_pct,
                "negative_percentage": negative_pct,
                "neutral_percentage": neutral_pct,
            }
        )

    return pd.DataFrame(summary_rows, columns=PRODUCT_SUMMARY_COLUMNS)


def save_product_summary(df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH) -> None:
    """Replaces the contents of the product_summary table."""
    initialize_database(db_path)

    summary_df = build_product_summary(df)

    conn = get_connection(db_path)
    try:
        conn.execute("DELETE FROM product_summary")
        summary_df.to_sql("product_summary", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()


def load_reviews_from_db(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Loads the full reviews table into a DataFrame. Returns an empty DataFrame if unavailable."""
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=REVIEWS_COLUMNS)

    conn = get_connection(db_path)
    try:
        try:
            df = pd.read_sql_query("SELECT * FROM reviews", conn)
        except (pd.errors.DatabaseError, sqlite3.OperationalError):
            return pd.DataFrame(columns=REVIEWS_COLUMNS)
    finally:
        conn.close()

    return df


def load_product_summary_from_db(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Loads the product_summary table into a DataFrame. Returns an empty DataFrame if unavailable."""
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=PRODUCT_SUMMARY_COLUMNS)

    conn = get_connection(db_path)
    try:
        try:
            df = pd.read_sql_query("SELECT * FROM product_summary", conn)
        except (pd.errors.DatabaseError, sqlite3.OperationalError):
            return pd.DataFrame(columns=PRODUCT_SUMMARY_COLUMNS)
    finally:
        conn.close()

    return df
