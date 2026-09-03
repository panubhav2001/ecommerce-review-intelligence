"""
pipeline.py

Runs the complete end-to-end data processing workflow for the
E-Commerce Product Review Intelligence project:

    CSV
     -> Load data
     -> Clean data / remove duplicates / handle missing values / normalize text
     -> PySpark processing (cleaning + summary statistics at scale)
     -> Sentiment analysis (VADER)
     -> Keyword analysis (TF-IDF)
     -> Save to SQLite

Run with:
    python pipeline.py
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore")

from src.data_loader import load_reviews
from src.preprocessing import clean_reviews_dataframe
from src.spark_processing import get_spark_session, process_reviews_with_spark
from src.sentiment import add_sentiment_columns
from src.keyword_analysis import get_keyword_analysis
from src.database import save_reviews, save_product_summary

DATA_PATH = "data/foods.txt"
DB_PATH = "database/reviews.db"


def run_pipeline():
    start_time = time.time()

    print("=" * 60)
    print("E-Commerce Product Review Intelligence - Pipeline")
    print("=" * 60)

    # --- 1. Load data ---
    print("\nLoading dataset...")
    try:
        raw_df = load_reviews(DATA_PATH)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print(f"Records loaded: {len(raw_df):,}")

    # --- 2. Clean data (pandas: normalize text, handle missing, dedupe) ---
    print("\nCleaning data...")
    clean_df = clean_reviews_dataframe(raw_df)
    print(f"Records after cleaning: {len(clean_df):,}")

    if clean_df.empty:
        print("ERROR: No valid reviews remain after cleaning. Check your dataset.")
        sys.exit(1)

    # --- 3. PySpark processing (scalable cleaning + summary stats) ---
    print("\nRunning PySpark processing...")
    spark = get_spark_session()
    try:
        spark_df, summary_stats = process_reviews_with_spark(clean_df, spark=spark)
    finally:
        spark.stop()
    print(f"Records after PySpark processing: {len(spark_df):,}")
    print(f"Average rating: {summary_stats['average_rating']}")
    print("PySpark processing completed.")

    if spark_df.empty:
        print("ERROR: No valid reviews remain after PySpark processing.")
        sys.exit(1)

    # --- 4. Sentiment analysis ---
    print("\nRunning sentiment analysis...")
    sentiment_df = add_sentiment_columns(spark_df)
    print("Sentiment analysis completed.")
    sentiment_counts = sentiment_df["sentiment"].value_counts()
    for label in ["Positive", "Neutral", "Negative"]:
        print(f"  {label}: {sentiment_counts.get(label, 0):,}")

    # --- 5. Keyword analysis ---
    print("\nExtracting keywords...")
    keyword_results = get_keyword_analysis(sentiment_df)
    print("Keyword analysis completed.")
    top_positive = [w for w, _ in keyword_results["positive"][:5]]
    top_negative = [w for w, _ in keyword_results["negative"][:5]]
    print(f"  Top positive keywords: {', '.join(top_positive) if top_positive else 'N/A'}")
    print(f"  Top negative keywords: {', '.join(top_negative) if top_negative else 'N/A'}")

    # --- 6. Save results to SQLite ---
    print("\nSaving results to SQLite...")
    save_reviews(sentiment_df, DB_PATH)
    save_product_summary(sentiment_df, DB_PATH)
    print(f"Results saved to '{DB_PATH}'.")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Pipeline completed successfully in {elapsed:.1f} seconds.")
    print("=" * 60)
    print("\nNext step: run 'streamlit run app.py' to explore the dashboard.")


if __name__ == "__main__":
    run_pipeline()
