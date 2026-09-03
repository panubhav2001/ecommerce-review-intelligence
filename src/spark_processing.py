"""
spark_processing.py

Uses PySpark (running in local mode) to demonstrate how the review
cleaning and aggregation logic can scale to much larger datasets than
pandas could comfortably handle on a single machine.

This module takes a pandas DataFrame (already mapped to the standard
schema), converts it to a Spark DataFrame, performs cleaning /
de-duplication / null-handling / transformations and summary
statistics using Spark's DataFrame API, then returns the cleaned
result back as a pandas DataFrame for the rest of the pipeline
(sentiment analysis, TF-IDF, SQLite) to consume.

Spark runs entirely locally (local[*]) - no cluster, no external
services required.
"""

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import os
import sys
import tempfile
import uuid

def get_spark_session(app_name: str = "EcommerceReviewIntelligence") -> SparkSession:
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[2]")  # fewer concurrent workers = fewer Windows socket/AV timing issues
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.python.worker.reuse", "false")
        .config("spark.local.dir", os.path.join(os.environ.get("TEMP", "C:\\temp"), "spark-tmp"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


_SPARK_SCHEMA = StructType(
    [
        StructField("product_id", StringType(), True),
        StructField("product_name", StringType(), True),
        StructField("rating", DoubleType(), True),
        StructField("review_title", StringType(), True),
        StructField("review_text", StringType(), True),
        StructField("review_date", StringType(), True),
        StructField("cleaned_text", StringType(), True),
    ]
)

_INPUT_COLUMNS = [
    "product_id",
    "product_name",
    "rating",
    "review_title",
    "review_text",
    "review_date",
    "cleaned_text",
]


def process_reviews_with_spark(df: pd.DataFrame, spark: SparkSession = None):
    """
    Runs the Spark-based cleaning/transformation stage of the pipeline.

    Expects `df` to already have gone through the pandas-level cleaning
    step (src/preprocessing.py) so it includes a `cleaned_text` column.
    This stage demonstrates how the same kind of cleaning, null-handling,
    de-duplication and aggregation logic can be expressed with PySpark's
    DataFrame API so it scales to much larger datasets.

    Steps performed with PySpark's DataFrame API:
        1. Convert the pandas DataFrame to a Spark DataFrame
        2. Handle null / blank values
        3. Trim whitespace on text columns (basic transformation)
        4. Enforce a valid rating range (1-5)
        5. Drop exact duplicate reviews (product_id + cleaned_text)
        6. Compute summary statistics (total reviews, avg rating,
           reviews by rating, reviews by product)
        7. Convert the cleaned Spark DataFrame back to pandas

    Returns:
        (cleaned_pandas_df, summary_stats_dict)
    """
    owns_spark = spark is None
    if spark is None:
        spark = get_spark_session()

    # Spark is picky about NaN vs None and mixed types coming from
    # pandas, so coerce ratings to a plain float/None first.
    prep_df = df.copy()
    prep_df["rating"] = pd.to_numeric(prep_df["rating"], errors="coerce").astype("float64")
    for col in ["product_id", "product_name", "review_title", "review_text", "review_date", "cleaned_text"]:
        prep_df[col] = prep_df[col].astype("object").where(prep_df[col].notna(), None)

    tmp_dir = tempfile.mkdtemp(prefix="spark_input_")
    tmp_csv_path = os.path.join(tmp_dir, f"input_{uuid.uuid4().hex}.csv")
    prep_df[_INPUT_COLUMNS].to_csv(tmp_csv_path, index=False)

    spark_df = (
        spark.read
        .option("header", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .schema(_SPARK_SCHEMA)
        .csv(tmp_csv_path)
    )

    total_loaded = spark_df.count()

    # --- Null / missing value handling ---
    spark_df = spark_df.filter(F.col("cleaned_text").isNotNull() & (F.trim(F.col("cleaned_text")) != ""))
    spark_df = spark_df.na.fill({"product_name": "(unknown product)", "product_id": "UNKNOWN", "review_title": "(no title)"})

    # --- Basic transformations: trim whitespace on text fields ---
    spark_df = spark_df.withColumn("product_name", F.trim(F.col("product_name")))
    spark_df = spark_df.withColumn("review_text", F.trim(F.col("review_text")))
    spark_df = spark_df.withColumn("cleaned_text", F.trim(F.col("cleaned_text")))
    spark_df = spark_df.withColumn("review_title", F.trim(F.col("review_title")))

    # --- Valid rating range only (1-5) ---
    spark_df = spark_df.filter(F.col("rating").isNotNull() & F.col("rating").between(1, 5))

    # --- Remove exact duplicate reviews (same product + same cleaned text) ---
    spark_df = spark_df.dropDuplicates(["product_id", "cleaned_text"])

    total_after_cleaning = spark_df.count()

    # --- Summary statistics computed with Spark ---
    avg_rating_row = spark_df.agg(F.avg("rating").alias("avg_rating")).collect()[0]
    average_rating = round(avg_rating_row["avg_rating"], 2) if avg_rating_row["avg_rating"] is not None else 0.0

    reviews_by_rating = (
        spark_df.groupBy("rating")
        .count()
        .orderBy("rating")
        .toPandas()
        .rename(columns={"count": "review_count"})
    )

    reviews_by_product = (
        spark_df.groupBy("product_id", "product_name")
        .agg(F.count("*").alias("review_count"), F.avg("rating").alias("average_rating"))
        .orderBy(F.desc("review_count"))
        .toPandas()
    )
    reviews_by_product["average_rating"] = reviews_by_product["average_rating"].round(2)

    summary_stats = {
        "total_loaded": total_loaded,
        "total_after_cleaning": total_after_cleaning,
        "average_rating": average_rating,
        "reviews_by_rating": reviews_by_rating,
        "reviews_by_product": reviews_by_product,
    }

    cleaned_pandas_df = spark_df.toPandas()

    if owns_spark:
        spark.stop()
    try:
        os.remove(tmp_csv_path)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return cleaned_pandas_df, summary_stats
