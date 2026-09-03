"""
data_loader.py

Handles loading the e-commerce review dataset from CSV.

Responsibilities:
    - Locate (or create) the dataset at data/reviews.csv
    - Map flexible / inconsistent column names to a standard internal schema
    - Generate a small sample dataset automatically if no file is found,
      so the project can always be demonstrated end-to-end.
"""

import os
import random
import datetime
import pandas as pd

# Standard internal column names used everywhere else in the project.
STANDARD_COLUMNS = [
    "product_id",
    "product_name",
    "rating",
    "review_title",
    "review_text",
    "review_date",
]

# Maps many possible "raw" column name variants (lowercased) to the
# standard internal column name. This makes the loader tolerant of
# datasets like the classic Amazon Fine Foods Reviews CSV
# (ProductId, ProductName, Score, Summary, Text, Time) as well as
# more human-friendly headers.
COLUMN_ALIASES = {
    "product_id": ["product_id", "productid", "asin", "id", "product id"],
    "product_name": ["product_name", "productname", "product", "name", "title_product", "product title"],
    "rating": ["rating", "score", "stars", "star_rating", "overall"],
    "review_title": ["review_title", "summary", "title", "headline", "review title"],
    "review_text": ["review_text", "text", "review", "body", "reviewtext", "review body", "content"],
    "review_date": ["review_date", "time", "date", "review_time", "timestamp", "reviewdate"],
}


def _build_reverse_lookup():
    """Builds a flat {raw_name_lowercase: standard_name} lookup table."""
    reverse = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            reverse[alias.strip().lower()] = standard_name
    return reverse


def map_columns_to_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames the columns of `df` to the standard internal schema, based on
    flexible alias matching. Any column that cannot be matched is left
    untouched (and will simply be ignored downstream).

    Any standard column that never gets matched is created and filled
    with sensible defaults, so downstream code can always rely on the
    full STANDARD_COLUMNS set being present.
    """
    reverse_lookup = _build_reverse_lookup()

    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in reverse_lookup:
            rename_map[col] = reverse_lookup[key]

    df = df.rename(columns=rename_map)

    # Ensure every standard column exists, even if the source file was
    # missing one entirely (e.g. no review_date column at all).
    for standard_col in STANDARD_COLUMNS:
        if standard_col not in df.columns:
            df[standard_col] = None

    # If duplicate standard column names resulted from renaming (e.g. two
    # source columns mapped to the same standard name), keep the first
    # occurrence only.
    df = df.loc[:, ~df.columns.duplicated()]

    return df[STANDARD_COLUMNS + [c for c in df.columns if c not in STANDARD_COLUMNS]]


def generate_sample_dataset(path: str, n_rows: int = 900) -> None:
    """
    Creates a small, realistic sample review dataset at `path` so the
    project can be run and demonstrated even without a real dataset.
    """
    random.seed(42)

    products = [
        ("P001", "Wireless Bluetooth Earbuds"),
        ("P002", "Stainless Steel Water Bottle"),
        ("P003", "Ergonomic Office Chair"),
        ("P004", "USB-C Fast Charger"),
        ("P005", "Non-Stick Frying Pan"),
        ("P006", "Portable Power Bank"),
        ("P007", "Memory Foam Pillow"),
        ("P008", "Smart Fitness Watch"),
    ]

    positive_snippets = [
        "The battery life is amazing and lasts all day.",
        "Great quality for the price, very happy with this purchase.",
        "Works perfectly and the build quality feels premium.",
        "Excellent value, I would definitely buy this again.",
        "Super comfortable and well designed.",
        "Fast shipping and the product exceeded my expectations.",
        "The performance is outstanding, highly recommend.",
        "Good customer service and a solid, reliable product.",
        "This is exactly what I needed, works great every time.",
        "Impressive design and the material quality feels durable.",
        "Setup was quick and easy, works flawlessly out of the box.",
        "The value for money here is fantastic, very pleased.",
        "Sturdy construction and it looks even better in person.",
        "My whole family loves this, well worth the purchase.",
        "Battery charges fast and holds a charge for days.",
        "Customer support was quick to help and very friendly.",
        "The design is sleek and it feels premium in the hand.",
        "Reliable performance every single day, no complaints at all.",
        "Great gift idea, everyone who received one loved it.",
        "Packaging was neat and the product arrived in perfect condition.",
    ]

    negative_snippets = [
        "The battery died after just two weeks, very disappointed.",
        "Arrived damaged and the packaging was terrible.",
        "Poor quality, it broke after a few days of normal use.",
        "Delivery took way too long and customer service was unhelpful.",
        "Not worth the price, feels cheap and flimsy.",
        "Stopped working after a month, would not recommend.",
        "The color was wrong and the item was not as described.",
        "Terrible experience, the product was defective on arrival.",
        "Packaging was crushed and the box was falling apart.",
        "Customer service never responded to my complaint about the defect.",
        "The material feels cheap and started peeling after a week.",
        "Shipping was delayed by over two weeks with no updates.",
        "It stopped charging properly after only a few uses.",
        "The size was completely different from what was advertised.",
        "Very disappointed, the product does not match the description.",
        "Poor packaging led to the item breaking in transit.",
        "The price is too high for such low build quality.",
        "It arrived late and missing several parts from the box.",
        "This broke almost immediately, a complete waste of money.",
        "Refund process was slow and the support team was unhelpful.",
    ]

    neutral_snippets = [
        "It's an average product, does what it says.",
        "Okay for the price, nothing special but works fine.",
        "Decent product, met my basic expectations.",
        "It's fine, not great but not bad either.",
        "Does the job but nothing stands out about it.",
        "Average quality, similar to other products in this price range.",
        "It works as described, no major issues so far.",
        "Reasonable purchase, though I expected slightly better quality.",
    ]

    rows = []
    for i in range(n_rows):
        product_id, product_name = random.choice(products)
        sentiment_bucket = random.choices(
            ["positive", "negative", "neutral"], weights=[0.55, 0.30, 0.15]
        )[0]

        if sentiment_bucket == "positive":
            text = random.choice(positive_snippets)
            rating = random.choices([5, 4], weights=[0.7, 0.3])[0]
            title = "Great product"
        elif sentiment_bucket == "negative":
            text = random.choice(negative_snippets)
            rating = random.choices([1, 2], weights=[0.6, 0.4])[0]
            title = "Not satisfied"
        else:
            text = random.choice(neutral_snippets)
            rating = 3
            title = "It's okay"

        # Occasionally duplicate a review or blank out a field, so the
        # cleaning step in the pipeline has real work to do.
        if random.random() < 0.03:
            text = None
        if random.random() < 0.02:
            title = None

        review_date = f"202{random.randint(2, 4)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"

        rows.append(
            {
                "product_id": product_id,
                "product_name": product_name,
                "rating": rating,
                "review_title": title,
                "review_text": text,
                "review_date": review_date,
            }
        )

    # Inject a handful of exact duplicate rows to exercise de-duplication.
    for _ in range(5):
        rows.append(dict(random.choice(rows)))

    sample_df = pd.DataFrame(rows, columns=STANDARD_COLUMNS)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    sample_df.to_csv(path, index=False)


# Field prefixes used by the raw Amazon-style "key: value" review dump
# (e.g. foods.txt from the Amazon Fine Foods Reviews dataset).
_AMAZON_TXT_KEYS = {
    "product/productId", "review/userId", "review/profileName",
    "review/helpfulness", "review/score", "review/time",
    "review/summary", "review/text",
}


def parse_amazon_reviews_txt(path: str) -> pd.DataFrame:
    """
    Parses the raw Amazon-style review dump format used by datasets like
    foods.txt (Amazon Fine Foods Reviews), where each review is a block
    of "key: value" lines separated by blank lines, e.g.:

        product/productId: B001E4KFG0
        review/userId: A3SGXH7AUHU8GW
        review/profileName: delmartian
        review/helpfulness: 1/1
        review/score: 5.0
        review/time: 1303862400
        review/summary: Good Quality Dog Food
        review/text: I have bought several of the Vitality canned...

    Handles the rare case where review/text or review/summary contains
    an embedded newline (the continuation line is appended to whichever
    field was most recently seen).

    This dataset has no product name field, so product_name is set
    equal to product_id.

    Returns a DataFrame with the standard internal column schema.
    """
    records = []
    current = {}
    current_key = None

    with open(path, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line.strip() == "":
                if current:
                    records.append(current)
                current = {}
                current_key = None
                continue

            prefix, sep, value = line.partition(": ")
            if sep and prefix in _AMAZON_TXT_KEYS:
                current[prefix] = value
                current_key = prefix
            elif current_key:
                # Continuation of a multi-line field (rare).
                current[current_key] += "\n" + line

        if current:
            records.append(current)

    if not records:
        raise ValueError(f"No reviews could be parsed from '{path}'.")

    raw_df = pd.DataFrame(records)

    def _to_date(unix_ts):
        try:
            return datetime.datetime.utcfromtimestamp(int(unix_ts)).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return None

    df = pd.DataFrame(
        {
            "product_id": raw_df.get("product/productId"),
            "product_name": raw_df.get("product/productId"),  # no product name in this dataset
            "rating": pd.to_numeric(raw_df.get("review/score"), errors="coerce"),
            "review_title": raw_df.get("review/summary"),
            "review_text": raw_df.get("review/text"),
            "review_date": raw_df.get("review/time").apply(_to_date) if "review/time" in raw_df else None,
        }
    )

    return df[STANDARD_COLUMNS]


def load_reviews(path: str = "data/reviews.csv") -> pd.DataFrame:
    """
    Loads the review dataset from `path`, generating a sample dataset
    automatically if the file does not exist.

    Supports two formats, dispatched on file extension:
        - .csv  -> standard CSV with flexible column mapping
        - .txt  -> raw Amazon-style "key: value" review dump
                   (e.g. foods.txt / Amazon Fine Foods Reviews)

    Returns a DataFrame with the standard internal column schema.
    """
    if not os.path.exists(path):
        if path.lower().endswith(".txt"):
            raise ValueError(f"Dataset file '{path}' was not found.")
        print(f"No dataset found at '{path}'. Generating a sample dataset for demonstration...")
        generate_sample_dataset(path)

    if path.lower().endswith(".txt"):
        df = parse_amazon_reviews_txt(path)
        if df.empty:
            raise ValueError(f"The dataset at '{path}' contains no rows.")
        return df

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"The dataset at '{path}' is empty.")
    except Exception as exc:
        raise ValueError(f"Failed to read dataset at '{path}': {exc}")

    if df.empty:
        raise ValueError(f"The dataset at '{path}' contains no rows.")

    df = map_columns_to_standard(df)
    return df