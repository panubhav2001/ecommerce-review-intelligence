"""
preprocessing.py

Text cleaning utilities for review text.

The cleaning is intentionally gentle: it strips HTML, URLs, and stray
punctuation/whitespace, but it does NOT remove stopwords or "negative"
words like "not", "never", "bad", "poor", etc., because those words
carry real sentiment signal that VADER (and a human reader) relies on.
"""

import re
import html
import pandas as pd

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"http[s]?://\S+|www\.\S+")
# Keep letters, numbers, basic sentence punctuation that matters for
# sentiment (! and ?), and whitespace. Strip everything else.
_UNWANTED_PUNCT_RE = re.compile(r"[^a-z0-9\s!?.,']")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_review_text(text) -> str:
    """
    Cleans a single review's text.

    Steps:
        1. Handle missing / non-string input -> empty string
        2. Unescape HTML entities (&amp; -> &)
        3. Strip HTML tags
        4. Strip URLs
        5. Lowercase
        6. Remove unnecessary punctuation (keeps letters, numbers, ! ? . , ')
        7. Collapse repeated whitespace
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    text = str(text)

    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.lower()
    text = _UNWANTED_PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    return text


def clean_reviews_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies clean_review_text to the review_text column of a DataFrame,
    handles missing review titles, drops exact duplicate reviews, and
    removes rows with no usable review text.

    This is the pandas-level cleaning used before/around the PySpark
    processing stage (PySpark does its own de-duplication and null
    handling on the Spark DataFrame as required by the project spec).
    """
    df = df.copy()

    # Fill missing titles/product names so downstream display code
    # never has to deal with NaN.
    df["review_title"] = df["review_title"].fillna("(no title)")
    df["product_name"] = df["product_name"].fillna("(unknown product)")
    df["product_id"] = df["product_id"].fillna("UNKNOWN")

    # Coerce rating to numeric; invalid ratings become NaN and are
    # handled by the caller / PySpark stage.
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Clean the review text itself.
    df["cleaned_text"] = df["review_text"].apply(clean_review_text)

    # Drop rows that have no usable review text after cleaning.
    df = df[df["cleaned_text"].str.len() > 0]

    # Remove exact duplicate reviews (same product, same cleaned text).
    df = df.drop_duplicates(subset=["product_id", "cleaned_text"])

    # Drop rows with an invalid/missing rating (must be 1-5).
    df = df[df["rating"].between(1, 5, inclusive="both")]

    df = df.reset_index(drop=True)
    return df
