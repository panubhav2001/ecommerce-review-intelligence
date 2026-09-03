"""
sentiment.py

Local, rule-based sentiment analysis using VADER (Valence Aware Dictionary
and sEntiment Reasoner). No API keys or LLMs required.

For each review, computes:
    - positive_score
    - negative_score
    - neutral_score
    - compound_score
    - sentiment label (Positive / Neutral / Negative)

Classification rule (transparent, no black box):
    compound >= 0.05   -> Positive
    compound <= -0.05  -> Negative
    otherwise          -> Neutral
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def classify_compound(compound: float) -> str:
    """Applies the transparent threshold rule to a compound score."""
    if compound >= POSITIVE_THRESHOLD:
        return "Positive"
    elif compound <= NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def analyze_review_sentiment(text: str) -> dict:
    """
    Runs VADER on a single piece of text and returns a dict with the
    four VADER scores plus the derived sentiment label.
    """
    if not text:
        scores = {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}
    else:
        scores = _analyzer.polarity_scores(text)

    return {
        "positive_score": scores["pos"],
        "negative_score": scores["neg"],
        "neutral_score": scores["neu"],
        "compound_score": scores["compound"],
        "sentiment": classify_compound(scores["compound"]),
    }


def add_sentiment_columns(df: pd.DataFrame, text_column: str = "cleaned_text") -> pd.DataFrame:
    """
    Applies VADER sentiment analysis to every row of `df` and appends
    the resulting columns:
        positive_score, negative_score, neutral_score, compound_score, sentiment
    """
    df = df.copy()

    sentiment_results = df[text_column].apply(analyze_review_sentiment)
    sentiment_df = pd.DataFrame(list(sentiment_results), index=df.index)

    for col in ["positive_score", "negative_score", "neutral_score", "compound_score", "sentiment"]:
        df[col] = sentiment_df[col]

    return df
