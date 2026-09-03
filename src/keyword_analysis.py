"""
keyword_analysis.py

Uses TF-IDF (scikit-learn) to identify the most important words/terms
in positive reviews, negative reviews, and all reviews combined.

This answers business questions like:
    - What are customers praising?
    - What are customers complaining about?
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_TOP_N = 20


def _top_keywords_for_texts(texts: list, top_n: int = DEFAULT_TOP_N) -> list:
    """
    Fits a TF-IDF vectorizer over `texts` and returns the top_n terms
    ranked by their summed TF-IDF weight across the corpus.

    Returns a list of (keyword, score) tuples. Returns an empty list if
    there is not enough text to build a vocabulary.
    """
    texts = [t for t in texts if isinstance(t, str) and t.strip()]

    if len(texts) < 2:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 1),
        min_df=1,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # Happens if the vocabulary ends up empty (e.g. all stopwords).
        return []

    feature_names = vectorizer.get_feature_names_out()
    summed_scores = tfidf_matrix.sum(axis=0).A1

    scored_terms = list(zip(feature_names, summed_scores))
    scored_terms.sort(key=lambda pair: pair[1], reverse=True)

    return scored_terms[:top_n]


def get_keyword_analysis(df: pd.DataFrame, text_column: str = "cleaned_text", top_n: int = DEFAULT_TOP_N) -> dict:
    """
    Computes top TF-IDF keywords for positive reviews, negative reviews,
    and all reviews.

    Returns a dict:
        {
            "positive": [(word, score), ...],
            "negative": [(word, score), ...],
            "overall":  [(word, score), ...],
        }
    """
    if "sentiment" not in df.columns:
        raise ValueError("DataFrame must contain a 'sentiment' column before running keyword analysis.")

    positive_texts = df.loc[df["sentiment"] == "Positive", text_column].tolist()
    negative_texts = df.loc[df["sentiment"] == "Negative", text_column].tolist()
    all_texts = df[text_column].tolist()

    return {
        "positive": _top_keywords_for_texts(positive_texts, top_n=top_n),
        "negative": _top_keywords_for_texts(negative_texts, top_n=top_n),
        "overall": _top_keywords_for_texts(all_texts, top_n=top_n),
    }


def keywords_to_dataframe(keyword_list: list) -> pd.DataFrame:
    """Converts a list of (word, score) tuples into a tidy DataFrame for display/plotting."""
    if not keyword_list:
        return pd.DataFrame(columns=["keyword", "score"])
    return pd.DataFrame(keyword_list, columns=["keyword", "score"])
