"""
insights.py

Generates simple, rule-based, data-driven business insights from the
fully processed reviews DataFrame. No LLM is used - every insight is
computed directly from the data with plain pandas logic and returned
as plain-language strings.
"""

import pandas as pd

# Words that, when they show up among the top keywords for a sentiment
# category, get called out by name as a "theme" in the generated
# insights (keeps the wording natural rather than just dumping a raw
# keyword list).
THEME_KEYWORDS = [
    "battery", "quality", "price", "delivery", "shipping", "packaging",
    "design", "performance", "value", "customer", "service", "size",
    "material", "comfort", "durability", "charging",
]


def _extract_theme_words(keyword_list, limit=3):
    """Pulls out up to `limit` recognizable theme words from a TF-IDF keyword list."""
    words = [word for word, _score in keyword_list]
    themes = [w for w in words if w in THEME_KEYWORDS]
    if not themes:
        themes = words[:limit]
    return themes[:limit]


def generate_business_insights(df: pd.DataFrame, keyword_results: dict) -> list:
    """
    Generates a list of short, human-readable business insight strings
    based on the processed reviews DataFrame and TF-IDF keyword results.

    Returns a list of dicts: [{"category": str, "text": str}, ...]
    """
    insights = []

    if df.empty:
        return [{"category": "Data", "text": "No reviews are available to generate insights yet."}]

    total_reviews = len(df)
    sentiment_counts = df["sentiment"].value_counts(normalize=True) * 100
    positive_pct = round(sentiment_counts.get("Positive", 0), 1)
    negative_pct = round(sentiment_counts.get("Negative", 0), 1)
    neutral_pct = round(sentiment_counts.get("Neutral", 0), 1)

    # --- Overall sentiment insight ---
    if positive_pct >= negative_pct and positive_pct >= neutral_pct:
        insights.append(
            {
                "category": "Customer Sentiment",
                "text": f"{positive_pct}% of the {total_reviews:,} reviews analyzed are positive, "
                        f"suggesting overall favorable customer sentiment.",
            }
        )
    elif negative_pct > positive_pct:
        insights.append(
            {
                "category": "Customer Sentiment",
                "text": f"{negative_pct}% of reviews are negative, which is higher than the "
                        f"{positive_pct}% that are positive - customer sentiment needs attention.",
            }
        )
    else:
        insights.append(
            {
                "category": "Customer Sentiment",
                "text": f"Sentiment is mixed: {positive_pct}% positive, {negative_pct}% negative, "
                        f"and {neutral_pct}% neutral.",
            }
        )

    # --- Positive themes (from TF-IDF) ---
    positive_keywords = keyword_results.get("positive", [])
    if positive_keywords:
        themes = _extract_theme_words(positive_keywords)
        if themes:
            insights.append(
                {
                    "category": "Product Quality",
                    "text": f"Customers frequently praise {', '.join(themes)} in positive reviews.",
                }
            )

    # --- Negative themes (from TF-IDF) ---
    negative_keywords = keyword_results.get("negative", [])
    if negative_keywords:
        themes = _extract_theme_words(negative_keywords)
        if themes:
            insights.append(
                {
                    "category": "Customer Complaints",
                    "text": f"{', '.join(themes).capitalize()}-related issues appear frequently in negative reviews.",
                }
            )

    # --- Rating vs. sentiment relationship ---
    if "rating" in df.columns:
        low_rating_df = df[df["rating"] < 3]
        if len(low_rating_df) > 0:
            low_rating_negative_pct = round((low_rating_df["sentiment"] == "Negative").mean() * 100, 1)
            insights.append(
                {
                    "category": "Rating Patterns",
                    "text": f"Products rated below 3 stars have a {low_rating_negative_pct}% negative "
                            f"sentiment rate, confirming ratings and review sentiment are closely aligned.",
                }
            )

    # --- Best and worst performing products ---
    if "product_name" in df.columns and df["product_id"].nunique() > 1:
        product_stats = (
            df.groupby(["product_id", "product_name"])
            .agg(review_count=("rating", "count"), average_rating=("rating", "mean"))
            .reset_index()
        )
        product_stats = product_stats[product_stats["review_count"] >= 3]

        if not product_stats.empty:
            best = product_stats.sort_values("average_rating", ascending=False).iloc[0]
            worst = product_stats.sort_values("average_rating", ascending=True).iloc[0]

            insights.append(
                {
                    "category": "Product Comparison",
                    "text": f"'{best['product_name']}' has the highest average rating "
                            f"({best['average_rating']:.1f}/5), while '{worst['product_name']}' has the "
                            f"lowest ({worst['average_rating']:.1f}/5) among products with 3+ reviews.",
                }
            )

    # --- Volume insight ---
    top_product = df["product_name"].value_counts().idxmax()
    top_product_count = df["product_name"].value_counts().max()
    insights.append(
        {
            "category": "Review Volume",
            "text": f"'{top_product}' received the most customer feedback, with {top_product_count} reviews.",
        }
    )

    return insights
