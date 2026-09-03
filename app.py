"""
app.py

E-Commerce Product Review Intelligence - Streamlit Dashboard

Run with:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import pandas as pd
import plotly.express as px

from pipeline import run_pipeline
from src.database import load_reviews_from_db, load_product_summary_from_db
from src.keyword_analysis import get_keyword_analysis, keywords_to_dataframe
from src.insights import generate_business_insights

DB_PATH = "database/reviews.db"

st.set_page_config(
    page_title="E-Commerce Product Review Intelligence",
    page_icon="🛍️",
    layout="wide",
)


# ----------------------------------------------------------------------
# Data loading (cached so the dashboard stays fast when filters change)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DB_PATH):
        with st.spinner("Preparing the demo review database..."):
            run_pipeline()
    reviews_df = load_reviews_from_db(DB_PATH)
    summary_df = load_product_summary_from_db(DB_PATH)
    return reviews_df, summary_df


def main():
    st.title("🛍️ E-Commerce Product Review Intelligence")
    st.caption("Turning customer reviews into actionable business insights")

    reviews_df, summary_df = load_data()

    # --- Error handling: missing / empty database ---
    if reviews_df.empty:
        st.warning(
            "No processed review data was found. "
            "Please run `python pipeline.py` first to generate the database, "
            "then reload this dashboard."
        )
        st.stop()

    required_columns = {"product_id", "product_name", "rating", "sentiment", "cleaned_text"}
    missing_columns = required_columns - set(reviews_df.columns)
    if missing_columns:
        st.error(f"The reviews database is missing expected columns: {', '.join(missing_columns)}")
        st.stop()

    # ------------------------------------------------------------------
    # Sidebar filters
    # ------------------------------------------------------------------
    st.sidebar.header("Filters")

    product_options = ["All Products"] + sorted(reviews_df["product_name"].dropna().unique().tolist())
    selected_product = st.sidebar.selectbox("Product", product_options)

    rating_options = sorted(reviews_df["rating"].dropna().unique().tolist())
    selected_ratings = st.sidebar.multiselect(
        "Rating", options=rating_options, default=rating_options, format_func=lambda r: f"{int(r)} star"
    )

    sentiment_options = ["All", "Positive", "Neutral", "Negative"]
    selected_sentiment = st.sidebar.radio("Sentiment", sentiment_options)

    # --- Apply filters ---
    filtered_df = reviews_df.copy()

    if selected_product != "All Products":
        filtered_df = filtered_df[filtered_df["product_name"] == selected_product]

    if selected_ratings:
        filtered_df = filtered_df[filtered_df["rating"].isin(selected_ratings)]
    else:
        filtered_df = filtered_df.iloc[0:0]

    if selected_sentiment != "All":
        filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]

    if filtered_df.empty:
        st.warning("No reviews match the selected filters. Try adjusting the filters in the sidebar.")
        st.stop()

    # ------------------------------------------------------------------
    # KPI section
    # ------------------------------------------------------------------
    total_reviews = len(filtered_df)
    average_rating = round(filtered_df["rating"].mean(), 2)
    positive_pct = round((filtered_df["sentiment"] == "Positive").mean() * 100, 1)
    negative_pct = round((filtered_df["sentiment"] == "Negative").mean() * 100, 1)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Reviews", f"{total_reviews:,}")
    kpi2.metric("Average Rating", f"{average_rating} / 5")
    kpi3.metric("Positive Reviews %", f"{positive_pct}%")
    kpi4.metric("Negative Reviews %", f"{negative_pct}%")

    st.divider()

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    chart_row1_col1, chart_row1_col2 = st.columns(2)

    with chart_row1_col1:
        st.subheader("Rating Distribution")
        rating_counts = (
            filtered_df["rating"].value_counts().sort_index().reset_index()
        )
        rating_counts.columns = ["rating", "count"]
        rating_counts["rating_label"] = rating_counts["rating"].apply(lambda r: f"{int(r)} star")
        fig_rating = px.bar(
            rating_counts, x="rating_label", y="count",
            labels={"rating_label": "Rating", "count": "Number of Reviews"},
            color="rating_label",
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig_rating.update_layout(showlegend=False)
        st.plotly_chart(fig_rating, width='stretch')

    with chart_row1_col2:
        st.subheader("Sentiment Distribution")
        sentiment_counts = filtered_df["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["sentiment", "count"]
        fig_sentiment = px.pie(
            sentiment_counts, names="sentiment", values="count", hole=0.45,
            color="sentiment",
            color_discrete_map={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"},
        )
        st.plotly_chart(fig_sentiment, width='stretch')

    chart_row2_col1, chart_row2_col2 = st.columns(2)

    with chart_row2_col1:
        st.subheader("Sentiment by Rating")
        sentiment_by_rating = (
            filtered_df.groupby(["rating", "sentiment"]).size().reset_index(name="count")
        )
        sentiment_by_rating["rating_label"] = sentiment_by_rating["rating"].apply(lambda r: f"{int(r)} star")
        fig_sent_rating = px.bar(
            sentiment_by_rating, x="rating_label", y="count", color="sentiment",
            barmode="stack",
            labels={"rating_label": "Rating", "count": "Number of Reviews"},
            color_discrete_map={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"},
        )
        st.plotly_chart(fig_sent_rating, width='stretch')

    with chart_row2_col2:
        st.subheader("Product Comparison")
        if not summary_df.empty:
            metric_choice = st.selectbox(
                "Compare products by", ["average_rating", "review_count", "positive_percentage", "negative_percentage"],
                format_func=lambda m: {
                    "average_rating": "Average Rating",
                    "review_count": "Review Count",
                    "positive_percentage": "Positive %",
                    "negative_percentage": "Negative %",
                }[m],
            )
            product_chart_df = summary_df.sort_values(metric_choice, ascending=False).head(15)
            fig_product = px.bar(
                product_chart_df,
                x=metric_choice, y="product_name", orientation="h",
                labels={"product_name": "Product", metric_choice: metric_choice.replace("_", " ").title()},
                color=metric_choice,
                color_continuous_scale="Blues",
            )
            fig_product.update_layout(
                height=500,
                yaxis={"categoryorder": "total ascending"},
                margin={"l": 10, "r": 10, "t": 20, "b": 20},
            )
            st.plotly_chart(fig_product, width='stretch')
        else:
            st.info("Product summary data is not available.")

    # --- Keyword charts ---
    st.subheader("Keyword Analysis")
    kw_col1, kw_col2 = st.columns(2)

    try:
        keyword_results = get_keyword_analysis(filtered_df)
    except ValueError:
        keyword_results = {"positive": [], "negative": [], "overall": []}

    with kw_col1:
        st.markdown("**Top Positive Keywords**")
        pos_kw_df = keywords_to_dataframe(keyword_results["positive"])
        if not pos_kw_df.empty:
            fig_pos_kw = px.bar(
                pos_kw_df.sort_values("score"), x="score", y="keyword", orientation="h",
                color_discrete_sequence=["#2ecc71"],
            )
            fig_pos_kw.update_layout(showlegend=False, yaxis_title="", xaxis_title="TF-IDF score")
            st.plotly_chart(fig_pos_kw, width='stretch')
        else:
            st.info("Not enough positive reviews to extract keywords.")

    with kw_col2:
        st.markdown("**Top Negative Keywords**")
        neg_kw_df = keywords_to_dataframe(keyword_results["negative"])
        if not neg_kw_df.empty:
            fig_neg_kw = px.bar(
                neg_kw_df.sort_values("score"), x="score", y="keyword", orientation="h",
                color_discrete_sequence=["#e74c3c"],
            )
            fig_neg_kw.update_layout(showlegend=False, yaxis_title="", xaxis_title="TF-IDF score")
            st.plotly_chart(fig_neg_kw, width='stretch')
        else:
            st.info("Not enough negative reviews to extract keywords.")

    st.divider()

    # ------------------------------------------------------------------
    # Review Explorer
    # ------------------------------------------------------------------
    st.subheader("Review Explorer")

    search_term = st.text_input("Search reviews by keyword", placeholder="e.g. battery")

    explorer_df = filtered_df.copy()
    if search_term:
        mask = explorer_df["cleaned_text"].str.contains(search_term.lower(), na=False)
        explorer_df = explorer_df[mask]

    if explorer_df.empty:
        st.warning(f"No reviews found matching '{search_term}'." if search_term else "No reviews to display.")
    else:
        display_cols = ["product_name", "review_title", "review_text", "rating", "sentiment"]
        display_cols = [c for c in display_cols if c in explorer_df.columns]
        st.dataframe(
            explorer_df[display_cols].rename(
                columns={
                    "product_name": "Product",
                    "review_title": "Title",
                    "review_text": "Review",
                    "rating": "Rating",
                    "sentiment": "Sentiment",
                }
            ),
            width='stretch',
            height=350,
        )
        st.caption(f"Showing {len(explorer_df):,} of {len(filtered_df):,} filtered reviews.")

    st.divider()

    # ------------------------------------------------------------------
    # Business Insights
    # ------------------------------------------------------------------
    st.subheader("Key Business Insights")

    try:
        insights = generate_business_insights(filtered_df, keyword_results)
        for insight in insights:
            st.markdown(f"**{insight['category']}**")
            st.write(insight["text"])
    except Exception as exc:
        st.error(f"Could not generate insights: {exc}")


if __name__ == "__main__":
    main()
