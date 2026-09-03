# E-Commerce Product Review Intelligence

Turning customer reviews into actionable business insights — fully local, no cloud, no API keys.

Repository: https://github.com/panubhav2001/ecommerce-review-intelligence

## Project Overview

Online retailers and product teams receive thousands of customer reviews, and manually reading through all of them to understand what customers like and dislike is impractical. Important signals — recurring complaints about delivery, praise for a particular feature, a product with unusually negative feedback — get lost in the noise.

This project builds a small, self-contained system that ingests raw product reviews and turns them into structured, queryable business intelligence: sentiment scores, keyword themes, product-level comparisons, and plain-language insights.

## Solution

The project is a local Python pipeline plus a Streamlit dashboard:

1. **Load** review data from the Amazon Fine Foods Reviews text file (the loader also supports CSV files with common review column names).
2. **Clean** the text and the underlying data (missing values, duplicates, normalization).
3. **Process at scale with PySpark**, demonstrating how the same cleaning and aggregation logic would scale to much larger datasets.
4. **Score sentiment** for every review using VADER, a local rule-based sentiment model (no API key, no internet required).
5. **Extract keywords** from positive and negative reviews using TF-IDF, surfacing what customers praise and complain about.
6. **Store** everything in a local SQLite database.
7. **Explore** the results interactively in a Streamlit dashboard with filters, charts, a searchable review explorer, and automatically generated business insights.

## Architecture

```mermaid
flowchart TD
    A[Dataset: data/foods.txt] --> B[PySpark: local mode]
    B --> C[Data Cleaning]
    C --> D[Sentiment Analysis - VADER]
    D --> E[Keyword Analysis - TF-IDF]
    E --> F[(SQLite: database/reviews.db)]
    F --> G[Streamlit Dashboard]
```

## Technologies Used

| Technology | Why it's used |
|---|---|
| **Python 3.11+** | Core language for the entire project. |
| **Pandas** | Flexible, fast data loading, column mapping, and text cleaning. |
| **PySpark (local mode)** | Demonstrates DataFrame-based cleaning, null handling, de-duplication, and aggregation that scales beyond what pandas alone can comfortably handle — while still running entirely on a single machine (`local[*]`). |
| **VADER Sentiment** | A lightweight, rule-based sentiment analyzer tuned for short, informal text like reviews. Runs fully offline, no model download or API key needed. |
| **Scikit-learn (TF-IDF)** | Identifies the most distinctive words in positive vs. negative reviews without needing a large language model. |
| **SQLite** | A zero-configuration, file-based database — perfect for a fully local project with no external database server. |
| **Streamlit** | Turns the analysis into an interactive dashboard with minimal boilerplate. |
| **Plotly** | Interactive charts (bar, pie/donut) for the dashboard. |

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate it (Windows)

```bash
.venv\Scripts\activate
```

On macOS/Linux, use `source .venv/bin/activate` instead.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PySpark requires a Java runtime (Java 8, 11, 17, or 21) to be installed and on your `PATH`. Most systems already have this; if not, install a JDK (e.g. from [Adoptium](https://adoptium.net/)) before running the pipeline.

### 4. Add the dataset

Place the Amazon Fine Foods Reviews dataset at `data/foods.txt`. The dataset is intentionally not included in GitHub because it is approximately 354 MB. The expected format is the standard block-based format with fields such as `ProductId`, `ProfileName`, `Score`, `Time`, `Summary`, and `Text`.

Alternatively, update `DATA_PATH` in `pipeline.py` to point to a compatible CSV file.

### 5. Run the pipeline

```bash
python pipeline.py
```

This loads, cleans, and analyzes the review data, then saves the results into `database/reviews.db`.

### 6. Run the dashboard

```bash
streamlit run app.py
```

This opens the dashboard in your browser (usually at `http://localhost:8501`).

## Dataset

The default pipeline input is `data/foods.txt`, using the Amazon Fine Foods Reviews format. For a CSV input, update `DATA_PATH` in `pipeline.py` and place the file in `data/`. The loader flexibly maps several common column naming conventions to a standard internal schema:

| Standard column | Accepted variants (case-insensitive) |
|---|---|
| `product_id` | `product_id`, `ProductId`, `asin`, `id` |
| `product_name` | `product_name`, `ProductName`, `product`, `name` |
| `rating` | `rating`, `Score`, `stars`, `star_rating`, `overall` |
| `review_title` | `review_title`, `Summary`, `title`, `headline` |
| `review_text` | `review_text`, `Text`, `review`, `body`, `content` |
| `review_date` | `review_date`, `Time`, `date`, `timestamp` |

> **Note:** Input datasets and generated database files are excluded by `.gitignore`. Keep local datasets out of source control, especially large review exports.

## Features

- **KPI overview:** total reviews, average rating, % positive, % negative.
- **Rating distribution** bar chart.
- **Sentiment distribution** donut chart.
- **Sentiment by rating** stacked bar chart.
- **Product comparison** horizontal bar chart showing the top 15 products by average rating, review count, positive %, or negative %.
- **Top positive / negative keyword** charts (TF-IDF).
- **Sidebar filters:** product, rating, sentiment — all charts and tables update live.
- **Review Explorer:** full-text keyword search across reviews, with sentiment/rating filters applied.
- **Key Business Insights:** dynamically generated, data-driven summary statements (not hard-coded).
- Graceful handling of missing data, empty results, and database errors — no raw Python tracebacks shown to the user.

## Business Value

Review intelligence like this helps businesses:

- **Product improvement** — see which features drive complaints vs. praise, and prioritize fixes.
- **Customer experience** — spot recurring friction points (e.g., delivery, packaging) before they escalate.
- **Quality monitoring** — track sentiment and rating trends per product over time.
- **Identifying recurring complaints** — TF-IDF keyword extraction surfaces themes without reading every review.
- **Product comparison** — quickly see which products are outperforming or underperforming peers.
- **Marketing insights** — understand which product qualities resonate most with customers, to highlight in messaging.

## Limitations

- **VADER is a rule-based sentiment model.** It works well for short, informal review text but can misjudge sarcasm, nuanced context, or domain-specific jargon.
- **TF-IDF keyword analysis does not fully understand context** — it surfaces statistically important words, not verified causal relationships between a word and customer satisfaction.
- **Dataset quality affects insights.** Garbage in, garbage out — mislabeled ratings or spam reviews will skew the analysis.
- **Amazon scraping is not part of this project.** You must supply the Amazon Fine Foods Reviews text file or a compatible CSV dataset.
- **The local SQLite database is intended for a project/demo environment**, not high-concurrency production use.

## Future Improvements

The following are intentionally *not* implemented in this version, but would be natural next steps:

- Aspect-based sentiment analysis (sentiment per product feature, not just per review).
- Real-time review ingestion (e.g., a live feed instead of a static CSV).
- Support for larger, distributed datasets across an actual Spark cluster.
- Product recommendation based on review sentiment.
- LLM-based review summarization.
- Cloud deployment of the dashboard.

## Project Structure

```text
ecommerce-review-intelligence/
│
├── data/
│   └── foods.txt              # local Amazon Fine Foods Reviews dataset (not tracked)
│
├── database/
│   └── reviews.db             # created by pipeline.py
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # CSV loading + flexible column mapping + sample data
│   ├── preprocessing.py       # text cleaning
│   ├── spark_processing.py    # PySpark cleaning + summary statistics
│   ├── sentiment.py           # VADER sentiment scoring
│   ├── keyword_analysis.py    # TF-IDF keyword extraction
│   ├── insights.py            # rule-based business insight generation
│   └── database.py            # SQLite read/write
│
├── app.py                     # Streamlit dashboard
├── pipeline.py                # end-to-end processing pipeline (run this first)
├── requirements.txt
├── README.md
└── .gitignore
```
