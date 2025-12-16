import os
import sys
import pandas as pd
import streamlit as st

# Base directory = repo root (works on Streamlit Cloud and locally)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(PROJECT_DIR, "app")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PROCESSED_FILE = os.path.join(PROCESSED_DIR, "reviews_with_topics.csv")

if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)

from preprocessing import clean_text
from sentiment import get_sentiment_score_and_label
from topic_modeling import get_topic_for_review


@st.cache_data
def load_data():
    df = pd.read_csv(PROCESSED_FILE)
    if "topic_id" not in df.columns:
        df["topic_id"] = -1
    if "unixReviewTime" in df.columns and "date" not in df.columns:
        df["date"] = pd.to_datetime(df["unixReviewTime"], unit="s")
    return df

df = load_data()

st.set_page_config(page_title="Customer Feedback NLP Dashboard", layout="wide")
st.title("Customer Feedback Analysis Dashboard")
st.caption("VADER sentiment + BERTopic topics over customer reviews")

st.sidebar.header("Filters")

sentiment_options = ["all"] + sorted(df["sentiment_label"].dropna().unique().tolist())
sentiment_filter = st.sidebar.selectbox("Sentiment", sentiment_options)

rating_min, rating_max = float(df["overall"].min()), float(df["overall"].max())
rating_range = st.sidebar.slider("Rating range", rating_min, rating_max,
                                 (rating_min, rating_max), step=0.5)

if "date" in df.columns:
    min_date, max_date = df["date"].min(), df["date"].max()
    date_range = st.sidebar.date_input("Date range", (min_date, max_date))
else:
    date_range = None

filtered = df.copy()
if sentiment_filter != "all":
    filtered = filtered[filtered["sentiment_label"] == sentiment_filter]

filtered = filtered[
    (filtered["overall"] >= rating_range[0]) &
    (filtered["overall"] <= rating_range[1])
]

if date_range and "date" in filtered.columns:
    start_date, end_date = date_range
    filtered = filtered[
        (filtered["date"].dt.date >= start_date) &
        (filtered["date"].dt.date <= end_date)
    ]

col1, col2, col3 = st.columns(3)
col1.metric("Total reviews", len(filtered))
pos_pct = (filtered["sentiment_label"] == "positive").mean() * 100 if len(filtered) else 0
neg_pct = (filtered["sentiment_label"] == "negative").mean() * 100 if len(filtered) else 0
col2.metric("% Positive", f"{pos_pct:.1f}%")
col3.metric("% Negative", f"{neg_pct:.1f}%")

st.markdown("---")
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Sentiment distribution")
    if len(filtered):
        st.bar_chart(filtered["sentiment_label"].value_counts())

    if "date" in filtered.columns:
        st.subheader("Sentiment over time")
        daily = (
            filtered
            .groupby(filtered["date"].dt.to_period("M"))["sentiment_score"]
            .mean()
            .to_timestamp()
            .reset_index()
        )
        daily = daily.rename(columns={"date": "Month",
                                      "sentiment_score": "Avg sentiment score"})
        st.line_chart(daily.set_index("Month"))

with right_col:
    st.subheader("Top complaint words (negative)")
    neg_reviews = filtered[filtered["sentiment_label"] == "negative"]
    if "clean_text" in neg_reviews.columns and len(neg_reviews):
        from collections import Counter
        all_words = " ".join(neg_reviews["clean_text"].astype(str)).split()
        counts = Counter(all_words)
        common = counts.most_common(20)
        if common:
            top_df = pd.DataFrame(common, columns=["word", "count"])
            st.dataframe(top_df)

st.markdown("---")
st.subheader("Review explorer")

search_text = st.text_input("Search in original reviews")

explore = filtered.copy()
if search_text:
    explore = explore[explore["reviewText"].str.contains(search_text,
                                                         case=False, na=False)]

st.write(f"Showing {len(explore)} reviews after filters/search.")
st.dataframe(
    explore[["reviewText", "overall", "sentiment_label",
             "sentiment_score", "topic_id"]].head(200)
)

st.markdown("---")
st.subheader("Analyze a new review")

new_review = st.text_area("Paste a customer review here")

if st.button("Analyze"):
    if not new_review.strip():
        st.warning("Please enter some text.")
    else:
        clean = clean_text(new_review)
        sent = get_sentiment_score_and_label(clean)
        topic_id, prob, words = get_topic_for_review(clean)

        st.write("**Cleaned text:**")
        st.write(clean)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Sentiment label", sent["label"])
        col_b.metric("Sentiment score", f"{sent['score']:.3f}")
        col_c.metric("Topic ID", topic_id)

        st.write("**Topic keywords:**", ", ".join(words))
