"""
Customer Feedback NLP Dashboard
Interactive Streamlit app for analyzing customer reviews
"""

import os
import sys
import pandas as pd
import streamlit as st
from pathlib import Path
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

# Setup paths
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from nlp import (
    clean_text,
    get_sentiment_score_and_label,
    get_topic_for_review,
    extract_keywords_and_entities,
    get_bert_sentiment,
)

# Data paths
PROCESSED_FILE = PROJECT_DIR / "data" / "processed" / "reviews_with_topics.csv"

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Customer Feedback NLP Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Customer Feedback Analysis Dashboard")
st.markdown("*Powered by VADER, BERTopic, and spaCy for intelligent review analysis*")
st.divider()

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_data():
    """Load processed reviews from CSV"""
    try:
        df = pd.read_csv(PROCESSED_FILE)
        
        # Ensure required columns exist
        if "topic_id" not in df.columns:
            df["topic_id"] = -1
        
        # Convert unixReviewTime to datetime
        if "unixReviewTime" in df.columns and "date" not in df.columns:
            df["date"] = pd.to_datetime(df["unixReviewTime"], unit="s")
        
        return df
    except FileNotFoundError:
        st.error(f"❌ Data file not found: {PROCESSED_FILE}")
        st.info("Please run the preprocessing pipeline first:\n`python scripts/pipeline_offline.py`")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None


df = load_data()

if df is None:
    st.stop()

# ============================================================================
# SIDEBAR FILTERS
# ============================================================================
st.sidebar.header("🔍 Filters")

# Sentiment filter
sentiment_options = ["All"] + sorted(df[df["sentiment_label"].notna()]["sentiment_label"].unique().tolist())
selected_sentiment = st.sidebar.selectbox("Sentiment", sentiment_options)

# Rating range
if "overall" in df.columns:
    rating_min, rating_max = float(df["overall"].min()), float(df["overall"].max())
    selected_rating_range = st.sidebar.slider(
        "Rating Range",
        rating_min, rating_max,
        (rating_min, rating_max),
        step=0.5
    )
else:
    selected_rating_range = (1, 5)

# Date range
if "date" in df.columns:
    min_date, max_date = df["date"].min(), df["date"].max()
    selected_date_range = st.sidebar.date_input(
        "Date Range",
        (min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
else:
    selected_date_range = None

# Topic filter (if available)
if "topic_id" in df.columns:
    topic_options = ["All"] + sorted([t for t in df["topic_id"].unique() if t >= 0])
    selected_topic = st.sidebar.selectbox("Topic ID", topic_options)
else:
    selected_topic = "All"

# ============================================================================
# APPLY FILTERS
# ============================================================================
filtered = df.copy()

if selected_sentiment != "All":
    filtered = filtered[filtered["sentiment_label"] == selected_sentiment]

if "overall" in filtered.columns:
    filtered = filtered[
        (filtered["overall"] >= selected_rating_range[0]) &
        (filtered["overall"] <= selected_rating_range[1])
    ]

if selected_date_range and "date" in filtered.columns:
    start_date, end_date = selected_date_range
    filtered = filtered[
        (filtered["date"].dt.date >= start_date) &
        (filtered["date"].dt.date <= end_date)
    ]

if selected_topic != "All" and "topic_id" in filtered.columns:
    filtered = filtered[filtered["topic_id"] == selected_topic]

# ============================================================================
# KEY METRICS
# ============================================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Total Reviews", f"{len(filtered):,}")

with col2:
    if len(filtered) > 0:
        pos_pct = (filtered["sentiment_label"] == "positive").mean() * 100
        st.metric("😊 % Positive", f"{pos_pct:.1f}%")
    else:
        st.metric("😊 % Positive", "0%")

with col3:
    if len(filtered) > 0:
        neg_pct = (filtered["sentiment_label"] == "negative").mean() * 100
        st.metric("😞 % Negative", f"{neg_pct:.1f}%")
    else:
        st.metric("😞 % Negative", "0%")

with col4:
    if "overall" in filtered.columns and len(filtered) > 0:
        avg_rating = filtered["overall"].mean()
        st.metric("⭐ Avg Rating", f"{avg_rating:.2f}")
    else:
        st.metric("⭐ Avg Rating", "N/A")

st.divider()

# ============================================================================
# SENTIMENT DISTRIBUTION
# ============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Distribution")
    if len(filtered) > 0:
        sentiment_counts = filtered["sentiment_label"].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to display")

with col2:
    st.subheader("Sentiment Over Time")
    if "date" in filtered.columns and len(filtered) > 0:
        try:
            daily = (
                filtered
                .groupby(filtered["date"].dt.to_period("M"))["sentiment_score"]
                .agg(["mean", "count"])
                .reset_index()
            )
            daily.columns = ["Month", "Avg Sentiment Score", "Review Count"]
            daily["Month"] = daily["Month"].astype(str)
            
            fig = px.line(
                daily,
                x="Month",
                y="Avg Sentiment Score",
                markers=True,
                title="Average Sentiment Score by Month"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate time series: {e}")
    else:
        st.info("No date data available")

st.divider()

# ============================================================================
# TOPIC ANALYSIS
# ============================================================================
if "topic_id" in filtered.columns:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Topics by Volume")
        topic_counts = filtered[filtered["topic_id"] >= 0]["topic_id"].value_counts().head(10)
        if len(topic_counts) > 0:
            fig = px.bar(x=topic_counts.values, y=topic_counts.index.astype(str))
            fig.update_layout(xaxis_title="Number of Reviews", yaxis_title="Topic ID")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No topic data available")
    
    with col2:
        st.subheader("Topic Sentiment Analysis")
        if len(filtered) > 0:
            topic_sentiment = filtered.groupby("topic_id")["sentiment_label"].value_counts().unstack(fill_value=0)
            topic_sentiment = topic_sentiment[[col for col in ["positive", "negative", "neutral"] if col in topic_sentiment.columns]]
            fig = px.bar(topic_sentiment, barmode="stack", title="Sentiment by Topic")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data to display")

st.divider()

# ============================================================================
# TOP COMPLAINT KEYWORDS
# ============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Complaint Keywords (Negative)")
    neg_reviews = filtered[filtered["sentiment_label"] == "negative"]
    
    if "clean_text" in neg_reviews.columns and len(neg_reviews) > 0:
        all_words = " ".join(neg_reviews["clean_text"].astype(str)).split()
        word_counts = Counter(all_words)
        common_words = word_counts.most_common(15)
        
        if common_words:
            top_words_df = pd.DataFrame(common_words, columns=["word", "count"])
            fig = px.bar(top_words_df, x="count", y="word", orientation="h")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No negative reviews with text")
    else:
        st.info("No negative reviews to analyze")

with col2:
    st.subheader("Top Positive Keywords")
    pos_reviews = filtered[filtered["sentiment_label"] == "positive"]
    
    if "clean_text" in pos_reviews.columns and len(pos_reviews) > 0:
        all_words = " ".join(pos_reviews["clean_text"].astype(str)).split()
        word_counts = Counter(all_words)
        common_words = word_counts.most_common(15)
        
        if common_words:
            top_words_df = pd.DataFrame(common_words, columns=["word", "count"])
            fig = px.bar(top_words_df, x="count", y="word", orientation="h")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No positive reviews with text")
    else:
        st.info("No positive reviews to analyze")

st.divider()

# ============================================================================
# REVIEW EXPLORER
# ============================================================================
st.subheader("📋 Review Explorer")

search_text = st.text_input("🔎 Search in reviews")
sort_by = st.selectbox("Sort by", ["Recent", "Rating (High to Low)", "Rating (Low to High)", "Sentiment Score"])

explore = filtered.copy()

if search_text:
    explore = explore[explore["reviewText"].astype(str).str.contains(search_text, case=False, na=False)]

# Apply sorting
if "date" in explore.columns:
    if sort_by == "Recent":
        explore = explore.sort_values("date", ascending=False)
    elif sort_by == "Rating (High to Low)" and "overall" in explore.columns:
        explore = explore.sort_values("overall", ascending=False)
    elif sort_by == "Rating (Low to High)" and "overall" in explore.columns:
        explore = explore.sort_values("overall", ascending=True)
    elif sort_by == "Sentiment Score":
        explore = explore.sort_values("sentiment_score", ascending=False)

st.write(f"**Showing {len(explore)} reviews** (after filters and search)")

# Display reviews
display_cols = ["reviewText", "overall", "sentiment_label", "sentiment_score"]
if "topic_id" in explore.columns:
    display_cols.append("topic_id")
if "date" in explore.columns:
    display_cols.append("date")

st.dataframe(
    explore[display_cols].head(100),
    use_container_width=True,
    height=400
)

st.divider()

# ============================================================================
# SINGLE REVIEW ANALYZER
# ============================================================================
st.subheader("🔬 Analyze a New Review")

new_review = st.text_area("Paste a customer review here:", height=100)

col1, col2 = st.columns([1, 4])

with col1:
    sentiment_model = st.radio("Sentiment Model:", ["VADER", "BERT"])
    analyze_btn = st.button("🚀 Analyze")

if analyze_btn and new_review.strip():
    with st.spinner("Analyzing..."):
        clean = clean_text(new_review)
        
        st.write("**Cleaned Text:**")
        st.info(clean)
        
        # Sentiment analysis
        col1, col2, col3 = st.columns(3)
        
        if sentiment_model == "VADER":
            vader_result = get_sentiment_score_and_label(clean)
            with col1:
                st.metric("Sentiment Label", vader_result["label"].upper())
            with col2:
                st.metric("Sentiment Score", f"{vader_result['score']:.3f}")
        else:
            try:
                bert_result = get_bert_sentiment(clean)
                with col1:
                    st.metric("Sentiment Label", bert_result["label"])
                with col2:
                    st.metric("Confidence Score", f"{bert_result['score']:.3f}")
            except Exception as e:
                st.error(f"BERT model not available: {e}")
        
        # Topic analysis
        topic_id, prob, words = get_topic_for_review(clean)
        with col3:
            st.metric("Topic ID", topic_id)
        
        st.write("**Topic Keywords:**", ", ".join(words))
        
        # Entity extraction
        try:
            entities = extract_keywords_and_entities(clean)
            if entities["keywords"]:
                st.write("**Extracted Keywords:**", ", ".join(entities["keywords"][:5]))
            if entities["entities"]:
                st.write("**Named Entities:**")
                for ent_text, ent_label in entities["entities"][:5]:
                    st.write(f"  • **{ent_text}** ({ent_label})")
        except Exception as e:
            st.warning(f"Could not extract entities: {e}")

elif analyze_btn and not new_review.strip():
    st.warning("⚠️ Please enter a review to analyze")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
---
**Customer Feedback NLP Dashboard** | Built with Streamlit, VADER, BERTopic & spaCy
- 📊 Analyze customer sentiment and identify complaint themes
- 🤖 Powered by state-of-the-art NLP models
- 📁 Data: `data/processed/reviews_with_topics.csv`
""")
