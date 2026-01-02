import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
import subprocess

# Set page config
st.set_page_config(page_title="Customer Feedback NLP Dashboard", layout="wide")

# ==================== SETUP & DOWNLOADS ====================

@st.cache_resource
def setup_nltk():
    """Download NLTK data if not present"""
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)

@st.cache_resource
def setup_spacy():
    """Download spacy model if not present"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        import spacy
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"])
        nlp = spacy.load("en_core_web_sm")
        return nlp

@st.cache_resource
def load_models():
    """Load all NLP models"""
    setup_nltk()
    nlp = setup_spacy()
    
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    return {
        'spacy': nlp,
        'vader': sentiment_analyzer
    }

@st.cache_data
def load_data():
    """Load processed reviews"""
    csv_path = "data/processed/reviews_with_topics.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        st.error("❌ Data file not found. Please run: `python scripts/pipeline_offline.py`")
        return None

# ==================== MAIN APP ====================

def main():
    st.title("📊 Customer Feedback NLP Dashboard")
    st.markdown("Analyze sentiment, topics, and trends from customer reviews")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    models = load_models()
    
    # ==================== METRICS ====================
    st.subheader("📈 Overall Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", len(df))
    
    with col2:
        positive_pct = (df['sentiment_label'] == 'positive').sum() / len(df) * 100
        st.metric("Positive %", f"{positive_pct:.1f}%")
    
    with col3:
        negative_pct = (df['sentiment_label'] == 'negative').sum() / len(df) * 100
        st.metric("Negative %", f"{negative_pct:.1f}%")
    
    with col4:
        neutral_pct = (df['sentiment_label'] == 'neutral').sum() / len(df) * 100
        st.metric("Neutral %", f"{neutral_pct:.1f}%")
    
    # ==================== FILTERS ====================
    st.divider()
    st.subheader("🔍 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_sentiment = st.multiselect(
            "Sentiment",
            options=df['sentiment_label'].unique(),
            default=df['sentiment_label'].unique()
        )
    
    with col2:
        rating_range = st.slider("Rating Range", 1, 5, (1, 5))
    
    with col3:
        search_text = st.text_input("Search reviews", "")
    
    # Apply filters
    filtered_df = df[df['sentiment_label'].isin(selected_sentiment)]
    filtered_df = filtered_df[(filtered_df['overall'] >= rating_range[0]) & (filtered_df['overall'] <= rating_range[1])]
    
    if search_text:
        filtered_df = filtered_df[filtered_df['reviewText'].str.contains(search_text, case=False, na=False)]
    
    st.info(f"Showing {len(filtered_df)} of {len(df)} reviews")
    
    # ==================== VISUALIZATIONS ====================
    st.divider()
    st.subheader("📊 Visualizations")
    
    col1, col2 = st.columns(2)
    
    # Sentiment Distribution
    with col1:
        sentiment_counts = filtered_df['sentiment_label'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color_discrete_map={'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Rating Distribution
    with col2:
        rating_counts = filtered_df['overall'].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            title="Rating Distribution",
            labels={'x': 'Rating', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Topics
    st.subheader("🏷️ Top Topics")
    if 'topic_keywords' in filtered_df.columns:
        topic_counts = filtered_df['topic_keywords'].value_counts().head(10)
        fig = px.bar(
            x=topic_counts.values,
            y=topic_counts.index,
            orientation='h',
            title="Top 10 Topics",
            labels={'x': 'Count', 'y': 'Topic'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== REVIEW EXPLORER ====================
    st.divider()
    st.subheader("📝 Review Explorer")
    
    if len(filtered_df) > 0:
        # Display reviews
        for idx, row in filtered_df.head(20).iterrows():
            sentiment_color = {'positive': '🟢', 'neutral': '🟡', 'negative': '🔴'}
            color = sentiment_color.get(row['sentiment_label'], '⚪')
            
            st.markdown(f"""
            {color} **Rating: {row['overall']}/5** | Sentiment: {row['sentiment_label'].upper()}
            
            {row['reviewText'][:500]}...
            """)
            st.divider()
    else:
        st.warning("No reviews match your filters")
    
    # ==================== EXPORT ====================
    st.divider()
    st.subheader("📥 Export Results")
    
    if len(filtered_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="customer_reviews_analysis.csv",
                mime="text/csv"
            )
        
        with col2:
            try:
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    filtered_df.to_excel(writer, index=False, sheet_name='Reviews')
                buffer.seek(0)
                st.download_button(
                    label="📊 Download as Excel",
                    data=buffer.getvalue(),
                    file_name="customer_reviews_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.warning(f"Excel export unavailable: {e}")

if __name__ == "__main__":
    main()
