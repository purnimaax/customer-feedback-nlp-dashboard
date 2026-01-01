"""
Offline pipeline: Processes raw reviews and generates sentiment/topic labels.
Usage: python scripts/pipeline_offline.py
"""

import os
import sys
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from nlp import (
    clean_text,
    get_sentiment_score_and_label,
    get_topic_for_review,
    extract_noun_phrases
)

# Define data paths
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "reviews.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "reviews_with_topics.csv"

def process_reviews(input_file: str, output_file: str, sample: bool = False, sample_size: int = 5000):
    """
    Process raw reviews and add sentiment, topic, and keyword columns.
    
    Args:
        input_file: Path to raw reviews CSV
        output_file: Path to save processed CSV
        sample: Whether to sample the data (useful for testing)
        sample_size: Size of sample if sample=True
    """
    logger.info(f"Loading reviews from {input_file}...")
    
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} reviews")
    except FileNotFoundError:
        logger.error(f"File not found: {input_file}")
        return
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        return
    
    # Sample if requested
    if sample and len(df) > sample_size:
        logger.info(f"Sampling {sample_size} reviews for testing...")
        df = df.sample(n=sample_size, random_state=42)
    
    # Check if required columns exist
    if "reviewText" not in df.columns:
        logger.error("'reviewText' column not found in reviews CSV")
        return
    
    # Initialize new columns
    logger.info("Processing reviews...")
    df["clean_text"] = ""
    df["sentiment_score"] = 0.0
    df["sentiment_label"] = ""
    df["topic_id"] = -1
    df["topic_prob"] = 0.0
    df["topic_keywords"] = ""
    df["keywords"] = ""
    
    # Process each review
    total = len(df)
    for idx, row in df.iterrows():
        if (idx + 1) % 500 == 0:
            logger.info(f"  Processing {idx + 1}/{total} reviews...")
        
        try:
            # Clean text
            raw_text = str(row["reviewText"]) if pd.notna(row["reviewText"]) else ""
            clean = clean_text(raw_text)
            df.at[idx, "clean_text"] = clean
            
            # Get sentiment
            if clean:
                sentiment = get_sentiment_score_and_label(clean)
                df.at[idx, "sentiment_score"] = sentiment["score"]
                df.at[idx, "sentiment_label"] = sentiment["label"]
                
                # Get topic
                topic_id, topic_prob, topic_words = get_topic_for_review(clean)
                df.at[idx, "topic_id"] = topic_id
                df.at[idx, "topic_prob"] = topic_prob
                df.at[idx, "topic_keywords"] = ", ".join(topic_words)
                
                # Extract noun phrases as keywords
                keywords = extract_noun_phrases(clean)
                df.at[idx, "keywords"] = ", ".join(keywords[:5])  # Top 5 keywords
        
        except Exception as e:
            logger.warning(f"Error processing row {idx}: {e}")
            continue
    
    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    logger.info(f"Saving processed reviews to {output_file}...")
    df.to_csv(output_file, index=False)
    logger.info(f"✓ Processing complete! Saved {len(df)} reviews to {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total reviews: {len(df)}")
    print(f"\nSentiment Distribution:")
    print(df["sentiment_label"].value_counts())
    print(f"\nTop 10 Topics:")
    print(df["topic_id"].value_counts().head(10))
    print("="*60 + "\n")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process customer reviews offline")
    parser.add_argument(
        "--input",
        type=str,
        default=str(RAW_DATA_PATH),
        help=f"Input CSV path (default: {RAW_DATA_PATH})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROCESSED_FILE),
        help=f"Output CSV path (default: {PROCESSED_FILE})"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample the data for testing (default: process all)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Sample size if --sample is used (default: 5000)"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting offline review processing pipeline...")
    process_reviews(
        input_file=args.input,
        output_file=args.output,
        sample=args.sample,
        sample_size=args.sample_size
    )
