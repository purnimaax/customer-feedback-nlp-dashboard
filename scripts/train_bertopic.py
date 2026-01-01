"""
Train BERTopic model on customer reviews.
Usage: python scripts/train_bertopic.py
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

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Define paths
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "reviews.csv"
PROCESSED_FILE = PROJECT_DIR / "data" / "processed" / "reviews_with_topics.csv"
MODEL_OUTPUT_DIR = PROJECT_DIR / "models" / "bertopic_model"


def train_bertopic_model(
    input_file: str,
    output_dir: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    sample_size: int = None,
    min_topic_size: int = 10,
):
    """
    Train a BERTopic model on reviews.
    
    Args:
        input_file: Path to CSV with reviews
        output_dir: Directory to save the trained model
        embedding_model: Sentence transformer model to use
        sample_size: Sample size for training (None = use all)
        min_topic_size: Minimum number of documents per topic
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
    
    # Use clean_text if available, otherwise use reviewText
    if "clean_text" in df.columns:
        documents = df["clean_text"].dropna().tolist()
        logger.info("Using 'clean_text' column")
    elif "reviewText" in df.columns:
        documents = df["reviewText"].dropna().astype(str).tolist()
        logger.info("Using 'reviewText' column")
    else:
        logger.error("No 'reviewText' or 'clean_text' column found")
        return
    
    if len(documents) == 0:
        logger.error("No documents to process")
        return
    
    # Sample if requested
    if sample_size and len(documents) > sample_size:
        logger.info(f"Sampling {sample_size} documents for training...")
        import random
        random.seed(42)
        documents = random.sample(documents, sample_size)
    
    logger.info(f"Training BERTopic on {len(documents)} documents...")
    logger.info(f"  - Embedding model: {embedding_model}")
    logger.info(f"  - Min topic size: {min_topic_size}")
    
    try:
        # Load embedding model
        logger.info("Loading sentence transformer model...")
        embedding_model_obj = SentenceTransformer(embedding_model)
        
        # Create and train BERTopic
        logger.info("Initializing and training BERTopic...")
        topic_model = BERTopic(
            embedding_model=embedding_model_obj,
            min_topic_size=min_topic_size,
            verbose=True,
            calculate_probabilities=True,
        )
        
        # Fit the model
        topics, probs = topic_model.fit_transform(documents)
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model to {output_dir}...")
        topic_model.save(output_dir)
        
        logger.info("✓ Model training complete!")
        
        # Print statistics
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total documents: {len(documents)}")
        print(f"Number of topics: {len(set(topics)) - 1 if -1 in topics else len(set(topics))}")
        print(f"Outliers: {sum(1 for t in topics if t == -1)}")
        print("\nTop 5 topics:")
        for topic_id in sorted(set(topics))[:5]:
            if topic_id >= 0:
                words = topic_model.get_topic(topic_id)
                top_words = ", ".join([word for word, _ in words[:5]])
                count = sum(1 for t in topics if t == topic_id)
                print(f"  Topic {topic_id}: {count} docs | Words: {top_words}")
        print("="*60 + "\n")
        
        return topic_model
    
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train BERTopic model on reviews")
    parser.add_argument(
        "--input",
        type=str,
        default=str(RAW_DATA_PATH),
        help=f"Input CSV path (default: {RAW_DATA_PATH})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(MODEL_OUTPUT_DIR),
        help=f"Output model directory (default: {MODEL_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model to use"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for training (None = use all)"
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=10,
        help="Minimum documents per topic"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting BERTopic training...")
    train_bertopic_model(
        input_file=args.input,
        output_dir=args.output,
        embedding_model=args.embedding_model,
        sample_size=args.sample_size,
        min_topic_size=args.min_topic_size,
    )
