from typing import Dict, Any, Optional
import torch

# Try to load BERT sentiment model
try:
    from transformers import pipeline
    BERT_SENTIMENT_PIPELINE = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=0 if torch.cuda.is_available() else -1
    )
    BERT_AVAILABLE = True
except Exception as e:
    print(f"⚠ BERT sentiment model not available: {e}")
    BERT_AVAILABLE = False
    BERT_SENTIMENT_PIPELINE = None


def get_bert_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyzes sentiment using a BERT-based model.
    More accurate than VADER for complex sentences, but slower.
    
    Args:
        text: Input text string (will be truncated to 512 tokens)
    
    Returns:
        Dictionary with:
        - label: "5 stars", "4 stars", "3 stars", "2 stars", "1 star"
        - score: confidence score (0-1)
        - numeric_rating: Extracted numeric rating (1-5)
    """
    if not text:
        return {"label": "3 stars", "score": 0.0, "numeric_rating": 3}
    
    if not BERT_AVAILABLE or BERT_SENTIMENT_PIPELINE is None:
        return {"label": "unavailable", "score": 0.0, "numeric_rating": 0}
    
    try:
        # Truncate to avoid exceeding token limit
        truncated_text = text[:512]
        
        result = BERT_SENTIMENT_PIPELINE(truncated_text)[0]
        label = result["label"]  # e.g., "5 stars", "1 star"
        score = result["score"]
        
        # Convert label to numeric rating
        numeric_rating = int(label.split()[0])
        
        return {
            "label": label,
            "score": score,
            "numeric_rating": numeric_rating
        }
    except Exception as e:
        print(f"Error in BERT sentiment analysis: {e}")
        return {"label": "error", "score": 0.0, "numeric_rating": 0}


def get_vader_and_bert_comparison(text: str) -> Dict[str, Any]:
    """
    Compare VADER and BERT sentiment for a single review.
    Useful for understanding different model outputs.
    
    Args:
        text: Input text string
    
    Returns:
        Dictionary with both VADER and BERT results
    """
    from sentiment import get_sentiment_score_and_label
    
    vader_result = get_sentiment_score_and_label(text)
    bert_result = get_bert_sentiment(text)
    
    return {
        "vader": vader_result,
        "bert": bert_result
    }
