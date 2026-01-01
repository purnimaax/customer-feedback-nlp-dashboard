from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict, Any

# --- GLOBAL MODEL LOADING (Executes only once at app startup) ---
ANALYZER = SentimentIntensityAnalyzer()

def get_sentiment_score_and_label(text: str) -> Dict[str, Any]:
    """
    Analyzes the sentiment of a single text string using VADER.
    
    Args:
        text: The preprocessed input string.
    
    Returns:
        A dictionary containing the compound score and the derived label.
    """
    if not text:
        return {"score": 0.0, "label": "neutral"}
    
    # Get VADER's compound polarity score
    score: float = ANALYZER.polarity_scores(text)["compound"]
    
    # Determine the sentiment label based on the score
    if score >= 0.05:
        label = "positive"
    elif score <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    
    return {"score": score, "label": label}


def get_all_sentiment_scores(text: str) -> Dict[str, float]:
    """
    Get detailed VADER sentiment scores (neg, neu, pos, compound).
    
    Args:
        text: The input string.
    
    Returns:
        Dictionary with all VADER polarity scores.
    """
    if not text:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    
    return ANALYZER.polarity_scores(text)
