import os
from typing import List, Tuple

from bertopic import BERTopic

# Try to load a local model; if missing, fall back to None
MODEL_PATH = os.path.join("models", "bertopic_model")

TOPIC_MODEL = None
try:
    if os.path.exists(MODEL_PATH):
        TOPIC_MODEL = BERTopic.load(MODEL_PATH)
        print(f"Successfully loaded BERTopic model from {MODEL_PATH}")
    else:
        print(f"No local BERTopic model found at {MODEL_PATH}. Topics for new reviews will be approximate.")
except Exception as e:
    print(f"Error loading BERTopic model from {MODEL_PATH}: {e}")
    TOPIC_MODEL = None


def get_topic_for_review(text: str) -> Tuple[int, float, List[str]]:
    """
    Performs topic inference on a single, cleaned text string.
    If the BERTopic model is not available, returns a fallback topic based on simple keywords.
    """
    if not text:
        return -2, 0.0, ["no_text"]

    # If BERTopic is available, use it
    if TOPIC_MODEL is not None:
        topics, probabilities = TOPIC_MODEL.transform([text])
        topic_id = topics[0]
        probability = float(probabilities[0]) if probabilities is not None else 0.0

        if topic_id == -1:
            representative_words = ["outlier", "no_clear_topic"]
        else:
            topic_info = TOPIC_MODEL.get_topic(topic_id)
            representative_words = [item[0] for item in topic_info[:5]] if topic_info else []

        return topic_id, probability, representative_words

    # -------- Fallback logic when no model is present --------
    text_lower = text.lower()
    if any(k in text_lower for k in ["battery", "life", "charge"]):
        return 100, 0.5, ["battery", "life", "power"]
    if any(k in text_lower for k in ["price", "expensive", "cheap", "cost"]):
        return 101, 0.5, ["price", "cost", "value"]
    if any(k in text_lower for k in ["delivery", "shipping", "late"]):
        return 102, 0.5, ["delivery", "shipping", "time"]
    if any(k in text_lower for k in ["support", "service", "help"]):
        return 103, 0.5, ["support", "service", "help"]

    # Generic fallback
    words = [w for w in text_lower.split() if len(w) > 3][:5]
    return -1, 0.0, words or ["no_topic_found"]
