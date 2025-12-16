# app/topic_modeling.py

import os
from bertopic import BERTopic
from typing import List, Tuple

# Base dir = project root (.. from app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "bertopic_model")

try:
    TOPIC_MODEL = BERTopic.load(MODEL_PATH)
    print(f"Successfully loaded BERTopic model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading BERTopic model from {MODEL_PATH}: {e}")
    TOPIC_MODEL = None

def get_topic_for_review(text: str) -> Tuple[int, float, List[str]]:
    if not TOPIC_MODEL or not text:
        return -2, 0.0, ["model_error", "no_topic_found"]

    topics, probabilities = TOPIC_MODEL.transform([text])
    topic_id = topics[0]
    probability = float(probabilities[0]) if probabilities is not None else 0.0

    if topic_id == -1:
        representative_words = ["outlier", "no_clear_topic"]
    else:
        topic_info = TOPIC_MODEL.get_topic(topic_id)
        representative_words = [item[0] for item in topic_info[:5]] if topic_info else []

    return topic_id, probability, representative_words
