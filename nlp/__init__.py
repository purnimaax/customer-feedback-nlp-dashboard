# NLP module for customer feedback analysis
from .preprocessing import clean_text
from .sentiment import get_sentiment_score_and_label, get_all_sentiment_scores
from .topic_modeling import get_topic_for_review
from .nlp_spacy import extract_keywords_and_entities, extract_noun_phrases
from .bert_sentiment import get_bert_sentiment, get_vader_and_bert_comparison

__all__ = [
    "clean_text",
    "get_sentiment_score_and_label",
    "get_all_sentiment_scores",
    "get_topic_for_review",
    "extract_keywords_and_entities",
    "extract_noun_phrases",
    "get_bert_sentiment",
    "get_vader_and_bert_comparison",
]
