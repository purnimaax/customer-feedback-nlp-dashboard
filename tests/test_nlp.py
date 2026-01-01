"""
Unit tests for NLP functions
Run with: pytest tests/test_nlp.py -v
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from nlp import (
    clean_text,
    get_sentiment_score_and_label,
    get_topic_for_review,
    extract_noun_phrases,
)


class TestPreprocessing:
    """Test text preprocessing"""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning"""
        text = "This is a GREAT product!!!"
        result = clean_text(text)
        assert isinstance(result, str)
        assert "great" in result.lower()
        assert "!" not in result
    
    def test_clean_text_stopwords(self):
        """Test stopword removal"""
        text = "the quick brown fox"
        result = clean_text(text)
        assert "the" not in result
        assert "quick" in result
    
    def test_clean_text_empty(self):
        """Test empty string handling"""
        assert clean_text("") == ""
        assert clean_text(None) == ""
    
    def test_clean_text_special_chars(self):
        """Test special character removal"""
        text = "Hello@World#123"
        result = clean_text(text)
        assert "@" not in result
        assert "#" not in result
        assert "123" not in result


class TestSentiment:
    """Test sentiment analysis"""
    
    def test_positive_sentiment(self):
        """Test positive sentiment detection"""
        text = "excellent amazing wonderful fantastic"
        result = get_sentiment_score_and_label(text)
        assert result["label"] == "positive"
        assert result["score"] > 0
    
    def test_negative_sentiment(self):
        """Test negative sentiment detection"""
        text = "terrible awful horrible bad"
        result = get_sentiment_score_and_label(text)
        assert result["label"] == "negative"
        assert result["score"] < 0
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment detection"""
        text = "the product is a product"
        result = get_sentiment_score_and_label(text)
        assert result["label"] == "neutral"
    
    def test_empty_sentiment(self):
        """Test empty text handling"""
        result = get_sentiment_score_and_label("")
        assert result["label"] == "neutral"
        assert result["score"] == 0.0


class TestTopic:
    """Test topic modeling"""
    
    def test_topic_for_review_empty(self):
        """Test empty text handling"""
        topic_id, prob, words = get_topic_for_review("")
        assert topic_id == -2
        assert prob == 0.0
        assert words == ["no_text"]
    
    def test_topic_for_review_battery(self):
        """Test battery keyword detection"""
        text = "battery life is terrible"
        topic_id, prob, words = get_topic_for_review(text)
        assert topic_id == 100  # Battery topic
        assert "battery" in words
    
    def test_topic_for_review_price(self):
        """Test price keyword detection"""
        text = "too expensive for the quality"
        topic_id, prob, words = get_topic_for_review(text)
        assert topic_id == 101  # Price topic
        assert "price" in words or "cost" in words
    
    def test_topic_returns_tuple(self):
        """Test return type"""
        result = get_topic_for_review("some text")
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], int)  # topic_id
        assert isinstance(result[1], float)  # probability
        assert isinstance(result[2], list)  # words


class TestKeywordExtraction:
    """Test keyword/entity extraction"""
    
    def test_extract_noun_phrases(self):
        """Test noun phrase extraction"""
        text = "The battery life is very good"
        result = extract_noun_phrases(text)
        assert isinstance(result, list)
        # spaCy might not be available, so just check type
    
    def test_extract_noun_phrases_empty(self):
        """Test empty text handling"""
        result = extract_noun_phrases("")
        assert result == []


class TestIntegration:
    """Integration tests"""
    
    def test_full_pipeline(self):
        """Test full processing pipeline"""
        raw_text = "This product is EXCELLENT!!! Amazing quality at a great price."
        
        # Step 1: Clean
        clean = clean_text(raw_text)
        assert isinstance(clean, str)
        assert len(clean) > 0
        
        # Step 2: Sentiment
        sentiment = get_sentiment_score_and_label(clean)
        assert "label" in sentiment
        assert "score" in sentiment
        
        # Step 3: Topic
        topic_id, prob, words = get_topic_for_review(clean)
        assert isinstance(topic_id, int)
        assert isinstance(prob, float)
        assert isinstance(words, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
