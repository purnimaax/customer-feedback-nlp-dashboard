import spacy
from typing import List, Dict, Any

# Load spaCy model
try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    print("⚠ spaCy model not found. Run: python -m spacy download en_core_web_sm")
    NLP = None


def extract_keywords_and_entities(text: str) -> Dict[str, Any]:
    """
    Extract noun chunks (keywords) and named entities from text using spaCy.
    
    Args:
        text: Input text string
    
    Returns:
        Dictionary containing:
        - keywords: List of noun chunks
        - entities: List of (entity_text, entity_label)
    """
    if not text or NLP is None:
        return {"keywords": [], "entities": []}
    
    try:
        doc = NLP(text)
        
        # Extract noun chunks (potential keywords)
        keywords = [chunk.text.lower() for chunk in doc.noun_chunks]
        
        # Extract named entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        return {
            "keywords": keywords,
            "entities": entities
        }
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return {"keywords": [], "entities": []}


def extract_noun_phrases(text: str) -> List[str]:
    """
    Extract noun phrases (noun chunks) from text.
    Useful for finding complaint themes.
    
    Args:
        text: Input text string
    
    Returns:
        List of noun phrases
    """
    if not text or NLP is None:
        return []
    
    try:
        doc = NLP(text)
        phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
        return phrases
    except Exception as e:
        print(f"Error extracting noun phrases: {e}")
        return []
