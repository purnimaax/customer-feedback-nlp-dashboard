import re
import nltk
from nltk.corpus import stopwords
from typing import List

# Download stopwords once (if not present)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """
    Simple text cleaner for Streamlit Cloud:
    - lowercases
    - removes non-letters
    - removes stopwords
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove special chars & digits (keep only letters and spaces)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    # 3. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # 4. Remove stopwords using nltk
    tokens: List[str] = [
        word
        for word in text.split()
        if word not in STOP_WORDS and len(word) > 2  # Also remove very short tokens
    ]
    
    return " ".join(tokens)
