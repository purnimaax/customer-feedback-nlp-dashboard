import re
import spacy
import nltk
from nltk.corpus import stopwords
from typing import List

# --- GLOBAL MODEL LOADING (Executes only once at app startup) ---

# Download stopwords once (if not present)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))

NLP = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """
    Performs text cleaning including lowercasing, special character removal,
    stopword removal, and lemmatization.

    Args:
        text: The raw input string (e.g., reviewText).

    Returns:
        The cleaned and preprocessed string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove special chars & digits (keeps only letters and spaces)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 3. Process with spaCy for tokenization and lemmatization
    doc = NLP(text)

    tokens: List[str] = [
        token.lemma_
        for token in doc
        # Filter: remove stopwords and punctuation (though most punctuation is already gone)
        if token.text not in STOP_WORDS and not token.is_punct
    ]

    return " ".join(tokens)
