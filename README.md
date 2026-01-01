# Customer Feedback NLP Dashboard

An intelligent, end-to-end customer feedback analysis platform using **VADER**, **BERTopic**, and **spaCy** to extract sentiment, identify recurring complaints, and visualize insights.

## 🎯 Features

- **Sentiment Analysis** (VADER & BERT)
- **Topic Modeling** (BERTopic with fallback heuristics)
- **Keyword & Entity Extraction** (spaCy)
- **Interactive Dashboard** (Streamlit)
- **Batch Processing** Pipeline
- **Automated Model Training**

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/purnimaax/customer-feedback-nlp-dashboard.git
cd customer-feedback-nlp-dashboard
```

### 2. Create a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## 🚀 Quick Start

### Step 1: Process your reviews
```bash
# Basic processing (uses fallback topic heuristics)
python scripts/pipeline_offline.py --input data/raw/reviews.csv

# Or sample 5000 reviews for testing
python scripts/pipeline_offline.py --sample --sample-size 5000
```

### Step 2 (Optional): Train BERTopic model
```bash
# Train a BERTopic model on your data
python scripts/train_bertopic.py --sample-size 50000

# Or use custom embedding model
python scripts/train_bertopic.py --embedding-model "all-MiniLM-L6-v2"
```

### Step 3: Launch the dashboard
```bash
streamlit run app/app.py
```

Open your browser to `http://localhost:8501` 🎉

## 📁 Project Structure

```
customer-feedback-nlp-dashboard/
├── nlp/                          # Core NLP modules
│   ├── __init__.py
│   ├── preprocessing.py          # Text cleaning with NLTK
│   ├── sentiment.py              # VADER sentiment analysis
│   ├── bert_sentiment.py         # BERT-based sentiment (optional)
│   ├── topic_modeling.py         # BERTopic with fallbacks
│   └── nlp_spacy.py              # Keyword & entity extraction
├── app/
│   ├── __init__.py
│   └── app.py                    # Streamlit dashboard
├── scripts/
│   ├── pipeline_offline.py       # Batch processing pipeline
│   └── train_bertopic.py         # BERTopic training script
├── data/
│   ├── raw/
│   │   └── reviews.csv          # Input reviews
│   └── processed/
│       └── reviews_with_topics.csv  # Processed output
├── models/
│   └── bertopic_model/          # Trained BERTopic model
├── tests/
│   └── test_nlp.py              # Unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 Pipeline Architecture

```
reviews.csv (raw)
    ↓
[pipeline_offline.py]
    ├─→ clean_text()              → lowercase, remove special chars, stopwords
    ├─→ get_sentiment_score_and_label()  → VADER sentiment
    ├─→ get_topic_for_review()    → BERTopic or keyword fallback
    └─→ extract_noun_phrases()    → spaCy keyword extraction
    ↓
reviews_with_topics.csv (processed)
    ↓
[app.py (Streamlit Dashboard)]
    ├─→ Sentiment distribution charts
    ├─→ Topic analysis & trends
    ├─→ Complaint keyword extraction
    ├─→ Review search & filtering
    └─→ Single review analyzer
```

## 🔧 Configuration

### Sentiment Analysis
- **VADER** (default): Fast, works well on social media/reviews
- **BERT** (optional): More accurate for complex sentences, slower

In `app/app.py`, select sentiment model in the analyzer section:
```python
sentiment_model = st.radio("Sentiment Model:", ["VADER", "BERT"])
```

### Topic Modeling
Configure BERTopic in `scripts/train_bertopic.py`:
```bash
python scripts/train_bertopic.py \
    --embedding-model "all-MiniLM-L6-v2" \
    --min-topic-size 10 \
    --sample-size 50000
```

## 📈 API Reference

### Preprocessing
```python
from nlp import clean_text

clean = clean_text("This is AMAZING!!! Great product.")
# Output: "amazing great product"
```

### Sentiment Analysis
```python
from nlp import get_sentiment_score_and_label

result = get_sentiment_score_and_label("excellent product")
# Output: {"score": 0.8, "label": "positive"}
```

### Topic Modeling
```python
from nlp import get_topic_for_review

topic_id, prob, keywords = get_topic_for_review("battery life is short")
# Output: (100, 0.5, ["battery", "life", "power"])
```

### Keyword Extraction
```python
from nlp import extract_keywords_and_entities

result = extract_keywords_and_entities("Apple's new iPhone is amazing")
# Output: {
#     "keywords": ["apple", "new iphone"],
#     "entities": [("Apple", "ORG"), ("iPhone", "PRODUCT")]
# }
```

## 🧪 Testing

Run unit tests:
```bash
pytest tests/test_nlp.py -v
```

Run specific test:
```bash
pytest tests/test_nlp.py::TestSentiment::test_positive_sentiment -v
```

## 📤 Deployment

### Streamlit Community Cloud
1. Push your repo to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Select your repo and branch
4. Set app path: `app/app.py`

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && python -m spacy download en_core_web_sm
COPY . .
CMD ["streamlit", "run", "app/app.py"]
```

## 📝 Performance Notes

| Component | Time | Notes |
|-----------|------|-------|
| Clean text | 0.1ms/review | Regex + NLTK |
| VADER sentiment | 0.5ms/review | Lightweight |
| BERT sentiment | 10-50ms/review | GPU recommended |
| BERTopic inference | 20-100ms/review | Depends on embedding model |
| Full pipeline (1K reviews) | ~10-30 sec | Fallback topics are faster |

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Add multi-language support
- [ ] Implement aspect-based sentiment analysis
- [ ] Add SQLite/PostgreSQL backend for scalability
- [ ] FastAPI endpoints for model serving
- [ ] Advanced visualization dashboards

## 📚 References

- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [BERTopic](https://maartengr.github.io/BERTopic/)
- [spaCy](https://spacy.io/)
- [Streamlit](https://streamlit.io/)

## 📄 License

MIT License - Feel free to use for personal/commercial projects

## 👤 Author

**Purnima** - [GitHub](https://github.com/purnimaax)

---

**Built with ❤️ for intelligent customer feedback analysis**
