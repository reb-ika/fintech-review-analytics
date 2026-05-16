# fintech-review-analytics
cat > README.md << 'EOF'
# Customer Experience Analytics for Fintech Apps
**Kifiya AI Training Program — Week 2**
*Omega Consultancy | Ethiopian Bank Mobile App Analysis*

---

## Business Objective
Omega Consultancy has been engaged by three Ethiopian banks — Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), and Dashen Bank — to analyze what their mobile app users are saying and why it matters.

This project builds a rigorous analytics pipeline that:
1. **Scrapes** user reviews from Google Play Store
2. **Analyzes sentiment** using VADER, TextBlob, and DistilBERT
3. **Extracts themes** using TF-IDF, bigrams, trigrams, and noun extraction
4. **Stores data** in a PostgreSQL relational database
5. **Synthesizes insights** into actionable recommendations for bank product teams

---

## Project Structure
fintech-review-analytics/
├── .github/
│   └── workflows/
│       └── unittests.yml       # CI/CD pipeline
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── raw/                    # Raw and processed data (NOT committed)
│       ├── bank_reviews.csv          # Scraped reviews
│       └── bank_reviews_analyzed.csv # Reviews with sentiment and themes
├── notebooks/
│   ├── task_1_scraping.ipynb         # Data collection and preprocessing
│   ├── task_2_sentiment_analysis.ipynb # Sentiment and thematic analysis
│   ├── task_3_database.ipynb         # PostgreSQL database engineering
│   └── task_4_insights.ipynb         # Insights and recommendations
├── scripts/
│   ├── schema.sql                    # PostgreSQL database schema
│   └── README.md
├── src/
│   └── init.py
└── tests/
└── init.py
---

## Banks Analyzed

| Bank | App | Google Play ID | Reviews |
|------|-----|---------------|---------|
| Commercial Bank of Ethiopia | CBE Mobile Banking | com.combanketh.mobilebanking | 455 |
| Bank of Abyssinia | BOA Mobile Banking | com.boa.boaMobileBanking | 499 |
| Dashen Bank | Dashen Super App | com.dashen.dashensuperapp | 498 |
| **Total** | | | **1,452** |

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/reb-ika/fintech-review-analytics.git
cd fintech-review-analytics
```

### 2. Create and activate virtual environment
```bash
# Create
python -m venv venv

# Activate (Windows)
source venv/Scripts/activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up PostgreSQL
```bash
# Install PostgreSQL from https://www.postgresql.org/download/
# Create the database
psql -U postgres -c "CREATE DATABASE bank_reviews;"
```

### 5. Run notebooks in order
1.notebooks/task_1_scraping.ipynb
2.notebooks/task_2_sentiment_analysis.ipynb
3.notebooks/task_3_database.ipynb
4.notebooks/task_4_insights.ipynb
---

## Scraping Methodology

Reviews were collected using `google-play-scraper` library:
- **Sort order**: Newest first to capture most recent user feedback
- **Target**: 600 reviews per bank (after deduplication: 400+ per bank)
- **Date range**: November 2024 to May 2026
- **Fields collected**: review text, rating (1-5), date, bank name, source

### Limitations
- Dashen Bank's original app ID (`com.dashen.dashensmart`) returned 0 reviews. Correct ID (`com.dashen.dashensuperapp`) was identified via Google Play search.
- Google Play scraper may hit rate limits for very large requests.
- Reviews are in English only — Amharic reviews are not captured.

---

## Sentiment Analysis Tools

| Tool | Type | Score Range | Strength |
|------|------|-------------|---------|
| VADER | Lexicon-based | -1 to +1 | Fast, handles informal text |
| TextBlob | Lexicon-based | -1 to +1 | Polarity + subjectivity |
| DistilBERT | Transformer | POSITIVE/NEGATIVE | Most accurate, context-aware |

### Tool Selection Rationale
VADER was selected as the primary tool because it is specifically designed for short, informal social media text — ideal for app reviews. DistilBERT was applied as validation. VADER-DistilBERT correlation of 0.65 confirms reasonable agreement.

---

## Key Findings

### Sentiment Overview
| Bank | Positive | Neutral | Negative | Avg Rating |
|------|---------|---------|---------|-----------|
| CBE | 56.7% | 29.9% | 13.4% | 3.94 ⭐ |
| Dashen | 57.8% | 25.5% | 16.7% | 3.78 ⭐ |
| BOA | 43.5% | 34.7% | 21.8% | 3.28 ⭐ |

### Top Themes (All Banks)
- 🔴 **App Stability** — #1 pain point (90 negative reviews)
- 🔴 **Transaction Performance** — #2 pain point (36 negative reviews)
- 🟢 **UI & User Experience** — #1 satisfaction driver (112 positive reviews)

---

## Database Schema

```sql
-- Banks table
CREATE TABLE banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) NOT NULL UNIQUE,
    app_name VARCHAR(200),
    app_id VARCHAR(200)
);

-- Reviews table
CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    bank_id INTEGER REFERENCES banks(bank_id),
    review_text TEXT,
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    review_date DATE,
    sentiment_label VARCHAR(20),
    sentiment_score FLOAT,
    identified_theme VARCHAR(100),
    source VARCHAR(50)
);
```

---

## CI/CD

GitHub Actions workflow runs automatically on every push to main:
- Sets up Python 3.11
- Installs all dependencies from `requirements.txt`
- Validates the environment is reproducible

---

## Ethical Considerations

1. **Negativity Bias** — Users are more likely to review after bad experiences
2. **Language Bias** — VADER trained on English; Amharic reviews may be misclassified
3. **Sampling Bias** — Only most recent reviews captured
4. **Rating Inflation** — Apps often prompt ratings after positive experiences

---

## Tools & Libraries

| Tool | Purpose |
|------|---------|
| google-play-scraper | Web scraping |
| Pandas / NumPy | Data manipulation |
| VADER (NLTK) | Sentiment analysis |
| TextBlob | Sentiment + subjectivity |
| DistilBERT (HuggingFace) | Transformer sentiment |
| scikit-learn TF-IDF | Keyword extraction |
| WordCloud | Visual keyword analysis |
| Matplotlib / Seaborn | Visualization |
| psycopg2 / PostgreSQL | Database engineering |
| Jupyter | Interactive notebooks |

---

## Author
**Rebika Woldeyesus**
Kifiya AI Training Program — Week 2
GitHub: [@reb-ika](https://github.com/reb-ika)
EOF