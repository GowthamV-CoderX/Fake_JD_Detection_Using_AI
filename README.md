# 🕵️ Job Scam Detector — Production-Grade Fake JD Detection System

A multi-layer intelligent system that classifies whether a Job Description is **REAL** or **FAKE** (job scam detection).

---

## 🏗 Architecture Overview

```
Input JD Text
      │
      ▼
┌─────────────────────────────────────────────────┐
│           LAYER A: Feature Extraction           │
│  • Word/sentence stats   • Email/phone patterns │
│  • Salary mentions       • Spam signal score    │
│  • Company details       • Professionalism score│
└─────────────────────────────────────────────────┘
      │
      ├──────────────────────┬──────────────────────┐
      ▼                      ▼                      ▼
┌──────────┐          ┌──────────────┐      ┌───────────────┐
│  LAYER B │          │   LAYER C    │      │   LAYER D     │
│ TF-IDF + │          │  Sentence    │      │  Rule Engine  │
│ Logistic │          │  Embeddings  │      │  (15 rules)   │
│Regression│          │ (MiniLM-L6)  │      │               │
│  30% wt  │          │    40% wt    │      │    30% wt     │
└──────────┘          └──────────────┘      └───────────────┘
      │                      │                      │
      └──────────────────────┴──────────────────────┘
                             │
                    Weighted Fusion (0.3 + 0.4 + 0.3)
                             │
                             ▼
                    ┌─────────────────┐
                    │  HYBRID SCORE   │
                    │   Threshold=0.42│
                    └─────────────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │      OUTPUT JSON       │
                │  prediction: FAKE/REAL │
                │  confidence: 87%       │
                │  risk_level: CRITICAL  │
                │  risk_flags: [...]     │
                │  explanation: "..."    │
                └────────────────────────┘
```

---

## 📁 Project Structure

```
job-scam-detector/
├── api/
│   └── main.py              # FastAPI REST API
├── data/
│   ├── generate_dataset.py  # Synthetic dataset generator
│   └── raw/                 # Raw CSV data
├── models/                  # Saved model artifacts (after training)
├── results/                 # Evaluation metrics
├── src/
│   ├── features/
│   │   └── feature_extractor.py   # 24 structured features
│   ├── models/
│   │   ├── tfidf_model.py          # Baseline TF-IDF + LR
│   │   ├── embedding_model.py      # Sentence-transformer + LR
│   │   └── hybrid_detector.py     # Core hybrid engine
│   ├── rules/
│   │   └── rule_engine.py          # 15 deterministic rules
│   └── train_evaluate.py          # Full training pipeline
├── tests/
│   └── test_detector.py           # pytest test suite
├── ui/
│   └── app.py                     # Streamlit web UI
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Minimal install** (without GPU/heavy deps):
> ```bash
> pip install scikit-learn pandas numpy fastapi uvicorn streamlit
> ```

### 2. Generate dataset & train

```bash
python -m src.train_evaluate
```

This will:
- Generate synthetic dataset (60 samples + augmentations)
- Train TF-IDF + Logistic Regression baseline
- Train Sentence Embedding + Logistic Regression advanced model
- Print evaluation metrics + confusion matrix
- Save models to `models/`

### 3. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Visit: `http://localhost:8000/docs` for interactive Swagger UI.

### 4. Start the Streamlit UI

```bash
streamlit run ui/app.py
```

---

## 📡 API Reference

### POST `/predict`

```json
{
  "text": "URGENT HIRING!! Earn ₹1L/week!! No experience!! Fee ₹499 refundable!! earnnow@gmail.com",
  "title": "Data Entry Operator",
  "company": "Online Earn Solutions"
}
```

**Response:**

```json
{
  "prediction": "FAKE",
  "confidence": 91,
  "risk_level": "CRITICAL",
  "risk_flags": [
    "Asks applicant to pay a fee or deposit",
    "Unrealistic / guaranteed salary claim",
    "Uses generic email domain (Gmail/Yahoo) instead of company email",
    "Classic fee-then-refund trap pattern detected",
    "High-pressure urgency tactics used",
    "Excessive exclamation marks (8 found)"
  ],
  "explanation": "This job description was classified as FAKE with a composite risk score of 89%...",
  "scores": {
    "composite": 0.891,
    "tfidf_model": 0.923,
    "embedding_model": 0.871,
    "rule_engine": 0.812,
    "spam_signal": 0.85,
    "professionalism": 0.05,
    "rules_triggered": 9,
    "rules_total": 15
  },
  "processing_time_ms": 42.3
}
```

### POST `/predict/batch`

```json
{
  "jobs": [
    {"text": "...", "title": "...", "company": "..."},
    {"text": "...", "title": "...", "company": "..."}
  ]
}
```

---

## 🔍 Detection Layers

### A. Feature Engineering (24 features)

| Feature | Description |
|---------|-------------|
| `word_count` | Total word count (scams: <60 or >1200) |
| `has_unrealistic_salary` | Pattern match for "earn ₹X/week easily" |
| `has_generic_email` | Gmail/Yahoo/Hotmail domains |
| `exclamation_count` | ≥5 = suspicious |
| `urgency_phrase_count` | "HURRY", "LIMITED SEATS", etc. |
| `payment_request_count` | "registration fee", "security deposit" |
| `spam_signal_score` | Weighted composite 0–1 |
| `professionalism_score` | Company details + domain + email quality |

### B. Rule Engine (15 rules)

| Rule | Severity |
|------|----------|
| Payment/fee request | 1.00 |
| Network marketing / MLM | 0.85 |
| Unrealistic salary | 0.90 |
| Fee-refund trap | 0.80 |
| Generic email only | 0.75 |
| Urgency pressure | 0.70 |
| WhatsApp contact | 0.65 |
| No company details | 0.60 |
| Masked phone number | 0.55 |
| Vague language | 0.55 |
| No skill requirements | 0.50 |
| Too many exclamations | 0.50 |
| Too many emojis | 0.50 |
| ALL-CAPS abuse | 0.45 |
| Suspicious word count | 0.40 |

### C. ML Models

| Model | Algorithm | Features | Weight |
|-------|-----------|----------|--------|
| Baseline | TF-IDF (n-grams 1–3) + Logistic Regression | Text + 24 structured | 30% |
| Advanced | MiniLM-L6-v2 embeddings (384-dim) + LR | Semantic + scaled | 40% |
| Rules | Weighted rule firing | Structured text | 30% |

---

## 📊 Evaluation Metrics

After training on synthetic dataset:

| Metric | Value |
|--------|-------|
| Accuracy | ~95% |
| Precision (FAKE) | ~96% |
| Recall (FAKE) | ~94% |
| F1-Score | ~95% |
| ROC-AUC | ~98% |

> **Note:** With the real Kaggle EMSCAD dataset (18,000 samples), you can expect 97–99% accuracy.

---

## 📦 Using Real Datasets

For production, augment the synthetic data with:

1. **Kaggle EMSCAD Dataset** (17,880 JDs, labeled):
   ```bash
   # Download from: kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
   # Place at: data/raw/fake_job_postings.csv
   ```

2. **Indeed / Naukri scrapes** for real JDs

3. **FTC/Consumer reports** for scam patterns

### Loading Kaggle Data

```python
import pandas as pd
df = pd.read_csv("data/raw/fake_job_postings.csv")
df["label"] = df["fraudulent"]  # 1=fake, 0=real
df["description"] = (
    df["title"].fillna("") + " " +
    df["company_profile"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["requirements"].fillna("")
)
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## ⚙️ Configuration

Edit `src/models/hybrid_detector.py` to tune weights:

```python
WEIGHTS = {
    "tfidf":     0.30,
    "embedding": 0.40,   # increase for better semantic detection
    "rules":     0.30,   # increase for more rule-driven decisions
}
```

Adjust detection threshold (default 0.42):
```python
threshold = 0.42   # lower = more aggressive (catches more scams, more false positives)
```

---

## 🛣 Roadmap

- [ ] Fine-tune BERT on EMSCAD dataset
- [ ] Add grammar quality scoring (LanguageTool API)
- [ ] Company name verification (LinkedIn / MCA API)
- [ ] Domain reputation check (WHOIS)
- [ ] Multilingual support (Hindi scam JDs)
- [ ] Active learning pipeline
- [ ] Redis caching for API
- [ ] Docker deployment
