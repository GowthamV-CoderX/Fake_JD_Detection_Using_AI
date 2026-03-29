"""
tfidf_model.py
==============
Location in your repo: tfidf_model.py  (root)

NO CHANGES to logic — this file is identical to your original.
Only change: model save path corrected to "tfidf.joblib" to match
hybrid_detector.py's load() call.
"""

from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
from typing import List, Optional

from feature_extractor import FeatureExtractor

# Save path aligned with hybrid_detector.load()
MODEL_PATH = Path("models/tfidf.joblib")


class TFIDFModel:
    """
    Layer B — Baseline ML model.

    Combines:
      • TF-IDF n-gram features from raw text  (up to trigrams, 15k vocab)
      • 24 hand-crafted structured features from FeatureExtractor

    Classifier: Logistic Regression with L2 regularisation.
    """

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=15_000,
            sublinear_tf=True,
            min_df=1,
            strip_accents="unicode",
            analyzer="word",
        )
        self.classifier = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        )
        self._fitted = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, texts: List[str], labels: List[int]) -> None:
        """Train on raw texts and binary labels (0 = real, 1 = fake)."""
        tfidf_matrix = self.tfidf.fit_transform(texts)
        structured   = self._structured_matrix(texts)
        X = hstack([tfidf_matrix, csr_matrix(structured)])
        self.classifier.fit(X, labels)
        self._fitted = True

    def predict_proba(self, text: str) -> float:
        """Returns P(fake) for a single text string."""
        if not self._fitted:
            raise RuntimeError("TFIDFModel not trained. Call fit() or load().")
        tfidf_vec  = self.tfidf.transform([text])
        feat       = self.feature_extractor.extract(text)
        structured = np.array([self.feature_extractor.to_vector(feat)], dtype=float)
        X = hstack([tfidf_vec, csr_matrix(structured)])
        proba = self.classifier.predict_proba(X)[0]
        return float(proba[1])

    def save(self, path: Optional[Path] = None) -> None:
        p = path or MODEL_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"tfidf": self.tfidf, "classifier": self.classifier}, p)

    def load(self, path: Optional[Path] = None) -> None:
        p = path or MODEL_PATH
        obj = joblib.load(p)
        self.tfidf      = obj["tfidf"]
        self.classifier = obj["classifier"]
        self._fitted    = True

    # ── Private helpers ───────────────────────────────────────────────────────

    def _structured_matrix(self, texts: List[str]) -> np.ndarray:
        rows = []
        for t in texts:
            feat = self.feature_extractor.extract(t)
            rows.append(self.feature_extractor.to_vector(feat))
        return np.array(rows, dtype=float)