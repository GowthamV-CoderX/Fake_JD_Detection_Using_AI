from __future__ import annotations
import numpy as np
import joblib
from pathlib import Path
from typing import List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path("models/embedding_lr.joblib")
EMBEDDING_MODEL_NAME = "all-MiniLM-L12-v2"  # better semantic understanding

class EmbeddingModel:

    def __init__(self):
        self._encoder = None
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            C=0.5,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self._fitted = False

    @property
    def encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                print("✅ Loading embedding model...")
                self._encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
                print("✅ Embedding model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Embedding model failed to load: {e}")
        return self._encoder

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def fit(self, texts: List[str], labels: List[int]) -> None:
        embeddings = self.encode(texts)
        X = self.scaler.fit_transform(embeddings)
        self.classifier.fit(X, labels)
        self._fitted = True

    def predict_proba(self, text: str) -> float:
        if not self._fitted:
            raise RuntimeError("Embedding model not trained. Call fit() or load().")
        emb = self.encode([text])
        X = self.scaler.transform(emb)
        proba = self.classifier.predict_proba(X)[0]
        return float(proba[1])

    def save(self, path: Optional[Path] = None) -> None:
        p = path or MODEL_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "classifier": self.classifier}, p)

    def load(self, path: Optional[Path] = None) -> None:
        p = path or MODEL_PATH
        obj = joblib.load(p)
        self.scaler = obj["scaler"]
        self.classifier = obj["classifier"]
        self._fitted = True