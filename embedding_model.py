"""
embedding_model.py
==================
Location in your repo: embedding_model.py  (root)

WHAT CHANGED vs your original:
  - Removed: sentence-transformers, torch (these were 4.2 GB combined)
  - Added:   requests call to Hugging Face Inference API (free, same model)
  - The model used is IDENTICAL: all-MiniLM-L12-v2
  - Drop-in replacement — all method signatures stay the same

SETUP (one-time):
  1. Sign up free at https://huggingface.co
  2. Go to https://huggingface.co/settings/tokens
  3. Click "New token" → name it anything → Role: Read → Create token
  4. Copy the token (starts with hf_...)
  5. Locally:  create a file named .env in your repo root with:
                 HF_API_TOKEN=hf_xxxxxxxxxxxxxxxx
  6. Vercel:   Project → Settings → Environment Variables → add HF_API_TOKEN
"""

from __future__ import annotations

import os
import time
import numpy as np
import joblib
import requests
from pathlib import Path
from typing import List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load .env file when running locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Same model as your original — just running on HF servers instead of locally
HF_API_URL = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction/"
    "sentence-transformers/all-MiniLM-L12-v2"
)

MODEL_PATH = Path("models/embedding_lr.joblib")

# Read token once at module level
_HF_TOKEN: Optional[str] = None


def _get_token() -> Optional[str]:
    global _HF_TOKEN
    if _HF_TOKEN is None:
        _HF_TOKEN = os.environ.get("HF_API_TOKEN", "").strip()
    return _HF_TOKEN or None


def _encode_via_api(texts: List[str], retries: int = 3) -> np.ndarray:
    """
    Calls HF Inference API to get 384-dim sentence embeddings.
    Mirrors SentenceTransformer(normalize_embeddings=True) behaviour exactly.
    Falls back to zero-vectors on failure so rules + TF-IDF still run.
    """
    token = _get_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "inputs": texts,
        "options": {"wait_for_model": True},
    }

    for attempt in range(retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                embeddings = np.array(data, dtype=np.float32)

                # Normalize to unit length — same as normalize_embeddings=True
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                return embeddings / norms

            elif response.status_code == 503:
                # HF is loading the model — wait and retry
                wait_secs = min(
                    int(response.json().get("estimated_time", 20)), 30
                )
                print(f"[EmbeddingModel] HF model loading, waiting {wait_secs}s...")
                time.sleep(wait_secs)

            elif response.status_code == 401:
                print("[EmbeddingModel] HF token invalid or missing — running without embeddings.")
                break

            else:
                print(
                    f"[EmbeddingModel] HF API error {response.status_code}: "
                    f"{response.text[:200]}"
                )
                break

        except requests.exceptions.Timeout:
            wait = 2 ** attempt
            print(f"[EmbeddingModel] Timeout (attempt {attempt + 1}/{retries}), retrying in {wait}s...")
            time.sleep(wait)

        except Exception as exc:
            print(f"[EmbeddingModel] Request failed: {exc}")
            break

    # Graceful fallback — zero vectors keep the rest of the pipeline alive
    print("[EmbeddingModel] Falling back to zero embeddings. Rules + TF-IDF still active.")
    return np.zeros((len(texts), 384), dtype=np.float32)


class EmbeddingModel:
    """
    Drop-in replacement for the original sentence-transformers EmbeddingModel.
    Public API is 100% identical — only the backend changed.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            C=0.5,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self._fitted = False

    # ── Keep the same encode() signature ───────────────────────────────────
    def encode(self, texts: List[str]) -> np.ndarray:
        """Returns (N, 384) normalized embedding matrix via HF API."""
        return _encode_via_api(texts)

    # ── Training ────────────────────────────────────────────────────────────
    def fit(self, texts: List[str], labels: List[int]) -> None:
        embeddings = self.encode(texts)
        X = self.scaler.fit_transform(embeddings)
        self.classifier.fit(X, labels)
        self._fitted = True

    # ── Inference ───────────────────────────────────────────────────────────
    def predict_proba(self, text: str) -> float:
        """Returns P(fake) for a single text string."""
        if not self._fitted:
            raise RuntimeError("Embedding model not trained. Call fit() or load().")
        emb = self.encode([text])
        X = self.scaler.transform(emb)
        proba = self.classifier.predict_proba(X)[0]
        return float(proba[1])

    # ── Persistence ─────────────────────────────────────────────────────────
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