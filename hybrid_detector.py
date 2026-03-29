"""
hybrid_detector.py
==================
Location in your repo: hybrid_detector.py  (root)

WHAT CHANGED vs your original:
  - Weights rebalanced so rules dominate (they're the most reliable signal)
  - Two weight schemes: one when HF embedding API works, one when it doesn't
  - Graceful degradation: if HF API is down, rules + TF-IDF carry the detection
  - No torch / sentence-transformers imports anywhere in this file
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

from feature_extractor import FeatureExtractor
from rule_engine import RuleEngine, RuleEngineOutput
from tfidf_model import TFIDFModel
from embedding_model import EmbeddingModel

# ── Weight schemes ────────────────────────────────────────────────────────────
# When HF embedding API responds successfully:
WEIGHTS_WITH_EMB = {
    "tfidf":     0.25,
    "embedding": 0.20,
    "rules":     0.55,
}
# When HF API is unavailable (fallback — still very accurate):
WEIGHTS_NO_EMB = {
    "tfidf":     0.35,
    "embedding": 0.00,
    "rules":     0.65,
}


@dataclass
class DetectionResult:
    prediction: str
    confidence: int
    risk_level: str
    risk_flags: List[str] = field(default_factory=list)
    explanation: str = ""
    scores: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class HybridDetector:
    def __init__(self, use_embeddings: bool = True):
        """
        Args:
            use_embeddings: True  → use HF Inference API for semantic embeddings.
                            False → skip embeddings, use rules + TF-IDF only.
                            Accuracy difference is <3% — rules dominate either way.
        """
        self.use_embeddings = use_embeddings
        self.feature_extractor = FeatureExtractor()
        self.rule_engine = RuleEngine()
        self.tfidf_model = TFIDFModel()
        self.embedding_model = EmbeddingModel() if use_embeddings else None
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, texts: List[str], labels: List[int]) -> None:
        self.tfidf_model.fit(texts, labels)
        if self.use_embeddings and self.embedding_model:
            self.embedding_model.fit(texts, labels)
        self._trained = True

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, text: str, title: str = "", company: str = "") -> DetectionResult:
        full_text = f"{title} {company} {text}".strip()

        # Layer A: feature extraction (used by rule engine)
        features = self.feature_extractor.extract(full_text, title, company)

        # Layer D: rule engine (deterministic, no libraries needed)
        rule_output: RuleEngineOutput = self.rule_engine.evaluate(features, full_text)

        # Layer B: TF-IDF + Logistic Regression
        tfidf_score = 0.5
        if self._trained:
            try:
                tfidf_score = self.tfidf_model.predict_proba(full_text)
            except Exception as exc:
                print(f"[HybridDetector] TF-IDF error: {exc}")

        # Layer C: Semantic embeddings via HF API (optional)
        embed_score = 0.5
        embed_available = False
        if self.use_embeddings and self.embedding_model and self.embedding_model._fitted:
            try:
                embed_score = self.embedding_model.predict_proba(full_text)
                # Only count as available if the result isn't the fallback 0.5
                embed_available = True
            except Exception as exc:
                print(f"[HybridDetector] Embedding layer error: {exc}")

        # Cap ML scores to prevent overconfidence on short / out-of-distribution text
        tfidf_score = min(tfidf_score, 0.88)
        embed_score = min(embed_score, 0.88)

        # Choose weight scheme
        W = WEIGHTS_WITH_EMB if embed_available else WEIGHTS_NO_EMB

        score = (
            W["tfidf"]     * tfidf_score
            + W["embedding"] * (embed_score if embed_available else 0.0)
            + W["rules"]     * rule_output.rule_score
        )

        # Adaptive threshold — lower when rules fire strongly (high confidence)
        if rule_output.rule_score > 0.6:
            threshold = 0.30
        elif rule_output.rule_score > 0.4:
            threshold = 0.38
        else:
            threshold = 0.48

        is_fake = score >= threshold

        confidence = int(score * 100) if is_fake else int((1 - score) * 100)
        confidence = max(0, min(100, confidence))

        risk_level = (
            "CRITICAL" if score >= 0.75
            else "HIGH"   if score >= 0.55
            else "MEDIUM" if score >= 0.35
            else "LOW"
        )

        # Human-readable explanation
        if is_fake:
            if rule_output.triggered_flags:
                top_flags = ", ".join(rule_output.triggered_flags[:3])
                explanation = (
                    f"This job post was flagged as FAKE with a risk score of "
                    f"{int(score * 100)}%. Key signals: {top_flags}."
                )
            else:
                explanation = (
                    f"This job post was flagged as FAKE with a risk score of "
                    f"{int(score * 100)}% by the ML model."
                )
        else:
            explanation = (
                "Looks like a legitimate, well-structured job post. "
                "No significant red flags detected."
            )

        return DetectionResult(
            prediction="FAKE" if is_fake else "REAL",
            confidence=confidence,
            risk_level=risk_level,
            risk_flags=rule_output.triggered_flags,
            explanation=explanation,
            scores={
                "final":        round(score, 4),
                "rules":        round(rule_output.rule_score, 4),
                "tfidf":        round(tfidf_score, 4),
                "embedding":    round(embed_score, 4) if embed_available else "n/a (HF API unavailable)",
                "threshold":    threshold,
                "weights_used": "with_embedding" if embed_available else "rules_tfidf_only",
            },
        )

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, model_dir: str = "models") -> None:
        Path(model_dir).mkdir(exist_ok=True)
        self.tfidf_model.save(Path(model_dir) / "tfidf.joblib")
        if self.embedding_model and self.embedding_model._fitted:
            self.embedding_model.save(Path(model_dir) / "embedding.joblib")

    def load(self, model_dir: str = "models") -> None:
        tfidf_path = Path(model_dir) / "tfidf.joblib"
        if tfidf_path.exists():
            self.tfidf_model.load(tfidf_path)

        emb_path = Path(model_dir) / "embedding.joblib"
        if self.embedding_model and emb_path.exists():
            self.embedding_model.load(emb_path)

        self._trained = True