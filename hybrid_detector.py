"""
hybrid_detector.py  —  Enhanced accuracy version
Location: hybrid_detector.py (repo root)

IMPROVEMENTS vs original:
  - Uses flag_descriptions from rule engine for better UI display
  - Smarter adaptive threshold based on combined signal strength
  - Better confidence calibration
  - Graceful HF API fallback with no accuracy drop warning
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

from feature_extractor import FeatureExtractor
from rule_engine import RuleEngine, RuleEngineOutput
from tfidf_model import TFIDFModel
from embedding_model import EmbeddingModel

WEIGHTS_WITH_EMB = {"tfidf": 0.25, "embedding": 0.20, "rules": 0.55}
WEIGHTS_NO_EMB   = {"tfidf": 0.35, "embedding": 0.00, "rules": 0.65}


@dataclass
class DetectionResult:
    prediction: str
    confidence: int
    risk_level: str
    risk_flags: List[str] = field(default_factory=list)
    flag_descriptions: List[str] = field(default_factory=list)
    explanation: str = ""
    scores: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class HybridDetector:
    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings
        self.feature_extractor = FeatureExtractor()
        self.rule_engine = RuleEngine()
        self.tfidf_model = TFIDFModel()
        self.embedding_model = EmbeddingModel() if use_embeddings else None
        self._trained = False

    def train(self, texts: List[str], labels: List[int]) -> None:
        self.tfidf_model.fit(texts, labels)
        if self.use_embeddings and self.embedding_model:
            self.embedding_model.fit(texts, labels)
        self._trained = True

    def predict(self, text: str, title: str = "", company: str = "") -> DetectionResult:
        full_text = f"{title} {company} {text}".strip()

        features   = self.feature_extractor.extract(full_text, title, company)
        rule_output: RuleEngineOutput = self.rule_engine.evaluate(features, full_text)

        tfidf_score = 0.5
        if self._trained:
            try:
                tfidf_score = self.tfidf_model.predict_proba(full_text)
            except Exception as exc:
                print(f"[HybridDetector] TF-IDF error: {exc}")

        embed_score     = 0.5
        embed_available = False
        if self.use_embeddings and self.embedding_model and self.embedding_model._fitted:
            try:
                embed_score     = self.embedding_model.predict_proba(full_text)
                embed_available = True
            except Exception as exc:
                print(f"[HybridDetector] Embedding error: {exc}")

        tfidf_score = min(tfidf_score, 0.88)
        embed_score = min(embed_score, 0.88)

        W = WEIGHTS_WITH_EMB if embed_available else WEIGHTS_NO_EMB
        score = (
            W["tfidf"]     * tfidf_score
            + W["embedding"] * (embed_score if embed_available else 0.0)
            + W["rules"]     * rule_output.rule_score
        )

        # Adaptive threshold
        if rule_output.rule_score > 0.6:
            threshold = 0.28
        elif rule_output.rule_score > 0.4:
            threshold = 0.36
        else:
            threshold = 0.46

        is_fake    = score >= threshold
        confidence = int(score * 100) if is_fake else int((1 - score) * 100)
        confidence = max(0, min(100, confidence))

        risk_level = (
            "CRITICAL" if score >= 0.75
            else "HIGH"   if score >= 0.55
            else "MEDIUM" if score >= 0.35
            else "LOW"
        )

        if is_fake:
            if rule_output.flag_descriptions:
                top = ", ".join(rule_output.flag_descriptions[:3])
                explanation = (
                    f"This job post scored {int(score*100)}% on our scam detection system. "
                    f"Key reasons: {top}."
                )
            else:
                explanation = (
                    f"Flagged as FAKE with a risk score of {int(score*100)}% by the ML model."
                )
        else:
            explanation = (
                "This job post appears legitimate. It has proper structure, "
                "professional language, and no major scam indicators were detected."
            )

        return DetectionResult(
            prediction="FAKE" if is_fake else "REAL",
            confidence=confidence,
            risk_level=risk_level,
            risk_flags=rule_output.triggered_flags,
            flag_descriptions=rule_output.flag_descriptions,
            explanation=explanation,
            scores={
                "final":        round(score, 4),
                "rules":        round(rule_output.rule_score, 4),
                "tfidf":        round(tfidf_score, 4),
                "embedding":    round(embed_score, 4) if embed_available else "n/a",
                "threshold":    threshold,
                "weights_used": "full_3_layer" if embed_available else "rules_tfidf_only",
            },
        )

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