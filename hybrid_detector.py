from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from pathlib import Path

from feature_extractor import FeatureExtractor
from rule_engine import RuleEngine, RuleEngineOutput
from tfidf_model import TFIDFModel
from embedding_model import EmbeddingModel


WEIGHTS = {
    "tfidf": 0.2,
    "embedding": 0.2,
    "rules": 0.6   # RULES DOMINATE
}


@dataclass
class DetectionResult:
    prediction: str
    confidence: int
    risk_level: str
    risk_flags: List[str] = field(default_factory=list)
    explanation: str = ""
    scores: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class HybridDetector:

    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings
        self.feature_extractor = FeatureExtractor()
        self.rule_engine = RuleEngine()
        self.tfidf_model = TFIDFModel()
        self.embedding_model = EmbeddingModel() if use_embeddings else None
        self._trained = False

    def train(self, texts: List[str], labels: List[int]):
        self.tfidf_model.fit(texts, labels)

        if self.use_embeddings:
            self.embedding_model.fit(texts, labels)

        self._trained = True

    def predict(self, text: str, title: str = "", company: str = "") -> DetectionResult:

        full_text = f"{title} {company} {text}"

        features = self.feature_extractor.extract(full_text, title, company)
        rule_output: RuleEngineOutput = self.rule_engine.evaluate(features, full_text)

        tfidf_score = 0.5
        embed_score = 0.5

        if self._trained:
            tfidf_score = self.tfidf_model.predict_proba(full_text)
            if self.use_embeddings:
                embed_score = self.embedding_model.predict_proba(full_text)

        # 🔥 LIMIT ML OVERCONFIDENCE
        tfidf_score = min(tfidf_score, 0.85)
        embed_score = min(embed_score, 0.85)

        # 🔥 COMPOSITE SCORE
        score = (
            WEIGHTS["tfidf"] * tfidf_score +
            WEIGHTS["embedding"] * embed_score +
            WEIGHTS["rules"] * rule_output.rule_score
        )

        # 🔥 SMART THRESHOLD
        if rule_output.rule_score > 0.6:
            threshold = 0.30
        elif rule_output.rule_score > 0.4:
            threshold = 0.38
        else:
            threshold = 0.48

        is_fake = score >= threshold

        confidence = int(score * 100) if is_fake else int((1 - score) * 100)

        risk_level = (
            "CRITICAL" if score >= 0.75 else
            "HIGH" if score >= 0.55 else
            "MEDIUM" if score >= 0.35 else
            "LOW"
        )

        explanation = (
            f"Fake detected due to: {rule_output.triggered_flags}"
            if is_fake else
            "Looks like a legitimate structured job post"
        )

        return DetectionResult(
            prediction="FAKE" if is_fake else "REAL",
            confidence=confidence,
            risk_level=risk_level,
            risk_flags=rule_output.triggered_flags,
            explanation=explanation,
            scores={
                "final": round(score, 4),
                "rules": round(rule_output.rule_score, 4),
                "tfidf": round(tfidf_score, 4),
                "embedding": round(embed_score, 4),
            },
        )

    def save(self, model_dir="models"):
        Path(model_dir).mkdir(exist_ok=True)
        self.tfidf_model.save(Path(model_dir) / "tfidf.joblib")
        if self.embedding_model:
            self.embedding_model.save(Path(model_dir) / "embedding.joblib")

    def load(self, model_dir="models"):
        self.tfidf_model.load(Path(model_dir) / "tfidf.joblib")
        if self.embedding_model:
            self.embedding_model.load(Path(model_dir) / "embedding.joblib")
        self._trained = True