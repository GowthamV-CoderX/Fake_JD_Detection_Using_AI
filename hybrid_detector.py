"""
hybrid_detector.py  —  v3: Adversarial AI Fake Detection
Location: hybrid_detector.py (repo root)

KEY CHANGES in v3:
  - adversarial_score from rule engine now contributes to FAKE verdict
  - AI-GENERATED verdict only fires when adversarial_score is LOW
    (i.e., AI-written but actually looks like a real job)
  - New scoring formula: adversarial posts get boosted scam_score
  - Explanation text distinguishes adversarial AI fakes from regular fakes
  - 4-way internal logic: FAKE(obvious) / FAKE(adversarial) / AI-GENERATED / REAL
    displayed as 3 verdicts but with better explanations
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from pathlib import Path

from feature_extractor import FeatureExtractor
from rule_engine import RuleEngine, RuleEngineOutput
from tfidf_model import TFIDFModel
from embedding_model import EmbeddingModel

WEIGHTS_WITH_EMB = {"tfidf": 0.25, "embedding": 0.20, "rules": 0.55}
WEIGHTS_NO_EMB   = {"tfidf": 0.35, "embedding": 0.00, "rules": 0.65}

# Adversarial AI fake: high AI score + adversarial signals → FAKE
ADVERSARIAL_FAKE_THRESHOLD  = 0.45   # adversarial_score threshold
AI_ASSIST_THRESHOLD         = 0.30   # minimum ai_score for adversarial to apply
# Pure AI-generated (not fake): high AI score, low adversarial
AI_VERDICT_THRESHOLD        = 0.48


@dataclass
class DetectionResult:
    prediction: str          # "FAKE" | "AI-GENERATED" | "REAL"
    confidence: int
    risk_level: str
    risk_flags: List[str] = field(default_factory=list)
    flag_descriptions: List[str] = field(default_factory=list)
    explanation: str = ""
    scores: Dict[str, Any] = field(default_factory=dict)
    ai_score: float = 0.0
    ai_signals: List[str] = field(default_factory=list)
    adversarial_score: float = 0.0
    adversarial_signals: List[str] = field(default_factory=list)
    is_adversarial: bool = False

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
        base_scam_score = (
            W["tfidf"]     * tfidf_score
            + W["embedding"] * (embed_score if embed_available else 0.0)
            + W["rules"]     * rule_output.rule_score
        )

        # ── Adversarial AI fake boost ─────────────────────────────────────────
        # If AI wrote this post AND adversarial signals are present,
        # boost the scam score so it crosses the fake threshold.
        # This is the core fix for "AI fake looks like real job" problem.
        ai_score          = rule_output.ai_score
        adversarial_score = rule_output.adversarial_score
        is_adversarial    = rule_output.is_adversarial_fake

        adversarial_boost = 0.0
        if ai_score >= AI_ASSIST_THRESHOLD and adversarial_score >= ADVERSARIAL_FAKE_THRESHOLD:
            # Scale boost: max 0.35 added to scam score
            # This ensures a "clean" AI post with strong adversarial signals crosses fake threshold
            adversarial_boost = min(adversarial_score * 0.60, 0.35)

        scam_score = min(base_scam_score + adversarial_boost, 1.0)

        # ── Adaptive scam threshold ───────────────────────────────────────────
        # Lower threshold when adversarial signals are present (we're more sure)
        if rule_output.rule_score > 0.6:
            threshold = 0.28
        elif rule_output.rule_score > 0.4:
            threshold = 0.36
        elif is_adversarial:
            # Adversarial AI fake: lower threshold since these evade rule engine
            threshold = 0.32
        else:
            threshold = 0.46

        is_fake = scam_score >= threshold

        # ── AI-Generated verdict ──────────────────────────────────────────────
        # Only fire AI-GENERATED if:
        # - Not fake by scam score
        # - High AI score
        # - LOW adversarial score (genuinely AI-written but not a scam)
        is_ai_gen = (
            not is_fake
            and ai_score >= AI_VERDICT_THRESHOLD
            and adversarial_score < ADVERSARIAL_FAKE_THRESHOLD
        )

        # ── Final verdict ─────────────────────────────────────────────────────
        if is_fake:
            prediction = "FAKE"
        elif is_ai_gen:
            prediction = "AI-GENERATED"
        else:
            prediction = "REAL"

        # ── Confidence ────────────────────────────────────────────────────────
        if prediction == "FAKE":
            if is_adversarial and adversarial_boost > 0:
                # Adversarial fake — blend scam + adversarial for confidence
                confidence = int((scam_score * 0.6 + adversarial_score * 0.4) * 100)
            else:
                confidence = int(scam_score * 100)
        elif prediction == "AI-GENERATED":
            confidence = int(ai_score * 100)
        else:
            real_conf = int((1 - max(scam_score, ai_score * 0.5)) * 100)
            confidence = real_conf

        confidence = max(0, min(100, confidence))

        # ── Risk level ────────────────────────────────────────────────────────
        if prediction == "FAKE":
            if is_adversarial:
                # Adversarial fakes are always at least HIGH — they're deliberately deceptive
                risk_level = "CRITICAL" if adversarial_score >= 0.70 else "HIGH"
            else:
                risk_level = (
                    "CRITICAL" if scam_score >= 0.75
                    else "HIGH"   if scam_score >= 0.55
                    else "MEDIUM" if scam_score >= 0.35
                    else "LOW"
                )
        elif prediction == "AI-GENERATED":
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # ── Explanation ───────────────────────────────────────────────────────
        if prediction == "FAKE":
            if is_adversarial and adversarial_boost > 0:
                # Adversarial fake explanation — explains WHY a "professional" post is fake
                top_adversarial = "; ".join(
                    s.replace("⚠ ", "") for s in rule_output.adversarial_signals[:3]
                ) if rule_output.adversarial_signals else "structural inconsistencies"
                explanation = (
                    f"⚠️ This post appears to be an AI-crafted fake job designed to look legitimate. "
                    f"While it uses professional language and proper structure, our adversarial detection "
                    f"found {len(rule_output.adversarial_signals)} deception signal(s). "
                    f"Key findings: {top_adversarial}. "
                    f"Adversarial confidence: {int(adversarial_score*100)}%."
                )
            elif rule_output.flag_descriptions:
                top = ", ".join(rule_output.flag_descriptions[:3])
                explanation = (
                    f"This job post scored {int(scam_score*100)}% on our scam detection system. "
                    f"Key reasons: {top}."
                )
            else:
                explanation = (
                    f"Flagged as FAKE with a risk score of {int(scam_score*100)}% by the ML model."
                )

        elif prediction == "AI-GENERATED":
            top_signals = ", ".join(rule_output.ai_signals[:3]) if rule_output.ai_signals else "multiple linguistic patterns"
            explanation = (
                f"This job description appears to be written by an AI tool (ChatGPT, Gemini, etc.) "
                f"with {int(ai_score*100)}% confidence. It is not necessarily a scam, but was not "
                f"authored by a human recruiter. Key signals: {top_signals}."
            )

        else:
            explanation = (
                "This job post appears legitimate. It has proper structure, "
                "professional language, and no major scam indicators were detected."
            )

        # Collect all flags for display
        all_flag_descriptions = list(rule_output.flag_descriptions)

        return DetectionResult(
            prediction=prediction,
            confidence=confidence,
            risk_level=risk_level,
            risk_flags=rule_output.triggered_flags,
            flag_descriptions=all_flag_descriptions,
            explanation=explanation,
            ai_score=round(ai_score, 3),
            ai_signals=rule_output.ai_signals,
            adversarial_score=round(adversarial_score, 3),
            adversarial_signals=rule_output.adversarial_signals,
            is_adversarial=is_adversarial and is_fake,
            scores={
                "final":              round(scam_score, 4),
                "base_scam":          round(base_scam_score, 4),
                "adversarial_boost":  round(adversarial_boost, 4),
                "rules":              round(rule_output.rule_score, 4),
                "tfidf":              round(tfidf_score, 4),
                "embedding":          round(embed_score, 4) if embed_available else "n/a",
                "ai_score":           round(ai_score, 4),
                "adversarial_score":  round(adversarial_score, 4),
                "threshold":          threshold,
                "weights_used":       "full_3_layer" if embed_available else "rules_tfidf_only",
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