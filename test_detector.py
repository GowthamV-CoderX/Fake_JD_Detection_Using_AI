"""
tests/test_detector.py
Unit tests for feature extractor, rule engine, and hybrid detector.
Run: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pytest
from feature_extractor import FeatureExtractor
from rule_engine import RuleEngine
from hybrid_detector import HybridDetector


FAKE_JD = (
    "URGENT HIRING!! Work From Home!! 🔥 Earn ₹1,00,000 per week easily!! "
    "Anyone can do it!! No experience needed!! HURRY only 5 seats!! "
    "Registration fee ₹499 refundable!! WhatsApp 9XXXXXXXXX or earnnow@gmail.com"
)

REAL_JD = (
    "Infosys is hiring a Senior Software Engineer for our Cloud Platform team. "
    "Requirements: 4+ years Python/Go, AWS/Kubernetes. B.Tech in CS. "
    "CTC: ₹18–28 LPA. Apply: careers.infosys.com | hr.pune@infosys.com"
)


class TestFeatureExtractor:
    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_fake_has_high_spam_score(self):
        feat = self.extractor.extract(FAKE_JD)
        assert feat.spam_signal_score > 0.5, "Fake JD should have high spam score"

    def test_real_has_low_spam_score(self):
        feat = self.extractor.extract(REAL_JD)
        assert feat.spam_signal_score < 0.4, "Real JD should have low spam score"

    def test_fake_has_generic_email(self):
        feat = self.extractor.extract(FAKE_JD)
        assert feat.has_generic_email, "Fake JD uses Gmail"

    def test_real_has_company_email(self):
        feat = self.extractor.extract(REAL_JD)
        assert feat.has_company_email, "Real JD uses company domain email"

    def test_fake_has_unrealistic_salary(self):
        feat = self.extractor.extract(FAKE_JD)
        assert feat.has_unrealistic_salary

    def test_fake_has_payment_request(self):
        feat = self.extractor.extract(FAKE_JD)
        assert feat.payment_request_count > 0

    def test_vector_length(self):
        feat = self.extractor.extract(REAL_JD)
        vec = self.extractor.to_vector(feat)
        assert len(vec) == len(FeatureExtractor.feature_names())

    def test_professionalism_real_higher_than_fake(self):
        real_feat = self.extractor.extract(REAL_JD)
        fake_feat = self.extractor.extract(FAKE_JD)
        assert real_feat.professionalism_score > fake_feat.professionalism_score


class TestRuleEngine:
    def setup_method(self):
        self.extractor = FeatureExtractor()
        self.engine = RuleEngine()

    def test_fake_triggers_multiple_rules(self):
        feat = self.extractor.extract(FAKE_JD)
        output = self.engine.evaluate(feat, FAKE_JD)
        assert len(output.active_rules) >= 4, (
            f"Expected ≥4 rules triggered, got {len(output.active_rules)}"
        )

    def test_real_triggers_few_rules(self):
        feat = self.extractor.extract(REAL_JD)
        output = self.engine.evaluate(feat, REAL_JD)
        assert len(output.active_rules) <= 2

    def test_fake_risk_level_high(self):
        feat = self.extractor.extract(FAKE_JD)
        output = self.engine.evaluate(feat, FAKE_JD)
        assert output.risk_level in ("HIGH", "CRITICAL")

    def test_real_risk_level_low(self):
        feat = self.extractor.extract(REAL_JD)
        output = self.engine.evaluate(feat, REAL_JD)
        assert output.risk_level in ("LOW", "MEDIUM")

    def test_payment_rule_triggered(self):
        feat = self.extractor.extract(FAKE_JD)
        output = self.engine.evaluate(feat, FAKE_JD)
        ids = [r.rule_id for r in output.active_rules]
        assert "payment_request" in ids

    def test_fee_refund_trap_triggered(self):
        feat = self.extractor.extract(FAKE_JD)
        output = self.engine.evaluate(feat, FAKE_JD)
        ids = [r.rule_id for r in output.active_rules]
        assert "fee_refund_trap" in ids


class TestHybridDetector:
    def setup_method(self):
        # Use rule-only mode for unit tests (no training required)
        self.detector = HybridDetector(use_embeddings=False)

    def test_fake_classified_correctly_rule_mode(self):
        result = self.detector.predict(FAKE_JD)
        assert result.prediction == "FAKE", (
            f"Expected FAKE, got {result.prediction} (confidence {result.confidence}%)"
        )

    def test_real_classified_correctly_rule_mode(self):
        result = self.detector.predict(REAL_JD)
        assert result.prediction == "REAL", (
            f"Expected REAL, got {result.prediction} (confidence {result.confidence}%)"
        )

    def test_result_has_all_fields(self):
        result = self.detector.predict(FAKE_JD)
        assert result.prediction in ("REAL", "FAKE")
        assert 0 <= result.confidence <= 100
        assert result.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert isinstance(result.risk_flags, list)
        assert isinstance(result.explanation, str) and len(result.explanation) > 0
        assert "composite" in result.scores

    def test_batch_predict(self):
        jobs = [
            {"text": FAKE_JD, "title": "WFH Job"},
            {"text": REAL_JD, "title": "Senior SWE"},
        ]
        results = self.detector.predict_batch(jobs)
        assert len(results) == 2

    def test_to_dict(self):
        result = self.detector.predict(FAKE_JD)
        d = result.to_dict()
        assert "prediction" in d
        assert "risk_flags" in d

    def test_to_json(self):
        import json
        result = self.detector.predict(REAL_JD)
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["prediction"] in ("REAL", "FAKE")


class TestTrainedDetector:
    """Tests that require actual model training."""

    @pytest.fixture(scope="class")
    def trained_detector(self):
        from generate_dataset import REAL_JDS, FAKE_JDS
        texts = [j["text"] for j in REAL_JDS + FAKE_JDS]
        labels = [j["label"] for j in REAL_JDS + FAKE_JDS]
        d = HybridDetector(use_embeddings=False)
        d.train(texts, labels)
        return d

    def test_trained_fake_prediction(self, trained_detector):
        result = trained_detector.predict(FAKE_JD)
        assert result.prediction == "FAKE"

    def test_trained_real_prediction(self, trained_detector):
        result = trained_detector.predict(REAL_JD)
        assert result.prediction == "REAL"
