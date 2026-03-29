#!/usr/bin/env python3
"""
Quick test for fake job detection
"""

from hybrid_detector import HybridDetector

def test_fake_job():
    detector = HybridDetector(use_embeddings=False)

    fake_job = {
        "text": "URGENT HIRING!! Work From Home!! Earn 100000 per week easily!! No experience needed!! HURRY only 5 seats!! Registration fee 499 refundable!! WhatsApp or email for details",
        "title": "Easy Money",
        "company": "Unknown"
    }

    result = detector.predict(
        text=fake_job["text"],
        title=fake_job["title"],
        company=fake_job["company"]
    )

    print("Prediction - FAKE Job")
    print("-" * 70)
    print(f"Verdict: {result.prediction}")
    print(f"Confidence: {result.confidence}%")
    print(f"Risk Level: {result.risk_level}")
    print(f"Risk Flags: {len(result.risk_flags)}")
    for flag in result.risk_flags:
        print(f"  - {flag}")

if __name__ == "__main__":
    test_fake_job()