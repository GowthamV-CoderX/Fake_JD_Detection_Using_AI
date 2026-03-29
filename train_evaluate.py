"""
src/train_evaluate.py
Full training pipeline with evaluation metrics, confusion matrix, and test examples.
Run: python -m src.train_evaluate
"""

from __future__ import annotations
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import generate_dataset
from hybrid_detector import HybridDetector

RESULTS_DIR = Path("results")
MODEL_DIR = Path("models")


def load_or_generate_data(csv_path: str = "data/raw/jd_dataset.csv"):
    p = Path(csv_path)
    if not p.exists():
        print("Generating synthetic dataset…")
        df = generate_dataset(csv_path)
    else:
        df = pd.read_csv(csv_path)
    return df


def evaluate_model(detector: HybridDetector, X_test: list, y_test: list) -> dict:
    print("\n[Evaluation] Running predictions on test set…")
    y_pred = []
    y_proba = []

    for text in X_test:
        result = detector.predict(text)
        y_pred.append(1 if result.prediction == "FAKE" else 0)
        y_proba.append(result.scores["composite"])

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy":  round(acc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1_score":  round(f1, 4),
        "roc_auc":   round(auc, 4),
        "confusion_matrix": cm.tolist(),
    }

    print("\n" + "="*55)
    print("  EVALUATION METRICS")
    print("="*55)
    print(f"  Accuracy  : {acc:.2%}")
    print(f"  Precision : {prec:.2%}  (of predicted FAKE, how many are truly fake)")
    print(f"  Recall    : {rec:.2%}  (of all FAKE JDs, how many did we catch)")
    print(f"  F1-Score  : {f1:.2%}")
    print(f"  ROC-AUC   : {auc:.2%}")
    print("="*55)

    print("\nConfusion Matrix:")
    print("              Predicted REAL  Predicted FAKE")
    print(f"  Actual REAL       {cm[0][0]:4d}            {cm[0][1]:4d}")
    print(f"  Actual FAKE       {cm[1][0]:4d}            {cm[1][1]:4d}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

    return metrics


def run_demo_predictions(detector: HybridDetector):
    """Show sample predictions with full explanations."""
    print("\n" + "="*55)
    print("  DEMO PREDICTIONS")
    print("="*55)

    demo_jds = [
        {
            "label": "FAKE (obvious scam)",
            "text": (
                "URGENT HIRING!! Work From Home!! 🔥 Earn ₹1,00,000 per week "
                "easily!! Anyone can do it!! No experience needed!! "
                "HURRY only 5 seats left!! Registration fee ₹499 refundable!! "
                "WhatsApp 9XXXXXXXXX or email earnnow@gmail.com"
            )
        },
        {
            "label": "REAL (professional JD)",
            "text": (
                "Infosys is hiring a Senior Software Engineer for our Cloud "
                "Platform team in Pune. You will design scalable microservices "
                "using Python and Go. Requirements: 4+ years backend experience, "
                "AWS/Kubernetes proficiency, B.Tech in CS. "
                "CTC: ₹18–28 LPA. Apply: careers.infosys.com"
            )
        },
        {
            "label": "BORDERLINE (subtle scam)",
            "text": (
                "Looking for part-time digital marketing executives. "
                "Work from home, earn ₹25,000/month. No specific qualifications. "
                "Must purchase starter kit ₹1,500 (refundable). "
                "Contact: bizjobs2024@gmail.com"
            )
        },
    ]

    for demo in demo_jds:
        result = detector.predict(demo["text"])
        print(f"\n[{demo['label']}]")
        print("-" * 45)
        print(f"Prediction  : {result.prediction}")
        print(f"Confidence  : {result.confidence}%")
        print(f"Risk Level  : {result.risk_level}")
        print(f"Risk Flags  : {result.risk_flags}")
        print(f"Explanation : {result.explanation[:200]}…")
        print(f"Scores      : {result.scores}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_or_generate_data()
    print(f"\nDataset: {len(df)} samples | REAL: {(df['label']==0).sum()} | FAKE: {(df['label']==1).sum()}")

    texts = df["description"].tolist()
    labels = df["label"].tolist()

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # 3. Train
    # Try with embeddings; fall back to rule-only if sentence-transformers unavailable
    try:
        from sentence_transformers import SentenceTransformer
        use_embeddings = True
        print("\n[Info] sentence-transformers available – full hybrid mode enabled.")
    except ImportError:
        use_embeddings = False
        print("\n[Warning] sentence-transformers not installed – running without embedding layer.")

    detector = HybridDetector(use_embeddings=use_embeddings)
    detector.train(X_train, y_train)

    # 4. Evaluate
    metrics = evaluate_model(detector, X_test, y_test)

    # 5. Save metrics
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Saved] Metrics → {RESULTS_DIR}/metrics.json")

    # 6. Save model
    detector.save(str(MODEL_DIR))

    # 7. Demo
    run_demo_predictions(detector)

    return detector, metrics


if __name__ == "__main__":
    main()
