import pandas as pd
from hybrid_detector import HybridDetector

print("Loading cleaned dataset...")

# 🔥 USE REAL DATASET (NOT MANUAL LISTS)
df = pd.read_csv("data/clean_jobs.csv")

texts = df["text"].tolist()
labels = df["label"].tolist()

print(f"Dataset size: {len(texts)}")

# sanity check
print("Fake samples:", sum(labels))
print("Real samples:", len(labels) - sum(labels))

detector = HybridDetector(use_embeddings=True)

print("Training model...")

detector.train(texts, labels)

detector.save("models")

print("✅ Training complete and models saved!")