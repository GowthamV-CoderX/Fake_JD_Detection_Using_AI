from datasets import load_dataset
import pandas as pd

print("Downloading dataset...")

dataset = load_dataset("gplsi/fake_job_postings_balanced_en")

df = dataset["train"].to_pandas()

df.to_csv("balanced_jobs.csv", index=False)

print("✅ Done. File saved as balanced_jobs.csv")
print(df.head())