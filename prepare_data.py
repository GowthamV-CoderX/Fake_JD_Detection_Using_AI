import pandas as pd

print("Loading dataset...")

# correct path to your dataset
df = pd.read_csv("data/raw/balanced_jobs.csv")

df = df.fillna("")

df["text"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"]
)

df = df[["text", "fraudulent"]]
df = df.rename(columns={"fraudulent": "label"})

df = df[df["text"].str.strip().astype(bool)]
df = df.sample(frac=1, random_state=42)

# save cleaned file
df.to_csv("data/clean_jobs.csv", index=False)

print("✅ Clean dataset ready at data/clean_jobs.csv")
print(df.head())