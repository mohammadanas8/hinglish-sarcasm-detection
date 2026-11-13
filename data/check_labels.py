import pandas as pd

df = pd.read_csv("data/preprocessed final.csv")

print("Missing values:")
print(df[["text", "label"]].isnull().sum())

print("\nUnique labels:")
print(sorted(df["label"].dropna().unique()))

print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())

print("\nDuplicate original texts:")
print(df["text"].duplicated().sum())

print("\nDataset validation completed.")