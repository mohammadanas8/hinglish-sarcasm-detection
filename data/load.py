import pandas as pd
from pathlib import Path

# Locate the dataset
file_path = Path("data") / "preprocessed final.csv"

# Load the CSV
df = pd.read_csv(file_path)

# Display dataset information
print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())
