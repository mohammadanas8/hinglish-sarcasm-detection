import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# =====================================================
# 1. PATH CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "preprocessed final.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. LOAD DATASET
# =====================================================

print("\nLoading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Check the filename inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)

# =====================================================
# 3. VALIDATE DATASET
# =====================================================

required_columns = {"text", "label"}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Keep original text and labels for transformers
df = df[["text", "label"]].copy()

# Remove records with missing text or labels
initial_count = len(df)

df = df.dropna(subset=["text", "label"])

print("Rows removed due to missing values:", initial_count - len(df))

# Normalize text field types
df["text"] = df["text"].astype(str).str.strip()

# Remove empty text records
df = df[df["text"] != ""]

# Validate binary labels
df["label"] = pd.to_numeric(df["label"], errors="raise")

if not df["label"].isin([0, 1]).all():
    raise ValueError("Labels must contain only 0 and 1.")

df["label"] = df["label"].astype(int)

# Stop if the dataset has exact duplicate texts
# We do not silently discard records here.
duplicate_count = df["text"].duplicated().sum()

print("Duplicate texts:", duplicate_count)

if duplicate_count > 0:
    raise ValueError(
        "Duplicate texts detected. Investigate them before splitting."
    )

if set(df["label"].unique()) != {0, 1}:
    raise ValueError("Both classes 0 and 1 must be present.")

print("Validated dataset shape:", df.shape)

# =====================================================
# 4. SPLIT: 80% TRAIN, 10% VALIDATION, 10% TEST
# =====================================================

print("\nCreating stratified splits...")

train_df, temporary_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

validation_df, test_df = train_test_split(
    temporary_df,
    test_size=0.50,
    random_state=42,
    stratify=temporary_df["label"]
)

# =====================================================
# 5. SAVE SPLITS
# =====================================================

train_path = OUTPUT_DIR / "train.csv"
validation_path = OUTPUT_DIR / "validation.csv"
test_path = OUTPUT_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
validation_df.to_csv(validation_path, index=False)
test_df.to_csv(test_path, index=False)

# =====================================================
# 6. DISPLAY RESULTS
# =====================================================

print("\n" + "=" * 55)
print("DATA SPLITTING COMPLETED")
print("=" * 55)

splits = {
    "TRAINING": train_df,
    "VALIDATION": validation_df,
    "TEST": test_df
}

for name, split_df in splits.items():

    print(f"\n{name} SET")
    print("-" * 35)

    print("Total samples:", len(split_df))

    print("\nLabel counts:")
    print(split_df["label"].value_counts().sort_index())

    print("\nLabel percentages:")
    print(
        (
            split_df["label"]
            .value_counts(normalize=True)
            .sort_index() * 100
        ).round(2)
    )

print("\nSaved files:")
print(train_path)
print(validation_path)
print(test_path)

print("\nFinal split sizes:")
print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))
print("Total:", len(train_df) + len(validation_df) + len(test_df))

print("\nDataset preparation finished successfully!")

