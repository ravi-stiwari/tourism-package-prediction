"""
Data preparation step.

- Loads the registered dataset from the repository data folder.
- Cleans the data (fixes the "Fe Male" typo, drops identifier/index columns,
  imputes missing values as a safeguard).
- Splits the cleaned data into stratified train/test sets.
- Saves Xtrain/Xtest/ytrain/ytest as CSV files so the next pipeline job can
  pick them up as a workflow artifact.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Loaded data from {path} with shape {df.shape}")
    return df


def clean_data(df):
    # Drop the unnamed leading column (empty header; it is a row index that
    # equals CustomerID - 200000 in tourism.csv).
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    df = df.drop(columns=unnamed, errors="ignore")

    # CustomerID is a unique identifier and carries no predictive signal.
    df = df.drop(columns=["CustomerID"], errors="ignore")

    # Fix the "Fe Male" data-entry typo of the existing "Female" label.
    # Other categories (e.g. MaritalStatus "Unmarried" vs "Single") are kept
    # exactly as they appear in the data.
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # Impute missing values (median for numeric, mode for categorical).
    # Safeguard only: tourism.csv has no missing values, so nothing changes.
    print("Missing values before imputation:", int(df.isnull().sum().sum()))
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(exclude=["number"]).columns
    for c in num_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode()[0])

    print("After cleaning:", df.shape)
    print("Remaining missing values:", int(df.isnull().sum().sum()))
    return df


def main():
    df = load_data()
    df = clean_data(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save the splits at the repo root so the next job can consume them as an
    # uploaded artifact (matches the paths in .github/workflows/pipeline.yml).
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
    print("Train shape:", Xtrain.shape, "Test shape:", Xtest.shape)


if __name__ == "__main__":
    main()
