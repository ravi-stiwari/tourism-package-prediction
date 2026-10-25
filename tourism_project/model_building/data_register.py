"""
Data registration step.

The dataset lives inside the GitHub repository (tourism_project/data/), so no
external data store is needed. This script "registers" it by validating that
the expected columns are present and printing a short summary that the pipeline
log captures.
"""
import os
import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = {
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
}


def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: dataset not found at {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    columns = set(df.columns)

    missing = EXPECTED_COLUMNS - columns
    if missing:
        print("ERROR: the dataset is missing expected columns:", sorted(missing))
        sys.exit(1)

    print("Dataset registered successfully.")
    print("Path    :", DATA_PATH)
    print("Shape   :", df.shape)
    print("Columns :", list(df.columns))
    print("\nTarget distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts())
    print("\nMissing values per column:")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()
