"""Clean the tourism dataset and produce train / test splits."""

import os

import pandas as pd
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


HF_USERNAME = os.getenv("HF_USERNAME", "prashanth-merwyn")
REPO_ID = f"{HF_USERNAME}/wellness-tourism-dataset"
DATASET_PATH = f"hf://datasets/{REPO_ID}/tourism.csv"

TARGET_COL = "ProdTaken"
DROP_COLS = ["Unnamed: 0", "CustomerID"]

NUMERIC_COLS = [
    "Age",
    "DurationOfPitch",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]
CATEGORICAL_COLS = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

api = HfApi(token=os.getenv("HF_TOKEN"))

print(f"Loading dataset from {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
print(f"Raw shape: {df.shape}")

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

for col in NUMERIC_COLS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

for col in CATEGORICAL_COLS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode().iloc[0])

print(f"Cleaned shape: {df.shape}")
print(f"Class balance: {df[TARGET_COL].value_counts().to_dict()}")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

for path in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"Uploaded {path}")

print("Data preparation complete.")
