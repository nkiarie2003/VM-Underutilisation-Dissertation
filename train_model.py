"""
Train and save the Random Forest model used by the Flask app.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Project folders
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "reference_labels.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "random_forest.joblib"


# Same 10 features used in the dissertation model
FEATURES = [
    "cpu_cores",
    "cpu_mean",
    "cpu_last_3_days_mean",
    "cpu_p95",
    "cpu_max",
    "cpu_below_5_pct",
    "memory_mean",
    "disk_mean",
    "outbound_network_mean",
    "network_mean",
]


def train():

    print("Loading VM data...")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df["reference_label"]

    # Same split used in the dissertation experiment
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Same Random Forest configuration
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    # Make sure the models folder exists
    MODEL_DIR.mkdir(exist_ok=True)

    # Save the model and feature list together
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
        },
        MODEL_PATH,
    )

    print("Model trained successfully.")
    print("Training VMs:", len(X_train))
    print("Testing VMs:", len(X_test))
    print("Saved to:", MODEL_PATH)

    return model


if __name__ == "__main__":
    train()