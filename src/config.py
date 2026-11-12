"""Shared paths and model settings used by training, inference and the app."""

from pathlib import Path

# Paths are resolved from this file, not from the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "final_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES  # exact column order the Pipeline expects
TARGET = "Machine failure"

# Identifiers and failure-type labels must never be model inputs (leakage).
EXCLUDED_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]

# Same split as notebooks 02-04.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Selected in notebooks/04_model_iyilestirme.ipynb using the validation set only.
DECISION_THRESHOLD = 0.40
