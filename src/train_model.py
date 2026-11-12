"""Train and save the final Stage 3.1 model.

Run from the project root:
    python -m src.train_model

Steps:
1. Same 80/20 stratified split as the notebooks (random_state=42).
2. Fit the final Random Forest Pipeline on the 8,000 training rows only.
3. Evaluate once on the 2,000-row held-out test set (no tuning here) and
   check the result against Stage 3.1 (notebook 04).
4. Save the fitted Pipeline (joblib) and a small JSON metadata file.
"""

import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    DECISION_THRESHOLD,
    EXCLUDED_COLUMNS,
    FEATURES,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)

# Confusion-matrix counts reported by the notebooks; used only as a reproducibility check.
EXPECTED_COUNTS = {
    "Logistic Regression": {"TP": 7, "FP": 4, "TN": 1928, "FN": 61},     # notebook 02/03
    "Random Forest baseline": {"TP": 34, "FP": 4, "TN": 1928, "FN": 34},  # notebook 03
    "Random Forest @ 0.40": {"TP": 43, "FP": 9, "TN": 1923, "FN": 25},    # notebook 04
}


def build_random_forest() -> Pipeline:
    # Numeric columns pass through unchanged (trees do not need scaling); Type is one-hot encoded.
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ])),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])


def build_logistic_regression() -> Pipeline:
    # Stage 2 baseline, only re-evaluated here for the dashboard comparison.
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ])),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


def evaluate(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def load_split():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    X, y = df[FEATURES], df[TARGET]
    assert not (set(EXCLUDED_COLUMNS) | {TARGET}) & set(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return df, X_train, X_test, y_train, y_test


def feature_summary(X_train: pd.DataFrame) -> dict:
    # Training-data ranges: used for UI defaults and out-of-range warnings, not as hard limits.
    numeric = {
        col: {
            "min": float(X_train[col].min()),
            "median": float(X_train[col].median()),
            "max": float(X_train[col].max()),
        }
        for col in NUMERIC_FEATURES
    }
    type_counts = X_train["Type"].value_counts()
    return {
        "numeric": numeric,
        "type_categories": sorted(type_counts.index.tolist()),
        "type_most_frequent": str(type_counts.idxmax()),
    }


def main() -> None:
    df, X_train, X_test, y_train, y_test = load_split()
    print(f"Train: {len(X_train)} rows, test: {len(X_test)} rows (held out)")

    # Final model: fitted on the 8,000 training rows only.
    model = build_random_forest()
    model.fit(X_train, y_train)

    test_score = model.predict_proba(X_test)[:, 1]
    lr = build_logistic_regression().fit(X_train, y_train)

    # The RF baseline is the same fitted forest with sklearn's default decision (predict()).
    evaluation = {
        "Logistic Regression": evaluate(y_test, lr.predict(X_test)),
        "Random Forest baseline": evaluate(y_test, model.predict(X_test)),
        "Random Forest @ 0.40": evaluate(y_test, (test_score >= DECISION_THRESHOLD).astype(int)),
    }

    for name, expected in EXPECTED_COUNTS.items():
        got = {k: evaluation[name][k] for k in expected}
        if got != expected:
            raise SystemExit(
                f"{name}: {got} does not match the notebook result {expected}. "
                f"Check the scikit-learn version (this run: {sklearn.__version__}) and the data file."
            )
        print(f"{name:<24} {evaluation[name]}  -> matches notebooks")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # Reload check: the saved Pipeline must give exactly the same scores.
    reloaded = joblib.load(MODEL_PATH)
    assert np.array_equal(reloaded.predict_proba(X_test)[:, 1], test_score)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "RandomForestClassifier(random_state=42), default hyperparameters, no class_weight",
        "decision_threshold": DECISION_THRESHOLD,
        "threshold_selected_on": "validation split inside the training data (notebook 04)",
        "features": FEATURES,
        "target": TARGET,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "dataset": {
            "rows": len(df),
            "failures": int(df[TARGET].sum()),
            "non_failures": int((df[TARGET] == 0).sum()),
        },
        "feature_summary": feature_summary(X_train),
        "test_evaluation": evaluation,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model    -> {MODEL_PATH}")
    print(f"Saved metadata -> {METADATA_PATH}")


if __name__ == "__main__":
    main()
