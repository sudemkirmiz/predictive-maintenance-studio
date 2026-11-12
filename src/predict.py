"""Load the saved model and make single predictions. No training happens here."""

import json
import math
from numbers import Real

import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from src.config import FEATURES, METADATA_PATH, MODEL_PATH, NUMERIC_FEATURES

TRAIN_COMMAND = "python -m src.train_model"

# Broad physical sanity limits. They catch typos and unit mistakes (e.g. a Celsius value
# typed into a Kelvin field) without restricting readings to the training min/max.
SANITY_LIMITS = {
    "Air temperature [K]": (200.0, 500.0),
    "Process temperature [K]": (200.0, 500.0),
    "Rotational speed [rpm]": (1.0, 10_000.0),
    "Torque [Nm]": (0.0, 500.0),
    "Tool wear [min]": (0.0, 1_000.0),
}


class ModelLoadError(Exception):
    """The model artifacts are missing, unreadable or incompatible."""


class InputValidationError(ValueError):
    """One or more user inputs are invalid."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def load_model_bundle(model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    """Return (pipeline, metadata). Only load joblib files produced locally by train_model.py."""
    if not model_path.exists() or not metadata_path.exists():
        raise ModelLoadError(f"Model dosyaları bulunamadı ({model_path.parent}). Çalıştırın: {TRAIN_COMMAND}")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pipeline = joblib.load(model_path)
    except Exception as exc:  # corrupted or truncated files
        raise ModelLoadError(
            f"Model dosyaları okunamadı ({type(exc).__name__}: {exc}). Çalıştırın: {TRAIN_COMMAND}"
        ) from exc

    if not isinstance(pipeline, Pipeline) or not hasattr(pipeline, "predict_proba"):
        raise ModelLoadError(f"{model_path.name} beklenen model nesnesini içermiyor. Çalıştırın: {TRAIN_COMMAND}")

    if list(getattr(pipeline, "feature_names_in_", [])) != FEATURES:
        raise ModelLoadError(f"Model farklı girdi sütunlarıyla eğitilmiş. Çalıştırın: {TRAIN_COMMAND}")

    # A pickled sklearn model is only reliable with the same sklearn version.
    if metadata.get("sklearn_version") != sklearn.__version__:
        raise ModelLoadError(
            f"Model scikit-learn {metadata.get('sklearn_version')} ile eğitilmiş, bu ortamda "
            f"{sklearn.__version__} var. Projenin .venv ortamını kullanın veya çalıştırın: {TRAIN_COMMAND}"
        )
    return pipeline, metadata


def validate_inputs(inputs: dict, metadata: dict) -> list[str]:
    """Raise InputValidationError for invalid values; return warnings for unusual ones."""
    errors, warnings = [], []
    summary = metadata["feature_summary"]

    missing = [f for f in FEATURES if f not in inputs]
    if missing:
        raise InputValidationError([f"Eksik girdi: {name}" for name in missing])

    if inputs["Type"] not in summary["type_categories"]:
        errors.append(f"Type şunlardan biri olmalı: {summary['type_categories']} (girilen: {inputs['Type']!r}).")

    for name in NUMERIC_FEATURES:
        value = inputs[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            errors.append(f"{name} sonlu bir sayı olmalı (girilen: {value!r}).")
            continue

        low, high = SANITY_LIMITS[name]
        if not low <= value <= high:
            errors.append(f"{name} = {value:g} makul aralığın dışında ({low:g} – {high:g}). Birimi kontrol edin.")
            continue

        observed = summary["numeric"][name]
        if not observed["min"] <= value <= observed["max"]:
            warnings.append(
                f"{name} = {value:g} eğitim verisi aralığının dışında "
                f"({observed['min']:g} – {observed['max']:g}); model skoru daha az güvenilirdir."
            )

    if errors:
        raise InputValidationError(errors)
    return warnings


def predict_failure(pipeline: Pipeline, metadata: dict, inputs: dict) -> dict:
    """Validate inputs, score one record with the saved Pipeline and apply the saved threshold."""
    warnings = validate_inputs(inputs, metadata)

    # One-row DataFrame with the exact column names and order used during training.
    row = pd.DataFrame([{name: inputs[name] for name in FEATURES}], columns=FEATURES)

    score = float(pipeline.predict_proba(row)[0, 1])  # model score for class 1 (failure)
    threshold = float(metadata["decision_threshold"])
    prediction = int(score >= threshold)

    return {
        "score": score,
        "threshold": threshold,
        "prediction": prediction,
        "label": "Failure" if prediction == 1 else "No Failure",
        "warnings": warnings,
    }
