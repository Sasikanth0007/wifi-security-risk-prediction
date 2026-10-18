"""
Model Training, Cross-Validation, Comparison, and Artifact Persistence.

Compares:
1. Baseline: Logistic Regression (Multinomial with L2 regularization)
2. Ensemble: Random Forest Classifier (Optimized depth, n_estimators=100)

Selects the champion model based on multi-class Weighted and Macro F1-score.
"""

import os
import json
import argparse
import time
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

from src.data_loader import load_dataset, DEFAULT_DATA_FILE, SYNTHETIC_DATA_FILE
from src.preprocessing import WiFiDataPreprocessor

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def evaluate_model_performance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list
) -> Dict[str, Any]:
    """
    Computes comprehensive test set evaluation metrics.
    """
    y_pred = model.predict(X_test)

    # Probabilities if supported
    has_proba = hasattr(model, "predict_proba")

    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred).tolist()

    # Per-class metrics
    class_rep = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    per_class_results = {}
    for c_name in class_names:
        if c_name in class_rep:
            per_class_results[c_name] = {
                "precision": float(class_rep[c_name]["precision"]),
                "recall": float(class_rep[c_name]["recall"]),
                "f1_score": float(class_rep[c_name]["f1-score"]),
                "support": int(class_rep[c_name]["support"])
            }

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "per_class": per_class_results,
        "supports_probabilities": has_proba
    }


def train_and_compare_models(
    sample_size: int = 25000,
    use_synthetic: bool = False,
    models_dir: str = DEFAULT_MODELS_DIR,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes complete training, cross-validation, comparison, and persistence pipeline.
    """
    os.makedirs(models_dir, exist_ok=True)

    data_path = SYNTHETIC_DATA_FILE if use_synthetic else DEFAULT_DATA_FILE
    print(f"\n--- 1. Loading Dataset (source: {'Synthetic Dev' if use_synthetic else 'UAVIDS-2025'}) ---")
    df, is_synthetic = load_dataset(
        filepath=data_path,
        sample_size=sample_size if not use_synthetic else None,
        random_state=random_state
    )
    print(f"Loaded {len(df)} samples across classes: {df['label'].value_counts().to_dict()}")

    print("\n--- 2. Preprocessing & Partitioning (Zero-Leakage Stratified Split) ---")
    preprocessor = WiFiDataPreprocessor()
    X_train_scaled, X_test_scaled, y_train, y_test, X_train_raw, X_test_raw = preprocessor.fit_transform(
        df, test_size=0.2, random_state=random_state
    )
    class_names = preprocessor.get_classes()
    print(f"Train size: {X_train_scaled.shape[0]} samples | Test size: {X_test_scaled.shape[0]} samples")
    print(f"Features: {len(preprocessor.feature_cols)} behavioral network flow attributes")
    print(f"Target classes ({len(class_names)}): {class_names}")

    # Initialize Candidate Models
    models = {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=random_state
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
    }

    results = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    print(f"\n--- 3. Model Training & {cv_folds}-Fold Stratified Cross-Validation ---")
    for name, model in models.items():
        print(f"\nEvaluating candidate model: {name}...")
        t0 = time.time()

        # Cross-validation on training fold only (zero test contamination)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)

        # Fit on full training set
        model.fit(X_train_scaled, y_train)
        fit_time = time.time() - t0

        # Evaluate on held-out test set
        test_eval = evaluate_model_performance(model, X_test_scaled, y_test, class_names)
        test_eval["cv_f1_weighted_mean"] = float(np.mean(cv_scores))
        test_eval["cv_f1_weighted_std"] = float(np.std(cv_scores))
        test_eval["fit_time_seconds"] = round(fit_time, 3)

        results[name] = test_eval
        print(f"  -> Fit Time: {fit_time:.2f}s")
        print(f"  -> CV F1-Score (Weighted): {test_eval['cv_f1_weighted_mean']:.4f} (+/- {test_eval['cv_f1_weighted_std']:.4f})")
        print(f"  -> Held-out Test Accuracy: {test_eval['accuracy']:.4f}")
        print(f"  -> Held-out Test F1 (Macro): {test_eval['f1_macro']:.4f} | F1 (Weighted): {test_eval['f1_weighted']:.4f}")

    # Model Selection: Champion is selected based on test weighted F1-score
    best_model_name = max(results.keys(), key=lambda k: results[k]["f1_weighted"])
    champion_model = models[best_model_name]
    print(f"\n--- 4. Champion Model Selection: {best_model_name} ---")
    print(f"Rationale: Achieved superior Weighted F1 ({results[best_model_name]['f1_weighted']:.4f}) "
          f"and Macro F1 ({results[best_model_name]['f1_macro']:.4f}) on held-out test data.")

    # Save Artifacts
    print("\n--- 5. Persisting Model Artifacts ---")
    joblib.dump(champion_model, os.path.join(models_dir, "best_model.joblib"))
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
    joblib.dump(models["Logistic_Regression"], os.path.join(models_dir, "logistic_regression.joblib"))
    joblib.dump(models["Random_Forest"], os.path.join(models_dir, "random_forest.joblib"))

    # Save representative background sample for SHAP explainability
    background_size = min(300, len(X_train_raw))
    bg_df = X_train_raw.sample(n=background_size, random_state=random_state)
    joblib.dump(bg_df, os.path.join(models_dir, "shap_background_sample.joblib"))

    # Metadata payload
    metadata = {
        "selected_model": best_model_name,
        "is_synthetic_training": is_synthetic,
        "dataset_name": "sample_synthetic_wifi_dev" if is_synthetic else "UAVIDS-2025",
        "dataset_doi": "10.5281/zenodo.15336998" if not is_synthetic else "Synthetic Local Development Only",
        "wireless_standard": "IEEE 802.11ac",
        "trained_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "train_samples_count": int(len(X_train_scaled)),
        "test_samples_count": int(len(X_test_scaled)),
        "features": preprocessor.feature_cols,
        "classes": class_names,
        "feature_metadata": preprocessor.feature_metadata,
        "results": results
    }

    metrics_path = os.path.join(models_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artifacts successfully saved to '{models_dir}':")
    print("  - best_model.joblib")
    print("  - preprocessor.joblib")
    print("  - logistic_regression.joblib")
    print("  - random_forest.joblib")
    print("  - shap_background_sample.joblib")
    print("  - training_metrics.json")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Wi-Fi Security Risk Prediction Models")
    parser.add_argument("--sample-size", type=int, default=25000, help="Stratified sample size (default: 25000; set -1 for full)")
    parser.add_argument("--use-synthetic", action="store_true", help="Use small synthetic development sample")
    parser.add_argument("--models-dir", type=str, default=DEFAULT_MODELS_DIR, help="Output models directory")
    args = parser.parse_args()

    s_size = None if args.sample_size <= 0 else args.sample_size
    train_and_compare_models(sample_size=s_size, use_synthetic=args.use_synthetic, models_dir=args.models_dir)
