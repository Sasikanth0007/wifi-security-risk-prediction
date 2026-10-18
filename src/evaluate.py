"""
Model Evaluation, Security Trade-off Analysis, and Metrics Reporting.

Computes and explains:
- Precision, Recall, F1-score (macro and weighted).
- Per-class confusion matrices and detection sensitivity.
- Security analysis: False Positives (Alert Fatigue) vs. False Negatives (Silent Breach Risk).
- Why accuracy is misleading in imbalanced network attack detection.
"""

import os
import json
import argparse
from typing import Dict, Any
import numpy as np

DEFAULT_METRICS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "training_metrics.json"
)


def load_evaluation_metrics(metrics_path: str = DEFAULT_METRICS_PATH) -> Dict[str, Any]:
    """
    Loads saved evaluation results. Raises FileNotFoundError if model has not been trained.
    """
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Metrics file not found at '{metrics_path}'. Train the model first via: python -m src.train"
        )
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_evaluation_report(metrics_path: str = DEFAULT_METRICS_PATH) -> None:
    """
    Prints a detailed, interview-grade evaluation summary.
    """
    data = load_evaluation_metrics(metrics_path)
    selected_model = data["selected_model"]
    results = data["results"]
    classes = data["classes"]

    print("=" * 80)
    print("WI-FI SECURITY RISK PREDICTION: MODEL EVALUATION & COMPARISON REPORT")
    print("=" * 80)
    print(f"Dataset: {data['dataset_name']} ({'SYNTHETIC DEV SAMPLE' if data.get('is_synthetic_training') else 'REAL BENCHMARK'})")
    if data.get("dataset_doi"):
        print(f"DOI/Reference: {data['dataset_doi']}")
    print(f"Wireless Standard: {data.get('wireless_standard', 'IEEE 802.11ac')}")
    print(f"Training Samples: {data['train_samples_count']:,} | Test Samples: {data['test_samples_count']:,}")
    print(f"Evaluation Timestamp: {data.get('trained_timestamp', 'N/A')}")
    print(f"Selected Champion Model: {selected_model}")
    print("-" * 80)

    # Comparative Summary Table
    print(f"{'Model Name':<22} | {'Test Acc':<9} | {'F1 (Macro)':<11} | {'F1 (Weighted)':<13} | {'CV F1 (Mean)':<12}")
    print("-" * 80)
    for model_name, res in results.items():
        is_sel = "*" if model_name == selected_model else " "
        acc = f"{res['accuracy'] * 100:.2f}%"
        f1_m = f"{res['f1_macro']:.4f}"
        f1_w = f"{res['f1_weighted']:.4f}"
        cv_f1 = f"{res.get('cv_f1_weighted_mean', 0.0):.4f}"
        print(f"{is_sel + model_name:<22} | {acc:<9} | {f1_m:<11} | {f1_w:<13} | {cv_f1:<12}")

    print("\n" + "=" * 80)
    print(f"DETAILED PER-CLASS METRICS: CHAMPION MODEL ({selected_model})")
    print("=" * 80)
    champ_res = results[selected_model]
    per_class = champ_res["per_class"]

    print(f"{'Class / Attack Vector':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Test Support':<12}")
    print("-" * 80)
    for c_name in classes:
        if c_name in per_class:
            c_data = per_class[c_name]
            p = f"{c_data['precision']:.4f}"
            r = f"{c_data['recall']:.4f}"
            f1 = f"{c_data['f1_score']:.4f}"
            s = f"{c_data['support']:,}"
            print(f"{c_name:<22} | {p:<10} | {r:<10} | {f1:<10} | {s:<12}")

    # Confusion Matrix
    print("\n" + "-" * 80)
    print("CONFUSION MATRIX (Rows: True Class, Columns: Predicted Class):")
    cm = np.array(champ_res["confusion_matrix"])
    header = "          " + "  ".join([f"{c[:6]:>7}" for c in classes])
    print(header)
    for idx, row in enumerate(cm):
        c_short = classes[idx][:8]
        row_str = "  ".join([f"{val:>7}" for val in row])
        print(f"{c_short:<10} {row_str}")

    # Honest Security Analysis
    print("\n" + "=" * 80)
    print("SECURITY ENGINEERING ANALYSIS: FALSE POSITIVES vs. FALSE NEGATIVES")
    print("=" * 80)
    print(
        "1. False Negatives (Attack misclassified as Normal Traffic):\n"
        "   - Impact: Critical security failure. The malicious packet flow continues unimpeded.\n"
        "   - E.g., An active Sybil attack (MAC spoofing) or Flooding DoS evades the intrusion filter.\n"
        "   - In wireless security operations, Recall on attack classes must be maximized.\n\n"
        "2. False Positives (Normal Traffic misclassified as an Attack):\n"
        "   - Impact: Operational friction and alert fatigue for SOC analysts.\n"
        "   - If automated mitigation (such as deauthenticating the client or blocking ports) is triggered,\n"
        "     legitimate users experience service disruption.\n\n"
        "3. Why Accuracy is a Supplementary Metric:\n"
        "   - In highly imbalanced network intrusion environments, a naive classifier predicting the majority\n"
        "     class could report high accuracy while completely failing to detect targeted zero-day attacks.\n"
        "   - Weighted and Macro F1-scores represent the true measure of detection balance across classes."
    )
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Wi-Fi Security Risk Models")
    parser.add_argument("--metrics-path", type=str, default=DEFAULT_METRICS_PATH, help="Path to training_metrics.json")
    args = parser.parse_args()
    print_evaluation_report(args.metrics_path)
