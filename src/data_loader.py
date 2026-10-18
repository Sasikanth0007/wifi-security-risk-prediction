"""
Data Loader and Integrity Module for Wi-Fi Security Risk Prediction.

Dataset: UAVIDS-2025 (IEEE 802.11ac Wireless Network Intrusion and Security Risk Benchmark)
DOI: 10.5281/zenodo.15336998
License: See the Zenodo record for current terms; the dataset is not redistributed by this project.
Authors: Qingli Zeng, Abdalrahman Bashir, Farid Nait-Abdesselam (2025)
"""

import os
import requests
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any

ZENODO_RECORD_URL = "https://zenodo.org/api/records/15336998/files/UAVIDS-2025.csv/content"
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEFAULT_DATA_FILE = os.path.join(DEFAULT_DATA_DIR, "UAVIDS-2025.csv")
SYNTHETIC_DATA_FILE = os.path.join(DEFAULT_DATA_DIR, "sample_synthetic_wifi_dev.csv")

TARGET_COLUMN = "label"

# Network flow behavioral features under IEEE 802.11ac wireless conditions
FEATURE_COLUMNS = [
    "FlowDuration/s",
    "TxPackets",
    "RxPackets",
    "LostPackets",
    "TxBytes",
    "RxBytes",
    "TxPacketRate/s",
    "RxPacketRate/s",
    "TxByteRate/s",
    "RxByteRate/s",
    "MeanDelay/s",
    "MeanJitter/s",
    "Throughput/Kbps",
    "MeanPacketSize",
    "PacketDropRate",
    "AverageHopCount"
]

# Non-generalizable identifiers / static metadata explicitly excluded to prevent data leakage and memorization
EXCLUDED_COLUMNS = [
    "FlowID",
    "SrcAddr",
    "SrcPort",
    "DstAddr",
    "DstPort",
    "Protocol"  # Constant UDP across this dataset
]

# Risk category severity mapping for security operations
RISK_CATEGORY_MAP = {
    "Normal Traffic": "Low Risk (Normal Operation)",
    "Flooding Attack": "High Risk (DoS / Buffer Exhaustion)",
    "Sybil Attack": "High Risk (Identity Impersonation)",
    "Blackhole Attack": "Critical Risk (Routing Disruption / Packet Drop)",
    "Wormhole Attack": "Critical Risk (Unauthorized Tunnel Diversion)"
}


def download_dataset(destination_path: str = DEFAULT_DATA_FILE, chunk_size: int = 1024 * 1024) -> str:
    """
    Downloads the UAVIDS-2025 benchmark dataset from Zenodo if not present.
    """
    if os.path.exists(destination_path) and os.path.getsize(destination_path) > 1024 * 1024:
        return destination_path

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    print(f"Downloading dataset from {ZENODO_RECORD_URL} to {destination_path}...")

    response = requests.get(ZENODO_RECORD_URL, stream=True, timeout=90)
    response.raise_for_status()

    total_downloaded = 0
    with open(destination_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                total_downloaded += len(chunk)

    print(f"Dataset successfully downloaded ({total_downloaded / (1024 * 1024):.2f} MB).")
    return destination_path


def create_synthetic_dev_sample(output_path: str = SYNTHETIC_DATA_FILE, n_samples: int = 250) -> str:
    """
    Creates a small, clearly labeled synthetic development sample with the identical schema.
    Used exclusively for offline testing, CI pipelines, and rapid unit testing.
    NEVER to be confused with real-world validation.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rng = np.random.default_rng(seed=42)

    labels = ["Normal Traffic", "Flooding Attack", "Blackhole Attack", "Sybil Attack", "Wormhole Attack"]
    sampled_labels = rng.choice(labels, size=n_samples, p=[0.22, 0.18, 0.20, 0.20, 0.20])

    data = {
        "FlowID": np.arange(1, n_samples + 1),
        "FlowDuration/s": np.round(rng.uniform(0.1, 1200.0, size=n_samples), 4),
        "SrcAddr": [f"10.0.0.{rng.integers(1, 254)}" for _ in range(n_samples)],
        "SrcPort": rng.integers(1024, 65535, size=n_samples),
        "DstAddr": [f"10.0.0.{rng.integers(1, 254)}" for _ in range(n_samples)],
        "DstPort": rng.integers(1024, 65535, size=n_samples),
        "Protocol": ["UDP"] * n_samples,
        "TxPackets": rng.integers(1, 5000, size=n_samples),
        "RxPackets": rng.integers(0, 4500, size=n_samples),
        "LostPackets": rng.integers(0, 500, size=n_samples),
        "TxBytes": rng.integers(30, 500000, size=n_samples),
        "RxBytes": rng.integers(0, 450000, size=n_samples),
        "TxPacketRate/s": np.round(rng.uniform(0.01, 100.0, size=n_samples), 4),
        "RxPacketRate/s": np.round(rng.uniform(0.0, 90.0, size=n_samples), 4),
        "TxByteRate/s": np.round(rng.uniform(1.0, 10000.0, size=n_samples), 4),
        "RxByteRate/s": np.round(rng.uniform(0.0, 9000.0, size=n_samples), 4),
        "MeanDelay/s": np.round(rng.uniform(0.001, 2.0, size=n_samples), 6),
        "MeanJitter/s": np.round(rng.uniform(0.0001, 1.5, size=n_samples), 6),
        "Throughput/Kbps": np.round(rng.uniform(0.0, 80.0, size=n_samples), 4),
        "MeanPacketSize": rng.integers(30, 80, size=n_samples),
        "PacketDropRate": np.round(rng.uniform(0.0, 1.0, size=n_samples), 6),
        "AverageHopCount": np.round(rng.uniform(0.0, 5.0, size=n_samples), 4),
        "label": sampled_labels
    }

    df_synth = pd.DataFrame(data)
    df_synth.to_csv(output_path, index=False)
    print(f"Created synthetic development dataset at {output_path} ({n_samples} samples).")
    return output_path


def validate_data_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates data integrity: missing values, feature presence, data types, and potential leakage.
    """
    report = {
        "is_valid": True,
        "missing_values": {},
        "inf_values": {},
        "missing_features": [],
        "has_target": TARGET_COLUMN in df.columns,
        "class_distribution": {},
        "warnings": []
    }

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        report["is_valid"] = False
        report["missing_features"] = missing_cols
        report["warnings"].append(f"Missing required feature columns: {missing_cols}")

    if not report["has_target"]:
        report["is_valid"] = False
        report["warnings"].append(f"Target column '{TARGET_COLUMN}' is missing.")
    else:
        report["class_distribution"] = df[TARGET_COLUMN].value_counts().to_dict()

    # Check nulls and infs in feature columns
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    null_counts = df[available_features].isnull().sum()
    report["missing_values"] = null_counts[null_counts > 0].to_dict()
    if report["missing_values"]:
        report["warnings"].append(f"Null values detected in columns: {report['missing_values']}")

    numeric_cols = df[available_features].select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    report["inf_values"] = inf_counts[inf_counts > 0].to_dict()
    if report["inf_values"]:
        report["warnings"].append(f"Infinite values detected in columns: {report['inf_values']}")

    return report


def load_dataset(
    filepath: Optional[str] = None,
    allow_download: bool = True,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, bool]:
    """
    Loads and validates the dataset.
    Returns (df, is_synthetic_flag).
    """
    is_synthetic = False

    # Determine filepath
    if filepath is None:
        filepath = DEFAULT_DATA_FILE

    # If primary dataset doesn't exist, attempt download or fallback to synthetic
    if not os.path.exists(filepath):
        if allow_download and filepath == DEFAULT_DATA_FILE:
            try:
                download_dataset(filepath)
            except Exception as e:
                print(f"Warning: Download failed ({e}). Falling back to synthetic dev data.")
                if not os.path.exists(SYNTHETIC_DATA_FILE):
                    create_synthetic_dev_sample(SYNTHETIC_DATA_FILE)
                filepath = SYNTHETIC_DATA_FILE
                is_synthetic = True
        else:
            if not os.path.exists(SYNTHETIC_DATA_FILE):
                create_synthetic_dev_sample(SYNTHETIC_DATA_FILE)
            filepath = SYNTHETIC_DATA_FILE
            is_synthetic = True

    if "synthetic" in os.path.basename(filepath).lower():
        is_synthetic = True

    df = pd.read_csv(filepath)

    integrity_report = validate_data_integrity(df)
    if not integrity_report["is_valid"]:
        raise ValueError(f"Data integrity validation failed: {integrity_report['warnings']}")

    if sample_size is not None and sample_size < len(df):
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df[TARGET_COLUMN],
            random_state=random_state
        )
        df = df.reset_index(drop=True)

    return df, is_synthetic
