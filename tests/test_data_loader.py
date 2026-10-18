"""
Tests for Data Loader, Validation, and Integrity.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    validate_data_integrity,
    create_synthetic_dev_sample,
    load_dataset,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)


def test_synthetic_dev_sample_generation(tmp_path):
    output_csv = os.path.join(tmp_path, "test_synthetic.csv")
    path = create_synthetic_dev_sample(output_csv, n_samples=50)
    assert os.path.exists(path)

    df = pd.read_csv(path)
    assert len(df) == 50
    assert TARGET_COLUMN in df.columns
    for col in FEATURE_COLUMNS:
        assert col in df.columns


def test_data_integrity_validation_success():
    # Valid synthetic batch
    rng = np.random.default_rng(42)
    data = {c: rng.uniform(0, 100, size=20) for c in FEATURE_COLUMNS}
    data[TARGET_COLUMN] = ["Normal Traffic"] * 10 + ["Flooding Attack"] * 10
    df = pd.DataFrame(data)

    report = validate_data_integrity(df)
    assert report["is_valid"] is True
    assert len(report["missing_features"]) == 0
    assert report["has_target"] is True


def test_data_integrity_validation_missing_feature():
    # DataFrame missing some required features
    df = pd.DataFrame({"FlowDuration/s": [1.0, 2.0], TARGET_COLUMN: ["Normal Traffic", "Normal Traffic"]})
    report = validate_data_integrity(df)
    assert report["is_valid"] is False
    assert len(report["missing_features"]) > 0


def test_data_integrity_validation_missing_target():
    data = {c: [1.0, 2.0] for c in FEATURE_COLUMNS}
    df = pd.DataFrame(data)
    report = validate_data_integrity(df)
    assert report["is_valid"] is False
    assert report["has_target"] is False


def test_load_dataset_fallback_to_synthetic(tmp_path):
    # Non-existent path without download will fallback to synthetic
    non_existent = os.path.join(tmp_path, "missing_data.csv")
    df, is_synthetic = load_dataset(filepath=non_existent, allow_download=False)
    assert is_synthetic is True
    assert len(df) > 0
    assert TARGET_COLUMN in df.columns
