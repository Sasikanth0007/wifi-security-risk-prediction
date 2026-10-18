"""
Tests for Preprocessing and Zero-Leakage Pipeline.
"""

import pytest
import numpy as np
import pandas as pd

from src.preprocessing import WiFiDataPreprocessor
from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN


@pytest.fixture
def mock_dataset():
    rng = np.random.default_rng(42)
    n = 100
    data = {c: rng.uniform(1.0, 50.0, size=n) for c in FEATURE_COLUMNS}
    data[TARGET_COLUMN] = rng.choice(
        ["Normal Traffic", "Flooding Attack", "Blackhole Attack", "Sybil Attack", "Wormhole Attack"],
        size=n
    )
    return pd.DataFrame(data)


def test_preprocessor_fit_transform(mock_dataset):
    preprocessor = WiFiDataPreprocessor()
    X_train, X_test, y_train, y_test, X_train_raw, X_test_raw = preprocessor.fit_transform(
        mock_dataset, test_size=0.2, random_state=42
    )

    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20
    assert X_train.shape[1] == len(FEATURE_COLUMNS)
    assert X_test.shape[1] == len(FEATURE_COLUMNS)
    assert len(y_train) == 80
    assert len(y_test) == 20
    assert preprocessor.is_fitted is True

    # Check zero-leakage: training feature stats should match preprocessor feature_metadata
    for col in FEATURE_COLUMNS:
        assert col in preprocessor.feature_metadata
        assert "min" in preprocessor.feature_metadata[col]
        assert "max" in preprocessor.feature_metadata[col]
        assert "mean" in preprocessor.feature_metadata[col]


def test_preprocessor_transform_single(mock_dataset):
    preprocessor = WiFiDataPreprocessor()
    preprocessor.fit_transform(mock_dataset, test_size=0.2, random_state=42)

    sample_dict = {col: 10.0 for col in FEATURE_COLUMNS}
    X_single, df_single = preprocessor.transform_single(sample_dict)

    assert X_single.shape == (1, len(FEATURE_COLUMNS))
    assert df_single.shape == (1, len(FEATURE_COLUMNS))


def test_preprocessor_unfitted_raises_error():
    preprocessor = WiFiDataPreprocessor()
    with pytest.raises(RuntimeError):
        preprocessor.transform_single({"FlowDuration/s": 10.0})
