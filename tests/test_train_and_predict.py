"""
Tests for Training Pipeline and Prediction Interface.
"""

import os
import pytest
import pandas as pd

from src.train import train_and_compare_models
from src.predict import WiFiRiskPredictor, PRESET_PROFILES
from src.data_loader import FEATURE_COLUMNS


@pytest.fixture(scope="module")
def trained_artifacts_dir(tmp_path_factory):
    temp_dir = str(tmp_path_factory.mktemp("test_models"))
    # Run rapid training on synthetic data
    train_and_compare_models(sample_size=100, use_synthetic=True, models_dir=temp_dir, cv_folds=2)
    return temp_dir


def test_model_training_creates_artifacts(trained_artifacts_dir):
    expected_files = [
        "best_model.joblib",
        "preprocessor.joblib",
        "logistic_regression.joblib",
        "random_forest.joblib",
        "shap_background_sample.joblib",
        "training_metrics.json"
    ]
    for filename in expected_files:
        assert os.path.exists(os.path.join(trained_artifacts_dir, filename))


def test_predictor_single_inference(trained_artifacts_dir):
    predictor = WiFiRiskPredictor(models_dir=trained_artifacts_dir)
    preset_data = list(PRESET_PROFILES.values())[0]

    result = predictor.predict_single(preset_data)

    assert "predicted_class" in result
    assert "risk_category" in result
    assert "probabilities" in result
    assert result["predicted_class"] in predictor.classes
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


def test_predictor_handles_missing_and_invalid_inputs(trained_artifacts_dir):
    predictor = WiFiRiskPredictor(models_dir=trained_artifacts_dir)

    # Incomplete input with negative packet count
    bad_input = {
        "FlowDuration/s": -10.0,
        "TxPackets": -50,
        "PacketDropRate": 2.5
    }

    result = predictor.predict_single(bad_input)
    assert "predicted_class" in result
    assert len(result["validation_warnings"]) > 0


def test_predictor_batch_inference(trained_artifacts_dir):
    predictor = WiFiRiskPredictor(models_dir=trained_artifacts_dir)
    batch_rows = list(PRESET_PROFILES.values())
    df_batch = pd.DataFrame(batch_rows)

    res_df = predictor.predict_batch(df_batch)
    assert len(res_df) == len(batch_rows)
    assert "predicted_threat" in res_df.columns
    assert "predicted_risk_category" in res_df.columns
