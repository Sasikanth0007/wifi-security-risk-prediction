"""
Tests for Explainability and SHAP Fallback Module.
"""

import pytest
from src.predict import WiFiRiskPredictor, PRESET_PROFILES
from src.explain import WiFiRiskExplainer


def test_explainer_initialization():
    explainer = WiFiRiskExplainer()
    assert explainer.model is not None
    assert explainer.preprocessor is not None
    assert len(explainer.feature_cols) > 0


def test_explainer_instance_explanation():
    predictor = WiFiRiskPredictor()
    explainer = WiFiRiskExplainer()

    sample = list(PRESET_PROFILES.values())[0]
    pred_res = predictor.predict_single(sample)

    exp_res = explainer.explain_instance(
        pred_res["scaled_features"],
        pred_res["raw_features_df"],
        pred_res["predicted_class"]
    )

    assert "method" in exp_res
    assert "top_features" in exp_res
    assert len(exp_res["top_features"]) <= 5
    assert "narrative" in exp_res
    assert isinstance(exp_res["narrative"], str)
    assert len(exp_res["narrative"]) > 10
    assert "disclaimer" in exp_res


def test_explainer_global_importance():
    explainer = WiFiRiskExplainer()
    global_imp = explainer.get_global_importance()

    assert len(global_imp) == len(explainer.feature_cols)
    assert "feature" in global_imp[0]
    assert "importance_percent" in global_imp[0]
    # Sum of normalized importance should be approximately 100%
    total_pct = sum(item["importance_percent"] for item in global_imp)
    assert 99.0 <= total_pct <= 101.0


def test_explainer_fallback_on_shap_disabled():
    explainer = WiFiRiskExplainer()
    # Force shap_explainer to None to test the fallback mechanism
    explainer.shap_explainer = None

    predictor = WiFiRiskPredictor()
    sample = list(PRESET_PROFILES.values())[0]
    pred_res = predictor.predict_single(sample)

    exp_res = explainer.explain_instance(
        pred_res["scaled_features"],
        pred_res["raw_features_df"],
        pred_res["predicted_class"]
    )

    assert exp_res["is_fallback"] is True
    assert "Fallback" in exp_res["method"]
    assert len(exp_res["top_features"]) > 0
    assert "narrative" in exp_res
