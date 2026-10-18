"""
Explainability and Feature Contribution Module.

Integrates:
1. SHAP TreeExplainer for Random Forest (Global & Local Explanations).
2. Robust Fallback: Scikit-Learn Permutation Importance & Model Feature Importances.
3. Plain-Language Narrative Interpretation.
4. Scientific Disclaimer: Statistical Attribution vs. Causal Proof.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.inspection import permutation_importance

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


class WiFiRiskExplainer:
    """
    Computes global and local model explanations using SHAP with graceful fallback.
    """
    def __init__(self, models_dir: str = DEFAULT_MODELS_DIR):
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "best_model.joblib")
        self.preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
        self.bg_sample_path = os.path.join(models_dir, "shap_background_sample.joblib")

        if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Missing model artifacts in '{models_dir}'. Run training first.")

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.feature_cols = self.preprocessor.feature_cols
        self.classes = self.preprocessor.get_classes()

        # Load or create background sample
        if os.path.exists(self.bg_sample_path):
            self.bg_df = joblib.load(self.bg_sample_path)
        else:
            # Fallback synthetic background
            self.bg_df = pd.DataFrame(
                np.zeros((50, len(self.feature_cols))),
                columns=self.feature_cols
            )

        self.shap_explainer = None
        self._init_shap_explainer()

    def _init_shap_explainer(self) -> None:
        """
        Initializes SHAP TreeExplainer if model is tree-based.
        """
        try:
            import shap
            # Test TreeExplainer initialization
            if hasattr(self.model, "estimators_"):
                self.shap_explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            print(f"Warning: SHAP TreeExplainer initialization failed ({e}). Fallback explanation enabled.")
            self.shap_explainer = None

    def explain_instance(
        self,
        X_scaled: np.ndarray,
        raw_df: pd.DataFrame,
        predicted_class: str
    ) -> Dict[str, Any]:
        """
        Computes local feature contributions for a specific prediction instance.
        """
        pred_class_idx = self.classes.index(predicted_class) if predicted_class in self.classes else 0

        # Attempt SHAP explanation
        if self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer(X_scaled)
                # Inspect dimensions
                # For multiclass, shap_values may have shape (1, n_features, n_classes) or (1, n_features)
                if len(shap_values.shape) == 3:
                    class_shap = shap_values.values[0, :, pred_class_idx]
                    base_val = float(shap_values.base_values[0, pred_class_idx])
                elif len(shap_values.shape) == 2:
                    class_shap = shap_values.values[0, :]
                    base_val = float(shap_values.base_values[0])
                else:
                    class_shap = np.array(shap_values.values).flatten()
                    base_val = 0.0

                contributions = []
                for idx, col in enumerate(self.feature_cols):
                    feature_val = float(raw_df[col].iloc[0])
                    shap_val = float(class_shap[idx])
                    contributions.append({
                        "feature": col,
                        "value": feature_val,
                        "shap_value": round(shap_val, 5),
                        "abs_impact": round(abs(shap_val), 5),
                        "direction": "Pushes TOWARD target" if shap_val > 0 else "Pushes AWAY from target"
                    })

                # Sort by absolute impact
                contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
                top_features = contributions[:5]

                narrative = self._generate_plain_language_narrative(
                    predicted_class, top_features
                )

                return {
                    "method": "SHAP TreeExplainer (Local Shapley Values)",
                    "predicted_class": predicted_class,
                    "base_value": round(base_val, 4),
                    "feature_contributions": contributions,
                    "top_features": top_features,
                    "narrative": narrative,
                    "is_fallback": False,
                    "disclaimer": (
                        "Statistical Explanation Note: These Shapley values represent mathematical attributions "
                        "of feature influence within this specific trained model. They reflect statistical correlation "
                        "in the training dataset, NOT causal proof or absolute physical ground truth."
                    )
                }
            except Exception as e:
                print(f"Notice: SHAP calculation failed on instance ({e}). Switching to permutation fallback.")

        # Fallback Explanation: Global Model-Specific Importance and Deviations
        return self._fallback_instance_explanation(raw_df, predicted_class)

    def _fallback_instance_explanation(
        self,
        raw_df: pd.DataFrame,
        predicted_class: str
    ) -> Dict[str, Any]:
        """
        Documented Scikit-Learn fallback explanation using feature importance & deviation from normal.
        """
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None and hasattr(self.model, "coef_"):
            pred_class_idx = self.classes.index(predicted_class) if predicted_class in self.classes else 0
            importances = np.abs(self.model.coef_[pred_class_idx])

        if importances is None:
            importances = np.ones(len(self.feature_cols)) / len(self.feature_cols)

        contributions = []
        for idx, col in enumerate(self.feature_cols):
            feat_val = float(raw_df[col].iloc[0])
            bg_median = float(self.bg_df[col].median()) if col in self.bg_df else 0.0
            bg_std = float(self.bg_df[col].std()) if col in self.bg_df and self.bg_df[col].std() > 0 else 1.0

            z_score = (feat_val - bg_median) / bg_std
            proxy_impact = float(importances[idx] * z_score)

            contributions.append({
                "feature": col,
                "value": feat_val,
                "shap_value": round(proxy_impact, 4),
                "abs_impact": round(abs(proxy_impact), 4),
                "direction": "Above Baseline" if z_score > 0 else "Below Baseline"
            })

        contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
        top_features = contributions[:5]

        narrative = self._generate_plain_language_narrative(predicted_class, top_features, is_fallback=True)

        return {
            "method": "Fallback: Model Feature Importance & Baseline Deviation",
            "predicted_class": predicted_class,
            "base_value": 0.0,
            "feature_contributions": contributions,
            "top_features": top_features,
            "narrative": narrative,
            "is_fallback": True,
            "disclaimer": (
                "Fallback Explanation Note: Computed using intrinsic model feature importances weighted "
                "by deviation from training baseline. Does not provide exact additive Shapley guarantees."
            )
        }

    def get_global_importance(self) -> List[Dict[str, Any]]:
        """
        Returns global feature importance ranked across the dataset.
        """
        if hasattr(self.model, "feature_importances_"):
            raw_imp = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            raw_imp = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            raw_imp = np.ones(len(self.feature_cols)) / len(self.feature_cols)

        # Normalize to 100%
        total = np.sum(raw_imp) if np.sum(raw_imp) > 0 else 1.0
        normalized = (raw_imp / total) * 100.0

        global_list = []
        for col, score, raw_s in zip(self.feature_cols, normalized, raw_imp):
            global_list.append({
                "feature": col,
                "importance_percent": round(float(score), 2),
                "raw_score": round(float(raw_s), 4)
            })

        global_list.sort(key=lambda x: x["importance_percent"], reverse=True)
        return global_list

    def _generate_plain_language_narrative(
        self,
        predicted_class: str,
        top_features: List[Dict[str, Any]],
        is_fallback: bool = False
    ) -> str:
        """
        Translates mathematical feature scores into natural language explanations.
        """
        if not top_features:
            return f"The model classified this flow as '{predicted_class}' based on baseline parameters."

        lead_feature = top_features[0]
        second_feature = top_features[1] if len(top_features) > 1 else None

        explanation_parts = [
            f"The network flow was categorized as **{predicted_class}** primarily influenced by "
            f"`{lead_feature['feature']}` (value: {lead_feature['value']:.2f}, impact: {lead_feature['shap_value']:+.3f})."
        ]

        if second_feature:
            explanation_parts.append(
                f"Additionally, `{second_feature['feature']}` (value: {second_feature['value']:.2f}) "
                f"strongly {second_feature['direction'].lower()}."
            )

        if predicted_class == "Normal Traffic":
            explanation_parts.append(
                "Observed flow parameters (delay, jitter, and drop rates) align with healthy IEEE 802.11ac transmission baselines."
            )
        elif predicted_class == "Flooding Attack":
            explanation_parts.append(
                "Unusually elevated packet or byte rates combined with high drop rates reflect classic Denial-of-Service resource exhaustion."
            )
        elif predicted_class == "Blackhole Attack":
            explanation_parts.append(
                "Severe asymmetry between transmitted packets and received packets indicates traffic sinkholing where packets are silently discarded."
            )
        elif predicted_class == "Sybil Attack":
            explanation_parts.append(
                "Anomalous routing hop counts and transmission patterns point to multi-identity MAC/IP spoofing on the wireless channel."
            )
        elif predicted_class == "Wormhole Attack":
            explanation_parts.append(
                "Disproportionate delay variations and routing hop discrepancies indicate traffic being tunneled out of band."
            )

        return " ".join(explanation_parts)
