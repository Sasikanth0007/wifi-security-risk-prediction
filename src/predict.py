"""
Inference and Prediction Interface for Wi-Fi Security Risk Assessment.

Provides:
- Strict schema and physical range validation.
- Multi-class threat classification and mapped Risk Severity Category.
- Calibrated class probabilities.
- Security disclaimer and limitation statement.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from src.data_loader import FEATURE_COLUMNS, RISK_CATEGORY_MAP

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


class WiFiRiskPredictor:
    """
    Loads trained artifacts and performs validated inferences.
    """
    def __init__(self, models_dir: str = DEFAULT_MODELS_DIR):
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "best_model.joblib")
        self.preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")

        if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(
                f"Model artifacts not found in '{models_dir}'. Please train the model first using: python -m src.train"
            )

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.feature_cols = self.preprocessor.feature_cols
        self.classes = self.preprocessor.get_classes()
        self.feature_meta = getattr(self.preprocessor, "feature_metadata", {})

    def validate_inputs(self, input_dict: Dict[str, Any]) -> Tuple[Dict[str, float], List[str]]:
        """
        Validates input fields, handles type conversions, and checks range plausibility.
        Returns cleaned numeric dictionary and list of any validation warnings.
        """
        cleaned = {}
        warnings = []

        for col in self.feature_cols:
            if col not in input_dict or input_dict[col] is None or input_dict[col] == "":
                # Fallback to training median if missing
                fallback_val = self.feature_meta.get(col, {}).get("median", 0.0)
                cleaned[col] = float(fallback_val)
                warnings.append(f"Missing '{col}', imputed with training median ({fallback_val:.2f}).")
            else:
                try:
                    val = float(input_dict[col])
                    # Physical plausibility checks
                    if "Packets" in col or "Bytes" in col or "Duration" in col or "Rate" in col:
                        if val < 0:
                            warnings.append(f"Field '{col}' cannot be negative (got {val}). Adjusted to 0.0.")
                            val = 0.0
                    if col == "PacketDropRate":
                        if val < 0:
                            val = 0.0
                        elif val > 1.0:
                            warnings.append(f"PacketDropRate ({val}) exceeds 1.0. Clipped to 1.0.")
                            val = 1.0
                    cleaned[col] = val
                except (ValueError, TypeError):
                    fallback_val = self.feature_meta.get(col, {}).get("median", 0.0)
                    cleaned[col] = float(fallback_val)
                    warnings.append(f"Invalid non-numeric value for '{col}'. Imputed with median ({fallback_val:.2f}).")

        return cleaned, warnings

    def predict_single(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts threat category, risk severity, and class probabilities for a single flow record.
        """
        cleaned_inputs, validation_warnings = self.validate_inputs(input_dict)
        X_scaled, df_raw = self.preprocessor.transform_single(cleaned_inputs)

        # Inference
        pred_idx = int(self.model.predict(X_scaled)[0])
        pred_class = self.classes[pred_idx]
        risk_category = RISK_CATEGORY_MAP.get(pred_class, "Unknown Risk")

        # Probabilities
        probabilities = {}
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)[0]
            for c_name, p_val in zip(self.classes, probs):
                probabilities[c_name] = round(float(p_val), 4)

        return {
            "predicted_class": pred_class,
            "risk_category": risk_category,
            "probabilities": probabilities,
            "confidence": probabilities.get(pred_class, 1.0),
            "validated_features": cleaned_inputs,
            "raw_features_df": df_raw,
            "scaled_features": X_scaled,
            "validation_warnings": validation_warnings,
            "disclaimer": (
                "Notice: This prediction is a probabilistic estimate derived from offline machine learning "
                "models trained on the UAVIDS-2025 benchmark dataset. It does NOT guarantee that the live network "
                "is secure or compromised. Real-world security audits require active packet inspection, physical "
                "perimeter verification, and certified human analyst review."
            )
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs batch inference on a DataFrame of network flows.
        """
        X_scaled, _ = self.preprocessor.transform_batch(df)
        preds_idx = self.model.predict(X_scaled)
        pred_classes = [self.classes[i] for i in preds_idx]

        results_df = df.copy()
        results_df["predicted_threat"] = pred_classes
        results_df["predicted_risk_category"] = [RISK_CATEGORY_MAP.get(c, "Unknown Risk") for c in pred_classes]

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)
            for idx, c_name in enumerate(self.classes):
                results_df[f"prob_{c_name.replace(' ', '_')}"] = np.round(probs[:, idx], 4)

        return results_df


# Preset test profiles derived from actual median parameters of the UAVIDS-2025 dataset
PRESET_PROFILES = {
    "Normal Traffic (Baseline Flow)": {
        "FlowDuration/s": 1588.38,
        "TxPackets": 21.0,
        "RxPackets": 18.0,
        "LostPackets": 3.0,
        "TxBytes": 1292.0,
        "RxBytes": 1064.0,
        "TxPacketRate/s": 0.0240,
        "RxPacketRate/s": 0.0194,
        "TxByteRate/s": 1.2180,
        "RxByteRate/s": 0.9884,
        "MeanDelay/s": 0.0410,
        "MeanJitter/s": 0.0461,
        "Throughput/Kbps": 0.0079,
        "MeanPacketSize": 46.0,
        "PacketDropRate": 0.1667,
        "AverageHopCount": 0.0
    },
    "Flooding Attack (DoS Spike)": {
        "FlowDuration/s": 58.97,
        "TxPackets": 182.0,
        "RxPackets": 136.0,
        "LostPackets": 13.0,
        "TxBytes": 13832.0,
        "RxBytes": 10336.0,
        "TxPacketRate/s": 3.2572,
        "RxPacketRate/s": 2.5404,
        "TxByteRate/s": 247.5455,
        "RxByteRate/s": 193.0675,
        "MeanDelay/s": 0.0428,
        "MeanJitter/s": 0.0298,
        "Throughput/Kbps": 1.5445,
        "MeanPacketSize": 76.0,
        "PacketDropRate": 0.0496,
        "AverageHopCount": 0.0460
    },
    "Blackhole Attack (Packet Sinkhole)": {
        "FlowDuration/s": 261.77,
        "TxPackets": 26.0,
        "RxPackets": 7.0,
        "LostPackets": 17.0,
        "TxBytes": 1585.0,
        "RxBytes": 456.0,
        "TxPacketRate/s": 0.0983,
        "RxPacketRate/s": 0.0229,
        "TxByteRate/s": 5.4029,
        "RxByteRate/s": 1.3706,
        "MeanDelay/s": 0.1539,
        "MeanJitter/s": 0.1151,
        "Throughput/Kbps": 0.0110,
        "MeanPacketSize": 47.0,
        "PacketDropRate": 0.6951,
        "AverageHopCount": 0.0104
    },
    "Sybil Attack (Identity Spoofing)": {
        "FlowDuration/s": 633.51,
        "TxPackets": 896.0,
        "RxPackets": 800.0,
        "LostPackets": 90.0,
        "TxBytes": 68096.0,
        "RxBytes": 60800.0,
        "TxPacketRate/s": 1.4303,
        "RxPacketRate/s": 1.2763,
        "TxByteRate/s": 108.6990,
        "RxByteRate/s": 96.9980,
        "MeanDelay/s": 0.0258,
        "MeanJitter/s": 0.0264,
        "Throughput/Kbps": 0.7760,
        "MeanPacketSize": 76.0,
        "PacketDropRate": 0.1183,
        "AverageHopCount": 0.0203
    },
    "Wormhole Attack (Tunnel Diversion)": {
        "FlowDuration/s": 170.10,
        "TxPackets": 13.0,
        "RxPackets": 4.0,
        "LostPackets": 7.0,
        "TxBytes": 988.0,
        "RxBytes": 304.0,
        "TxPacketRate/s": 0.1011,
        "RxPacketRate/s": 0.0333,
        "TxByteRate/s": 7.6854,
        "RxByteRate/s": 2.5324,
        "MeanDelay/s": 0.2256,
        "MeanJitter/s": 0.1272,
        "Throughput/Kbps": 0.0203,
        "MeanPacketSize": 76.0,
        "PacketDropRate": 0.5714,
        "AverageHopCount": 0.2143
    }
}
