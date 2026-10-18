"""
Preprocessing and Data Transformation Pipeline for Wi-Fi Security Risk Prediction.

Ensures strict zero-leakage standards:
- Stratified train/test partitioning prior to any fitting.
- Transformers fitted exclusively on training instances.
- Robust label encoding and feature metadata extraction for UI validation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN


class WiFiDataPreprocessor:
    """
    Manages data splitting, transformation, and feature metadata computation.
    """
    def __init__(self, feature_cols: List[str] = FEATURE_COLUMNS, target_col: str = TARGET_COLUMN):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.label_encoder = LabelEncoder()
        self.feature_metadata: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset into stratified train and test sets, then fits transformers ONLY on train.
        Returns:
            X_train_scaled, X_test_scaled, y_train, y_test, X_train_raw_df, X_test_raw_df
        """
        # Extract features and targets
        X = df[self.feature_cols].copy()
        y_raw = df[self.target_col].copy()

        # Stratified train/test split to avoid leakage
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X, y_raw, test_size=test_size, stratify=y_raw, random_state=random_state
        )

        # Fit label encoder exclusively on training labels
        self.label_encoder.fit(y_train_raw)
        y_train = self.label_encoder.transform(y_train_raw)
        y_test = self.label_encoder.transform(y_test_raw)

        # Compute feature statistics strictly from training data for validation/UI bounds
        self.feature_metadata = {}
        for col in self.feature_cols:
            vals = X_train_raw[col].astype(float)
            self.feature_metadata[col] = {
                "min": float(vals.min()),
                "max": float(vals.max()),
                "mean": float(vals.mean()),
                "std": float(vals.std()) if vals.std() > 0 else 1.0,
                "median": float(vals.median()),
                "q25": float(vals.quantile(0.25)),
                "q75": float(vals.quantile(0.75))
            }

        # Impute and scale strictly fitted on training features
        X_train_imp = self.imputer.fit_transform(X_train_raw)
        X_train_scaled = self.scaler.fit_transform(X_train_imp)

        # Transform test data without re-fitting
        X_test_imp = self.imputer.transform(X_test_raw)
        X_test_scaled = self.scaler.transform(X_test_imp)

        self.is_fitted = True
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train_raw, X_test_raw

    def transform_single(self, input_dict: Dict[str, Any]) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Transforms a single raw feature dictionary into scaled numpy array and single-row DataFrame.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming new samples.")

        # Ensure correct column ordering
        row = [float(input_dict.get(c, self.feature_metadata[c]["median"])) for c in self.feature_cols]
        df_row = pd.DataFrame([row], columns=self.feature_cols)

        row_imp = self.imputer.transform(df_row)
        row_scaled = self.scaler.transform(row_imp)

        return row_scaled, df_row

    def transform_batch(self, df_new: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Transforms a batch DataFrame into scaled numpy array.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming new samples.")

        df_ordered = df_new[self.feature_cols].copy().astype(float)
        batch_imp = self.imputer.transform(df_ordered)
        batch_scaled = self.scaler.transform(batch_imp)
        return batch_scaled, df_ordered

    def get_classes(self) -> List[str]:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor is not fitted.")
        return list(self.label_encoder.classes_)
