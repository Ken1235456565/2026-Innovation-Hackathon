"""
ClimaHealth AI — Disease Outbreak Prediction Module
=====================================================
Ensemble classifier that predicts disease outbreak risk from
climate features, historical patterns, and NLP signal scores.

Ensemble components:
1. Random Forest — captures nonlinear feature interactions
2. Gradient Boosting — sequential error correction
3. Logistic Regression — interpretable baseline

The ensemble uses weighted soft voting for final predictions.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, 
    f1_score, 
    roc_auc_score,
    mean_absolute_error,
    r2_score
)
import joblib
import os


# Feature columns used by the disease prediction model
FEATURE_COLS = [
    "temperature", "temp_anomaly", "precipitation", "humidity", "ndvi",
    "nlp_signal",
    "temp_lag1", "temp_lag2", "temp_lag3", "temp_lag4",
    "precip_lag1", "precip_lag2", "precip_lag3", "precip_lag4",
    "humidity_lag1", "humidity_lag2", "humidity_lag3", "humidity_lag4",
    "ndvi_lag1", "ndvi_lag2", "ndvi_lag3", "ndvi_lag4",
    "temp_rolling4", "temp_rolling8", "temp_rolling12",
    "precip_rolling4", "precip_rolling8", "precip_rolling12",
    "humidity_rolling4", "humidity_rolling8", "humidity_rolling12",
    "temp_change_4w", "precip_change_4w",
    "sin_week", "cos_week",
    "temp_x_precip", "temp_x_humidity",
]


class DiseasePredictor:
    """
    Ensemble model for disease outbreak prediction.
    Supports both:
    - Binary classification (outbreak yes/no)
    - Risk score regression (0-100 continuous score)
    """
    
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.scaler = StandardScaler()
        self.disease_encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_names = None
    
    def _get_feature_cols(self, df):
        """Get available feature columns from the dataframe."""
        available = [c for c in FEATURE_COLS if c in df.columns]
        return available
    
    def _prepare_data(self, df):
        """Prepare features and encode disease type."""
        feature_cols = self._get_feature_cols(df)
        self.feature_names = feature_cols
        
        X = df[feature_cols].copy()
        
        # Add encoded disease type as a feature
        if "disease" in df.columns:
            X["disease_encoded"] = self.disease_encoder.fit_transform(df["disease"])
            self.feature_names = feature_cols + ["disease_encoded"]
        
        return X
    
    def fit(self, df):
        """
        Train the ensemble model on the full dataset.
        
        Args:
            df: DataFrame with climate features, NLP signals, and target columns
                (must include 'outbreak' for classification, 'risk_score' for regression)
        """
        print("\n  Training Disease Prediction Ensemble...")
        
        X = self._prepare_data(df)
        
        # === CLASSIFICATION: Outbreak prediction ===
        y_class = df["outbreak"].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build ensemble
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        gb = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        lr = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=42
        )
        
        self.classifier = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
            voting="soft",
            weights=[2, 2, 1]  # RF and GB get higher weight
        )
        
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate classification
        y_pred = self.classifier.predict(X_test_scaled)
        y_prob = self.classifier.predict_proba(X_test_scaled)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"\n  === Classification Results ===")
        print(f"  F1 Score: {f1:.3f}")
        print(f"  AUC-ROC:  {auc:.3f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['No Outbreak', 'Outbreak'])}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.classifier, X_train_scaled, y_train, cv=5, scoring="f1")
        print(f"  5-Fold CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # === REGRESSION: Risk score prediction ===
        y_reg = df["risk_score"].values
        
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )
        
        X_train_r_scaled = self.scaler.transform(X_train_r)
        X_test_r_scaled = self.scaler.transform(X_test_r)
        
        self.regressor = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        self.regressor.fit(X_train_r_scaled, y_train_r)
        
        y_pred_r = self.regressor.predict(X_test_r_scaled)
        mae = mean_absolute_error(y_test_r, y_pred_r)
        r2 = r2_score(y_test_r, y_pred_r)
        
        print(f"\n  === Risk Score Regression ===")
        print(f"  MAE:  {mae:.2f}")
        print(f"  R²:   {r2:.3f}")
        
        self.is_fitted = True
        
        return {
            "classification": {"f1": f1, "auc": auc, "cv_f1_mean": cv_scores.mean()},
            "regression": {"mae": mae, "r2": r2}
        }
    
    def predict_outbreak(self, features_df):
        """Predict outbreak probability for new data."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        
        X = features_df[self.feature_names].copy() if all(f in features_df.columns for f in self.feature_names) else self._prepare_data(features_df)
        X_scaled = self.scaler.transform(X)
        
        proba = self.classifier.predict_proba(X_scaled)[:, 1]
        binary = self.classifier.predict(X_scaled)
        
        return {
            "outbreak_probability": proba,
            "outbreak_predicted": binary,
        }
    
    def predict_risk_score(self, features_df):
        """Predict continuous risk score (0-100)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        
        X = features_df[self.feature_names].copy() if all(f in features_df.columns for f in self.feature_names) else self._prepare_data(features_df)
        X_scaled = self.scaler.transform(X)
        
        scores = self.regressor.predict(X_scaled)
        return np.clip(np.round(scores), 0, 100).astype(int)
    
    def get_feature_importance(self, model_type="ensemble"):
        """
        Get feature importance from the ensemble.
        
        This serves as our SHAP-like explainability layer.
        In production, we'd use actual SHAP values for per-prediction explanations.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        
        importances = {}
        
        # Get from Random Forest
        rf = self.classifier.named_estimators_["rf"]
        rf_imp = dict(zip(self.feature_names, rf.feature_importances_))
        
        # Get from Gradient Boosting
        gb = self.classifier.named_estimators_["gb"]
        gb_imp = dict(zip(self.feature_names, gb.feature_importances_))
        
        # Get from Logistic Regression (absolute coefficients)
        lr = self.classifier.named_estimators_["lr"]
        lr_imp = dict(zip(self.feature_names, np.abs(lr.coef_[0])))
        lr_total = sum(lr_imp.values())
        lr_imp = {k: v / lr_total for k, v in lr_imp.items()}
        
        # Weighted average matching voting weights
        for feat in self.feature_names:
            importances[feat] = (
                2 * rf_imp.get(feat, 0) + 
                2 * gb_imp.get(feat, 0) + 
                1 * lr_imp.get(feat, 0)
            ) / 5
        
        # Normalize
        total = sum(importances.values())
        importances = {k: round(v / total, 4) for k, v in importances.items()}
        
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    def get_shap_summary(self):
        """
        Generate a SHAP-like summary by grouping features into interpretable categories.
        This maps the raw feature importances to the categories shown in the dashboard.
        """
        raw = self.get_feature_importance()
        
        categories = {
            "temperature": ["temperature", "temp_anomaly", "temp_lag1", "temp_lag2",
                          "temp_lag3", "temp_lag4", "temp_rolling4", "temp_rolling8",
                          "temp_rolling12", "temp_change_4w", "temp_x_precip", "temp_x_humidity"],
            "precipitation": ["precipitation", "precip_lag1", "precip_lag2", "precip_lag3",
                            "precip_lag4", "precip_rolling4", "precip_rolling8",
                            "precip_rolling12", "precip_change_4w"],
            "humidity": ["humidity", "humidity_lag1", "humidity_lag2", "humidity_lag3",
                        "humidity_lag4", "humidity_rolling4", "humidity_rolling8",
                        "humidity_rolling12"],
            "seasonality": ["sin_week", "cos_week", "ndvi", "ndvi_lag1", "ndvi_lag2",
                          "ndvi_lag3", "ndvi_lag4"],
            "nlp_signals": ["nlp_signal"],
        }
        
        summary = {}
        for cat, features in categories.items():
            summary[cat] = round(sum(raw.get(f, 0) for f in features), 3)
        
        # Normalize
        total = sum(summary.values())
        summary = {k: round(v / total, 3) for k, v in summary.items()}
        
        return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))
    
    def save(self, path):
        """Save models to disk."""
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            "classifier": self.classifier,
            "regressor": self.regressor,
            "scaler": self.scaler,
            "disease_encoder": self.disease_encoder,
            "feature_names": self.feature_names,
        }, os.path.join(path, "disease_predictor.pkl"))
        print(f"  Disease models saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load models from disk."""
        data = joblib.load(os.path.join(path, "disease_predictor.pkl"))
        obj = cls()
        obj.classifier = data["classifier"]
        obj.regressor = data["regressor"]
        obj.scaler = data["scaler"]
        obj.disease_encoder = data["disease_encoder"]
        obj.feature_names = data["feature_names"]
        obj.is_fitted = True
        return obj
