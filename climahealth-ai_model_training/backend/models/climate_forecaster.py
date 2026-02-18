"""
ClimaHealth AI — Climate Forecasting Module
=============================================
Time-series forecasting for temperature and precipitation.

In a full implementation, this would use:
- LSTM neural networks (PyTorch/TensorFlow) for nonlinear temporal patterns
- Facebook Prophet for decomposable time-series with seasonality

For the hackathon prototype, we use sklearn-based approaches that
capture the key dynamics: seasonality, trend, and autoregressive features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os


class ClimateForecaster:
    """
    Forecasts temperature and precipitation 1-8 weeks ahead.
    Uses Gradient Boosting with autoregressive features to capture
    temporal dynamics similar to what an LSTM would learn.
    """
    
    def __init__(self, forecast_horizon=8):
        self.forecast_horizon = forecast_horizon
        self.temp_models = {}   # One model per forecast horizon
        self.precip_models = {}
        self.scalers = {}
        self.is_fitted = False
    
    def _create_ar_features(self, series, lookback=12):
        """Create autoregressive features from a time series."""
        features = {}
        for lag in range(1, lookback + 1):
            features[f"lag_{lag}"] = series.shift(lag)
        
        # Rolling statistics
        for window in [4, 8, 12]:
            features[f"rolling_mean_{window}"] = series.rolling(window).mean()
            features[f"rolling_std_{window}"] = series.rolling(window).std()
        
        # Rate of change
        features["diff_1"] = series.diff(1)
        features["diff_4"] = series.diff(4)
        
        return pd.DataFrame(features)
    
    def _create_seasonal_features(self, week_indices):
        """Create Fourier-based seasonal features."""
        features = {}
        for period in [52, 26]:  # Annual and semi-annual
            features[f"sin_{period}"] = np.sin(2 * np.pi * week_indices / period)
            features[f"cos_{period}"] = np.cos(2 * np.pi * week_indices / period)
        features["week_of_year"] = week_indices % 52
        return pd.DataFrame(features)
    
    def _prepare_features(self, climate_df, target_col, horizon):
        """Prepare feature matrix for a specific forecast horizon."""
        # Autoregressive features (shifted by horizon to avoid data leakage)
        ar_features = self._create_ar_features(climate_df[target_col], lookback=12)
        ar_features = ar_features.shift(horizon)  # Shift to avoid leakage
        
        # Seasonal features
        seasonal = self._create_seasonal_features(climate_df["week_index"])
        
        # Combine
        X = pd.concat([ar_features, seasonal], axis=1)
        y = climate_df[target_col]
        
        # Drop NaN rows
        valid = X.dropna().index
        X = X.loc[valid]
        y = y.loc[valid]
        
        return X, y
    
    def fit(self, climate_df):
        """Train forecasting models for each horizon."""
        print("\n  Training Climate Forecasting Models...")
        print(f"  Forecast horizons: 1 to {self.forecast_horizon} weeks")
        
        results = {"temperature": {}, "precipitation": {}}
        
        for target, models_dict in [("temperature", self.temp_models), 
                                      ("precipitation", self.precip_models)]:
            print(f"\n  Target: {target}")
            
            for h in range(1, self.forecast_horizon + 1):
                X, y = self._prepare_features(climate_df, target, horizon=h)
                
                # Train/test split (last 52 weeks for test)
                split_idx = len(X) - 52
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42
                )
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                models_dict[h] = model
                self.scalers[f"{target}_h{h}"] = scaler
                results[target][h] = {"mae": mae, "r2": r2}
                
                if h in [1, 4, 8]:
                    print(f"    Horizon {h}w: MAE={mae:.2f}, R²={r2:.3f}")
        
        self.is_fitted = True
        self.feature_names_ = X.columns.tolist()
        return results
    
    def predict(self, climate_df, horizon=4):
        """Generate forecast for a specific horizon."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        forecasts = {}
        for target, models_dict in [("temperature", self.temp_models),
                                      ("precipitation", self.precip_models)]:
            h = min(horizon, self.forecast_horizon)
            X, _ = self._prepare_features(climate_df, target, horizon=h)
            scaler = self.scalers[f"{target}_h{h}"]
            X_scaled = scaler.transform(X.iloc[[-1]])
            forecasts[target] = models_dict[h].predict(X_scaled)[0]
        
        return forecasts
    
    def forecast_sequence(self, climate_df):
        """Generate 8-week forecast sequence for both variables."""
        sequence = []
        for h in range(1, self.forecast_horizon + 1):
            pred = self.predict(climate_df, horizon=h)
            sequence.append({
                "week": f"W{h}",
                "horizon_weeks": h,
                "temperature": round(pred["temperature"], 1),
                "precipitation": round(max(0, pred["precipitation"]), 0),
            })
        return sequence
    
    def get_feature_importance(self, target="temperature", horizon=4):
        """Return feature importances for a given model."""
        if target == "temperature":
            model = self.temp_models[horizon]
        else:
            model = self.precip_models[horizon]
        
        importances = dict(zip(self.feature_names_, model.feature_importances_))
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def save(self, path):
        """Save all models to disk."""
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            "temp_models": self.temp_models,
            "precip_models": self.precip_models,
            "scalers": self.scalers,
            "feature_names": self.feature_names_,
            "forecast_horizon": self.forecast_horizon,
        }, os.path.join(path, "climate_forecaster.pkl"))
        print(f"  Climate models saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load models from disk."""
        data = joblib.load(os.path.join(path, "climate_forecaster.pkl"))
        obj = cls(forecast_horizon=data["forecast_horizon"])
        obj.temp_models = data["temp_models"]
        obj.precip_models = data["precip_models"]
        obj.scalers = data["scalers"]
        obj.feature_names_ = data["feature_names"]
        obj.is_fitted = True
        return obj
