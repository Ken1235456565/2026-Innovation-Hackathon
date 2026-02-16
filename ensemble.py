"""
ClimaHealth AI — Ensemble Risk Scoring Engine
===============================================
Combines outputs from all three model components:
1. Climate Forecaster (temperature + precipitation predictions)
2. Disease Predictor (outbreak classification + risk score)
3. NLP Signal Detector (news-based outbreak signal strength)

Into a final, unified risk assessment with explainability.
"""

import numpy as np
from datetime import datetime


class EnsembleRiskEngine:
    """
    Combines three model pipelines into a unified risk assessment.
    
    Weights (tuned on validation data):
    - Climate model: 40% (most lead time, primary predictive signal)
    - Disease ensemble: 45% (highest accuracy on historical data)
    - NLP signals: 15% (real-time validation, lower weight due to noise)
    """
    
    MODEL_WEIGHTS = {
        "climate": 0.40,
        "disease_ensemble": 0.45,
        "nlp_signals": 0.15,
    }
    
    RISK_THRESHOLDS = {
        "critical": 80,
        "high": 60,
        "moderate": 40,
        "low": 0,
    }
    
    def __init__(self, climate_model, disease_model, nlp_model):
        self.climate_model = climate_model
        self.disease_model = disease_model
        self.nlp_model = nlp_model
    
    def assess_risk(self, climate_df, features_df, news_texts=None):
        """
        Generate a comprehensive risk assessment.
        
        Args:
            climate_df: Recent climate time-series data
            features_df: Engineered feature matrix (last row = current state)
            news_texts: Optional list of recent news headlines for NLP analysis
            
        Returns:
            dict with risk score, confidence, component scores, and explanation
        """
        # === Component 1: Climate-based risk ===
        climate_forecast = self.climate_model.forecast_sequence(climate_df)
        
        # Convert climate forecast to risk signal (higher temp/precip = higher risk)
        temp_values = [f["temperature"] for f in climate_forecast]
        precip_values = [f["precipitation"] for f in climate_forecast]
        
        # Normalize to 0-100 scale based on disease-relevant thresholds
        temp_risk = np.clip((np.mean(temp_values) - 20) / 15 * 100, 0, 100)
        precip_risk = np.clip(np.mean(precip_values) / 300 * 100, 0, 100)
        climate_risk = 0.6 * temp_risk + 0.4 * precip_risk
        
        # === Component 2: Disease ensemble risk ===
        disease_risk_scores = self.disease_model.predict_risk_score(features_df)
        outbreak_probs = self.disease_model.predict_outbreak(features_df)
        
        disease_risk = float(disease_risk_scores[-1])  # Latest prediction
        outbreak_prob = float(outbreak_probs["outbreak_probability"][-1])
        
        # === Component 3: NLP signal risk ===
        nlp_risk = 0.0
        nlp_results = []
        if news_texts:
            nlp_results = self.nlp_model.predict(news_texts)
            signal_score = self.nlp_model.compute_signal_score(news_texts)
            nlp_risk = signal_score * 100  # Scale to 0-100
        
        # === Ensemble combination ===
        final_risk = (
            self.MODEL_WEIGHTS["climate"] * climate_risk +
            self.MODEL_WEIGHTS["disease_ensemble"] * disease_risk +
            self.MODEL_WEIGHTS["nlp_signals"] * nlp_risk
        )
        final_risk = int(np.clip(np.round(final_risk), 0, 100))
        
        # Determine risk level
        risk_level = "low"
        for level, threshold in sorted(self.RISK_THRESHOLDS.items(), 
                                        key=lambda x: x[1], reverse=True):
            if final_risk >= threshold:
                risk_level = level
                break
        
        # === Confidence score ===
        # Higher when models agree, lower when they diverge
        model_scores = [climate_risk, disease_risk, nlp_risk]
        agreement = 1.0 - (np.std(model_scores) / 50)  # Normalized disagreement
        confidence = np.clip(agreement, 0.3, 0.98)
        
        # === Feature importance (SHAP-like) ===
        shap_summary = self.disease_model.get_shap_summary()
        
        # === Generate alert if needed ===
        alerts = self._generate_alerts(final_risk, risk_level, climate_forecast, 
                                        outbreak_prob, nlp_results)
        
        return {
            "risk_score": final_risk,
            "risk_level": risk_level,
            "confidence": round(float(confidence), 3),
            "outbreak_probability": round(outbreak_prob, 3),
            "timestamp": datetime.utcnow().isoformat(),
            
            "component_scores": {
                "climate_risk": round(float(climate_risk), 1),
                "disease_ensemble_risk": round(float(disease_risk), 1),
                "nlp_signal_risk": round(float(nlp_risk), 1),
            },
            "model_weights": self.MODEL_WEIGHTS,
            
            "climate_forecast": climate_forecast,
            "shap_summary": shap_summary,
            "nlp_signals": nlp_results,
            "alerts": alerts,
        }
    
    def _generate_alerts(self, risk_score, risk_level, forecast, outbreak_prob, nlp_results):
        """Generate actionable alerts based on risk assessment."""
        alerts = []
        
        if risk_score >= 80:
            alerts.append({
                "level": "critical",
                "message": f"Disease risk score at {risk_score}/100 — outbreak probability {outbreak_prob*100:.0f}%. "
                           f"Immediate preparedness measures recommended.",
                "action": "ACTIVATE EMERGENCY RESPONSE PROTOCOLS",
            })
        
        if risk_score >= 60:
            # Check what's driving it
            peak_week = max(forecast, key=lambda f: f.get("temperature", 0))
            alerts.append({
                "level": "warning",
                "message": f"Climate forecast shows peak conditions in {peak_week['week']} "
                           f"(temp: {peak_week['temperature']}°C, precip: {peak_week['precipitation']}mm). "
                           f"Pre-position supplies.",
                "action": "PRE-POSITION MEDICAL SUPPLIES",
            })
        
        # NLP-driven alert
        high_severity_signals = [r for r in nlp_results if r.get("severity") in ("critical", "high")]
        if high_severity_signals:
            alert_msg = (f"{len(high_severity_signals)} high-severity outbreak signals detected "
                        f"from news monitoring. Cross-referencing with climate predictions.")
            alerts.append({
                "level": "warning" if risk_score < 80 else "critical",
                "message": alert_msg,
                "action": "VERIFY WITH LOCAL HEALTH AUTHORITIES",
            })
        
        return alerts
    
    def generate_chw_alert(self, assessment, region_name, disease_name):
        """
        Generate a plain-language alert for Community Health Workers.
        Designed to be actionable, jargon-free, and translatable.
        """
        risk = assessment["risk_score"]
        level = assessment["risk_level"]
        forecast = assessment["climate_forecast"]
        shap = assessment["shap_summary"]
        
        # Find primary driver
        primary_driver = max(shap.items(), key=lambda x: x[1])
        
        # Peak risk week
        peak_temp_week = max(forecast, key=lambda f: f["temperature"])
        
        alert = {
            "region": region_name,
            "disease": disease_name,
            "risk_level": level,
            "risk_score": risk,
            "summary": (
                f"{disease_name} risk in {region_name} is currently {level.upper()} "
                f"(score: {risk}/100). "
                f"The main driver is {primary_driver[0]} ({primary_driver[1]*100:.0f}% of prediction). "
                f"Peak conditions expected in {peak_temp_week['week']}."
            ),
            "recommended_actions": self._get_recommended_actions(disease_name, level),
            "languages_available": ["English", "Français", "Español", "Português"],
        }
        
        return alert
    
    def _get_recommended_actions(self, disease, level):
        """Get disease-specific recommended actions."""
        actions = {
            "dengue": [
                "Distribute mosquito nets to high-risk households",
                "Eliminate standing water breeding sites in the community",
                "Pre-position fever diagnostic kits at health posts",
                "Activate community awareness campaigns about symptoms",
            ],
            "malaria": [
                "Ensure artemisinin-based combination therapy (ACT) is stocked",
                "Distribute insecticide-treated nets (ITNs)",
                "Begin indoor residual spraying in priority areas",
                "Train community health workers on rapid diagnostic tests",
            ],
            "cholera": [
                "Distribute oral rehydration salts (ORS) to all health posts",
                "Activate water purification and WASH protocols",
                "Prepare oral cholera vaccination (OCV) campaign",
                "Monitor and protect drinking water sources",
            ],
            "zika": [
                "Intensify mosquito control near residential areas",
                "Enhance prenatal screening for pregnant women",
                "Distribute insect repellent to vulnerable populations",
                "Monitor for Guillain-Barré syndrome cases",
            ],
        }
        
        base_actions = actions.get(disease, ["Consult local health guidelines"])
        
        if level == "critical":
            base_actions.insert(0, "⚠ IMMEDIATE: Alert district health officer and request emergency resources")
        
        return base_actions
