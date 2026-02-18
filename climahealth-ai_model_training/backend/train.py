"""
ClimaHealth AI — Model Training Pipeline
==========================================
Generates training data, trains all three ML models, evaluates performance,
and saves trained models to disk.

Usage:
    python train.py

This will:
1. Generate synthetic climate-disease training data (simulating NASA/WHO APIs)
2. Train the Climate Forecaster (Gradient Boosting time-series models)
3. Train the Disease Predictor (RF + GB + LR ensemble)
4. Train the NLP Outbreak Detector (TF-IDF + Logistic Regression)
5. Save all models to ./saved_models/
6. Print comprehensive evaluation metrics
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generate_training_data import (
    generate_full_dataset, 
    generate_news_corpus,
    REGION_PROFILES,
    generate_climate_timeseries,
    create_feature_matrix,
    generate_disease_incidence,
    generate_nlp_signals,
    compute_risk_score,
)
from models.climate_forecaster import ClimateForecaster
from models.disease_predictor import DiseasePredictor
from models.nlp_detector import OutbreakSignalDetector
from models.ensemble import EnsembleRiskEngine


def print_banner(text, char="="):
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def train_all():
    """Full training pipeline."""
    start_time = time.time()
    
    print_banner("ClimaHealth AI — Training Pipeline", "█")
    print("  Climate-Driven Disease Outbreak Early Warning System")
    print("  InnovAIte Hackathon 2026")
    print(f"{'█' * 70}")
    
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(model_dir, exist_ok=True)
    
    # =================================================================
    # Step 1: Generate Training Data
    # =================================================================
    print_banner("STEP 1/5: Generating Training Data")
    
    full_df = generate_full_dataset()
    full_df.to_csv(os.path.join(data_dir, "training_data.csv"), index=False)
    
    news_df = generate_news_corpus()
    news_df.to_csv(os.path.join(data_dir, "news_corpus.csv"), index=False)
    
    print(f"\n  Dataset Summary:")
    print(f"    Total samples: {len(full_df):,}")
    print(f"    Regions: {full_df['region'].nunique()}")
    print(f"    Diseases: {full_df['disease'].nunique()}")
    print(f"    Features: {full_df.shape[1] - 4} (excluding metadata)")
    print(f"    Outbreak rate: {full_df['outbreak'].mean()*100:.1f}%")
    print(f"    News corpus: {len(news_df)} headlines")
    
    # =================================================================
    # Step 2: Train Climate Forecaster
    # =================================================================
    print_banner("STEP 2/5: Training Climate Forecaster")
    
    # Train on one representative region (Dhaka — most data-rich)
    dhaka_profile = REGION_PROFILES["dhaka_bangladesh"]
    climate_df = generate_climate_timeseries(dhaka_profile, n_weeks=520)
    
    climate_model = ClimateForecaster(forecast_horizon=8)
    climate_results = climate_model.fit(climate_df)
    climate_model.save(model_dir)
    
    print(f"\n  Climate Model Summary:")
    for target in ["temperature", "precipitation"]:
        for h in [1, 4, 8]:
            r = climate_results[target][h]
            print(f"    {target.capitalize()} h={h}w: MAE={r['mae']:.2f}, R²={r['r2']:.3f}")
    
    # =================================================================
    # Step 3: Train Disease Predictor
    # =================================================================
    print_banner("STEP 3/5: Training Disease Prediction Ensemble")
    
    disease_model = DiseasePredictor()
    disease_results = disease_model.fit(full_df)
    disease_model.save(model_dir)
    
    print(f"\n  Disease Model Summary:")
    print(f"    Classification F1:  {disease_results['classification']['f1']:.3f}")
    print(f"    Classification AUC: {disease_results['classification']['auc']:.3f}")
    print(f"    CV F1 (mean):       {disease_results['classification']['cv_f1_mean']:.3f}")
    print(f"    Regression MAE:     {disease_results['regression']['mae']:.2f}")
    print(f"    Regression R²:      {disease_results['regression']['r2']:.3f}")
    
    # Feature importance
    print(f"\n  SHAP-like Feature Category Importance:")
    shap_summary = disease_model.get_shap_summary()
    for cat, imp in shap_summary.items():
        bar = "█" * int(imp * 50)
        print(f"    {cat:20s} {imp*100:5.1f}% {bar}")
    
    # =================================================================
    # Step 4: Train NLP Detector
    # =================================================================
    print_banner("STEP 4/5: Training NLP Outbreak Signal Detector")
    
    nlp_model = OutbreakSignalDetector()
    nlp_results = nlp_model.fit(news_df)
    nlp_model.save(model_dir)
    
    print(f"\n  NLP Model Summary:")
    print(f"    Classification F1:  {nlp_results['f1']:.3f}")
    print(f"    Classification AUC: {nlp_results['auc']:.3f}")
    
    # Top features
    top_features = nlp_model.get_top_features(n=10)
    print(f"\n  Top Outbreak Indicator Words:")
    for word, coef in top_features["outbreak_indicators"][:8]:
        print(f"    {word:30s} coef={coef:+.3f}")
    
    # =================================================================
    # Step 5: Run Ensemble Demo Prediction
    # =================================================================
    print_banner("STEP 5/5: Ensemble Integration Demo")
    
    # Create ensemble engine
    ensemble = EnsembleRiskEngine(climate_model, disease_model, nlp_model)
    
    # Run a demo prediction for Dhaka
    print("\n  Running ensemble prediction for Dhaka, Bangladesh (Dengue)...")
    
    # Prepare demo data
    dhaka_climate = generate_climate_timeseries(dhaka_profile, n_weeks=520)
    dhaka_disease_params = dhaka_profile["diseases"]["dengue"]
    dhaka_cases = generate_disease_incidence(dhaka_climate, dhaka_disease_params, n_weeks=520)
    dhaka_nlp = generate_nlp_signals(dhaka_climate, dhaka_cases, n_weeks=520)
    dhaka_features = create_feature_matrix(dhaka_climate, dhaka_nlp, lag_weeks=4)
    
    aligned_cases = dhaka_cases[len(dhaka_cases) - len(dhaka_features):]
    aligned_risk = compute_risk_score(dhaka_cases)[len(dhaka_cases) - len(dhaka_features):]
    dhaka_features["cases"] = aligned_cases
    dhaka_features["risk_score"] = aligned_risk
    dhaka_features["outbreak"] = (aligned_risk >= 60).astype(int)
    dhaka_features["region"] = "dhaka_bangladesh"
    dhaka_features["disease"] = "dengue"
    
    # Demo news headlines
    demo_headlines = [
        "WHO reports surge in dengue cases across Bangladesh",
        "Dhaka hospitals overwhelmed by fever patients amid monsoon season",
        "Aedes mosquito breeding sites multiply after heavy rainfall in Dhaka slums",
        "New hospital opens in Dhaka with expanded ICU capacity",
    ]
    
    # Run ensemble assessment
    assessment = ensemble.assess_risk(
        climate_df=dhaka_climate,
        features_df=dhaka_features,
        news_texts=demo_headlines,
    )
    
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  ENSEMBLE RISK ASSESSMENT — DHAKA, BD       │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │  Risk Score:     {assessment['risk_score']:>3}/100                    │")
    print(f"  │  Risk Level:     {assessment['risk_level'].upper():>8s}                   │")
    print(f"  │  Confidence:     {assessment['confidence']*100:>5.1f}%                    │")
    print(f"  │  Outbreak Prob:  {assessment['outbreak_probability']*100:>5.1f}%                    │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │  Component Scores:                          │")
    for comp, score in assessment["component_scores"].items():
        print(f"  │    {comp:25s} {score:>5.1f}/100     │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │  SHAP Feature Importance:                   │")
    for cat, imp in assessment["shap_summary"].items():
        bar = "█" * int(imp * 30)
        print(f"  │    {cat:15s} {imp*100:5.1f}% {bar:<15s}│")
    print(f"  └─────────────────────────────────────────────┘")
    
    # Climate forecast
    print(f"\n  8-Week Climate Forecast:")
    for f in assessment["climate_forecast"]:
        print(f"    {f['week']}: Temp={f['temperature']}°C, Precip={f['precipitation']}mm")
    
    # NLP signals
    print(f"\n  NLP Signal Analysis:")
    for sig in assessment["nlp_signals"]:
        status = "🔴 OUTBREAK" if sig["is_outbreak"] else "🟢 Non-outbreak"
        print(f"    {status} (conf={sig['confidence']:.2f}, sev={sig['severity']})")
        print(f"      \"{sig['text'][:60]}...\"")
    
    # Alerts
    print(f"\n  Active Alerts ({len(assessment['alerts'])}):")
    for alert in assessment["alerts"]:
        level = alert["level"].upper()
        emoji = "🔴" if level == "CRITICAL" else "🟡"
        print(f"    {emoji} [{level}] {alert['message'][:70]}...")
        print(f"       → {alert['action']}")
    
    # CHW Alert
    chw_alert = ensemble.generate_chw_alert(assessment, "Dhaka, Bangladesh", "Dengue Fever")
    print(f"\n  Community Health Worker Alert:")
    print(f"    {chw_alert['summary']}")
    print(f"\n    Recommended Actions:")
    for i, action in enumerate(chw_alert["recommended_actions"], 1):
        print(f"      {i}. {action}")
    
    # =================================================================
    # Summary
    # =================================================================
    elapsed = time.time() - start_time
    
    print_banner("TRAINING COMPLETE", "█")
    print(f"  Total time:     {elapsed:.1f}s")
    print(f"  Models saved:   {model_dir}/")
    print(f"  Files created:")
    for f in sorted(os.listdir(model_dir)):
        size = os.path.getsize(os.path.join(model_dir, f))
        print(f"    {f:30s} {size/1024:.0f} KB")
    
    print(f"\n  Model Performance Summary:")
    print(f"    Climate Forecaster:    Temp MAE={climate_results['temperature'][4]['mae']:.2f}°C (4-week)")
    print(f"    Disease Classifier:    F1={disease_results['classification']['f1']:.3f}, AUC={disease_results['classification']['auc']:.3f}")
    print(f"    Disease Regressor:     MAE={disease_results['regression']['mae']:.2f}, R²={disease_results['regression']['r2']:.3f}")
    print(f"    NLP Detector:          F1={nlp_results['f1']:.3f}, AUC={nlp_results['auc']:.3f}")
    
    print(f"\n  Next steps:")
    print(f"    1. Start API:  cd climahealth_backend && uvicorn api.main:app --reload")
    print(f"    2. Run demo:   python demo.py")
    print(f"    3. Dashboard:  Open the React frontend at localhost:3000")
    print(f"{'█' * 70}\n")
    
    return {
        "climate": climate_results,
        "disease": disease_results,
        "nlp": nlp_results,
        "ensemble_demo": assessment,
    }


if __name__ == "__main__":
    train_all()
