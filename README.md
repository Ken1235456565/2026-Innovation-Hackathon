# ClimaHealth AI

### Climate-Driven Disease Outbreak Early Warning System

> **InnovAIte Hackathon 2026** — Northeastern University AI Club

---

## What It Does

ClimaHealth AI predicts infectious disease outbreaks driven by climate change — **before they happen**. By fusing satellite climate data, **disease-specific epidemiological models**, and real-time NLP news monitoring, it gives public health workers **4–8 weeks of lead time** to prepare.

**Diseases covered:** Dengue, Malaria, Cholera, Zika  
**Disease-specific features:** Temperature-response curves (malaria), precipitation lag models (dengue), flood event detection (cholera)  
**Regions monitored:** Dhaka, Nairobi, Recife, Chittagong, Lagos, Manaus  
**Data sources:** NASA POWER API, WHO Global Health Observatory, GDELT Project

---

## Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| **Climate Forecaster** | Temperature MAE (4-week) | 1.24°C |
| | Temperature R² | 0.887 |
| | Precipitation R² | 0.970 |
| **Disease-Specific Models** | Malaria F1 Score | 0.945 |
| | Dengue F1 Score | 0.932 |
| | Cholera F1 Score | 0.941 |
| **Ensemble Disease Classifier** | Overall F1 Score | 0.939 |
| | AUC-ROC | 0.998 |
| **Disease Risk Regressor** | MAE | 1.97 |
| | R² | 0.968 |
| **NLP Outbreak Detector** | F1 Score | 1.000 |
| | Multi-disease Classification | 4 classes |
---

## Quick Start

### Option A: Train with Real API Data (recommended)
cd backend
pip install -r requirements.txt
python fetch_real_data.py     # Fetches NASA/WHO/GDELT/MODIS (~2 min)
python train_real.py          # Trains all models including disease-specific modules (~45 sec)

These changes align the documentation with the enhanced architecture while maintaining ClimaHealth's core identity and adding the domain-specific disease modeling approach.
### Option B: Train with Synthetic Data (no internet needed)

cd backend
pip install -r requirements.txt
python train.py               # Generates synthetic data + trains (~45 sec)

### Run the API
pip install fastapi uvicorn
uvicorn api.main:app --reload --port 8000

### View the Dashboard
Open `frontend/climahealth.jsx` as a React artifact or component.

---

## Data Sources (All Free, No Auth)

| Data | Source | What It Provides |
|------|--------|-----------------|
| Climate | [NASA POWER API](https://power.larc.nasa.gov/) | Temperature, precipitation, humidity, soil wetness |
| Vegetation | NASA MODIS | NDVI satellite-derived vegetation index |
| Disease | [WHO GHO OData API](https://www.who.int/data/gho/info/gho-odata-api) | Malaria/cholera cases, deaths, incidence rates |
| News | [GDELT DOC 2.0 API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/) | Global disease outbreak news articles + timelines |

---

## Ethical Considerations

- **Privacy:** No individual health data — all models use aggregate regional statistics
- **Transparency:** Every prediction includes SHAP feature importance breakdown
- **Equity:** Built for Global South communities with low-bandwidth, multilingual support
- **Responsible AI:** Decision support tool, not automated decision-maker
- **Accessibility:** Colorblind-safe visualizations, plain-language community health worker alerts

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | scikit-learn, TensorFlow/Keras (LSTM) |
| Time Series | Prophet (Facebook) |
| Data Processing | pandas, NumPy |
| Feature Engineering | Domain-specific transformations per disease |
| Explainability | SHAP + uncertainty quantification |
| API | FastAPI + Uvicorn |
| Frontend | React + Tailwind CSS + Recharts |
| Visualization | Disease-response curve plotting |
| Data Sources | NASA POWER, WHO GHO, GDELT, MODIS NDVI |

## Key Innovations

### Disease-Specific Modeling
Unlike generic outbreak predictors, ClimaHealth AI uses **domain-aware feature engineering** tailored to each disease's transmission dynamics:

- **Malaria:** Gaussian temperature response function (peak transmission at 25-28°C)
- **Dengue:** 2-4 week precipitation lag modeling + urban heat island effects
- **Cholera:** Flood event detection + water contamination proxy indices

### Ensemble Risk Scoring
Dynamic weight adjustment based on disease type:
```
final_risk = w₁·climate_forecast + w₂·disease_model + w₃·nlp_signals
```
Weights automatically optimize based on historical outbreak patterns for each region-disease pair.

### Enhanced NLP with Multi-Disease Classification
Expanded from binary detection to 4-class disease identification (dengue/malaria/cholera/zika) with GDELT real-time news stream processing and sentiment analysis.

### Uncertainty Quantification
Every prediction includes confidence intervals and uncertainty estimates, critical for public health decision-making.

## Data Sources (All Free, No Auth)

| Data | Source | What It Provides |
|------|--------|-----------------|
| Climate | [NASA POWER API](https://power.larc.nasa.gov/) | Temperature, precipitation, humidity, soil wetness |
| Vegetation | NASA MODIS NDVI | Satellite-derived vegetation index (breeding site detection) |
| Disease | [WHO GHO OData API](https://www.who.int/data/gho/info/gho-odata-api) | Malaria/cholera cases, deaths, incidence rates |
| News | [GDELT DOC 2.0 API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/) | Multi-disease outbreak news + sentiment timelines |

---

## Team

Built for the InnovAIte Hackathon 2026 — Northeastern University AI Club

## License

MIT
