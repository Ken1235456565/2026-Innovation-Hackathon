# 🌡 ClimaHealth AI

### Climate-Driven Disease Outbreak Early Warning System

> **InnovAIte Hackathon 2026** — Northeastern University AI Club

---

## What It Does

ClimaHealth AI predicts infectious disease outbreaks driven by climate change — **before they happen**. By fusing satellite climate data, epidemiological models, and real-time NLP news monitoring, it gives public health workers **4–8 weeks of lead time** to prepare.

**Diseases covered:** Dengue, Malaria, Cholera, Zika  
**Regions monitored:** Dhaka, Nairobi, Recife, Chittagong, Lagos, Manaus  
**Data sources:** NASA POWER API, WHO Global Health Observatory, GDELT Project

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                  │
│  NASA POWER API │ WHO GHO │ MODIS Satellite │ GDELT News    │
└────────┬──────────────┬──────────────────┬───────────────────┘
         │              │                  │
┌────────▼──────────────▼──────────────────▼───────────────────┐
│                    ML PIPELINE (Python)                       │
│                                                              │
│  ┌─────────────────┐ ┌──────────────────┐ ┌───────────────┐ │
│  │ Climate         │ │ Disease Ensemble │ │ NLP Outbreak   │ │
│  │ Forecaster      │ │ RF + GB + LR     │ │ Signal         │ │
│  │ (GBR time-      │ │ Classifier +     │ │ Detector       │ │
│  │  series)        │ │ Regressor        │ │ (TF-IDF + LR) │ │
│  └────────┬────────┘ └────────┬─────────┘ └───────┬────────┘ │
│           └───────────────────┼────────────────────┘          │
│                     ┌─────────▼──────────┐                   │
│                     │  Ensemble Risk     │                   │
│                     │  Scoring Engine    │                   │
│                     │  + SHAP Explainer  │                   │
│                     └────────────────────┘                   │
└──────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│         FRONTEND — React Interactive Dashboard               │
│  Global Risk Map │ Climate Charts │ NLP Feed │ SHAP Panel   │
└──────────────────────────────────────────────────────────────┘
```

---

## Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| **Climate Forecaster** | Temperature MAE (4-week) | 1.24°C |
| | Temperature R² | 0.887 |
| | Precipitation R² | 0.970 |
| **Disease Classifier** (Ensemble) | F1 Score | 0.939 |
| | AUC-ROC | 0.998 |
| **Disease Risk Regressor** | MAE | 1.97 |
| | R² | 0.968 |
| **NLP Outbreak Detector** | F1 Score | 1.000 |

---

## Quick Start

### Option A: Train with Real API Data (recommended)
```bash
cd backend
pip install -r requirements.txt
python fetch_real_data.py     # Fetches real data from NASA/WHO/GDELT (~2 min)
python train_real.py          # Trains models on real data (~30 sec)
```

### Option B: Train with Synthetic Data (no internet needed)
```bash
cd backend
pip install -r requirements.txt
python train.py               # Generates synthetic data + trains (~45 sec)
```

### Run the API
```bash
pip install fastapi uvicorn
uvicorn api.main:app --reload --port 8000
```

### View the Dashboard
Open `frontend/climahealth.jsx` as a React artifact or component.

---

## Project Structure

```
climahealth-ai/
│
├── README.md                          # This file
├── PROPOSAL.md                        # Full hackathon proposal + demo script
│
├── backend/
│   ├── fetch_real_data.py             # Fetches real data from NASA/WHO/GDELT APIs
│   ├── train.py                       # Train models on synthetic data
│   ├── train_real.py                  # Train models on real API data
│   ├── requirements.txt               # Python dependencies
│   │
│   ├── models/
│   │   ├── climate_forecaster.py      # Temperature + precipitation forecasting
│   │   ├── disease_predictor.py       # RF + GB + LR ensemble classifier
│   │   ├── nlp_detector.py            # TF-IDF outbreak signal detector
│   │   └── ensemble.py               # Combined risk scoring engine
│   │
│   ├── data/
│   │   └── generate_training_data.py  # Synthetic data generator
│   │
│   ├── api/
│   │   └── main.py                    # FastAPI REST API
│   │
│   └── saved_models/                  # Pre-trained model files
│       ├── climate_forecaster.pkl
│       ├── disease_predictor.pkl
│       └── nlp_detector.pkl
│
└── frontend/
    └── climahealth.jsx                # React interactive dashboard
```

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
| ML Framework | scikit-learn |
| Data Processing | pandas, NumPy |
| API | FastAPI + Uvicorn |
| Explainability | SHAP-compatible feature importance |
| Frontend | React + Tailwind CSS + Recharts |
| Data Sources | NASA POWER, WHO GHO, GDELT |

---

## Team

Built for the InnovAIte Hackathon 2026 — Northeastern University AI Club

## License

MIT
