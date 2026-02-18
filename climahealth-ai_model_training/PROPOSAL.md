# ClimaHealth AI — Climate-Driven Disease Outbreak Early Warning System

## InnovAIte Hackathon Submission

---

## The Idea (Elevator Pitch)

ClimaHealth AI is an AI-powered early warning platform that predicts infectious disease outbreaks driven by climate change — before they happen. By fusing satellite climate data, epidemiological records, and real-time news/social media signals, it forecasts where and when climate conditions will trigger disease surges (malaria, dengue, cholera, Zika) in vulnerable communities, giving public health workers weeks of lead time to prepare.

**Why this is an existential threat:** Climate change is already expanding disease zones, and the WHO estimates climate-driven diseases will cause 250,000+ additional deaths per year by 2030. The intersection of climate change and pandemic risk is one of the least-prepared-for catastrophic scenarios facing humanity.

---

## Rubric Alignment Map

Below is how ClimaHealth AI targets **every rubric criterion at the Excellent (5) level**:

| Rubric Criterion | How ClimaHealth AI Scores a 5 |
|---|---|
| **Alignment to Problem Statements** | Directly addresses two existential threats (climate collapse + pandemic risk) from the problem statement, and spans 4 suggested domains: Healthcare, Environmental Science, Social Good, and Text Analysis |
| **Responsiveness to Human Needs** | Built for the communities most affected — low-income tropical/subtropical regions with limited health infrastructure. Features community health worker interface, multilingual alerts, and offline-capable mobile design |
| **Technicality** | Multi-model ML pipeline: climate time-series forecasting (LSTM/Prophet), satellite image classification (CNN), NLP-based outbreak signal detection, ensemble risk scoring, and interactive geospatial dashboard |
| **Ethical Considerations** | Privacy-preserving design (no individual health data), open-source datasets, accessibility-first UI (colorblind-safe, low-bandwidth), equitable focus on Global South communities, transparent model explainability |
| **Feasibility & Scalability** | Uses freely available data (NASA, WHO, GDELT), lightweight models runnable on consumer hardware, modular architecture scales to any region/disease, clear deployment roadmap |
| **Innovation & Creativity** | Novel fusion of climate forecasting + epidemiological prediction + NLP signal detection — most existing tools address these in isolation, not as an integrated early warning system |
| **Presentation Skills** | Clear narrative: "Climate change is making disease outbreaks harder to predict — we built an AI that sees them coming" with a live demo showing a real-world prediction |

---

## Problem Deep Dive

### The Climate-Disease Nexus

Climate change doesn't just cause floods and heatwaves — it fundamentally reshapes the global disease landscape:

- **Rising temperatures** expand the habitat range of mosquitoes carrying malaria, dengue, and Zika into regions that have never experienced these diseases
- **Extreme rainfall and flooding** create breeding grounds for waterborne diseases like cholera and leptospirosis
- **Drought cycles** force population displacement, concentrating people in areas with poor sanitation
- **Shifting seasons** disrupt traditional disease patterns, making historical surveillance data unreliable

### Why Current Systems Fail

Traditional disease surveillance is **reactive** — it detects outbreaks after people start getting sick. By then, it's often too late to prevent exponential spread, especially in communities with limited healthcare infrastructure. Existing climate models and epidemiological models operate in silos, and there is no integrated system that connects climate forecasts to disease risk predictions with actionable lead time.

### Who Is Affected

The communities most vulnerable to climate-driven disease outbreaks are those with the fewest resources to respond: rural communities in sub-Saharan Africa, South/Southeast Asia, and Latin America. These are also the communities with the least access to early warning tools. ClimaHealth AI is built for them first.

---

## Core Features

### 1. Climate Risk Forecasting Engine
- Ingests satellite-derived climate data (temperature, precipitation, humidity, vegetation index) from NASA POWER and MODIS
- Uses LSTM neural networks and Facebook Prophet to forecast climate conditions 4-8 weeks ahead
- Identifies when conditions cross disease-specific thresholds (e.g., sustained temperatures above 25°C + high humidity = elevated dengue risk)

### 2. Epidemiological Prediction Model
- Trained on historical disease incidence data (WHO Global Health Observatory, local health ministry records)
- Correlates past outbreaks with preceding climate patterns to learn region-specific predictive signals
- Uses a Random Forest / Gradient Boosting ensemble to generate disease probability scores by region

### 3. Real-Time Signal Detection (NLP Module)
- Monitors news articles and public health reports using NLP to detect early outbreak signals
- Classifies reports by disease type, severity, and location using a fine-tuned text classifier
- Serves as a "reality check" that validates or boosts the climate model's predictions with ground-truth signals
- Data sources: GDELT Project (global news), WHO Disease Outbreak News, ProMED-mail

### 4. Interactive Risk Dashboard
- Geospatial heatmap showing predicted disease risk by region, updated weekly
- Time-slider to see how risk evolves over the coming 4-8 weeks
- Drill-down views: click any region to see which climate factors are driving the risk, model confidence level, and recommended preparedness actions
- Built with React + Leaflet/Mapbox for interactive mapping

### 5. Community Alert System
- Generates plain-language alerts for community health workers: "Dengue risk in [Region] is predicted to increase by 60% in the next 3 weeks due to sustained high temperatures and recent heavy rainfall"
- Alerts include recommended actions: distribute mosquito nets, pre-position oral rehydration supplies, activate community health volunteer networks
- Designed for multilingual deployment and low-bandwidth environments

### 6. Model Explainability Layer
- Every prediction includes a feature importance breakdown: "This prediction is driven 45% by temperature anomaly, 30% by precipitation levels, 15% by historical outbreak seasonality, 10% by news signal detection"
- Builds trust with public health decision-makers by making the AI's reasoning transparent
- Uses SHAP (SHapley Additive exPlanations) values for interpretable ML

---

## Technical Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │ NASA POWER / │  │ WHO / Local  │  │ GDELT / ProMED /   │ │
│  │ MODIS Climate│  │ Health Data  │  │ News APIs          │ │
│  │ APIs         │  │              │  │ (NLP Pipeline)     │ │
│  └──────┬───────┘  └──────┬───────┘  └─────────┬──────────┘ │
└─────────┼─────────────────┼────────────────────┼────────────┘
          │                 │                    │
┌─────────┼─────────────────┼────────────────────┼────────────┐
│         ▼                 ▼                    ▼            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              FEATURE ENGINEERING LAYER                │   │
│  │  - Climate feature extraction (temp, precip, NDVI)   │   │
│  │  - Epidemiological feature engineering (lag, season)  │   │
│  │  - NLP feature extraction (outbreak mentions, severity│   │
│  └──────────────────────────┬───────────────────────────┘   │
│                ML PIPELINE  │                               │
│  ┌──────────────────────────┼───────────────────────────┐   │
│  │                          ▼                           │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌──────────────┐ │   │
│  │  │ Climate    │  │ Epi Ensemble │  │ NLP Signal   │ │   │
│  │  │ Forecaster │  │ (RF + XGBoost│  │ Classifier   │ │   │
│  │  │ (LSTM +    │  │  + Logistic  │  │ (TF-IDF +    │ │   │
│  │  │  Prophet)  │  │  Regression) │  │  BERT/Claude)│ │   │
│  │  └─────┬──────┘  └──────┬───────┘  └──────┬───────┘ │   │
│  │        │                │                  │         │   │
│  │        └────────────────┼──────────────────┘         │   │
│  │                         ▼                            │   │
│  │              ┌────────────────────┐                  │   │
│  │              │ ENSEMBLE RISK      │                  │   │
│  │              │ SCORING ENGINE     │                  │   │
│  │              │ + SHAP Explainer   │                  │   │
│  │              └─────────┬──────────┘                  │   │
│  └────────────────────────┼─────────────────────────────┘   │
│                BACKEND    │  (Python / FastAPI)              │
└───────────────────────────┼─────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────┐
│                    FRONTEND (React)                          │
│                           ▼                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │ Geospatial   │  │ Time-Series  │  │ Alert Generator    │ │
│  │ Risk Heatmap │  │ Risk Charts  │  │ & Explainability   │ │
│  │ (Leaflet)    │  │ (Recharts)   │  │ Dashboard          │ │
│  └──────────────┘  └──────────────┘  └────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Justification |
|---|---|---|
| Frontend | React + Tailwind CSS + Leaflet.js + Recharts | Interactive maps + charts, responsive design |
| Backend | Python + FastAPI | Fast, async-ready, ML-ecosystem compatible |
| Climate Forecasting | LSTM (PyTorch) + Facebook Prophet | Time-series forecasting with seasonality |
| Disease Prediction | scikit-learn (Random Forest, XGBoost, Logistic Regression) | Proven ensemble methods for tabular classification |
| NLP Module | TF-IDF + Claude API (or lightweight BERT) | News signal classification and outbreak detection |
| Explainability | SHAP | Feature importance for every prediction |
| Data Sources | NASA POWER API, WHO GHO, GDELT Project | All freely available, no authentication barriers |
| Deployment | Docker + Vercel (frontend) / Railway (backend) | Quick deployment for hackathon demo |

---

## Data Sources (All Free & Open)

| Data | Source | What It Provides |
|---|---|---|
| Climate (temperature, precipitation, humidity) | NASA POWER API | Global gridded climate data, daily/monthly |
| Vegetation index (NDVI) | NASA MODIS | Satellite-derived vegetation health — proxy for mosquito habitat |
| Disease incidence | WHO Global Health Observatory | Historical outbreak data by country/region |
| News & outbreak reports | GDELT Project + ProMED-mail | Real-time global news events and disease alerts |
| Population & vulnerability data | World Bank Open Data | Demographics for risk-weighted scoring |

---

## Ethical Considerations (Targeting Excellent)

### Privacy
- **No individual health data** — all models operate on aggregate, regional-level statistics
- No personally identifiable information is collected, stored, or processed
- NLP module processes public news articles only, not social media from individuals

### Transparency & Explainability
- Every prediction includes a SHAP-based feature importance breakdown
- Model confidence scores are always displayed — the system never presents uncertain predictions as definitive
- All data sources are publicly documented and verifiable

### Equity & Inclusivity
- **Built for the Global South first** — the communities most affected and least served by current surveillance tools
- Multilingual alert generation (English, French, Spanish, Portuguese — the primary languages of the most affected regions)
- Low-bandwidth, mobile-first design for areas with limited internet
- Colorblind-accessible visualization palette (viridis color scale)

### Sustainability
- Lightweight ML models (no massive GPU requirements) — trainable on a single consumer laptop
- Uses existing public satellite and health data rather than requiring new data collection infrastructure

### Responsible AI
- The system provides *decision support*, not automated decisions — it empowers health workers with information, not directives
- Clear documentation of model limitations and uncertainty
- Bias monitoring: regular checks that prediction accuracy doesn't systematically differ across regions or demographic groups

### Accessibility
- Screen-reader compatible dashboard components
- High-contrast mode for visually impaired users
- Plain-language alerts (no technical jargon) for community health workers

---

## Community Engagement Plan (Targeting "Responsiveness to Human Needs")

### Who Are the Target Communities?
- Community health workers (CHWs) in climate-vulnerable regions
- District-level public health officers in sub-Saharan Africa, South/Southeast Asia, and Latin America
- Ministries of Health disease surveillance teams
- International NGOs (Doctors Without Borders, Red Cross health teams)

### How the Solution Integrates Their Needs
1. **Alert format designed for CHWs** — plain language, actionable steps, no medical jargon
2. **Offline-capable mobile interface** — alerts can be cached and viewed without connectivity
3. **Feedback loop** — CHWs can report back whether predictions matched ground reality, improving the model over time
4. **Localized risk thresholds** — different regions have different baseline disease levels; the system adapts thresholds to local context rather than applying global averages
5. **Co-design principle** — the alert templates and dashboard layout would be iteratively refined with CHW input (in a real deployment scenario)

---

## Demo Script (10-Minute Presentation + 2-Minute Q&A)

### [0:00 – 1:30] The Problem — Hook with Impact
"Climate change isn't just about rising seas and wildfires — it's rewriting the global disease map. Mosquitoes that carry dengue are now found in regions they've never reached before. Cholera outbreaks follow floods that are becoming more extreme every year. The WHO estimates 250,000 additional deaths per year by 2030 from climate-driven diseases alone. And the communities hit hardest — rural populations across the Global South — have no early warning system. We built one."

### [1:30 – 3:00] The Solution — What ClimaHealth AI Does
- Explain the three-signal approach: climate forecasting + historical disease correlation + NLP news monitoring
- Show the architecture diagram briefly
- Key differentiator: "Most disease surveillance is reactive — it tells you about an outbreak after people are already sick. ClimaHealth AI gives you 4-8 weeks of lead time."

### [3:00 – 6:00] Live Demo
1. Open the dashboard — show the global risk heatmap
2. Zoom into a region (e.g., Bangladesh or East Africa) — show the rising dengue risk prediction
3. Click into the region detail view:
   - Show the climate forecasts driving the prediction (temperature + precipitation charts)
   - Show the SHAP explainability panel: "45% temperature, 30% precipitation, 15% seasonal pattern, 10% news signals"
   - Show the NLP module detecting early outbreak reports from GDELT news data
4. Show a generated alert: "Dengue risk in [Region] predicted to increase 60% over the next 3 weeks. Recommended: distribute mosquito nets, pre-position medical supplies at district hospitals."
5. Toggle the time slider — show how risk changes week by week into the future

### [6:00 – 7:30] Technicality Deep Dive
- Walk through the ML pipeline: data ingestion → feature engineering → three parallel models → ensemble scoring → SHAP
- Highlight sophisticated elements: LSTM for non-linear climate patterns, ensemble methods for robustness, NLP as a real-time validation layer
- Show a brief code snippet or model architecture diagram

### [7:30 – 8:30] Ethical Considerations
- Privacy: no individual data, aggregate only
- Equity: built for Global South first, multilingual, low-bandwidth
- Transparency: every prediction is explainable
- Responsible AI: decision support, not automated decisions

### [8:30 – 9:30] Feasibility & Impact
- All data sources are free and publicly available
- Models run on consumer hardware (no GPU required)
- Modular architecture: add new diseases or regions by plugging in new data
- Real-world impact: even 2 weeks of lead time can reduce outbreak mortality by 20-30% through pre-positioning supplies and activating response teams

### [9:30 – 10:00] Close
"The next pandemic won't just come from a lab or a wet market — it might come from a heatwave, a flood, or a shifting monsoon pattern. ClimaHealth AI connects the dots between climate and disease, so communities can prepare before it's too late."

### [Q&A — 2 minutes]
Prepared answers for likely judge questions:
- **"How accurate is it?"** — "Our ensemble approach combines multiple models to reduce individual model errors. In our prototype, we validated against historical outbreaks in [region] and achieved [X]% precision. The NLP signal layer acts as a real-time reality check."
- **"How is this different from existing tools?"** — "Existing tools like HealthMap or ProMED are reactive. Climate models exist separately. We're the integration layer that connects climate forecasts to disease predictions with actionable lead time."
- **"Can this scale?"** — "The architecture is modular — adding a new disease or region means adding a new data source and retraining. All our data sources are global, so the system scales to any geography."

---

## Implementation Roadmap (Hackathon Timeline)

### Hours 0–2: Setup & Data
- Set up React + FastAPI project structure
- Download sample climate data from NASA POWER API
- Obtain historical disease data from WHO GHO

### Hours 2–5: Climate Forecasting Module
- Build LSTM/Prophet pipeline for temperature and precipitation forecasting
- Feature engineering: rolling averages, seasonal decomposition, anomaly detection

### Hours 5–8: Disease Prediction Model
- Train Random Forest / XGBoost ensemble on historical climate ↔ disease correlations
- Implement SHAP explainability for feature importance

### Hours 8–11: NLP Signal Detection
- Build news classifier using TF-IDF + Claude API
- Connect to GDELT API for real-time news ingestion
- Classify articles by disease type, severity, and location

### Hours 11–14: Ensemble Integration
- Build the risk scoring engine that combines all three model outputs
- Implement confidence scoring and uncertainty quantification

### Hours 14–18: Dashboard & Visualization
- Build React dashboard with Leaflet geospatial heatmap
- Add time-series charts (Recharts), drill-down views, and time slider
- Build alert generation system

### Hours 18–22: Polish & Ethics
- Implement colorblind-safe palette, high-contrast mode
- Add model explainability panel to the UI
- Add multilingual alert templates
- Testing and bug fixes

### Hours 22–24: Demo Prep
- Load compelling demo data (real historical outbreak scenario)
- Practice the 10-minute presentation
- Prepare Q&A answers

---

## Why This Wins

1. **Realistic & Impactful** — addresses a real, documented existential threat with real data sources
2. **Technically Sophisticated** — multi-model ML pipeline with three distinct AI approaches (time-series, classification, NLP) working together
3. **Deeply Ethical** — comprehensive privacy, equity, accessibility, and transparency story
4. **Human-Centered** — designed for the communities most affected, with their constraints (low-bandwidth, multilingual, non-technical users) built into the design
5. **Feasible** — all data sources are free, models are lightweight, architecture is modular
6. **Innovative** — the integration of climate forecasting + disease prediction + NLP outbreak detection into a single early warning system is novel
7. **Spans 4 Suggested Domains** — Healthcare, Environmental Science, Social Good, and Text Analysis
8. **Strong Narrative** — "Climate change is rewriting the disease map, and we're building the AI to read it first"
