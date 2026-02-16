"""
ClimaHealth AI — Synthetic Training Data Generator
====================================================
Generates realistic training data that mirrors documented climate-disease 
relationships for hackathon demonstration. In production, this would be 
replaced by real data from NASA POWER, WHO GHO, and GDELT APIs.

Data sources this simulates:
- NASA POWER API: Temperature, precipitation, humidity, solar radiation
- MODIS Satellite: NDVI (vegetation index)
- WHO GHO: Historical disease incidence by region
- GDELT Project: News-based outbreak signal scores
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

# =============================================================================
# REGION PROFILES — Based on real epidemiological literature
# =============================================================================

REGION_PROFILES = {
    "dhaka_bangladesh": {
        "lat": 23.8, "lon": 90.4,
        "base_temp": 28.0, "temp_amplitude": 6.0,   # Tropical monsoon
        "base_precip": 150, "precip_amplitude": 200,
        "base_humidity": 75, "humidity_amplitude": 12,
        "base_ndvi": 0.45,
        "diseases": {
            "dengue": {"base_rate": 120, "temp_threshold": 25, "temp_coeff": 8.5,
                       "precip_coeff": 0.15, "humidity_coeff": 1.2, "lag_weeks": 3},
            "cholera": {"base_rate": 45, "temp_threshold": 20, "temp_coeff": 3.2,
                        "precip_coeff": 0.25, "humidity_coeff": 0.8, "lag_weeks": 2},
        }
    },
    "nairobi_kenya": {
        "lat": -1.3, "lon": 36.8,
        "base_temp": 22.0, "temp_amplitude": 3.0,   # Highland tropical
        "base_precip": 80, "precip_amplitude": 120,
        "base_humidity": 62, "humidity_amplitude": 15,
        "base_ndvi": 0.55,
        "diseases": {
            "malaria": {"base_rate": 85, "temp_threshold": 22, "temp_coeff": 12.0,
                        "precip_coeff": 0.20, "humidity_coeff": 1.5, "lag_weeks": 4},
        }
    },
    "recife_brazil": {
        "lat": -8.05, "lon": -34.9,
        "base_temp": 26.5, "temp_amplitude": 3.5,   # Tropical Atlantic
        "base_precip": 120, "precip_amplitude": 150,
        "base_humidity": 76, "humidity_amplitude": 8,
        "base_ndvi": 0.50,
        "diseases": {
            "zika": {"base_rate": 60, "temp_threshold": 24, "temp_coeff": 7.0,
                     "precip_coeff": 0.12, "humidity_coeff": 1.0, "lag_weeks": 3},
            "dengue": {"base_rate": 95, "temp_threshold": 25, "temp_coeff": 7.8,
                       "precip_coeff": 0.14, "humidity_coeff": 1.1, "lag_weeks": 3},
        }
    },
    "chittagong_bangladesh": {
        "lat": 22.3, "lon": 91.8,
        "base_temp": 27.0, "temp_amplitude": 6.5,
        "base_precip": 200, "precip_amplitude": 280,
        "base_humidity": 78, "humidity_amplitude": 10,
        "base_ndvi": 0.40,
        "diseases": {
            "cholera": {"base_rate": 70, "temp_threshold": 20, "temp_coeff": 4.5,
                        "precip_coeff": 0.30, "humidity_coeff": 0.9, "lag_weeks": 2},
        }
    },
    "lagos_nigeria": {
        "lat": 6.5, "lon": 3.4,
        "base_temp": 27.5, "temp_amplitude": 3.0,
        "base_precip": 130, "precip_amplitude": 180,
        "base_humidity": 77, "humidity_amplitude": 10,
        "base_ndvi": 0.42,
        "diseases": {
            "malaria": {"base_rate": 200, "temp_threshold": 22, "temp_coeff": 10.0,
                        "precip_coeff": 0.18, "humidity_coeff": 1.3, "lag_weeks": 4},
        }
    },
    "manaus_brazil": {
        "lat": -3.1, "lon": -60.0,
        "base_temp": 28.5, "temp_amplitude": 2.0,   # Equatorial
        "base_precip": 200, "precip_amplitude": 150,
        "base_humidity": 82, "humidity_amplitude": 6,
        "base_ndvi": 0.65,
        "diseases": {
            "dengue": {"base_rate": 110, "temp_threshold": 25, "temp_coeff": 9.0,
                       "precip_coeff": 0.16, "humidity_coeff": 1.3, "lag_weeks": 3},
        }
    },
}


def generate_climate_timeseries(profile, n_weeks=520):
    """Generate 10 years of weekly climate data with realistic patterns."""
    weeks = np.arange(n_weeks)
    
    # Seasonal cycle (annual)
    season = 2 * np.pi * weeks / 52
    
    # Temperature: seasonal + trend + noise
    temp_trend = 0.02 * weeks / 52  # ~0.02°C/year warming trend
    temp = (profile["base_temp"]
            + profile["temp_amplitude"] * np.sin(season - np.pi/6)
            + temp_trend
            + np.random.normal(0, 1.2, n_weeks))
    
    # Precipitation: seasonal + ENSO-like interannual variability + noise
    enso_cycle = 15 * np.sin(2 * np.pi * weeks / (52 * 3.7))  # ~3.7 year quasi-cycle
    precip = np.maximum(0,
        profile["base_precip"]
        + profile["precip_amplitude"] * np.sin(season + np.pi/4)
        + enso_cycle
        + np.random.exponential(20, n_weeks)
    )
    
    # Humidity: correlated with temp and precip
    humidity = (profile["base_humidity"]
                + profile["humidity_amplitude"] * np.sin(season)
                + 0.05 * precip
                + np.random.normal(0, 3, n_weeks))
    humidity = np.clip(humidity, 30, 100)
    
    # NDVI: inverse lag with temperature, positive with moderate precip
    ndvi = (profile["base_ndvi"]
            + 0.08 * np.sin(season + np.pi/2)
            + 0.0002 * precip
            - 0.005 * np.maximum(0, temp - 30)
            + np.random.normal(0, 0.03, n_weeks))
    ndvi = np.clip(ndvi, 0.1, 0.9)
    
    # Temperature anomaly (deviation from 30-year "normal")
    temp_anomaly = temp - profile["base_temp"]
    
    return pd.DataFrame({
        "week_index": weeks,
        "temperature": np.round(temp, 2),
        "temp_anomaly": np.round(temp_anomaly, 2),
        "precipitation": np.round(precip, 1),
        "humidity": np.round(humidity, 1),
        "ndvi": np.round(ndvi, 3),
    })


def generate_disease_incidence(climate_df, disease_params, n_weeks=520):
    """
    Generate disease incidence based on climate drivers with realistic
    epidemiological relationships.
    
    The model encodes documented relationships:
    - Dengue/Zika: Strong positive association with temperature >25°C and precipitation
    - Malaria: Positive with temperature >22°C, strong precipitation effect  
    - Cholera: Strong precipitation/flooding effect, temperature secondary
    """
    p = disease_params
    cases = np.zeros(n_weeks)
    
    for i in range(n_weeks):
        # Lagged climate features (using the documented lag for each disease)
        lag = min(i, p["lag_weeks"])
        temp_lag = climate_df["temperature"].iloc[i - lag]
        precip_lag = climate_df["precipitation"].iloc[i - lag]
        humidity_lag = climate_df["humidity"].iloc[i - lag]
        
        # Temperature effect (nonlinear — kicks in above threshold)
        temp_effect = max(0, temp_lag - p["temp_threshold"]) * p["temp_coeff"]
        
        # Precipitation effect (positive, with saturation)
        precip_effect = precip_lag * p["precip_coeff"]
        
        # Humidity effect
        humidity_effect = max(0, humidity_lag - 60) * p["humidity_coeff"]
        
        # Seasonal multiplier
        week_of_year = i % 52
        seasonal = 1.0 + 0.4 * np.sin(2 * np.pi * week_of_year / 52 - np.pi/4)
        
        # Combine: base rate + climate effects + seasonal + noise
        expected = (p["base_rate"] + temp_effect + precip_effect + humidity_effect) * seasonal
        
        # Add occasional outbreak spikes (representing real epidemic dynamics)
        if np.random.random() < 0.03:  # ~3% chance of outbreak spike per week
            expected *= np.random.uniform(2.0, 4.0)
        
        # Poisson-distributed case counts
        cases[i] = max(0, np.random.poisson(max(1, expected)))
    
    return cases.astype(int)


def generate_nlp_signals(climate_df, disease_cases, n_weeks=520):
    """
    Generate simulated NLP outbreak signal scores.
    In production, these come from GDELT/ProMED-mail text classification.
    Signal strength correlates with actual case surges.
    """
    signals = np.zeros(n_weeks)
    
    for i in range(2, n_weeks):
        # NLP signals tend to appear 1-2 weeks AFTER cases start rising
        case_trend = disease_cases[i] - disease_cases[i-2]
        
        # Base signal from case trend
        if case_trend > 0:
            signals[i] = min(1.0, case_trend / 200 + np.random.uniform(0, 0.2))
        else:
            signals[i] = max(0, np.random.uniform(-0.05, 0.15))
        
        # Occasional false positives (noise in news data)
        if np.random.random() < 0.05:
            signals[i] = min(1.0, signals[i] + np.random.uniform(0.1, 0.3))
    
    return np.round(np.clip(signals, 0, 1), 3)


def compute_risk_score(cases, max_cases=800):
    """Convert raw case counts to 0-100 risk score."""
    return np.clip(np.round((cases / max_cases) * 100), 0, 100).astype(int)


def create_feature_matrix(climate_df, nlp_signals, lag_weeks=4):
    """
    Create the ML feature matrix with lagged climate features,
    rolling statistics, and NLP signal scores.
    """
    df = climate_df.copy()
    df["nlp_signal"] = nlp_signals
    
    # Lagged features (key for prediction — we predict FUTURE outbreaks from PAST climate)
    for lag in range(1, lag_weeks + 1):
        df[f"temp_lag{lag}"] = df["temperature"].shift(lag)
        df[f"precip_lag{lag}"] = df["precipitation"].shift(lag)
        df[f"humidity_lag{lag}"] = df["humidity"].shift(lag)
        df[f"ndvi_lag{lag}"] = df["ndvi"].shift(lag)
    
    # Rolling statistics (capture trends)
    for window in [4, 8, 12]:
        df[f"temp_rolling{window}"] = df["temperature"].rolling(window).mean()
        df[f"precip_rolling{window}"] = df["precipitation"].rolling(window).mean()
        df[f"humidity_rolling{window}"] = df["humidity"].rolling(window).mean()
    
    # Rate of change
    df["temp_change_4w"] = df["temperature"] - df["temperature"].shift(4)
    df["precip_change_4w"] = df["precipitation"] - df["precipitation"].shift(4)
    
    # Seasonal features
    df["week_of_year"] = df["week_index"] % 52
    df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    
    # Interaction features
    df["temp_x_precip"] = df["temperature"] * df["precipitation"]
    df["temp_x_humidity"] = df["temperature"] * df["humidity"]
    
    # Drop rows with NaN from lagging
    df = df.dropna().reset_index(drop=True)
    
    return df


def generate_full_dataset():
    """Generate the complete training dataset for all regions and diseases."""
    
    all_data = []
    
    for region_name, profile in REGION_PROFILES.items():
        print(f"\n{'='*60}")
        print(f"Generating data for: {region_name}")
        print(f"{'='*60}")
        
        # Generate 10 years of climate data
        climate_df = generate_climate_timeseries(profile, n_weeks=520)
        print(f"  Climate data: {len(climate_df)} weeks")
        print(f"  Temp range: {climate_df['temperature'].min():.1f} - {climate_df['temperature'].max():.1f}°C")
        print(f"  Precip range: {climate_df['precipitation'].min():.0f} - {climate_df['precipitation'].max():.0f}mm")
        
        for disease_name, disease_params in profile["diseases"].items():
            print(f"\n  Disease: {disease_name}")
            
            # Generate disease incidence
            cases = generate_disease_incidence(climate_df, disease_params, n_weeks=520)
            print(f"    Total cases over 10yr: {cases.sum():,}")
            print(f"    Weekly avg: {cases.mean():.1f}, max: {cases.max()}")
            
            # Generate NLP signals
            nlp_signals = generate_nlp_signals(climate_df, cases, n_weeks=520)
            
            # Create feature matrix
            features_df = create_feature_matrix(climate_df, nlp_signals, lag_weeks=4)
            
            # Align targets (risk score) with feature matrix
            risk_scores = compute_risk_score(cases)
            aligned_cases = cases[len(cases) - len(features_df):]
            aligned_risk = risk_scores[len(risk_scores) - len(features_df):]
            
            features_df["cases"] = aligned_cases
            features_df["risk_score"] = aligned_risk
            features_df["outbreak"] = (aligned_risk >= 60).astype(int)  # Binary outbreak label
            features_df["region"] = region_name
            features_df["disease"] = disease_name
            
            all_data.append(features_df)
            print(f"    Feature matrix: {features_df.shape}")
            print(f"    Outbreak weeks: {features_df['outbreak'].sum()} / {len(features_df)} ({features_df['outbreak'].mean()*100:.1f}%)")
    
    # Combine all regions/diseases
    full_df = pd.concat(all_data, ignore_index=True)
    
    return full_df


def generate_news_corpus():
    """
    Generate a synthetic corpus of outbreak-related news headlines for NLP training.
    In production, this comes from GDELT API and ProMED-mail.
    """
    
    outbreak_headlines = [
        "WHO reports surge in dengue cases across Southeast Asia",
        "Bangladesh hospitals overwhelmed by fever patients amid monsoon",
        "Dengue fever outbreak kills dozens in Dhaka slums",
        "Aedes mosquito breeding sites multiply after heavy rainfall",
        "Emergency declared as malaria cases spike in Kenya highlands",
        "Climate change pushing malaria into new altitudes in East Africa",
        "Cholera epidemic follows severe flooding in coastal Bangladesh",
        "Water contamination leads to cholera outbreak in refugee camps",
        "Zika virus detected in new regions of northeastern Brazil",
        "Pregnant women warned as Zika cases rise in Recife area",
        "Record dengue hospitalizations reported in Amazonas state",
        "Mosquito-borne diseases surge as temperatures hit new highs",
        "Health officials scramble to contain waterborne disease outbreak",
        "Malaria drug shortages reported in Lagos public hospitals",
        "Vector control teams deployed to combat dengue in urban areas",
        "Scientists warn of climate-driven expansion of disease vectors",
        "Flood-damaged sanitation systems fuel cholera spread",
        "Emergency vaccination campaign launched against cholera",
        "Rising temperatures accelerate mosquito reproduction cycles",
        "Tropical disease surveillance systems detect anomalous case patterns",
    ]
    
    non_outbreak_headlines = [
        "New hospital opens in Dhaka with expanded ICU capacity",
        "Government announces healthcare infrastructure investment plan",
        "Annual monsoon season begins across South Asia",
        "Tourism sector reports strong growth in tropical destinations",
        "Agricultural output exceeds expectations despite weather challenges",
        "Education reform bill passes in national parliament",
        "Technology companies expand operations in emerging markets",
        "International trade negotiations progress at summit",
        "New study examines long-term effects of urbanization",
        "Sports championship draws large crowds in capital city",
        "Environmental conservation program shows positive results",
        "Transportation improvements reduce commute times in metro areas",
        "Cultural festival celebrates regional traditions and cuisine",
        "Research university receives funding for climate studies",
        "Renewable energy installation reaches milestone in developing nations",
        "Telecommunications network expansion improves rural connectivity",
        "Banking sector introduces new digital financial services",
        "Community health workers receive updated training programs",
        "Weather forecast predicts normal seasonal patterns this year",
        "Public parks renovation project completed in urban district",
    ]
    
    headlines = []
    labels = []
    
    for h in outbreak_headlines:
        headlines.append(h)
        labels.append(1)
        # Augment with slight variations
        for _ in range(4):
            words = h.split()
            if len(words) > 4:
                # Random word swap/drop for augmentation
                idx = np.random.randint(1, len(words) - 1)
                augmented = words[:idx] + words[idx+1:]
                headlines.append(" ".join(augmented))
                labels.append(1)
    
    for h in non_outbreak_headlines:
        headlines.append(h)
        labels.append(0)
        for _ in range(4):
            words = h.split()
            if len(words) > 4:
                idx = np.random.randint(1, len(words) - 1)
                augmented = words[:idx] + words[idx+1:]
                headlines.append(" ".join(augmented))
                labels.append(0)
    
    return pd.DataFrame({"headline": headlines, "is_outbreak": labels})


if __name__ == "__main__":
    print("=" * 60)
    print("ClimaHealth AI — Training Data Generator")
    print("=" * 60)
    
    # Generate climate-disease dataset
    data_dir = os.path.join(os.path.dirname(__file__))
    
    full_df = generate_full_dataset()
    output_path = os.path.join(data_dir, "training_data.csv")
    full_df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Full dataset saved: {output_path}")
    print(f"Shape: {full_df.shape}")
    print(f"Regions: {full_df['region'].nunique()}")
    print(f"Diseases: {full_df['disease'].nunique()}")
    print(f"Outbreak rate: {full_df['outbreak'].mean()*100:.1f}%")
    
    # Generate NLP corpus
    news_df = generate_news_corpus()
    news_path = os.path.join(data_dir, "news_corpus.csv")
    news_df.to_csv(news_path, index=False)
    print(f"\nNews corpus saved: {news_path}")
    print(f"Headlines: {len(news_df)} ({news_df['is_outbreak'].sum()} outbreak, {(~news_df['is_outbreak'].astype(bool)).sum()} non-outbreak)")
