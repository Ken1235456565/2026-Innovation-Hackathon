"""
ClimaHealth AI — Real Data Fetcher
=====================================
Fetches REAL climate, disease, and news data from public APIs.

Data Sources:
  1. NASA POWER API — Temperature, precipitation, humidity (free, no auth)
  2. WHO GHO OData API — Disease incidence by country (free, no auth)  
  3. GDELT DOC 2.0 API — Global news articles about disease outbreaks (free, no auth)

Usage:
    pip install requests pandas
    python fetch_real_data.py

All APIs are free and require NO authentication.
This script fetches 10 years of monthly climate data for all 6 regions,
disease incidence from WHO, and recent outbreak news from GDELT.
"""

import requests
import pandas as pd
import json
import os
import time
from datetime import datetime, timedelta

# Output directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "real")
os.makedirs(DATA_DIR, exist_ok=True)


# =============================================================================
# REGION DEFINITIONS
# =============================================================================

REGIONS = {
    "dhaka_bangladesh": {"lat": 23.8, "lon": 90.4, "country_code": "BGD", "fips": "BG"},
    "nairobi_kenya": {"lat": -1.3, "lon": 36.8, "country_code": "KEN", "fips": "KE"},
    "recife_brazil": {"lat": -8.05, "lon": -34.9, "country_code": "BRA", "fips": "BR"},
    "chittagong_bangladesh": {"lat": 22.3, "lon": 91.8, "country_code": "BGD", "fips": "BG"},
    "lagos_nigeria": {"lat": 6.5, "lon": 3.4, "country_code": "NGA", "fips": "NI"},
    "manaus_brazil": {"lat": -3.1, "lon": -60.0, "country_code": "BRA", "fips": "BR"},
}


# =============================================================================
# 1. NASA POWER API — Climate Data
# =============================================================================
# Docs: https://power.larc.nasa.gov/docs/services/api/
# Parameters: T2M (temp), PRECTOTCORR (precip), RH2M (humidity)
# Free, no auth, 0.5° x 0.625° resolution

def fetch_nasa_power(region_name, lat, lon, start_year=2015, end_year=2024):
    """
    Fetch monthly climate data from NASA POWER API.
    
    Parameters fetched:
    - T2M: Temperature at 2 Meters (°C)
    - T2M_MAX: Max Temperature at 2 Meters (°C) 
    - T2M_MIN: Min Temperature at 2 Meters (°C)
    - PRECTOTCORR: Precipitation Corrected (mm/day)
    - RH2M: Relative Humidity at 2 Meters (%)
    - GWETTOP: Surface Soil Wetness (fraction)
    """
    
    url = (
        f"https://power.larc.nasa.gov/api/temporal/monthly/point"
        f"?parameters=T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,GWETTOP"
        f"&community=AG"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={start_year}"
        f"&end={end_year}"
        f"&format=JSON"
    )
    
    print(f"  Fetching NASA POWER data for {region_name}...")
    print(f"    URL: {url}")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # Extract parameter data
        params = data.get("properties", {}).get("parameter", {})
        
        # Build DataFrame
        records = []
        if "T2M" in params:
            for date_key, temp in params["T2M"].items():
                if date_key == "ANN":  # Skip annual average
                    continue
                year = int(date_key[:4])
                month = int(date_key[4:])
                records.append({
                    "year": year,
                    "month": month,
                    "date": f"{year}-{month:02d}",
                    "temperature": temp if temp != -999 else None,
                    "temp_max": params.get("T2M_MAX", {}).get(date_key),
                    "temp_min": params.get("T2M_MIN", {}).get(date_key),
                    "precipitation_mm_day": params.get("PRECTOTCORR", {}).get(date_key),
                    "humidity": params.get("RH2M", {}).get(date_key),
                    "soil_wetness": params.get("GWETTOP", {}).get(date_key),
                    "region": region_name,
                    "lat": lat,
                    "lon": lon,
                })
        
        df = pd.DataFrame(records)
        
        # Replace -999 (NASA's missing value code) with None
        for col in ["temperature", "temp_max", "temp_min", "precipitation_mm_day", "humidity", "soil_wetness"]:
            if col in df.columns:
                df[col] = df[col].replace(-999, None)
        
        # Convert daily precipitation to monthly total (approximate)
        if "precipitation_mm_day" in df.columns:
            df["precipitation_mm_month"] = df["precipitation_mm_day"] * 30
        
        print(f"    ✓ Got {len(df)} months of data")
        print(f"    Temp range: {df['temperature'].min():.1f} - {df['temperature'].max():.1f}°C")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"    ✗ Error: {e}")
        return pd.DataFrame()


def fetch_all_nasa_data():
    """Fetch NASA POWER data for all regions."""
    print("\n" + "=" * 60)
    print("  FETCHING NASA POWER CLIMATE DATA")
    print("=" * 60)
    
    all_climate = []
    for region_name, info in REGIONS.items():
        df = fetch_nasa_power(region_name, info["lat"], info["lon"])
        if not df.empty:
            all_climate.append(df)
        time.sleep(2)  # Be nice to the API
    
    if all_climate:
        combined = pd.concat(all_climate, ignore_index=True)
        output_path = os.path.join(DATA_DIR, "nasa_power_climate.csv")
        combined.to_csv(output_path, index=False)
        print(f"\n  ✓ Saved {len(combined)} records to {output_path}")
        return combined
    
    return pd.DataFrame()


# =============================================================================
# 2. WHO GHO OData API — Disease Incidence Data
# =============================================================================
# Docs: https://www.who.int/data/gho/info/gho-odata-api
# Endpoint: https://ghoapi.azureedge.net/api/{INDICATOR_CODE}
# Free, no auth

# WHO GHO indicator codes for relevant diseases
WHO_INDICATORS = {
    # Malaria
    "MALARIA_EST_CASES": "Estimated number of malaria cases",
    "MALARIA_EST_DEATHS": "Estimated number of malaria deaths",
    "MALARIA_EST_INCIDENCE": "Estimated malaria incidence (per 1000 population at risk)",
    
    # Dengue  
    # Note: WHO doesn't have a dedicated dengue indicator in GHO;
    # we use the broader NTD indicator and supplement with GDELT news data
    
    # Cholera
    "CHOLERA_0000000001": "Number of reported cases of cholera",
    "CHOLERA_0000000002": "Number of reported deaths from cholera",
    "CHOLERA_0000000003": "Cholera case fatality rate (%)",
}

# Country codes for our regions
TARGET_COUNTRIES = ["BGD", "KEN", "BRA", "NGA"]


def fetch_who_indicator(indicator_code, indicator_name):
    """
    Fetch data for a specific WHO GHO indicator.
    
    API format: https://ghoapi.azureedge.net/api/{INDICATOR_CODE}
    Filter by country: ?$filter=SpatialDim eq 'BGD'
    """
    
    # Build filter for target countries
    country_filter = " or ".join([f"SpatialDim eq '{c}'" for c in TARGET_COUNTRIES])
    
    url = (
        f"https://ghoapi.azureedge.net/api/{indicator_code}"
        f"?$filter=({country_filter})"
    )
    
    print(f"  Fetching WHO data: {indicator_name}...")
    print(f"    Indicator: {indicator_code}")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        records = []
        for item in data.get("value", []):
            records.append({
                "indicator_code": indicator_code,
                "indicator_name": indicator_name,
                "country_code": item.get("SpatialDim"),
                "year": item.get("TimeDim"),
                "value": item.get("NumericValue"),
                "value_low": item.get("Low"),
                "value_high": item.get("High"),
                "dim1": item.get("Dim1"),  # e.g., sex, age group
            })
        
        df = pd.DataFrame(records)
        print(f"    ✓ Got {len(df)} records")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"    ✗ Error: {e}")
        return pd.DataFrame()


def fetch_all_who_data():
    """Fetch all WHO GHO disease data."""
    print("\n" + "=" * 60)
    print("  FETCHING WHO GLOBAL HEALTH OBSERVATORY DATA")
    print("=" * 60)
    
    all_who = []
    for code, name in WHO_INDICATORS.items():
        df = fetch_who_indicator(code, name)
        if not df.empty:
            all_who.append(df)
        time.sleep(1)
    
    if all_who:
        combined = pd.concat(all_who, ignore_index=True)
        output_path = os.path.join(DATA_DIR, "who_disease_data.csv")
        combined.to_csv(output_path, index=False)
        print(f"\n  ✓ Saved {len(combined)} records to {output_path}")
        return combined
    
    return pd.DataFrame()


# =============================================================================
# 3. GDELT DOC 2.0 API — News-Based Outbreak Signals
# =============================================================================
# Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
# Endpoint: https://api.gdeltproject.org/api/v2/doc/doc
# Free, no auth, searches last 3 months of global news

DISEASE_QUERIES = {
    "dengue": '(dengue OR "dengue fever" OR "aedes mosquito")',
    "malaria": '(malaria OR "anopheles mosquito" OR antimalarial)',
    "cholera": '(cholera OR "vibrio cholerae" OR "waterborne disease outbreak")',
    "zika": '(zika OR "zika virus" OR microcephaly)',
    "disease_outbreak_general": '("disease outbreak" OR epidemic OR "public health emergency")',
}

GDELT_COUNTRY_FILTERS = {
    "BGD": "sourcecountry:BG",  # GDELT uses FIPS codes
    "KEN": "sourcecountry:KE",
    "BRA": "sourcecountry:BR",
    "NGA": "sourcecountry:NI",
}


def fetch_gdelt_articles(disease, query, country_fips=None, max_records=75):
    """
    Fetch recent news articles about a disease from GDELT DOC 2.0 API.
    
    API returns articles from the last 3 months matching the query.
    Returns: title, url, date, source country, language, domain
    """
    
    full_query = f"{query} outbreak"
    if country_fips:
        full_query += f" sourcecountry:{country_fips}"
    
    url = (
        f"https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={requests.utils.quote(full_query)}"
        f"&mode=artlist"
        f"&maxrecords={max_records}"
        f"&format=json"
        f"&sort=datedesc"
    )
    
    print(f"  Fetching GDELT articles: {disease}" + (f" ({country_fips})" if country_fips else ""))
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        articles = data.get("articles", [])
        records = []
        for art in articles:
            records.append({
                "disease_query": disease,
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "url_mobile": art.get("url_mobile", ""),
                "date": art.get("seendate", ""),
                "source_country": art.get("sourcecountry", ""),
                "language": art.get("language", ""),
                "domain": art.get("domain", ""),
                "social_image": art.get("socialimage", ""),
            })
        
        df = pd.DataFrame(records)
        print(f"    ✓ Got {len(df)} articles")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"    ✗ Error: {e}")
        return pd.DataFrame()


def fetch_gdelt_timeline(disease, query, timespan="3m"):
    """
    Fetch a volume timeline for disease mentions from GDELT.
    Shows the proportion of global news matching the query over time.
    """
    
    url = (
        f"https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={requests.utils.quote(query + ' outbreak')}"
        f"&mode=timelinevol"
        f"&timespan={timespan}"
        f"&format=json"
    )
    
    print(f"  Fetching GDELT timeline: {disease}...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        records = []
        timeline = data.get("timeline", [])
        if timeline and len(timeline) > 0:
            for point in timeline[0].get("data", []):
                records.append({
                    "disease": disease,
                    "date": point.get("date", ""),
                    "volume": point.get("value", 0),
                })
        
        df = pd.DataFrame(records)
        print(f"    ✓ Got {len(df)} timeline points")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"    ✗ Error: {e}")
        return pd.DataFrame()


def fetch_all_gdelt_data():
    """Fetch all GDELT disease outbreak news data."""
    print("\n" + "=" * 60)
    print("  FETCHING GDELT NEWS & OUTBREAK SIGNAL DATA")
    print("=" * 60)
    
    # --- Article lists ---
    all_articles = []
    for disease, query in DISEASE_QUERIES.items():
        # Global articles
        df = fetch_gdelt_articles(disease, query, max_records=75)
        if not df.empty:
            all_articles.append(df)
        time.sleep(1)
        
        # Country-specific articles for key regions
        for country, fips_filter in GDELT_COUNTRY_FILTERS.items():
            fips_code = fips_filter.split(":")[1]
            df = fetch_gdelt_articles(disease, query, country_fips=fips_code, max_records=25)
            if not df.empty:
                all_articles.append(df)
            time.sleep(1)
    
    if all_articles:
        articles_combined = pd.concat(all_articles, ignore_index=True).drop_duplicates(subset=["url"])
        output_path = os.path.join(DATA_DIR, "gdelt_outbreak_articles.csv")
        articles_combined.to_csv(output_path, index=False)
        print(f"\n  ✓ Saved {len(articles_combined)} unique articles to {output_path}")
    
    # --- Volume timelines ---
    all_timelines = []
    for disease, query in DISEASE_QUERIES.items():
        df = fetch_gdelt_timeline(disease, query)
        if not df.empty:
            all_timelines.append(df)
        time.sleep(1)
    
    if all_timelines:
        timelines_combined = pd.concat(all_timelines, ignore_index=True)
        output_path = os.path.join(DATA_DIR, "gdelt_disease_timelines.csv")
        timelines_combined.to_csv(output_path, index=False)
        print(f"  ✓ Saved {len(timelines_combined)} timeline points to {output_path}")
    
    return articles_combined if all_articles else pd.DataFrame()


# =============================================================================
# MAIN
# =============================================================================

def print_banner(text, char="="):
    width = 60
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


if __name__ == "__main__":
    print_banner("ClimaHealth AI — Real Data Fetcher", "█")
    print("  Fetching from NASA POWER, WHO GHO, and GDELT APIs")
    print("  All APIs are free and require no authentication")
    print(f"{'█' * 60}")
    
    start = time.time()
    
    # 1. NASA POWER — Climate data (10 years, 6 regions)
    climate_df = fetch_all_nasa_data()
    
    # 2. WHO GHO — Disease incidence data
    who_df = fetch_all_who_data()
    
    # 3. GDELT — News outbreak signals
    gdelt_df = fetch_all_gdelt_data()
    
    # Summary
    elapsed = time.time() - start
    print_banner("DATA FETCH COMPLETE", "█")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"  Output directory: {DATA_DIR}/")
    print(f"\n  Files created:")
    for f in sorted(os.listdir(DATA_DIR)):
        filepath = os.path.join(DATA_DIR, f)
        size = os.path.getsize(filepath)
        print(f"    {f:40s} {size/1024:.1f} KB")
    
    print(f"\n  Data Summary:")
    if not climate_df.empty:
        print(f"    NASA POWER:  {len(climate_df)} monthly records across {climate_df['region'].nunique()} regions")
    if not who_df.empty:
        print(f"    WHO GHO:     {len(who_df)} disease records across {who_df['country_code'].nunique()} countries")
    if not gdelt_df.empty:
        print(f"    GDELT News:  {len(gdelt_df)} unique articles across {gdelt_df['disease_query'].nunique()} disease queries")
    
    print(f"\n  Next: Run 'python train_real.py' to train models on this real data.")
    print(f"{'█' * 60}\n")
