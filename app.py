"""
ClimaHealth AI – Enhanced with Personalized Action Center
"""
import os,sys,shutil,time,numpy as np,pandas as pd,gradio as gr,requests
import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
from collections import Counter
import html as html_lib
import json

try:
    import folium
except ImportError:
    os.system("pip install folium -q")
    import folium

print("="*70);print("ClimaHealth AI – Initializing...");print("="*70)
from climate_forecaster import ClimateForecaster
from disease_predictor import DiseasePredictor
from nlp_detector import OutbreakSignalDetector
from ensemble import EnsembleRiskEngine
from generate_training_data import (REGION_PROFILES,generate_full_dataset,generate_news_corpus,generate_climate_timeseries,generate_disease_incidence,generate_nlp_signals,create_feature_matrix,compute_risk_score)

REAL_DATA={"nasa":None,"who":None,"gdelt":{}}
DATA_SOURCE="synthetic"
REGIONS_API={
    "Nairobi, Kenya":{"lat":-1.3,"lon":36.8,"country":"KEN"},
    "Chittagong, Bangladesh":{"lat":22.3,"lon":91.8,"country":"BGD"},
    "Lagos, Nigeria":{"lat":6.5,"lon":3.4,"country":"NGA"},
    "Dhaka, Bangladesh":{"lat":23.8,"lon":90.4,"country":"BGD"},
}

WHO_DISEASE_INDICATORS={
    "malaria":{"codes":["MALARIA_EST_CASES","MALARIA_EST_DEATHS","MALARIA_EST_INCIDENCE"],"names":{"MALARIA_EST_CASES":"Malaria cases","MALARIA_EST_DEATHS":"Malaria deaths","MALARIA_EST_INCIDENCE":"Malaria incidence/1000"}},
    "cholera":{"codes":["CHOLERA_0000000001","CHOLERA_0000000002"],"names":{"CHOLERA_0000000001":"Cholera cases","CHOLERA_0000000002":"Cholera deaths"}},
}

def generate_global_map():
    # 创建基础地图
    m = folium.Map(location=[5, 20], zoom_start=2, tiles='CartoDB dark_matter', width='100%', height='400px')
    # 定义疾病颜色
    risk_colors = {"Malaria": "red", "Cholera": "blue"}
    # 为每个区域添加标记
    for r_name, info in REGIONS.items():
        if info['api_name'] in REGIONS_API:
            lat = REGIONS_API[info['api_name']]['lat']
            lon = REGIONS_API[info['api_name']]['lon']
            color = risk_colors.get(info['disease_name'], 'green')
            
            # 创建弹出窗口内容
            popup = f"""
            <div style='font-family:monospace; font-size:12px; color:black;'>
               <b>{info['api_name']}</b><br>
               Disease: {info['disease_name']}<br>
               Pop: {info['pop']}<br>
               CHWs: {info['chws']}<br>
               Vector: {info['vector']}
            </div>
            """
            
            # 添加标记
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup, max_width=200),
                tooltip=f"{r_name}",
                icon=folium.Icon(color=color, icon="heart", prefix="fa")
            ).add_to(m)
    # 将地图渲染为HTML并嵌入iframe
    map_doc = html_lib.escape(m.get_root().render(), quote=True)
    return f'<iframe srcdoc="{map_doc}" style="width:100%;height:400px;border:none;border-radius:10px;display:block;background:#0a1628;"></iframe>'
    
def fetch_real_data():
    global REAL_DATA,DATA_SOURCE
    print("\n📡 Fetching REAL data...")
    sc=0
    try:
        print("  [1/3] NASA POWER...")
        nr=[]
        for name,info in REGIONS_API.items():
            url=f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=T2M,PRECTOTCORR,RH2M&community=AG&longitude={info['lon']}&latitude={info['lat']}&start=2015&end=2024&format=JSON"
            r=requests.get(url,timeout=30);r.raise_for_status()
            params=r.json().get("properties",{}).get("parameter",{})
            if "T2M" in params:
                for dk,temp in params["T2M"].items():
                    if dk=="ANN":continue
                    if int(dk[4:])>12:continue
                    nr.append({"region":name,"year":int(dk[:4]),"month":int(dk[4:]),
                        "temperature_C":round(temp,2) if temp!=-999 else None,
                        "precip_mm_day":round(params.get("PRECTOTCORR",{}).get(dk,-999),2) if params.get("PRECTOTCORR",{}).get(dk,-999)!=-999 else None,
                        "humidity_pct":round(params.get("RH2M",{}).get(dk,-999),1) if params.get("RH2M",{}).get(dk,-999)!=-999 else None})
            time.sleep(0.5)
        REAL_DATA["nasa"]=pd.DataFrame(nr);print(f"✓ {len(REAL_DATA['nasa'])} records");sc+=1
    except Exception as e:print(f"     ✗ {e}")

    try:
        print("  [2/3] WHO GHO...")
        wr=[];countries=["BGD","KEN","NGA"]
        cf=" or ".join([f"SpatialDim eq '{c}'" for c in countries])
        all_codes={}
        for d,info in WHO_DISEASE_INDICATORS.items():
            for code in info["codes"]:all_codes[code]=info["names"].get(code,code)
        for code,iname in all_codes.items():
            try:
                url=f"https://ghoapi.azureedge.net/api/{code}?$filter=({cf})"
                r=requests.get(url,timeout=20);r.raise_for_status()
                for item in r.json().get("value",[]):
                    val=item.get("NumericValue")
                    if val is not None:wr.append({"indicator":iname,"indicator_code":code,"country":item.get("SpatialDim"),"year":item.get("TimeDim"),"value":val})
            except:pass
            time.sleep(0.3)
        REAL_DATA["who"]=pd.DataFrame(wr);print(f"✓ {len(REAL_DATA['who'])} records");sc+=1
    except Exception as e:print(f"     ✗ {e}")

    try:
        print("  [3/3] GDELT...")
        for disease,query in {"malaria":"malaria outbreak cases deaths mosquito","cholera":"cholera outbreak water contamination cases"}.items():
            try:
                url=f"https://api.gdeltproject.org/api/v2/doc/doc?query={requests.utils.quote(query)}&mode=artlist&maxrecords=20&format=json&sort=datedesc"
                r=requests.get(url,timeout=15);r.raise_for_status()
                arts=[{"title":a.get("title","").strip(),"url":a.get("url",""),"date":a.get("seendate","")[:10],"domain":a.get("domain","")} for a in r.json().get("articles",[]) if a.get("title","").strip() and len(a.get("title",""))>15]
                REAL_DATA["gdelt"][disease]=arts;print(f"✓ {disease}: {len(arts)}")
            except Exception as ex:REAL_DATA["gdelt"][disease]=[];print(f"     ⚠ {disease}: {ex}")
            time.sleep(0.5)
        if any(len(v)>0 for v in REAL_DATA["gdelt"].values()):sc+=1
    except Exception as e:print(f"     ✗ {e}")

    if sc==3:DATA_SOURCE="real"
    elif sc>0:DATA_SOURCE="mixed"
    print(f"\n✓ Status: {DATA_SOURCE} ({sc}/3 APIs)")

fetch_real_data()

MODEL_DIR="saved_models";os.makedirs(MODEL_DIR,exist_ok=True)
def train_models():
    if os.path.exists(MODEL_DIR):shutil.rmtree(MODEL_DIR)
    os.makedirs(MODEL_DIR)
    d=REGION_PROFILES["dhaka_bangladesh"];c=generate_climate_timeseries(d,520)
    cm=ClimateForecaster(8);cm.fit(c);cm.save(MODEL_DIR)
    dm=DiseasePredictor();dm.fit(generate_full_dataset());dm.save(MODEL_DIR)
    nm=OutbreakSignalDetector();nm.fit(generate_news_corpus());nm.save(MODEL_DIR)
    return cm,dm,nm
try:
    rq=["climate_forecaster.pkl","disease_predictor.pkl","nlp_detector.pkl"]
    assert all(os.path.exists(os.path.join(MODEL_DIR,f)) for f in rq)
    climate_model=ClimateForecaster.load(MODEL_DIR);disease_model=DiseasePredictor.load(MODEL_DIR);nlp_model=OutbreakSignalDetector.load(MODEL_DIR)
    _=disease_model.get_shap_summary()
except:climate_model,disease_model,nlp_model=train_models()
ensemble=EnsembleRiskEngine(climate_model,disease_model,nlp_model)

REGIONS={
    "Nairobi, Kenya (Malaria)":{"key":"nairobi_kenya","disease":"malaria","disease_name":"Malaria","pop":"5.3M","facilities":312,"chws":"4,200","icon":"🦟","vector":"Anopheles mosquito","api_name":"Nairobi, Kenya"},
    "Lagos, Nigeria (Malaria)":{"key":"lagos_nigeria","disease":"malaria","disease_name":"Malaria","pop":"16.6M","facilities":1120,"chws":"15,600","icon":"🦟","vector":"Anopheles mosquito","api_name":"Lagos, Nigeria"},
    "Chittagong, Bangladesh (Cholera)":{"key":"chittagong_bangladesh","disease":"cholera","disease_name":"Cholera","pop":"5.2M","facilities":234,"chws":"3,100","icon":"💧","vector":"Waterborne (V. cholerae)","api_name":"Chittagong, Bangladesh"},
    "Dhaka, Bangladesh (Cholera)":{"key":"dhaka_bangladesh","disease":"cholera","disease_name":"Cholera","pop":"22.4M","facilities":847,"chws":"12,400","icon":"💧","vector":"Waterborne (V. cholerae)","api_name":"Dhaka, Bangladesh"},
}

CACHE={}
def get_data(rk,dk):
    ck=f"{rk}_{dk}"
    if ck in CACHE:return CACHE[ck]
    p=REGION_PROFILES[rk];c=generate_climate_timeseries(p,520);cs=generate_disease_incidence(c,p["diseases"][dk],520)
    ns=generate_nlp_signals(c,cs,520);f=create_feature_matrix(c,ns,4)
    ac=cs[len(cs)-len(f):];ar=compute_risk_score(cs)[len(cs)-len(f):]
    f["cases"]=ac;f["risk_score"]=ar;f["outbreak"]=(ar>=60).astype(int);f["region"]=rk;f["disease"]=dk
    CACHE[ck]=(c,f);return c,f

BG="#0a1628"
def sax(ax):
    ax.set_facecolor(BG);ax.tick_params(colors="#8899aa",labelsize=9)
    ax.xaxis.label.set_color("white");ax.yaxis.label.set_color("white");ax.title.set_color("white")
    for s in ax.spines.values():s.set_color("#1a3050")
    ax.grid(alpha=0.08,color="white")
def dfig(nr=1,nc=1,fs=(10,5)):
    f,a=plt.subplots(nr,nc,figsize=fs);f.patch.set_facecolor(BG)
    for x in(a.flat if isinstance(a,np.ndarray)else[a]):sax(x)
    return f,a
def grc(s):return "#DC2626" if s>=80 else "#F59E0B" if s>=60 else "#3B82F6" if s>=40 else "#10B981"
def grl(s):return "CRITICAL" if s>=80 else "HIGH" if s>=60 else "MODERATE" if s>=40 else "LOW"

def get_headlines(dk):
    ga=REAL_DATA["gdelt"].get(dk,[])
    if ga and len(ga)>=3:return [a["title"] for a in ga[:8]],ga[:8],True
    fallbacks={
        "malaria":[
            "Kenya highland malaria cases increase 30% as temperatures rise above seasonal norms",
            "Nigeria records over 68 million malaria cases in 2024 according to WHO estimates",
            "Anopheles mosquitoes detected at higher altitudes near Nairobi due to warming climate",
            "Lagos state hospitals report surge in malaria admissions during extended rainy season",
            "WHO warns climate change expanding malaria transmission zones into new regions",
            "East Africa faces increased malaria risk following El Nino flooding events",
            "Artemisinin-based combination therapy shortages reported in sub-Saharan Africa",
            "New study links rising temperatures to 15% increase in malaria transmission rates",
        ],
        "cholera":[
            "Bangladesh reports cholera spike in flood-affected Chittagong district",
            "Rohingya refugee camps report acute watery diarrhea outbreak linked to contaminated water",
            "WHO deploys oral cholera vaccines to Bangladesh following monsoon flooding",
            "Cholera cases surge in coastal Bangladesh as cyclone damages water infrastructure",
            "Dhaka water treatment facilities overwhelmed after record monsoon rainfall",
            "Climate-driven flooding creates ideal conditions for Vibrio cholerae transmission",
            "UNICEF warns of cholera risk in displacement camps across South Asia",
            "Global cholera pandemic enters seventh decade with climate change accelerating spread",
        ],
    }
    hl=fallbacks.get(dk,["Disease outbreak reported in the region","Hospitals report increased patient admissions","Health authorities monitor developing situation","New medical facility opens in the district"])
    return hl,[],False

def source_badge(source_type):
    badges={'nasa':'NASA','who':'WHO','gdelt':'GDELT','synthetic':'SYN'}
    colors={'nasa':'#EF4444','who':'#F59E0B','gdelt':'#8B5CF6','synthetic':'#3B82F6'}
    return f'<span style="font-size:8px;padding:2px 6px;border-radius:4px;background:{colors[source_type]}22;color:{colors[source_type]};font-weight:600;">{badges[source_type]}</span>'

def get_trend_indicator(current, historical_avg):
    if historical_avg is None or historical_avg == 0:return ""
    diff_pct = (current / historical_avg - 1) * 100
    if diff_pct > 10:
        return f'<span style="font-size:9px;color:#EF4444;">📈 +{diff_pct:.1f}%</span>'
    elif diff_pct < -10:
        return f'<span style="font-size:9px;color:#10B981;">📉 {diff_pct:.1f}%</span>'
    return '<span style="font-size:9px;color:#8899aa;">➡️ stable</span>'

def create_sparkline(values, color="#3B82F6", width=60, height=20):
    if not values or len(values) < 2:return ""
    max_v, min_v = max(values), min(values)
    if max_v == min_v:return ""
    step = width / (len(values) - 1)
    points = " ".join([f"{i*step},{height - (v-min_v)/(max_v-min_v)*(height-2)}" for i, v in enumerate(values)])
    return f'<svg width="{width}" height="{height}" style="margin-left:8px;vertical-align:middle;"><polyline points="{points}" stroke="{color}" stroke-width="1.5" fill="none"/></svg>'

def create_risk_gauge(score):
    color = grc(score)
    angle = (score / 100) * 180
    end_x = 60 + 50 * np.cos(np.radians(180 - angle))
    end_y = 60 - 50 * np.sin(np.radians(180 - angle))
    return f'''<svg width="140" height="80" viewBox="0 0 140 80" style="display:block;margin:0 auto;">
        <path d="M 20 60 A 50 50 0 0 1 120 60" stroke="#1a3050" stroke-width="10" fill="none" stroke-linecap="round"/>
        <path d="M 20 60 A 50 50 0 0 1 {end_x} {end_y}" stroke="{color}" stroke-width="10" fill="none" stroke-linecap="round"/>
        <circle cx="70" cy="60" r="3" fill="{color}"/>
        <text x="70" y="52" text-anchor="middle" font-size="22" fill="{color}" font-weight="bold" font-family="monospace">{score}</text>
    </svg>'''

def extract_keywords(headlines):
    stop_words = {'the', 'and', 'for', 'with', 'from', 'cases', 'reports', 'report', 'health', 'says', 'over'}
    keywords = []
    for h in headlines:
        words = [w.lower() for w in h.split() if len(w) > 4 and w.lower() not in stop_words]
        keywords.extend(words)
    return Counter(keywords).most_common(6)

def create_radar_chart(scores):
    fig = plt.figure(figsize=(4.5, 4.5), facecolor=BG)
    ax = fig.add_subplot(111, projection='polar')
    ax.set_facecolor(BG)
    
    categories = ['Climate\nRisk', 'Medical\nRisk', 'Public\nAnxiety']
    values = [scores['climate_risk'], scores['disease_ensemble_risk'], scores['nlp_signal_risk']]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_plot = values + values[:1]
    angles_plot = angles + angles[:1]
    
    ax.plot(angles_plot, values_plot, 'o-', linewidth=2.5, color='#3B82F6', markersize=6)
    ax.fill(angles_plot, values_plot, alpha=0.2, color='#3B82F6')
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, color='white', size=10, fontweight='500')
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75])
    ax.set_yticklabels(['25', '50', '75'], color='#667788', size=8)
    ax.grid(color='white', alpha=0.15, linewidth=0.5)
    ax.spines['polar'].set_color('#1a3050')
    plt.tight_layout()
    return fig

def generate_action_center_html(user_profile, disease, risk_score, region):
    """Generate Personalized Action Center with Dynamic Content"""
    
    # Personalized metrics
    liquid_assets = user_profile.get("liquid_assets", 140)
    daily_income = user_profile.get("daily_income", 35)
    daily_expenses = user_profile.get("daily_expenses", 40)
    
    # User-friendly calculations
    days_secure = int(liquid_assets / daily_expenses) if daily_expenses > 0 else 0
    emergency_mode = risk_score >= 70
    
    # Dynamic action generation based on risk level and region
    actions = []
    total_impact = 0
    potential_days = 0
    
    # HIGH RISK ACTIONS (>70)
    if risk_score >= 70:
        # 1. Medical Screening
        impact_medical = 200
        days_medical = int(impact_medical / daily_expenses)
        actions.append({
            "title": f"Emergency {disease.title()} Screening",
            "category": "🏥 HEALTH",
            "priority": 1,
            "impact_dollars": impact_medical,
            "impact_days": days_medical,
            "time_hours": 2,
            "difficulty": "Easy",
            "description": f"Get tested for {disease} at nearest clinic. Early detection prevents costly treatment later.",
            "steps": [
                {"order": "1", "label": "Find Nearest Testing Center", "automated": True},
                {"order": "2", "label": "Book Your Appointment", "automated": False},
                {"order": "3", "label": "Get Test Results & Treatment", "automated": True}
            ],
            "resources": [
                {"name": f"{region} Health Department", "phone": "+254-20-2217475" if "Kenya" in region else "+880-1234-5678"},
                {"name": "WHO Disease Information", "url": f"https://www.who.int/health-topics/{disease}"}
            ]
        })
        total_impact += impact_medical
        potential_days += days_medical
        
        # 2. Financial Protection
        if liquid_assets < 200:
            impact_cash = 300
            days_cash = int(impact_cash / daily_expenses)
            actions.append({
                "title": "Emergency Cash Relief",
                "category": "💰 FINANCIAL",
                "priority": 1,
                "impact_dollars": impact_cash,
                "impact_days": days_cash,
                "time_hours": 3,
                "difficulty": "Moderate",
                "description": "Apply for emergency government relief. Average approval: $300 within 48 hours.",
                "steps": [
                    {"order": "1", "label": "Complete Online Application", "automated": False},
                    {"order": "2", "label": "Upload Required Documents", "automated": False},
                    {"order": "3", "label": "Track Your Application", "automated": True}
                ],
                "resources": [
                    {"name": "National Relief Portal", "url": "https://relief.gov.bd" if "Bangladesh" in region else "https://socialprotection.or.ke"},
                    {"name": "Emergency Hotline", "phone": "*247#"}
                ]
            })
            total_impact += impact_cash
            potential_days += days_cash
    
    # MODERATE RISK ACTIONS (40-70)
    if 40 <= risk_score < 70:
        impact_prevent = 50
        days_prevent = int(impact_prevent / daily_expenses)
        actions.append({
            "title": "Prevention Supplies Kit",
            "category": "🛡️ PREVENTION",
            "priority": 2,
            "impact_dollars": impact_prevent,
            "impact_days": days_prevent,
            "time_hours": 1,
            "difficulty": "Easy",
            "description": "Stock water purification tablets, ORS, and hygiene supplies to prevent illness.",
            "steps": [
                {"order": "1", "label": "Download Supply Checklist", "automated": True},
                {"order": "2", "label": "Visit Nearby Pharmacy", "automated": True},
                {"order": "3", "label": "Store Supplies Safely", "automated": False}
            ],
            "resources": [
                {"name": "Supply Checklist (PDF)", "url": "https://www.who.int/emergencies/diseases/cholera/prevention"},
                {"name": "Find Pharmacy", "url": f"https://maps.google.com/?q=pharmacy+near+{region.replace(' ', '+')}"}
            ]
        })
        total_impact += impact_prevent
        potential_days += days_prevent
    
    # FOOD SECURITY (if low assets)
    if liquid_assets < 150:
        impact_food = 240
        days_food = int(impact_food / daily_expenses)
        actions.append({
            "title": "Emergency Food Assistance",
            "category": "🍚 NUTRITION",
            "priority": 2,
            "impact_dollars": impact_food,
            "impact_days": days_food,
            "time_hours": 3,
            "difficulty": "Moderate",
            "description": "WFP food vouchers covering 30 days of basic nutrition for your household.",
            "steps": [
                {"order": "1", "label": "Submit Application Online", "automated": False},
                {"order": "2", "label": "Verify Household Members", "automated": False},
                {"order": "3", "label": "Receive Digital Voucher", "automated": True}
            ],
            "resources": [
                {"name": "WFP Food Aid", "url": "https://www.wfp.org/countries/bangladesh" if "Bangladesh" in region else "https://www.wfp.org/countries/kenya"},
                {"name": "Food Aid Hotline", "phone": "+880-9612-345678" if "Bangladesh" in region else "+254-20-7622000"}
            ]
        })
        total_impact += impact_food
        potential_days += days_food
    
    optimized_days = days_secure + potential_days
    
    # HTML Generation
    html = f'''
    <div style="background:linear-gradient(135deg,rgba(139,92,246,0.12),rgba(59,130,246,0.12));
                border:2px solid rgba(139,92,246,0.3);border-radius:16px;padding:24px;margin-bottom:20px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
            <div style="display:flex;align-items:center;gap:12px;">
                <div style="width:48px;height:48px;background:linear-gradient(135deg,#8B5CF6,#3B82F6);
                            border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:24px;">⚡</div>
                <div>
                    <div style="font-size:24px;font-weight:700;color:white;font-family:'Inter',sans-serif;">
                        Your<span style="color:#8B5CF6;"> Personal</span> Action Plan
                    </div>
                    <div style="font-size:11px;color:#8899aa;letter-spacing:1px;">
                        Customized for {region} • {len(actions)} actions available
                    </div>
                </div>
            </div>
            <div style="padding:8px 16px;background:{'rgba(220,38,38,0.15)' if emergency_mode else 'rgba(59,130,246,0.15)'};
                        border:2px solid {'#DC2626' if emergency_mode else '#3B82F6'};border-radius:8px;">
                <div style="font-size:10px;color:#8899aa;font-family:monospace;margin-bottom:2px;">URGENCY</div>
                <div style="font-size:18px;font-weight:700;color:{'#EF4444' if emergency_mode else '#3B82F6'};">
                    {'HIGH' if emergency_mode else 'MODERATE'}
                </div>
            </div>
        </div>
    </div>
    '''
    
    # Emergency Warning
    if emergency_mode:
        html += '''
        <div style="background:rgba(220,38,38,0.12);border:2px solid #DC2626;border-left:6px solid #DC2626;
                    border-radius:12px;padding:16px 20px;margin-bottom:16px;animation:pulse-border 2s infinite;">
            <div style="display:flex;align-items:center;gap:12px;">
                <div style="font-size:32px;">⚠️</div>
                <div>
                    <div style="font-size:13px;font-weight:700;color:#FCA5A5;letter-spacing:1px;margin-bottom:6px;">
                        ⚠️ IMMEDIATE ACTION REQUIRED
                    </div>
                    <div style="font-size:12px;color:rgba(255,255,255,0.85);line-height:1.6;">
                        Complete at least 1 action in the next 24 hours to protect your household's health and finances.
                    </div>
                </div>
            </div>
        </div>
        <style>
            @keyframes pulse-border {
                0%, 100% { border-color: #DC2626; }
                50% { border-color: #F87171; }
            }
        </style>
        '''
    
    # Financial Security Dashboard
    html += f'''
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                border-radius:12px;padding:20px;margin-bottom:20px;">
        <div style="font-size:12px;color:#8899aa;font-family:monospace;letter-spacing:1px;margin-bottom:16px;
                    display:flex;align-items:center;gap:8px;">
            <span>📊</span> YOUR HOUSEHOLD FINANCIAL SECURITY
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;">
            <div style="text-align:center;padding:16px;background:rgba(59,130,246,0.08);
                        border:1px solid rgba(59,130,246,0.2);border-radius:10px;">
                <div style="font-size:11px;color:#60A5FA;font-family:monospace;margin-bottom:8px;">CURRENT STATUS</div>
                <div style="font-size:32px;font-weight:700;color:#3B82F6;margin:8px 0;">
                    {days_secure}
                </div>
                <div style="font-size:11px;color:#8899aa;">days of financial security</div>
                <div style="font-size:9px;color:#556677;margin-top:6px;">Based on ${liquid_assets} savings</div>
            </div>
            <div style="text-align:center;padding:16px;background:rgba(16,185,129,0.08);
                        border:1px solid rgba(16,185,129,0.2);border-radius:10px;">
                <div style="font-size:11px;color:#6EE7B7;font-family:monospace;margin-bottom:8px;">IF YOU ACT NOW</div>
                <div style="font-size:32px;font-weight:700;color:#10B981;margin:8px 0;">
                    {optimized_days}
                </div>
                <div style="font-size:11px;color:#8899aa;">days protected</div>
                <div style="font-size:9px;color:#556677;margin-top:6px;">+{potential_days} days extension</div>
            </div>
            <div style="text-align:center;padding:16px;background:rgba(245,158,11,0.08);
                        border:1px solid rgba(245,158,11,0.2);border-radius:10px;">
                <div style="font-size:11px;color:#FDE68A;font-family:monospace;margin-bottom:8px;">TOTAL VALUE</div>
                <div style="font-size:32px;font-weight:700;color:#F59E0B;margin:8px 0;">
                    ${total_impact}
                </div>
                <div style="font-size:11px;color:#8899aa;">potential savings</div>
                <div style="font-size:9px;color:#556677;margin-top:6px;">All actions combined</div>
            </div>
        </div>
    </div>
    '''
    
    # Action Cards
    for idx, action in enumerate(actions):
        priority = action.get("priority", 1)
        
        priority_colors = {
            1: {"bg": "rgba(220,38,38,0.08)", "border": "#DC2626", "text": "#FCA5A5", "icon": "🔴"},
            2: {"bg": "rgba(245,158,11,0.08)", "border": "#F59E0B", "text": "#FDE68A", "icon": "🟡"}
        }
        colors = priority_colors.get(priority, priority_colors[2])
        
        html += f'''
        <div style="background:{colors['bg']};border:2px solid {colors['border']};
                    border-left:6px solid {colors['border']};border-radius:12px;padding:20px;margin-bottom:16px;">
            
            <div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:14px;">
                <div style="flex:1;">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                        <span style="font-size:11px;padding:4px 12px;border-radius:5px;
                                     background:{colors['border']}22;color:{colors['text']};
                                     font-weight:700;font-family:monospace;">
                            {colors['icon']} PRIORITY {priority}
                        </span>
                        <span style="font-size:11px;padding:4px 12px;border-radius:5px;
                                     background:rgba(139,92,246,0.15);color:#A78BFA;
                                     font-weight:600;font-family:monospace;">
                            {action.get('category', 'GENERAL')}
                        </span>
                        <span style="font-size:11px;padding:4px 12px;border-radius:5px;
                                     background:rgba(6,182,212,0.15);color:#67E8F9;
                                     font-weight:600;font-family:monospace;">
                            {action.get('difficulty', 'Easy')}
                        </span>
                    </div>
                    <div style="font-size:22px;font-weight:700;color:white;margin-bottom:8px;">
                        {action.get('title', 'Emergency Action')}
                    </div>
                    <div style="font-size:13px;color:rgba(255,255,255,0.8);line-height:1.6;margin-bottom:10px;">
                        {action.get('description', '')}
                    </div>
                    <div style="display:flex;gap:12px;flex-wrap:wrap;">
                        <div style="font-size:12px;color:#10B981;font-weight:600;">
                            💰 Saves ${action.get('impact_dollars', 0)}
                        </div>
                        <div style="font-size:12px;color:#3B82F6;font-weight:600;">
                            📅 +{action.get('impact_days', 0)} days security
                        </div>
                        <div style="font-size:12px;color:#F59E0B;font-weight:600;">
                            ⏱️ Takes {action.get('time_hours', 1)}h
                        </div>
                    </div>
                </div>
            </div>
            
            <div style="background:rgba(0,0,0,0.15);border-radius:10px;padding:16px;margin-bottom:14px;">
                <div style="font-size:11px;font-weight:600;color:#FDE68A;font-family:monospace;margin-bottom:12px;">
                    📋 STEP-BY-STEP INSTRUCTIONS
                </div>
        '''
        
        # Steps
        for step in action.get("steps", []):
            is_auto = step.get("automated", False)
            auto_badge = '<span style="font-size:9px;padding:2px 8px;border-radius:3px;background:rgba(16,185,129,0.2);color:#6EE7B7;font-weight:700;margin-left:8px;">✓ READY</span>' if is_auto else ''
            
            html += f'''
            <div style="display:flex;align-items:center;gap:12px;padding:12px;
                        background:rgba(255,255,255,0.03);border-radius:8px;margin-bottom:8px;">
                <div style="width:36px;height:36px;
                            background:{'rgba(16,185,129,0.2)' if is_auto else 'rgba(59,130,246,0.2)'};
                            border-radius:50%;display:flex;align-items:center;justify-content:center;
                            font-weight:700;color:{'#6EE7B7' if is_auto else '#60A5FA'};
                            font-family:monospace;font-size:15px;">
                    {step.get('order', '?')}
                </div>
                <div style="flex:1;font-size:13px;color:white;font-weight:500;">
                    {step.get('label', 'Unknown Step')} {auto_badge}
                </div>
            </div>
            '''
        
        html += '</div>'
        
        # Resources Section
        if action.get("resources"):
            html += '''
            <div style="background:rgba(0,0,0,0.2);border:1px solid rgba(255,255,255,0.1);
                        border-radius:10px;padding:16px;margin-bottom:14px;">
                <div style="font-size:11px;font-weight:600;color:#8B5CF6;font-family:monospace;margin-bottom:12px;">
                    📞 HELPFUL CONTACTS & LINKS
                </div>
                <div style="display:grid;gap:10px;">
            '''
            
            for resource in action["resources"]:
                if "url" in resource:
                    html += f'''
                    <a href="{resource['url']}" target="_blank" 
                       style="display:flex;align-items:center;gap:10px;padding:10px;
                              background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.2);
                              border-radius:8px;text-decoration:none;transition:all 0.2s;"
                       onmouseover="this.style.background='rgba(59,130,246,0.2)'"
                       onmouseout="this.style.background='rgba(59,130,246,0.1)'">
                        <span style="font-size:20px;">🔗</span>
                        <span style="color:#60A5FA;font-size:13px;font-weight:500;">{resource['name']}</span>
                    </a>
                    '''
                elif "phone" in resource:
                    html += f'''
                    <div style="display:flex;align-items:center;gap:10px;padding:10px;
                                background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.2);
                                border-radius:8px;">
                        <span style="font-size:20px;">📞</span>
                        <div>
                            <div style="color:#6EE7B7;font-size:13px;font-weight:500;">{resource['name']}</div>
                            <div style="color:#10B981;font-size:14px;font-weight:700;font-family:monospace;">{resource['phone']}</div>
                        </div>
                    </div>
                    '''
            
            html += '</div></div>'
        
        # Action Buttons
        action_id = f"action_{idx}"
        html += f'''
            <div style="display:flex;gap:10px;flex-wrap:wrap;">
                <button id="btn_{action_id}" 
                        onclick="
                            this.innerHTML='✅ STARTED';
                            this.style.background='rgba(16,185,129,0.4)';
                            this.disabled=true;
                            document.getElementById('status_{action_id}').style.display='block';
                        "
                        style="background:linear-gradient(135deg,#10B981,#059669);color:white;
                               border:none;padding:14px 28px;border-radius:10px;font-size:14px;
                               font-weight:700;cursor:pointer;font-family:'Inter',sans-serif;
                               transition:all 0.3s ease;box-shadow:0 4px 12px rgba(16,185,129,0.3);"
                        onmouseover="this.style.transform='translateY(-2px)';this.style.boxShadow='0 6px 20px rgba(16,185,129,0.5)';"
                        onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='0 4px 12px rgba(16,185,129,0.3)';">
                    ⚡ START THIS ACTION
                </button>
                <button style="background:rgba(59,130,246,0.2);border:1px solid rgba(59,130,246,0.4);
                               color:#60A5FA;padding:14px 28px;border-radius:10px;font-size:14px;
                               font-weight:600;font-family:'Inter',sans-serif;cursor:pointer;
                               transition:all 0.2s;"
                        onmouseover="this.style.background='rgba(59,130,246,0.3)'"
                        onmouseout="this.style.background='rgba(59,130,246,0.2)'"
                        onclick="window.open('https://maps.google.com/?q=health+clinic+{region.replace(' ', '+')}', '_blank')">
                    🗺️ VIEW MAP
                </button>
                <button onclick="
                            this.innerHTML='⏭️ SKIPPED';
                            this.style.opacity='0.5';
                            this.disabled=true;
                        "
                        style="background:rgba(245,158,11,0.2);border:1px solid rgba(245,158,11,0.4);
                               color:#FDE68A;padding:14px 28px;border-radius:10px;font-size:14px;
                               font-weight:600;font-family:'Inter',sans-serif;cursor:pointer;">
                    ⏭️ SKIP
                </button>
            </div>
            
            <div id="status_{action_id}" style="display:none;margin-top:14px;padding:12px;
                        background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);
                        border-radius:8px;">
                <div style="font-size:13px;color:#6EE7B7;font-weight:600;">
                    ✅ Great start! You've extended your financial security by +{action.get('impact_days', 0)} days (${action.get('impact_dollars', 0)} saved).
                </div>
            </div>
        </div>
        '''
    
    # Summary Footer
    if not actions:
        html += '''
        <div style="padding:40px 20px;text-align:center;background:rgba(16,185,129,0.08);
                    border:1px solid rgba(16,185,129,0.2);border-radius:12px;">
            <div style="font-size:48px;margin-bottom:16px;">✅</div>
            <div style="font-size:20px;font-weight:700;color:#10B981;margin-bottom:10px;">
                You're in Good Shape!
            </div>
            <div style="font-size:14px;color:rgba(255,255,255,0.8);">
                No urgent actions needed at this time. Continue monitoring your health and finances.
            </div>
        </div>
        '''
    
    return html

def run_prediction(region_name):
    info = REGIONS[region_name]
    rk, dk, dn = info["key"], info["disease"], info["disease_name"]
    c, f = get_data(rk, dk)
    hl, ga, is_real = get_headlines(dk)
    a = ensemble.assess_risk(climate_df=c, features_df=f, news_texts=hl)
    chw = ensemble.generate_chw_alert(a, region_name.split(" (")[0], dn)
    rc = grc(a["risk_score"])
    rl = grl(a["risk_score"])
    cs = a["component_scores"]
    cc = REGIONS_API.get(info["api_name"], {}).get("country", "")
    
    # Generate personalized Action Plan based on region
    region_profiles = {
        "Nairobi, Kenya": {"liquid_assets": 180, "daily_income": 40, "daily_expenses": 35},
        "Lagos, Nigeria": {"liquid_assets": 120, "daily_income": 30, "daily_expenses": 42},
        "Chittagong, Bangladesh": {"liquid_assets": 100, "daily_income": 25, "daily_expenses": 38},
        "Dhaka, Bangladesh": {"liquid_assets": 90, "daily_income": 28, "daily_expenses": 45}
    }
    
    region_key = region_name.split(" (")[0]
    profile = region_profiles.get(region_key, {"liquid_assets": 140, "daily_income": 35, "daily_expenses": 40})
    
    user_profile = {
        **profile,
        "risk_score": a["risk_score"],
        "outbreak_prob": a["outbreak_probability"],
        "region": region_key
    }
    
    action_html = generate_action_center_html(user_profile, dn, a["risk_score"], region_key)
    
    # OVERVIEW TAB
    gauge_svg = create_risk_gauge(a["risk_score"])
    overview_html = f'''
    <div style="background:linear-gradient(135deg,{rc}08,{rc}18);border:1px solid {rc}40;border-radius:16px;padding:20px 24px;margin-bottom:16px;">
        <div style="display:grid;grid-template-columns:1fr auto;gap:24px;align-items:center;">
            <div>
                <div style="font-size:28px;font-weight:700;color:white;margin-bottom:8px;">{region_name.split(" (")[0]}</div>
                <div style="font-size:12px;color:#8899aa;">Pop: {info["pop"]} • Facilities: {info["facilities"]} • CHWs: {info["chws"]}</div>
                <div style="margin-top:16px;display:grid;grid-template-columns:repeat(2,1fr);gap:10px;">
                    <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px;">
                        <div style="font-size:9px;color:#667788;text-transform:uppercase;font-family:monospace;">Confidence</div>
                        <div style="font-size:20px;font-weight:700;color:white;margin-top:2px;">{a["confidence"]*100:.1f}%</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px;">
                        <div style="font-size:9px;color:#667788;text-transform:uppercase;font-family:monospace;">Outbreak Prob</div>
                        <div style="font-size:20px;font-weight:700;color:white;margin-top:2px;">{a["outbreak_probability"]*100:.1f}%</div>
                    </div>
                </div>
            </div>
            <div style="text-align:center;">
                {gauge_svg}
                <div style="font-size:11px;font-weight:700;color:{rc};letter-spacing:2px;font-family:monospace;margin-top:8px;">{rl} RISK</div>
                <div style="margin-top:12px;font-size:11px;color:#8899aa;">
                    <div>{info["icon"]} {dn}</div>
                    <div style="font-size:9px;color:#667788;margin-top:2px;">{info["vector"]}</div>
                </div>
            </div>
        </div>
    </div>
    '''
    
    fig_radar = create_radar_chart(cs)
    
    # SHAP chart
    fig_shap,ax=dfig(fs=(8,3.5))
    cats=list(a["shap_summary"].keys());vals=[v*100 for v in a["shap_summary"].values()];clrs=["#EF4444","#06B6D4","#F59E0B","#8B5CF6","#3B82F6"]
    bars=ax.barh(cats[::-1],vals[::-1],color=clrs[::-1],height=0.6)
    ax.set_xlabel("Feature Contribution (%)",fontsize=10,color="white")
    ax.set_title("SHAP FEATURE IMPORTANCE",fontsize=11,fontweight="bold",fontfamily="monospace",loc="left",color="white")
    for b,v in zip(bars,vals[::-1]):ax.text(b.get_width()+0.3,b.get_y()+b.get_height()/2,f"{v:.1f}%",va="center",fontsize=9,color="white",fontweight='600')
    ax.set_xlim(0,max(vals)*1.25);plt.tight_layout()
    
    # CLIMATE TAB
    api_name=info["api_name"]
    climate_html = ""
    
    if REAL_DATA["nasa"] is not None and len(REAL_DATA["nasa"])>0:
        rd=REAL_DATA["nasa"][REAL_DATA["nasa"]["region"]==api_name].dropna()
        rd=rd[rd["month"]<=12].tail(12)
        if len(rd)>=6:
            climate_html+='<div style="margin-bottom:20px;">'
            climate_html+=f'<div style="font-size:11px;color:#667788;text-transform:uppercase;letter-spacing:1px;font-family:monospace;margin-bottom:12px;">{source_badge("nasa")} REAL-TIME CLIMATE DATA</div>'
            climate_html+='<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">'
            
            l=rd.tail(1).iloc[0]
            temps = rd["temperature_C"].tail(6).tolist()
            precips = rd["precip_mm_day"].tail(6).tolist()
            humids = rd["humidity_pct"].tail(6).tolist()
            
            temp_avg = rd["temperature_C"].mean()
            temp_trend = get_trend_indicator(l["temperature_C"], temp_avg)
            temp_spark = create_sparkline(temps, "#EF4444")
            
            climate_html+=f'''<div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);border-radius:12px;padding:14px;">
                <div style="font-size:9px;color:#667788;font-family:monospace;margin-bottom:6px;">TEMPERATURE</div>
                <div style="display:flex;align-items:center;justify-content:space-between;">
                    <div style="font-size:26px;font-weight:700;color:#EF4444;">{l["temperature_C"]}°C</div>
                    {temp_spark}
                </div>
                <div style="margin-top:6px;">{temp_trend}</div>
                <div style="font-size:9px;color:#556677;margin-top:4px;">{int(l["year"])}-{int(l["month"]):02d}</div>
            </div>'''
            
            precip_spark = create_sparkline(precips, "#3B82F6")
            climate_html+=f'''<div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.2);border-radius:12px;padding:14px;">
                <div style="font-size:9px;color:#667788;font-family:monospace;margin-bottom:6px;">PRECIPITATION</div>
                <div style="display:flex;align-items:center;justify-content:space-between;">
                    <div style="font-size:26px;font-weight:700;color:#3B82F6;">{l["precip_mm_day"]}mm/d</div>
                    {precip_spark}
                </div>
                <div style="font-size:9px;color:#556677;margin-top:10px;">{int(l["year"])}-{int(l["month"]):02d}</div>
            </div>'''
            
            humid_spark = create_sparkline(humids, "#06B6D4")
            climate_html+=f'''<div style="background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.2);border-radius:12px;padding:14px;">
                <div style="font-size:9px;color:#667788;font-family:monospace;margin-bottom:6px;">HUMIDITY</div>
                <div style="display:flex;align-items:center;justify-content:space-between;">
                    <div style="font-size:26px;font-weight:700;color:#06B6D4;">{l["humidity_pct"]}%</div>
                    {humid_spark}
                </div>
                <div style="font-size:9px;color:#556677;margin-top:10px;">{int(l["year"])}-{int(l["month"]):02d}</div>
            </div>'''
            climate_html+='</div></div>'
    
    # Temperature & Precipitation charts
    fig_temp,ax1=dfig(fs=(6,3.5))
    used_real=False
    if REAL_DATA["nasa"] is not None and len(REAL_DATA["nasa"])>0:
        rd=REAL_DATA["nasa"][REAL_DATA["nasa"]["region"]==api_name].dropna()
        rd=rd[rd["month"]<=12].tail(12)
        if len(rd)>=6:
            months=[f"{int(r['year'])}-{int(r['month']):02d}" for _,r in rd.iterrows()]
            temps=[r["temperature_C"] for _,r in rd.iterrows()]
            ax1.plot(months,temps,"o-",color="#EF4444",lw=2.5,ms=6);ax1.fill_between(months,temps,alpha=0.15,color="#EF4444")
            ax1.set_title("TEMPERATURE TREND (°C)",fontsize=11,fontweight="bold",fontfamily="monospace",loc="left",color="white")
            ax1.tick_params(axis='x',rotation=45);used_real=True
    if not used_real:
        wks=[ff["week"] for ff in a["climate_forecast"]];ts=[ff["temperature"] for ff in a["climate_forecast"]]
        ax1.plot(wks,ts,"o-",color="#EF4444",lw=2.5,ms=6);ax1.fill_between(wks,ts,alpha=0.15,color="#EF4444")
        ax1.set_title("TEMPERATURE FORECAST (°C)",fontsize=11,fontweight="bold",fontfamily="monospace",loc="left",color="white")
    ax1.set_ylabel("°C",fontsize=10,color="white");plt.tight_layout()
    
    fig_precip,ax2=dfig(fs=(6,3.5))
    used_real2=False
    if REAL_DATA["nasa"] is not None and len(REAL_DATA["nasa"])>0:
        rd=REAL_DATA["nasa"][REAL_DATA["nasa"]["region"]==api_name].dropna()
        rd=rd[rd["month"]<=12].tail(12)
        if len(rd)>=6:
            months=[f"{int(r['year'])}-{int(r['month']):02d}" for _,r in rd.iterrows()]
            precips=[r["precip_mm_day"]*30 for _,r in rd.iterrows()]
            ax2.bar(months,precips,color="#3B82F6",alpha=0.85,width=0.6)
            ax2.set_title("PRECIPITATION TREND (mm/month)",fontsize=11,fontweight="bold",fontfamily="monospace",loc="left",color="white")
            ax2.tick_params(axis='x',rotation=45);used_real2=True
    if not used_real2:
        wks=[ff["week"] for ff in a["climate_forecast"]];ps=[ff["precipitation"] for ff in a["climate_forecast"]]
        ax2.bar(wks,ps,color="#3B82F6",alpha=0.85,width=0.6)
        ax2.set_title("PRECIPITATION FORECAST (mm)",fontsize=11,fontweight="bold",fontfamily="monospace",loc="left",color="white")
    ax2.set_ylabel("mm",fontsize=10,color="white");plt.tight_layout()
    
    # DISEASE TAB
    disease_html = ""
    who_info=WHO_DISEASE_INDICATORS.get(dk,{})
    who_codes=who_info.get("codes",[])
    if who_codes and REAL_DATA["who"] is not None and len(REAL_DATA["who"])>0:
        wd=REAL_DATA["who"][(REAL_DATA["who"]["country"]==cc)&(REAL_DATA["who"]["indicator_code"].isin(who_codes))]
        if len(wd)>0:
            disease_html+=f'<div style="margin-bottom:20px;"><div style="font-size:11px;color:#667788;text-transform:uppercase;letter-spacing:1px;font-family:monospace;margin-bottom:12px;">{source_badge("who")} {dn.upper()} INDICATORS</div>'
            disease_html+='<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;">'
            for ind in wd["indicator"].unique():
                row=wd[wd["indicator"]==ind].sort_values("year").tail(1).iloc[0]
                disease_html+=f'''<div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);border-radius:12px;padding:14px;">
                    <div style="font-size:9px;color:#667788;font-family:monospace;margin-bottom:4px;">{ind.upper()}</div>
                    <div style="font-size:24px;font-weight:700;color:#F59E0B;margin:8px 0;">{row["value"]:,.0f}</div>
                    <div style="font-size:9px;color:#556677;">{cc} • {int(row["year"])}</div>
                </div>'''
            disease_html+='</div></div>'
    
    # NEWS TAB
    keywords = extract_keywords(hl)
    news_html = f'<div style="margin-bottom:16px;"><div style="font-size:11px;color:#667788;text-transform:uppercase;letter-spacing:1px;font-family:monospace;margin-bottom:12px;">KEY TOPICS</div><div style="display:flex;flex-wrap:wrap;gap:8px;">'
    for word, count in keywords:
        news_html+=f'<span style="padding:6px 14px;background:rgba(139,92,246,0.15);border:1px solid rgba(139,92,246,0.3);border-radius:20px;font-size:12px;color:#C4B5FD;font-weight:500;">{word} ({count})</span>'
    news_html+='</div></div>'
    
    news_html+=f'<div style="font-size:11px;color:#667788;text-transform:uppercase;letter-spacing:1px;font-family:monospace;margin-bottom:12px;">{source_badge("gdelt" if ga else "synthetic")} NLP SIGNAL ANALYSIS ({len(hl)} headlines)</div>'
    for sig in a["nlp_signals"][:5]:
        if sig["is_outbreak"]:bg,bc,badge_col="rgba(220,38,38,0.1)","rgba(220,38,38,0.3)","#FCA5A5"
        else:bg,bc,badge_col="rgba(16,185,129,0.1)","rgba(16,185,129,0.3)","#6EE7B7"
        badge_text="🔴 OUTBREAK" if sig["is_outbreak"] else "🟢 NORMAL"
        news_html+=f'''<div style="background:{bg};border:1px solid {bc};border-radius:10px;padding:12px;margin-bottom:8px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="font-size:9px;padding:3px 8px;border-radius:4px;background:{bc};color:{badge_col};font-weight:700;">{badge_text}</span>
                <span style="font-size:9px;color:#667788;font-family:monospace;">Conf: {sig["confidence"]:.0%} | {sig["severity"]}</span>
            </div>
            <div style="font-size:12px;color:rgba(255,255,255,0.85);line-height:1.5;">"{sig["text"][:100]}..."</div>
        </div>'''
    
    # ALERTS TAB
    alerts_html='<div style="font-size:11px;color:#667788;text-transform:uppercase;letter-spacing:1px;font-family:monospace;margin-bottom:12px;">ACTIVE ALERTS</div>'
    if a["alerts"]:
        sorted_alerts = sorted(a["alerts"], key=lambda x: 0 if x['level']=='critical' else 1)
        alerts_html+='<div style="position:relative;padding-left:24px;">'
        for i,al in enumerate(sorted_alerts):
            if al["level"]=="critical":color,icon="#DC2626","⚠️"
            else:color,icon="#F59E0B","⚡"
            is_last = i == len(sorted_alerts) - 1
            line_style = "" if is_last else "border-left:2px solid rgba(255,255,255,0.1);"
            alerts_html+=f'''<div style="position:relative;margin-bottom:20px;">
                <div style="position:absolute;left:-24px;top:12px;width:10px;height:10px;border-radius:50%;background:{color};border:2px solid {BG};z-index:1;"></div>
                <div style="{line_style}padding-left:16px;padding-bottom:16px;">
                    <div style="background:rgba(255,255,255,0.03);border:1px solid {color}40;border-left:3px solid {color};border-radius:10px;padding:14px;">
                        <div style="font-size:10px;font-weight:700;color:{color};letter-spacing:1px;font-family:monospace;margin-bottom:8px;">{icon} {al["level"].upper()}</div>
                        <div style="font-size:13px;color:rgba(255,255,255,0.9);line-height:1.6;margin-bottom:10px;">{al["message"]}</div>
                        <div style="display:inline-block;font-size:10px;font-weight:600;color:{color};background:{color}15;padding:6px 12px;border-radius:6px;font-family:monospace;">→ {al["action"]}</div>
                    </div>
                </div>
            </div>'''
        alerts_html+='</div>'
    else:
        alerts_html+='<div style="padding:20px;text-align:center;color:#667788;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:10px;">✓ No active alerts</div>'
    
    # CHW recommendations
    chw_html = f'''<div style="font-size:11px;color:#667788;text-transform:uppercase;letter-spacing:1px;font-family:monospace;margin:24px 0 12px 0;">COMMUNITY HEALTH WORKER ALERT</div>
    <div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.2);border-radius:12px;padding:18px;">
        <div style="font-size:12px;color:#93C5FD;font-weight:600;font-family:monospace;margin-bottom:10px;">📋 {dn.upper()} – {region_name.split(" (")[0].upper()}</div>
        <div style="font-size:13px;color:rgba(255,255,255,0.8);line-height:1.7;margin-bottom:14px;">{chw["summary"]}</div>
        <div style="border-top:1px solid rgba(255,255,255,0.1);padding-top:14px;">
            <div style="font-size:11px;font-weight:600;color:#FDE68A;font-family:monospace;margin-bottom:10px;">RECOMMENDED ACTIONS:</div>
            <div style="margin-bottom:16px;">
                <a href="https://www.who.int/health-topics/{info["disease"]}" target="_blank" style="text-decoration:none;">
                    <button style="background:linear-gradient(135deg,#10B981,#059669);
                           color:white;
                           border:none;
                           padding:12px 24px;
                           border-radius:10px;
                           font-size:14px;
                           font-weight:600;
                           cursor:pointer;
                           font-family:'Inter',sans-serif;
                           display:inline-flex;
                           align-items:center;
                           gap:10px;
                           transition:all 0.3s ease;
                           box-shadow:0 4px 8px rgba(16,185,129,0.25);"
                           onmouseover="this.style.transform='translateY(-2px) scale(1.02)';this.style.boxShadow='0 6px 16px rgba(16,185,129,0.4)';"
                           onmouseout="this.style.transform='translateY(0) scale(1)';this.style.boxShadow='0 4px 8px rgba(16,185,129,0.25)';">
                        <span style="font-size:18px;">📚</span>
                        <span>Get WHO Information</span>
                        <span style="font-size:14px;opacity:0.8;">→</span>
                    </button>
                </a>
            </div>
            <div style="font-size:12px;color:rgba(255,255,255,0.8);line-height:1.9;">'''
    for i, act in enumerate(chw["recommended_actions"][1:], 2):
        chw_html += f'<div style="margin-bottom:6px;"><span style="display:inline-block;width:20px;color:#3B82F6;font-weight:700;">{i}.</span>{act}</div>'
    chw_html += '</div></div></div>'
    
    return overview_html, fig_radar, fig_shap, climate_html, fig_temp, fig_precip, disease_html, news_html, alerts_html, chw_html, action_html

CSS="""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
.gradio-container{max-width:1280px!important;background:#050b18!important;font-family:'Inter',sans-serif!important}
.main,.contain,#component-0{background:#050b18!important}
.block{background:transparent!important;border:none!important}
.wrap,.panel{background:#0a1628!important}
input,textarea,.input-text,.scroll-hide{background:#0d1f3c!important;border:1px solid rgba(59,130,246,0.2)!important;color:white!important;border-radius:10px!important}
label,.label-wrap{color:#8899aa!important;font-family:'JetBrains Mono',monospace!important;font-size:11px!important;text-transform:uppercase!important;letter-spacing:1px!important}
.primary{background:linear-gradient(135deg,#3B82F6,#2563EB)!important;border:none!important;border-radius:10px!important;font-family:'JetBrains Mono',monospace!important;font-weight:600!important}
.markdown{color:white!important}
footer{display:none!important}
.tabs{border-bottom:1px solid rgba(59,130,246,0.15)!important}
.tab-nav button{color:#ffffff!important;border:none!important;background:transparent!important;font-family:'JetBrains Mono',monospace!important;font-size:11px!important;letter-spacing:1px!important;padding:12px 20px!important}
.tab-nav button.selected{color:#3B82F6!important;border-bottom:2px solid #3B82F6!important;font-weight:600!important}
@media (max-width: 768px) {
    .gradio-container{max-width:100%!important}
}
"""

ds_badge="🟢 LIVE" if DATA_SOURCE=="real" else "🟡 MIXED" if DATA_SOURCE=="mixed" else "🔵 SYNTHETIC"
HEADER=f'''<div style="padding:12px 0 20px 0;border-bottom:1px solid rgba(59,130,246,0.12);margin-bottom:20px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:14px;">
            <div style="width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,#3B82F6,#06B6D4);display:flex;align-items:center;justify-content:center;font-size:22px;">🌡</div>
            <div>
                <div style="font-size:22px;font-weight:700;color:white;">ClimaHealth<span style="color:#3B82F6;">AI</span></div>
                <div style="font-size:9px;color:#667788;letter-spacing:2px;text-transform:uppercase;font-family:monospace;">Climate-Driven Disease Early Warning + Action Center</div>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:12px;">
            <span style="font-size:10px;padding:4px 10px;border-radius:6px;background:rgba(16,185,129,0.15);color:#10B981;font-family:monospace;font-weight:600;">{ds_badge}</span>
            <div style="display:flex;align-items:center;gap:6px;">
                <div style="width:6px;height:6px;border-radius:50%;background:#10B981;animation:pulse 2s infinite;"></div>
                <span style="font-size:10px;color:#667788;font-family:monospace;">LIVE</span>
            </div>
        </div>
    </div>
    <div style="font-size:11px;color:#8899aa;line-height:1.6;">
        <strong style="color:white;">Pipeline:</strong> Domain-specific Gaussian/Rolling Features → Disease Ensemble (RF+GB+LR) → NLP TF-IDF/N-gram → Personalized Action Generation<br/>
        <strong style="color:white;">Sources:</strong> NASA POWER • WHO GHO • GDELT Project
    </div>
</div>
<style>@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.4}}}}</style>'''

with gr.Blocks(title="ClimaHealth AI",css=CSS,theme=gr.themes.Base(primary_hue="blue",secondary_hue="cyan",neutral_hue="slate")) as app:
    gr.HTML(HEADER)

    with gr.Accordion("GLOBAL RISK MONITOR (INTERACTIVE MAP)", open=True):
        gr.HTML(generate_global_map())
    
    with gr.Row():
        dd=gr.Dropdown(choices=list(REGIONS.keys()),value="Nairobi, Kenya (Malaria)",label="SELECT REGION",scale=2)
        btn=gr.Button("🔍 Run Analysis",variant="primary",scale=1)
    
    with gr.Tabs():
        with gr.Tab("Risk Overview"):
            overview_out=gr.HTML()
            with gr.Row():
                radar_plot=gr.Plot(label="Component Risk Analysis")
                shap_plot=gr.Plot(label="Feature Importance")
        
        with gr.Tab("Climate Analysis"):
            climate_out=gr.HTML()
            with gr.Row():
                temp_plot=gr.Plot(label="")
                precip_plot=gr.Plot(label="")
        
        with gr.Tab("Disease Intelligence"):
            disease_out=gr.HTML()
        
        with gr.Tab("News Signals"):
            news_out=gr.HTML()
        
        with gr.Tab("Alerts & Actions"):
            alerts_out=gr.HTML()
            chw_out=gr.HTML()
        
        with gr.Tab("Action Center"):
            action_out=gr.HTML()
    
    btn.click(fn=run_prediction,inputs=[dd],outputs=[overview_out,radar_plot,shap_plot,climate_out,temp_plot,precip_plot,disease_out,news_out,alerts_out,chw_out,action_out])

print("\n ClimaHealth AI ready with Enhanced Action Center!")
if __name__=="__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )