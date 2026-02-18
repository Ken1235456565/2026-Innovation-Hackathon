import { useState, useEffect, useCallback, useMemo } from "react";
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Cell, PieChart, Pie } from "recharts";

// ============================================================
// DATA LAYER - Realistic simulated data for hackathon demo
// ============================================================

const DISEASES = {
  dengue: { name: "Dengue Fever", color: "#F59E0B", icon: "🦟", vectorType: "Aedes mosquito" },
  malaria: { name: "Malaria", color: "#EF4444", icon: "🦟", vectorType: "Anopheles mosquito" },
  cholera: { name: "Cholera", color: "#3B82F6", icon: "💧", vectorType: "Waterborne (V. cholerae)" },
  zika: { name: "Zika Virus", color: "#8B5CF6", icon: "🦟", vectorType: "Aedes mosquito" },
  lyme: { name: "Lyme Disease", color: "#10B981", icon: "🪲", vectorType: "Ixodes tick" },
};

const REGIONS = [
  {
    id: "dhaka", name: "Dhaka, Bangladesh", lat: 23.8, lon: 90.4,
    population: "22.4M", healthFacilities: 847, chws: 12400,
    primaryRisk: "dengue", riskScore: 87, trend: "rising",
    climate: { tempCurrent: 32.1, tempAnomaly: +2.3, precipMm: 340, humidity: 82, ndvi: 0.42 },
    historicalOutbreaks: 14,
    shapValues: { temperature: 0.42, precipitation: 0.28, humidity: 0.15, seasonality: 0.10, nlpSignals: 0.05 },
    nlpSignals: [
      { source: "WHO DON", date: "2026-02-10", title: "Bangladesh reports surge in dengue hospitalizations", severity: "high", confidence: 0.94 },
      { source: "ProMED-mail", date: "2026-02-08", title: "Aedes aegypti breeding sites identified in Dhaka slums", severity: "medium", confidence: 0.87 },
      { source: "GDELT", date: "2026-02-12", title: "Dhaka hospitals overwhelmed with fever cases", severity: "high", confidence: 0.78 },
    ],
    weeklyForecast: [
      { week: "W1", risk: 72, temp: 31.2, precip: 45 },
      { week: "W2", risk: 78, temp: 31.8, precip: 62 },
      { week: "W3", risk: 83, temp: 32.4, precip: 88 },
      { week: "W4", risk: 87, temp: 32.1, precip: 120 },
      { week: "W5", risk: 91, temp: 33.0, precip: 145 },
      { week: "W6", risk: 88, temp: 32.6, precip: 110 },
      { week: "W7", risk: 82, temp: 31.9, precip: 78 },
      { week: "W8", risk: 76, temp: 31.2, precip: 55 },
    ],
    alerts: [
      { level: "critical", message: "Dengue risk predicted to peak in 3-5 weeks. Pre-position mosquito nets and insecticide at district hospitals.", action: "DEPLOY VECTOR CONTROL" },
      { level: "warning", message: "Heavy monsoon rainfall forecast to increase Aedes breeding sites by 40%.", action: "ACTIVATE LARVICIDING TEAMS" },
    ]
  },
  {
    id: "nairobi", name: "Nairobi, Kenya", lat: -1.3, lon: 36.8,
    population: "5.3M", healthFacilities: 312, chws: 4200,
    primaryRisk: "malaria", riskScore: 74, trend: "rising",
    climate: { tempCurrent: 24.8, tempAnomaly: +1.8, precipMm: 210, humidity: 68, ndvi: 0.55 },
    historicalOutbreaks: 9,
    shapValues: { temperature: 0.38, precipitation: 0.22, humidity: 0.18, seasonality: 0.14, nlpSignals: 0.08 },
    nlpSignals: [
      { source: "WHO DON", date: "2026-02-09", title: "Kenya highland malaria cases increase 30% year-over-year", severity: "high", confidence: 0.91 },
      { source: "GDELT", date: "2026-02-11", title: "Anopheles mosquitoes detected at higher altitudes near Nairobi", severity: "medium", confidence: 0.83 },
    ],
    weeklyForecast: [
      { week: "W1", risk: 62, temp: 23.5, precip: 30 },
      { week: "W2", risk: 66, temp: 24.0, precip: 42 },
      { week: "W3", risk: 70, temp: 24.8, precip: 58 },
      { week: "W4", risk: 74, temp: 24.6, precip: 72 },
      { week: "W5", risk: 78, temp: 25.2, precip: 85 },
      { week: "W6", risk: 75, temp: 24.8, precip: 68 },
      { week: "W7", risk: 69, temp: 24.1, precip: 48 },
      { week: "W8", risk: 63, temp: 23.6, precip: 35 },
    ],
    alerts: [
      { level: "warning", message: "Malaria risk elevated due to warming highlands expanding mosquito habitat.", action: "DISTRIBUTE ARTEMISININ SUPPLIES" },
    ]
  },
  {
    id: "recife", name: "Recife, Brazil", lat: -8.05, lon: -34.9,
    population: "4.1M", healthFacilities: 523, chws: 6800,
    primaryRisk: "zika", riskScore: 69, trend: "stable",
    climate: { tempCurrent: 28.5, tempAnomaly: +1.4, precipMm: 185, humidity: 78, ndvi: 0.48 },
    historicalOutbreaks: 11,
    shapValues: { temperature: 0.35, precipitation: 0.25, humidity: 0.20, seasonality: 0.12, nlpSignals: 0.08 },
    nlpSignals: [
      { source: "ProMED-mail", date: "2026-02-07", title: "Zika-positive mosquito pools identified in Recife metropolitan area", severity: "medium", confidence: 0.86 },
    ],
    weeklyForecast: [
      { week: "W1", risk: 65, temp: 28.0, precip: 25 },
      { week: "W2", risk: 67, temp: 28.3, precip: 30 },
      { week: "W3", risk: 69, temp: 28.5, precip: 38 },
      { week: "W4", risk: 69, temp: 28.4, precip: 42 },
      { week: "W5", risk: 71, temp: 28.8, precip: 48 },
      { week: "W6", risk: 68, temp: 28.2, precip: 35 },
      { week: "W7", risk: 64, temp: 27.8, precip: 28 },
      { week: "W8", risk: 61, temp: 27.5, precip: 22 },
    ],
    alerts: [
      { level: "advisory", message: "Zika risk moderately elevated. Monitor pregnant women in affected districts.", action: "ENHANCE PRENATAL SCREENING" },
    ]
  },
  {
    id: "chittagong", name: "Chittagong, Bangladesh", lat: 22.3, lon: 91.8,
    population: "5.2M", healthFacilities: 234, chws: 3100,
    primaryRisk: "cholera", riskScore: 81, trend: "rising",
    climate: { tempCurrent: 30.2, tempAnomaly: +1.9, precipMm: 410, humidity: 85, ndvi: 0.38 },
    historicalOutbreaks: 18,
    shapValues: { temperature: 0.20, precipitation: 0.40, humidity: 0.18, seasonality: 0.12, nlpSignals: 0.10 },
    nlpSignals: [
      { source: "WHO DON", date: "2026-02-11", title: "Cholera cases spike in flood-affected Chittagong district", severity: "high", confidence: 0.92 },
      { source: "GDELT", date: "2026-02-13", title: "Rohingya camps report acute watery diarrhea outbreak", severity: "critical", confidence: 0.89 },
    ],
    weeklyForecast: [
      { week: "W1", risk: 70, temp: 29.5, precip: 55 },
      { week: "W2", risk: 74, temp: 29.8, precip: 78 },
      { week: "W3", risk: 78, temp: 30.2, precip: 105 },
      { week: "W4", risk: 81, temp: 30.0, precip: 135 },
      { week: "W5", risk: 85, temp: 30.5, precip: 160 },
      { week: "W6", risk: 83, temp: 30.2, precip: 128 },
      { week: "W7", risk: 77, temp: 29.7, precip: 90 },
      { week: "W8", risk: 72, temp: 29.3, precip: 65 },
    ],
    alerts: [
      { level: "critical", message: "Cholera outbreak imminent — flooding has contaminated water sources in 12 districts.", action: "DEPLOY ORS + WATER PURIFICATION" },
      { level: "warning", message: "Refugee camp populations at extreme risk. Activate emergency WASH protocols.", action: "MOBILIZE WASH TEAMS" },
    ]
  },
  {
    id: "lagos", name: "Lagos, Nigeria", lat: 6.5, lon: 3.4,
    population: "16.6M", healthFacilities: 1120, chws: 15600,
    primaryRisk: "malaria", riskScore: 72, trend: "stable",
    climate: { tempCurrent: 29.8, tempAnomaly: +1.5, precipMm: 195, humidity: 79, ndvi: 0.44 },
    historicalOutbreaks: 22,
    shapValues: { temperature: 0.32, precipitation: 0.28, humidity: 0.18, seasonality: 0.15, nlpSignals: 0.07 },
    nlpSignals: [
      { source: "GDELT", date: "2026-02-10", title: "Lagos state reports increased malaria admissions in public hospitals", severity: "medium", confidence: 0.81 },
    ],
    weeklyForecast: [
      { week: "W1", risk: 68, temp: 29.2, precip: 28 },
      { week: "W2", risk: 70, temp: 29.5, precip: 35 },
      { week: "W3", risk: 72, temp: 29.8, precip: 45 },
      { week: "W4", risk: 72, temp: 29.6, precip: 52 },
      { week: "W5", risk: 74, temp: 30.0, precip: 60 },
      { week: "W6", risk: 71, temp: 29.5, precip: 48 },
      { week: "W7", risk: 68, temp: 29.1, precip: 35 },
      { week: "W8", risk: 65, temp: 28.8, precip: 28 },
    ],
    alerts: [
      { level: "warning", message: "Seasonal malaria transmission intensifying. Ensure ACT availability at primary care centers.", action: "RESTOCK ANTIMALARIALS" },
    ]
  },
  {
    id: "manaus", name: "Manaus, Brazil", lat: -3.1, lon: -60.0,
    population: "2.3M", healthFacilities: 187, chws: 2400,
    primaryRisk: "dengue", riskScore: 78, trend: "rising",
    climate: { tempCurrent: 31.2, tempAnomaly: +2.0, precipMm: 290, humidity: 84, ndvi: 0.65 },
    historicalOutbreaks: 16,
    shapValues: { temperature: 0.38, precipitation: 0.30, humidity: 0.15, seasonality: 0.10, nlpSignals: 0.07 },
    nlpSignals: [
      { source: "ProMED-mail", date: "2026-02-09", title: "Dengue serotype 3 re-emerges in Amazonas state", severity: "high", confidence: 0.88 },
      { source: "GDELT", date: "2026-02-12", title: "Manaus emergency services report record dengue hospitalizations", severity: "high", confidence: 0.84 },
    ],
    weeklyForecast: [
      { week: "W1", risk: 70, temp: 30.5, precip: 42 },
      { week: "W2", risk: 73, temp: 30.8, precip: 55 },
      { week: "W3", risk: 76, temp: 31.2, precip: 72 },
      { week: "W4", risk: 78, temp: 31.0, precip: 88 },
      { week: "W5", risk: 81, temp: 31.5, precip: 102 },
      { week: "W6", risk: 79, temp: 31.1, precip: 85 },
      { week: "W7", risk: 74, temp: 30.6, precip: 62 },
      { week: "W8", risk: 70, temp: 30.2, precip: 48 },
    ],
    alerts: [
      { level: "warning", message: "Dengue serotype shift detected — population immunity may be lower. Increase surveillance.", action: "INTENSIFY CASE MONITORING" },
    ]
  },
];

const GLOBAL_STATS = {
  regionsMonitored: 847,
  activePredictions: 2341,
  alertsIssued: 156,
  avgLeadTimeDays: 32,
  modelAccuracy: 0.847,
  livesProtected: "2.1M",
};

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

const getRiskColor = (score) => {
  if (score >= 80) return "#DC2626";
  if (score >= 60) return "#F59E0B";
  if (score >= 40) return "#3B82F6";
  return "#10B981";
};

const getRiskLabel = (score) => {
  if (score >= 80) return "CRITICAL";
  if (score >= 60) return "HIGH";
  if (score >= 40) return "MODERATE";
  return "LOW";
};

const getAlertColor = (level) => {
  if (level === "critical") return { bg: "rgba(220, 38, 38, 0.15)", border: "#DC2626", text: "#FCA5A5" };
  if (level === "warning") return { bg: "rgba(245, 158, 11, 0.15)", border: "#F59E0B", text: "#FDE68A" };
  return { bg: "rgba(59, 130, 246, 0.15)", border: "#3B82F6", text: "#93C5FD" };
};

// ============================================================
// COMPONENTS
// ============================================================

// --- Animated counter ---
const AnimatedNumber = ({ value, duration = 1200 }) => {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    const num = typeof value === 'string' ? parseFloat(value) : value;
    if (isNaN(num)) { setDisplay(value); return; }
    let start = 0;
    const step = num / (duration / 16);
    const timer = setInterval(() => {
      start += step;
      if (start >= num) { setDisplay(num); clearInterval(timer); }
      else setDisplay(Math.floor(start));
    }, 16);
    return () => clearInterval(timer);
  }, [value, duration]);
  return <>{typeof value === 'string' && isNaN(parseFloat(value)) ? value : display}</>;
};

// --- SVG World Map with risk dots ---
const WorldMap = ({ regions, selectedRegion, onSelectRegion }) => {
  const mapProjection = (lat, lon) => {
    const x = ((lon + 180) / 360) * 800;
    const y = ((90 - lat) / 180) * 400;
    return { x, y };
  };

  return (
    <div style={{ position: "relative", width: "100%", aspectRatio: "2/1", background: "linear-gradient(180deg, #0a1628 0%, #0d1f3c 100%)", borderRadius: 12, overflow: "hidden", border: "1px solid rgba(59, 130, 246, 0.2)" }}>
      <svg viewBox="0 0 800 400" style={{ width: "100%", height: "100%" }}>
        {/* Grid lines */}
        {[...Array(18)].map((_, i) => (
          <line key={`vg${i}`} x1={i * 44.4} y1={0} x2={i * 44.4} y2={400} stroke="rgba(59,130,246,0.06)" strokeWidth={0.5} />
        ))}
        {[...Array(9)].map((_, i) => (
          <line key={`hg${i}`} x1={0} y1={i * 44.4} x2={800} y2={i * 44.4} stroke="rgba(59,130,246,0.06)" strokeWidth={0.5} />
        ))}

        {/* Simplified continent outlines */}
        {/* Africa */}
        <path d="M370,140 L400,130 L420,145 L430,170 L435,200 L430,240 L420,270 L400,290 L380,285 L365,260 L360,230 L355,200 L360,170 Z" fill="rgba(59,130,246,0.08)" stroke="rgba(59,130,246,0.15)" strokeWidth={1}/>
        {/* South America */}
        <path d="M220,210 L250,195 L270,200 L280,220 L285,250 L280,280 L265,310 L245,330 L230,320 L220,290 L215,260 L218,230 Z" fill="rgba(59,130,246,0.08)" stroke="rgba(59,130,246,0.15)" strokeWidth={1}/>
        {/* Asia */}
        <path d="M440,80 L520,70 L580,80 L620,100 L640,130 L630,160 L600,170 L560,175 L520,165 L490,170 L460,160 L440,140 L435,110 Z" fill="rgba(59,130,246,0.08)" stroke="rgba(59,130,246,0.15)" strokeWidth={1}/>
        {/* SE Asia */}
        <path d="M560,160 L590,155 L610,165 L620,180 L610,200 L590,205 L570,195 L560,180 Z" fill="rgba(59,130,246,0.08)" stroke="rgba(59,130,246,0.15)" strokeWidth={1}/>
        {/* North America */}
        <path d="M100,60 L180,50 L220,70 L240,100 L235,130 L220,155 L190,170 L160,165 L130,150 L110,120 L100,90 Z" fill="rgba(59,130,246,0.08)" stroke="rgba(59,130,246,0.15)" strokeWidth={1}/>
        {/* Europe */}
        <path d="M370,60 L410,55 L440,65 L445,90 L435,110 L410,115 L390,110 L375,95 L370,75 Z" fill="rgba(59,130,246,0.08)" stroke="rgba(59,130,246,0.15)" strokeWidth={1}/>

        {/* Risk region markers */}
        {regions.map((region) => {
          const pos = mapProjection(region.lat, region.lon);
          const isSelected = selectedRegion?.id === region.id;
          const pulseRadius = region.riskScore >= 80 ? 20 : region.riskScore >= 60 ? 15 : 10;
          return (
            <g key={region.id} onClick={() => onSelectRegion(region)} style={{ cursor: "pointer" }}>
              {/* Pulse animation */}
              <circle cx={pos.x} cy={pos.y} r={pulseRadius} fill="none" stroke={getRiskColor(region.riskScore)} strokeWidth={1} opacity={0.3}>
                <animate attributeName="r" values={`${pulseRadius * 0.5};${pulseRadius * 1.5}`} dur="2s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.5;0" dur="2s" repeatCount="indefinite" />
              </circle>
              {/* Risk circle */}
              <circle cx={pos.x} cy={pos.y} r={isSelected ? 10 : 7} fill={getRiskColor(region.riskScore)} opacity={0.8} stroke={isSelected ? "#fff" : "none"} strokeWidth={isSelected ? 2 : 0} />
              {/* Label */}
              <text x={pos.x + 14} y={pos.y + 4} fill="rgba(255,255,255,0.7)" fontSize={9} fontFamily="'JetBrains Mono', monospace">{region.name.split(",")[0]}</text>
              {/* Risk score */}
              <text x={pos.x} y={pos.y + 3} fill="#fff" fontSize={7} fontFamily="'JetBrains Mono', monospace" textAnchor="middle" fontWeight="bold">{region.riskScore}</text>
            </g>
          );
        })}
      </svg>

      {/* Legend */}
      <div style={{ position: "absolute", bottom: 12, left: 12, display: "flex", gap: 12, fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}>
        {[
          { label: "CRITICAL (80+)", color: "#DC2626" },
          { label: "HIGH (60-79)", color: "#F59E0B" },
          { label: "MODERATE (40-59)", color: "#3B82F6" },
          { label: "LOW (<40)", color: "#10B981" },
        ].map(item => (
          <div key={item.label} style={{ display: "flex", alignItems: "center", gap: 4, color: "rgba(255,255,255,0.5)" }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: item.color }} />
            {item.label}
          </div>
        ))}
      </div>
    </div>
  );
};

// --- SHAP Explainability Bar ---
const SHAPExplainer = ({ shapValues, diseaseColor }) => {
  const data = Object.entries(shapValues)
    .map(([key, val]) => ({
      name: key === "nlpSignals" ? "NLP Signals" : key.charAt(0).toUpperCase() + key.slice(1),
      value: Math.round(val * 100),
      fill: key === "temperature" ? "#EF4444" : key === "precipitation" ? "#3B82F6" : key === "humidity" ? "#06B6D4" : key === "seasonality" ? "#8B5CF6" : "#F59E0B"
    }))
    .sort((a, b) => b.value - a.value);

  return (
    <div>
      <div style={{ fontSize: 11, color: "rgba(255,255,255,0.5)", marginBottom: 8, fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", letterSpacing: 1 }}>
        SHAP Feature Importance
      </div>
      {data.map((item, i) => (
        <div key={item.name} style={{ marginBottom: 6 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "rgba(255,255,255,0.7)", marginBottom: 2 }}>
            <span>{item.name}</span>
            <span style={{ color: item.fill, fontWeight: 600 }}>{item.value}%</span>
          </div>
          <div style={{ height: 6, background: "rgba(255,255,255,0.05)", borderRadius: 3, overflow: "hidden" }}>
            <div style={{
              height: "100%", borderRadius: 3, background: item.fill,
              width: `${item.value}%`, transition: "width 0.8s ease", transitionDelay: `${i * 100}ms`
            }} />
          </div>
        </div>
      ))}
    </div>
  );
};

// --- NLP Signal Feed ---
const NLPSignalFeed = ({ signals }) => {
  const getSeverityStyle = (sev) => {
    if (sev === "critical") return { bg: "rgba(220,38,38,0.2)", color: "#FCA5A5", border: "rgba(220,38,38,0.4)" };
    if (sev === "high") return { bg: "rgba(245,158,11,0.2)", color: "#FDE68A", border: "rgba(245,158,11,0.4)" };
    return { bg: "rgba(59,130,246,0.2)", color: "#93C5FD", border: "rgba(59,130,246,0.4)" };
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ fontSize: 11, color: "rgba(255,255,255,0.5)", fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", letterSpacing: 1 }}>
        NLP Signal Detection — Live Feed
      </div>
      {signals.map((sig, i) => {
        const style = getSeverityStyle(sig.severity);
        return (
          <div key={i} style={{
            background: style.bg, border: `1px solid ${style.border}`, borderRadius: 8, padding: "10px 12px",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
              <span style={{ fontSize: 10, color: "rgba(255,255,255,0.4)", fontFamily: "'JetBrains Mono', monospace" }}>{sig.source}</span>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 4, background: style.bg, color: style.color, textTransform: "uppercase", fontWeight: 700, letterSpacing: 0.5 }}>{sig.severity}</span>
                <span style={{ fontSize: 9, color: "rgba(255,255,255,0.3)" }}>{sig.date}</span>
              </div>
            </div>
            <div style={{ fontSize: 12, color: "rgba(255,255,255,0.85)", lineHeight: 1.4 }}>{sig.title}</div>
            <div style={{ fontSize: 10, color: "rgba(255,255,255,0.35)", marginTop: 4, fontFamily: "'JetBrains Mono', monospace" }}>
              Confidence: {(sig.confidence * 100).toFixed(0)}%
              <span style={{ display: "inline-block", width: 40, height: 3, background: "rgba(255,255,255,0.1)", borderRadius: 2, marginLeft: 6, verticalAlign: "middle" }}>
                <span style={{ display: "block", width: `${sig.confidence * 100}%`, height: "100%", background: style.color, borderRadius: 2, opacity: 0.6 }} />
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
};

// --- Alert Card ---
const AlertCard = ({ alert }) => {
  const colors = getAlertColor(alert.level);
  return (
    <div style={{
      background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 10, padding: 14,
      borderLeft: `4px solid ${colors.border}`,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <span style={{ fontSize: 10, fontWeight: 700, color: colors.text, textTransform: "uppercase", letterSpacing: 1, fontFamily: "'JetBrains Mono', monospace" }}>
          {alert.level === "critical" ? "⚠ CRITICAL" : alert.level === "warning" ? "⚡ WARNING" : "ℹ ADVISORY"}
        </span>
      </div>
      <div style={{ fontSize: 13, color: "rgba(255,255,255,0.85)", lineHeight: 1.5, marginBottom: 8 }}>{alert.message}</div>
      <div style={{
        display: "inline-block", fontSize: 10, fontWeight: 700, color: colors.text,
        background: `${colors.border}22`, padding: "4px 10px", borderRadius: 4,
        fontFamily: "'JetBrains Mono', monospace", letterSpacing: 0.5,
      }}>
        → {alert.action}
      </div>
    </div>
  );
};

// --- Stat Card ---
const StatCard = ({ label, value, sub, icon }) => (
  <div style={{
    background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12,
    padding: "16px 18px", flex: 1, minWidth: 130,
  }}>
    <div style={{ fontSize: 10, color: "rgba(255,255,255,0.4)", textTransform: "uppercase", letterSpacing: 1, fontFamily: "'JetBrains Mono', monospace", marginBottom: 6 }}>{label}</div>
    <div style={{ fontSize: 26, fontWeight: 700, color: "#fff", fontFamily: "'Space Grotesk', sans-serif" }}>
      {icon && <span style={{ marginRight: 4, fontSize: 18 }}>{icon}</span>}
      {value}
    </div>
    {sub && <div style={{ fontSize: 10, color: "rgba(255,255,255,0.35)", marginTop: 2 }}>{sub}</div>}
  </div>
);

// ============================================================
// MAIN APP
// ============================================================

export default function ClimaHealthApp() {
  const [selectedRegion, setSelectedRegion] = useState(REGIONS[0]);
  const [activeTab, setActiveTab] = useState("overview");
  const [currentTime, setCurrentTime] = useState(new Date());
  const [animateIn, setAnimateIn] = useState(false);

  useEffect(() => {
    setAnimateIn(true);
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    setAnimateIn(false);
    const t = setTimeout(() => setAnimateIn(true), 50);
    return () => clearTimeout(t);
  }, [selectedRegion]);

  const disease = DISEASES[selectedRegion.primaryRisk];

  const customTooltipStyle = {
    backgroundColor: 'rgba(10, 20, 40, 0.95)',
    border: '1px solid rgba(59, 130, 246, 0.3)',
    borderRadius: '8px',
    padding: '8px 12px',
    fontSize: 11,
    color: 'rgba(255,255,255,0.8)',
    fontFamily: "'JetBrains Mono', monospace"
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #050b18 0%, #0a1628 40%, #0d1a30 100%)",
      color: "#fff",
      fontFamily: "'Inter', -apple-system, sans-serif",
      overflow: "auto",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />

      {/* === HEADER === */}
      <div style={{
        padding: "16px 24px",
        borderBottom: "1px solid rgba(59, 130, 246, 0.1)",
        display: "flex", justifyContent: "space-between", alignItems: "center",
        background: "rgba(10, 22, 40, 0.8)", backdropFilter: "blur(10px)",
        position: "sticky", top: 0, zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: "linear-gradient(135deg, #3B82F6, #06B6D4)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 18, fontWeight: 700,
          }}>🌡</div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, fontFamily: "'Space Grotesk', sans-serif", letterSpacing: -0.5 }}>
              ClimaHealth<span style={{ color: "#3B82F6" }}>AI</span>
            </div>
            <div style={{ fontSize: 9, color: "rgba(255,255,255,0.4)", letterSpacing: 2, textTransform: "uppercase", fontFamily: "'JetBrains Mono', monospace" }}>
              Climate-Driven Disease Early Warning System
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#10B981", animation: "pulse 2s infinite" }}>
              <style>{`@keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.4 } }`}</style>
            </div>
            <span style={{ fontSize: 10, color: "rgba(255,255,255,0.5)", fontFamily: "'JetBrains Mono', monospace" }}>LIVE</span>
          </div>
          <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", fontFamily: "'JetBrains Mono', monospace" }}>
            {currentTime.toISOString().split('T')[0]} {currentTime.toTimeString().split(' ')[0]}
          </div>
        </div>
      </div>

      <div style={{ padding: "20px 24px", maxWidth: 1400, margin: "0 auto" }}>

        {/* === GLOBAL STATS BAR === */}
        <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
          <StatCard label="Regions Monitored" value={GLOBAL_STATS.regionsMonitored} icon="🌍" />
          <StatCard label="Active Predictions" value={GLOBAL_STATS.activePredictions} sub="Updated hourly" />
          <StatCard label="Alerts Issued (30d)" value={GLOBAL_STATS.alertsIssued} icon="⚠" />
          <StatCard label="Avg Lead Time" value={`${GLOBAL_STATS.avgLeadTimeDays}d`} sub="Before outbreak onset" />
          <StatCard label="Model Accuracy" value={`${(GLOBAL_STATS.modelAccuracy * 100).toFixed(1)}%`} sub="Validation F1-score" />
        </div>

        {/* === MAP + REGION SELECTOR === */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 16, marginBottom: 20 }}>
          <div>
            <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginBottom: 8, fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", letterSpacing: 1 }}>
              Global Disease Risk Map — 8-Week Forecast
            </div>
            <WorldMap regions={REGIONS} selectedRegion={selectedRegion} onSelectRegion={setSelectedRegion} />
          </div>

          {/* Region Selector */}
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginBottom: 2, fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", letterSpacing: 1 }}>
              Monitored Regions
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6, overflowY: "auto", maxHeight: 360 }}>
              {REGIONS.map(region => {
                const isSelected = selectedRegion?.id === region.id;
                const d = DISEASES[region.primaryRisk];
                return (
                  <div key={region.id} onClick={() => setSelectedRegion(region)} style={{
                    background: isSelected ? "rgba(59, 130, 246, 0.12)" : "rgba(255,255,255,0.02)",
                    border: `1px solid ${isSelected ? "rgba(59, 130, 246, 0.4)" : "rgba(255,255,255,0.05)"}`,
                    borderRadius: 10, padding: "10px 12px", cursor: "pointer",
                    transition: "all 0.2s ease",
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 600, color: isSelected ? "#fff" : "rgba(255,255,255,0.8)" }}>{region.name}</div>
                        <div style={{ fontSize: 10, color: "rgba(255,255,255,0.4)", display: "flex", alignItems: "center", gap: 4, marginTop: 2 }}>
                          <span>{d.icon}</span> {d.name}
                          {region.trend === "rising" && <span style={{ color: "#EF4444", fontSize: 9 }}>▲ RISING</span>}
                          {region.trend === "stable" && <span style={{ color: "#F59E0B", fontSize: 9 }}>● STABLE</span>}
                        </div>
                      </div>
                      <div style={{
                        width: 42, height: 42, borderRadius: 10,
                        background: `${getRiskColor(region.riskScore)}18`,
                        border: `2px solid ${getRiskColor(region.riskScore)}`,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 16, fontWeight: 700, color: getRiskColor(region.riskScore),
                        fontFamily: "'JetBrains Mono', monospace",
                      }}>
                        {region.riskScore}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* === REGION DETAIL SECTION === */}
        <div style={{
          background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: 16, overflow: "hidden", marginBottom: 20,
          opacity: animateIn ? 1 : 0, transform: animateIn ? "translateY(0)" : "translateY(8px)",
          transition: "all 0.4s ease",
        }}>
          {/* Region Header */}
          <div style={{
            padding: "16px 20px",
            background: `linear-gradient(135deg, ${getRiskColor(selectedRegion.riskScore)}08, ${getRiskColor(selectedRegion.riskScore)}15)`,
            borderBottom: `1px solid ${getRiskColor(selectedRegion.riskScore)}20`,
            display: "flex", justifyContent: "space-between", alignItems: "center",
          }}>
            <div>
              <div style={{ fontSize: 20, fontWeight: 700, fontFamily: "'Space Grotesk', sans-serif" }}>
                {selectedRegion.name}
              </div>
              <div style={{ fontSize: 12, color: "rgba(255,255,255,0.5)", marginTop: 2, display: "flex", gap: 16 }}>
                <span>Pop: {selectedRegion.population}</span>
                <span>Health Facilities: {selectedRegion.healthFacilities}</span>
                <span>CHWs: {selectedRegion.chws.toLocaleString()}</span>
                <span>Historical Outbreaks: {selectedRegion.historicalOutbreaks}</span>
              </div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{
                fontSize: 36, fontWeight: 700, color: getRiskColor(selectedRegion.riskScore),
                fontFamily: "'Space Grotesk', sans-serif", lineHeight: 1,
              }}>
                {selectedRegion.riskScore}
              </div>
              <div style={{
                fontSize: 10, fontWeight: 700, color: getRiskColor(selectedRegion.riskScore),
                letterSpacing: 2, fontFamily: "'JetBrains Mono', monospace",
              }}>
                {getRiskLabel(selectedRegion.riskScore)} RISK
              </div>
            </div>
          </div>

          {/* Tabs */}
          <div style={{ display: "flex", gap: 0, borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
            {[
              { id: "overview", label: "Risk Overview" },
              { id: "climate", label: "Climate Forecast" },
              { id: "signals", label: "NLP Signals" },
              { id: "explain", label: "Explainability" },
              { id: "alerts", label: `Alerts (${selectedRegion.alerts.length})` },
            ].map(tab => (
              <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
                padding: "10px 18px", fontSize: 12, fontWeight: activeTab === tab.id ? 600 : 400,
                color: activeTab === tab.id ? "#fff" : "rgba(255,255,255,0.4)",
                background: activeTab === tab.id ? "rgba(59,130,246,0.1)" : "transparent",
                border: "none", borderBottom: activeTab === tab.id ? "2px solid #3B82F6" : "2px solid transparent",
                cursor: "pointer", transition: "all 0.2s ease",
                fontFamily: "'JetBrains Mono', monospace", letterSpacing: 0.5,
              }}>
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div style={{ padding: 20 }}>

            {/* OVERVIEW TAB */}
            {activeTab === "overview" && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                <div>
                  <div style={{ fontSize: 11, color: "rgba(255,255,255,0.5)", marginBottom: 10, fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", letterSpacing: 1 }}>
                    8-Week Risk Forecast — {disease.name}
                  </div>
                  <ResponsiveContainer width="100%" height={220}>
                    <AreaChart data={selectedRegion.weeklyForecast}>
                      <defs>
                        <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={disease.color} stopOpacity={0.3} />
                          <stop offset="95%" stopColor={disease.color} stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="week" tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.4)' }} axisLine={{ stroke: 'rgba(255,255,255,0.1)' }} />
                      <YAxis domain={[40, 100]} tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.4)' }} axisLine={{ stroke: 'rgba(255,255,255,0.1)' }} />
                      <Tooltip contentStyle={customTooltipStyle} />
                      <Area type="monotone" dataKey="risk" stroke={disease.color} fill="url(#riskGrad)" strokeWidth={2} dot={{ r: 3, fill: disease.color }} name="Risk Score" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                <div>
                  <div style={{ fontSize: 11, color: "rgba(255,255,255,0.5)", marginBottom: 10, fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", letterSpacing: 1 }}>
                    Current Climate Conditions
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                    {[
                      { label: "Temperature", value: `${selectedRegion.climate.tempCurrent}°C`, anomaly: `+${selectedRegion.climate.tempAnomaly}°C`, color: "#EF4444" },
                      { label: "Precipitation", value: `${selectedRegion.climate.precipMm}mm`, anomaly: "above avg", color: "#3B82F6" },
                      { label: "Humidity", value: `${selectedRegion.climate.humidity}%`, anomaly: "elevated", color: "#06B6D4" },
                      { label: "Vegetation (NDVI)", value: selectedRegion.climate.ndvi.toFixed(2), anomaly: "declining", color: "#10B981" },
                    ].map(item => (
                      <div key={item.label} style={{
                        background: "rgba(255,255,255,0.03)", borderRadius: 10, padding: 12,
                        border: "1px solid rgba(255,255,255,0.05)",
                      }}>
                        <div style={{ fontSize: 10, color: "rgba(255,255,255,0.4)", fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", letterSpacing: 0.5 }}>{item.label}</div>
                        <div style={{ fontSize: 22, fontWeight: 700, color: item.color, fontFamily: "'Space Grotesk', sans-serif", marginTop: 4 }}>{item.value}</div>
                        <div style={{ fontSize: 10, color: `${item.color}88`, marginTop: 2 }}>{item.anomaly}</div>
                      </div>
                    ))}
                  </div>

                  <div style={{ marginTop: 14 }}>
                    <div style={{
                      background: `${disease.color}12`, border: `1px solid ${disease.color}30`,
                      borderRadius: 10, padding: 12, display: "flex", alignItems: "center", gap: 10,
                    }}>
                      <span style={{ fontSize: 24 }}>{disease.icon}</span>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 600, color: disease.color }}>{disease.name}</div>
                        <div style={{ fontSize: 10, color: "rgba(255,255,255,0.5)" }}>Vector: {disease.vectorType}</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* CLIMATE FORECAST TAB */}
            {activeTab === "climate" && (
              <div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                  <div>
                    <div style={{ fontSize: 11, color: "rgba(255,255,255,0.5)", marginBottom: 10, fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", letterSpacing: 1 }}>
                      Temperature Forecast (°C) — LSTM Model
                    </div>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={selectedRegion.weeklyForecast}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="week" tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.4)' }} axisLine={{ stroke: 'rgba(255,255,255,0.1)' }} />
                        <YAxis tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.4)' }} axisLine={{ stroke: 'rgba(255,255,255,0.1)' }} />
                        <Tooltip contentStyle={customTooltipStyle} />
                        <Line type="monotone" dataKey="temp" stroke="#EF4444" strokeWidth={2} dot={{ r: 3 }} name="Temp (°C)" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: "rgba(255,255,255,0.5)", marginBottom: 10, fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", letterSpacing: 1 }}>
                      Precipitation Forecast (mm) — Prophet Model
                    </div>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={selectedRegion.weeklyForecast}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="week" tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.4)' }} axisLine={{ stroke: 'rgba(255,255,255,0.1)' }} />
                        <YAxis tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.4)' }} axisLine={{ stroke: 'rgba(255,255,255,0.1)' }} />
                        <Tooltip contentStyle={customTooltipStyle} />
                        <Bar dataKey="precip" fill="#3B82F6" radius={[4, 4, 0, 0]} name="Precip (mm)" opacity={0.8} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div style={{ marginTop: 16, padding: 14, background: "rgba(59,130,246,0.06)", borderRadius: 10, border: "1px solid rgba(59,130,246,0.15)" }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: "#93C5FD", marginBottom: 4, fontFamily: "'JetBrains Mono', monospace" }}>MODEL INSIGHT</div>
                  <div style={{ fontSize: 12, color: "rgba(255,255,255,0.7)", lineHeight: 1.6 }}>
                    Climate conditions in <strong>{selectedRegion.name}</strong> show a temperature anomaly of <strong style={{ color: "#EF4444" }}>+{selectedRegion.climate.tempAnomaly}°C</strong> above the 30-year average, combined with precipitation levels <strong style={{ color: "#3B82F6" }}>{selectedRegion.climate.precipMm}mm</strong> above seasonal norms.
                    These conditions are {selectedRegion.riskScore >= 80 ? "strongly" : "moderately"} correlated with historical {disease.name.toLowerCase()} outbreaks in this region.
                    The LSTM model forecasts sustained elevated temperatures through weeks 3–5, creating optimal conditions for {disease.vectorType} proliferation.
                  </div>
                </div>
              </div>
            )}

            {/* NLP SIGNALS TAB */}
            {activeTab === "signals" && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                <NLPSignalFeed signals={selectedRegion.nlpSignals} />
                <div>
                  <div style={{ fontSize: 11, color: "rgba(255,255,255,0.5)", marginBottom: 10, fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", letterSpacing: 1 }}>
                    NLP Pipeline Architecture
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {[
                      { step: "1", label: "Data Ingestion", desc: "GDELT + ProMED + WHO DON feeds", color: "#3B82F6" },
                      { step: "2", label: "Text Classification", desc: "TF-IDF + Fine-tuned BERT / Claude API", color: "#8B5CF6" },
                      { step: "3", label: "Entity Extraction", desc: "Disease, location, severity extraction", color: "#F59E0B" },
                      { step: "4", label: "Confidence Scoring", desc: "Multi-source corroboration weighting", color: "#10B981" },
                      { step: "5", label: "Signal Integration", desc: "Boost or validate climate model predictions", color: "#EF4444" },
                    ].map(item => (
                      <div key={item.step} style={{
                        display: "flex", alignItems: "center", gap: 12,
                        background: "rgba(255,255,255,0.02)", borderRadius: 8, padding: "10px 12px",
                        border: "1px solid rgba(255,255,255,0.05)",
                      }}>
                        <div style={{
                          width: 28, height: 28, borderRadius: 8, background: `${item.color}20`,
                          border: `1px solid ${item.color}40`, display: "flex", alignItems: "center", justifyContent: "center",
                          fontSize: 12, fontWeight: 700, color: item.color, fontFamily: "'JetBrains Mono', monospace",
                        }}>{item.step}</div>
                        <div>
                          <div style={{ fontSize: 12, fontWeight: 600, color: "rgba(255,255,255,0.85)" }}>{item.label}</div>
                          <div style={{ fontSize: 10, color: "rgba(255,255,255,0.4)" }}>{item.desc}</div>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div style={{ marginTop: 12, padding: 10, background: "rgba(245,158,11,0.06)", borderRadius: 8, border: "1px solid rgba(245,158,11,0.15)" }}>
                    <div style={{ fontSize: 10, color: "#FDE68A", fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>SIGNAL SUMMARY</div>
                    <div style={{ fontSize: 11, color: "rgba(255,255,255,0.6)", marginTop: 4, lineHeight: 1.5 }}>
                      {selectedRegion.nlpSignals.length} signal{selectedRegion.nlpSignals.length !== 1 ? "s" : ""} detected from {new Set(selectedRegion.nlpSignals.map(s => s.source)).size} sources.
                      Average confidence: {(selectedRegion.nlpSignals.reduce((a, s) => a + s.confidence, 0) / selectedRegion.nlpSignals.length * 100).toFixed(0)}%.
                      NLP signals are {selectedRegion.nlpSignals.some(s => s.severity === "high" || s.severity === "critical") ? "corroborating" : "weakly supporting"} the climate model's risk prediction.
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* EXPLAINABILITY TAB */}
            {activeTab === "explain" && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                <SHAPExplainer shapValues={selectedRegion.shapValues} diseaseColor={disease.color} />
                <div>
                  <div style={{ fontSize: 11, color: "rgba(255,255,255,0.5)", marginBottom: 10, fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", letterSpacing: 1 }}>
                    Model Confidence & Ensemble Breakdown
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 16 }}>
                    {[
                      { model: "LSTM Climate", weight: "40%", confidence: "87%", color: "#EF4444" },
                      { model: "RF/XGBoost Epi", weight: "45%", confidence: "82%", color: "#3B82F6" },
                      { model: "NLP Signals", weight: "15%", confidence: `${(selectedRegion.nlpSignals.reduce((a, s) => a + s.confidence, 0) / selectedRegion.nlpSignals.length * 100).toFixed(0)}%`, color: "#F59E0B" },
                    ].map(m => (
                      <div key={m.model} style={{
                        background: `${m.color}08`, border: `1px solid ${m.color}25`,
                        borderRadius: 10, padding: 12, textAlign: "center",
                      }}>
                        <div style={{ fontSize: 10, color: "rgba(255,255,255,0.5)", fontFamily: "'JetBrains Mono', monospace" }}>{m.model}</div>
                        <div style={{ fontSize: 20, fontWeight: 700, color: m.color, fontFamily: "'Space Grotesk', sans-serif", margin: "4px 0" }}>{m.confidence}</div>
                        <div style={{ fontSize: 9, color: "rgba(255,255,255,0.35)" }}>Ensemble Weight: {m.weight}</div>
                      </div>
                    ))}
                  </div>

                  <div style={{ padding: 14, background: "rgba(16,185,129,0.06)", borderRadius: 10, border: "1px solid rgba(16,185,129,0.15)" }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: "#6EE7B7", marginBottom: 6, fontFamily: "'JetBrains Mono', monospace" }}>EXPLAINABILITY STATEMENT</div>
                    <div style={{ fontSize: 12, color: "rgba(255,255,255,0.7)", lineHeight: 1.6 }}>
                      This prediction for <strong>{selectedRegion.name}</strong> is driven primarily by <strong style={{ color: Object.entries(selectedRegion.shapValues).sort((a, b) => b[1] - a[1])[0][0] === "temperature" ? "#EF4444" : "#3B82F6" }}>
                      {Object.entries(selectedRegion.shapValues).sort((a, b) => b[1] - a[1])[0][0]}</strong> ({Math.round(Object.entries(selectedRegion.shapValues).sort((a, b) => b[1] - a[1])[0][1] * 100)}% contribution).
                      The ensemble combines three independent models with weighted voting. All feature importances are computed using SHAP (SHapley Additive exPlanations) for full transparency.
                      Model confidence represents validation performance on held-out historical outbreak data for this specific region.
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ALERTS TAB */}
            {activeTab === "alerts" && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  <div style={{ fontSize: 11, color: "rgba(255,255,255,0.5)", marginBottom: 2, fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", letterSpacing: 1 }}>
                    Active Alerts — {selectedRegion.name}
                  </div>
                  {selectedRegion.alerts.map((alert, i) => (
                    <AlertCard key={i} alert={alert} />
                  ))}
                </div>

                <div>
                  <div style={{ fontSize: 11, color: "rgba(255,255,255,0.5)", marginBottom: 10, fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", letterSpacing: 1 }}>
                    Community Health Worker Alert (Plain Language)
                  </div>
                  <div style={{
                    background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: 12, padding: 16,
                  }}>
                    <div style={{ fontSize: 11, color: "#93C5FD", fontWeight: 600, marginBottom: 8, fontFamily: "'JetBrains Mono', monospace" }}>
                      📋 ALERT FOR COMMUNITY HEALTH WORKERS
                    </div>
                    <div style={{ fontSize: 13, color: "rgba(255,255,255,0.85)", lineHeight: 1.7, marginBottom: 12 }}>
                      <strong>{disease.name} risk in {selectedRegion.name.split(",")[0]}</strong> is predicted to
                      {selectedRegion.trend === "rising" ? " increase significantly" : " remain elevated"} over the next <strong>3–5 weeks</strong>.
                    </div>
                    <div style={{ fontSize: 12, color: "rgba(255,255,255,0.7)", lineHeight: 1.7 }}>
                      <strong>Why:</strong> {selectedRegion.primaryRisk === "cholera"
                        ? "Heavy rainfall and flooding are contaminating water sources, creating ideal conditions for cholera transmission."
                        : `Rising temperatures (+${selectedRegion.climate.tempAnomaly}°C above normal) and increased rainfall are creating ideal breeding conditions for ${disease.vectorType.toLowerCase()}.`
                      }
                    </div>
                    <div style={{ marginTop: 12, borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: 12 }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: "#FDE68A", marginBottom: 6, fontFamily: "'JetBrains Mono', monospace" }}>RECOMMENDED ACTIONS:</div>
                      <div style={{ fontSize: 12, color: "rgba(255,255,255,0.7)", lineHeight: 1.8 }}>
                        {selectedRegion.primaryRisk === "dengue" && "① Distribute mosquito nets to households in high-risk zones\n② Eliminate standing water breeding sites\n③ Pre-position fever diagnostic kits at clinics\n④ Activate community awareness campaigns".split("\n").map((l, i) => <div key={i}>{l}</div>)}
                        {selectedRegion.primaryRisk === "malaria" && "① Ensure artemisinin-based combination therapy (ACT) is stocked\n② Distribute insecticide-treated nets (ITNs)\n③ Begin indoor residual spraying in priority areas\n④ Train CHWs on rapid diagnostic test (RDT) protocols".split("\n").map((l, i) => <div key={i}>{l}</div>)}
                        {selectedRegion.primaryRisk === "cholera" && "① Distribute oral rehydration salts (ORS) to all health posts\n② Activate water purification and WASH protocols\n③ Set up oral cholera vaccination (OCV) campaign\n④ Monitor and protect drinking water sources".split("\n").map((l, i) => <div key={i}>{l}</div>)}
                        {selectedRegion.primaryRisk === "zika" && "① Intensify mosquito control near residential areas\n② Enhance prenatal screening for pregnant women\n③ Distribute insect repellent to vulnerable populations\n④ Monitor for Guillain-Barré syndrome cases".split("\n").map((l, i) => <div key={i}>{l}</div>)}
                      </div>
                    </div>

                    <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
                      {["English", "Français", "Español", "Português"].map(lang => (
                        <span key={lang} style={{
                          fontSize: 10, padding: "3px 8px", borderRadius: 4,
                          background: lang === "English" ? "rgba(59,130,246,0.2)" : "rgba(255,255,255,0.05)",
                          color: lang === "English" ? "#93C5FD" : "rgba(255,255,255,0.4)",
                          border: `1px solid ${lang === "English" ? "rgba(59,130,246,0.3)" : "rgba(255,255,255,0.08)"}`,
                          cursor: "pointer",
                        }}>{lang}</span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* === FOOTER === */}
        <div style={{
          padding: "16px 0",
          borderTop: "1px solid rgba(255,255,255,0.05)",
          display: "flex", justifyContent: "space-between", alignItems: "center",
          fontSize: 10, color: "rgba(255,255,255,0.25)", fontFamily: "'JetBrains Mono', monospace",
        }}>
          <div>
            ClimaHealth AI — InnovAIte Hackathon 2026 | Climate-Driven Disease Early Warning System
          </div>
          <div style={{ display: "flex", gap: 16 }}>
            <span>Data: NASA POWER • WHO GHO • GDELT</span>
            <span>Models: LSTM • XGBoost • BERT</span>
            <span>Explainability: SHAP</span>
          </div>
        </div>
      </div>
    </div>
  );
}
