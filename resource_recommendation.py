import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class ResourceRecommendationModel:
    """
    根据用户画像推荐最合适的帮扶资源
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # 真实可用的帮扶资源库
        self.resource_database = {
            "kenya": {
                "free_clinics": [
                    {
                        "name": "Nairobi County Health Services",
                        "type": "free_malaria_treatment",
                        "services": ["Free malaria testing", "Artemisinin combination therapy (ACT)", "Follow-up care"],
                        "contact": "+254-20-2217475",
                        "website": "https://nairobi.go.ke/health-services/",
                        "application": "Walk-in with Kenyan ID or Huduma Namba",
                        "cost": "Free",
                        "eligibility": "All Kenyan residents"
                    },
                    {
                        "name": "AMREF Flying Doctors",
                        "type": "mobile_clinic",
                        "services": ["Mobile malaria testing", "Treatment", "Community outreach"],
                        "contact": "+254-20-6993000",
                        "website": "https://amref.org/kenya/",
                        "application": "Check schedule on website or call hotline",
                        "cost": "Free for low-income",
                        "eligibility": "Priority to remote areas"
                    }
                ],
                "financial_aid": [
                    {
                        "name": "Kenya National Hospital Insurance Fund (NHIF)",
                        "type": "health_insurance",
                        "services": ["Covers malaria treatment", "Hospitalization", "Medication subsidy"],
                        "contact": "https://www.nhif.or.ke/",
                        "application": "Online enrollment with ID/Passport and payslip",
                        "cost": "KES 500/month (~$4)",
                        "eligibility": "All Kenyan citizens"
                    },
                    {
                        "name": "Equity Bank Emergency Loan",
                        "type": "emergency_loan",
                        "services": ["Quick cash up to KES 50,000", "1% monthly interest"],
                        "contact": "Dial *247# or Equitel *247#",
                        "application": "Instant via mobile phone",
                        "cost": "1% interest per month",
                        "eligibility": "Must have M-Pesa account with 6+ months history"
                    }
                ],
                "food_assistance": [
                    {
                        "name": "World Food Programme (WFP) Kenya",
                        "type": "emergency_food",
                        "services": ["Monthly food ration", "Cash transfer program"],
                        "contact": "+254-20-7622000",
                        "application": "Register through local chief's office",
                        "cost": "Free",
                        "eligibility": "Vulnerable households in informal settlements"
                    }
                ]
            },
            "bangladesh": {
                "free_clinics": [
                    {
                        "name": "Chittagong Medical College Hospital",
                        "type": "free_cholera_treatment",
                        "services": ["Free ORS distribution", "IV rehydration", "Antibiotics if severe"],
                        "contact": "+880-31-2525969",
                        "website": "Chawk Bazar, Chittagong",
                        "application": "Walk-in, 24/7 emergency",
                        "cost": "Free",
                        "eligibility": "All patients"
                    },
                    {
                        "name": "ICDDR,B Cholera Hospital",
                        "type": "specialized_treatment",
                        "services": ["World-class cholera treatment", "Free care", "Research facility"],
                        "contact": "+880-2-9827001-10",
                        "website": "https://www.icddrb.org/",
                        "application": "Walk-in, no appointment needed",
                        "cost": "Free",
                        "eligibility": "All patients"
                    }
                ],
                "financial_aid": [
                    {
                        "name": "bKash Emergency Cash",
                        "type": "mobile_money_loan",
                        "services": ["Quick loan up to BDT 10,000", "24-hour approval"],
                        "contact": "Dial *247# from bKash number",
                        "application": "Instant mobile application",
                        "cost": "Interest rates apply",
                        "eligibility": "Active bKash users"
                    }
                ],
                "food_assistance": [
                    {
                        "name": "Government Food Distribution",
                        "type": "subsidized_food",
                        "services": ["Rice at subsidized rates", "Emergency rations"],
                        "contact": "Local Union Parishad office",
                        "application": "Register with local authorities",
                        "cost": "Heavily subsidized",
                        "eligibility": "Low-income families"
                    }
                ]
            },
            "nigeria": {
                "free_clinics": [
                    {
                        "name": "Lagos State Malaria Control Programme",
                        "type": "free_malaria_treatment",
                        "services": ["Free RDT testing", "ACT treatment", "Insecticide-treated nets"],
                        "contact": "+234-1-7747500",
                        "website": "https://health.lagosstate.gov.ng/",
                        "application": "Visit any Primary Health Center in Lagos",
                        "cost": "Free",
                        "eligibility": "All Lagos residents"
                    }
                ],
                "financial_aid": [
                    {
                        "name": "Lagos State Social Security",
                        "type": "cash_transfer",
                        "services": ["Monthly stipend", "Healthcare subsidy"],
                        "contact": "LASRRA office",
                        "application": "Register with Lagos State Resident card",
                        "cost": "Free",
                        "eligibility": "Vulnerable Lagos residents"
                    }
                ]
            }
        }
    
    def generate_training_data(self):
        """生成训练数据"""
        data = []
        
        for _ in range(1000):
            user = {
                "liquid_assets": np.random.randint(50, 1000),
                "daily_income": np.random.randint(5, 100),
                "household_size": np.random.randint(1, 8),
                "has_mobile_money": np.random.choice([0, 1], p=[0.3, 0.7]),
                "disease_risk_score": np.random.randint(40, 100),
                "health_critical": np.random.choice([0, 1], p=[0.7, 0.3]),
                
                "needs_clinic": 0,
                "needs_loan": 0,
                "needs_food": 0
            }
            
            # 规则引擎
            if user["liquid_assets"] < 150 or user["health_critical"]:
                user["needs_clinic"] = 1
            
            if user["liquid_assets"] < 200 and user["daily_income"] < 20:
                user["needs_loan"] = 1
                user["needs_food"] = 1
            
            data.append(user)
        
        return pd.DataFrame(data)
    
    def train(self):
        """训练模型"""
        df = self.generate_training_data()
        
        features = ["liquid_assets", "daily_income", "household_size", 
                   "has_mobile_money", "disease_risk_score", "health_critical"]
        
        X = df[features]
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练三个分类器
        for target in ["needs_clinic", "needs_loan", "needs_food"]:
            y = df[target]
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            self.models[target] = model
        
        self.is_trained = True
        print("✅ Resource recommendation model trained")
    
    def predict_needs(self, user_profile):
        """预测用户需求"""
        if not self.is_trained:
            self.train()
        
        features = np.array([[
            user_profile.get("liquid_assets", 150),
            user_profile.get("daily_income", 35),
            user_profile.get("household_size", 4),
            user_profile.get("has_mobile_money", 1),
            user_profile.get("disease_risk_score", 70),
            user_profile.get("health_critical", 0)
        ]])
        
        features_scaled = self.scaler.transform(features)
        
        predictions = {}
        for need, model in self.models.items():
            prob = model.predict_proba(features_scaled)[0][1]
            predictions[need] = prob
        
        return sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    def get_resources(self, country, needs):
        """获取匹配的资源"""
        country_key = country.lower().replace(" ", "_")
        resources = self.resource_database.get(country_key, {})
        
        recommendations = []
        
        for need_type, probability in needs:
            if probability < 0.4:
                continue
            
            priority = 1 if probability > 0.7 else 2
            
            if "clinic" in need_type:
                for resource in resources.get("free_clinics", []):
                    recommendations.append({
                        "priority": priority,
                        "category": "🏥 Medical Care",
                        "resource": resource,
                        "match_score": probability
                    })
            
            elif "loan" in need_type:
                for resource in resources.get("financial_aid", []):
                    if "loan" in resource["type"].lower():
                        recommendations.append({
                            "priority": priority,
                            "category": "💰 Financial Aid",
                            "resource": resource,
                            "match_score": probability
                        })
            
            elif "food" in need_type:
                for resource in resources.get("food_assistance", []):
                    recommendations.append({
                        "priority": priority,
                        "category": "🍚 Food Security",
                        "resource": resource,
                        "match_score": probability
                    })
        
        recommendations.sort(key=lambda x: (x["priority"], -x["match_score"]))
        return recommendations[:4]  # 返回前4个
