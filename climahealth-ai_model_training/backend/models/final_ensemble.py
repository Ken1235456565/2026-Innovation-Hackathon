# ensemble_final.py
"""
最终集成风险评估引擎
融合：
1. ClimaHealth AI 的 LSTM/Prophet 气候预测
2. DL_climate 的疾病特异性风险逻辑
3. 增强版 NLP 检测器
4. SHAP 可解释性（ClimaHealth AI）
5. 动态权重调整（新增）
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class FinalRiskEngine:
    """
    融合风险评估引擎
    
    组件权重：
    - Climate Forecast: 35%
    - Disease Model: 50%
    - NLP Signals: 15%
    
    权重根据数据可用性和疾病特性动态调整
    """
    
    def __init__(
        self,
        climate_forecaster,     # ClimaHealth AI 的 ClimateForecaster
        disease_predictor,      # 增强版 EnhancedDiseasePredictor
        nlp_detector,          # 增强版 EnhancedOutbreakDetector
    ):
        self.climate_forecaster = climate_forecaster
        self.disease_predictor = disease_predictor
        self.nlp_detector = nlp_detector
        
        # 基础权重（可根据情况调整）
        self.base_weights = {
            'climate': 0.35,
            'disease': 0.50,
            'nlp': 0.15
        }
    
    def assess_risk(
        self,
        region: str,
        disease: str,
        climate_df: pd.DataFrame,
        recent_cases: np.ndarray,
        news_texts: List[str],
        forecast_weeks: int = 8
    ) -> Dict:
        """
        综合风险评估
        
        参数:
            region: 地区名称 (如 'dhaka_bangladesh')
            disease: 疾病类型 ('malaria'/'dengue'/...)
            climate_df: 历史气候数据（至少 12 周）
            recent_cases: 近期病例数（至少 12 周）
            news_texts: 近期新闻标题列表
            forecast_weeks: 预测未来周数
        
        返回:
            {
                'risk_score': float (0-100),
                'risk_level': str,
                'confidence': float,
                'component_scores': {...},
                'climate_forecast': [...],
                'nlp_signals': [...],
                'recommended_actions': [...],
                'shap_summary': {...},
                'uncertainty_bounds': {...}
            }
        """
        
        # =====================================
        # 1. 气候预测（使用 ClimaHealth AI 的 LSTM+Prophet）
        # =====================================
        climate_forecast = self.climate_forecaster.forecast(
            climate_df, 
            weeks=forecast_weeks
        )
        climate_risk = self._calculate_climate_risk(
            climate_df.iloc[-1], 
            climate_forecast,
            disease
        )
        
        # =====================================
        # 2. 疾病模型预测（使用增强版集成模型）
        # =====================================
        # 应用领域特征工程
        from disease_domain_features import DomainFeatureManager
        feature_manager = DomainFeatureManager()
        enhanced_features = feature_manager.engineer_features(climate_df, disease)
        
        # 提取最新特征向量
        latest_features = self._extract_latest_features(
            enhanced_features, 
            recent_cases,
            disease
        )
        
        # 预测
        _, disease_prob = self.disease_predictor.predict(latest_features.reshape(1, -1))
        disease_risk = disease_prob[0] * 100
        
        # =====================================
        # 3. NLP 信号分析（使用增强版检测器）
        # =====================================
        nlp_results = self.nlp_detector.predict(news_texts)
        
        # 筛选相关疾病的爆发信号
        relevant_signals = [
            r for r in nlp_results 
            if r['is_outbreak'] and r['disease'] == disease
        ]
        
        if relevant_signals:
            nlp_risk = np.mean([s['confidence'] for s in relevant_signals]) * 100
            nlp_urgency = np.max([s['urgency_score'] for s in relevant_signals])
        else:
            nlp_risk = 0
            nlp_urgency = 0
        
        # =====================================
        # 4. 动态权重调整
        # =====================================
        adjusted_weights = self._adjust_weights(
            climate_data_quality=self._assess_data_quality(climate_df),
            nlp_signal_count=len(relevant_signals),
            disease=disease
        )
        
        # =====================================
        # 5. 计算最终风险分数
        # =====================================
        final_score = (
            adjusted_weights['climate'] * climate_risk +
            adjusted_weights['disease'] * disease_risk +
            adjusted_weights['nlp'] * nlp_risk
        )
        
        # 紧急程度加权（NLP 高紧急度时提升总分）
        if nlp_urgency > 0.8:
            final_score = min(100, final_score * 1.15)
        
        # =====================================
        # 6. 风险等级划分
        # =====================================
        risk_level = self._categorize_risk(final_score, nlp_urgency)
        
        # =====================================
        # 7. 不确定性量化（新增）
        # =====================================
        uncertainty = self._compute_uncertainty(
            climate_forecast,
            disease_prob[0],
            nlp_results
        )
        
        # =====================================
        # 8. SHAP 特征重要性（简化版）
        # =====================================
        shap_summary = self._compute_shap_approximation(
            climate_risk,
            disease_risk,
            nlp_risk,
            adjusted_weights
        )
        
        # =====================================
        # 9. 生成行动建议
        # =====================================
        actions = self._generate_actions(
            risk_level, 
            disease,
            enhanced_features.iloc[-1],
            nlp_urgency
        )
        
        # =====================================
        # 10. 返回完整评估
        # =====================================
        return {
            'region': region,
            'disease': disease,
            'risk_score': round(final_score, 1),
            'risk_level': risk_level,
            'confidence': round(1 - uncertainty['total'], 2),
            'outbreak_probability': round(disease_prob[0], 3),
            
            'component_scores': {
                'climate_risk': round(climate_risk, 1),
                'disease_ensemble_risk': round(disease_risk, 1),
                'nlp_signal_risk': round(nlp_risk, 1),
            },
            
            'component_weights': adjusted_weights,
            
            'climate_forecast': climate_forecast,
            
            'nlp_signals': nlp_results,
            'nlp_urgency': round(nlp_urgency, 2),
            
            'recommended_actions': actions,
            
            'shap_summary': shap_summary,
            
            'uncertainty_bounds': {
                'lower': round(max(0, final_score - uncertainty['margin']), 1),
                'upper': round(min(100, final_score + uncertainty['margin']), 1),
                'sources': uncertainty
            }
        }
    
    def _calculate_climate_risk(
        self, 
        current_climate: pd.Series,
        forecast: List[Dict],
        disease: str
    ) -> float:
        """
        计算气候风险分数
        融合 DL_climate 的疾病特异性逻辑
        """
        from disease_domain_features import DomainFeatureManager
        
        manager = DomainFeatureManager()
        model = manager.disease_models[disease]
        optimal_range = model.get_optimal_climate_range()
        
        # 当前气候条件评分
        temp = current_climate['temperature']
        precip = current_climate.get('precipitation', 0)
        
        # 温度风险
        temp_low, temp_high = optimal_range['temperature']
        if temp_low <= temp <= temp_high:
            temp_risk = 80 + (1 - abs(temp - np.mean([temp_low, temp_high])) / (temp_high - temp_low)) * 20
        elif abs(temp - temp_low) < 5 or abs(temp - temp_high) < 5:
            temp_risk = 60
        else:
            temp_risk = 30
        
        # 降水风险
        precip_low, precip_high = optimal_range['precipitation']
        if precip_low <= precip <= precip_high:
            precip_risk = 75
        elif precip > precip_high:
            precip_risk = 85  # 过量降水高风险
        else:
            precip_risk = 40
        
        # 预测趋势调整
        forecast_temps = [f['temperature'] for f in forecast]
        if np.mean(forecast_temps) > temp_high:
            trend_adjustment = 1.1  # 趋势向高风险
        elif np.mean(forecast_temps) < temp_low:
            trend_adjustment = 0.9
        else:
            trend_adjustment = 1.0
        
        base_risk = (temp_risk + precip_risk) / 2
        return min(100, base_risk * trend_adjustment)
    
    def _extract_latest_features(
        self,
        enhanced_features: pd.DataFrame,
        recent_cases: np.ndarray,
        disease: str
    ) -> np.ndarray:
        """提取最新的特征向量"""
        # 基础特征
        latest = enhanced_features.iloc[-1]
        features = [
            latest['temperature'],
            latest['precipitation'],
            latest['humidity'],
        ]
        
        # 滞后特征
        for lag in [1, 2, 4]:
            features.append(latest.get(f'temp_lag_{lag}', latest['temperature']))
            features.append(latest.get(f'precip_lag_{lag}', latest['precipitation']))
        
        # 滚动统计
        features.append(latest.get('temp_rolling_4w', latest['temperature']))
        features.append(latest.get('precip_rolling_4w', latest['precipitation']))
        
        # 疾病特异性特征
        features.append(latest.get(f'{disease}_transmission_factor', 1.0))
        features.append(latest.get(f'{disease}_temp_deviation', 0))
        features.append(latest.get(f'{disease}_precip_deviation', 0))
        features.append(latest.get(f'{disease}_high_risk', 0))
        
        return np.array(features)
    
    def _adjust_weights(
        self,
        climate_data_quality: float,
        nlp_signal_count: int,
        disease: str
    ) -> Dict[str, float]:
        """动态调整组件权重"""
        weights = self.base_weights.copy()
        
        # 1. 气候数据质量低 → 降低气候权重，提升疾病模型权重
        if climate_data_quality < 0.7:
            weights['climate'] *= 0.8
            weights['disease'] += (self.base_weights['climate'] - weights['climate'])
        
        # 2. NLP 信号强烈 → 提升 NLP 权重
        if nlp_signal_count >= 3:
            boost = 0.05
            weights['nlp'] += boost
            weights['climate'] -= boost * 0.5
            weights['disease'] -= boost * 0.5
        
        # 3. 疾病特性调整
        disease_adjustments = {
            'malaria': {'climate': 1.1, 'disease': 1.0, 'nlp': 0.9},  # 气候敏感
            'dengue': {'climate': 1.0, 'disease': 1.1, 'nlp': 1.0},   # 疾病模型重要
            'cholera': {'climate': 0.9, 'disease': 1.0, 'nlp': 1.2},  # 突发事件敏感
            'zika': {'climate': 1.0, 'disease': 1.0, 'nlp': 1.1},
        }
        
        if disease in disease_adjustments:
            for key in weights:
                weights[key] *= disease_adjustments[disease][key]
        
        # 归一化
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def _assess_data_quality(self, climate_df: pd.DataFrame) -> float:
        """评估气候数据质量"""
        missing_rate = climate_df.isnull().mean().mean()
        return 1 - missing_rate
    
    def _categorize_risk(self, score: float, nlp_urgency: float) -> str:
        """风险等级分类"""
        # 高紧急度时降低阈值
        if nlp_urgency > 0.8:
            if score >= 65:
                return 'critical'
            elif score >= 50:
                return 'high'
            elif score >= 35:
                return 'medium'
            else:
                return 'low'
        else:
            if score >= 75:
                return 'critical'
            elif score >= 60:
                return 'high'
            elif score >= 40:
                return 'medium'
            else:
                return 'low'
    
    def _compute_uncertainty(
        self,
        climate_forecast: List[Dict],
        disease_prob: float,
        nlp_results: List[Dict]
    ) -> Dict:
        """不确定性量化"""
        # 1. 气候预测不确定性（假设有置信区间）
        climate_uncertainty = 0.15  # 15% 基础不确定性
        
        # 2. 疾病模型不确定性（基于概率）
        disease_uncertainty = abs(0.5 - disease_prob)  # 越接近 0.5 越不确定
        
        # 3. NLP 不确定性（基于信号一致性）
        if nlp_results:
            nlp_confidences = [r['confidence'] for r in nlp_results]
            nlp_uncertainty = 1 - np.mean(nlp_confidences)
        else:
            nlp_uncertainty = 0.5  # 无信号时中等不确定性
        
        total_uncertainty = np.mean([
            climate_uncertainty,
            disease_uncertainty,
            nlp_uncertainty
        ])
        
        margin = total_uncertainty * 100  # 转换为分数范围
        
        return {
            'climate': round(climate_uncertainty, 2),
            'disease': round(disease_uncertainty, 2),
            'nlp': round(nlp_uncertainty, 2),
            'total': round(total_uncertainty, 2),
            'margin': round(margin, 1)
        }
    
    def _compute_shap_approximation(
        self,
        climate_risk: float,
        disease_risk: float,
        nlp_risk: float,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """SHAP 特征重要性近似"""
        contributions = {
            'climate_factors': climate_risk * weights['climate'] / 100,
            'disease_model': disease_risk * weights['disease'] / 100,
            'nlp_signals': nlp_risk * weights['nlp'] / 100,
        }
        
        # 归一化为比例
        total = sum(contributions.values())
        if total > 0:
            return {k: v/total for k, v in contributions.items()}
        else:
            return {k: 1/3 for k in contributions.keys()}
    
    def _generate_actions(
        self,
        risk_level: str,
        disease: str,
        current_features: pd.Series,
        nlp_urgency: float
    ) -> List[str]:
        """
        生成行动建议
        融合 DL_climate 的分级建议逻辑
        """
        actions = []
        
        # === Critical 级别 ===
        if risk_level == 'critical':
            actions.extend([
                f'🚨 立即启动{disease}紧急防控响应',
                '扩大室内残留喷雾（IRS）覆盖范围' if disease in ['malaria', 'dengue'] else '紧急清洁水源系统',
                f'加强{disease}快速诊断和治疗',
                '分发长效杀虫蚊帐（LLINs）' if disease in ['malaria', 'dengue', 'zika'] else '分发口服补液盐',
                '开展大规模媒介控制活动',
                '加强疫情监测和报告系统',
            ])
            
            if nlp_urgency > 0.8:
                actions.append('📢 启动公众紧急警报系统')
        
        # === High 级别 ===
        elif risk_level == 'high':
            actions.extend([
                '加强蚊虫滋生地清理' if disease in ['malaria', 'dengue', 'zika'] else '监测水源污染',
                f'增加{disease}筛查频率',
                '提升社区健康教育',
                f'确保抗{disease}药物库存充足',
                '准备应急响应资源',
            ])
        
        # === Medium 级别 ===
        elif risk_level == 'medium':
            actions.extend([
                f'维持常规{disease}监测',
                '继续蚊帐使用宣传' if disease in ['malaria', 'dengue', 'zika'] else '维持卫生设施运转',
                '监控气候变化趋势',
                '加强高风险人群保护',
            ])
        
        # === Low 级别 ===
        else:
            actions.extend([
                '继续常规预防措施',
                '保持社区卫生意识',
                '定期检查防护设施完整性',
            ])
        
        return actions


# ============================================
# 使用示例
# ============================================

if __name__ == '__main__':
    # 假设已加载模型
    from models.climate_forecaster import ClimateForecaster
    from train_enhanced import EnhancedDiseasePredictor
    from nlp_detector_enhanced import EnhancedOutbreakDetector
    
    climate_model = ClimateForecaster.load('saved_models/climate_forecaster.pkl')
    malaria_model = EnhancedDiseasePredictor.load('saved_models/enhanced_malaria_predictor.pkl')
    nlp_model = EnhancedOutbreakDetector.load('saved_models/enhanced_nlp_detector.pkl')
    
    # 创建引擎
    engine = FinalRiskEngine(climate_model, malaria_model, nlp_model)
    
    # 模拟输入数据
    climate_data = pd.DataFrame({
        'temperature': [26, 27, 28, 27, 26, 25, 24, 25, 26, 27, 28, 29],
        'precipitation': [120, 150, 180, 200, 220, 180, 160, 140, 130, 150, 170, 190],
        'humidity': [75, 78, 80, 82, 85, 83, 80, 78, 76, 77, 79, 81],
    })
    
    recent_cases = np.array([45, 52, 60, 75, 90, 110, 95, 85, 80, 88, 95, 105])
    
    news = [
        'Malaria cases surge in rural districts following heavy monsoon rains',
        'WHO warns of severe malaria season ahead as temperatures rise',
        'Emergency malaria clinics set up in affected communities',
    ]
    
    # 评估风险
    assessment = engine.assess_risk(
        region='sub_saharan_africa',
        disease='malaria',
        climate_df=climate_data,
        recent_cases=recent_cases,
        news_texts=news,
        forecast_weeks=8
    )
    
    # 打印结果
    print('\n' + '='*70)
    print(f'  {assessment["disease"].upper()} 风险评估 - {assessment["region"]}')
    print('='*70)
    print(f"  风险分数:      {assessment['risk_score']}/100")
    print(f"  风险等级:      {assessment['risk_level'].upper()}")
    print(f"  置信度:        {assessment['confidence']*100:.1f}%")
    print(f"  爆发概率:      {assessment['outbreak_probability']*100:.1f}%")
    
    print(f"\n  组件分数:")
    for comp, score in assessment['component_scores'].items():
        weight = assessment['component_weights'][comp.split('_')[0]]
        print(f"    {comp:30s} {score:>6.1f}/100 (权重: {weight*100:.1f}%)")
    
    print(f"\n  不确定性边界: [{assessment['uncertainty_bounds']['lower']}, {assessment['uncertainty_bounds']['upper']}]")
    
    print(f"\n  推荐行动:")
    for i, action in enumerate(assessment['recommended_actions'], 1):
        print(f"    {i}. {action}")
    
    print('='*70)
