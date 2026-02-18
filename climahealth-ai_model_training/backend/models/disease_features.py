# disease_domain_features.py
"""
融合 DL_climate 的领域建模 + ClimaHealth AI 的通用框架
为每种疾病定制特征工程，基于流行病学原理
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class DiseaseFeatureEngineer(ABC):
    """疾病特征工程基类"""
    
    @abstractmethod
    def compute_transmission_factor(self, climate_df: pd.DataFrame) -> np.ndarray:
        """计算疾病传播因子（基于生物学/环境机制）"""
        pass
    
    @abstractmethod
    def get_optimal_climate_range(self) -> dict:
        """返回疾病最适宜的气候条件"""
        pass


class MalariaFeatures(DiseaseFeatureEngineer):
    """疟疾特异性特征（融合 DL_climate 的高斯响应曲线）"""
    
    def compute_transmission_factor(self, climate_df: pd.DataFrame) -> np.ndarray:
        """
        基于 Anopheles 蚊媒生物学的传播因子
        参考：DL_climate.py 的疟疾建模
        """
        temp = climate_df['temperature'].values
        precip = climate_df['precipitation'].values
        humidity = climate_df['humidity'].values
        
        # 1. 温度响应：高斯曲线（20-30°C最适宜）
        # 峰值在 25°C，标准差 5°C
        temp_factor = np.exp(-0.5 * ((temp - 25) / 5) ** 2)
        
        # 2. 降水响应：适度降水增加风险，过量反而降低
        # 50-200mm 最佳积水环境
        precip_factor = np.where(
            precip < 200,
            1 + 0.003 * precip,
            1 + 0.003 * 200 - 0.002 * (precip - 200)
        )
        precip_factor = np.clip(precip_factor, 0.5, 2.0)
        
        # 3. 湿度响应：>60% 显著增加风险
        humidity_factor = np.where(
            humidity > 60,
            0.5 + 0.01 * humidity,
            0.8
        )
        
        # 4. 滞后效应：蚊媒生命周期约 2-4 周
        lag_factor = self._compute_lag_effect(climate_df, lag_weeks=[2, 4])
        
        return temp_factor * precip_factor * humidity_factor * lag_factor
    
    def _compute_lag_effect(self, df: pd.DataFrame, lag_weeks: list) -> np.ndarray:
        """计算滞后累积效应"""
        lag_temp = np.zeros(len(df))
        lag_precip = np.zeros(len(df))
        
        for lag in lag_weeks:
            if len(df) > lag:
                lag_temp += df['temperature'].shift(lag).fillna(df['temperature'].mean()).values
                lag_precip += df['precipitation'].shift(lag).fillna(0).values
        
        # 归一化
        lag_temp /= len(lag_weeks)
        lag_precip /= len(lag_weeks)
        
        return 1 + 0.01 * (lag_temp - 25) + 0.0005 * lag_precip
    
    def get_optimal_climate_range(self) -> dict:
        return {
            'temperature': (20, 32),
            'precipitation': (50, 300),
            'humidity': (60, 90),
            'altitude_max': 2500  # 疟疾在高海拔受限
        }


class DengueFeatures(DiseaseFeatureEngineer):
    """登革热特异性特征（Aedes 蚊媒）"""
    
    def compute_transmission_factor(self, climate_df: pd.DataFrame) -> np.ndarray:
        temp = climate_df['temperature'].values
        precip = climate_df['precipitation'].values
        
        # 1. 温度响应：Aedes 最适温度 22-34°C（比疟疾更耐热）
        temp_factor = np.exp(-0.5 * ((temp - 28) / 6) ** 2)
        
        # 2. 降水响应：城市积水容器（轮胎、花盆）
        # 短时强降雨后 1-2 周风险最高
        precip_factor = 1 + 0.004 * np.minimum(precip, 250)
        
        # 3. 城市化效应（如有数据）：登革热是城市疾病
        # 这里用降水变异性作为代理变量
        precip_std = climate_df['precipitation'].rolling(4).std().fillna(0).values
        urban_proxy = 1 + 0.002 * precip_std
        
        # 4. 滞后效应：1-3 周（比疟疾更短）
        lag_factor = self._compute_short_lag(climate_df, lag_weeks=[1, 2, 3])
        
        return temp_factor * precip_factor * urban_proxy * lag_factor
    
    def _compute_short_lag(self, df: pd.DataFrame, lag_weeks: list) -> np.ndarray:
        """短期滞后效应"""
        lag_precip = sum(
            df['precipitation'].shift(lag).fillna(0).values 
            for lag in lag_weeks
        ) / len(lag_weeks)
        return 1 + 0.001 * lag_precip
    
    def get_optimal_climate_range(self) -> dict:
        return {
            'temperature': (22, 34),
            'precipitation': (100, 400),
            'humidity': (65, 85),
            'altitude_max': 1800
        }


class CholeraFeatures(DiseaseFeatureEngineer):
    """霍乱特异性特征（水源传播）"""
    
    def compute_transmission_factor(self, climate_df: pd.DataFrame) -> np.ndarray:
        temp = climate_df['temperature'].values
        precip = climate_df['precipitation'].values
        
        # 1. 温度响应：霍乱弧菌在 25-35°C 快速繁殖
        temp_factor = np.where(
            (temp >= 25) & (temp <= 35),
            1.5,
            0.8
        )
        
        # 2. 洪水事件检测：短期内极端降水
        # 使用 7 日滑动窗口检测异常
        precip_7d = climate_df['precipitation'].rolling(7).sum().fillna(0).values
        precip_threshold = np.percentile(precip_7d, 90)
        flood_factor = np.where(precip_7d > precip_threshold, 2.0, 1.0)
        
        # 3. 水源污染代理：持续高降水 + 高温
        contamination_risk = (precip > 100) & (temp > 25)
        contamination_factor = np.where(contamination_risk, 1.5, 1.0)
        
        # 4. 滞后效应：霍乱潜伏期短（1-5天），但传播链长
        lag_factor = self._compute_waterborne_lag(climate_df)
        
        return temp_factor * flood_factor * contamination_factor * lag_factor
    
    def _compute_waterborne_lag(self, df: pd.DataFrame) -> np.ndarray:
        """水源性疾病的滞后模式"""
        # 1 周滞后：洪水后水源污染
        lag_precip_1w = df['precipitation'].shift(1).fillna(0).values
        return 1 + 0.005 * lag_precip_1w
    
    def get_optimal_climate_range(self) -> dict:
        return {
            'temperature': (15, 40),
            'precipitation': (0, 500),  # 极端降水是主要风险
            'humidity': (50, 95),
            'flood_threshold': 150  # mm/week
        }


class ZikaFeatures(DiseaseFeatureEngineer):
    """寨卡病毒特征（与登革热相似但传播力更弱）"""
    
    def compute_transmission_factor(self, climate_df: pd.DataFrame) -> np.ndarray:
        # 与登革热类似，但系数更保守
        dengue_model = DengueFeatures()
        base_factor = dengue_model.compute_transmission_factor(climate_df)
        
        # Zika 传播效率约为登革热的 0.6-0.8
        return base_factor * 0.7
    
    def get_optimal_climate_range(self) -> dict:
        return {
            'temperature': (22, 34),
            'precipitation': (100, 350),
            'humidity': (65, 85),
            'altitude_max': 1500
        }


# ============================================
# 特征工程管理器
# ============================================

class DomainFeatureManager:
    """管理所有疾病的特征工程"""
    
    def __init__(self):
        self.disease_models = {
            'malaria': MalariaFeatures(),
            'dengue': DengueFeatures(),
            'cholera': CholeraFeatures(),
            'zika': ZikaFeatures(),
        }
    
    def engineer_features(
        self, 
        climate_df: pd.DataFrame, 
        disease: str
    ) -> pd.DataFrame:
        """
        为指定疾病生成增强特征
        
        参数:
            climate_df: 包含 temperature, precipitation, humidity 的数据框
            disease: 'malaria' | 'dengue' | 'cholera' | 'zika'
        
        返回:
            增强后的特征数据框
        """
        if disease not in self.disease_models:
            raise ValueError(f"Unknown disease: {disease}")
        
        model = self.disease_models[disease]
        
        # 1. 计算疾病特异性传播因子
        transmission_factor = model.compute_transmission_factor(climate_df)
        
        # 2. 创建特征副本
        features = climate_df.copy()
        features[f'{disease}_transmission_factor'] = transmission_factor
        
        # 3. 添加最优气候偏离度
        optimal_range = model.get_optimal_climate_range()
        features[f'{disease}_temp_deviation'] = self._compute_deviation(
            climate_df['temperature'],
            optimal_range['temperature']
        )
        features[f'{disease}_precip_deviation'] = self._compute_deviation(
            climate_df['precipitation'],
            optimal_range['precipitation']
        )
        
        # 4. 添加风险警报
        features[f'{disease}_high_risk'] = (
            (climate_df['temperature'].between(*optimal_range['temperature'])) &
            (climate_df['precipitation'].between(*optimal_range['precipitation']))
        ).astype(int)
        
        return features
    
    def _compute_deviation(
        self, 
        values: pd.Series, 
        optimal_range: tuple
    ) -> pd.Series:
        """计算与最优区间的偏离度"""
        low, high = optimal_range
        return values.apply(lambda x: 
            0 if low <= x <= high else min(abs(x - low), abs(x - high))
        )


# ============================================
# 使用示例
# ============================================

if __name__ == '__main__':
    # 模拟气候数据
    np.random.seed(42)
    n_weeks = 100
    
    climate_data = pd.DataFrame({
        'temperature': 25 + 5 * np.sin(2 * np.pi * np.arange(n_weeks) / 52) + np.random.normal(0, 2, n_weeks),
        'precipitation': np.maximum(0, 150 + 50 * np.sin(2 * np.pi * np.arange(n_weeks) / 52) + np.random.gamma(2, 20, n_weeks)),
        'humidity': 75 + 10 * np.sin(2 * np.pi * np.arange(n_weeks) / 52) + np.random.normal(0, 5, n_weeks),
    })
    
    # 为疟疾生成特征
    manager = DomainFeatureManager()
    malaria_features = manager.engineer_features(climate_data, 'malaria')
    
    print("疟疾特征工程示例：")
    print(malaria_features[['temperature', 'malaria_transmission_factor', 'malaria_high_risk']].head(10))
    print(f"\n高风险周数：{malaria_features['malaria_high_risk'].sum()}/{len(malaria_features)}")
