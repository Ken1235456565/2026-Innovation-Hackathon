# train_enhanced.py
"""
融合训练流程：
1. 使用 ClimaHealth AI 的真实数据源（NASA/WHO/GDELT）
2. 应用 DL_climate 的领域特征工程
3. 训练疾病特异性集成模型
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# 假设已实现的模块
from disease_domain_features import DomainFeatureManager
from models.climate_forecaster import ClimateForecaster  # ClimaHealth AI 的 LSTM+Prophet
from models.nlp_detector import OutbreakSignalDetector   # ClimaHealth AI 的 NLP


class EnhancedDiseasePredictor:
    """
    增强版疾病预测器
    结合：
    - ClimaHealth AI 的 RF+GB+LR 集成架构
    - DL_climate 的疾病特异性特征
    """
    
    def __init__(self, disease: str):
        self.disease = disease
        self.feature_manager = DomainFeatureManager()
        
        # 三模型集成（保持 ClimaHealth AI 的设计）
        self.rf_model = RandomForestClassifier(
            n_estimators=150,  # 比原始更多
            max_depth=12,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.lr_model = LogisticRegression(
            max_iter=1000,
            C=0.5,
            random_state=42
        )
        
        # 动态权重（根据疾病特性调整）
        self.weights = self._get_disease_weights(disease)
        self.feature_names = None
        self.feature_importance_ = None
    
    def _get_disease_weights(self, disease: str) -> dict:
        """
        不同疾病的模型权重
        - 疟疾：更依赖环境因素 → RF 权重高
        - 登革热：城市传播 → GB 捕捉非线性
        - 霍乱：突发性强 → LR 快速响应
        """
        weights_map = {
            'malaria': {'rf': 0.50, 'gb': 0.40, 'lr': 0.10},
            'dengue': {'rf': 0.40, 'gb': 0.50, 'lr': 0.10},
            'cholera': {'rf': 0.35, 'gb': 0.45, 'lr': 0.20},
            'zika': {'rf': 0.45, 'gb': 0.45, 'lr': 0.10},
        }
        return weights_map.get(disease, {'rf': 0.45, 'gb': 0.45, 'lr': 0.10})
    
    def train(self, climate_df: pd.DataFrame, labels: np.ndarray):
        """
        训练疾病特异性模型
        
        参数:
            climate_df: 包含气候特征的数据框
            labels: 爆发标签 (0/1)
        """
        # 1. 应用领域特征工程
        print(f"应用 {self.disease} 领域特征工程...")
        enhanced_features = self.feature_manager.engineer_features(
            climate_df, 
            self.disease
        )
        
        # 2. 选择特征列
        base_features = [
            'temperature', 'precipitation', 'humidity',
            'temp_lag_1', 'temp_lag_2', 'temp_lag_4',
            'precip_lag_1', 'precip_lag_2', 'precip_lag_4',
            'temp_rolling_4w', 'precip_rolling_4w',
        ]
        
        # 添加疾病特异性特征
        disease_features = [
            f'{self.disease}_transmission_factor',
            f'{self.disease}_temp_deviation',
            f'{self.disease}_precip_deviation',
            f'{self.disease}_high_risk',
        ]
        
        all_features = base_features + disease_features
        
        # 检查特征存在性
        available_features = [f for f in all_features if f in enhanced_features.columns]
        self.feature_names = available_features
        
        X = enhanced_features[available_features].values
        y = labels
        
        # 3. 划分训练/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 4. 训练三个模型
        print(f"训练 {self.disease} 集成模型...")
        
        print("  [1/3] Random Forest...")
        self.rf_model.fit(X_train, y_train)
        rf_cv = cross_val_score(self.rf_model, X_train, y_train, cv=5, scoring='f1').mean()
        print(f"        CV F1: {rf_cv:.3f}")
        
        print("  [2/3] Gradient Boosting...")
        self.gb_model.fit(X_train, y_train)
        gb_cv = cross_val_score(self.gb_model, X_train, y_train, cv=5, scoring='f1').mean()
        print(f"        CV F1: {gb_cv:.3f}")
        
        print("  [3/3] Logistic Regression...")
        self.lr_model.fit(X_train, y_train)
        lr_cv = cross_val_score(self.lr_model, X_train, y_train, cv=5, scoring='f1').mean()
        print(f"        CV F1: {lr_cv:.3f}")
        
        # 5. 测试集评估
        y_pred, y_pred_proba = self.predict(X_test)
        
        print(f"\n{'='*60}")
        print(f"{self.disease.upper()} 模型性能（测试集）")
        print('='*60)
        print(classification_report(y_test, y_pred, target_names=['正常', '爆发']))
        print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
        
        # 6. 特征重要性
        self.feature_importance_ = dict(zip(
            self.feature_names,
            self.rf_model.feature_importances_
        ))
        
        print(f"\nTop 10 重要特征:")
        for feat, imp in sorted(self.feature_importance_.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {feat:40s} {imp*100:5.2f}%")
    
    def predict(self, X: np.ndarray) -> tuple:
        """集成预测"""
        rf_pred = self.rf_model.predict_proba(X)[:, 1]
        gb_pred = self.gb_model.predict_proba(X)[:, 1]
        lr_pred = self.lr_model.predict_proba(X)[:, 1]
        
        # 加权平均
        ensemble_pred = (
            self.weights['rf'] * rf_pred +
            self.weights['gb'] * gb_pred +
            self.weights['lr'] * lr_pred
        )
        
        return (ensemble_pred > 0.5).astype(int), ensemble_pred
    
    def save(self, path: str):
        """保存模型"""
        joblib.dump({
            'disease': self.disease,
            'rf_model': self.rf_model,
            'gb_model': self.gb_model,
            'lr_model': self.lr_model,
            'weights': self.weights,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance_,
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """加载模型"""
        data = joblib.load(path)
        instance = cls(data['disease'])
        instance.rf_model = data['rf_model']
        instance.gb_model = data['gb_model']
        instance.lr_model = data['lr_model']
        instance.weights = data['weights']
        instance.feature_names = data['feature_names']
        instance.feature_importance_ = data['feature_importance']
        return instance


# ============================================
# 完整训练流程
# ============================================

def train_all_diseases(data_path: str = 'data/real_data.csv'):
    """
    训练所有疾病模型
    
    假设数据格式：
    - 列：date, region, disease, temperature, precipitation, humidity, 
          cases, outbreak (0/1), ...
    """
    print("加载真实数据（来自 fetch_real_data.py）...")
    df = pd.read_csv(data_path)
    
    diseases = ['malaria', 'dengue', 'cholera', 'zika']
    models = {}
    
    for disease in diseases:
        print(f"\n{'='*60}")
        print(f"训练 {disease.upper()} 模型")
        print('='*60)
        
        # 筛选该疾病数据
        disease_df = df[df['disease'] == disease].copy()
        
        if len(disease_df) < 100:
            print(f"⚠️  {disease} 数据不足 (<100 样本)，跳过")
            continue
        
        # 准备特征
        climate_features = disease_df[[
            'temperature', 'precipitation', 'humidity'
        ]].copy()
        
        # 添加滞后特征
        for lag in [1, 2, 4]:
            climate_features[f'temp_lag_{lag}'] = disease_df['temperature'].shift(lag)
            climate_features[f'precip_lag_{lag}'] = disease_df['precipitation'].shift(lag)
        
        # 滚动统计
        climate_features['temp_rolling_4w'] = disease_df['temperature'].rolling(4).mean()
        climate_features['precip_rolling_4w'] = disease_df['precipitation'].rolling(4).sum()
        
        # 删除 NaN
        climate_features = climate_features.dropna()
        labels = disease_df.loc[climate_features.index, 'outbreak'].values
        
        # 训练模型
        model = EnhancedDiseasePredictor(disease)
        model.train(climate_features, labels)
        
        # 保存
        model.save(f'saved_models/enhanced_{disease}_predictor.pkl')
        models[disease] = model
        
        print(f"✅ {disease} 模型已保存")
    
    return models


# ============================================
# 主流程
# ============================================

if __name__ == '__main__':
    import sys
    
    # 检查数据文件
    data_file = 'data/real_data.csv'
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        print("请先运行: python fetch_real_data.py")
        sys.exit(1)
    
    # 训练所有模型
    models = train_all_diseases(data_file)
    
    print("\n" + "="*60)
    print("✅ 所有疾病模型训练完成！")
    print("="*60)
    
    # 输出模型性能总结
    print("\n模型权重配置：")
    for disease, model in models.items():
        print(f"  {disease:10s} → RF:{model.weights['rf']:.2f} | "
              f"GB:{model.weights['gb']:.2f} | LR:{model.weights['lr']:.2f}")
