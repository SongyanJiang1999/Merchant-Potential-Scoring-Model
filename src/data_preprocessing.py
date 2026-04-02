"""
Data preprocessing module for merchant potential scoring model
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor


class DataPreprocessor:
    """Handle data loading and initial preprocessing"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.target_col = "YabandPay客户"
        
        # Feature categories
        self.candidate_numeric = ["人气指数", "平均客单价", "经营年限", "营业时长", "竞争指数", "商圈等级"]
        self.candidate_binary = ["网站", "社交媒体", "外卖"]
        self.candidate_categorical = ["商圈等级"]
        
    def load_data(self) -> pd.DataFrame:
        """Load data from Excel file"""
        self.data = pd.read_excel(self.data_path)
        return self.data
    
    def get_feature_columns(self) -> dict:
        """Get categorized feature columns"""
        if self.data is None:
            self.load_data()
            
        numeric_cols = [c for c in self.candidate_numeric if c in self.data.columns]
        binary_cols = [c for c in self.candidate_binary if c in self.data.columns]
        categorical_cols = [c for c in self.candidate_categorical if c in self.data.columns]
        
        return {
            'numeric': numeric_cols,
            'binary': binary_cols,
            'categorical': categorical_cols
        }
    
    def get_descriptive_stats(self) -> pd.DataFrame:
        """Get descriptive statistics for numeric features"""
        if self.data is None:
            self.load_data()
            
        feature_cols = self.get_feature_columns()
        return self.data[feature_cols['numeric']].describe().T
    
    def calculate_spearman_correlation(self) -> pd.DataFrame:
        """Calculate Spearman correlation between features and target"""
        if self.data is None:
            self.load_data()
            
        feature_cols = self.get_feature_columns()
        spearman_results = []
        
        for col in feature_cols['numeric']:
            df_tmp = self.data[[col, self.target_col]].dropna()
            if df_tmp[self.target_col].nunique() <= 1:
                continue
            r, p = spearmanr(df_tmp[col], df_tmp[self.target_col])
            spearman_results.append({'变量': col, 'SpearmanR': r, 'P值': p})
        
        return pd.DataFrame(spearman_results).sort_values(by='SpearmanR', ascending=False)
    
    def calculate_vif(self) -> pd.DataFrame:
        """Calculate Variance Inflation Factor for numeric features"""
        if self.data is None:
            self.load_data()
            
        feature_cols = self.get_feature_columns()
        vif_df = pd.DataFrame()
        vif_df['变量'] = feature_cols['numeric']
        
        X_vif = self.data[feature_cols['numeric']].dropna()
        vif_values = []
        
        for i in range(X_vif.shape[1]):
            vif_values.append(variance_inflation_factor(X_vif.values, i))
        
        vif_df['VIF'] = vif_values
        return vif_df
    
    def get_selected_features(self) -> list:
        """Get pre-selected features for modeling"""
        return ["营业时长", "竞争指数", "人气指数", "经营年限", "外卖"]
