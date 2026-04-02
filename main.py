#!/usr/bin/env python3
"""
Merchant Potential Scoring Model - Main Entry Point

This script implements a complete merchant potential scoring pipeline:
1. Data loading and preprocessing
2. Feature binning and WOE calculation
3. Model training and evaluation
4. Scorecard generation and segmentation
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import DataPreprocessor
from src.binning_woe import BinningWOE
from src.model_training import ModelTrainer
from src.scorecard_generator import ScorecardGenerator


def main():
    """Main execution function"""
    print("=" * 60)
    print("商户潜力评分模型 - Merchant Potential Scoring Model")
    print("=" * 60)
    
    # Configuration
    config = {
        'data_file': 'data/海牙地区中餐厅_评分卡模型.xlsx',
        'output_file': 'results/海牙地区中餐厅_评分卡打分结果.xlsx',
        'target_col': 'YabandPay客户',
        'selected_features': ['营业时长', '竞争指数', '人气指数', '经营年限', '外卖'],
        'scorecard_params': {
            'pdo': 50,
            'base_score': 600,
            'base_odds': 1/20
        }
    }
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    try:
        # ===================== 1. Data Loading and Preprocessing =====================
        print("\n1. 数据加载和预处理...")
        preprocessor = DataPreprocessor(config['data_file'])
        data = preprocessor.load_data()
        
        # Display basic statistics
        print(f"数据形状: {data.shape}")
        print(f"目标变量分布: {data[config['target_col']].value_counts()}")
        
        # Get feature statistics
        feature_stats = preprocessor.get_descriptive_stats()
        print("\n数值特征统计:")
        print(feature_stats)
        
        # Calculate correlations
        spearman_corr = preprocessor.calculate_spearman_correlation()
        print("\nSpearman相关性分析:")
        print(spearman_corr)
        
        # Calculate VIF
        vif_analysis = preprocessor.calculate_vif()
        print("\nVIF分析:")
        print(vif_analysis)
        
        # ===================== 2. Feature Binning and WOE Calculation =====================
        print("\n2. 特征分箱和WOE计算...")
        binning_woe = BinningWOE(config['target_col'])
        
        # Get feature columns
        feature_cols = preprocessor.get_feature_columns()
        
        # Calculate IV summary
        iv_summary = binning_woe.calculate_iv_summary(
            data, 
            feature_cols['numeric'], 
            feature_cols['binary']
        )
        print("\nIV值汇总:")
        print(iv_summary)
        
        # Perform binning for selected features
        bin_map = {}
        for col in config['selected_features']:
            print(f"\n对变量 {col} 进行分箱...")
            bin_table = binning_woe.bin_and_check_monotonic(
                data, col, config['target_col'], n_bins=5, plot=False
            )
            bin_map[col] = bin_table
        
        # Apply WOE transformation
        X_woe = pd.DataFrame(index=data.index)
        for var in config['selected_features']:
            X_woe[f"{var}_WOE"] = binning_woe.apply_woe(data[var], bin_map[var])
        
        y = data[config['target_col']]
        print(f"\nWOE特征矩阵形状: {X_woe.shape}")
        
        # ===================== 3. Model Training =====================
        print("\n3. 模型训练...")
        trainer = ModelTrainer(config['target_col'])
        model = trainer.train_logistic_regression(X_woe, y)
        
        # Get model summary
        model_summary = trainer.get_model_summary()
        
        # ===================== 4. Scorecard Generation =====================
        print("\n4. 评分卡生成...")
        scorecard_gen = ScorecardGenerator(**config['scorecard_params'])
        
        # Calculate scores
        scores = scorecard_gen.calculate_scores(
            X_woe, 
            model_summary['coefficients'], 
            model_summary['intercept']
        )
        
        # Generate scorecard tables
        scorecard_tables = scorecard_gen.generate_scorecard_tables(
            X_woe, 
            model_summary['coefficients'], 
            bin_map
        )
        
        # Assign segments
        segments = scorecard_gen.assign_segments(scores)
        
        # ===================== 5. Model Evaluation =====================
        print("\n5. 模型评估...")
        auc_value, ks_value, ks_df = trainer.evaluate_model(
            y, scores, plot_roc=True
        )
        
        print(f"AUC = {auc_value:.4f}")
        print(f"KS = {ks_value:.4f}")
        
        # ===================== 6. Results Export =====================
        print("\n6. 结果导出...")
        
        # Add scores and segments to original data
        results_data = data.copy()
        results_data["scorecard_score"] = scores
        results_data["segment"] = segments
        
        # Display sample results
        print("\n样本结果预览:")
        print(results_data[["餐厅名", "scorecard_score", "segment"]].head(10))
        
        # Export results
        scorecard_gen.export_scorecard(
            results_data, scores, segments, config['output_file']
        )
        
        # Save model parameters
        model_params = {
            'coefficients': model_summary['coefficients'].tolist(),
            'intercept': float(model_summary['intercept']),
            'feature_names': X_woe.columns.tolist(),
            'scorecard_params': scorecard_gen.get_scorecard_summary()
        }
        
        import json
        with open('models/model_parameters.json', 'w', encoding='utf-8') as f:
            json.dump(model_params, f, ensure_ascii=False, indent=2)
        
        print("\n模型参数已保存到: models/model_parameters.json")
        
        # ===================== 7. Summary =====================
        print("\n" + "=" * 60)
        print("模型训练完成!")
        print("=" * 60)
        print(f"数据样本数: {len(data)}")
        print(f"特征数量: {len(config['selected_features'])}")
        print(f"AUC: {auc_value:.4f}")
        print(f"KS: {ks_value:.4f}")
        print(f"评分范围: {scores.min():.1f} - {scores.max():.1f}")
        print("\n分层结果:")
        print(segments.value_counts().sort_index())
        print(f"\n结果文件: {config['output_file']}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
