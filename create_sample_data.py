#!/usr/bin/env python3
"""
Create sample data for testing the merchant potential scoring model
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_samples = 104

# Restaurant names
restaurant_names = [f"Restaurant_{i}" for i in range(1, n_samples + 1)]

# Generate features
data = {
    '餐厅名': restaurant_names,
    '评分': np.random.uniform(3.5, 4.8, n_samples),
    '评论数': np.random.randint(100, 1500, n_samples),
    '人气指数': np.random.uniform(10, 35, n_samples),
    '人均消费': np.random.choice(['10-20', '20-30', '30-40', '40-50'], n_samples),
    '平均客单价': np.random.uniform(5, 100, n_samples),
    '营业时长': np.random.uniform(4, 12, n_samples),
    '成立时间': np.random.randint(1980, 2020, n_samples),
    '经营年限': np.random.randint(1, 50, n_samples),
    '地理位置': np.random.choice(['Centrum', 'Escamp', 'Zoetermeer', 'Scheveningen'], n_samples),
    '商圈等级': np.random.randint(1, 6, n_samples),
    '同一地区餐厅数量': np.random.randint(5, 50, n_samples),
    '竞争指数': np.random.uniform(0.5, 4.0, n_samples),
    '网站': np.random.randint(0, 2, n_samples),
    '社交媒体': np.random.randint(0, 2, n_samples),
    '外卖': np.random.randint(0, 2, n_samples),
    'YabandPay客户': np.random.randint(0, 2, n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
df.to_excel('data/海牙地区中餐厅_评分卡模型.xlsx', index=False)

print(f"Sample data created with {len(df)} records")
print("Data saved to: data/海牙地区中餐厅_评分卡模型.xlsx")
print("\nFirst few records:")
print(df.head())
print("\nTarget variable distribution:")
print(df['YabandPay客户'].value_counts())
