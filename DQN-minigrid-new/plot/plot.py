import os
import random

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

smooth_window_size = 1500
# 定义文件名和算法名称的映射
file_to_alg = {
    'run-6_EPER_10000-tag-reward.csv': 'EPER (Ours)',
    'run-2_CER_10000-tag-reward.csv': 'CER',
    'run-4_proportional_PER_10000-tag-reward.csv': 'PER',
    'run-1_default_10000-tag-reward.csv': 'Vanilla ER',
    # "run-runs_SimpleVSS-v0_24-tag-avg_sample_index_delta.csv":'APER (Ours)',
    # "run-runs_SimpleVSS-v0_25-tag-avg_sample_index_delta.csv":'Vanilla ER',
    # "run-runs_SimpleVSS-v0_26-tag-avg_sample_index_delta.csv":'PER',
    # "run-runs_SimpleVSS-v0_27-tag-avg_sample_index_delta.csv":'CER',
}

# 读取数据
data = {}
for file in os.listdir('plot/formal'):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join('plot/formal', file))
        alg = file_to_alg[file]
        if 'reward' in file:
            metric = "reward"
        if alg not in data:
            data[alg] = {}
        data[alg][metric] = df

# 绘制 reward 图
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
palette = sns.color_palette("deep", len(data))
for i, alg in enumerate(data):
    df = data[alg]['reward']
    df['Smoothed'] = df['Value'].rolling(window=smooth_window_size, min_periods=1).mean()
    df['Std'] = df['Value'].rolling(window=smooth_window_size, min_periods=1).std()/10
    plt.plot(df['Step'][100:], df['Smoothed'][100:], label=alg, color=palette[i])
    plt.fill_between(df['Step'][100:], df['Smoothed'][100:]-df['Std'][100:], df['Smoothed'][100:]+df['Std'][100:], alpha=0.1, color=palette[i])
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.title('Reward Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('plot/reward.jpg')