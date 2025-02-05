# model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# データの読み込み
df = pd.read_csv('models/model/training_progress/training_metrics.csv')

# 各カラムの補間
# 最初にNaNではない値のインデックスを取得
for column in ['loss', 'learning_rate', 'variance', 'bias']:
    valid_indices = df[df[column].notna()].index
    if len(valid_indices) > 0:  # データが存在する場合のみ補間
        valid_values = df.loc[valid_indices, column].values
        df[column] = np.interp(df.index, valid_indices, valid_values)

# グラフの作成
plt.style.use('default')
fig, ((ax1, ax2), (ax3, ax4), (ax5, _)) = plt.subplots(3, 2, figsize=(12, 15))

# Training Loss
ax1.plot(df['step'], df['loss'], label='Loss')
ax1.set_title('Training Loss')
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss')
ax1.legend()

# Learning Rate
ax2.plot(df['step'], df['learning_rate'], label='LR')
ax2.set_title('Learning Rate')
ax2.set_xlabel('Step')
ax2.set_ylabel('Learning Rate')
ax2.legend()

# Perplexity
ax3.plot(df['step'], df['perplexity'], label='Perplexity')
ax3.set_title('Perplexity')
ax3.set_xlabel('Step')
ax3.set_ylabel('Perplexity')
ax3.legend()

# Variance
mask_variance = df['variance'].notna()
ax4.plot(df.loc[mask_variance, 'step'], df.loc[mask_variance, 'variance'], label='Variance')
ax4.set_title('Variance')
ax4.set_xlabel('Step')
ax4.set_ylabel('Variance')
ax4.legend()

# Bias
mask_bias = df['bias'].notna()
ax5.plot(df.loc[mask_bias, 'step'], df.loc[mask_bias, 'bias'], label='Bias')
ax5.set_title('Bias')
ax5.set_xlabel('Step')
ax5.set_ylabel('Bias')
ax5.legend()

# レイアウトの調整
plt.tight_layout()

# 保存
plt.savefig('models/model/training_progress/training_metrics_filled.png')
plt.close() 