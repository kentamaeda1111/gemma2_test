import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSVファイルを読み込む
df = pd.read_csv('models/noattention_withprompt_1980/training_progress/training_metrics.csv')

# 欠損値を含む行を削除
df = df.dropna(subset=['loss', 'learning_rate'])

# プロットのスタイル設定
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1])
fig.tight_layout(pad=3.0)

# Training Lossのプロット
ax1.scatter(df['step'], df['loss'], label='Training Loss', color='blue', alpha=0.5, s=20)
ax1.plot(df['step'], df['loss'], color='blue', alpha=0.3)

# 移動平均の計算と描画
window_size = 5
df['moving_avg_loss'] = df['loss'].rolling(window=window_size, min_periods=1).mean()
ax1.plot(df['step'], df['moving_avg_loss'], '--', label='Moving Average Loss', color='orange', linewidth=2)

ax1.set_xlabel('Training Steps')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss Over Time')
ax1.legend()
ax1.grid(True)
ax1.set_xlim(0, 2000)
ax1.set_ylim(0, 5.0)

# Learning Rateのプロット
ax2.scatter(df['step'], df['learning_rate'], color='blue', alpha=0.5, s=20)
ax2.plot(df['step'], df['learning_rate'], color='blue')
ax2.set_xlabel('Training Steps')
ax2.set_ylabel('Learning Rate')
ax2.set_title('Learning Rate Schedule')
ax2.grid(True)
ax2.set_xlim(0, 2000)
ax2.set_ylim(0, 8e-5)

# グラフの保存
plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
plt.close() 