import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
df = pd.read_csv('models/2gouki/training_progress/training_metrics.csv')

# プロットのスタイル設定
plt.style.use('default')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Training Loss のプロット
ax1.plot(df['step'][df['loss'].notna()], df['loss'][df['loss'].notna()], label='Loss', color='blue')
ax1.set_title('Training Loss')
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Learning Rate のプロット
ax2.plot(df['step'][df['learning_rate'].notna()], 
         df['learning_rate'][df['learning_rate'].notna()], 
         label='LR', color='blue')
ax2.set_title('Learning Rate')
ax2.set_xlabel('Step')
ax2.set_ylabel('Learning Rate')
ax2.legend()
ax2.grid(True)

# Socratic Style Score のプロット
ax3.plot(df['step'][df['socratic_style'].notna()], 
         df['socratic_style'][df['socratic_style'].notna()], 
         label='Socratic Style', color='blue')
ax3.set_title('Socratic Style Score')
ax3.set_xlabel('Step')
ax3.set_ylabel('Score')
ax3.legend()
ax3.grid(True)

# グラフ間のスペースを調整
plt.tight_layout()

# 画像を保存
plt.savefig('models/2gouki/training_progress/training_metrics.png')
plt.close() 