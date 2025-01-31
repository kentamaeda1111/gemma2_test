import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_metrics(csv_path):
    # データの読み込み
    df = pd.read_csv(csv_path)
    
    # 出力先ディレクトリの設定（csvファイルと同じディレクトリ）
    output_dir = os.path.dirname(csv_path)
    output_path = os.path.join(output_dir, 'training_progress.png')
    
    # 欠損値を含む行を削除せずに、各メトリクスごとに有効なデータポイントを使用
    df_loss = df[df['loss'].notna()]
    df_lr = df[df['learning_rate'].notna()]
    
    # フィギュアとサブプロットの作成
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training Loss
    ax1.plot(df_loss['step'], df_loss['loss'], label='Loss', color='blue')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Learning Rate
    ax2.plot(df_lr['step'], df_lr['learning_rate'], label='LR', color='blue')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate')
    ax2.legend()
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax2.grid(True)
    
    # Evaluation Metrics
    valid_metrics = df[df['style_consistency'].notna() & df['dialogue_flow'].notna()]
    ax3.plot(valid_metrics['step'], valid_metrics['style_consistency'], 
             label='Style Consistency', color='blue')
    ax3.plot(valid_metrics['step'], valid_metrics['dialogue_flow'], 
             label='Dialogue Flow', color='orange')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Score')
    ax3.set_title('Evaluation Metrics')
    ax3.legend()
    ax3.grid(True)
    
    # Combined Score
    valid_combined = df[df['combined_score'].notna()]
    ax4.plot(valid_combined['step'], valid_combined['combined_score'], 
             label='Combined Score', color='blue')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Score')
    ax4.set_title('Combined Evaluation Score')
    ax4.legend()
    ax4.grid(True)
    
    # 共通の設定
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(0, 2000)  # x軸の範囲を0-2000に設定
    
    # レイアウトの調整
    plt.tight_layout()
    
    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_training_metrics('models/model9/training_progress_model/training_metrics_model.csv') 