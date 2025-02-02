import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_path):
    # データを格納する辞書
    data = {
        'step': [],
        'loss': [],
        'learning_rate': [],
        'style_consistency': [],
        'dialogue_flow': [],
        'combined_score': []
    }
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Training lossの抽出
            if "{'loss':" in line:
                match = re.search(r"'loss': ([\d.]+)", line)
                if match:
                    step = int(re.search(r"Step (\d+)", line).group(1))
                    data['step'].append(step)
                    data['loss'].append(float(match.group(1)))
            
            # Learning rateの抽出
            if "'learning_rate':" in line:
                match = re.search(r"'learning_rate': ([\d.e-]+)", line)
                if match:
                    data['learning_rate'].append(float(match.group(1)))
            
            # 評価メトリクスの抽出
            if "'eval_style_consistency':" in line:
                matches = {
                    'style': re.search(r"'eval_style_consistency': ([\d.]+)", line),
                    'flow': re.search(r"'eval_dialogue_flow': ([\d.]+)", line),
                    'combined': re.search(r"'eval_combined_score': ([\d.]+)", line)
                }
                if all(matches.values()):
                    data['style_consistency'].append(float(matches['style'].group(1)))
                    data['dialogue_flow'].append(float(matches['flow'].group(1)))
                    data['combined_score'].append(float(matches['combined'].group(1)))

    return data

def plot_training_progress(data):
    # フォントサイズの設定
    plt.rcParams.update({'font.size': 10})
    
    # サブプロットの作成
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training Progress', fontsize=12)
    
    # Training Loss
    ax1.plot(data['step'], data['loss'], 'b-', label='Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Learning Rate
    ax2.plot(data['step'][:len(data['learning_rate'])], data['learning_rate'], 'b-', label='LR')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate')
    ax2.grid(True)
    ax2.legend()
    
    # Evaluation Metrics
    eval_steps = np.linspace(0, max(data['step']), len(data['style_consistency']))
    ax3.plot(eval_steps, data['style_consistency'], 'b-', label='Style Consistency')
    ax3.plot(eval_steps, data['dialogue_flow'], 'orange', label='Dialogue Flow')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Score')
    ax3.set_title('Evaluation Metrics')
    ax3.grid(True)
    ax3.legend()
    
    # Combined Score
    ax4.plot(eval_steps, data['combined_score'], 'b-', label='Combined Score')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Score')
    ax4.set_title('Combined Evaluation Score')
    ax4.grid(True)
    ax4.legend()
    
    # レイアウトの調整
    plt.tight_layout()
    
    return fig

def main():
    # ログファイルのパス
    log_path = 'others/★model!!/kaggle_model_ver2/logs/training_log_20250127_000954.log'
    
    # データの解析
    data = parse_log_file(log_path)
    
    # グラフの作成
    fig = plot_training_progress(data)
    
    # グラフの保存
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main() 