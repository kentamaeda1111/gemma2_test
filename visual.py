import pandas as pd
import matplotlib.pyplot as plt
import re

# データを抽出する関数
def extract_metrics(text):
    data = {
        'step': [],
        'loss': [],
        'learning_rate': [],
        'eval_loss': [],
        'eval_perplexity': [],
        'epoch': []
    }
    
    for line in text.split('\n'):
        try:
            if line.strip().startswith('{') and line.strip().endswith('}'):
                metrics = eval(line.strip())
                
                if 'loss' in metrics and 'eval_loss' not in metrics:
                    data['loss'].append(metrics['loss'])
                    data['learning_rate'].append(metrics.get('learning_rate', None))
                    step = int(metrics.get('step', metrics['epoch'] * 132))
                    data['step'].append(step)
                    data['epoch'].append(metrics['epoch'])
                    data['eval_loss'].append(None)
                    data['eval_perplexity'].append(None)
                
                elif 'eval_loss' in metrics:
                    data['eval_loss'].append(metrics['eval_loss'])
                    data['eval_perplexity'].append(metrics['eval_perplexity'])
                    step = int(metrics.get('step', metrics['epoch'] * 132))
                    data['step'].append(step)
                    data['epoch'].append(metrics['epoch'])
                    data['loss'].append(None)
                    data['learning_rate'].append(None)
        except Exception as e:
            print(f"Error processing line: {e}")
            continue
    
    return pd.DataFrame(data)

# グラフの作成
def plot_training_metrics(df):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training Loss
    ax1.plot(df['step'][df['loss'].notna()], df['loss'].dropna(), 'b-', label='Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Learning Rate
    ax2.plot(df['step'][df['learning_rate'].notna()], df['learning_rate'].dropna(), 'b-', label='LR')
    ax2.set_title('Learning Rate')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Learning Rate')
    ax2.grid(True)
    ax2.legend()
    
    # Evaluation Loss
    ax3.plot(df['step'][df['eval_loss'].notna()], df['eval_loss'].dropna(), 'b-', label='Eval Loss')
    ax3.set_title('Evaluation Loss')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Loss')
    ax3.grid(True)
    ax3.legend()
    
    # Perplexity (対数スケール)
    ax4.plot(df['step'][df['eval_perplexity'].notna()], df['eval_perplexity'].dropna(), 'g-', label='Perplexity')
    ax4.set_title('Evaluation Perplexity')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Perplexity')
    ax4.set_yscale('log')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

# score.txtの内容を使ってデータフレームを作成
with open('score.txt', 'r') as file:
    text = file.read()
    
df = extract_metrics(text)
plot_training_metrics(df)