import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import numpy as np

def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load and prepare the CSV data for analysis"""
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    print("Original columns:", df.columns.tolist())  # デバッグ用
    print("Number of rows:", len(df))  # デバッグ用
    
    # Fill empty model_version with 'base'
    df['model_version'] = df['model_version'].fillna('base')
    df['checkpoint'] = df['checkpoint'].fillna('base')
    
    # Melt the dataframe to get all metrics in one column
    metric_columns = [col for col in df.columns if any(metric in col for metric in ['tone', 'approach', 'format', 'logic'])]
    print("Metric columns found:", metric_columns)  # デバッグ用
    
    if not metric_columns:
        raise ValueError("No metric columns found in the CSV file")
    
    # Print sample of data before melting
    print("\nSample of original data:")
    print(df[['model_version', 'checkpoint'] + metric_columns].head())
    
    df_melted = pd.melt(
        df,
        id_vars=['model_version', 'checkpoint'],
        value_vars=metric_columns,
        var_name='metric_pair',
        value_name='score'
    )
    
    # Print sample of melted data
    print("\nSample of melted data:")
    print(df_melted.head())
    
    # Split metric_pair into metric_type and pair_number
    df_melted[['metric_type', 'pair_num']] = df_melted['metric_pair'].str.extract(r'(\w+)_pair(\d+)')
    
    # Convert score to numeric, coercing errors to NaN
    df_melted['score'] = pd.to_numeric(df_melted['score'], errors='coerce')
    
    # Remove any rows with NaN values
    df_melted = df_melted.dropna()
    
    print("\nAfter processing:")
    print("Number of valid data points:", len(df_melted))  # デバッグ用
    print("Unique model versions:", df_melted['model_version'].unique())  # デバッグ用
    print("Unique metrics:", df_melted['metric_type'].unique())  # デバッグ用
    
    # Print sample of final data
    print("\nSample of final processed data:")
    print(df_melted.head())
    
    if len(df_melted) == 0:
        raise ValueError("No valid data points after processing")
    
    return df_melted

def calculate_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics for each model version, checkpoint, and metric type"""
    summary = df.groupby(['model_version', 'checkpoint', 'metric_type'])['score'].agg([
        'mean',
        'std',
        'min',
        'max',
        'count'
    ]).round(3)
    
    return summary

def plot_metric_comparisons(df: pd.DataFrame, output_dir: str):
    """Create visualization plots for metric comparisons"""
    if df.empty:
        print("Error: No data to plot")
        return
        
    # Set style - using a default matplotlib style
    plt.style.use('default')
    
    # Set the figure size and font sizes
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14
    })
    
    # Plot 1: Box plot comparing metrics across model versions
    plt.figure()
    
    # Prepare data for boxplot
    metrics = sorted(df['metric_type'].unique())
    models = sorted(df['model_version'].unique())
    
    if not metrics or not models:
        print("Error: No metrics or models found in data")
        return
    
    # 重み付けの定義
    metric_weights = {
        'tone': 0.40,      # トーンを最重視
        'logic': 0.25,     # ロジックは中程度
        'approach': 0.25,  # アプローチも中程度
        'format': 0.10     # フォーマットは最小
    }
    
    # 各モデルの最適なチェックポイントを特定
    best_checkpoints = {}
    for model in df[df['model_version'] != 'base']['model_version'].unique():
        model_data = df[df['model_version'] == model]
        checkpoint_scores = {}
        
        for checkpoint in model_data['checkpoint'].unique():
            checkpoint_data = model_data[model_data['checkpoint'] == checkpoint]
            metric_scores = checkpoint_data.groupby('metric_type')['score'].mean()
            
            weighted_score = sum(
                metric_scores[metric] * weight 
                for metric, weight in metric_weights.items()
            )
            checkpoint_scores[checkpoint] = weighted_score
        
        best_checkpoints[model] = max(checkpoint_scores.items(), key=lambda x: x[1])[0]
    
    # Box plotのデータ準備
    positions = []
    data = []
    labels = []
    colors = ['lightblue', 'lightgreen', 'lightpink']
    
    for i, metric in enumerate(metrics):
        for j, model in enumerate(models):
            if model == 'base':
                # baseモデルは全データを使用
                mask = (df['metric_type'] == metric) & (df['model_version'] == model)
            else:
                # ファインチューンモデルは最適なチェックポイントのデータのみ使用
                mask = ((df['metric_type'] == metric) & 
                       (df['model_version'] == model) & 
                       (df['checkpoint'] == best_checkpoints[model]))
            
            scores = df[mask]['score'].dropna().tolist()
            if scores:
                data.append(scores)
                positions.append(i * (len(models) + 1) + j)
                labels.append(f"{model}")
    
    if not data:
        print("Error: No valid data for plotting")
        return
    
    # Create boxplot
    bp = plt.boxplot(data, positions=positions, patch_artist=True)
    
    # Color the boxes
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor=colors[i % len(colors)])
    
    # Set the style of the chart
    plt.xticks([i * (len(models) + 1) + (len(models) - 1) / 2 for i in range(len(metrics))],
               metrics, rotation=45)
    plt.title('Metric Scores Distribution by Model Version\n' +
             '(Fine-tuned models shown at their best checkpoints)', 
             pad=20)
    plt.ylabel('Score')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], label=model)
                      for i, model in enumerate(models)]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add explanation text
    plt.figtext(0.02, 0.02, 
                "Note: For fine-tuned models, only the best checkpoint data is shown.\n"
                "Best checkpoints selected based on weighted average score across metrics.\n"
                "Weights: Tone(40%), Logic(25%), Approach(25%), Format(10%)",
                fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metric_distribution.png', bbox_inches='tight')
    plt.close()
    
    # Plot 2: Line plot showing progression across checkpoints
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))  # 2x2のサブプロット
    axes = axes.flatten()  # 扱いやすいように1次元配列に変換
    
    # Handle checkpoint numbers for non-base models
    df['checkpoint_num'] = df.apply(lambda x: 
        int(x['checkpoint'].replace('checkpoint-', '')) if 'checkpoint-' in str(x['checkpoint'])
        else 0 if x['checkpoint'] == 'base'  # base modelは0として扱う
        else None, axis=1)
    
    # Define colors for each model
    colors = {
        'attention-tuned': '#FF6347',    # 朱色
        'standard-tuned': '#DAA520',  # 黄土色
        'base': '#4169E1'               # 青
    }
    
    metrics = ['approach', 'format', 'logic', 'tone']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Plot for each model
        for model in ['attention-tuned', 'standard-tuned', 'base']:
            model_data = df[df['model_version'] == model]
            if not model_data.empty:
                means = model_data.groupby(['checkpoint_num', 'metric_type'])['score'].mean().unstack()
                if metric in means.columns:
                    if model == 'base':
                        # baseモデルは水平線として表示
                        base_score = means[metric].iloc[0]
                        ax.axhline(y=base_score, color=colors[model], 
                                 linestyle='-', label=f'{model}', linewidth=2)
                    else:
                        # 他のモデルは線と点で表示
                        ax.plot(means.index, means[metric], 
                               marker='o', label=f'{model}',
                               color=colors[model], linewidth=2)
        
        ax.set_title(f'{metric.capitalize()} Score Progression', pad=10)
        ax.set_xlabel('Checkpoint Number (0 = base model)')
        ax.set_ylabel('Average Score')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Y軸の範囲を0-4に設定
        ax.set_ylim(1.5, 4.0)
    
    plt.suptitle('Score Progression Across Checkpoints', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/checkpoint_progression.png',
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_improvement_from_base(df: pd.DataFrame, output_dir: str):
    """Plot improvements of fine-tuned models compared to base model"""
    # より大きなフィギュアサイズを設定し、上部に余裕を持たせる
    plt.figure(figsize=(12, 8))
    
    # サブプロットの位置を調整して上部に余裕を持たせる
    plt.subplots_adjust(top=0.9, bottom=0.2)
    
    # Calculate base model means for each metric
    base_means = df[df['model_version'] == 'base'].groupby('metric_type')['score'].mean()
    
    # 重み付けの定義
    metric_weights = {
        'tone': 0.40,      # トーンを最重視
        'logic': 0.25,     # ロジックは中程度
        'approach': 0.25,  # アプローチも中程度
        'format': 0.10     # フォーマットは最小
    }
    
    # Get the best checkpoint scores for each fine-tuned model
    ft_models = df[df['model_version'] != 'base']['model_version'].unique()
    improvements = []
    
    for model in ft_models:
        model_data = df[df['model_version'] == model]
        
        # 各チェックポイントの重み付き平均スコアを計算
        checkpoint_scores = {}
        for checkpoint in model_data['checkpoint'].unique():
            checkpoint_data = model_data[model_data['checkpoint'] == checkpoint]
            metric_scores = checkpoint_data.groupby('metric_type')['score'].mean()
            
            # 重み付き平均を計算
            weighted_score = sum(
                metric_scores[metric] * weight 
                for metric, weight in metric_weights.items()
            )
            checkpoint_scores[checkpoint] = weighted_score
        
        # 最高の重み付きスコアを持つチェックポイントを特定
        best_checkpoint = max(checkpoint_scores.items(), key=lambda x: x[1])[0]
        
        # 選択されたチェックポイントのスコアを取得
        best_checkpoint_data = model_data[model_data['checkpoint'] == best_checkpoint]
        best_scores = best_checkpoint_data.groupby('metric_type')['score'].mean()
        
        improvement = best_scores - base_means
        improvements.append((model, improvement, best_checkpoint))
    
    # Plot
    x = np.arange(len(base_means.index))
    width = 0.35
    
    for i, (model, improvement, best_checkpoint) in enumerate(improvements):
        plt.bar(x + i*width, improvement, width, label=f"{model}\n(Best: {best_checkpoint})",
               color=['lightblue', 'lightgreen'][i])
        
        # Add value labels on bars
        for j, v in enumerate(improvement):
            plt.text(x[j] + i*width, v + (0.1 if v >= 0 else -0.1),
                    f'{v:+.2f}',
                    ha='center', va='bottom' if v >= 0 else 'top')
    
    # グラフの要素を配置
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Metrics', labelpad=10)  # labelpadで軸ラベルの位置を調整
    plt.ylabel('Improvement from Base Model\n(Score Difference)', labelpad=10)  # 改行を追加して2行に
    plt.title('Best Improvement in Socratic Elements from Base Model', pad=20)
    plt.xticks(x + width/2, base_means.index)
    plt.legend(bbox_to_anchor=(1.02, 1))  # 凡例の位置を調整
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Add baseline scores as text（位置を調整）
    plt.text(-0.2, -1.2, f'Base Model Scores:', 
            fontsize=10, color='gray', ha='left')
    for i, (metric, score) in enumerate(base_means.items()):
        plt.text(i-0.2, -1.4, f'{metric}: {score:.2f}', 
                fontsize=9, color='gray', ha='left')
    
    # Add explanation text（重み付けの説明を追加）
    plt.figtext(0.02, 0.02, 
                "Note: Improvements shown are from the best performing checkpoint\n"
                "for each model, selected based on weighted average score across metrics.\n"
                "Weights: Tone(40%), Logic(25%), Approach(25%), Format(10%)",
                fontsize=8, style='italic')
    
    # Y軸の範囲を明示的に設定
    plt.ylim(-1.5, 1.5)  # ベーススコアのテキストが見えるように下限を調整
    
    plt.tight_layout()  # レイアウトを自動調整
    plt.savefig(f'{output_dir}/improvement_from_base.png', 
                bbox_inches='tight',  # 余白を適切に調整
                dpi=300)
    plt.close()

def analyze_quality_results(csv_path: str, output_dir: str):
    """Main function to analyze quality check results"""
    # Load and prepare data
    df = load_and_prepare_data(csv_path)
    
    # Calculate summary statistics
    summary_stats = calculate_summary_stats(df)
    
    # Save summary statistics to CSV
    summary_stats.to_csv(f'{output_dir}/summary_statistics.csv')
    
    # Create visualization plots
    plot_metric_comparisons(df, output_dir)
    plot_improvement_from_base(df, output_dir)
    
    # Print overall findings
    print("\nOverall Analysis Results:")
    print("-" * 50)
    
    # Compare model versions
    for metric in df['metric_type'].unique():
        print(f"\n{metric.upper()} Metric Summary:")
        for model in df['model_version'].unique():
            model_metric_mean = df[
                (df['model_version'] == model) & 
                (df['metric_type'] == metric)
            ]['score'].mean()
            print(f"{model}: {model_metric_mean:.3f}")

    # Print detailed summary
    print("\nDetailed Summary Statistics:")
    print("=" * 80)
    print(summary_stats)
    
    print("\nKey Findings:")
    print("-" * 80)
    print("1. Best performing metrics by model:")
    for model in df['model_version'].unique():
        model_means = df[df['model_version'] == model].groupby('metric_type')['score'].mean()
        best_metric = model_means.idxmax()
        print(f"   {model}: {best_metric} ({model_means[best_metric]:.3f})")

    # Add stability analysis
    print("\n2. Stability Analysis (Standard Deviation):")
    for model in df['model_version'].unique():
        print(f"\n{model}:")
        model_data = df[df['model_version'] == model]
        for metric in df['metric_type'].unique():
            metric_std = model_data[model_data['metric_type'] == metric]['score'].std()
            print(f"   {metric}: {metric_std:.3f}")
    
    # Add improvement analysis
    print("\n3. Improvement Analysis (First to Last Checkpoint):")
    for model in df['model_version'].unique():
        print(f"\n{model}:")
        for metric in df['metric_type'].unique():
            first_checkpoint = df[df['checkpoint'] == 'checkpoint-100']
            last_checkpoint = df[df['checkpoint'] == 'checkpoint-990']
            
            first_score = first_checkpoint[
                (first_checkpoint['model_version'] == model) & 
                (first_checkpoint['metric_type'] == metric)
            ]['score'].mean()
            
            last_score = last_checkpoint[
                (last_checkpoint['model_version'] == model) & 
                (last_checkpoint['metric_type'] == metric)
            ]['score'].mean()
            
            improvement = last_score - first_score
            print(f"   {metric}: {improvement:+.3f}")

    # Add improvement analysis from base model
    print("\n4. Improvement Analysis from Base Model:")
    base_scores = df[df['model_version'] == 'base'].groupby('metric_type')['score'].mean()
    
    for model in df[df['model_version'] != 'base']['model_version'].unique():
        print(f"\n{model}:")
        model_scores = df[df['model_version'] == model].groupby('metric_type')['score'].mean()
        for metric in base_scores.index:
            improvement = model_scores[metric] - base_scores[metric]
            print(f"   {metric}: {improvement:+.3f} ({base_scores[metric]:.2f} → {model_scores[metric]:.2f})")

def main():
    csv_path = "data/config/automation_gemma.csv"
    output_dir = "data/analysis"
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    analyze_quality_results(csv_path, output_dir)

if __name__ == "__main__":
    main() 