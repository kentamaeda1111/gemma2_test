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
    
    # model_version と checkpoint は既にCSVに含まれているので抽出は不要
    # df[['model_version', 'checkpoint']] = df['dialogue'].str.extract(r'(train_[yn]am)_(\d+)') を削除
    
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
    
    positions = []
    data = []
    labels = []
    colors = ['lightblue', 'lightgreen']
    
    for i, metric in enumerate(metrics):
        for j, model in enumerate(models):
            mask = (df['metric_type'] == metric) & (df['model_version'] == model)
            scores = df[mask]['score'].dropna().tolist()
            if scores:  # Only add if there are valid scores
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
    plt.title('Metric Scores Distribution by Model Version')
    plt.ylabel('Score')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], label=model)
                      for i, model in enumerate(models)]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metric_distribution.png', bbox_inches='tight')
    plt.close()
    
    # Plot 2: Line plot showing progression across checkpoints
    plt.figure(figsize=(15, 8))  # グラフサイズを大きくする
    
    # checkpointから'checkpoint-'を削除して数値のみにする
    df['checkpoint_num'] = df['checkpoint'].str.extract(r'checkpoint-(\d+)').astype(int)
    
    for model in models:
        model_data = df[df['model_version'] == model]
        if not model_data.empty:
            means = model_data.groupby(['checkpoint_num', 'metric_type'])['score'].mean().unstack()
            if not means.empty:
                for metric in means.columns:
                    plt.plot(means.index, means[metric], marker='o', 
                            label=f'{model}-{metric}', linewidth=2)
    
    plt.title('Score Progression Across Checkpoints', pad=20)
    plt.xlabel('Checkpoint Number')
    plt.ylabel('Average Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              borderaxespad=0., frameon=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/checkpoint_progression.png', 
                bbox_inches='tight', dpi=300)
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

def main():
    csv_path = "data/config/automation_gemma.csv"
    output_dir = "data/analysis/quality_check"
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    analyze_quality_results(csv_path, output_dir)

if __name__ == "__main__":
    main() 