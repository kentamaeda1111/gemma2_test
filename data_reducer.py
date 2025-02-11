import json
import sys
import math

def reduce_data(input_file, output_file, percentage):
    # 入力ファイルを読み込む
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 指定された割合に基づいて新しいデータ数を計算
    original_length = len(data)
    new_length = math.floor(original_length * (percentage / 100))
    
    # データを指定された量まで減らす
    reduced_data = data[:new_length]
    
    # 結果を新しいファイルに保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(reduced_data, f, ensure_ascii=False, indent=2)
    
    print(f"元のデータ数: {original_length}")
    print(f"削減後のデータ数: {new_length}")
    print(f"削減率: {percentage}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python data_reducer.py <削減率>")
        print("例: python data_reducer.py 50")
        sys.exit(1)
    
    try:
        percentage = float(sys.argv[1])
        if percentage <= 0 or percentage > 100:
            raise ValueError
    except ValueError:
        print("エラー: 削減率は0より大きく100以下の数値を指定してください")
        sys.exit(1)
    
    input_file = "data/dialogue/processed/kaggle_model.json"
    output_file = f"data/dialogue/processed/systemprompt_no_{int(percentage)}.json"
    
    reduce_data(input_file, output_file, percentage) 