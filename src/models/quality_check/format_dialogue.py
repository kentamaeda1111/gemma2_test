import json
from typing import Dict, List

def format_dialogue(input_file: str, output_file: str = None) -> Dict:
    """
    対話データを新しい形式に変換する関数
    
    Args:
        input_file: 入力JSONファイルのパス
        output_file: 出力JSONファイルのパス（省略可能）
    
    Returns:
        変換後のデータ（dict）
    """
    # 入力ファイルを読み込む
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 新しい形式のデータを構築
    formatted_data = {
        "metadata": {
            "question_id": data["question_id"],
            "timestamp": data["timestamp"],
            "model_version": data["model_version"],
            "checkpoint": data["checkpoint"],
            "topic": data["history"][0]["content"]
        },
        "pairs": []
    }
    
    # 対話ペアを構築
    history = data["history"][1:]  # 最初のGemmaの発言（topic）を除く
    for i in range(0, len(history), 2):
        if i + 1 >= len(history):
            break
            
        pair = {
            "pair_id": (i // 2) + 1,
            "claude": {
                "content": history[i]["content"]
            },
            "gemma": {
                "content": history[i + 1]["content"]
            },
            "evaluation": {
                "tone": {"quantitative": "", "qualitative": ""},
                "grammar": {"quantitative": "", "qualitative": ""},
                "format": {"quantitative": "", "qualitative": ""},
                "logic": {"quantitative": "", "qualitative": ""}
            }
        }
        formatted_data["pairs"].append(pair)
    
    # 出力ファイルが指定されている場合は保存
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
    
    return formatted_data

# 使用例
if __name__ == "__main__":
    input_path = "data/dialogue/raw_gemma/dialogue_76_20250212_100723.json"
    output_path = "data/dialogue/raw_gemma/dialogue_76_20250212_100723.json"
    formatted = format_dialogue(input_path, output_path) 