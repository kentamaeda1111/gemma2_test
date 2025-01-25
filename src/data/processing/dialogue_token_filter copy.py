import json
from transformers import AutoTokenizer
import os
import logging
from tqdm import tqdm
from pathlib import Path

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def filter_long_dialogues():
    # パスの設定
    input_path = "data/dialogue/processed/kaggle_model.json"
    output_path = "data/dialogue/processed/kaggle_model_processed.json"
    
    # Gemma tokenizer の読み込み
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2b-it",
        trust_remote_code=True
    )
    
    try:
        # 入力JSONファイルの読み込み
        with open(input_path, 'r', encoding='utf-8') as f:
            dialogues = json.load(f)
        
        # フィルタリング済みの対話を格納するリスト
        filtered_dialogues = []
        skipped_count = 0
        
        # 各対話をチェック
        for dialogue in tqdm(dialogues, desc="Filtering dialogues"):
            messages = dialogue.get('messages', [])
            
            if len(messages) >= 2 and messages[0]['role'] == 'user' and messages[1]['role'] == 'model':
                # 対話をGemmaのフォーマットに変換
                formatted_text = tokenizer.apply_chat_template(
                    messages[:2],
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # トークン数をカウント
                tokens = tokenizer.encode(formatted_text)
                if len(tokens) <= 256:
                    filtered_dialogues.append(dialogue)
                else:
                    skipped_count += 1
        
        # 出力ディレクトリの確認と作成
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # フィルタリング済みの対話を新しいJSONファイルに保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_dialogues, f, ensure_ascii=False, indent=2)
        
        # 結果の表示
        logging.info(f"Original dialogues: {len(dialogues)}")
        logging.info(f"Filtered dialogues: {len(filtered_dialogues)}")
        logging.info(f"Removed dialogues: {skipped_count}")
        logging.info(f"Filtered file saved to: {output_path}")
        
    except Exception as e:
        logging.error(f"Error processing dialogue file: {str(e)}")
        raise

if __name__ == "__main__":
    filter_long_dialogues() 