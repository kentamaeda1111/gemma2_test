import json
from transformers import AutoTokenizer
import os
import logging
from tqdm import tqdm

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_dialogue_stats():
    # 対話ファイルのパス
    DIALOGUE_PATH = "data/dialogue/processed/kaggle_model.json"
    
    # Gemma tokenizer の読み込み
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2b-it",
        trust_remote_code=True
    )
    
    try:
        # JSONファイルの読み込み
        with open(DIALOGUE_PATH, 'r', encoding='utf-8') as f:
            dialogues = json.load(f)
        
        total_dialogues = len(dialogues)
        long_dialogues = 0
        token_lengths = []
        
        # 各対話のトークン数をチェック
        for dialogue in tqdm(dialogues, desc="Checking dialogues"):
            messages = dialogue.get('messages', [])
            
            # user-model のペアをチェック
            if len(messages) >= 2 and messages[0]['role'] == 'user' and messages[1]['role'] == 'model':
                # 対話をGemmaのフォーマットに変換
                formatted_text = tokenizer.apply_chat_template(
                    messages[:2],  # 最初の2つのメッセージのみ
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # トークン数をカウント
                tokens = tokenizer.encode(formatted_text)
                token_count = len(tokens)
                token_lengths.append(token_count)
                
                if token_count > 256:
                    long_dialogues += 1
        
        # 統計情報の表示
        logging.info(f"Total number of dialogues: {total_dialogues}")
        logging.info(f"Number of dialogues over 256 tokens: {long_dialogues}")
        logging.info(f"Percentage of long dialogues: {(long_dialogues/total_dialogues)*100:.2f}%")
        
        # トークン長の分布
        if token_lengths:
            logging.info(f"Average token length: {sum(token_lengths)/len(token_lengths):.2f}")
            logging.info(f"Max token length: {max(token_lengths)}")
            logging.info(f"Min token length: {min(token_lengths)}")
            
            # トークン長の分布を表示
            ranges = [(0, 64), (65, 128), (129, 256), (257, 512), (513, float('inf'))]
            for start, end in ranges:
                count = sum(1 for length in token_lengths if start <= length <= end)
                logging.info(f"Dialogues with {start}-{end} tokens: {count} ({(count/total_dialogues)*100:.2f}%)")
                
    except Exception as e:
        logging.error(f"Error processing dialogue file: {str(e)}")
        raise

if __name__ == "__main__":
    check_dialogue_stats() 