import json
from transformers import AutoTokenizer
import logging
from pathlib import Path
from src.utils.config import get_api_keys

# グローバル設定
TARGET_FILE = "kaggle_model"  # 処理対象のファイル名（拡張子なし）
DIALOGUE_DIR = "data/dialogue/processed"  # 対話データのディレクトリ
MAX_TOKENS = 256  # トークン数の上限

def process_dialogue_file(target_file=TARGET_FILE, max_tokens=MAX_TOKENS):
    """
    対話データのトークン長をチェックし、制限を超えるペアを除外した新しいファイルを作成します。
    制限を超えるペアが見つからない場合は、新しいファイルは作成しません。
    
    Args:
        target_file (str): 処理対象のファイル名（拡張子なし）
        max_tokens (int): トークン数の制限（デフォルト: グローバル設定値）
    
    Returns:
        str or None: 作成された新しいファイルのパス。ファイルが作成されなかった場合はNone
    """
    # 入力ファイルパスの構築
    json_path = Path(DIALOGUE_DIR) / f"{target_file}.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    # API keyの取得
    try:
        api_keys = get_api_keys()
        huggingface_token = api_keys['huggingface_api_key']
    except Exception as e:
        logging.error(f"Failed to get API keys: {str(e)}")
        raise

    # トークナイザーの準備
    model_name = "google/gemma-2-2b-jpn-it"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=huggingface_token
    )

    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            dialogues = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON file: {str(e)}")
        return

    valid_dialogues = []
    over_limit_pairs = []
    total_pairs = 0

    for dialogue in dialogues:
        messages = dialogue.get('messages', [])
        if len(messages) != 2:  # user-modelのペアのみを処理
            continue

        total_pairs += 1
        user_msg = messages[0]['content']
        model_msg = messages[1]['content']

        # ペアを結合して会話形式に
        conversation = [
            {"role": "user", "content": user_msg},
            {"role": "model", "content": model_msg}
        ]

        # Gemmaの会話テンプレートを適用
        formatted_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # トークン数をカウント
        tokens = tokenizer.encode(formatted_text)
        token_length = len(tokens)

        if token_length > max_tokens:
            over_limit_pairs.append({
                'source_file': dialogue.get('source_file', 'unknown'),
                'extract_range': dialogue.get('extract_range', 'unknown'),
                'token_length': token_length,
                'user_msg': user_msg[:100] + '...' if len(user_msg) > 100 else user_msg,
                'model_msg': model_msg[:100] + '...' if len(model_msg) > 100 else model_msg
            })
        else:
            valid_dialogues.append(dialogue)

    # 結果の出力
    logging.info(f"\nAnalysis Results for {json_path.name}:")
    logging.info(f"Total dialogue pairs analyzed: {total_pairs}")
    logging.info(f"Pairs exceeding {max_tokens} tokens: {len(over_limit_pairs)}")
    
    if not over_limit_pairs:
        logging.info("\nNo dialogues exceeding token limit found. No new file will be created.")
        return None
    
    if over_limit_pairs:
        logging.info("\nDetailed report of pairs exceeding token limit:")
        for i, pair in enumerate(over_limit_pairs, 1):
            logging.info(f"\n{i}. Token length: {pair['token_length']}")
            logging.info(f"Source: {pair['source_file']}, Range: {pair['extract_range']}")
            logging.info(f"User message preview: {pair['user_msg']}")
            logging.info(f"Model message preview: {pair['model_msg']}")

        # 新しいファイルの作成
        output_path = json_path.parent / f"{target_file}_processed.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(valid_dialogues, f, ensure_ascii=False, indent=2)
            logging.info(f"\nProcessed file saved as: {output_path}")
            logging.info(f"Valid dialogues saved: {len(valid_dialogues)}")
            return str(output_path)
        except Exception as e:
            logging.error(f"Failed to save processed file: {str(e)}")
            raise

    return None

if __name__ == "__main__":
    result = process_dialogue_file()
    if result:
        logging.info(f"Processing completed. New file created: {result}")
    else:
        logging.info("Processing completed. No new file was needed.") 