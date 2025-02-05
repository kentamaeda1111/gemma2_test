# https://claude.ai/chat/a8e367d4-7896-474f-b1fb-f3e40e900341これをもとに修正

# もともとは2goukiのやつ
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
import logging
from datetime import datetime
import os
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.config import get_api_keys

# グローバル設定
DIALOGUE_JSON_PATH = "data/dialogue/processed/kaggle_model.json"  # 対話データのJSONファイルパス
MAX_SEQUENCE_LENGTH = 256  # 1つの対話の最大トークン数
MAX_TOKENIZE_LENGTH = 256  # トークナイズ時の最大トークン数

# グローバル設定の前に追加
try:
    # APIキーを取得
    api_keys = get_api_keys()
    huggingface_token = api_keys['huggingface_api_key']
    
    # Hugging Face APIキーを設定
    os.environ["HUGGINGFACE_TOKEN"] = huggingface_token
    
    logging.info("Successfully loaded Hugging Face API key")
except Exception as e:
    logging.error(f"Error loading API keys: {str(e)}")
    raise

# 設定のログ出力
logging.info(f"Using dialogue file: {DIALOGUE_JSON_PATH}")
logging.info(f"Max sequence length: {MAX_SEQUENCE_LENGTH}")
logging.info(f"Max tokenize length: {MAX_TOKENIZE_LENGTH}")

# 環境変数とwarningの設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore", category=FutureWarning)

def validate_message_format(message):
    """メッセージのフォーマットを検証"""
    if not isinstance(message, dict):
        return False
    if 'role' not in message or 'content' not in message:
        return False
    if message['role'] not in ['user', 'model']:
        return False
    if not isinstance(message['content'], str):
        return False
    return True

def prepare_dataset():
    conversations = []
    
    try:
        with open(DIALOGUE_JSON_PATH, 'r', encoding='utf-8') as f:
            dialogue_data = json.load(f)
            
        for dialogue in dialogue_data:
            messages = dialogue.get('messages', [])
            
            # メッセージのフォーマットを検証
            if not all(validate_message_format(msg) for msg in messages):
                logging.warning(f"Skipped dialogue due to invalid message format")
                continue
                
            # user->modelの順序を確認しながら会話を構築
            current_conversation = []
            valid_sequence = True
            
            for i in range(0, len(messages)-1, 2):
                if (i+1 < len(messages) and 
                    messages[i]['role'] == 'user' and 
                    messages[i+1]['role'] == 'model'):
                    current_conversation.extend([messages[i], messages[i+1]])
                else:
                    valid_sequence = False
                    break
            
            # 有効な会話のみを追加
            if valid_sequence and current_conversation:
                # Gemmaのチャットテンプレートを適用
                formatted_text = tokenizer.apply_chat_template(
                    current_conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # トークン数をチェック
                tokens = tokenizer.encode(formatted_text)
                if len(tokens) <= MAX_SEQUENCE_LENGTH:
                    conversations.append({"text": formatted_text})
                else:
                    logging.warning(f"Skipped conversation due to length: {len(tokens)} tokens")
            
    except Exception as e:
        logging.error(f"Error processing dialogue file: {str(e)}")
        raise
    
    if not conversations:
        raise ValueError("No valid conversations found in the dialogue file")
        
    logging.info(f"Processed {len(conversations)} valid conversations")
    return Dataset.from_list(conversations)

# モデルとトークナイザーの準備
model_name = "google/gemma-2-2b-jpn-it"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    token=huggingface_token  # APIトークンを追加
)

# BitsAndBytesConfigの設定をさらに最適化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_storage=torch.uint8,
)

# モデルの読み込みを修正
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation='eager',
    token=huggingface_token  # APIトークンを追加
)

# モデルをLoRA用に準備した後にキャッシュを無効化
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

# claude 修正1（前）
# LoRAの設定を調整
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# claude 修正1（後）
# lora_config = LoraConfig(
#     r=32,  # より大きな rank
#     lora_alpha=64,
#     # 出力に近い層を重点的に
#     target_modules=["o_proj"],  # 出力投影のみを対象
#     lora_dropout=0.05,  # より小さなドロップアウト
#     bias="none",
#     task_type="CAUSAL_LM",
# )

# LoRAモデルの作成
model = get_peft_model(model, lora_config)

# メモリ効率のための設定
model.config.use_cache = False

# データセットの準備
dataset = prepare_dataset()

# データセットの構造を確認
print("Dataset structure:")
print(dataset[0])  # 最初の要素を表示
print("\nDataset features:")
print(dataset.features)

# データセットのバッチ処理を最適化
dataset = dataset.select(range(len(dataset))).shuffle(seed=42)

# トークナイズ関数の修正
def tokenize_function(examples):
    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_TOKENIZE_LENGTH,      # グローバル設定を使用
        padding='max_length',
        add_special_tokens=True,
        return_tensors=None,
    )
    return result

# データセットの処理を最適化
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=32,  # 64から減少
    num_proc=4,     # 2から増加
    load_from_cache_file=True,
    desc="Tokenizing datasets",
    remove_columns=dataset.column_names,
)

# メモリ使用量を監視するログを追加
def log_memory_usage():
    import psutil
    process = psutil.Process()
    logging.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# データセットのサイズをログに記録
logging.info(f"Total dataset size: {len(dataset)}")
log_memory_usage()

# データセットの検証を追加
def validate_dataset(dataset):
    # 最初の要素をチェック
    first_item = dataset[0]
    print("Validated first item structure:")
    print(f"Keys: {first_item.keys()}")
    print(f"input_ids type: {type(first_item['input_ids'])}")
    print(f"input_ids length: {len(first_item['input_ids'])}")
    return dataset

tokenized_dataset = validate_dataset(tokenized_dataset)

# claude 修正2(前)
# # データセットの前処理を追加
# def preprocess_function(examples):
#     # パターン定義
#     end_patterns = [
#         "だろうか", "ではないか", "のではないか", "かね",
#         "なるほど", "興味深い", "考えてみよう"
#     ]
    
#     # 接続詞パターン
#     conjunctions = [
#         "しかし", "だから", "それでは", "すなわち",
#         "たとえば", "つまり", "ならば", "もし"
#     ]
    
#     # トークン化されたテキストを取得
#     texts = tokenizer.batch_decode(examples['input_ids'])
#     new_attention_masks = []
    
#     for text, mask in zip(texts, examples['attention_mask']):
#         if not isinstance(mask, list):
#             mask = mask.tolist()
        
#         # 新しいattention maskを作成（ベースは0.8）
#         new_mask = [0.8] * len(mask)
        
#         # 文を分割
#         sentences = text.split('。')
#         current_pos = 0
        
#         for sentence in sentences:
#             if not sentence.strip():
#                 continue
                
#             # 文末パターンの検出と強調
#             for pattern in end_patterns:
#                 if pattern in sentence:
#                     # パターンの位置を特定
#                     pattern_tokens = tokenizer.encode(pattern, add_special_tokens=False)
#                     pattern_len = len(pattern_tokens)
                    
#                     # パターンを含むトークンとその前後を強調
#                     pattern_start = current_pos + len(tokenizer.encode(sentence, add_special_tokens=False)) - pattern_len
#                     for i in range(max(0, pattern_start - 2), min(len(mask), pattern_start + pattern_len + 2)):
#                         new_mask[i] = 1.0  # パターン部分は最大の注意を向ける
            
#             # 接続詞の検出と強調
#             for conj in conjunctions:
#                 if conj in sentence:
#                     # 接続詞の位置を特定
#                     conj_tokens = tokenizer.encode(conj, add_special_tokens=False)
#                     conj_pos = len(tokenizer.encode(sentence.split(conj)[0], add_special_tokens=False))
                    
#                     # 接続詞の前後を強調（やや弱め）
#                     for i in range(max(0, current_pos + conj_pos - 1), 
#                                  min(len(mask), current_pos + conj_pos + len(conj_tokens) + 1)):
#                         new_mask[i] = 0.9
            
#             # 句読点の強調
#             for i, char in enumerate(sentence):
#                 if char in '、。！？':
#                     # 句読点の位置を特定
#                     punct_pos = len(tokenizer.encode(sentence[:i], add_special_tokens=False))
#                     # 句読点前後のトークンを強調
#                     for j in range(max(0, current_pos + punct_pos - 1),
#                                  min(len(mask), current_pos + punct_pos + 2)):
#                         new_mask[j] = 0.95
            
#             # 文の区切りごとの位置を更新
#             current_pos += len(tokenizer.encode(sentence + '。', add_special_tokens=False))
        
#         # 特殊トークンのマスクは1.0に設定
#         if tokenizer.bos_token_id is not None:
#             new_mask[0] = 1.0  # BOS token
#         if tokenizer.eos_token_id is not None:
#             new_mask[-1] = 1.0  # EOS token
            
#         new_attention_masks.append(new_mask)

#     examples['attention_mask'] = new_attention_masks
#     return examples

# claude 修正2（後）
def preprocess_function(examples):
    # ソクラテス的な口調と問いのパターンに絞る
    socratic_patterns = [
        # 問いかけのパターン
        "かね", "だろうか", "のかね", "ではないかね",
        # 問いの導入
        "では", "について",
        # 二人称（老練な口調の特徴）
        "君は", "君が", "君の"
    ]
    
    # トークン化されたテキストを取得
    texts = tokenizer.batch_decode(examples['input_ids'])
    new_attention_masks = []
    
    for text, mask in zip(texts, examples['attention_mask']):
        if not isinstance(mask, list):
            mask = mask.tolist()

        # claude 修正3(前)
        # # ベースのattention maskを作成
        # new_mask = [0.8] * len(mask)

        # claude 修正3(後)
        # ベースのattention maskをより自然な形に
        new_mask = mask.copy()  # 元のマスクを維持
        
        # 文を分割
        sentences = text.split('。')
        current_pos = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # ソクラテス的パターンの検出と強調
            for pattern in socratic_patterns:
                if pattern in sentence:
                    # パターンの位置を特定
                    pattern_tokens = tokenizer.encode(pattern, add_special_tokens=False)
                    pattern_len = len(pattern_tokens)
                    
                    # パターンを含むトークンとその前後を強調
                    pattern_start = current_pos + len(tokenizer.encode(sentence, add_special_tokens=False)) - pattern_len
                    for i in range(max(0, pattern_start - 2), min(len(mask), pattern_start + pattern_len + 2)):
                        new_mask[i] = 1.0  # パターン部分は最大の注意を向ける
            
            # 文の区切りごとの位置を更新
            current_pos += len(tokenizer.encode(sentence + '。', add_special_tokens=False))
        
        # 特殊トークンのマスクは1.0に設定
        if tokenizer.bos_token_id is not None:
            new_mask[0] = 1.0  # BOS token
        if tokenizer.eos_token_id is not None:
            new_mask[-1] = 1.0  # EOS token
            
        new_attention_masks.append(new_mask)

    examples['attention_mask'] = new_attention_masks
    return examples

tokenized_dataset = tokenized_dataset.map(
    preprocess_function,
    batched=True,
    desc="Applying attention masking"
)


# claude 修正4（消すか消さないか）
# # トークナイザーの設定を追加
# tokenizer.add_special_tokens({
#     'additional_special_tokens': [
#         '。', '、', '！', '？',  # 句読点
#     ]
# })

# ロギングの設定を更新
import os

# ログディレクトリを作成
log_dir = "model/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# 学習の設定を更新
training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=30,     # 「For smaller datasets: Use larger batch sizes, higher epoch counts」から30エポックに設定
    learning_rate=8e-5,      # 1e-4から若干低下
    weight_decay=0.06,       # わずかに増加
    warmup_ratio=0.25,       # より長いウォームアップ
    lr_scheduler_type="cosine_with_restarts",  # リスタートありのスケジューラに変更
    evaluation_strategy="steps",
    eval_steps=20,          # 25から20に変更してより頻繁に評価
    save_strategy="steps",
    save_steps=20,
    gradient_accumulation_steps=8,   # 累積回数を減らす
    max_steps=-1,
    disable_tqdm=False,
    logging_dir="./model/logs",
    logging_strategy="steps",
    logging_steps=50,
    no_cuda=False,
    dataloader_num_workers=2,
    report_to=[],
    run_name=None,
    per_device_train_batch_size=4,  # メモリが許す場合は増やす
    per_device_eval_batch_size=2,   # トレーニングバッチサイズの半分を評価用に設定
    gradient_checkpointing=True,
    max_grad_norm=0.5,       # 「Gradient clipping: Values between 0.5-1.0 help prevent divergence」から0.5を選択
    dataloader_pin_memory=True,
    save_total_limit=3,
    fp16=True,
    optim="adamw_torch_fused",
    eval_accumulation_steps=8,
    load_best_model_at_end=True,
    metric_for_best_model="socratic_style",  # 新しい評価指標を使用
)

# 環境変数でwandbを無効化（training_argsの前に追加）
import os
os.environ["WANDB_DISABLED"] = "true"

# データコレーターの修正
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# claude修正案5(前)
# # 評価メトリクスの修正
# def compute_metrics(eval_preds):
#     logits, labels = eval_preds  # eval_predsから logits と labels を取得
    
#     # 評価用データセットのサイズ制限を緩和
#     max_samples = 100
    
#     # デコード処理の改善
#     with torch.no_grad():
#         logits = torch.tensor(logits).cpu()
#         predictions = torch.argmax(logits, dim=-1)
        
#         # バッチ全体をデコード
#         decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
#         # より詳細なログ出力を追加
#         logging.info(f"Sample prediction: {decoded_preds[0][:100]}...")
        
#         del logits, predictions  # メモリ解放
#         torch.cuda.empty_cache()
#         # 文末パターンをより柔軟に定義
#         sentence_end_patterns = {
#             'question_patterns': [
#                 'かね', 'だろうか', 'ではないか',
#                 'のか', 'と思わないか', '考えてみよう',
#             ],
#             'statement_patterns': [
#                 'だね', 'なるほど', '興味深い',
#                 'といえよう', 'というべきだ'
#             ],
#             'reflection_patterns': [
#                 'かもしれない', 'のではないか',
#                 'と考えられる', 'といえそうだ'
#             ]
#         }
        
#         # 助動詞パターン
#         auxiliary_patterns = [
#             'である', 'だ', 'です', 'ます',
#             'のだ', 'のです', 'のである'
#         ]
        
#         def calculate_style_consistency(text):
#             sentences = text.split('。')
#             if not sentences:
#                 return 0.0
                
#             # 文末スタイルの一貫性を評価
#             end_style_scores = []
#             for sent in sentences:
#                 if not sent.strip():
#                     continue
                    
#                 # 文末パターンの評価（部分一致を許容）
#                 pattern_found = False
#                 for pattern_type, patterns in sentence_end_patterns.items():
#                     if any(p in sent[-10:] for p in patterns):  # 文末10文字以内で検索
#                         pattern_found = True
#                         break
#                 end_style_scores.append(1.0 if pattern_found else 0.0)
            
#             # 助動詞の一貫性を評価
#             aux_style_scores = []
#             for sent in sentences:
#                 if not sent.strip():
#                     continue
                    
#                 # 文中での助動詞使用を評価
#                 aux_found = any(p in sent for p in auxiliary_patterns)
#                 aux_style_scores.append(1.0 if aux_found else 0.0)
            
#             # 文の長さの一貫性を評価
#             lengths = [len(s.strip()) for s in sentences if s.strip()]
#             length_variance = np.var(lengths) if lengths else 0
#             length_score = 1.0 / (1.0 + length_variance/100)  # 分散が小さいほど高スコア
            
#             # 総合評価
#             end_style_avg = np.mean(end_style_scores) if end_style_scores else 0
#             aux_style_avg = np.mean(aux_style_scores) if aux_style_scores else 0
            
#             # 重み付け
#             weights = {
#                 'end_style': 0.5,
#                 'aux_style': 0.3,
#                 'length_consistency': 0.2
#             }
            
#             return (
#                 weights['end_style'] * end_style_avg +
#                 weights['aux_style'] * aux_style_avg +
#                 weights['length_consistency'] * length_score
#             )
        
#         # 各予測に対してスタイル一貫性を評価
#         style_scores = [calculate_style_consistency(pred) for pred in decoded_preds]
        
#         # 対話の流れも評価
#         def calculate_dialogue_flow(text):
#             sentences = text.split('。')
#             if not sentences:
#                 return 0.0
            
#             # より詳細な評価基準を追加
#             scores = []
            
#             # 1. 質問と説明のバランス（既存の評価）
#             questions = sum(1 for s in sentences if any(p in s for p in sentence_end_patterns['question_patterns']))
#             ratio = questions / len(sentences) if sentences else 0
#             balance_score = max(0.0, 1.0 - min(abs(0.3 - ratio), 0.2) * 2)
#             scores.append(balance_score)
            
#             # 2. 文の長さの変化
#             lengths = [len(s.strip()) for s in sentences if s.strip()]
#             length_variance = np.var(lengths) if len(lengths) > 1 else 0
#             length_score = 1.0 / (1.0 + length_variance/500)  # 分散が小さいほど高スコア
#             scores.append(length_score)
            
#             # 3. 接続詞の使用
#             conjunctions = ['しかし', 'だから', 'また', 'そして', 'したがって']
#             conj_count = sum(1 for s in sentences if any(c in s for c in conjunctions))
#             conj_ratio = conj_count / len(sentences)
#             conj_score = min(1.0, conj_ratio * 2)  # 適度な使用を評価
#             scores.append(conj_score)
            
#             # スコアの重み付け平均
#             weights = [0.5, 0.25, 0.25]  # バランス、長さ、接続詞の重み
#             final_score = sum(s * w for s, w in zip(scores, weights))
            
#             return max(0.1, min(1.0, final_score))  # 0.1から1.0の範囲に制限
        
#         flow_scores = [calculate_dialogue_flow(pred) for pred in decoded_preds]
        
#         style_score = np.mean(style_scores)
#         flow_score = np.mean(flow_scores)
        
#         # 総合評価スコアを追加
#         combined_score = (style_score * 0.6 + flow_score * 0.4)  # flow_scoreの重みを増加
        
#         return {
#             'style_consistency': style_score,
#             'dialogue_flow': flow_score,
#             'combined_score': combined_score
#         }

# claude修正案5(後)
def compute_metrics(eval_preds):
    logits, labels = eval_preds  # eval_predsから logits と labels を取得
    
    # デコード処理の改善
    with torch.no_grad():
        logits = torch.tensor(logits).cpu()
        predictions = torch.argmax(logits, dim=-1)
        
        # バッチ全体をデコード
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # より詳細なログ出力を追加
        logging.info(f"Sample prediction: {decoded_preds[0][:100]}...")
        
        del logits, predictions  # メモリ解放
        torch.cuda.empty_cache()

        socratic_patterns = {
            'question_endings': ['かね', 'だろうか', 'ではないかね'],
            'address_patterns': ['君は', '君が', '君の'],
            'inquiry_leads': ['では', 'について']
        }

        def calculate_socratic_style(text):
            sentences = text.split('。')
            if not sentences:
                return 0.0
            
            scores = []
            for sent in sentences:
                if not sent.strip():
                    continue
                
                # 問いで終わっているか
                ends_with_question = any(sent.endswith(p) for p in socratic_patterns['question_endings'])
                # 二人称の適切な使用
                uses_proper_address = any(p in sent for p in socratic_patterns['address_patterns'])
                # 問いの導入句の使用
                uses_inquiry_lead = any(p in sent for p in socratic_patterns['inquiry_leads'])
                
                # 各要素のスコア（問いで終わることを重視）
                sentence_score = (
                    (ends_with_question * 0.6) +
                    (uses_proper_address * 0.25) +
                    (uses_inquiry_lead * 0.15)
                )
                scores.append(sentence_score)
            
            return np.mean(scores) if scores else 0.0

        style_scores = [calculate_socratic_style(pred) for pred in decoded_preds]
        final_score = np.mean(style_scores)
        
        return {
            'socratic_style': final_score,
        }

# カスタムコールバックの修正
class StyleCallback(TrainerCallback):
    def __init__(self):
        self.socratic_scores = []
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if 'eval_socratic_style' in metrics:
            self.socratic_scores.append(metrics['eval_socratic_style'])
            
            # ログに詳細を記録
            logging.info(f"Step {state.global_step}:")
            logging.info(f"Socratic Style Score: {metrics['eval_socratic_style']:.3f}")
    
    def on_train_end(self, args, state, control, **kwargs):
        # 学習全体の評価をログに記録
        avg_score = sum(self.socratic_scores) / len(self.socratic_scores) if self.socratic_scores else 0
        
        logging.info("Training Complete!")
        logging.info(f"Average Socratic Style Score: {avg_score:.3f}")

# TrainingMonitorCallbackも修正
class TrainingMonitorCallback(TrainerCallback):
    def __init__(self):
        self.train_start_time = None
        self.metrics_history = {
            'step': [],
            'socratic_style': [],  # メトリクス名を変更
            'loss': [],
            'learning_rate': [],
            'epoch': []
        }
        self.output_dir = Path("model/training_progress")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = datetime.now()
        log_memory_usage()
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        # メトリクスの記録
        current_step = state.global_step
        
        # すべてのステップでlossとlearning_rateを記録
        if 'loss' in logs:
            self.metrics_history['step'].append(current_step)
            self.metrics_history['epoch'].append(state.epoch)
            self.metrics_history['loss'].append(logs['loss'])
            self.metrics_history['learning_rate'].append(logs.get('learning_rate', None))
            self.metrics_history['socratic_style'].append(None)  # 評価ステップ以外はNone
        
        # 評価ステップでsocratic_styleスコアを更新
        if 'eval_socratic_style' in logs:
            # 最後のエントリを更新（同じステップの場合）
            if self.metrics_history['step'] and self.metrics_history['step'][-1] == current_step:
                self.metrics_history['socratic_style'][-1] = logs['eval_socratic_style']
            else:
                # 新しいエントリを追加
                self.metrics_history['step'].append(current_step)
                self.metrics_history['epoch'].append(state.epoch)
                self.metrics_history['loss'].append(None)
                self.metrics_history['learning_rate'].append(None)
                self.metrics_history['socratic_style'].append(logs['eval_socratic_style'])
        
        # CSVファイルに保存
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.output_dir / 'training_metrics.csv', index=False)
        
        # 100ステップごとにグラフを更新
        if current_step % 100 == 0:
            self._plot_metrics()
            
    def _plot_metrics(self):
        """学習メトリクスをプロットして保存"""
        plt.figure(figsize=(15, 8))
        
        # Loss - Noneを除外して描画
        plt.subplot(2, 2, 1)
        valid_steps_loss = [s for s, v in zip(self.metrics_history['step'], self.metrics_history['loss']) if v is not None]
        valid_loss = [v for v in self.metrics_history['loss'] if v is not None]
        if valid_steps_loss:
            plt.plot(valid_steps_loss, valid_loss, label='Loss')
            plt.title('Training Loss')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.legend()
        
        # Learning Rate - Noneを除外して描画
        plt.subplot(2, 2, 2)
        valid_steps_lr = [s for s, v in zip(self.metrics_history['step'], self.metrics_history['learning_rate']) if v is not None]
        valid_lr = [v for v in self.metrics_history['learning_rate'] if v is not None]
        if valid_steps_lr:
            plt.plot(valid_steps_lr, valid_lr, label='LR')
            plt.title('Learning Rate')
            plt.xlabel('Step')
            plt.ylabel('Learning Rate')
            plt.legend()
        
        # Socratic Style Score - Noneを除外して描画
        plt.subplot(2, 2, 3)
        valid_steps = [s for s, v in zip(self.metrics_history['step'], self.metrics_history['socratic_style']) if v is not None]
        valid_scores = [v for v in self.metrics_history['socratic_style'] if v is not None]
        if valid_steps:
            plt.plot(valid_steps, valid_scores, label='Socratic Style')
            plt.title('Socratic Style Score')
            plt.xlabel('Step')
            plt.ylabel('Score')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_progress.png')
        plt.close()
    
    def on_train_end(self, args, state, control, **kwargs):
        # 最終的な学習結果のサマリーを保存
        summary = {
            'training_duration': str(datetime.now() - self.train_start_time),
            'final_loss': self.metrics_history['loss'][-1] if self.metrics_history['loss'] else None,
            'best_socratic_score': max(filter(None, self.metrics_history['socratic_style'])) if self.metrics_history['socratic_style'] else None,
            'total_steps': len(self.metrics_history['step']),
            'final_epoch': self.metrics_history['epoch'][-1] if self.metrics_history['epoch'] else None
        }
        
        # サマリーをJSONファイルとして保存
        with open(self.output_dir / 'training_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 最終的なグラフを保存
        self._plot_metrics()
        
        logging.info("Training Complete!")
        logging.info(f"Training duration: {summary['training_duration']}")
        
        # Noneチェックを追加
        if summary['final_loss'] is not None:
            logging.info(f"Final loss: {summary['final_loss']:.4f}")
        else:
            logging.info("Final loss: Not available")
        
        if summary['best_socratic_score'] is not None:
            logging.info(f"Best Socratic style score: {summary['best_socratic_score']:.4f}")
        else:
            logging.info("Best Socratic style score: Not available")

# データセットを訓練用と評価用に分割
dataset_size = len(tokenized_dataset)
indices = np.random.permutation(dataset_size)
split_idx = int(dataset_size * 0.8)

train_dataset = tokenized_dataset.select(indices[:split_idx])
# 評価データセットのサイズを制限
eval_dataset = tokenized_dataset.select(indices[split_idx:split_idx+100])  # 最大100サンプル

logging.info(f"Training dataset size: {len(train_dataset)}")
logging.info(f"Evaluation dataset size: {len(eval_dataset)}")

# メモリ解放の追加
def clear_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
class CustomTrainer(Trainer):
    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        if self.state.global_step % 100 == 0:
            clear_memory()
        return loss

# 評価用のカスタムTrainerクラスを作成
class CustomTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is not None:
            # 評価データセットを100サンプルに制限
            eval_dataset = eval_dataset.select(range(min(100, len(eval_dataset))))
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

# トレーナーの設定を更新
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[StyleCallback(), TrainingMonitorCallback()],
)

# 学習の実行
logging.info("Starting training...")
try:
    checkpoint_dir = "./model"
    resume_from_checkpoint = None
    
    # チェックポイントの確認と処理を修正
    if os.path.exists(checkpoint_dir):
        logging.info("\nChecking checkpoint status...")
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint-")]
        if checkpoints:
            # 最新のチェックポイントを取得
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            logging.info(f"Found latest checkpoint: {latest_checkpoint}")
            
            # チェックポイントの状態を確認
            state_path = os.path.join(checkpoint_path, "trainer_state.json")
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                current_epoch = state.get('epoch', 0)
                logging.info(f"\nCurrent training status:")
                logging.info(f"Current epoch: {current_epoch}")
                logging.info(f"Target epochs: {training_args.num_train_epochs}")
                
                # 完了している場合は安全に終了
                if current_epoch >= training_args.num_train_epochs - 0.1:
                    logging.info("\n" + "="*50)
                    logging.info("IMPORTANT NOTICE:")
                    logging.info(f"Training has already been completed at epoch {current_epoch}!")
                    logging.info(f"Target epochs was {training_args.num_train_epochs}")
                    logging.info(f"Trained model is available at: {checkpoint_dir}")
                    logging.info("="*50 + "\n")
                    exit(0)
            else:
                logging.warning("Invalid checkpoint state found. Proceeding with training...")
                logging.warning(f"Checkpoint directory: {checkpoint_dir}")
        else:
            logging.warning("Checkpoint directory exists but no checkpoints found. Proceeding with training...")

    # 学習を開始（または再開）
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logging.info("Training completed successfully!")
    
    # 設定の保存（JSONとして）
    import json

    def convert_to_serializable(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(x) for x in obj]
        return obj

    # 各設定を変換
    training_args_dict = convert_to_serializable(training_args.to_dict())
    lora_config_dict = convert_to_serializable(lora_config.to_dict())

    config_dict = {
        "model_name": model_name,
        "training_args": training_args_dict,
        "lora_config": lora_config_dict,
        "bnb_config": {
            "load_in_4bit": bnb_config.load_in_4bit,
            "bnb_4bit_use_double_quant": bnb_config.bnb_4bit_use_double_quant,
            "bnb_4bit_quant_type": bnb_config.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": str(bnb_config.bnb_4bit_compute_dtype),
        }
    }
    
    with open(os.path.join(training_args.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    # モデルの保存
    trainer.save_model()
    # 設定の保存
    model.config.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logging.info("Model and configuration saved successfully!")

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    # エラー発生時もチェックポイントは保持される
    raise 