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
DIALOGUE_JSON_PATH = "data/dialogue/processed/kaggle_model_40.json"  # 対話データのJSONファイルパス
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

# LoRAの設定を調整
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

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
    batch_size=32,
    num_proc=4,
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

# トークナイザーの設定を追加
tokenizer.add_special_tokens({
    'additional_special_tokens': [
        '。', '、', '！', '？',  # 句読点
    ]
})

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
    save_steps=50,          # 20ステップごとにチェックポイントを保存
    gradient_accumulation_steps=8,   # 累積回数を減らす
    max_steps=-1,
    disable_tqdm=False,
    logging_dir="./model/logs",
    logging_strategy="steps",
    logging_steps=50,        # 50ステップごとにログを記録
    no_cuda=False,
    dataloader_num_workers=2,
    report_to=[],
    run_name=None,
    per_device_train_batch_size=4,  # メモリが許す場合は増やす
    per_device_eval_batch_size=2,   # トレーニングバッチサイズの半分を評価用に設定
    gradient_checkpointing=True,
    max_grad_norm=0.5,       # 「Gradient clipping: Values between 0.5-1.0 help prevent divergence」から0.5を選択
    dataloader_pin_memory=True,
    save_total_limit=4,     # 最大3つのチェックポイントを保持
    fp16=True,
    optim="adamw_torch_fused",
    eval_accumulation_steps=8,
    load_best_model_at_end=False,  # 変更: メトリクスがないため
    metric_for_best_model=None,    # 変更: メトリクスがないため
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

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is not None:
            eval_dataset = eval_dataset.select(range(min(100, len(eval_dataset))))
        
        # 評価を実行
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Perplexityを計算（loss from metrics）
        try:
            perplexity = torch.exp(torch.tensor(metrics[f"{metric_key_prefix}_loss"]))
            metrics[f"{metric_key_prefix}_perplexity"] = perplexity.item()
            
            # ログに記録
            logging.info(f"Step {self.state.global_step}: Perplexity: {perplexity:.2f}")
        except Exception as e:
            logging.warning(f"Failed to calculate perplexity: {str(e)}")
        
        return metrics

class TrainingMonitorCallback(TrainerCallback):
    def __init__(self):
        self.metrics_history = {
            'step': [],           # トレーニングのステップ数
            'loss': [],          # 各ステップでの損失値
            'learning_rate': [], # 各ステップでの学習率
            'epoch': [],         # 現在のエポック数
            'perplexity': [],    # パープレキシティ（評価ステップ時のみ計算）
            'variance': [],      # 各ステップでの損失値の分散
            'bias': []           # 各ステップでの損失値のバイアス
        }
        # ディレクトリパスを設定して作成
        self.output_dir = Path("model/training_progress")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.running_mean = 0
        self.n_steps = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = datetime.now()
        log_memory_usage()
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        # デバッグ用のログ出力
        logging.info(f"Received logs: {logs}")
        logging.info(f"Current step: {state.global_step}")
        
        # 損失値の取得方法を修正
        current_loss = logs.get('loss', None)
        if current_loss is not None:
            current_loss = float(current_loss)  # テンソルを float に変換
        
        # 学習率の取得方法を修正
        current_lr = logs.get('learning_rate', None)
        if current_lr is not None:
            current_lr = float(current_lr)
        
        # メトリクスの記録
        self.metrics_history['step'].append(state.global_step)
        self.metrics_history['epoch'].append(state.epoch)
        self.metrics_history['loss'].append(current_loss)  # 修正された損失値
        self.metrics_history['learning_rate'].append(current_lr)  # 修正された学習率
        
        # Variance と Bias の計算
        if current_loss is not None:
            self.n_steps += 1
            delta = current_loss - self.running_mean
            self.running_mean += delta / self.n_steps
            
            # Variance の計算 (実際の損失値と移動平均との差の二乗)
            variance = (current_loss - self.running_mean) ** 2
            # Bias の計算 (移動平均と理想的な損失値0との差)
            bias = abs(self.running_mean - 0)
            
            self.metrics_history['variance'].append(variance)
            self.metrics_history['bias'].append(bias)
        else:
            self.metrics_history['variance'].append(None)
            self.metrics_history['bias'].append(None)
        
        # CSVファイルに保存
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.output_dir / 'training_metrics.csv', index=False)
        
        # 100ステップごとにグラフを更新
        if state.global_step % 100 == 0:
            self._plot_metrics()
            
    def _plot_metrics(self):
        """学習メトリクスをプロットして保存"""
        plt.figure(figsize=(15, 12))
        
        # Loss
        plt.subplot(3, 2, 1)
        plt.plot(self.metrics_history['step'], self.metrics_history['loss'], label='Loss')
        plt.title('Training Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        
        # Learning Rate
        plt.subplot(3, 2, 2)
        plt.plot(self.metrics_history['step'], self.metrics_history['learning_rate'], label='LR')
        plt.title('Learning Rate')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.legend()
        
        # Perplexity
        plt.subplot(3, 2, 3)
        valid_perplexity = [p for p in self.metrics_history['perplexity'] if p is not None]
        if valid_perplexity:
            plt.plot(
                [s for s, p in zip(self.metrics_history['step'], self.metrics_history['perplexity']) if p is not None],
                valid_perplexity,
                label='Perplexity'
            )
            plt.title('Perplexity')
            plt.xlabel('Step')
            plt.ylabel('Perplexity')
            plt.legend()
            
        # Variance
        plt.subplot(3, 2, 4)
        valid_variance = [v for v in self.metrics_history['variance'] if v is not None]
        if valid_variance:
            plt.plot(
                [s for s, v in zip(self.metrics_history['step'], self.metrics_history['variance']) if v is not None],
                valid_variance,
                label='Variance'
            )
            plt.title('Variance')
            plt.xlabel('Step')
            plt.ylabel('Variance')
            plt.legend()
            
        # Bias
        plt.subplot(3, 2, 5)
        valid_bias = [b for b in self.metrics_history['bias'] if b is not None]
        if valid_bias:
            plt.plot(
                [s for s, b in zip(self.metrics_history['step'], self.metrics_history['bias']) if b is not None],
                valid_bias,
                label='Bias'
            )
            plt.title('Bias')
            plt.xlabel('Step')
            plt.ylabel('Bias')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_progress.png')
        plt.close()

# トレーナーの設定を更新
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[TrainingMonitorCallback()],  # StyleCallbackを削除
)

# 学習の実行
logging.info("Starting training...")
try:
    checkpoint_dir = "./model"
    resume_from_checkpoint = None
    
    # チェックポイントの確認と処理
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
                logging.info(f"Current epoch: {current_epoch}")
                logging.info(f"Target epochs: {training_args.num_train_epochs}")
                
                # 完了している場合は安全に終了
                if current_epoch >= training_args.num_train_epochs - 0.1:
                    logging.info("Training has already been completed. Exiting to protect existing model.")
                    logging.info(f"Trained model is available at: {checkpoint_dir}")
                    exit(0)
                else:
                    resume_from_checkpoint = checkpoint_path
                    logging.info(f"Resuming from checkpoint: {checkpoint_path}")
            else:
                logging.warning("Invalid checkpoint state found. Proceeding with training from scratch.")
        else:
            logging.warning("Checkpoint directory exists but no checkpoints found. Starting fresh training.")
    else:
        logging.info("No checkpoint directory found. Starting fresh training.")
        os.makedirs(checkpoint_dir, exist_ok=True)

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