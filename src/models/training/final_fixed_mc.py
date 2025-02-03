# 1.初期設定とインポート部分
### 1.1 ライブラリインポートとグローバル定数設定
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
import psutil
import gc
import shutil

# Global Setting
DIALOGUE_JSON_PATH = "data/dialogue/processed/kaggle_model.json"  
MAX_SEQUENCE_LENGTH = 256
TOKENIZE_MAX_LENGTH = 256  

### 1.2 出力ディレクトリとロギング設定
# ディレクトリ設定
BASE_OUTPUT_DIR = "models/kaggle_model_ver2"  
MODEL_OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/model"
LOG_OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/logs" 

# ディレクトリの作成
for dir_path in [BASE_OUTPUT_DIR, MODEL_OUTPUT_DIR, LOG_OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_OUTPUT_DIR, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)


# Initial logging messages
logging.info("Training script started")
logging.info(f"Using dialogue file: {DIALOGUE_JSON_PATH}")
logging.info(f"Max sequence length: {MAX_SEQUENCE_LENGTH}")
logging.info(f"Output directory: {BASE_OUTPUT_DIR}")

### 1.3 環境設定とAPI認証
# Environment variables and warning settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore", category=FutureWarning)

# API keys
api_keys = get_api_keys()
os.environ["HUGGINGFACE_TOKEN"] = api_keys['huggingface_api_key']

# 2. データパイプライン
### 2.1 トークナイザー設定
# Model and tokenizer preparation
model_name = "google/gemma-2-2b-jpn-it"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=os.environ["HUGGINGFACE_TOKEN"],  
    trust_remote_code=True
)


# Add special tokens to tokenizer
tokenizer.add_special_tokens({
    'additional_special_tokens': [
        '。', '、', '！', '？',  # Punctuation marks
    ]
})

### 2.2 データセット準備と検証
def validate_message_format(message):
    """Validate message format"""
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
            
            # Validate message format
            if not all(validate_message_format(msg) for msg in messages):
                logging.warning(f"Skipped dialogue due to invalid message format")
                continue
                
            # Build conversation checking user->model sequence
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
            
            # Add only valid conversations
            if valid_sequence and current_conversation:
                # Apply Gemma chat template
                formatted_text = tokenizer.apply_chat_template(
                    current_conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Check token count
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
dataset = prepare_dataset()

# Check dataset structure
print("Dataset structure:")
print(dataset[0])  # Display first element
print("\nDataset features:")
print(dataset.features)
dataset = dataset.select(range(len(dataset))).shuffle(seed=42)

### 2.3 トークン化関数の定義

def tokenize_function(examples):
    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=TOKENIZE_MAX_LENGTH,      # 256 から TOKENIZE_MAX_LENGTH に変更
        padding='max_length',
        add_special_tokens=True,
        return_tensors=None,
    )
    return result

# Add dataset preprocessing
def preprocess_function(examples):
    """シンプルな前処理"""
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=TOKENIZE_MAX_LENGTH,
        padding='max_length',
        add_special_tokens=True,
        return_tensors=None
    )


### 2.4 データセットのトークン化と検証
# Optimize dataset processing
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=16,
    num_proc=2,
    load_from_cache_file=True,
    desc="Tokenizing datasets",
    remove_columns=dataset.column_names,
)

# Add dataset validation
def validate_dataset(dataset):
    # Check first element
    first_item = dataset[0]
    print("Validated first item structure:")
    print(f"Keys: {first_item.keys()}")
    print(f"input_ids type: {type(first_item['input_ids'])}")
    print(f"input_ids length: {len(first_item['input_ids'])}")
    return dataset

tokenized_dataset = validate_dataset(tokenized_dataset)

# 3. モデル設定
### 3.1 量子化設定（BitsAndBytes）
# Optimize BitsAndBytesConfig settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_storage=torch.uint8,
)

### 3.2 モデルロードと学習設定
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=os.environ["HUGGINGFACE_TOKEN"],  
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation='eager'
)


for param in model.parameters():
    param.requires_grad = True

model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# パラメータの勾配計算が有効になっているか確認
def check_requires_grad(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            logging.warning(f"Parameter {name} does not require gradients")

check_requires_grad(model)

### 3.3 LoRA設定とモデル変換
# Adjust LoRA configuration
lora_config = LoraConfig(
    r=8,                # 16から8に減少
    lora_alpha=16,      # 32から16に減少
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,  # 0.1から0.05に減少
    bias="none",
    task_type="CAUSAL_LM",
)

# Create LoRA model
model = get_peft_model(model, lora_config)

# 4. トレーニングインフラ
### 4.1 評価メトリクス定義
def compute_metrics(eval_preds):
    """基本的な評価メトリクスの計算"""
    logits, labels = eval_preds
    
    with torch.no_grad():
        # Convert logits to CPU tensor
        logits = torch.tensor(logits).cpu()
        labels = torch.tensor(labels).cpu()
        
        # Calculate perplexity
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            labels.view(-1),
            ignore_index=-100
        )
        perplexity = torch.exp(loss)
        
        # Clean up memory
        del logits, labels
        torch.cuda.empty_cache()
        
        return {
            'perplexity': perplexity.item(),
            'loss': loss.item()
        }


### 4.2 システムリソース監視
def log_memory_usage():
    import psutil
    import torch
    
    # CPU memory
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # GPU memory
    gpu_memory = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory.append({
                'device': i,
                'allocated': torch.cuda.memory_allocated(i) / 1024 / 1024,  # MB
                'reserved': torch.cuda.memory_reserved(i) / 1024 / 1024,    # MB
                'max_allocated': torch.cuda.max_memory_allocated(i) / 1024 / 1024  # MB
            })
    
    logging.info(f"CPU Memory usage: {cpu_memory:.2f} MB")
    for gpu in gpu_memory:
        logging.info(f"GPU {gpu['device']} Memory:")
        logging.info(f"  - Allocated: {gpu['allocated']:.2f} MB")
        logging.info(f"  - Reserved: {gpu['reserved']:.2f} MB")
        logging.info(f"  - Max Allocated: {gpu['max_allocated']:.2f} MB")

# Log dataset size
logging.info(f"Total dataset size: {len(dataset)}")
log_memory_usage()

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()



### 4.3 データセット分割とトレーニング設定
# Split dataset into training and evaluation sets
dataset_size = len(tokenized_dataset)
indices = np.random.permutation(dataset_size)
split_idx = int(dataset_size * 0.8)
train_dataset = tokenized_dataset.select(indices[:split_idx])
# Limit evaluation dataset size
eval_dataset = tokenized_dataset.select(indices[split_idx:split_idx+50])  # Maximum 50 samples

logging.info(f"Training dataset size: {len(train_dataset)}")
logging.info(f"Evaluation dataset size: {len(eval_dataset)}")

# Disable wandb via environment variable
os.environ["WANDB_DISABLED"] = "true"

# Update training arguments
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,  
    num_train_epochs=30,
    learning_rate=2e-4,           # 8e-5から2e-4に増加
    weight_decay=0.01,            # 0.06から0.01に減少
    warmup_ratio=0.1,             # 0.25から0.1に減少
    lr_scheduler_type="cosine",   # cosine_with_restartsからcosineに変更
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    gradient_accumulation_steps=8,    # バッチサイズを小さくした分、これを8に増やして補完
    max_steps=-1,
    disable_tqdm=False,
    logging_dir=LOG_OUTPUT_DIR,   
    logging_strategy="steps",
    logging_steps=10,
    no_cuda=False,
    dataloader_num_workers=1,         # CPUメモリ節約のため1に戻す
    report_to=[],
    run_name=None,
    per_device_train_batch_size=2,    # CPUメモリ節約のため2に戻す
    per_device_eval_batch_size=2,     # 同上
    gradient_checkpointing=True,
    max_grad_norm=1.0,             # 0.5から1.0に増加
    dataloader_pin_memory=True,
    save_total_limit=10,  # 10に増やす
    fp16=False,
    bf16=True,
    optim="adamw_torch",
    eval_accumulation_steps=4,
    load_best_model_at_end=True,
    metric_for_best_model="perplexity",
)



### 4.4 トレーニング監視システム実装
class TrainingMonitorCallback(TrainerCallback):
    def __init__(self):
        self.train_start_time = None
        self.metrics_history = {
            'step': [],
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'perplexity': [],
            'grad_norm': [],
            'gpu_memory_usage': [],
        }
        self.output_dir = Path(f"{BASE_OUTPUT_DIR}/training_progress")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 安定性監視用の設定
        self.window_size = 5  # 移動平均のウィンドウサイズ
        self.last_checkpoint_step = 0
        self.min_steps_between_checkpoints = 100  # チェックポイント間の最小ステップ数
        self.stable_checkpoints = []  # 安定したチェックポイントを記録
        
        # メトリクスの閾値設定
        self.perplexity_threshold = 2.5  # 良好なperplexityの閾値
        self.eval_loss_variance_threshold = 0.1  # eval_lossの許容変動幅
        self.grad_norm_bounds = (0.1, 2.0)  # grad_normの適正範囲
        self.max_stable_checkpoints = 5  # 安定チェックポイントの最大数を設定
        self.error_count = 0  # エラー回数を追跡
        self.max_consecutive_errors = 3  # 連続エラーの許容回数
        
        # バランス監視用の設定を追加
        self.variance_bias_window = 10  # より長いウィンドウで傾向を見る
        self.train_losses = []  # 訓練損失の履歴
        self.eval_losses = []   # 評価損失の履歴
        self.optimal_gap_range = (0.1, 0.3)  # 訓練損失と評価損失の理想的な差分範囲
    
    def _calculate_stability_metrics(self, state):
        """安定性メトリクスを計算"""
        if len(self.metrics_history['perplexity']) < self.window_size:
            return None
            
        recent_perplexity = self.metrics_history['perplexity'][-self.window_size:]
        recent_eval_loss = self.metrics_history['eval_loss'][-self.window_size:]
        recent_grad_norm = self.metrics_history['grad_norm'][-self.window_size:]
        
        # 移動平均と標準偏差を計算
        perplexity_mean = np.mean(recent_perplexity)
        eval_loss_std = np.std(recent_eval_loss)
        grad_norm_mean = np.mean(recent_grad_norm)
        
        return {
            'perplexity_mean': perplexity_mean,
            'eval_loss_std': eval_loss_std,
            'grad_norm_mean': grad_norm_mean
        }
    
    def _calculate_variance_bias_metrics(self):
        """分散と偏りのバランスを計算"""
        if len(self.train_losses) < self.variance_bias_window or \
           len(self.eval_losses) < self.variance_bias_window:
            return None
            
        recent_train = self.train_losses[-self.variance_bias_window:]
        recent_eval = self.eval_losses[-self.variance_bias_window:]
        
        # 訓練損失と評価損失の差（バイアスの指標）
        loss_gap = np.mean(recent_eval) - np.mean(recent_train)
        
        # 損失の変動（分散の指標）
        train_variance = np.var(recent_train)
        eval_variance = np.var(recent_eval)
        
        return {
            'loss_gap': loss_gap,
            'train_variance': train_variance,
            'eval_variance': eval_variance,
            'total_variance': (train_variance + eval_variance) / 2
        }
    
    def _is_balanced_state(self, variance_bias_metrics):
        """バランスの取れた状態かを判断"""
        if variance_bias_metrics is None:
            return False
            
        # 理想的な差分範囲内にあるか
        good_gap = (self.optimal_gap_range[0] <= variance_bias_metrics['loss_gap'] <= self.optimal_gap_range[1])
        
        # 分散が適度に小さいか
        stable_variance = variance_bias_metrics['total_variance'] < 0.1
        
        # 訓練と評価の分散が近いか（安定性の指標）
        variance_ratio = min(variance_bias_metrics['train_variance'], variance_bias_metrics['eval_variance']) / \
                        max(variance_bias_metrics['train_variance'], variance_bias_metrics['eval_variance'])
        balanced_variance = variance_ratio > 0.7  # 70%以上の類似性
        
        return good_gap and stable_variance and balanced_variance
    
    def _should_save_checkpoint(self, state, metrics):
        """チェックポイント保存の判断を拡張"""
        # 既存の条件をチェック
        basic_conditions = super()._should_save_checkpoint(state, metrics)
        
        # バランス状態もチェック
        variance_bias_metrics = self._calculate_variance_bias_metrics()
        balanced_state = self._is_balanced_state(variance_bias_metrics)
        
        if balanced_state:
            logging.info(f"Found balanced state at step {state.global_step}")
            logging.info(f"Variance-Bias metrics: {variance_bias_metrics}")
        
        return basic_conditions or balanced_state  # どちらかの条件を満たせば保存
    
    def _safe_save_checkpoint(self, checkpoint_dir, state, metrics, stability_metrics):
        """安全にチェックポイントを保存"""
        try:
            # チェックポイントディレクトリの作成を試みる
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # モデルの保存を試みる
            try:
                self.trainer.save_model(checkpoint_dir)
            except Exception as e:
                logging.error(f"Failed to save model checkpoint: {str(e)}")
                return False
            
            # メトリクス情報の保存を試みる
            try:
                metrics_path = os.path.join(checkpoint_dir, "stability_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump({
                        'step': state.global_step,
                        'metrics': stability_metrics,
                        'eval_metrics': metrics,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)
            except Exception as e:
                logging.error(f"Failed to save metrics: {str(e)}")
                # メトリクスの保存に失敗してもチェックポイントは有効
                
            return True
            
        except Exception as e:
            self.error_count += 1
            logging.error(f"Checkpoint creation failed (attempt {self.error_count}): {str(e)}")
            if self.error_count >= self.max_consecutive_errors:
                logging.warning("Too many consecutive checkpoint errors. Will skip future checkpoint attempts.")
            return False
    
    def _safe_remove_checkpoint(self, checkpoint_path):
        """安全にチェックポイントを削除"""
        try:
            if os.path.exists(checkpoint_path):
                shutil.rmtree(checkpoint_path)
                logging.info(f"Successfully removed old checkpoint: {checkpoint_path}")
        except Exception as e:
            logging.error(f"Failed to remove old checkpoint {checkpoint_path}: {str(e)}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """評価時のチェックポイント判断"""
        if not metrics:
            return
        
        try:
            stability_metrics = self._calculate_stability_metrics(state)
            if stability_metrics and self._should_save_checkpoint(state, stability_metrics):
                # 安定したチェックポイントとして保存を試みる
                checkpoint_dir = os.path.join(
                    args.output_dir,
                    f"stable_checkpoint-{state.global_step}"
                )
                
                if self._safe_save_checkpoint(checkpoint_dir, state, metrics, stability_metrics):
                    self.stable_checkpoints.append({
                        'step': state.global_step,
                        'path': checkpoint_dir,
                        'metrics': stability_metrics
                    })
                    self.last_checkpoint_step = state.global_step
                    self.error_count = 0  # 成功したらエラーカウントをリセット
                    logging.info(f"Saved stable checkpoint at step {state.global_step}")
                    logging.info(f"Stability metrics: {stability_metrics}")
                
        except Exception as e:
            logging.error(f"Error during evaluation callback: {str(e)}")
            # エラーが発生しても処理を継続
    
    def on_train_end(self, args, state, control, **kwargs):
        """トレーニング終了時の処理"""
        try:
            training_duration = datetime.now() - self.train_start_time
            
            # 安定したチェックポイントの概要を保存
            try:
                checkpoints_summary = os.path.join(self.output_dir, 'stable_checkpoints_summary.json')
                with open(checkpoints_summary, 'w') as f:
                    json.dump({
                        'total_checkpoints': len(self.stable_checkpoints),
                        'checkpoints': self.stable_checkpoints
                    }, f, indent=2)
            except Exception as e:
                logging.error(f"Failed to save checkpoints summary: {str(e)}")
            
            # メトリクス履歴の保存を試みる
            try:
                history_file = self.output_dir / 'training_metrics.json'
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump(self.metrics_history, f, indent=2)
            except Exception as e:
                logging.error(f"Failed to save training metrics: {str(e)}")
            
            # 終了ログの出力
            logging.info(f"Training completed. Duration: {training_duration}")
            self._log_final_metrics()
            
        except Exception as e:
            logging.error(f"Error during training end callback: {str(e)}")
    
    def _log_final_metrics(self):
        """最終メトリクスのログ出力"""
        try:
            if self.metrics_history['train_loss']:
                logging.info(f"Final training loss: {self.metrics_history['train_loss'][-1]:.4f}")
            if self.metrics_history['eval_loss']:
                logging.info(f"Final evaluation loss: {self.metrics_history['eval_loss'][-1]:.4f}")
            if self.metrics_history['perplexity']:
                logging.info(f"Final perplexity: {self.metrics_history['perplexity'][-1]:.4f}")
            logging.info(f"Total stable checkpoints saved: {len(self.stable_checkpoints)}")
        except Exception as e:
            logging.error(f"Error logging final metrics: {str(e)}")

### 4.5 トレーナー実装と初期化
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

class CustomTrainer(Trainer):
    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        if self.state.global_step % 50 == 0:
            clear_memory()
            gc.collect()
            torch.cuda.empty_cache()
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is not None:
            # Limit evaluation dataset to 100 samples
            eval_dataset = eval_dataset.select(range(min(100, len(eval_dataset))))
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

# Trainer initialization
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[TrainingMonitorCallback()],
)


# 5. トレーニング実行
### 5.1 チェックポイント管理とトレーニング実行

# Start training
logging.info("Starting training...")
try:
    checkpoint_dir = MODEL_OUTPUT_DIR  
    resume_from_checkpoint = None
    
    # Check if running in Kaggle environment
    is_kaggle = os.path.exists('/kaggle/working')
    
    # Checkpoint status and processing
    if os.path.exists(checkpoint_dir):
        print("\nChecking checkpoint status...")  
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint-")]
        if checkpoints:
            # Get latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"Found latest checkpoint: {latest_checkpoint}") 
            
            # Check checkpoint status
            state_path = os.path.join(checkpoint_path, "trainer_state.json")
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                current_epoch = state.get('epoch', 0)
                print(f"\nCurrent training status:")  
                print(f"Current epoch: {current_epoch}")  
                print(f"Target epochs: {training_args.num_train_epochs}")  
                
                # Exit safely if completed
                if current_epoch >= training_args.num_train_epochs - 0.1:
                    print("\n" + "="*50)
                    print("IMPORTANT NOTICE:")
                    print(f"Training has already been completed at epoch {current_epoch}!")
                    print(f"Target epochs was {training_args.num_train_epochs}")  
                    print(f"Trained model is available at: {checkpoint_dir}")
                    print("="*50 + "\n")
                    logging.info("Training has already been completed. Exiting to protect existing model.")
                    logging.info(f"Trained model is available at: {checkpoint_dir}")
                    exit(0)
            else:
                logging.warning("Invalid checkpoint state found. Please check manually.")
                logging.warning(f"Checkpoint directory: {checkpoint_dir}")
        else:
            logging.warning("Checkpoint directory exists but no checkpoints found.")
            logging.info("Continuing with training...")  # 追加: 自動的に続行

    # Start training (or resume)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logging.info("Training completed successfully!")

    ### 5.2 ベストモデルと設定の保存
    # Save settings (as JSON)
    def convert_to_serializable(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(x) for x in obj]
        return obj

    # Convert each setting
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
    
    # Save configurations
    with open(os.path.join(training_args.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    # トレーナーが保持している最良のモデルを保存
    # load_best_model_at_end=Trueにより、この時点で既にbestモデルがロードされている
    best_model_path = os.path.join(training_args.output_dir, "best_model")
    os.makedirs(best_model_path, exist_ok=True)
    
    # Save best model and its configuration
    trainer.model.save_pretrained(best_model_path)
    model.config.save_pretrained(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    
    # Save a marker file indicating this is the best model
    with open(os.path.join(best_model_path, "best_model_info.json"), "w", encoding="utf-8") as f:
        best_metrics = {
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_perplexity": trainer.state.best_metric
        }
        json.dump(best_metrics, f, indent=2)
    
    logging.info(f"Best model saved to {best_model_path}")
    logging.info(f"Best perplexity: {trainer.state.best_metric}")
    logging.info("Model and configuration saved successfully!")

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    raise