#train_finalをkaggleでも使えるよにしたやつ。ただT4x2用だからGPU二つある

# 1.初期設定とインポート部分
### 1.1 インポートとグローバル設定
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

# Global Setting
DIALOGUE_JSON_PATH = "data/dialogue/processed/kaggle_model.json"  
MAX_SEQUENCE_LENGTH = 256
TOKENIZE_MAX_LENGTH = 256  

### 1.2 ディレクトリ設定とロギング設定
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

### 1.3 環境変数とAPI設定
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

### 2.2 データセット準備
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

### 2.3 データ前処理と検証

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


### 2.4 データセット最適化
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


tokenized_dataset = tokenized_dataset.map(
    preprocess_function,
    batched=True,
    desc="Applying attention masking"
)


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

### 3.2 モデルロードと初期化
# Load model with modifications
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=os.environ["HUGGINGFACE_TOKEN"],  
    quantization_config=bnb_config,
    device_map="balanced",
    torch_dtype=torch.float16,
    attn_implementation='sdpa',
    max_memory={0: "4GiB", 1: "4GiB", "cpu": "24GB"}
)

# Prepare model for LoRA and disable cache
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

### 3.3 LoRA設定
# Adjust LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
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


### 4.2 メモリ管理とモニタリング
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


### 4.3 トレーニング引数設定
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
    learning_rate=8e-5,
    weight_decay=0.06,
    warmup_ratio=0.25,
    lr_scheduler_type="cosine_with_restarts",
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    gradient_accumulation_steps=8,
    max_steps=-1,
    disable_tqdm=False,
    logging_dir=LOG_OUTPUT_DIR,   
    logging_strategy="steps",
    logging_steps=10,
    no_cuda=False,
    dataloader_num_workers=1,
    report_to=[],
    run_name=None,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_checkpointing=True,
    max_grad_norm=0.5,
    dataloader_pin_memory=True,
    save_total_limit=2,
    fp16=True,
    optim="adamw_torch_fused",
    eval_accumulation_steps=4,
    load_best_model_at_end=True,
    metric_for_best_model="perplexity",
)



### 4.4 コールバック実装
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
    
    def _record_gpu_memory(self):
        """Record GPU memory usage"""
        if torch.cuda.is_available():
            memory_allocated = []
            for i in range(torch.cuda.device_count()):
                memory_allocated.append(
                    torch.cuda.memory_allocated(i) / (1024 * 1024 * 1024)  # Convert to GB
                )
            return memory_allocated
        return None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = datetime.now()
        logging.info("Training started at: %s", self.train_start_time)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Record basic metrics
            step = state.global_step
            self.metrics_history['step'].append(step)
            
            # Training loss
            if 'loss' in logs:
                self.metrics_history['train_loss'].append(logs['loss'])
            
            # Learning rate
            if 'learning_rate' in logs:
                self.metrics_history['learning_rate'].append(logs['learning_rate'])
            
            # Gradient norm
            if 'grad_norm' in logs:
                self.metrics_history['grad_norm'].append(logs['grad_norm'])
            
            # Evaluation metrics
            if 'eval_loss' in logs:
                self.metrics_history['eval_loss'].append(logs['eval_loss'])
            if 'eval_perplexity' in logs:
                self.metrics_history['perplexity'].append(logs['eval_perplexity'])
            
            # GPU memory usage
            gpu_memory = self._record_gpu_memory()
            if gpu_memory is not None:
                self.metrics_history['gpu_memory_usage'].append(gpu_memory)
            
            # Log current metrics
            logging.info(
                f"Step {step} - "
                f"Loss: {logs.get('loss', 'N/A'):.4f}, "
                f"LR: {logs.get('learning_rate', 'N/A'):.2e}, "
                f"Grad Norm: {logs.get('grad_norm', 'N/A'):.4f}"
            )
    
    def on_train_end(self, args, state, control, **kwargs):
        training_duration = datetime.now() - self.train_start_time
        
        # Save training history
        history_file = self.output_dir / 'training_metrics.json'
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Log final summary
        logging.info(f"Training completed. Duration: {training_duration}")
        if self.metrics_history['train_loss']:
            logging.info(f"Final training loss: {self.metrics_history['train_loss'][-1]:.4f}")
        if self.metrics_history['eval_loss']:
            logging.info(f"Final evaluation loss: {self.metrics_history['eval_loss'][-1]:.4f}")
        if self.metrics_history['perplexity']:
            logging.info(f"Final perplexity: {self.metrics_history['perplexity'][-1]:.4f}")

### 4.5 カスタムトレーナー定義とコレータ設定
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
### 5.1 トレーニング実行と例外処理


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
                if not is_kaggle:  
                    user_input = input("Do you want to continue and overwrite? (yes/no): ")
                    if user_input.lower() != 'yes':
                        logging.info("Aborting to protect existing data.")
                        exit(0)
        else:
            logging.warning("Checkpoint directory exists but no checkpoints found.")
            if not is_kaggle:  
                user_input = input("Do you want to continue and overwrite the directory? (yes/no): ")
                if user_input.lower() != 'yes':
                    logging.info("Aborting to protect existing data.")
                    exit(0)

    # Start training (or resume)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logging.info("Training completed successfully!")

    ### 5.2 モデル保存と設定エクスポート
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
    
    # Save model and settings
    trainer.save_model()
    model.config.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logging.info("Model and configuration saved successfully!")

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    raise