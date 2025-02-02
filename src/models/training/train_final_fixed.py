#train_finalをkaggleでも使えるよにしたやつ。ただT4x2用だからGPU二つある
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
MAX_SEQUENCE_LENGTH = 512
TOKENIZE_MAX_LENGTH = 256  # 追加: トークン化時の最大長

# Setup output directory paths
BASE_OUTPUT_DIR = "models/kaggle_model_ver2"  
MODEL_OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/model"
LOG_OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/logs"

# Create directories
for dir_path in [BASE_OUTPUT_DIR, MODEL_OUTPUT_DIR, LOG_OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Setup logging configuration immediately after directory creation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_OUTPUT_DIR, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)

# Initial logging messages to verify logging is working
logging.info("Training script started")
logging.info(f"Using dialogue file: {DIALOGUE_JSON_PATH}")
logging.info(f"Max sequence length: {MAX_SEQUENCE_LENGTH}")
logging.info(f"Output directory: {BASE_OUTPUT_DIR}")

# Environment variables and warning settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore", category=FutureWarning)

# API keys
api_keys = get_api_keys()
os.environ["HUGGINGFACE_TOKEN"] = api_keys['huggingface_api_key']

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

# Model and tokenizer preparation
model_name = "google/gemma-2-2b-jpn-it"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=os.environ["HUGGINGFACE_TOKEN"],  
    trust_remote_code=True
)

# Optimize BitsAndBytesConfig settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_storage=torch.uint8,
)

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

# Memory efficiency settings
model.config.use_cache = False

# Dataset preparation
dataset = prepare_dataset()

# Check dataset structure
print("Dataset structure:")
print(dataset[0])  # Display first element
print("\nDataset features:")
print(dataset.features)

# Optimize dataset batch processing
dataset = dataset.select(range(len(dataset))).shuffle(seed=42)

# Optimize tokenize function
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

# Add memory usage monitoring log
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

# Add dataset preprocessing
def preprocess_function(examples):
    # Pattern definitions
    end_patterns = [
        "だろうか", "ではないか", "のではないか", "かね",
        "なるほど", "興味深い", "考えてみよう"
    ]
    
    # Conjunction patterns
    conjunctions = [
        "しかし", "だから", "それでは", "すなわち",
        "たとえば", "つまり", "ならば", "もし"
    ]
    
    # Get tokenized texts
    texts = tokenizer.batch_decode(examples['input_ids'])
    new_attention_masks = []
    
    for text, mask in zip(texts, examples['attention_mask']):
        if not isinstance(mask, list):
            mask = mask.tolist()
        
        # Create new attention mask (base value 0.8)
        new_mask = [0.8] * len(mask)
        
        # Split into sentences
        sentences = text.split('。')
        current_pos = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Detect and emphasize end patterns
            for pattern in end_patterns:
                if pattern in sentence:
                    # Locate pattern position
                    pattern_tokens = tokenizer.encode(pattern, add_special_tokens=False)
                    pattern_len = len(pattern_tokens)
                    
                    # Emphasize tokens containing pattern and surrounding tokens
                    pattern_start = current_pos + len(tokenizer.encode(sentence, add_special_tokens=False)) - pattern_len
                    for i in range(max(0, pattern_start - 2), min(len(mask), pattern_start + pattern_len + 2)):
                        new_mask[i] = 1.0  # Maximum attention for pattern parts
            
            # Detect and emphasize conjunctions
            for conj in conjunctions:
                if conj in sentence:
                    # Locate conjunction position
                    conj_tokens = tokenizer.encode(conj, add_special_tokens=False)
                    conj_pos = len(tokenizer.encode(sentence.split(conj)[0], add_special_tokens=False))
                    
                    # Emphasize tokens before and after conjunction (slightly lower)
                    for i in range(max(0, current_pos + conj_pos - 1), 
                                 min(len(mask), current_pos + conj_pos + len(conj_tokens) + 1)):
                        new_mask[i] = 0.9
            
            # Emphasize punctuation marks
            for i, char in enumerate(sentence):
                if char in '、。！？':
                    # Locate punctuation position
                    punct_pos = len(tokenizer.encode(sentence[:i], add_special_tokens=False))
                    # Emphasize tokens around punctuation
                    for j in range(max(0, current_pos + punct_pos - 1),
                                 min(len(mask), current_pos + punct_pos + 2)):
                        new_mask[j] = 0.95
            
            # Update position for next sentence
            current_pos += len(tokenizer.encode(sentence + '。', add_special_tokens=False))
        
        # Set special token masks to 1.0
        if tokenizer.bos_token_id is not None:
            new_mask[0] = 1.0  # BOS token
        if tokenizer.eos_token_id is not None:
            new_mask[-1] = 1.0  # EOS token
            
        new_attention_masks.append(new_mask)

    examples['attention_mask'] = new_attention_masks
    return examples

# Add special tokens to tokenizer
tokenizer.add_special_tokens({
    'additional_special_tokens': [
        '。', '、', '！', '？',  # Punctuation marks
    ]
})


tokenized_dataset = tokenized_dataset.map(
    preprocess_function,
    batched=True,
    desc="Applying attention masking"
)

# Create log directory
log_dir = f"{BASE_OUTPUT_DIR}/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)

# Update training arguments
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,  
    num_train_epochs=30,
    learning_rate=8e-5,
    weight_decay=0.06,
    warmup_ratio=0.25,
    lr_scheduler_type="cosine_with_restarts",
    evaluation_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=20,
    gradient_accumulation_steps=8,
    max_steps=-1,
    disable_tqdm=False,
    logging_dir=LOG_OUTPUT_DIR,   
    logging_strategy="steps",
    logging_steps=50,
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
    metric_for_best_model="combined_score",
)

# Disable wandb via environment variable (add before training_args)
import os
os.environ["WANDB_DISABLED"] = "true"

# Modify data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# Update evaluation metrics
def compute_metrics(eval_preds):
    logits, labels = eval_preds  # Get logits and labels from eval_preds
    
    # Relax size limit for evaluation dataset
    max_samples = 100
    
    # Improve decoding process
    with torch.no_grad():
        logits = torch.tensor(logits).cpu()
        predictions = torch.argmax(logits, dim=-1)
        
        # Decode batch
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Add more detailed logging
        logging.info(f"Sample prediction: {decoded_preds[0][:100]}...")
        
        del logits, predictions  # Memory release
        torch.cuda.empty_cache()
        
        # Define sentence ending patterns more flexibly
        sentence_end_patterns = {
            'question_patterns': [
                'かね', 'だろうか', 'ではないか',
                'のか', 'と思わないか', '考えてみよう',
            ],
            'statement_patterns': [
                'だね', 'なるほど', '興味深い',
                'といえよう', 'というべきだ'
            ],
            'reflection_patterns': [
                'かもしれない', 'のではないか',
                'と考えられる', 'といえそうだ'
            ]
        }
        
        # Auxiliary verb patterns
        auxiliary_patterns = [
            'である', 'だ', 'です', 'ます',
            'のだ', 'のです', 'のである'
        ]
        
        def calculate_style_consistency(text):
            sentences = text.split('。')
            if not sentences:
                return 0.0
                
            # Evaluate sentence ending style consistency
            end_style_scores = []
            for sent in sentences:
                if not sent.strip():
                    continue
                    
                # Evaluate sentence ending patterns (partial match allowed)
                pattern_found = False
                for pattern_type, patterns in sentence_end_patterns.items():
                    if any(p in sent[-10:] for p in patterns):  # Search within 10 characters at the end
                        pattern_found = True
                        break
                end_style_scores.append(1.0 if pattern_found else 0.0)
            
            # Evaluate auxiliary verb consistency
            aux_style_scores = []
            for sent in sentences:
                if not sent.strip():
                    continue
                    
                # Evaluate auxiliary verb usage in the sentence
                aux_found = any(p in sent for p in auxiliary_patterns)
                aux_style_scores.append(1.0 if aux_found else 0.0)
            
            # Evaluate sentence length consistency
            lengths = [len(s.strip()) for s in sentences if s.strip()]
            length_variance = np.var(lengths) if lengths else 0
            length_score = 1.0 / (1.0 + length_variance/100)  # Higher score if variance is small
            
            # Overall evaluation
            end_style_avg = np.mean(end_style_scores) if end_style_scores else 0
            aux_style_avg = np.mean(aux_style_scores) if aux_style_scores else 0
            
            # Weighting
            weights = {
                'end_style': 0.5,
                'aux_style': 0.3,
                'length_consistency': 0.2
            }
            
            return (
                weights['end_style'] * end_style_avg +
                weights['aux_style'] * aux_style_avg +
                weights['length_consistency'] * length_score
            )
        
        # Evaluate style consistency for each prediction
        style_scores = [calculate_style_consistency(pred) for pred in decoded_preds]
        
        # Evaluate dialogue flow
        def calculate_dialogue_flow(text):
            sentences = text.split('。')
            if not sentences:
                return 0.0
            
            # Add more detailed evaluation criteria
            scores = []
            
            # 1. Balance between questions and statements
            questions = sum(1 for s in sentences if any(p in s for p in sentence_end_patterns['question_patterns']))
            ratio = questions / len(sentences) if sentences else 0
            balance_score = max(0.0, 1.0 - min(abs(0.3 - ratio), 0.2) * 2)
            scores.append(balance_score)
            
            # 2. Sentence length change
            lengths = [len(s.strip()) for s in sentences if s.strip()]
            length_variance = np.var(lengths) if len(lengths) > 1 else 0
            length_score = 1.0 / (1.0 + length_variance/500)  # Higher score if variance is small
            scores.append(length_score)
            
            # 3. Use of conjunctions
            conjunctions = ['しかし', 'だから', 'また', 'そして', 'したがって']
            conj_count = sum(1 for s in sentences if any(c in s for c in conjunctions))
            conj_ratio = conj_count / len(sentences)
            conj_score = min(1.0, conj_ratio * 2)  # Evaluate moderate usage
            scores.append(conj_score)
            
            # Weighted average of scores
            weights = [0.5, 0.25, 0.25]  # Balance, length, conjunction weights
            final_score = sum(s * w for s, w in zip(scores, weights))
            
            return max(0.1, min(1.0, final_score))  # Limit to range 0.1 to 1.0
        
        flow_scores = [calculate_dialogue_flow(pred) for pred in decoded_preds]
        
        style_score = np.mean(style_scores)
        flow_score = np.mean(flow_scores)
        
        # Add overall evaluation score
        combined_score = (style_score * 0.6 + flow_score * 0.4)  # Increase flow_score weight
        
        return {
            'style_consistency': style_score,
            'dialogue_flow': flow_score,
            'combined_score': combined_score
        }

# Update custom callbacks
class StyleCallback(TrainerCallback):
    def __init__(self):
        self.style_scores = []
        self.flow_scores = []
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if 'eval_style_consistency' in metrics:
            self.style_scores.append(metrics['eval_style_consistency'])
            self.flow_scores.append(metrics['eval_dialogue_flow'])
            
            # Log detailed information
            logging.info(f"Step {state.global_step}:")
            logging.info(f"Style Consistency: {metrics['eval_style_consistency']:.3f}")
            logging.info(f"Dialogue Flow: {metrics['eval_dialogue_flow']:.3f}")
    
    def on_train_end(self, args, state, control, **kwargs):
        # Log overall evaluation
        avg_style = sum(self.style_scores) / len(self.style_scores) if self.style_scores else 0
        avg_flow = sum(self.flow_scores) / len(self.flow_scores) if self.flow_scores else 0
        
        logging.info("Training Complete!")
        logging.info(f"Average Style Consistency: {avg_style:.3f}")
        logging.info(f"Average Dialogue Flow: {avg_flow:.3f}")

# Extend custom callbacks
class TrainingMonitorCallback(TrainerCallback):
    def __init__(self):
        # Import psutil here as well for safety
        import psutil
        self.train_start_time = None
        self.metrics_history = {
            'step': [],
            'style_consistency': [],
            'dialogue_flow': [],
            'combined_score': [],
            'loss': [],
            'learning_rate': [],
            'epoch': [],
            'cpu_ram_usage': [],
            'gpu_vram_usage': [],
            'gpu_utilization': [],
            'batch_size': [],
            'moving_avg_loss': [],
            # 新しい詳細メトリクス
            'lr_schedule': [],
            'batch_metrics': [],
            'gpu_metrics': [],
            'grad_norm': []
        }
        self.peak_metrics = {
            'cpu_ram': 0,
            'gpu_vram': 0,
            'gpu_util': 0
        }
        self.output_dir = Path(f"{BASE_OUTPUT_DIR}/training_progress")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _record_resource_usage(self):
        """Record current resource usage with timestamp"""
        import psutil
        import torch
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # CPU RAM
        cpu_ram = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)  # GB
        self.peak_metrics['cpu_ram'] = max(self.peak_metrics['cpu_ram'], cpu_ram)
        
        # GPU metrics with timestamp
        if torch.cuda.is_available():
            gpu_metrics = []
            for i in range(torch.cuda.device_count()):
                vram_used = torch.cuda.memory_allocated(i) / (1024 * 1024 * 1024)  # GB
                self.peak_metrics['gpu_vram'] = max(self.peak_metrics['gpu_vram'], vram_used)
                
                # GPU utilization (requires nvidia-smi)
                try:
                    import subprocess
                    result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
                    gpu_util = float(result.decode('utf-8').strip())
                    self.peak_metrics['gpu_util'] = max(self.peak_metrics['gpu_util'], gpu_util)
                except:
                    gpu_util = 0
                
                gpu_metrics.append({
                    'device': i,
                    'vram_used': vram_used,
                    'utilization': gpu_util
                })
                
            # 時系列データとして保存
            self.metrics_history['gpu_metrics'].append({
                'timestamp': current_time,
                'metrics': gpu_metrics
            })
                
            self.metrics_history['cpu_ram_usage'].append(cpu_ram)
            self.metrics_history['gpu_vram_usage'].append(vram_used)
            self.metrics_history['gpu_utilization'].append(gpu_util)
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = datetime.now()
        logging.info("Training started at: %s", self.train_start_time)
        self._record_resource_usage()
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 学習率とスケジューリングの記録
            if 'learning_rate' in logs:
                self.metrics_history['lr_schedule'].append({
                    'timestamp': current_time,
                    'step': state.global_step,
                    'learning_rate': logs['learning_rate'],
                    'schedule_type': args.lr_scheduler_type
                })
                self.metrics_history['learning_rate'].append(logs['learning_rate'])
            
            # バッチサイズと損失値の関連を記録
            if 'loss' in logs:
                self.metrics_history['batch_metrics'].append({
                    'timestamp': current_time,
                    'step': state.global_step,
                    'batch_size': args.per_device_train_batch_size,
                    'loss': logs['loss'],
                    'grad_norm': logs.get('grad_norm', None)
                })
                self.metrics_history['loss'].append(logs['loss'])
                self.metrics_history['batch_size'].append(args.per_device_train_batch_size)
                if 'grad_norm' in logs:
                    self.metrics_history['grad_norm'].append(logs['grad_norm'])
            
            # 移動平均の計算と記録
            if len(self.metrics_history['loss']) > 10:
                avg_loss = sum(self.metrics_history['loss'][-10:]) / 10
                self.metrics_history['moving_avg_loss'].append(avg_loss)
                logging.info(f"Moving average loss (last 10 steps): {avg_loss:.4f}")
            
            logging.info(f"Step {state.global_step}: {logs}")
            if 'grad_norm' in logs:
                logging.info(f"Gradient norm: {logs['grad_norm']:.4f}")
            
        self._record_resource_usage()
        
    def on_train_end(self, args, state, control, **kwargs):
        training_duration = datetime.now() - self.train_start_time
        
        # 詳細な学習履歴の保存
        training_history = {
            'lr_schedule': self.metrics_history['lr_schedule'],
            'batch_metrics': self.metrics_history['batch_metrics'],
            'gpu_metrics': self.metrics_history['gpu_metrics'],
            'moving_avg_loss': self.metrics_history['moving_avg_loss']
        }
        
        # 学習履歴をJSONファイルとして保存
        history_file = self.output_dir / 'training_history.json'
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(training_history, f, indent=2, ensure_ascii=False)
        
        # 基本的なメトリクスのログ出力
        logging.info(f"Training completed. Total duration: {training_duration}")
        logging.info(f"Peak CPU RAM usage: {self.peak_metrics['cpu_ram']:.2f} GB")
        logging.info(f"Peak GPU VRAM usage: {self.peak_metrics['gpu_vram']:.2f} GB")
        logging.info(f"Peak GPU utilization: {self.peak_metrics['gpu_util']:.1f}%")
        
        # 最終サマリーの作成と保存
        summary = {
            'training_duration': str(training_duration),
            'final_loss': self.metrics_history['loss'][-1] if self.metrics_history['loss'] else None,
            'best_combined_score': max(filter(None, self.metrics_history['combined_score'])) if self.metrics_history['combined_score'] else None,
            'total_steps': len(self.metrics_history['step']),
            'final_epoch': self.metrics_history['epoch'][-1] if self.metrics_history['epoch'] else None,
            'learning_rate_summary': {
                'initial': self.metrics_history['learning_rate'][0] if self.metrics_history['learning_rate'] else None,
                'final': self.metrics_history['learning_rate'][-1] if self.metrics_history['learning_rate'] else None,
                'schedule_type': args.lr_scheduler_type
            },
            'loss_summary': {
                'final_moving_avg': self.metrics_history['moving_avg_loss'][-1] if self.metrics_history['moving_avg_loss'] else None,
                'best_loss': min(self.metrics_history['loss']) if self.metrics_history['loss'] else None
            },
            'resource_usage': {
                'peak_cpu_ram_gb': self.peak_metrics['cpu_ram'],
                'peak_gpu_vram_gb': self.peak_metrics['gpu_vram'],
                'peak_gpu_utilization': self.peak_metrics['gpu_util']
            },
            'hardware_info': {
                'cpu_info': self._get_cpu_info(),
                'gpu_info': self._get_gpu_info(),
                'total_ram': self._get_total_ram()
            }
        }
        
        # サマリーをJSONファイルとして保存
        with open(self.output_dir / 'training_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        logging.info("Training Complete!")
        logging.info(f"Training duration: {summary['training_duration']}")
        
        # Noneチェックを追加
        if summary['loss_summary']['final_moving_avg'] is not None:
            logging.info(f"Final moving average loss: {summary['loss_summary']['final_moving_avg']:.4f}")
        if summary['loss_summary']['best_loss'] is not None:
            logging.info(f"Best loss achieved: {summary['loss_summary']['best_loss']:.4f}")
        
        logging.info(f"Peak CPU RAM usage: {summary['resource_usage']['peak_cpu_ram_gb']:.2f} GB")
        logging.info(f"Peak GPU VRAM usage: {summary['resource_usage']['peak_gpu_vram_gb']:.2f} GB")
        logging.info(f"Peak GPU utilization: {summary['resource_usage']['peak_gpu_utilization']:.1f}%")

    def _get_cpu_info(self):
        import cpuinfo
        try:
            info = cpuinfo.get_cpu_info()
            return {
                'model': info.get('brand_raw', 'Unknown'),
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True)
            }
        except:
            return "Failed to get CPU info"
            
    def _get_gpu_info(self):
        if not torch.cuda.is_available():
            return "No GPU available"
        try:
            import subprocess
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=gpu_name,memory.total', '--format=csv,noheader,nounits'])
            gpus = result.decode('utf-8').strip().split('\n')
            return [{'model': g.split(',')[0], 'memory': float(g.split(',')[1])/1024} for g in gpus]
        except:
            return "Failed to get GPU info"
            
    def _get_total_ram(self):
        return psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB

# Split dataset into training and evaluation sets
dataset_size = len(tokenized_dataset)
indices = np.random.permutation(dataset_size)
split_idx = int(dataset_size * 0.8)

train_dataset = tokenized_dataset.select(indices[:split_idx])
# Limit evaluation dataset size
eval_dataset = tokenized_dataset.select(indices[split_idx:split_idx+50])  # Maximum 50 samples

logging.info(f"Training dataset size: {len(train_dataset)}")
logging.info(f"Evaluation dataset size: {len(eval_dataset)}")

# Add memory cleanup
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    
class CustomTrainer(Trainer):
    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        if self.state.global_step % 50 == 0:
            clear_memory()
            gc.collect()
            torch.cuda.empty_cache()
        return loss

# Create custom Trainer class for evaluation
class CustomTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is not None:
            # Limit evaluation dataset to 100 samples
            eval_dataset = eval_dataset.select(range(min(100, len(eval_dataset))))
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

# Update trainer settings
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[StyleCallback(), TrainingMonitorCallback()],
)

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
    
    # Save settings (as JSON)
    import json

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
    
    with open(os.path.join(training_args.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    # Save model
    trainer.save_model()
    # Save settings
    model.config.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logging.info("Model and configuration saved successfully!")

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    # Checkpoints are preserved even if an error occurs
    raise 