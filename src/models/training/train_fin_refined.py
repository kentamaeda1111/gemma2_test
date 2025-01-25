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
from src.utils.config import get_api_keys  # 追加

# グローバル設定を更新
DIALOGUE_JSON_PATH = "data/dialogue/processed/kaggle_model_processed.json"  # 対話データのパスを更新
MODEL_OUTPUT_DIR = "models/gemma_refined"  # モデル出力ディレクトリを更新
LOG_DIR = f"{MODEL_OUTPUT_DIR}/logs"  # ログディレクトリを更新
MAX_SEQUENCE_LENGTH = 256  # 1つの対話の最大トークン数

# API keyの取得を追加
try:
    api_keys = get_api_keys()
    huggingface_token = api_keys['huggingface_api_key']
except Exception as e:
    logging.error(f"Failed to get API keys: {str(e)}")
    raise

# 設定のログ出力
logging.info(f"Using dialogue file: {DIALOGUE_JSON_PATH}")
logging.info(f"Max sequence length: {MAX_SEQUENCE_LENGTH}")

# Environment variables and warning settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore", category=FutureWarning)
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
    trust_remote_code=True,
    token=huggingface_token  # トークンを追加
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
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation='eager',
    token=huggingface_token  # トークンを追加
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
        max_length=256,        # 最大長を256に制限
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
# メモリ使用量を監視するログを追加
def log_memory_usage():
    import psutil
    process = psutil.Process()
    logging.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

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

# ログディレクトリを作成
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_DIR}/training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Update training arguments
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    num_train_epochs=3,  # エポック数を減らす
    logging_dir=LOG_DIR,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=1,  # バッチサイズを1に削減
    per_device_eval_batch_size=1,   # 評価用バッチサイズも1に削減
    gradient_accumulation_steps=4,  # 勾配蓄積を追加
    gradient_checkpointing=True,    # 勾配チェックポイントを有効化
    max_grad_norm=0.5,
    dataloader_pin_memory=True,
    save_total_limit=2,            # 保存するチェックポイント数を制限
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
        self.train_start_time = None
        self.metrics_history = {
            'step': [],
            'style_consistency': [],
            'dialogue_flow': [],
            'combined_score': [],
            'loss': [],
            'learning_rate': [],
            'epoch': []
        }
        self.output_dir = Path(f"{MODEL_OUTPUT_DIR}/training_progress")  # 出力ディレクトリを更新
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = datetime.now()
        log_memory_usage()
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # Record metrics
        self.metrics_history['step'].append(state.global_step)
        self.metrics_history['epoch'].append(state.epoch)
        self.metrics_history['loss'].append(logs.get('loss', None))
        self.metrics_history['learning_rate'].append(logs.get('learning_rate', None))
        self.metrics_history['style_consistency'].append(logs.get('eval_style_consistency', None))
        self.metrics_history['dialogue_flow'].append(logs.get('eval_dialogue_flow', None))
        self.metrics_history['combined_score'].append(logs.get('eval_combined_score', None))
        
        # Save to CSV file
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.output_dir / 'training_metrics.csv', index=False)
        
        # Update graph every 100 steps
        if state.global_step % 100 == 0:
            self._plot_metrics()
            
    def _plot_metrics(self):
        """Plot learning metrics and save"""
        plt.figure(figsize=(15, 10))
        
        # Loss
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics_history['step'], self.metrics_history['loss'], label='Loss')
        plt.title('Training Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        
        # Learning Rate
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics_history['step'], self.metrics_history['learning_rate'], label='LR')
        plt.title('Learning Rate')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.legend()
        
        # Style and Flow Scores
        plt.subplot(2, 2, 3)
        valid_steps = [s for s, v in zip(self.metrics_history['step'], self.metrics_history['style_consistency']) if v is not None]
        valid_style = [v for v in self.metrics_history['style_consistency'] if v is not None]
        valid_flow = [v for v in self.metrics_history['dialogue_flow'] if v is not None]
        
        if valid_steps:
            plt.plot(valid_steps, valid_style, label='Style Consistency')
            plt.plot(valid_steps, valid_flow, label='Dialogue Flow')
            plt.title('Evaluation Metrics')
            plt.xlabel('Step')
            plt.ylabel('Score')
            plt.legend()
        
        # Combined Score
        plt.subplot(2, 2, 4)
        valid_combined = [v for v in self.metrics_history['combined_score'] if v is not None]
        if valid_steps:
            plt.plot(valid_steps, valid_combined, label='Combined Score')
            plt.title('Combined Evaluation Score')
            plt.xlabel('Step')
            plt.ylabel('Score')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_progress.png')
        plt.close()
    
    def on_train_end(self, args, state, control, **kwargs):
        # Save final learning result summary
        summary = {
            'training_duration': str(datetime.now() - self.train_start_time),
            'final_loss': self.metrics_history['loss'][-1] if self.metrics_history['loss'] else None,
            'best_combined_score': max(filter(None, self.metrics_history['combined_score'])) if self.metrics_history['combined_score'] else None,
            'total_steps': len(self.metrics_history['step']),
            'final_epoch': self.metrics_history['epoch'][-1] if self.metrics_history['epoch'] else None
        }
        
        # Save summary as JSON file
        with open(self.output_dir / 'training_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save final graph
        self._plot_metrics()
        
        logging.info("Training Complete!")
        logging.info(f"Training duration: {summary['training_duration']}")
        # None check added
        if summary['final_loss'] is not None:
            logging.info(f"Final loss: {summary['final_loss']:.4f}")
        else:
            logging.info("Final loss: Not available")
        if summary['best_combined_score'] is not None:
            logging.info(f"Best combined score: {summary['best_combined_score']:.4f}")
        else:
            logging.info("Best combined score: Not available")

# Split dataset into training and evaluation sets
dataset_size = len(tokenized_dataset)
indices = np.random.permutation(dataset_size)
split_idx = int(dataset_size * 0.8)

train_dataset = tokenized_dataset.select(indices[:split_idx])
# Limit evaluation dataset size
eval_dataset = tokenized_dataset.select(indices[split_idx:split_idx+50])  

logging.info(f"Training dataset size: {len(train_dataset)}")
logging.info(f"Evaluation dataset size: {len(eval_dataset)}")

# Add memory cleanup
def clear_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
class CustomTrainer(Trainer):
    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        if self.state.global_step % 50 == 0:  # より頻繁にメモリをクリア
            clear_memory()
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is not None:
            eval_dataset = eval_dataset.select(range(min(50, len(eval_dataset))))  
        clear_memory()  
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
    checkpoint_dir = MODEL_OUTPUT_DIR  # チェックポイントディレクトリを更新
    resume_from_checkpoint = None
    
    # Checkpoint status and processing
    if os.path.exists(checkpoint_dir):
        print("\nChecking checkpoint status...")  # Added
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint-")]
        if checkpoints:
            # Get latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"Found latest checkpoint: {latest_checkpoint}")  # Added
            
            # Check checkpoint status
            state_path = os.path.join(checkpoint_path, "trainer_state.json")
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                current_epoch = state.get('epoch', 0)
                print(f"\nCurrent training status:")  # Added
                print(f"Current epoch: {current_epoch}")  # Added
                print(f"Target epochs: {training_args.num_train_epochs}")  # Added
                
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
                # 自動的にyesを選択
                logging.info("Automatically continuing with training...")
        else:
            logging.warning("Checkpoint directory exists but no checkpoints found.")
            # 自動的にyesを選択
            logging.info("Automatically continuing with training...")

    # Start training (or resume)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logging.info("Training completed successfully!")
    
    # Save settings (as JSON)
    with open(os.path.join(MODEL_OUTPUT_DIR, "training_config.json"), "w", encoding="utf-8") as f:  # 設定ファイルの保存先を更新
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    # Save model
    trainer.save_model()
    # Save settings
    model.config.save_pretrained(MODEL_OUTPUT_DIR)  # モデル設定の保存先を更新
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)  # トークナイザーの保存先を更新
    logging.info("Model and configuration saved successfully!")

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    # Checkpoints are preserved even if an error occurs
    raise 