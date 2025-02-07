# 1. Environment Setup and Imports
# 1.1 Import Dependencies
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

# 1.2 Global Constants and Environment Variables
# Define global constants
DIALOGUE_JSON_PATH = "data/dialogue/processed/final.json"  # Path to dialogue JSON file
MAX_SEQUENCE_LENGTH = 256  # Maximum number of tokens per dialogue
MAX_TOKENIZE_LENGTH = 256  # Maximum token length during tokenization

# Environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["WANDB_DISABLED"] = "true"

# API key
try:
    # Get API key
    api_keys = get_api_keys()
    huggingface_token = api_keys['huggingface_api_key']
    
    # Set Hugging Face API key
    os.environ["HUGGINGFACE_TOKEN"] = huggingface_token
    
    logging.info("Successfully loaded Hugging Face API key")
except Exception as e:
    logging.error(f"Error loading API keys: {str(e)}")
    raise

# 1.3 Logging Setup
# Create log directory
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

# Output settings
logging.info(f"Using dialogue file: {DIALOGUE_JSON_PATH}")
logging.info(f"Max sequence length: {MAX_SEQUENCE_LENGTH}")
logging.info(f"Max tokenize length: {MAX_TOKENIZE_LENGTH}")


# 2. Data Preprocessing Pipeline
# 2.1 Tokenizer Setup and Initialization
model_name = "google/gemma-2-2b-jpn-it"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    token=huggingface_token 
)

# 2.2 Data Validation Functions
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



def validate_dataset(dataset):
    """Validate dataset structure"""
    first_item = dataset[0]
    print("Validated first item structure:")
    print(f"Keys: {first_item.keys()}")
    print(f"input_ids type: {type(first_item['input_ids'])}")
    print(f"input_ids length: {len(first_item['input_ids'])}")
    return dataset


# 2.3 Data Set Preparation Function
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
                
            # Construct conversation in user->model order
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

# 2.4 Data Processing Pipeline Function
def tokenize_function(examples):
    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_TOKENIZE_LENGTH,      # Use global setting
        padding='max_length',
        add_special_tokens=True,
        return_tensors=None,
    )
    return result

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


# 2.5 Data Set Preparation
# Prepare base dataset
dataset = prepare_dataset()
logging.info(f"Total dataset size: {len(dataset)}")

# Validate dataset structure
print("Dataset structure:")
print(dataset[0])  # Display first element
print("\nDataset features:")
print(dataset.features)

# Optimize dataset batch processing
dataset = dataset.select(range(len(dataset))).shuffle(seed=42)


# 変更前
# Optimize dataset processing
# tokenized_dataset = dataset.map(
#     tokenize_function,
#     batched=True,
#     batch_size=32,  
#     num_proc=4,     
#     load_from_cache_file=True,
#     desc="Tokenizing datasets",
#     remove_columns=dataset.column_names,
# )

# KAGGLE用に変更後
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

# Apply preprocessing
tokenized_dataset = tokenized_dataset.map(
    preprocess_function,
    batched=True,
    desc="Applying attention masking"
)

# Final dataset validation
tokenized_dataset = validate_dataset(tokenized_dataset)


# 3. Model Architecture
# 3.1 Quantization Setup (BitsAndBytes)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_storage=torch.uint8,
)

# 3.2 Basic Model Initialization
# 変更前
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     attn_implementation='eager',
#     token=huggingface_token  # API token added
# )
# KAGGLE用に変更後
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="balanced",
    torch_dtype=torch.float16,
    attn_implementation='sdpa',
    token=os.environ["HUGGINGFACE_TOKEN"],  
    max_memory={0: "4GiB", 1: "4GiB", "cpu": "24GB"}
)

# 3.3 LoRA Setup and Application
# LoRA parameter setup
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Create and initialize LoRA model
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 3.4 Model Optimization Setup
# Optimize cache setup
model.config.use_cache = False

# Data collator setup
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)



# 4. Training Framework
# 4.1 System Resource Monitoring
def log_memory_usage():
    """Log memory usage"""
    import psutil
    process = psutil.Process()
    logging.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def clear_memory():
    """Memory release function during training"""
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# 4.2 Metrics Calculation System
def compute_metrics(eval_preds):
    logits, labels = eval_preds  # Get logits and labels from eval_preds
    
    # Improve decoding process
    with torch.no_grad():
        logits = torch.tensor(logits).cpu()
        predictions = torch.argmax(logits, dim=-1)
        
        # Decode entire batch
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Add more detailed log output
        logging.info(f"Sample prediction: {decoded_preds[0][:100]}...")
        
        del logits, predictions  # Memory release
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
                
                # Check if sentence ends with a question
                ends_with_question = any(sent.endswith(p) for p in socratic_patterns['question_endings'])
                # Proper use of second person
                uses_proper_address = any(p in sent for p in socratic_patterns['address_patterns'])
                # Use of inquiry introduction
                uses_inquiry_lead = any(p in sent for p in socratic_patterns['inquiry_leads'])
                
                # Sentence score (emphasize ending with a question)
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

# 4.3 Training Callbacks
# Custom callback modification
class StyleCallback(TrainerCallback):
    def __init__(self):
        self.socratic_scores = []
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if 'eval_socratic_style' in metrics:
            self.socratic_scores.append(metrics['eval_socratic_style'])
            
            # Log detailed information
            logging.info(f"Step {state.global_step}:")
            logging.info(f"Socratic Style Score: {metrics['eval_socratic_style']:.3f}")
    
    def on_train_end(self, args, state, control, **kwargs):
        # Log overall evaluation
        avg_score = sum(self.socratic_scores) / len(self.socratic_scores) if self.socratic_scores else 0
        
        logging.info("Training Complete!")
        logging.info(f"Average Socratic Style Score: {avg_score:.3f}")

# TrainingMonitorCallback also modified
class TrainingMonitorCallback(TrainerCallback):
    def __init__(self):
        self.train_start_time = None
        self.metrics_history = {
            'step': [],
            'socratic_style': [],  # Metric name changed
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
        
        # Record metrics
        current_step = state.global_step
        
        # Record loss and learning_rate for all steps
        if 'loss' in logs:
            self.metrics_history['step'].append(current_step)
            self.metrics_history['epoch'].append(state.epoch)
            self.metrics_history['loss'].append(logs['loss'])
            self.metrics_history['learning_rate'].append(logs.get('learning_rate', None))
            self.metrics_history['socratic_style'].append(None)  # None for non-evaluation steps
        
        # Update socratic_style score in evaluation step
        if 'eval_socratic_style' in logs:
            # Update last entry (same step)
            if self.metrics_history['step'] and self.metrics_history['step'][-1] == current_step:
                self.metrics_history['socratic_style'][-1] = logs['eval_socratic_style']
            else:
                # Add new entry
                self.metrics_history['step'].append(current_step)
                self.metrics_history['epoch'].append(state.epoch)
                self.metrics_history['loss'].append(None)
                self.metrics_history['learning_rate'].append(None)
                self.metrics_history['socratic_style'].append(logs['eval_socratic_style'])
        
        # Save to CSV file
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.output_dir / 'training_metrics.csv', index=False)
        
        # Update graph every 100 steps
        if current_step % 100 == 0:
            self._plot_metrics()
            
    def _plot_metrics(self):
        """Plot learning metrics and save"""
        plt.figure(figsize=(15, 8))
        
        # Plot Loss - Exclude None
        plt.subplot(2, 2, 1)
        valid_steps_loss = [s for s, v in zip(self.metrics_history['step'], self.metrics_history['loss']) if v is not None]
        valid_loss = [v for v in self.metrics_history['loss'] if v is not None]
        if valid_steps_loss:
            plt.plot(valid_steps_loss, valid_loss, label='Loss')
            plt.title('Training Loss')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.legend()
        
        # Plot Learning Rate - Exclude None
        plt.subplot(2, 2, 2)
        valid_steps_lr = [s for s, v in zip(self.metrics_history['step'], self.metrics_history['learning_rate']) if v is not None]
        valid_lr = [v for v in self.metrics_history['learning_rate'] if v is not None]
        if valid_steps_lr:
            plt.plot(valid_steps_lr, valid_lr, label='LR')
            plt.title('Learning Rate')
            plt.xlabel('Step')
            plt.ylabel('Learning Rate')
            plt.legend()
        
        # Plot Socratic Style Score - Exclude None
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
        # Final learning result summary to save
        summary = {
            'training_duration': str(datetime.now() - self.train_start_time),
            'final_loss': self.metrics_history['loss'][-1] if self.metrics_history['loss'] else None,
            'best_socratic_score': max(filter(None, self.metrics_history['socratic_style'])) if self.metrics_history['socratic_style'] else None,
            'total_steps': len(self.metrics_history['step']),
            'final_epoch': self.metrics_history['epoch'][-1] if self.metrics_history['epoch'] else None
        }
        
        # Save summary as JSON file
        with open(self.output_dir / 'training_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Final graph to save
        self._plot_metrics()
        
        logging.info("Training Complete!")
        logging.info(f"Training duration: {summary['training_duration']}")
        
        # None check added
        if summary['final_loss'] is not None:
            logging.info(f"Final loss: {summary['final_loss']:.4f}")
        else:
            logging.info("Final loss: Not available")
        
        if summary['best_socratic_score'] is not None:
            logging.info(f"Best Socratic style score: {summary['best_socratic_score']:.4f}")
        else:
            logging.info("Best Socratic style score: Not available")

# 4.4 Custom Trainer Implementation
class CustomTrainer(Trainer):
    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        if self.state.global_step % 100 == 0:
            clear_memory()
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is not None:
            # Limit evaluation dataset to 100 samples
            eval_dataset = eval_dataset.select(range(min(100, len(eval_dataset))))
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

# 4.5 Training Setup
training_args = TrainingArguments(
    output_dir="./model",
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
    logging_dir="./model/logs",
    logging_strategy="steps",
    logging_steps=50,
    no_cuda=False,
    dataloader_num_workers=1, #kaggle用は1, 修正前は2
    report_to=[],
    run_name=None,
    per_device_train_batch_size=2, #kaggle用は2, 修正前は4
    per_device_eval_batch_size=1, #kaggle用は1, 修正前は2
    gradient_checkpointing=True,
    max_grad_norm=0.5,
    dataloader_pin_memory=True,
    save_total_limit=2, #kaggle用は2, 修正前は3
    fp16=True,
    optim="adamw_torch_fused",
    eval_accumulation_steps=4, #kaggle用は4, 修正前は8
    load_best_model_at_end=True,
    metric_for_best_model="socratic_style",  
)

# 5. Execution and Model Management
# 5.1 Data Set Split and Validation
# Data set split
dataset_size = len(tokenized_dataset)
indices = np.random.permutation(dataset_size)
split_idx = int(dataset_size * 0.8)

# Create training and test datasets
train_dataset = tokenized_dataset.select(indices[:split_idx])
eval_dataset = tokenized_dataset.select(indices[split_idx:split_idx+100])  # Maximum 100 samples

# Record dataset size
logging.info(f"Training dataset size: {len(train_dataset)}")
logging.info(f"Evaluation dataset size: {len(eval_dataset)}")

# 5.2 Training Execution Control
# Trainer initialization
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[StyleCallback(), TrainingMonitorCallback()],
)

# Check memory state
log_memory_usage()

# Training execution
logging.info("Starting training...")
try:
    checkpoint_dir = "./model"
    resume_from_checkpoint = None
    
    # Checkpoint status and processing modification
    if os.path.exists(checkpoint_dir):
        logging.info("\nChecking checkpoint status...")
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint-")]
        if checkpoints:
            # Get latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            logging.info(f"Found latest checkpoint: {latest_checkpoint}")
            
            # Check checkpoint status
            state_path = os.path.join(checkpoint_path, "trainer_state.json")
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                current_epoch = state.get('epoch', 0)
                logging.info(f"\nCurrent training status:")
                logging.info(f"Current epoch: {current_epoch}")
                logging.info(f"Target epochs: {training_args.num_train_epochs}")
                
                # Exit safely if completed
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

    # Start learning (or resume)
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

    # Convert all settings
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
    # Checkpoint is also kept even on error
    raise 

