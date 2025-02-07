


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

