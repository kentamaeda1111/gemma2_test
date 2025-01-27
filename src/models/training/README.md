# Socratic Dialogue Training System

## Overview
A training system for fine-tuning language models on Socratic dialogue data using LoRA (Low-Rank Adaptation) techniques.

## Training Data Analysis
For the Kaggle competition submission, we used a carefully curated dataset (kaggle_model.json) with the following characteristics:

### Dataset Metrics
- **Scale**: 2,662 Socratic dialogue pairs
- **Token Distribution**:
  - Total tokens: 752,369
  - Per dialogue average: 282.6 tokens
    - Student messages: 169.4 tokens (average)
    - Socrates responses: 113.2 tokens (average)
  - Length range: 44-552 tokens per dialogue

This dataset was specifically designed to balance comprehensive philosophical discussions with model training efficiency, keeping individual dialogues within manageable token limits while maintaining meaningful Socratic interactions.

## Configuration Parameters

### Model Settings
```python
# Base model configuration
model_name = "google/gemma-2-2b-jpn-it"
MAX_SEQUENCE_LENGTH = 512

# 4-bit quantization settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# LoRA configuration
lora_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,          # Alpha scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Training Parameters
```python
training_args = TrainingArguments(
    num_train_epochs=30,
    learning_rate=8e-5,
    weight_decay=0.06,
    warmup_ratio=0.25,
    lr_scheduler_type="cosine_with_restarts",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    # ... other parameters
)
```

## Requirements
- torch
- transformers (from GitHub)
- peft>=0.7.1
- bitsandbytes>=0.43.2
- accelerate>=0.26.0
- datasets
- Hugging Face token (set in `.env` as HUGGINGFACE_API_KEY)

## Usage

### 1. Setup
1. Prepare dialogue data:
   ```
   data/dialogue/processed/1gouki.json  # Training data
   ```
2. Set up environment:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure model parameters in train.py if needed

### 2. Running
```bash
# From the project root directory
python -m src.models.training.train
```

### 3. Output Structure
```
models/
└── {MODEL_VERSION}/          # Named after input JSON (e.g., "test_short")
    ├── model/               # Model files and adapter config
    │   └── adapter_config.json
    ├── logs/               # Training logs
    │   └── training_log_{timestamp}.log
    └── training_progress/  # Training metrics and summary
        └── training_summary.json
```

## Features

### 1. Model Configuration
- 4-bit quantization for memory efficiency
- LoRA for efficient fine-tuning
- Dynamic attention masking
- Japanese language optimization

### 2. Training Management
- Custom evaluation metrics
- Style consistency tracking
- Dialogue flow monitoring
- Resource usage monitoring

### 3. Quality Assessment
- Socratic pattern evaluation
- Response consistency metrics
- Conversation flow analysis
- Multi-metric tracking

### 4. Progress Tracking
- Detailed training logs
- Resource usage tracking
- Checkpoint management
- Training summaries

## Monitoring

### Training Progress
The system logs the following metrics in real-time:
- Loss values and moving averages
- Learning rate changes
- Style consistency scores
- Dialogue flow metrics
- CPU/GPU resource usage
- Batch size tracking

### Output Files

#### 1. Training Logs (logs/training_log_{timestamp}.log)
- Detailed training progress
- Step-by-step metrics
- Evaluation results
- Resource usage statistics
- Error and warning messages

#### 2. Training Summary (training_progress/training_summary.json)
Contains final training results including:
- Training duration
- Final loss values
- Best combined score
- Total training steps
- Final epoch number
- Resource usage peaks:
  - Peak CPU RAM usage
  - Peak GPU VRAM usage
  - Peak GPU utilization
- Hardware specifications
- Batch size history
- Moving average loss tracking

## Note
This training system is optimized for fine-tuning language models for Socratic dialogue generation. It uses quantization and LoRA techniques to maintain efficiency while achieving high-quality results. The comprehensive logging system helps in monitoring training progress and resource usage for optimization purposes. 