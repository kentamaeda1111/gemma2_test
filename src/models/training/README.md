# Socratic Dialogue Training System

## Overview
A training system for fine-tuning language models on Socratic dialogue data using LoRA (Low-Rank Adaptation) techniques.

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
└── {MODEL_VERSION}/          # Named after input JSON (e.g., "1gouki")
    ├── model/               # Model files and checkpoints
    │   ├── checkpoint-{step}/
    │   └── training_config.json
    └── logs/               # Training logs and metrics
        ├── training_metrics.csv
        ├── training_summary.json
        └── metrics_plot.png
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
- Progress visualization

### 3. Quality Assessment
- Socratic pattern evaluation
- Response consistency metrics
- Conversation flow analysis
- Multi-metric tracking

### 4. Progress Tracking
- Detailed training logs
- Progress visualizations
- Checkpoint management
- Training summaries

## Monitoring

### Training Progress
The system generates real-time visualizations and logs:
- Loss curves
- Learning rate schedules
- Style consistency scores
- Dialogue flow metrics

### Output Files
- training_metrics.csv: Detailed metrics history
- training_summary.json: Final training results
- metrics_plot.png: Training progress visualization

## Note
This training system is optimized for fine-tuning language models for Socratic dialogue generation. It uses quantization and LoRA techniques to maintain efficiency while achieving high-quality results. 