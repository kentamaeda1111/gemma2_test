# Socratic AI Chat Interface

## Overview
An interactive chat interface for the fine-tuned Socratic dialogue model, supporting both Jupyter notebook widgets and console-based interactions.

## Configuration Parameters

### Global Settings
```python
MODEL_VERSION = "kaggle_model"           # Model version name
CHECKPOINT_NUMBER = "1980"         # Checkpoint to load
MAX_HISTORY = 5                    # Number of conversation turns to remember
BASE_MODEL = "google/gemma-2-2b-jpn-it"  # Base model name
```

### Generation Settings
```python
generation_config = {
    "max_new_tokens": 256,        # Maximum length of generated response
    "do_sample": True,            # Enable sampling
    "temperature": 0.7,           # Creativity level
    "top_p": 0.9,                # Nucleus sampling parameter
    "repetition_penalty": 1.1,    # Penalty for repeating tokens
}
```

## Requirements
- torch
- transformers
- peft
- IPython>=8.0.0 (for both console and Jupyter interfaces)
- ipywidgets>=8.0.0 (for Jupyter interface)
- Hugging Face token (set in `.env` as HUGGINGFACE_API_KEY)

### Hardware Requirements
- **Recommended**: NVIDIA GPU with 8GB+ VRAM
  - Peak VRAM usage: ~6GB
  - Additional VRAM buffer: 2GB
- **Minimum**: 16GB+ RAM for CPU-only execution

**Note**: While the model can run in CPU-only environments, inference speed will be 
significantly slower (several minutes per response vs. seconds on GPU). 
GPU execution is strongly recommended for optimal performance.

## Usage

### 1. Setup
1. Ensure model checkpoint exists at:
   ```
   models/{MODEL_VERSION}/model/checkpoint-{CHECKPOINT_NUMBER}
   ```
2. Set up environment:
   ```bash
   pip install -r requirements.txt
   ```
3. Create `.env` file with your Hugging Face token:
   ```env
   HUGGINGFACE_API_KEY="your_token_here"
   ```

### 2. Running
```bash
# From the project root directory
python -m src.models.inference.test
```

### 3. Interface Options

#### Jupyter Notebook
When run in a Jupyter environment, provides an interactive widget interface with:
- Message history display
- Input text box
- Send button
- Real-time response updates

#### Console
When run in a terminal, provides a console-based interface with:
- Command line input
- Message history display
- Exit commands: 'quit', 'exit', 'bye'

## Project Structure
```
src/models/inference/
├── test.py          # Main chat interface implementation
└── README.md        # This documentation

models/
└── {MODEL_VERSION}/
    └── model/
        └── checkpoint-{CHECKPOINT_NUMBER}/  # Model checkpoint files
```

## Features
- Automatic requirements checking and installation
- Dynamic UI selection based on environment
- Conversation history management
- Proper message turn alternation
- Error handling and logging
- Support for both local and Kaggle environments

## Initial Conversation

### Design Philosophy
The system implements a fixed initial dialogue pattern that embodies core Socratic principles:
- Focuses on the concept of "self" as a universal yet deeply philosophical topic
- Sets up the Socratic method through a carefully crafted opening question
- Establishes the appropriate tone and relationship between Socrates and the dialogue partner

### Technical Implementation
The system initializes with a predefined conversation context:
```python
initial_user_msg = "あなたは古代ギリシャの哲学者ソクラテスです。今日は何について話しますか？"
initial_model_msg = (
    "やぁ、よく来てくれたね。今日は『自分』という、これ以上ないほど身近な存在でありながら、"
    "あまり話すことのないトピックについて話そうではないか。..."
)

# These messages are added to conversation history but not displayed
chatai._update_history({"role": "user", "content": initial_user_msg})
chatai._update_history({"role": "model", "content": initial_model_msg})
```

### User Experience
- The initial user prompt is used internally to set context
- Users see only Socrates' opening message about exploring the concept of "self"
- This design creates a natural entry point while maintaining the philosophical depth

## Note
This interface is designed to work with the fine-tuned Gemma-2b model trained on Socratic dialogues. Make sure you have the correct model checkpoint and Hugging Face access token before running.
