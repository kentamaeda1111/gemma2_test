# Japanese-Speaking Socratic Gemma

A Kaggle competition submission demonstrating fine-tuning of Gemma-2b for Japanese Socratic dialogue generation. For the actual competition submission and results, please see the [Kaggle notebook](https://www.kaggle.com/code/kentamaeda/japanese-speaking-socratic-gemma).

This repository contains the supporting code, shared under the MIT License for reference purposes only. It is not actively maintained.

## Project Overview

This project showcases:
1. **Automated Dialogue Generation**: Using Claude API to create training data
2. **Data Processing & Quality Control**: Systematic evaluation and formatting of dialogues
3. **Model Training & Testing**: Fine-tuning and evaluating Gemma-2b for Socratic dialogue

## Components

1. **Generation Pipeline**:
   - Dialogue generation using Claude API
   - Dual AI system (Student & Socrates)
   - Configurable conversation parameters

2. **Processing Pipeline**:
   - Quality assessment and filtering
   - Format conversion for Gemma
   - Dialogue segmentation and cleaning

3. **Training & Testing Pipeline**:
   - Model fine-tuning with LoRA
   - Interactive chat interface
   - Performance evaluation metrics

## Setup & Usage

### API Configuration
Project requires the following API keys:
- `CLAUDE_API_KEY_1`, `CLAUDE_API_KEY_2`, `CLAUDE_API_KEY_QUALITY`: For dialogue generation and quality assessment
- `HUGGINGFACE_API_KEY`: For model access

Keys can be configured via:
- Local: `.env` file (template provided in `.env.template`)
- Kaggle: Using Kaggle Secrets or environment variables
- Colab: Using environment variables or built-in secure form prompt

### Quick Start

**Local Environment**
```bash
git clone https://github.com/kentamaeda1111/gemma2_test.git
cd gemma2_test
pip install -r requirements.txt
cp .env.template .env  # Configure API keys

# Run pipeline
python -m src.data.generation.automation
python -m src.data.processing.dialogue_quality_check
python -m src.data.processing.dialogue_extractor
python -m src.models.training.train
python -m src.models.inference.test
```

**Cloud Environment (Kaggle/Colab)**
```python
!git clone https://github.com/kentamaeda1111/gemma2_test
%cd gemma2_test
!pip install -r requirements.txt

# Configure keys via respective platform's method
# Run same pipeline as above
```

## Repository Structure

```
project_root/
├── src/
│   ├── data/
│   │   ├── generation/          
│   │   │   ├── automation.py
│   │   │   └── README.md
│   │   │
│   │   └── processing/         
│   │       ├── dialogue_quality_check.py
│   │       ├── dialogue_extractor.py
│   │       └── README.md
│   │
│   ├── models/                
│   │   ├── training/
│   │   │   ├── train_fin_copy.py
│   │   │   └── README.md
│   │   │
│   │   └── inference/
│   │       ├── test.py
│   │       └── README.md
│   │
│   └── utils/                 
│       ├── config.py   
│       └── README.md
│
├── data/
│   ├── config/                
│   │   └── automation.csv  
│   │ 
│   ├── prompts/              
│   │   ├── assistant_system_prompt/
│   │   │   ├── assistant_system_prompt.json
│   │   │   ├── response.json
│   │   │   └── update.json
│   │   ├── user_system_prompt/
│   │   │   ├── user_system_prompt.json
│   │   │   ├── others.json
│   │   │   ├── persona.json
│   │   │   └── transform.json
│   │   └── questions.json
│   │
│   └── dialogue/            
│       ├── raw/
│       ├── low_rated/
│       └── processed/
│
├── models/                  
│   └── [model_name]/
│       ├── model/
│       └── logs/
│
├── .env.template           
├── .gitignore             
├── README.md              
└── requirements.txt       
```

### Key Directories

- `src/data/generation/`: Dialogue generation system using Claude API
- `src/data/processing/`: Quality assessment and data formatting tools
- `src/models/training/`: Model fine-tuning implementation
- `src/models/inference/`: Testing and chat interface
- `src/utils/`: Configuration utilities and shared functions
- `data/config/automation.csv`: Controls dialogue generation parameters (turns, temperature, etc.)
- `data/prompts/`: System prompts for dialogue generation
  - `assistant_system_prompt/`: Prompts for Socrates role
  - `user_system_prompt/`: Prompts for student role
- `data/dialogue/`: Generated and processed dialogue data
  - `raw/`: Initial generated dialogues
  - `low_rated/`: Dialogues that didn't meet quality standards
  - `processed/`: Final processed dialogues ready for training
- `models/`: Trained model checkpoints and logs

For detailed documentation of each component, please refer to the README.md files in their respective directories:
- Generation system: [src/data/generation/README.md](src/data/generation/README.md)
- Data processing: [src/data/processing/README.md](src/data/processing/README.md)
- Model training: [src/models/training/README.md](src/models/training/README.md)
- Inference system: [src/models/inference/README.md](src/models/inference/README.md)
- Utilities: [src/utils/README.md](src/utils/README.md)

## Setup Requirements

### Basic Requirements
- Python: 3.10 or higher
- Operating System: Windows 10/11, macOS, or Linux
- Package Manager: pip (latest version)

### For Training
Note: Training large language models requires significant computational resources. Please refer to the official Gemma documentation for detailed hardware requirements. In our case, we used Google Colab Pro+ with A100 GPU for training.

### For Inference
- GPU: NVIDIA GPU with 8GB+ VRAM
- RAM: 16GB+
- Storage: 20GB+ free space
- API access:
  - Claude API (for dialogue generation)
  - Hugging Face (for model access)

Note: For inference, the model can be run with reduced precision (bfloat16) to decrease memory requirements. When loaded in bf16, it consumes approximately 8GB of VRAM for the 2b model.

## Data and Model Transparency

For transparency and reproducibility purposes, this repository intentionally includes several key files that would typically be excluded:

1. `data/config/automation.csv`: Contains the complete configuration parameters used for dialogue generation, including:
   - Temperature settings
   - Maximum turns/pairs
   - Prompt IDs
   - Quality metrics for each dialogue

2. `data/dialogue/processed/kaggle_model.json`: The actual training data fed into the model, allowing for:
   - Verification of training data quality
   - Understanding of dialogue patterns and structures
   - Reproduction of training results

3. `models/kaggle_model/`: The trained model outputs, including:
   - Model checkpoints
   - Training logs
   - Performance metrics

4. `data/prompts/*.json`: The complete set of prompt templates used for dialogue generation:
   - `assistant_system_prompt/`: Prompts for Socrates' role and behavior
   - `user_system_prompt/`: Prompts for the student's persona and responses
   - `questions.json`: The collection of philosophical questions used to initiate dialogues
   These prompts are crucial for understanding how the dialogues were structured and generated.

Note: All prompt templates are written in Japanese. For detailed English documentation of the prompts, please see [data/prompts/README.md](data/prompts/README.md).
