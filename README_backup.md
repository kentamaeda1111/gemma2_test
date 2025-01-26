# Japanese-Speaking Socratic Gemma

A Kaggle competition submission demonstrating fine-tuning of Gemma-2b for Japanese Socratic dialogue generation. For the actual competition submission and results, please see the [Kaggle notebook](https://www.kaggle.com/code/kentamaeda/japanese-speaking-socratic-gemma).

This repository contains the supporting code, shared under the MIT License for reference purposes only. It is not actively maintained.

## Project Overview & Components

This project implements a Japanese-speaking Socratic dialogue system using Gemma-2b, consisting of three main pipelines:

1. **Dialogue Generation** ([src/data/generation](src/data/generation/README.md))
   - Automated dialogue creation using Claude API
   - Dual AI system (User & Socrates roles)
   - Configurable conversation parameters
   - Quality control metrics

2. **Data Processing** ([src/data/processing](src/data/processing/README.md))
   - Quality assessment and filtering
   - Format conversion for Gemma
   - Dialogue segmentation and cleaning

3. **Model Pipeline** 
   - Training: Fine-tuning with LoRA ([src/models/training](src/models/training/README.md))
   - Inference: Interactive chat interface ([src/models/inference](src/models/inference/README.md))

Supporting documentation:
- Shared utilities: [src/utils](src/utils/README.md)
- Prompt system: [data/prompts](data/prompts/README.md)
  - Dialogue generation prompts
  - Japanese-English documentation
  - Configuration guidelines

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
├── data/
│   ├── config/     
│   │   ├── automation_kaggle_model.csv   # Configuration used for the Kaggle competition submission
│   │   └── automation.csv                # Controls dialogue generation (turns, temperature, etc.)
│   │ 
│   ├── dialogue/                         # Generated and processed dialogue data
│   │   ├── low_rated/                    # Dialogues that didn't meet quality standards
│   │   ├── processed/                    # Final processed dialogues ready for training
│       │   └── kaggle_model.json         # Training data used for Kaggle competition model
│   │   └── raw/                         # Initial generated dialogues
│   │
│   └── prompts/                         # System prompts used to generate Kaggle competition training data
│       ├── assistant_system_prompt/      # Prompts for Socrates role in training dialogues
│       │   ├── assistant_system_prompt.json
│       │   ├── response.json
│       │   └── update.json
│       ├── user_system_prompt/          # Prompts for user role in training dialogues
│       │   ├── user_system_prompt.json
│       │   ├── others.json
│       │   ├── persona.json
│       │   └── transform.json
│       ├── questions.json               # Philosophical questions used to initiate training dialogues
│       └── README.md
│
├── models/                              # Trained model checkpoints and logs
│   └── kaggle_model/                    # Model files from successful Kaggle competition submission
│       ├── logs/
│       └── model/
│ 
├── src/
│   ├── data/
│   │   ├── generation/                  # Dialogue generation system using Claude API
│   │   │   ├── automation.py
│   │   │   └── README.md
│   │   │
│   │   └── processing/                  # Quality assessment and data formatting tools
│   │       ├── dialogue_extractor.py
│   │       ├── dialogue_quality_check.py
│   │       └── README.md
│   │
│   ├── models/ 
│   │   ├── inference/                   # Testing and chat interface
│   │   │   ├── test.py
│   │   │   └── README.md
│   │   │ 
│   │   └── training/                    # Model fine-tuning implementation
│   │       ├── train_fin_copy.py
│   │       └── README.md
│   │
│   └── utils/                          # Configuration utilities and shared functions
│       ├── config.py   
│       └── README.md
│ 
├── .env.template           
├── .gitignore             
├── README.md              
└── requirements.txt       
```



## Setup Requirements

### Basic Requirements


### Hardware Requirements

#### For Training
- GPU: NVIDIA GPU with at least 24GB VRAM (e.g., A5000, A6000, or A100)
  - Peak VRAM usage during training: ~20GB
  - Additional VRAM buffer recommended: 4GB
- RAM: 32GB minimum
  - Peak RAM usage during training: ~24GB
  - Additional RAM buffer recommended: 8GB
- Storage: 50GB+ free space for model checkpoints and training data

Note: These requirements are based on using LoRA for fine-tuning and 4-bit quantization (QLoRA). 

#### For Inference
- GPU: NVIDIA GPU with 8GB+ VRAM
  - Peak VRAM usage during inference: ~6GB
  - Additional VRAM buffer recommended: 2GB
- RAM: 16GB+
  - Peak RAM usage during inference: ~8GB
  - Additional RAM buffer recommended: 8GB
- Storage: 20GB+ free space

Note: For inference, the model can be run with reduced precision (bfloat16) to decrease memory requirements. When loaded in bf16, it consumes approximately 8GB of VRAM for the 2b model.
While the model can run on CPU-only environments, inference speed will be significantly slower compared to GPU execution 
(response generation may take several minutes per query on CPU vs. seconds on GPU).
For optimal performance, GPU execution is strongly recommended.

## Data and Model Transparency

For transparency and reproducibility purposes, this repository intentionally includes several key files that would typically be excluded:

1. `data/config/automation_kaggle_model.csv`: Contains the complete configuration parameters used for dialogue generation, including:
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
   - `assistant_system_prompt/`: Prompts for Socrates' role
   - `user_system_prompt/`: Prompts for the user's persona and responses
   - `questions.json`: The collection of philosophical questions used to initiate dialogues

Note: All prompt templates are written in Japanese. For detailed English documentation of the prompts, please see [data/prompts/README.md](data/prompts/README.md).
