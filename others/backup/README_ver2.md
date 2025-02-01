# Japanese-Speaking Scratic Gemma

## About This Repository
This repository serves as a place to store additional explanations, supplementary information, and corrections for the content submitted to the Kaggle competition  [Kaggle notebook](https://www.kaggle.com/code/kentamaeda/japanese-speaking-socratic-gemma).
While the Kaggle notebook contains the complete inference code (test.py), it only includes partial code for other components. Therefore, this repository provides all the code, including the prompts and JSON data used, enabling complete reproduction of the process used to create the model used in Kaggle. Additionally, sections that couldn't be fully explained due to time constraints and areas requiring modifications will be gradually updated and corrected in the future.

Note: This repository is shared under the MIT License for reference purposes only. It is not actively maintained.

## Full Project Documentation

### Repository Structure

```
project_root/
├── src/                      # Source code for all pipeline components
├── data/                     # Configuration, prompts, and dialogue data
└── models/                   # Trained model checkpoints and logs
```
#### Project Overview
This project implements a pipeline for creating a Japanese-speaking Socratic dialogue model using Gemma-2b, consisting of five main stages:

1. **Dialogue Generation**: Automated creation of Socratic dialogues using Claude API (`automation.py`)
2. **Quality Assessment**: Evaluation of generated dialogues for adherence to Socratic method and dialogue quality (`dialogue_quality_check.py`)
3. **Data Preparation**: Processing and formatting of approved dialogues into training-ready format (`dataset_preparation.py`)
4. **Model Training**: Fine-tuning Gemma-2b model with the processed dialogue data (`train.py`)
5. **Model Inference**: Testing and deploying the trained model for interactive dialogue (`test.py`)

#### Data and Model Transparency

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

### Technical Requirements

#### Basic Requirements
- Python: 3.10 or higher
- Operating System: Windows 10/11, macOS, or Linux
- Package Manager: pip (latest version)

#### Hardware Requirements
##### For Training
- GPU: NVIDIA GPU with at least 24GB VRAM (e.g., A5000, A6000, or A100)
  - Peak VRAM usage during training: ~20GB (with 4-bit quantization)
  - Additional VRAM buffer recommended: 4GB
- RAM: 32GB minimum
  - Peak RAM usage during training: ~24GB
  - Additional RAM buffer recommended: 8GB
- Storage: 50GB+ free space for model checkpoints and training data
- Training Time: ~2.5 hours on A100 GPU

##### For Inference
- GPU: NVIDIA GPU with 8GB+ VRAM
  - Peak VRAM usage during inference: ~6GB
  - Additional VRAM buffer recommended: 2GB
- RAM: 16GB+
  - Peak RAM usage during inference: ~8GB
  - Additional RAM buffer recommended: 8GB
- Storage: 20GB+ free space

Note: These requirements are based on using LoRA for fine-tuning and 4-bit quantization (QLoRA). Running the model in full precision or without quantization would require significantly more memory.
Note: For inference, the model can be run with reduced precision (bfloat16) to decrease memory requirements. When loaded in bf16, it consumes approximately 8GB of VRAM for the 2b model.

### Setup & Usage

#### API Configuration
Project requires the following API keys:
- `CLAUDE_API_KEY_1`, `CLAUDE_API_KEY_2`, `CLAUDE_API_KEY_QUALITY`: For dialogue generation and quality assessment
- `HUGGINGFACE_API_KEY`: For model access

Keys can be configured via:
- Local: `.env` file (template provided in `.env.template`)
- Kaggle: Using Kaggle Secrets or environment variables
- Colab: Using built-in secure form prompt or environment variables

#### Quick Start

```bash
git clone https://github.com/kentamaeda1111/gemma2_test.git
cd gemma2_test
pip install -r requirements.txt
cp .env.template .env  # Configure API keys

# Run pipeline
python -m src.data.generation.automation
python -m src.data.quality_check.dialogue_quality_check
python -m src.data.dataset_preparation.dialogue_extractor
python -m src.models.training.train
python -m src.models.inference.test
```

