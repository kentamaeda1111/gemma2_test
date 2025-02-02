# Japanese-Speaking Socratic Gemma

## Supplementary Repository for Kaggle Competition
This repository provides the complete pipeline and resources necessary to replicate the model training process behind [Japanese-Speaking Socratic Gemma](https://www.kaggle.com/code/kentamaeda/japanese-speaking-socratic-gemma), our Kaggle notebook submission to the [Google - Unlock Global Communication with Gemma](https://www.kaggle.com/competitions/gemma-language-tuning) competition. While the Kaggle notebook only includes the inference code, this repository contains the full end-to-end training pipeline to enable complete model reproduction, including implementations of dialogue generation, quality assessment, data preparation, and model training.

Note: The repository is shared under the MIT License for reference purposes. While we may occasionally update it to address necessary fixes in the Kaggle notebook, it is not actively maintained.

## Repository Overview

### Repository Structure
```
project_root/
├── src/                      # Source code for all pipeline components
├── data/                     # Configuration, prompts, and dialogue data
└── models/                   # Trained model checkpoints and logs
```

### 1. Development Pipeline
This project implements a four-stage pipeline using Gemma-2b. Each stage corresponds to specific implementation files:

#### Stage 1: Dialogue Generation
- Implementation: `src/data/generation/automation.py`
- Documentation: `src/data/generation/README.md`
- Inputs:
  - Configuration: `data/config/automation.csv`
  - Prompts: `data/prompts/*.json`
- Output: Generated dialogue files in `data/dialogue/raw/`

#### Stage 2: Quality Assessment
- Implementation: `src/data/quality_check/dialogue_quality_check.py`
- Documentation: `src/data/quality_check/README.md`
- Input: Dialogue files from `data/dialogue/raw/`
- Action: Moves low-rated dialogues to `data/dialogue/low_rated/`

#### Stage 3: Data Preparation
- Implementation: `src/data/dataset_preparation/dialogue_extractor.py`
- Documentation: `src/data/dataset_preparation/README.md`
- Input: Dialogue files from `data/dialogue/raw/`
- Output: `data/dialogue/processed/kaggle_model.json`

#### Stage 4: Model Training
- Implementation: `src/models/training/train.py`
- Documentation: `src/models/training/README.md`
- Input: `data/dialogue/processed/kaggle_model.json`
- Output: Model artifacts in `models/kaggle_model/` 

### 2. Core Technical Documentation
Two comprehensive documents in the `data/` directory detail our approach:

- `data/README.md`: Training Data Strategy
  - Data generation policies and rationale
  - Quality assurance methodology
  - System prompt integration approach
  - Detailed model configuration statistics
  - Training data characteristics analysis

- `data/prompts/README.md`: Prompt Engineering System
  - Core design philosophy and architecture
  - Detailed prompt categories and templates
  - Assistant and user prompt implementations
  - Initial question design strategy
  - Anti-overfitting measures

### 3. Implementation Transparency
This repository intentionally includes key files that would typically be excluded:

1. `data/config/automation.csv`: Configuration parameters for dialogue generation
   - Temperature and maximum turns settings
   - Quality metrics thresholds

2. `data/dialogue/processed/kaggle_model.json`: Complete training dataset
   - Enables verification and reproduction of results

3. `models/kaggle_model/`: Model artifacts including:
   - Checkpoints
   - Training logs

4. `data/prompts/*.json`: Dialogue generation prompts
   - `assistant_system_prompt/`: Socrates role prompts
   - `user_system_prompt/`: Student role prompts
   - `questions.json`: Initial philosophical questions

## System Requirements

### Software Dependencies
- Python: 3.10+
- Latest pip version
- Compatible with Windows 10/11, macOS, and Linux

### Hardware Specifications for Training Environment
- GPU: NVIDIA GPU with 24GB+ VRAM
- RAM: 32GB minimum
- Storage: 50GB+ free space

## Setup Instructions

### Environment Configuration
Required API keys:
- `CLAUDE_API_KEY_1`, `CLAUDE_API_KEY_2`, `CLAUDE_API_KEY_QUALITY`
- `HUGGINGFACE_API_KEY`

Configure via:
- Local: `.env` file
- Kaggle: Secrets/environment variables
- Colab: Secure form/environment variables

### Quick Start Guide
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

