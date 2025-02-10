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

### Development Pipeline
This project implements a four-stage pipeline using Gemma-2b. Each stage corresponds to specific implementation files:

#### Stage 1: Dialogue Generation
- Implementation: `src/data/generation/automation.py`
- Inputs:
  - Configuration: `data/config/automation.csv`
  - Prompts: `data/prompts/*.json`
- Output: Generated dialogue files in `data/dialogue/raw/`
- Note: Detailed documentation about our prompt engineering system and dialogue generation strategy can be found in `data/prompts/README.md`

#### Stage 2: Quality Assessment
- Implementation: `src/data/quality_check/dialogue_quality_check.py`
- Input: Dialogue files from `data/dialogue/raw/`
- Action: Moves low-rated dialogues to `data/dialogue/low_rated/`

#### Stage 3: Data Preparation
- Implementation: `src/data/dataset_preparation/dialogue_extractor.py`
- Input: Dialogue files from `data/dialogue/raw/`
- Output: `data/dialogue/processed/kaggle_model.json`

#### Stage 4: Model Training
- Implementation: `src/models/training/train.py`
- Input: `data/dialogue/processed/kaggle_model.json`
- Output: Model artifacts in `models/kaggle_model/` 

### Implementation Transparency
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

### Hardware Specifications for Training Environment
- GPU: 2x NVIDIA Tesla T4 (15GB VRAM each) or equivalent
- CPU: Intel Xeon CPU @ 2.00GHz (4 cores)
- RAM: 32GB (30GB available recommended)
- Storage: 20GB+ free space

Note: Successfully tested on Kaggle's T4 x2 environment. Lower specifications may work but are not guaranteed.

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

