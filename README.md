# Japanese-Speaking Socratic Gemma

## Supplementary Repository for Kaggle Competition
This repository provides comprehensive implementation details and additional resources for the [Japanese-Speaking Socratic Gemma](https://www.kaggle.com/code/kentamaeda/japanese-speaking-socratic-gemma) Kaggle notebook. While the Kaggle notebook demonstrates the core inference functionality, this repository contains the complete codebase including dialogue generation, quality assessment, and training components.

Note: This repository is shared under the MIT License for reference purposes and is not actively maintained.

## Complete Implementation Details
### Development Pipeline
Our project implements a five-stage pipeline to create a Japanese-speaking Socratic dialogue model using Gemma-2b:

1. **Dialogue Generation**: Automated creation of Socratic dialogues using Claude API
2. **Quality Assessment**: Evaluation of generated dialogues for Socratic method adherence
3. **Data Preparation**: Processing of approved dialogues into training format
4. **Model Training**: Fine-tuning of Gemma-2b with processed dialogues
5. **Model Inference**: Deployment of trained model for interactive dialogue

### Repository Structure

project_root/
├── src/                      # Source code for all pipeline components
├── data/                     # Configuration, prompts, and dialogue data
└── models/                   # Trained model checkpoints and logs

Each component is documented in its respective directory's README.md file.

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
Note: All prompts are in Japanese. See [data/prompts/README.md](data/prompts/README.md) for English documentation.

## System Requirements

### Software Dependencies
- Python: 3.10+
- Latest pip version
- Compatible with Windows 10/11, macOS, and Linux

### Hardware Specifications

#### Training Environment
- GPU: NVIDIA GPU with 24GB+ VRAM (A5000/A6000/A100)
- RAM: 32GB minimum (24GB usage + 8GB buffer)
- Storage: 50GB+ free space
- Training Duration: ~2.5 hours on A100 GPU

#### Inference Environment
- GPU: NVIDIA GPU with 8GB+ VRAM
- RAM: 16GB minimum
- Storage: 20GB+ free space

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
python -m src.models.inference.test

