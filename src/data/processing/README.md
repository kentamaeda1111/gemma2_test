# Data Processing Modules

## Overview
This package contains modules for processing and evaluating Socratic dialogue data.

## Modules

### 1. Dialogue Quality Check
A system for evaluating the quality of generated Socratic dialogues.

#### Key Features
- Analyzes dialogue files for Socratic patterns
- Scores dialogues on tone and logical consistency (0-4 scale)
- Manages evaluation results in CSV format
- Identifies and segregates low-quality dialogues
- Generates statistical analysis reports

#### Model Settings
```python
MODEL_NAME = "claude-3-5-sonnet-20241022"
MAX_TOKENS = 3000
TEMPERATURE = 0.3
BATCH_SIZE = 1  # Number of files to process in each batch
```

#### Usage
```bash
# From the project root directory
python -m src.data.processing.dialogue_quality_check
```

#### Requirements
- Claude API key (set in `.env` file as CLAUDE_API_KEY_QUALITY)
- Input files in `data/dialogue/raw/`
- `automation.csv` with required columns

#### Output
- Updates `automation.csv` with evaluation scores
- Moves low-rated files to `data/dialogue/low_rated/`
- Generates analysis reports

### 2. Dialogue Extractor
A tool for extracting and reformatting dialogue segments for model training.

#### Key Features
- Extracts specific dialogue segments
- Reformats dialogues into Gemma-2 compatible format
- Manages output in JSON format
- Supports batch processing
- Configurable extraction settings

#### Configuration Settings
Edit these variables in `dialogue_extractor.py`:
```python
# Output settings
OUTPUT_FILE = '1gouki.json'       # Name of the output JSON file
PREPEND_KEYWORD = "ソクラテスさん。"  # Text to prepend to each first utterance

# Extraction range settings
EXTRACTION_SETTINGS = [
    {'start': 1, 'end': 2},      # Extract turns 1-2
    {'start': 3, 'end': 4},      # Extract turns 3-4
    # Add more ranges as needed...
]
```

#### Usage
```bash
# From the project root directory
python -m src.data.processing.dialogue_extractor
```

#### Configuration
Edit the following settings in `dialogue_extractor.py`:
- `OUTPUT_FILE`: Name of the output JSON file
- `PREPEND_KEYWORD`: Optional text to prepend to utterances
- `EXTRACTION_SETTINGS`: Array of start/end indices for extraction

#### Input/Output
- Input: Text files from `data/dialogue/raw/`
- Output: JSON files in `data/dialogue/processed/`
  - Format: Gemma-2 compatible dialogue format
  - Each extraction range creates a separate dialogue entry
  - Maintains source file tracking

## Project Structure
```
src/data/processing/
├── dialogue_quality_check.py  # Quality evaluation system
├── dialogue_extractor.py      # Data extraction and formatting
└── README.md                  # This file

data/
├── dialogue/
│   ├── raw/                  # Input dialogue files
│   ├── low_rated/           # Low-quality dialogues
│   └── processed/           # Processed JSON files
└── config/
    └── automation.csv       # Configuration and results tracking
``` 