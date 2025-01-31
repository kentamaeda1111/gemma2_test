# Dialogue Extractor

## Overview
A tool for extracting and reformatting dialogue segments for model training. This module processes raw dialogue files and converts them into a Gemma-2 compatible JSON format for training data preparation.

## Key Features
- Extracts specific dialogue segments based on configurable ranges
- Reformats dialogues into Gemma-2 compatible format
- Manages output in JSON format
- Supports batch processing
- Configurable extraction settings

## Configuration Settings
Edit these variables in `dialogue_extractor.py`:
```python
# Output settings
OUTPUT_FILE = 'test.json'       # Name of the output JSON file
PREPEND_KEYWORD = "あなたは古代ギリシャの哲学者ソクラテスです。"  # Text to prepend to each first utterance

# Extraction range settings
EXTRACTION_SETTINGS = [
    {'start': 1, 'end': 2},      # Extract turns 1-2
    {'start': 3, 'end': 4},      # Extract turns 3-4
    # Add more ranges as needed...
]
```

## Usage
```bash
# From the project root directory
python -m src.data.dataset_preparation.dialogue_extractor
```

## Input/Output
### Input
- Text files from `data/dialogue/raw/`
- Each file should contain dialogue logs in the format:
  ```
  === Dialogue Log ===
  User: ...
  Assistant: ...
  ```

### Output
- JSON files in `data/dialogue/processed/`
- Format: Gemma-2 compatible dialogue format
- Each extraction range creates a separate dialogue entry
- Maintains source file tracking

## Project Structure
```
src/data/dataset_preparation/
├── dialogue_extractor.py      # Data extraction and formatting
└── README.md                  # This file

data/
├── dialogue/
│   ├── raw/                  # Input dialogue files
│   └── processed/           # Processed JSON files
``` 