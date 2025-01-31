# Dialogue Quality Check

## Overview
A system for evaluating the quality of generated Socratic dialogues. This module analyzes dialogue files for Socratic patterns and scores them based on tone and logical consistency.

## Key Features
- Analyzes dialogue files for Socratic patterns
- Scores dialogues on tone and logical consistency (0-4 scale)
- Manages evaluation results in CSV format
- Identifies and segregates low-quality dialogues
- Generates statistical analysis reports

## Quality Check Prompt
The system uses a carefully crafted Japanese prompt to evaluate dialogue quality. The prompt implements both quantitative and qualitative evaluation methods:

### 1. Evaluate Socratic Style (Score 0-4)
- Checks for Socratic linguistic patterns (e.g., "かね?", "だろうか?")
- Evaluates addressing patterns (e.g., "友よ", "君")
- Scoring criteria:
  - 0: Not Socratic at all
  - 1: No Socratic elements
  - 2: Generally good with minor issues
  - 3: Successfully Socratic
  - 4: Perfect Socratic style

### 2. Evaluate Logical Consistency (Score 0-4)
- Focuses on natural conversation flow
- Checks response appropriateness
- Scoring criteria:
  - 0: Incomprehensible responses
  - 1: Misaligned conversation
  - 2: Generally good with minor issues
  - 3: Natural conversation flow
  - 4: Excellent Socratic responses

### 3. Brief Comments
- Short feedback on each dialogue pair
- Highlights specific strengths or issues
- Maintains concise format

## Model Settings
```python
MODEL_NAME = "claude-3-5-sonnet-20241022"
MAX_TOKENS = 3000
TEMPERATURE = 0.3
BATCH_SIZE = 1  # Number of files to process in each batch
```

## Usage
```bash
# From the project root directory
python -m src.data.quality_check.dialogue_quality_check
```

## Requirements
- Claude API key (set in `.env` file as CLAUDE_API_KEY_QUALITY)
- Input files in `data/dialogue/raw/`
- `automation.csv` with required columns

## Output
- Updates `automation.csv` with evaluation scores
- Moves low-rated files to `data/dialogue/low_rated/`
- Generates analysis reports

## Project Structure
```
src/data/quality_check/
├── dialogue_quality_check.py  # Quality evaluation system
└── README.md                  # This file

data/
├── dialogue/
│   ├── raw/                  # Input dialogue files
│   └── low_rated/           # Low-quality dialogues
└── config/
    └── automation.csv       # Configuration and results tracking
``` 