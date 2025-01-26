# Socratic Dialogue Generation System

## Overview
An automated system for generating Socratic-style dialogues using Claude AI models. The system manages conversations between two AI models - one acting as a student/user and another as a Socratic teacher.

## System Architecture

### 1. Dual AI Model System
- **AI1 (User/Student)**:
  - Implements diverse perspectives through 148 distinct personas
  - Temperature variations (0.3/0.7) for response style diversity
  - Total 296 unique dialogue patterns (148 personas × 2 temperatures)

- **AI2 (Socrates)**:
  - Maintains consistent Socratic methodology
  - Fixed personality and questioning style
  - Focuses on guiding self-discovery

### 2. Training Strategy
- **Dialogue Diversity**:
  - 74 philosophical topics for training
  - Fixed self-identity question for inference
  - Anti-overfitting through topic variation

- **Quality Control**:
  - Standardized templates
  - Consistent dialogue structure
  - Natural conversation flow

## Configuration Parameters

### Hard-coded Parameters in automation.py
These parameters are set directly in the code:

1. Model Settings:
```python
# AI1 (User) parameters
AI1_MODEL_NAME = "claude-3-5-sonnet-20241022"
AI1_MAX_TOKENS = 2048

# AI2 (Assistant) parameters
AI2_MODEL_NAME = "claude-3-5-sonnet-20241022"
AI2_MAX_TOKENS = 2048
```

### Configurable Parameters in automation.csv
Parameters that can be modified without code changes:

1. **Temperature Settings**:
   - `AI1_TEMPERATURE`: Controls user AI creativity (0.0-1.0)
   - `AI2_TEMPERATURE`: Controls Socrates AI consistency (0.0-1.0)

2. **Conversation Control**:
   - `MAX_MESSAGE_PAIRS`: History retention count
   - `MAX_TURNS`: Total conversation turns

3. **Prompt Selection**:
   - `INITIAL_QUESTION_ID`: Starting question ID
   - `USER_PROMPT_ID`: User AI system prompt ID
   - `ASSISTANT_PROMPT_ID`: Socrates AI system prompt ID
   - `OTHERS_ID`: Additional context ID
   - `PERSONA_ID`: Character personality ID
   - `TRANSFORM_ID`: Style transformation ID
   - `RESPONSE_ID`: Response pattern ID
   - `UPDATE_ID`: Update pattern ID

4. **Logging**:
   - `DIALOGUE_KEYWORD`: Log file identifier
   - `dialogue`: Output file path (auto-generated)

## Implementation Features

### 1. Prompt Management
- **Dynamic Loading**: JSON-based prompt configuration
- **Placeholder System**: Variable substitution in templates
- **Modular Design**: Separate prompts for each AI role

### 2. Conversation Control
- **Context Management**: Maintains dialogue history
- **Turn Limiting**: Controls conversation length
- **Memory Management**: Efficient history handling

### 3. Logging System
- **Detailed Records**: Comprehensive dialogue logs
- **Metadata Tracking**: Session information storage
- **Version Control**: Timestamp-based tracking

## Installation & Setup

### Dependencies
```
anthropic: Claude API communication
csv: Configuration management
json: Prompt handling
datetime: Logging timestamps
os: File operations
time: Process control
```

### Environment Setup
1. Create `.env` from template:
```env
CLAUDE_API_KEY_1="your_first_api_key"
CLAUDE_API_KEY_2="your_second_api_key"
```

2. Configure `automation.csv`
3. Prepare prompt JSON files
4. Set model parameters

## Usage

### Basic Operation
```bash
python -m src.data.generation.automation
```

### Output Structure
```
project_root/
└── data/
    └── dialogue/
        ├── raw/         # Generated dialogues
        ├── low_rated/   # Quality control failures
        └── processed/   # Final processed dialogues
```

## Quality Control
- Temperature variation for response diversity
- Standardized templates for consistency
- Automated logging and tracking
- Dialogue quality metrics

## Note
This system is designed for generating high-quality Socratic dialogues with a focus on philosophical exploration and self-discovery. The implementation balances consistency in methodology with diversity in perspective.

## API Keys
Two different Claude API keys required for the dual AI system. Obtain from https://console.anthropic.com 