# Socratic Dialogue Generation System

## Overview
An automated system for generating Socratic-style dialogues using Claude AI models. The system manages conversations between two AI models - one acting as a student/user and another as a Socratic teacher.

## Configuration Parameters

### Hard-coded Parameters in automation.py
These parameters are set directly in the code and require code modification to change:

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
These parameters can be modified through the CSV file without changing code:

1. Temperature Settings:
   - `AI1_TEMPERATURE`: Controls creativity level for the user AI (float: 0.0-1.0)
   - `AI2_TEMPERATURE`: Controls creativity level for the Socrates AI (float: 0.0-1.0)

2. Conversation Control:
   - `MAX_MESSAGE_PAIRS`: Number of message pairs to keep in history (integer)
   - `MAX_TURNS`: Total number of conversation turns (integer)

3. Prompt Selection:
   - `INITIAL_QUESTION_ID`: Starting question ID for dialogue (integer)
   - `USER_PROMPT_ID`: System prompt ID for user AI (integer)
   - `ASSISTANT_PROMPT_ID`: System prompt ID for Socrates AI (integer)

4. Character Settings:
   - `OTHERS_ID`: Additional context settings ID (integer)
   - `PERSONA_ID`: Character personality settings ID (integer)
   - `TRANSFORM_ID`: Dialogue style transformation rules ID (integer)
   - `RESPONSE_ID`: Response pattern settings ID (integer)
   - `UPDATE_ID`: Update pattern settings ID (integer)

5. Logging:
   - `DIALOGUE_KEYWORD`: Identifier for log files (string)
   - `dialogue`: Output dialogue file path (auto-filled by system)

## Key Features

### 1. Dual AI Model System
- AI1: Acts as the User/Student
- AI2: Acts as the Socratic Teacher
- Independent temperature and parameter control for each model

### 2. Dynamic Prompt Management
- Loads prompts from JSON configuration files
- Supports placeholder substitution
- Maintains separate system prompts for each AI

### 3. Conversation Control
- Manages message history and turn limits
- Controls conversation length and complexity
- Maintains context within specified bounds

### 4. Logging System
- Creates detailed dialogue logs with metadata
- Timestamps and versioning for each conversation
- Comprehensive header information for tracking

### 5. CSV Integration
- Reads configuration from automation.csv
- Supports batch processing of multiple configurations
- Tracks progress and prevents duplicate processing

## Installation

### Dependencies
- anthropic: For Claude API communication
- csv: For configuration management
- json: For prompt file handling
- datetime: For logging timestamps
- os: For file operations
- time: For process control

## Usage

### 1. Setup
1. Configure `automation.csv` with desired parameters
2. Ensure prompt JSON files are in place
3. Create `.env` file from `.env.template` and set your API keys:
   ```env
   # Required API Keys
   CLAUDE_API_KEY_1="your_first_api_key"
   CLAUDE_API_KEY_2="your_second_api_key"
   CLAUDE_API_KEY_QUALITY="your_quality_check_api_key"
   ```
4. Set other model parameters

### 2. Running
```bash
# From the project root directory
python -m src.data.generation.automation
```

### 3. Output
- Creates dialogue files in `data/dialogue/raw/`
- Updates `automation.csv` with results
- Generates timestamped conversation logs

## Project Structure
```
project_root/
├── src/
│   └── data/
│       └── generation/
│           ├── automation.py   # This module
│           └── README.md      # This file
│
└── data/
    ├── config/
    │   └── automation.csv     # Configuration and results tracking
    │
    ├── prompts/              # System prompts directory
    │   ├── assistant_system_prompt/
    │   │   ├── assistant_system_prompt.json
    │   │   ├── response.json
    │   │   └── update.json
    │   └── user_system_prompt/
    │       ├── user_system_prompt.json
    │       ├── others.json
    │       ├── persona.json
    │       └── transform.json
    │
    └── dialogue/            # Dialogue data
        ├── raw/             # Generated dialogue files
        ├── low_rated/       # Low-quality dialogues
        └── processed/       # Processed dialogue files
```

## Configuration

### Required Files
- `automation.csv`: Main configuration file
- JSON prompt files:
  - user_system_prompt.json
  - assistant_system_prompt.json
  - questions.json
  - persona.json
  - others.json
  - transform.json
  - response.json
  - update.json

### CSV Configuration Parameters
1. Temperature Settings:
   - AI1_TEMPERATURE: Creativity level for the user AI
   - AI2_TEMPERATURE: Creativity level for the Socrates AI

2. Conversation Control:
   - MAX_MESSAGE_PAIRS: Number of message pairs to keep in history
   - MAX_TURNS: Total number of conversation turns

3. Prompt Selection:
   - INITIAL_QUESTION_ID: Starting question for dialogue
   - USER_PROMPT_ID: System prompt for user AI
   - ASSISTANT_PROMPT_ID: System prompt for Socrates AI

4. Character Settings:
   - OTHERS_ID: Additional context settings
   - PERSONA_ID: Character personality settings
   - TRANSFORM_ID: Dialogue style transformation rules
   - RESPONSE_ID: Response pattern settings
   - UPDATE_ID: Update pattern settings

5. Logging:
   - DIALOGUE_KEYWORD: Identifier for log files
   - dialogue: Output dialogue file path (auto-filled)

## Note
This system is designed to generate high-quality Socratic dialogues for training and evaluation purposes. The dialogue style and content are controlled through the configuration files and prompt templates.

## API Keys
You need two different Claude API keys for the dual AI system. Visit https://console.anthropic.com to obtain your API keys. 