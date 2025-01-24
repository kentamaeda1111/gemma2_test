# Utility Modules

## Configuration Management (`config.py`)

Centralized configuration management module for handling API keys and environment variables.

### Purpose

- Provide a single source of truth for API key management
- Ensure consistent environment variable access across modules
- Handle configuration errors gracefully

### Usage

1. Set up environment variables in `.env` file:
```
CLAUDE_API_KEY_1=your_first_claude_key
CLAUDE_API_KEY_2=your_second_claude_key
CLAUDE_API_KEY_QUALITY=your_quality_check_key
HUGGINGFACE_API_KEY=your_HUGGINGFACE_API_KEY
```

2. Import and use in your modules:
```python
from src.utils.config import get_api_keys

api_keys = get_api_keys()
claude_key = api_keys['claude_api_key_1']
```

### API Reference

#### `get_api_keys()`

Returns a dictionary containing all API keys from environment variables.

```python
{
    'claude_api_key_1': str,      # For dialogue generation (User)
    'claude_api_key_2': str,      # For dialogue generation (Assistant)
    'claude_api_key_quality': str, # For quality assessment
    'huggingface_api_key': str    # For model access
}
```

Raises `ValueError` if any required keys are missing from the environment.

### Used By

- `src/data/generation/automation.py`: For dialogue generation
- `src/data/processing/dialogue_quality_check.py`: For quality assessment
- `src/models/training/train.py`: For model training
- `src/models/inference/test.py`: For model inference

### Dependencies

- `python-dotenv`: For loading environment variables from `.env` file
- `typing`: For type hints
