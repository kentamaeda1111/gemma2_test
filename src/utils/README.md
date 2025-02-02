# Utility Modules

## Configuration Management (`config.py`)

Centralized configuration management module for handling API keys and environment variables.

### Purpose

- Provide a single source of truth for API key management
- Support multiple environments (Local, Kaggle, Google Colab)
- Handle configuration errors gracefully

### Usage

#### 1. Local Environment
Set up environment variables in `.env` file:
```
CLAUDE_API_KEY_1=your_first_claude_key
CLAUDE_API_KEY_2=your_second_claude_key
CLAUDE_API_KEY_QUALITY=your_quality_check_key
HUGGINGFACE_API_KEY=your_HUGGINGFACE_API_KEY
```

#### 2. Kaggle Environment
Using Kaggle Secrets (Recommended):
```python
from src.utils.config import get_api_keys

# Will automatically fetch from Kaggle secrets
api_keys = get_api_keys()
```

#### 3. Google Colab Environment
The system will display a secure password form:
```python
from src.utils.config import get_api_keys

# Will display a form to enter API keys
api_keys = get_api_keys()
```

### API Reference

#### `get_api_keys()`

Returns a dictionary containing all API keys. The function automatically detects the environment and handles key retrieval appropriately.

```python
{
    'claude_api_key_1': str,      # For dialogue generation (User)
    'claude_api_key_2': str,      # For dialogue generation (Assistant)
    'claude_api_key_quality': str, # For quality assessment
    'huggingface_api_key': str    # For model access
}
```

#### `get_colab_api_keys()`

Internal function that handles API key input in Google Colab environment. Displays a secure password form for key entry.

Returns the same dictionary format as `get_api_keys()`.

### Environment Detection

The configuration system automatically:
1. Checks for `.env` file (local environment)
2. Checks for Kaggle environment (`/kaggle/working` path)
3. Checks for Google Colab environment (`/content` path)
4. Provides appropriate key input method for each environment

### Error Handling

Raises `ValueError` if:
- Required API keys are missing
- Environment cannot be determined
- Keys cannot be retrieved from Kaggle secrets

### Used By

- `src/data/generation/automation.py`: For dialogue generation
- `src/data/quality_check/dialogue_quality_check.py`: For quality assessment
- `src/models/training/train.py`: For model training
- `src/models/inference/test.py`: For model inference

### Dependencies

- `python-dotenv`: For loading environment variables from `.env` file
- `typing`: For type hints
- `ipywidgets`: For Colab form interface
- `IPython.display`: For Colab display functionality
