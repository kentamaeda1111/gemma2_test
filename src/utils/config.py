import os
from typing import Dict
from dotenv import load_dotenv

def get_api_keys() -> Dict[str, str]:
    """
    Load API keys from environment variables.
    
    Returns:
        Dict[str, str]: Dictionary containing API keys
        {
            'claude_api_key_1': str,
            'claude_api_key_2': str,
            'claude_api_key_quality': str,
            'huggingface_api_key': str
        }
    
    Raises:
        ValueError: If required API keys are not found in environment variables
    """
    # Load environment variables from .env file
    load_dotenv()
    
    required_keys = {
        'claude_api_key_1': 'CLAUDE_API_KEY_1',
        'claude_api_key_2': 'CLAUDE_API_KEY_2',
        'claude_api_key_quality': 'CLAUDE_API_KEY_QUALITY',
        'huggingface_api_key': 'HUGGINGFACE_API_KEY'
    }
    
    # Get API keys from environment variables
    api_keys = {}
    missing_keys = []
    
    for key, env_var in required_keys.items():
        value = os.getenv(env_var)
        if value is None:
            missing_keys.append(env_var)
        api_keys[key] = value
    
    # Raise error if any required keys are missing
    if missing_keys:
        raise ValueError(
            f"Missing required API keys in environment variables: {', '.join(missing_keys)}\n"
            "Please ensure these are set in your .env file."
        )
    
    return api_keys 