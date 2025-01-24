import os
from typing import Dict
from dotenv import load_dotenv

def get_api_keys() -> Dict[str, str]:
    """
    Load API keys from environment variables or Kaggle secrets.
    
    Returns:
        Dict[str, str]: Dictionary containing API keys
        {
            'claude_api_key_1': str,
            'claude_api_key_2': str,
            'claude_api_key_quality': str,
            'huggingface_api_key': str
        }
    
    Raises:
        ValueError: If required API keys are not found
    """
    # Try to load from .env file first (for local environment)
    load_dotenv(override=True)
    
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
    
    # If keys are missing, try to get from Kaggle secrets
    if missing_keys and os.path.exists('/kaggle/working'):
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            
            # Try getting each missing key from Kaggle secrets
            for env_var in missing_keys[:]:  # Create a copy to modify during iteration
                try:
                    secret_value = user_secrets.get_secret(env_var)
                    os.environ[env_var] = secret_value
                    missing_keys.remove(env_var)
                except Exception as e:
                    print(f"Could not get {env_var} from Kaggle secrets: {str(e)}")
            
            # Update api_keys with any new values
            for key, env_var in required_keys.items():
                value = os.getenv(env_var)
                if value is not None:
                    api_keys[key] = value
                    
        except ImportError:
            print("Not running in Kaggle environment or secrets not accessible")
    
    # If still missing keys and in Colab/Kaggle, try direct setting
    if missing_keys and (os.path.exists('/kaggle/working') or os.path.exists('/content')):
        try:
            # Direct environment variable setting
            if 'CLAUDE_API_KEY_1' in missing_keys:
                os.environ['CLAUDE_API_KEY_1'] = "your_claude_key_1"  # Set your key here
            if 'CLAUDE_API_KEY_2' in missing_keys:
                os.environ['CLAUDE_API_KEY_2'] = "your_claude_key_2"  # Set your key here
            if 'CLAUDE_API_KEY_QUALITY' in missing_keys:
                os.environ['CLAUDE_API_KEY_QUALITY'] = "your_claude_key_quality"  # Set your key here
            if 'HUGGINGFACE_API_KEY' in missing_keys:
                os.environ['HUGGINGFACE_API_KEY'] = "your_huggingface_key"  # Set your key here
            
            # Try getting the keys again
            api_keys = {}
            missing_keys = []
            for key, env_var in required_keys.items():
                value = os.getenv(env_var)
                if value is None:
                    missing_keys.append(env_var)
                api_keys[key] = value
        except Exception as e:
            print(f"Error setting environment variables directly: {str(e)}")
    
    # If still missing keys, raise error
    if missing_keys:
        raise ValueError(
            f"Missing required API keys: {', '.join(missing_keys)}\n"
            "Please either:\n"
            "1. Set them in your .env file (for local environment)\n"
            "2. Set them in Kaggle secrets\n"
            "3. Set them directly in the notebook (for Colab/Kaggle)\n"
            "4. Modify config.py to include your keys"
        )
    
    return api_keys 