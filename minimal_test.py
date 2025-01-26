import torch
import transformers
from peft import PeftModel
import anthropic

def test_basic_functionality():
    # Basic ML libraries test
    print("Testing PyTorch:", torch.cuda.is_available())
    print("Testing Transformers:", transformers.__version__)
    
    # API connection test (requires API key)
    try:
        client = anthropic.Client(api_key="dummy_key")
        print("Anthropic client initialized")
    except Exception as e:
        print("Anthropic client error:", e)

if __name__ == "__main__":
    test_basic_functionality() 