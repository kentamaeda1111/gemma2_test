# initialではなく、チューニングモデル推論用に修正

from src.utils.config import get_api_keys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
from queue import Queue
import logging
from IPython.display import clear_output
import ipywidgets as widgets
from huggingface_hub import login
from peft import PeftModel

# Global Settings
MODEL_VERSION = "train_nam"  
CHECKPOINT = "checkpoint-700"  
MAX_HISTORY = 5  
BASE_MODEL = "google/gemma-2-2b-jpn-it"

# Get API keys using config utility
api_keys = get_api_keys()
HF_TOKEN = api_keys['huggingface_api_key']

if not HF_TOKEN:
    logger.warning("HUGGINGFACE_API_KEY not found in environment variables")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
MODEL_PATH = os.path.join(ROOT_DIR, "models", MODEL_VERSION, "model", CHECKPOINT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow警告を抑制

class ChatAI:
    """
    ChatAI class
    - Loads the model and tokenizer
    - Manages message history
    - Generates responses
    """
    def __init__(
        self,
        model_path: str = "./model",
        base_model: str = "google/gemma-2-2b-jpn-it",
        max_history: int = 5,
        hf_token: str = None
    ):
        """
        Constructor
        Args:
            model_path (str): Path to the fine-tuned model
            base_model (str): Base model path on Hugging Face
            max_history (int): Number of turns to store in the history
            hf_token (str): Hugging Face access token
        """
        self.max_history = max_history
        self.message_history = Queue(maxsize=max_history)
        self.hf_token = hf_token
        
        try:
            logger.info("Loading model and tokenizer...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                token=hf_token,
                trust_remote_code=True
            )
            
            # Get configuration based on available hardware
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load the base model with appropriate configuration
            load_config = {
                "trust_remote_code": True,
                "token": hf_token,
                "low_cpu_mem_usage": True
            }
            
            # Adjust configuration based on device
            if device == "cuda":
                load_config["device_map"] = "auto"
                load_config["torch_dtype"] = torch.bfloat16
            else:
                load_config["device_map"] = "auto"
                load_config["torch_dtype"] = torch.float32
                load_config["offload_folder"] = "offload_folder"
                os.makedirs("offload_folder", exist_ok=True)

            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                **load_config
            )
            
            # Load the PEFT model
            self.model = PeftModel.from_pretrained(
                base_model_obj,
                model_path,
                torch_dtype=load_config["torch_dtype"]
            )

            logger.info(f"Model loaded successfully on {device}")
            
            self.generation_config = {
                "max_new_tokens": 256,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            }
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def _initialize_model(self, base_model, model_path, hf_token):
        """Initialize the model with appropriate device and memory settings"""
        try:
            # Check available memory and GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Configure model loading parameters based on available resources
            load_config = {
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
                "token": hf_token
            }
            
            # If running on CPU or limited memory, adjust loading strategy
            if device == "cpu":
                load_config["device_map"] = "auto"
                load_config["offload_folder"] = "offload_folder"  # Add offload directory
                os.makedirs("offload_folder", exist_ok=True)
            else:
                load_config["device_map"] = "balanced"

            # Load base model with configured parameters
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                **load_config
            )

            # Load fine-tuned model
            self.model = PeftModel.from_pretrained(
                base_model_obj,
                model_path,
                **load_config
            )

            logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def _update_history(self, message: dict) -> None:
        """
        Enqueues new user or model messages and manages removal of old messages
        """
        if self.message_history.full():
            removed = self.message_history.get()
            logger.debug(f"Removed message from history: {removed}")
        self.message_history.put(message)
        logger.debug(f"Added message to history: {message}")
        logger.debug(f"Current queue size: {self.message_history.qsize()}")

    def _format_messages(self):
        """
        Ensures that messages alternate correctly between user and model
        Returns a list of messages if valid, or an empty list if invalid
        """
        messages = list(self.message_history.queue)
        
        for i in range(len(messages)):
            expected_role = "user" if i % 2 == 0 else "model"
            if messages[i]["role"] != expected_role:
                logger.warning(f"Invalid message sequence detected at position {i}")
                return [messages[-1]] if messages[-1]["role"] == "user" else []
        
        return messages

    def generate_response(self, user_input: str, add_to_history: bool = True) -> str:
        """
        Generates a response from the model
        Args:
            user_input (str): The user's input text
            add_to_history (bool): Whether to add this turn to the conversation history
        Returns:
            str: The model's generated response
        """
        try:
            if add_to_history:
                if self.message_history.qsize() == 0:  # 最初の入力の場合
                    initial_setting = (
                        "あなたは老練なギリシャの哲学者ソクラテスです。\n"
                        "あなたは以下のような発言で会話をスタートしました。\n"
                        "\"今日は『自分』という、これ以上ないほど身近な存在でありながら、あまり話すことのないトピックについて話そうではないか。"
                        "人は「自分の意思で決めた」や、「自分らしさ」というような具合に、日々「自分」という言葉を多くの場面で使っておるが、そもそも「自分」という言葉を使っているとき、君は何を指していると思うかね？\"\n"
                        "それに対してあなたの対話者は以下のように返答をしてきました。文末に「かね？」や「ないか」等をいれつつ、引き続きソクラテスのような口調を使いながら、問いで返してください。\n"
                        f"\"{user_input}\""
                    )
                    contextualized_input = initial_setting
                else:
                    # 2回目以降は会話履歴を含める
                    conversation_history = "あなたは老練なギリシャの哲学者ソクラテスです。\n"
                    conversation_history += "あなたは以下のような発言で会話をスタートしました。\n"
                    conversation_history += "\"今日は『自分』という、これ以上ないほど身近な存在でありながら、あまり話すことのないトピックについて話そうではないか。"
                    conversation_history += "人は「自分の意思で決めた」や、「自分らしさ」というような具合に、日々「自分」という言葉を多くの場面で使っておるが、そもそも「自分」という言葉を使っているとき、君は何を指していると思うかね？\"\n"
                    conversation_history += "それに対して以下のように今のところ対話が進んでます。文末に「かね？」や「ないか」等をいれつつ、引き続きソクラテスのような口調を使いながら、問いで返してください。\n"

                    # これまでの会話履歴を追加
                    messages = list(self.message_history.queue)
                    for msg in messages:
                        if msg["role"] == "user":
                            conversation_history += f"\nUser: \"{msg['content']}\""
                        else:
                            conversation_history += f"\nModel: {msg['content']}"

                    # 新しい入力を追加
                    conversation_history += f"\n\nUser: \"{user_input}\""
                    contextualized_input = conversation_history

                self._update_history({"role": "user", "content": contextualized_input})
            
            messages = self._format_messages()
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            logger.debug(f"Generated prompt: {prompt}")
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                **self.generation_config
            )
            
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

            response_parts = decoded_output.split("<start_of_turn>model")
            if len(response_parts) > 1:
                last_response = response_parts[-1].split("<end_of_turn>")[0].strip()
                if "model" in last_response:
                    last_response = last_response.split("model", 1)[1].strip()
            else:
                last_response = "Failed to respond"
            
            if add_to_history:
                self._update_history({"role": "model", "content": last_response})
            
            return last_response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "There was an error"

    def _get_model_config(self):
        """Get model configuration based on available hardware"""
        config = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "token": self.hf_token
        }
        
        # GPUが利用可能かチェック
        if torch.cuda.is_available():
            config["device_map"] = "balanced"
            logger.info("GPU detected, using balanced device map")
        else:
            # CPU環境用の設定
            config["device_map"] = "auto"
            config["offload_folder"] = "offload_folder"
            os.makedirs("offload_folder", exist_ok=True)
            logger.info("CPU environment detected, using memory offloading")
        
        return config

def create_chat_ui(chatai: ChatAI):
    """
    Creates a basic interactive chat UI. If running in Jupyter/IPython environment,
    creates an interactive widget UI, otherwise runs in console mode.
    """
    try:
        # Check if we're in IPython/Jupyter environment
        get_ipython()
        _create_widget_ui(chatai)
    except:
        # If not in IPython, use console interface
        _create_console_ui(chatai)

def _create_widget_ui(chatai: ChatAI):
    """
    Creates a widget-based UI for Jupyter/IPython environment
    """
    chat_output = widgets.Output()
    text_input = widgets.Text(
        placeholder='Enter your message...',
        layout=widgets.Layout(width='80%')
    )
    send_button = widgets.Button(
        description='Send',
        layout=widgets.Layout(width='20%')
    )
    
    def on_send_button_clicked(_):
        user_input = text_input.value
        if not user_input.strip():
            return
            
        with chat_output:
            clear_output(wait=True)
            for i, msg in enumerate(chatai.message_history.queue):
                if msg["role"] == "user":
                    if i == 0:
                        pass
                    else:
                        print(f"\nYou: {msg['content']}")
                else:
                    print(f"\nSocrates: {msg['content']}")
            
            print(f"\nYou: {user_input}")
            response = chatai.generate_response(user_input)
            print(f"\nSocrates: {response}")
        
        text_input.value = ''
    
    send_button.on_click(on_send_button_clicked)
    
    with chat_output:
        for i, msg in enumerate(chatai.message_history.queue):
            if msg["role"] == "user":
                if i == 0:
                    pass  
                else:
                    print(f"\nYou: {msg['content']}")
            else:
                print(f"\nLaMDA: {msg['content']}")
    
    display(widgets.VBox([
        chat_output,
        widgets.HBox([text_input, send_button]),
    ]))

def _create_console_ui(chatai: ChatAI):
    """
    Creates a console-based UI for standard Python environment
    """
    # 最初の質問を表示
    initial_question = (
        "\nSocrates: 今日は『自分』という、これ以上ないほど身近な存在でありながら、"
        "あまり話すことのないトピックについて話そうではないか。\n"
        "人は「自分の意思で決めた」や、「自分らしさ」というような具合に、日々「自分」という言葉を多くの場面で使っておるが、そもそも「自分」という言葉を使っているとき、君は何を指していると思うかね？"
    )
    print(initial_question)
    
    # Start conversation loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nSocrates: また会おう。")
                break
            if not user_input:
                continue
                
            response = chatai.generate_response(user_input)
            print(f"\nSocrates: {response}")
            
        except KeyboardInterrupt:
            print("\nSocrates: 対話を終了します。")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            break

IS_KAGGLE_SUBMISSION = os.path.exists('/kaggle/working')

if __name__ == "__main__":
    try:
        chatai = ChatAI(
            model_path=MODEL_PATH,
            base_model=BASE_MODEL,
            max_history=MAX_HISTORY,
            hf_token=HF_TOKEN
        )
        
        print(f"\nSocratic AI Assistant with Fine-Tuned Gemma-2b (Model: {MODEL_VERSION}, Checkpoint: {CHECKPOINT})")
        create_chat_ui(chatai)
    
    except Exception as e:
        print(f"Error initializing ChatAI: {str(e)}")