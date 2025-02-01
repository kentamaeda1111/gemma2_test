#kaggleに提出したもので使ったコード、kaggle向けのコードなのでGPUのみに特化

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
from queue import Queue
import logging
from IPython.display import clear_output
import ipywidgets as widgets
from huggingface_hub import login
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        try:
            logger.info("Loading model and tokenizer...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                token=hf_token,
                trust_remote_code=True
            )
            
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="balanced",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                token=hf_token
            )
            
            self.model = PeftModel.from_pretrained(
                base_model_obj,
                model_path,
                device_map="balanced",
                torch_dtype=torch.bfloat16
            )
            
            logger.info("Model loaded successfully")
            
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

class ChatAI(ChatAI):
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

class ChatAI(ChatAI):
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
                self._update_history({"role": "user", "content": user_input})
            
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

def create_chat_ui(chatai: ChatAI):
    """
    Creates a basic interactive chat UI connected to the ChatAI instance
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
                print(f"\nSocrates: {msg['content']}")
    
    display(widgets.VBox([
        chat_output,
        widgets.HBox([text_input, send_button]),
    ]))

model_path = "/kaggle/input/gemma-2b-jpn-socrates/pytorch/default/1/gemma-2b-jpn-socrates/checkpoint-1980"
base_model = "google/gemma-2-2b-jpn-it"

IS_KAGGLE_SUBMISSION = os.path.exists('/kaggle/working')

if not os.path.exists(model_path):
    print(f"Error: Model path {model_path} does not exist")
else:
    try:
        if IS_KAGGLE_SUBMISSION:
            chatai = ChatAI(
                model_path=model_path,
                base_model=base_model
            )
        else:
            chatai = ChatAI(
                model_path=model_path,
                base_model=base_model,
                hf_token="your-actual-token-here" 
            )
        
        print("\nSocratic AI Assistant with Fine-Tuned Gemma-2b")
        initial_user_msg = "あなたは古代ギリシャの哲学者ソクラテスです。今日は何について話しますか？"
        initial_model_msg = (
            "やぁ、よく来てくれたね。今日は『自分』という、これ以上ないほど身近な存在でありながら、あまり話すことのないトピックについて話そうではないか。"
            "人は「自分の意思で決めた」や、「自分らしさ」というような具合に、日々「自分」という言葉を多くの場面で使っておるが、"
            "そもそも「自分」という言葉を使っているとき、君は何を指していると思うかね？"
        )
        
        chatai._update_history({"role": "user", "content": initial_user_msg})
        chatai._update_history({"role": "model", "content": initial_model_msg})
        
        create_chat_ui(chatai)
    
    except Exception as e:
        print(f"Error initializing ChatAI: {str(e)}")