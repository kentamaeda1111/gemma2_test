# initialではなく、チューニングモデル推論用に修正 あとtest_nsystempromptの設定も合流させた

from src.utils.config import get_api_keys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
from queue import Queue
import logging
from huggingface_hub import login
from peft import PeftModel
from anthropic import Anthropic
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

# Global Settings
DEFAULT_MODEL_VERSION = "noattention_noprompt_50_lolarefine"  
DEFAULT_CHECKPOINT = "checkpoint-990"  
MAX_HISTORY = 5  # 1回の対話でGemmaが記憶する対話履歴の数（状態保持用）
MAX_TURNS = 2    # 1つの対話における往復回数
BASE_MODEL = "google/gemma-2-2b-jpn-it"
SAVE_DIR = "data/dialogue/raw_gemma"  # 対話結果の保存先

# Add new constants for data files
CSV_CONFIG_PATH = "data/config/automation_gemma.csv"
QUESTIONS_JSON_PATH = "data/prompts/questions.json"

# Claude model settings
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"  # モデル名を更新

# Claude's system prompt
SYSTEM_PROMPT = """
あなたはソクラテスと哲学的な対話を行う対話者です。
ソクラテスの質問に対して、以下のように振る舞ってください：
- 哲学的な概念を持ち出しすぎず、一般の人の視点で答えてください
- 質問の意図や意味が分からない場合は無理にあわせず、分からない旨を素直に伝えてください
- 返答は文字数制限を越える可能性があるため、絶対端的にしてください。
"""

# Get API keys using config utility
api_keys = get_api_keys()
HF_TOKEN = api_keys['huggingface_api_key']
CLAUDE_API_KEY = api_keys['claude_api_key_quality2']

if not HF_TOKEN or not CLAUDE_API_KEY:
    logger.warning("Required API keys not found in environment variables")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
MODEL_PATH = os.path.join(ROOT_DIR, "models", DEFAULT_MODEL_VERSION, "model", DEFAULT_CHECKPOINT)
SAVE_PATH = os.path.join(ROOT_DIR, SAVE_DIR)

# Create save directory if it doesn't exist
os.makedirs(SAVE_PATH, exist_ok=True)

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

    def __del__(self):
        """デストラクタ: モデルのリソースを解放"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()  # GPUメモリをクリア

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
                        "\"{{QUESTION}}\"\n"
                        "それに対してあなたの対話者は以下のように返答をしてきました。文末に「かね？」や「ないか」等をいれつつ、引き続きソクラテスのような口調を使いながら、問いで返してください。\n"
                        f"\"{user_input}\""
                    )
                    contextualized_input = initial_setting
                else:
                    # 2回目以降は会話履歴を含める
                    conversation_history = "あなたは老練なギリシャの哲学者ソクラテスです。\n"
                    conversation_history += "あなたは以下のような発言で会話をスタートしました。\n"
                    conversation_history += "\"{{QUESTION}}\"\n"
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

def create_dialogue_session(chatai: ChatAI, dialogue_id: int):
    """
    ClaudeとGemmaの間で自動対話を実行し、結果を保存する
    1つの対話セッションで指定された回数（MAX_TURNS）の往復を行う
    """
    # Initialize Claude client
    claude = Anthropic(api_key=CLAUDE_API_KEY)
    
    dialogue_history = []
    
    try:
        # Initial Gemma message
        initial_gemma = (
            "{{QUESTION}}"
        )
        
        # Get first Claude response
        try:
            response = claude.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=150,
                temperature=0.7,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": initial_gemma
                }]
            )
            
            if not response.content:
                raise ValueError("Empty response from Claude")
                
            claude_message = response.content[0].text
            logger.info(f"Claude's first response: {claude_message}")
            
            # Add messages to history
            dialogue_history.extend([
                {"role": "gemma", "content": initial_gemma},
                {"role": "claude", "content": claude_message}
            ])
            
            # Get Gemma's response
            try:
                gemma_response = chatai.generate_response(claude_message)
                logger.info(f"Gemma's response: {gemma_response}")
                
                if not gemma_response:
                    raise ValueError("Empty response from Gemma")
                    
                dialogue_history.append({
                    "role": "gemma",
                    "content": gemma_response
                })
                
            except Exception as gemma_error:
                logger.error(f"Error in Gemma response: {str(gemma_error)}")
                raise
            
            # Continue dialogue loop
            for turn in range(MAX_TURNS - 1):
                messages = []
                for msg in dialogue_history:
                    if msg["role"] == "gemma":
                        messages.append({
                            "role": "user",
                            "content": msg["content"]
                        })
                    else:
                        messages.append({
                            "role": "assistant",
                            "content": msg["content"]
                        })
                
                # Get Claude's response
                response = claude.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=150,
                    temperature=0.7,
                    system=SYSTEM_PROMPT,
                    messages=messages
                )
                
                if not response.content:
                    raise ValueError("Empty response from Claude")
                    
                claude_message = response.content[0].text
                logger.info(f"Claude's response (turn {turn + 1}): {claude_message}")
                
                dialogue_history.append({
                    "role": "claude",
                    "content": claude_message
                })
                
                # Get Gemma's response
                try:
                    gemma_response = chatai.generate_response(claude_message)
                    logger.info(f"Gemma's response (turn {turn + 1}): {gemma_response}")
                    
                    if not gemma_response:
                        raise ValueError("Empty response from Gemma")
                        
                    dialogue_history.append({
                        "role": "gemma",
                        "content": gemma_response
                    })
                    
                except Exception as gemma_error:
                    logger.error(f"Error in Gemma response: {str(gemma_error)}")
                    raise
                
                logger.info(f"Completed turn {turn + 1} of dialogue {dialogue_id}")
            
            # 対話ループ終了後、対話履歴を保存
            save_dialogue(dialogue_history, dialogue_id)
            logger.info(f"Successfully completed dialogue {dialogue_id}")
            
        except Exception as claude_error:
            logger.error(f"Error in Claude response: {str(claude_error)}")
            raise
            
    except Exception as e:
        logger.error(f"Error in dialogue {dialogue_id}: {str(e)}")
        logger.error("Failed to complete dialogue")

def save_dialogue(dialogue_history: list, dialogue_id: int) -> None:
    """
    対話履歴をJSONファイルとして保存する
    
    Args:
        dialogue_history (list): 対話履歴のリスト
        dialogue_id (int): 対話のID
    """
    # 保存先ディレクトリの作成
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # ファイル名の生成（タイムスタンプ付き）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dialogue_{dialogue_id}_{timestamp}.json"
    filepath = os.path.join(SAVE_DIR, filename)
    
    # 対話履歴をJSON形式で保存
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'dialogue_id': dialogue_id,
            'timestamp': timestamp,
            'model_version': DEFAULT_MODEL_VERSION,
            'checkpoint': DEFAULT_CHECKPOINT,
            'history': dialogue_history
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved dialogue {dialogue_id} to {filepath}")

IS_KAGGLE_SUBMISSION = os.path.exists('/kaggle/working')

def load_questions():
    """Load questions from JSON file"""
    with open(QUESTIONS_JSON_PATH, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    # Convert to dictionary for easier lookup
    return {str(q['id']): q['content'] for q in questions_data['prompts']}

def load_config():
    """Load configuration from CSV file"""
    df = pd.read_csv(CSV_CONFIG_PATH)
    # QUESTION_IDを整数型に変換
    df['QUESTION_ID'] = df['QUESTION_ID'].astype(int)
    return df

def update_csv(csv_path: str, question_id: str, dialogue_filename: str):
    """Update CSV file with dialogue filename"""
    df = pd.read_csv(csv_path)
    df.loc[df['QUESTION_ID'] == int(question_id), 'dialogue'] = dialogue_filename
    df.to_csv(csv_path, index=False)

def main():
    """メイン実行関数"""
    # Load questions and config
    questions = load_questions()
    config_df = load_config()
    
    for _, row in config_df.iterrows():
        question_id = str(int(row['QUESTION_ID']))
        chatai = None  # 各イテレーションの開始時にNoneに設定
        
        try:
            # Get model version and checkpoint from CSV, use defaults if not specified
            model_version = row.get('model_version', DEFAULT_MODEL_VERSION)
            checkpoint = row.get('checkpoint', DEFAULT_CHECKPOINT)
            
            # Update MODEL_PATH for current iteration
            current_model_path = os.path.join(ROOT_DIR, "models", model_version, "model", checkpoint)
            
            # Skip if dialogue already exists
            if pd.notna(row['dialogue']):
                logger.info(f"Dialogue already exists for question {question_id}, skipping...")
                continue
                
            if question_id not in questions:
                logger.warning(f"Question ID {question_id} not found in questions.json")
                continue
                
            logger.info(f"Processing question ID {question_id} with model {model_version} checkpoint {checkpoint}")
            
            # Initialize model for current configuration
            if chatai is not None:
                del chatai  # 既存のモデルを明示的に解放
                torch.cuda.empty_cache()  # GPUメモリをクリア
            
            chatai = ChatAI(
                model_path=current_model_path,
                base_model=BASE_MODEL,
                max_history=MAX_HISTORY,
                hf_token=HF_TOKEN
            )
            logger.info(f"Model loaded successfully for question {question_id}")
            
            # Get question content
            question_content = questions[question_id]
            
            # Create dialogue session with the specific question
            dialogue_history = []
            
            # Initialize Claude client
            claude = Anthropic(api_key=CLAUDE_API_KEY)
            
            # Initial Gemma message (replace placeholder)
            initial_gemma = question_content
            
            # Get first Claude response
            response = claude.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=150,
                temperature=0.7,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": initial_gemma
                }]
            )
            
            claude_message = response.content[0].text
            logger.info(f"Claude's first response: {claude_message}")
            
            # Add messages to history
            dialogue_history.extend([
                {"role": "gemma", "content": initial_gemma},
                {"role": "claude", "content": claude_message}
            ])
            
            # Continue dialogue for specified number of turns
            for turn in range(MAX_TURNS - 1):
                # Get Gemma's response
                gemma_response = chatai.generate_response(claude_message)
                logger.info(f"Gemma's response (turn {turn + 1}): {gemma_response}")
                
                dialogue_history.append({
                    "role": "gemma",
                    "content": gemma_response
                })
                
                # Get Claude's response
                messages = []
                for msg in dialogue_history:
                    if msg["role"] == "gemma":
                        messages.append({
                            "role": "user",
                            "content": msg["content"]
                        })
                    else:
                        messages.append({
                            "role": "assistant",
                            "content": msg["content"]
                        })
                
                response = claude.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=150,
                    temperature=0.7,
                    system=SYSTEM_PROMPT,
                    messages=messages
                )
                
                claude_message = response.content[0].text
                logger.info(f"Claude's response (turn {turn + 1}): {claude_message}")
                
                dialogue_history.append({
                    "role": "claude",
                    "content": claude_message
                })
            
            # Generate filename and save dialogue
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dialogue_{question_id}_{timestamp}.json"
            filepath = os.path.join(SAVE_DIR, filename)
            
            # Save dialogue
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'question_id': question_id,
                    'timestamp': timestamp,
                    'model_version': model_version,
                    'checkpoint': checkpoint,
                    'history': dialogue_history
                }, f, ensure_ascii=False, indent=2)
            
            # Update CSV with filename
            update_csv(CSV_CONFIG_PATH, question_id, filename)
            logger.info(f"Completed dialogue {question_id} and saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {str(e)}")
            if chatai is not None:
                del chatai  # エラー時もモデルを解放
                torch.cuda.empty_cache()
            continue
        
        finally:
            # 各イテレーション終了時にモデルを解放
            if chatai is not None:
                del chatai
                torch.cuda.empty_cache()
    
    logger.info("Completed all dialogues")

if __name__ == "__main__":
    main()