import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
from queue import Queue
import logging
from src.utils.config import get_api_keys
import json

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self, model_path="./model", base_model="google/gemma-2-2b-jpn-it", max_history=5, config_path="1gouki.json"):
        self.max_history = max_history
        self.message_history = Queue(maxsize=max_history)
        
        # 設定ファイルの読み込み
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
            self.config_data = {}
        
        try:
            logger.info("Loading model and tokenizer...")
            
            # Hugging Face の認証を追加
            api_keys = get_api_keys()
            huggingface_token = api_keys['huggingface_api_key']
            
            # 設定ファイルから生成パラメータを読み込む
            self.generation_config = self.config_data.get('generation_config', {
                "max_new_tokens": 256,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            })
            
            # トークナイザーは元のモデルから（認証付き）
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True,
                token=huggingface_token
            )
            
            # ベースモデルの設定を読み込む
            config = AutoConfig.from_pretrained(
                base_model,
                trust_remote_code=True,
                token=huggingface_token
            )
            
            # モデルタイプを設定
            config.model_type = "gemma"
            
            # まずベースモデルを読み込む（認証付き）
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                config=config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation='eager',
                trust_remote_code=True,
                token=huggingface_token
            )
            
            # 次にチェックポイントを読み込む
            if os.path.exists(model_path):
                logger.info(f"Loading checkpoint from {model_path}")
                # safetensorsファイルを探す
                if os.path.exists(os.path.join(model_path, "model.safetensors")):
                    logger.info("Loading from safetensors format")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        config=config,
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                        attn_implementation='eager',
                        trust_remote_code=True,
                        ignore_mismatched_sizes=True
                    )
                # pytorch_model.binを探す
                elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
                    logger.info("Loading from pytorch format")
                    state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
                    self.model.load_state_dict(state_dict, strict=False)
                else:
                    logger.warning(f"No model file found in {model_path}")
                
                logger.info("Checkpoint loaded successfully")
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def _update_history(self, message):
        """履歴の更新"""
        if self.message_history.full():
            removed = self.message_history.get()
            logger.debug(f"Removed message from history: {removed}")
        self.message_history.put(message)
        logger.debug(f"Added message to history: {message}")
        logger.debug(f"Current queue size: {self.message_history.qsize()}")

    def _format_messages(self):
        """メッセージ履歴をGemma-2のフォーマットに変換"""
        messages = list(self.message_history.queue)
        
        # メッセージの順序を検証
        for i in range(len(messages)):
            expected_role = "user" if i % 2 == 0 else "model"
            if messages[i]["role"] != expected_role:
                logger.warning(f"Invalid message sequence detected at position {i}")
                # 必要に応じて履歴をクリア
                self.clear_history()
                # 最新のユーザーメッセージのみを保持
                return [messages[-1]] if messages[-1]["role"] == "user" else []
        
        return messages

    def generate_response(self, user_input, add_to_history=True):
        try:
            # ユーザー入力を履歴に追加（オプション）
            if add_to_history:
                self._update_history({"role": "user", "content": user_input})
            
            # チャットテンプレートの適用
            messages = self._format_messages()
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # デバッグ用
            logger.debug(f"Generated prompt: {prompt}")
            
            # 入力をトークン化
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.model.device)
            
            # 応答の生成
            outputs = self.model.generate(
                **inputs,
                **self.generation_config
            )
            
            # デコード
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            # 特殊トークンを保持
            
            # モデルの応答部分を抽出
            response_parts = decoded_output.split("<start_of_turn>model")
            if len(response_parts) > 1:
                last_response = response_parts[-1].split("<end_of_turn>")[0].strip()
                # 不要なプレフィックスを削除
                if "model" in last_response:
                    last_response = last_response.split("model", 1)[1].strip()
            else:
                last_response = "応答の解析に失敗しました。"
            
            # 応答を履歴に追加
            if add_to_history:
                self._update_history({"role": "model", "content": last_response})
            
            return last_response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "申し訳ありません。エラーが発生しました。"

    def clear_history(self):
        """会話履歴のクリア"""
        while not self.message_history.empty():
            self.message_history.get()
        logger.info("Conversation history cleared")

def main():
    # モデルパスとconfig_pathを指定
    model_path = "./model/checkpoint-1980"
    config_path = "1gouki.json"
    
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist")
        return
        
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} does not exist, using defaults")
        
    chatbot = ChatBot(
        model_path=model_path,
        base_model="google/gemma-2-2b-jpn-it",
        config_path=config_path
    )
    print("\nGemma-2チャットボットへようこそ。")
    
    # 初期の会話設定（1回だけ）
    initial_user_msg = "ソクラテスさん。今日は何について話しますか？"
    initial_model_msg = "今日は「正義」について話そうではないか、君はそもそも「正義」とはなんだと思う？"
    
    # 初期会話をセットアップ
    chatbot._update_history({"role": "user", "content": initial_user_msg})
    chatbot._update_history({"role": "model", "content": initial_model_msg})
    
    # 初期プロンプトを表示
    print(f"\nソクラテス: {initial_model_msg}")
    
    while True:
        try:
            user_input = input("\nあなた: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '終了']:
                print("チャットを終了します。")
                break
                
            if user_input.lower() in ['clear', 'クリア']:
                chatbot.clear_history()
                print("会話履歴をクリアしました。")
                chatbot._update_history({"role": "user", "content": initial_user_msg})
                chatbot._update_history({"role": "model", "content": initial_model_msg})
                print("\nソクラテス：今日は「正義」について話そうではないか、君はそもそも「正義」とはなんだと思う？")
                continue
                
            if not user_input:
                continue
            
            # ユーザー入力を追加（1回だけ）
            chatbot._update_history({"role": "user", "content": user_input})
            
            # 応答の生成（add_to_history=Falseで履歴の二重追加を防ぐ）
            response = chatbot.generate_response(user_input, add_to_history=False)
            
            # モデルの応答を追加
            chatbot._update_history({"role": "model", "content": response})
            print(f"\nソクラテス: {response}")
            
        except KeyboardInterrupt:
            print("\nチャットを終了します。")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            print("エラーが発生しました。もう一度試してください。")

if __name__ == "__main__":
    main() 