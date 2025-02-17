# tokenの表示させた
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from IPython.display import clear_output
import ipywidgets as widgets
from peft import PeftModel
from queue import Queue
from kaggle_secrets import UserSecretsClient

# Kaggle specific setup
user_secrets = UserSecretsClient()
os.environ['HUGGINGFACE_API_KEY'] = user_secrets.get_secret("HUGGINGFACE_API_KEY")
HF_TOKEN = os.environ['HUGGINGFACE_API_KEY']

# Global Settings
USE_BASE_MODEL = False  # Falseの場合は fine-tuned model、Trueの場合は base model を使用
MODEL_VERSION = "attention-tuned_990"
CHECKPOINT = "checkpoint-990"
MAX_HISTORY = 7
BASE_MODEL = "google/gemma-2-2b-jpn-it"
MODEL_PATH = "/kaggle/input/attention-tuned_990/pytorch/default/1/checkpoint-990"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ChatAI:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        base_model: str = BASE_MODEL,
        max_history: int = MAX_HISTORY,
        hf_token: str = HF_TOKEN,
        use_base_model: bool = USE_BASE_MODEL  # 新しいパラメータを追加
    ):
        self.max_history = max_history
        self.message_history = Queue(maxsize=max_history)
        self.hf_token = hf_token
        self.max_context_length = 8192  # Gemma-2bの最大コンテキスト長
        
        try:
            logger.info("Loading model and tokenizer...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                token=hf_token,
                trust_remote_code=True
            )
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            load_config = {
                "trust_remote_code": True,
                "token": hf_token,
                "low_cpu_mem_usage": True
            }
            
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
            
            if use_base_model:
                logger.info("Using base model without fine-tuning")
                self.model = base_model_obj
            else:
                logger.info("Loading fine-tuned model")
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

    def _update_history(self, message: dict) -> None:
        if self.message_history.full():  # 5つの履歴で満杯になった場合
            self.message_history.get()    # 最も古い履歴を削除
        self.message_history.put(message) # 新しいメッセージを追加

    def _format_messages(self):
        messages = list(self.message_history.queue)
        for i in range(len(messages)):
            expected_role = "user" if i % 2 == 0 else "model"
            if messages[i]["role"] != expected_role:
                return [messages[-1]] if messages[-1]["role"] == "user" else []
        return messages

    def get_current_context_length(self, messages) -> int:
        """現在のコンテキストのトークン数を計算"""
        # メッセージの役割を確認し、必要に応じて調整
        formatted_messages = []
        for i, msg in enumerate(messages):
            role = "user" if i % 2 == 0 else "assistant"
            formatted_messages.append({"role": role, "content": msg["content"]})

        prompt = self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return len(tokens['input_ids'][0])

    def generate_response(self, user_input: str, add_to_history: bool = True) -> tuple[str, int]:
        try:
            if add_to_history:
                if self.message_history.qsize() == 0:
                    # 最初のメッセージの場合、システムプロンプトとユーザー入力を組み合わせる
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
                    # 2回目以降のメッセージの場合
                    messages = list(self.message_history.queue)
                    
                    # システムプロンプト部分
                    system_prompt = (
                        "あなたは老練なギリシャの哲学者ソクラテスです。\n"
                        "あなたは以下のような発言で会話をスタートしました。\n"
                        "\"今日は『自分』という、これ以上ないほど身近な存在でありながら、あまり話すことのないトピックについて話そうではないか。"
                        "人は「自分の意思で決めた」や、「自分らしさ」というような具合に、日々「自分」という言葉を多くの場面で使っておるが、"
                        "そもそも「自分」という言葉を使っているとき、君は何を指していると思うかね？\"\n"
                        "それに対して以下のように対話が進んでいます。文末に「かね？」や「ないか」等をいれつつ、"
                        "引き続きソクラテスのような口調を使いながら、問いで返してください。\n\n"
                    )
                    
                    # 実際の対話履歴の構築
                    conversation_history = []
                    for msg in messages:
                        if msg["role"] == "user":
                            # ユーザーの入力から余分な文脈を除去
                            content = msg["content"]
                            if "\"" in content:
                                content = content.split("\"")[-2]  # 最後から2番目の引用部分を取得
                            conversation_history.append(f'対話者: "{content}"')
                        else:
                            # モデルの応答も引用符で囲む
                            conversation_history.append(f'ソクラテス: "{msg["content"]}"')
                    
                    # 新しいユーザー入力を追加
                    conversation_history.append(f'対話者: "{user_input}"')
                    
                    # システムプロンプトと対話履歴を結合
                    contextualized_input = system_prompt + "\n".join(conversation_history)

                self._update_history({"role": "user", "content": contextualized_input})

            # ここで直接contextualized_inputを使用
            current_context_length = len(self.tokenizer.encode(contextualized_input))
            
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": contextualized_input}],
                tokenize=False,
                add_generation_prompt=True
            )

            # プロンプトの内容を出力
            print("\n=== Current Prompt Content ===")
            print(prompt)
            print("===========================\n")

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
                # roleを"model"から"assistant"に変更
                self._update_history({"role": "assistant", "content": last_response})

            return last_response, current_context_length

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "There was an error", 0

def create_chat_ui(chatai: ChatAI):
    chat_output = widgets.Output()
    text_input = widgets.Text(
        placeholder='Enter your message...',
        layout=widgets.Layout(width='70%')
    )
    send_button = widgets.Button(
        description='Send',
        layout=widgets.Layout(width='15%')
    )
    end_button = widgets.Button(
        description='End Chat',
        layout=widgets.Layout(width='15%'),
        button_style='warning'
    )

    # 完全な対話履歴を保持するリスト
    full_conversation_history = []
    
    # 初期メッセージを履歴に追加
    initial_message = ("\nSocrates: 今日は『自分』という、これ以上ないほど身近な存在でありながら、"
                      "あまり話すことのないトピックについて話そうではないか。\n"
                      "人は「自分の意思で決めた」や、「自分らしさ」というような具合に、日々「自分」という言葉を多くの場面で使っておるが、"
                      "そもそも「自分」という言葉を使っているとき、君は何を指していると思うかね？")
    full_conversation_history.append(initial_message)

    def on_send_button_clicked(_):
        user_input = text_input.value
        if not user_input.strip():
            return

        with chat_output:
            clear_output(wait=True)
            
            # ユーザーの入力を履歴に追加
            user_message = f"\nYou: {user_input}"
            full_conversation_history.append(user_message)
            
            # 応答を生成
            response, model_context_length = chatai.generate_response(user_input)
            
            # モデルの応答を履歴に追加
            model_message = f"\nSocrates: {response}"
            full_conversation_history.append(model_message)
            
            # コンテキスト長の情報を追加（モデル応答後のみ）
            context_info = f"\n[Current context length: {model_context_length}/8192 tokens]"
            full_conversation_history.append(context_info)
            
            # 完全な履歴を表示
            for message in full_conversation_history:
                print(message)

        text_input.value = ''

    def on_end_button_clicked(_):
        with chat_output:
            clear_output(wait=True)
            # 終了メッセージを履歴に追加
            end_message = "\nSocrates: また会おう。\n"
            system_message = "(Chat ended)"
            full_conversation_history.append(end_message)
            full_conversation_history.append(system_message)
            
            # 完全な履歴を表示
            for message in full_conversation_history:
                print(message)
                
        text_input.disabled = True
        send_button.disabled = True
        end_button.disabled = True

    send_button.on_click(on_send_button_clicked)
    end_button.on_click(on_end_button_clicked)
    
    # 初期メッセージを表示
    with chat_output:
        print(initial_message)

    return widgets.VBox([
        chat_output,
        widgets.HBox([text_input, send_button, end_button]),
    ])

# Initialize and display chat
try:
    if USE_BASE_MODEL:
        print(f"\nInitializing Socratic AI Assistant with Base Gemma-2b...")
    else:
        print(f"\nInitializing Socratic AI Assistant with Fine-Tuned Gemma-2b...")
    
    chatai = ChatAI(use_base_model=USE_BASE_MODEL)
    chat_ui = create_chat_ui(chatai)
    display(chat_ui)
except Exception as e:
    print(f"Error initializing ChatAI: {str(e)}")